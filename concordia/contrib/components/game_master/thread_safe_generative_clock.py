# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A thread-safe generative clock that increments only after all players act.

This component provides a language-model-driven clock for Concordia game master
simulations. Unlike the standard GenerativeClock, this version is designed for
safe use with concurrent entity loops (e.g. the asynchronous engine):
  1. Guards all mutable state with a threading.Lock.
  2. Tracks which players have acted in the current round.
  3. Only advances the clock (via LLM call) once every player has acted at
     least once, then resets the per-round tracker.
"""

from collections.abc import Sequence
import re
import threading

from concordia.components.agent import action_spec_ignored
from concordia.components.game_master import player_roster
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class ThreadSafeGenerativeClock(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A thread-safe generative clock updated via language model.

  The clock only advances after every registered player has acted at least
  once in the current round.  All public methods that touch mutable state are
  guarded by a lock so the component is safe for use with concurrent entity
  loops (e.g. the asynchronous engine).
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      prompt: str,
      start_time: str,
      player_names: Sequence[str],
      components: Sequence[str] = (),
      format_description_key: str = 'Clock format description',
      pre_act_label: str = '\nClock',
      update_question: str | None = None,
      aliases: dict[str, str] | None = None,
      roster_component_key: str | None = None,
  ):
    """Initializes the component.

    Args:
      model: The language model to use.
      prompt: Description of what the clock represents, how it gets updated, and
        what it is used for.
      start_time: The initial time of the clock.
      player_names: Names of all player entities.  The clock only ticks after
        every one of them has acted at least once.
      components: Keys of components to condition clock updates on.
      format_description_key: The key to prepend to the description of the
        desired format to use in the prompt for the sample that produces the
        clock's update on each step.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
      update_question: Optional custom question to ask the model to update the
        clock. If None, a default question is used.
      aliases: Optional dictionary mapping aliases to full player names.
      roster_component_key: Optional key of a `PlayerRoster` component on the
        parent entity used to dynamically resolve active player names.
    """
    super().__init__()
    self._pre_act_label = pre_act_label
    self._model = model
    self._format_description_key = format_description_key
    self._prompt = prompt
    self._components = tuple(components)
    self._update_question = update_question
    self._aliases = aliases or {}
    self._roster_component_key = roster_component_key

    # ── Round-gating state ──
    self._player_names: frozenset[str] = frozenset(player_names)
    self._acted_this_round: set[str] = set()
    self._lock = threading.Lock()

    # ── Initial clock calibration (one-time LLM call) ──
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement(self._prompt)
    chain_of_thought.statement(f'Start time: {start_time}')
    self._clock_description = chain_of_thought.open_question(
        question=(
            'Given the context above, when is the clock updated? How are times '
            'represented internally? (usually as a number of steps). And, how '
            'do internal time repesentations map to the time representations '
            'communicated to players?'
        ),
        max_tokens=1000,
        terminators=(),
    )

    self._num_steps = 0
    self._time = start_time
    self._prompt_to_log = ''
    self._latest_action_spec: entity_lib.ActionSpec | None = None

  # ─── Component helpers ────────────────────────────────────────────────

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_value()
    )

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'{self.get_component_pre_act_label(key)}:\n'
        f'{self.get_named_component_pre_act_value(key)}'
    )

  def _resolve_roster_players(self) -> frozenset[str] | None:
    """Reads live player names from `PlayerRoster` if configured."""
    if not self._roster_component_key:
      return None
    try:
      roster = self.get_entity().get_component(
          self._roster_component_key, type_=player_roster.PlayerRoster
      )
      return frozenset(roster.get_player_names())
    except (KeyError, AttributeError, RuntimeError):
      return None

  def get_player_names(self) -> frozenset[str]:
    """Returns the currently tracked active player names."""
    roster_players = self._resolve_roster_players()
    with self._lock:
      if roster_players is not None:
        self._player_names = roster_players
      return self._player_names

  def get_step_count(self) -> int:
    """Returns the number of completed clock ticks (thread-safe)."""
    with self._lock:
      return self._num_steps

  def get_pre_act_label(self) -> str:
    """Returns the key used as a prefix in the string returned by `pre_act`."""
    return self._pre_act_label

  def get_pre_act_value(self) -> str:
    """Returns the current clock time (thread-safe)."""
    with self._lock:
      return self._time + '\n'

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    with self._lock:
      self._latest_action_spec = action_spec
      result = self._time + '\n'
    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result,
        'Value': result,
        'Prompt': self._prompt_to_log,
    })
    return result

  def _identify_actor(
      self, event: str, active_players: frozenset[str]
  ) -> str | None:
    """Best-effort extraction of the acting player from an event string."""
    actor_by_label = dict(self._aliases)
    # Preserve canonical names when an alias has the exact same spelling.
    actor_by_label.update(
        {name: name for name in active_players | self._player_names}
    )
    candidate_events = (event, event.removeprefix('Event: '))
    for label in sorted(actor_by_label, key=len, reverse=True):
      # Prefer the most specific known label and do not credit a shorter
      # identifier for an unrelated word that merely begins with it.
      pattern = re.escape(label) + r'(?!\w)'
      if any(re.match(pattern, text) for text in candidate_events):
        return actor_by_label[label]
    return None

  def post_act(
      self,
      event: str,
  ) -> str:
    roster_players = self._resolve_roster_players()
    with self._lock:
      if (
          self._latest_action_spec is None
          or self._latest_action_spec.output_type
          != entity_lib.OutputType.RESOLVE
      ):
        return ''

      active_players = (
          roster_players if roster_players is not None else self._player_names
      )
      if roster_players is not None:
        self._player_names = roster_players

      # Track which player just acted.
      actor = self._identify_actor(event, active_players)
      if actor:
        self._acted_this_round.add(actor)

      # Only advance the clock once all active players have acted.
      if not active_players or not active_players.issubset(
          self._acted_this_round
      ):
        return ''

      # ── All players have acted: tick the clock ──
      prompt = interactive_document.InteractiveDocument(self._model)

      component_states = '\n'.join(
          [self._component_pre_act_display(key) for key in self._components]
      )
      prompt.statement(f'\n{component_states}\n')

      prompt.statement(
          f'{self._format_description_key}:\n{self._prompt}\n***\n'
      )

      prompt.statement(f'Time prior to the latest event: {self._time}')

      self._num_steps += 1
      prompt.statement(f'The next event: {event}')
      prompt.statement(
          f'Internal simulation step count is now: {self._num_steps}'
      )

      if self._update_question:
        question = self._update_question
      else:
        question = (
            'Given the context above, and after the event, what is the new '
            'time? Never respond with a sentence like "the time is unchanged"'
            ' or anything to that effect, never respond with "unknown", and '
            'never mention the number of simulation steps. Always convert '
            'from internal simulation steps to the requested format.\n'
            'Correct responses always follow the format described under '
            f'"{self._format_description_key}" above.'
        )

      self._time = prompt.open_question(
          question=question,
          max_tokens=128,
      )

      self._prompt_to_log = prompt.view().text()

      # Reset for next round.
      self._acted_this_round.clear()

    return ''

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    with self._lock:
      return {
          'num_steps': self._num_steps,
          'time': self._time,
          'prompt_to_log': self._prompt_to_log,
          'acted_this_round': sorted(self._acted_this_round),
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    with self._lock:
      self._num_steps = state['num_steps']  # pyrefly: ignore[bad-assignment]
      self._time = state['time']  # pyrefly: ignore[bad-assignment]
      self._prompt_to_log = state['prompt_to_log']  # pyrefly: ignore[bad-assignment]
      self._latest_action_spec = None
      raw_acted = state.get('acted_this_round', [])
      if isinstance(raw_acted, list):
        self._acted_this_round = {str(x) for x in raw_acted}
      else:
        self._acted_this_round = set()
