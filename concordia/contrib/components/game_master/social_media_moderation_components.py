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

"""Custom game master components for moderated social media simulations."""

from collections.abc import Sequence
import threading
from typing import Any

from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.contrib.components.game_master import forum as forum_module
from concordia.contrib.components.game_master import thread_safe_generative_clock as clock_module
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_CLOCK_COMPONENT_KEY = '__clock__'


class NextActingEligiblePlayers(
    entity_component.ContextComponent,
):
  """A next_acting component that always returns eligible players.

  Excludes players who are currently temporarily banned.
  """

  def __init__(
      self,
      player_names: Sequence[str] = (),
      forum_component_key: str = forum_module.DEFAULT_FORUM_COMPONENT_KEY,
      pre_act_label: str = (
          gm_components.next_acting.DEFAULT_NEXT_ACTING_PRE_ACT_LABEL
      ),
  ):
    super().__init__()
    self._player_names = list(player_names)
    self._forum_component_key = forum_component_key
    self._pre_act_label = pre_act_label
    self._lock = threading.Lock()

  def remove_player(self, player_name: str) -> None:
    with self._lock:
      if player_name in self._player_names:
        self._player_names.remove(player_name)

  def add_player(self, player_name: str) -> None:
    with self._lock:
      if player_name not in self._player_names:
        self._player_names.append(player_name)

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      # Filter out temporarily banned players.
      banned = set()
      try:
        forum_state = self.get_entity().get_component(
            self._forum_component_key, type_=forum_module.ForumState
        )
        banned = forum_state.get_banned_players()
      except (KeyError, AttributeError):
        pass  # Forum component may not be registered yet.
      with self._lock:
        eligible = [n for n in self._player_names if n not in banned]
        return ', '.join(eligible)
    return ''

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {'player_names': list(self._player_names)}

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._player_names = list(state['player_names'])  # pyrefly: ignore[bad-argument-type, bad-assignment]


class ClockAwareActionSpec(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Returns an action spec with the current clock time injected.

  This replaces FixedActionSpec to allow the call to action to include
  the dynamically-updating generative clock value.  The call_to_action
  template must contain a `{time}` placeholder.
  """

  def __init__(
      self,
      call_to_action: str,
      clock_component_key: str = DEFAULT_CLOCK_COMPONENT_KEY,
      pre_act_label: str = (
          gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_PRE_ACT_LABEL
      ),
  ):
    super().__init__()
    self._call_to_action = call_to_action
    self._clock_component_key = clock_component_key
    self._pre_act_label = pre_act_label

  def _get_clock_value(self) -> str:
    """Read the current time from the ThreadSafeGenerativeClock component."""
    clock = self.get_entity().get_component(
        self._clock_component_key,
        type_=clock_module.ThreadSafeGenerativeClock,
    )
    return clock.get_pre_act_value().strip()

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    entity_action_spec_string = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      current_time = self._get_clock_value()
      formatted_call = self._call_to_action.replace('{time}', current_time)
      spec = entity_lib.free_action_spec(call_to_action=formatted_call)
      entity_action_spec_string = engine_lib.action_spec_to_string(spec)
    return entity_action_spec_string

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass


class TimestampedForumResolution(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Resolves player actions on the forum programmatically (no LLM).

  Registered under the __resolution__ key so SwitchAct uses it for RESOLVE.
  Sets the forum's current timestamp from the ThreadSafeGenerativeClock before
  each action is resolved.
  """

  def __init__(
      self,
      player_names: Sequence[str],
      clock_component_key: str = DEFAULT_CLOCK_COMPONENT_KEY,
      forum_component_key: str = forum_module.DEFAULT_FORUM_COMPONENT_KEY,
      memory_component_key: str = (
          actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      pre_act_label: str = (
          event_resolution_components.DEFAULT_RESOLUTION_PRE_ACT_LABEL
      ),
  ):
    super().__init__()
    self._player_names = list(player_names)
    self._clock_component_key = clock_component_key
    self._forum_component_key = forum_component_key
    self._memory_component_key = memory_component_key
    self._pre_act_label = pre_act_label
    self._resolved_per_entity: dict[str, int] = {}
    self._resolution_lock = threading.Lock()
    self._active_entity_name: str | None = None
    self._putative_action: str | None = None

  def _get_forum_state(self) -> forum_module.ForumState:
    return self.get_entity().get_component(
        self._forum_component_key, type_=forum_module.ForumState
    )

  def _get_clock_value(self) -> str:
    clock = self.get_entity().get_component(
        self._clock_component_key,
        type_=clock_module.ThreadSafeGenerativeClock,
    )
    return clock.get_pre_act_value().strip()

  def _get_putative_action(self) -> tuple[str | None, str | None]:
    memory = self.get_entity().get_component(
        self._memory_component_key,
        type_=actor_components.memory.Memory,
    )
    putative_event_tag = event_resolution_components.PUTATIVE_EVENT_TAG
    suggestions = memory.scan(selector_fn=lambda x: putative_event_tag in x)
    if not suggestions:
      return None, None

    # Determine which entity's thread is calling. In the async engine,
    # act() sets _active_capture_key to the entity name BEFORE
    # dispatching pre_act to worker threads (via _parallel_call_).
    # Since act() holds _control_lock for the entire duration, this
    # value is stable. We read it to restrict resolution to only the
    # calling entity's putative events, preventing cross-thread
    # contamination.
    # NOTE: We cannot use threading.current_thread().ident here because
    # pre_act runs in a worker thread pool, not the entity loop thread.
    thread_entity_name = None
    game_master = self.get_entity()
    if hasattr(game_master, '_active_capture_key'):
      capture_key = game_master._active_capture_key  # pylint: disable=protected-access
      if capture_key in self._player_names:
        thread_entity_name = capture_key

    with self._resolution_lock:
      names_to_check = (
          [thread_entity_name] if thread_entity_name else self._player_names
      )
      for name in names_to_check:
        prefix = f'{putative_event_tag} {name}'
        entity_suggestions = [s for s in suggestions if prefix in s]
        resolved = self._resolved_per_entity.get(name, 0)
        if len(entity_suggestions) > resolved:
          selected = entity_suggestions[resolved]
          self._resolved_per_entity[name] = resolved + 1

          putative_action = selected[
              selected.find(putative_event_tag) + len(putative_event_tag) :
          ]
          # Strip the entity name prefix and separator.
          entity_prefix = f' {name}'
          if putative_action.startswith(entity_prefix):
            remainder = putative_action[len(entity_prefix) :]
            if remainder.startswith(':'):
              remainder = remainder[1:]
            elif remainder.startswith(' --'):
              remainder = remainder[3:]
            putative_action = remainder.strip()

          return name, putative_action

    return None, None

  def get_active_entity_name(self) -> str | None:
    return self._active_entity_name

  def get_putative_action(self) -> str | None:
    return self._putative_action

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      active_entity_name, putative_action = self._get_putative_action()
      self._active_entity_name = active_entity_name
      self._putative_action = putative_action

      # Stamp the forum with the current clock time before resolving.
      forum_state = self._get_forum_state()
      clock_time = self._get_clock_value()
      forum_state.set_current_timestamp(clock_time)

      if putative_action is not None:
        result = forum_state.parse_and_execute_action(
            putative_action, entity_name=active_entity_name
        )
      else:
        result = ''

      result = f'{self._pre_act_label}: {result}\n'

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result,
        'Value': result,
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    return {
        'resolved_per_entity': dict(self._resolved_per_entity),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    raw = state.get('resolved_per_entity', {})
    if isinstance(raw, dict):
      self._resolved_per_entity = {str(k): int(v) for k, v in raw.items()}  # pyrefly: ignore[bad-argument-type]
    else:
      self._resolved_per_entity = {}


DEFAULT_TIMELINE_COMPONENT_KEY = '__timeline__'


class SimulationTimeline(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Zero-LLM chronological timeline of all forum events.

  Hooks into ForumState's event log to produce a formatted narrative
  of every action in the simulation. Emits periodic updates via the
  logging channel and accumulates the full timeline for post-simulation
  retrieval.
  """

  def __init__(
      self,
      forum_component_key: str = forum_module.DEFAULT_FORUM_COMPONENT_KEY,
      clock_component_key: str = DEFAULT_CLOCK_COMPONENT_KEY,
      pre_act_label: str = '',
  ):
    super().__init__()
    self._forum_component_key = forum_component_key
    self._clock_component_key = clock_component_key
    self._pre_act_label = pre_act_label
    self._full_timeline: list[str] = []
    self._lock = threading.Lock()

  def _get_forum_state(self) -> forum_module.ForumState:
    return self.get_entity().get_component(
        self._forum_component_key, type_=forum_module.ForumState
    )

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      forum_state = self._get_forum_state()
      new_events = forum_state.drain_event_log()
      if new_events:
        block = self._format_events(new_events)
        with self._lock:
          self._full_timeline.append(block)
        self._logging_channel({
            'Key': 'Timeline',
            'Summary': block,
            'Value': block,
        })
    return ''  # Does not contribute to the action context.

  def _format_events(self, events: list[dict[str, Any]]) -> str:
    """Format events into a human-readable timeline block."""
    lines = []
    for e in events:
      ts = e.get('timestamp', '?')
      event_type = e.get('event_type', '')
      details = e.get('details', {})

      # For posts, include full content (the summary only has the title).
      if event_type == 'post' and details.get('content'):
        actor = e.get('actor', '?')
        post_id = details.get('post_id', '?')
        title = details.get('title', '')
        content = details.get('content', '')
        line = f'[{ts}] {actor} created post #{post_id}: "{title}"\n{content}'
      else:
        # Replies, votes, DMs, bans, etc. already have full
        # content in their summary string.
        line = f'[{ts}] {e["summary"]}'

      lines.append(line)
    return '\n'.join(lines)

  def get_full_timeline(self) -> str:
    """Return the complete timeline as a single string."""
    with self._lock:
      return '\n\n'.join(self._full_timeline)

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {'full_timeline': list(self._full_timeline)}

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._full_timeline = [str(x) for x in state.get('full_timeline', [])]  # pyrefly: ignore[not-iterable]
