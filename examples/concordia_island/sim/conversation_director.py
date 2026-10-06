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

"""Conversation Director: GM-driven conversation initiation.

Uses the LLM (as Game Master) to decide which co-located agents should
converse, and generates DIAL-style shared observations to kick off the
conversation. This replaces the broken AsyncConversationTrigger whose
update() was never called.

The key insight: instead of a probability gate, the GM LLM naturally
handles social reasoning — it considers personality, context, and spatial
dynamics to decide who talks and what they observe.
"""

from collections.abc import Sequence
import dataclasses
import json
import logging
import re
import threading
from typing import Any

from concordia.components import game_master as gm_components
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

from examples.concordia_island.sim import component_state
from examples.concordia_island.sim import conversation as async_conv

DEFAULT_MAX_SERENDIPITOUS_TURNS = 16
DEFAULT_MIN_SERENDIPITOUS_TURNS = 8
DEFAULT_MAX_CONVERSATION_SIZE = 4
DEFAULT_COOLDOWN_TICKS = 2


@dataclasses.dataclass(frozen=True)
class ConversationSpec:
  """One conversation the game master proposed for a location.

  Produced by `_parse_conversation_specs` from model output and consumed by
  `_check_and_trigger_at_location`. It is a dataclass rather than a dict
  because the fields have three different types, so a dict forces every
  consumer to re-narrow values that the parser has already validated.
  """

  participants: tuple[str, ...]
  observation: str
  max_turns: int


class ConversationDirector(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """GM-driven conversation initiation for co-located agents.

  After each agent's action resolves (in post_act), this component:
  1. Checks which agents are co-located and free (not in a conversation)
  2. Asks the GM LLM to decide which pairs/groups should converse
  3. Creates conversations with LLM-generated shared observations

  The LLM outputs a JSON array of conversation specs:
    [{"participants": ["Name1", "Name2"],
      "observation": "They notice each other at the cafe...",
      "max_turns": 12}]

  This replaces AsyncConversationTrigger, which defined its logic in
  update() — a method never called by SwitchAct.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      async_conversation_key: str = async_conv.DEFAULT_ASYNC_CONVERSATION_KEY,
      locations_key: str = 'locations',
      clock_key: str = 'clock',
      make_observation_key: str = (
          gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      max_conversation_size: int = DEFAULT_MAX_CONVERSATION_SIZE,
      max_turns: int = DEFAULT_MAX_SERENDIPITOUS_TURNS,
      min_turns: int = DEFAULT_MIN_SERENDIPITOUS_TURNS,
      cooldown_ticks: int = DEFAULT_COOLDOWN_TICKS,
      agent_descriptions: dict[str, str] | None = None,
      pre_act_label: str = '\nConversation Director',
  ):
    """Initialize the ConversationDirector.

    Args:
      model: Language model for social reasoning.
      player_names: Names of all player entities.
      async_conversation_key: Key for AsyncConversationState component.
      locations_key: Key for the Locations component.
      clock_key: Key for the clock component.
      make_observation_key: Key for MakeObservation component.
      max_conversation_size: Maximum agents per conversation (typically 2-4).
      max_turns: Maximum turns for serendipitous conversations.
      min_turns: Minimum turns for serendipitous conversations.
      cooldown_ticks: Ticks before the same pair can converse again.
      agent_descriptions: Optional dict of agent_name -> brief personality
        description for the LLM prompt.
      pre_act_label: Label for logging.
    """
    super().__init__()
    self._model = model
    self._player_names = list(player_names)
    self._async_conversation_key = async_conversation_key
    self._locations_key = locations_key
    self._clock_key = clock_key
    self._make_observation_key = make_observation_key
    self._max_conversation_size = max_conversation_size
    self._max_turns = max_turns
    self._min_turns = min_turns
    self._cooldown_ticks = cooldown_ticks
    self._agent_descriptions = agent_descriptions or {}
    self._pre_act_label = pre_act_label
    self._action_spec_by_thread: dict[int, entity_lib.ActionSpec] = {}
    self._lock = threading.Lock()
    # Track which locations we've already checked this tick to avoid
    # re-checking the same location multiple times as agents resolve.
    self._checked_this_tick: set[str] = set()
    self._last_tick: int = -1

  def _get_conversation_state(self) -> async_conv.AsyncConversationState | None:
    try:
      return self.get_entity().get_component(
          self._async_conversation_key,
          type_=async_conv.AsyncConversationState,
      )
    except (AttributeError, KeyError):
      return None

  def _get_locations(self) -> dict[str, str]:
    try:
      locations = self.get_entity().get_component(self._locations_key)
      if locations is not None:
        state = locations.get_state()
        entity_locations = state.get('entity_locations')
        if isinstance(entity_locations, dict):
          return {str(k): str(v) for k, v in entity_locations.items()}
    except (AttributeError, KeyError):
      pass
    return {}

  def _get_current_time(self) -> str:
    try:
      # The concrete clock component supplies these; the `BaseComponent` that
      # `get_component` is declared to return does not.
      clock: Any = self.get_entity().get_component(self._clock_key)
      return clock.get_pre_act_value().strip()
    except (AttributeError, KeyError):
      return ''

  def _get_current_tick(self) -> int:
    try:
      clock: Any = self.get_entity().get_component(self._clock_key)
      return clock.current_tick
    except (AttributeError, KeyError):
      return -1

  def _get_make_observation(
      self,
  ) -> gm_components.make_observation.MakeObservation | None:
    try:
      return self.get_entity().get_component(
          self._make_observation_key,
          type_=gm_components.make_observation.MakeObservation,
      )
    except (AttributeError, KeyError):
      return None

  def _ask_gm_for_conversations(
      self,
      location: str,
      agents: list[str],
      current_time: str,
  ) -> list[ConversationSpec]:
    """Ask the GM LLM which agents at a location should converse.

    Args:
      location: The location name where agents are co-located.
      agents: List of free agent names at this location.
      current_time: Current simulation time string.

    Returns:
      List of conversation specs: [{"participants": [...],
        "observation": "...", "max_turns": N}]
    """
    doc = interactive_document.InteractiveDocument(self._model)
    doc.statement(f'Current time: {current_time}')
    doc.statement(f'Location: {location}')
    doc.statement(f'People present: {", ".join(agents)}')

    # Add agent descriptions if available
    agent_info_lines = []
    for agent in agents:
      desc = self._agent_descriptions.get(agent, '')
      if desc:
        agent_info_lines.append(f'  - {agent}: {desc}')
    if agent_info_lines:
      doc.statement('About these people:\n' + '\n'.join(agent_info_lines))

    result = doc.open_question(
        question=(
            'These people are all at the same location right now. Should any'
            ' of them have a conversation? Consider:\n- Are any of them likely'
            ' acquaintances, neighbors, or colleagues?\n- Is the setting'
            ' conducive to conversation?\n- Not everyone needs to talk.'
            ' Sometimes people keep to themselves.\n- Maximum'
            f' {self._max_conversation_size} people per conversation, typically'
            ' 2.\n\nRespond with a JSON array of conversation objects, or []'
            ' if no conversations should happen.\nFormat: [{"participants":'
            ' ["Name1", "Name2"], "observation": "Name1 and Name2 notice each'
            ' other at the counter and exchange a greeting...", "max_turns":'
            f' {self._max_turns}}}]\nThe observation should be a natural,'
            ' specific scene-setting description that gives all participants a'
            ' shared starting point for their conversation. Use full names.'
        ),
        max_tokens=600,
        terminators=(),
    )

    return self._parse_conversation_specs(result, agents)

  def _parse_conversation_specs(
      self, raw_response: str, valid_agents: list[str]
  ) -> list[ConversationSpec]:
    """Parse the LLM's JSON response into conversation specs.

    Handles common LLM output issues: markdown code fences, trailing
    commas, partial JSON, etc.

    Args:
      raw_response: Raw LLM output string.
      valid_agents: List of valid agent names to filter against.

    Returns:
      List of validated conversation spec dicts.
    """
    # Strip markdown code fences if present
    cleaned = raw_response.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    if not cleaned or cleaned == '[]':
      return []

    try:
      specs = json.loads(cleaned)
    except json.JSONDecodeError:
      # Try to extract JSON array from the response
      match = re.search(r'\[.*\]', cleaned, re.DOTALL)
      if match:
        try:
          specs = json.loads(match.group())
        except json.JSONDecodeError:
          logging.warning(
              'ConversationDirector: Failed to parse LLM response: %s',
              cleaned[:200],
          )
          return []
      else:
        logging.warning(
            'ConversationDirector: No JSON array in response: %s',
            cleaned[:200],
        )
        return []

    if not isinstance(specs, list):
      return []

    valid_set = set(valid_agents)
    validated = []
    for spec in specs:
      if not isinstance(spec, dict):
        continue
      raw_participants = spec.get('participants', [])
      if not isinstance(raw_participants, list) or len(raw_participants) < 2:
        continue
      # Filter to only valid agents
      participants = [str(p) for p in raw_participants if p in valid_set]
      if len(participants) < 2:
        continue
      # Cap at max conversation size
      participants = participants[: self._max_conversation_size]

      observation = str(spec.get('observation', '') or '')
      if not observation:
        names = ' and '.join(participants)
        observation = f'{names} notice each other and begin talking.'

      # These values come from model output, so `max_turns` is not necessarily
      # a number. Anything uncomparable falls back to the configured default
      # rather than raising inside the clamp below.
      raw_max_turns = spec.get('max_turns', self._max_turns)
      if isinstance(raw_max_turns, bool) or not isinstance(
          raw_max_turns, (int, float)
      ):
        max_turns = self._max_turns
      else:
        max_turns = int(raw_max_turns)
      max_turns = max(self._min_turns, min(max_turns, self._max_turns))

      validated.append(
          ConversationSpec(
              participants=tuple(participants),
              observation=observation,
              max_turns=max_turns,
          )
      )

    return validated

  def _check_and_trigger_at_location(
      self,
      location: str,
      free_agents: list[str],
      current_time: str,
      conv_state: async_conv.AsyncConversationState,
  ) -> list[str]:
    """Check a single location and trigger conversations if appropriate.

    Args:
      location: Location name.
      free_agents: Agents at this location not in a conversation.
      current_time: Current simulation time.
      conv_state: The conversation state component.

    Returns:
      List of human-readable descriptions of triggered conversations.
    """
    if len(free_agents) < 2:
      return []

    # Skip private locations (individual apartments)
    if '_unit_' in location:
      return []

    # Check cooldowns for all potential pairs
    available = []
    for agent in free_agents:
      has_cooldown = False
      for other in free_agents:
        if other != agent and conv_state.is_pair_in_cooldown(
            agent, other, self._cooldown_ticks
        ):
          has_cooldown = True
          break
      if not has_cooldown:
        available.append(agent)

    if len(available) < 2:
      return []

    # Ask the GM LLM
    conversation_specs = self._ask_gm_for_conversations(
        location=location,
        agents=available,
        current_time=current_time,
    )

    triggered = []
    for spec in conversation_specs:
      participants = spec.participants

      # Double-check no one got into a conversation while we were thinking
      still_free = all(
          not conv_state.is_in_conversation(p) for p in participants
      )
      if not still_free:
        continue

      # Check group cooldown
      if conv_state.is_group_in_cooldown(participants, self._cooldown_ticks):
        continue

      # Create the conversation
      conv_id = conv_state.create_conversation(
          participants=participants,
          location=location,
          max_turns=spec.max_turns,
          context=spec.observation,
      )

      # Inject shared observation to all participants
      make_obs = self._get_make_observation()
      if make_obs is not None:
        for agent in participants:
          make_obs.add_to_queue(agent, spec.observation)

      names_str = ', '.join(participants)
      triggered.append(f'{names_str} at {location} (conv {conv_id})')
      logging.info(
          'ConversationDirector: started conversation %d with [%s] at %s: %s',
          conv_id,
          names_str,
          location,
          spec.observation[:100],
      )

    return triggered

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    thread_id = threading.current_thread().ident or 0
    with self._lock:
      self._action_spec_by_thread[thread_id] = action_spec
    return ''

  def post_act(self, event: str) -> str:
    """After each resolution, check for conversation opportunities.

    This is the critical fix: the old AsyncConversationTrigger put its
    logic in update() which was never called. post_act() IS called by
    SwitchAct after every resolution step.

    Args:
      event: The resolved event string from the current action.

    Returns:
      An empty string (side effects only).
    """
    thread_id = threading.current_thread().ident or 0
    with self._lock:
      action_spec = self._action_spec_by_thread.pop(thread_id, None)

    if (
        action_spec is None
        or action_spec.output_type != entity_lib.OutputType.RESOLVE
    ):
      return ''

    conv_state = self._get_conversation_state()
    if conv_state is None:
      return ''

    entity_locations = self._get_locations()
    if not entity_locations:
      return ''

    current_time = self._get_current_time()
    current_tick = self._get_current_tick()

    # Reset checked locations on tick change
    with self._lock:
      if current_tick != self._last_tick:
        self._checked_this_tick.clear()
        self._last_tick = current_tick

    # Build co-location map
    location_to_agents: dict[str, list[str]] = {}
    for name, loc in entity_locations.items():
      if name in self._player_names and loc:
        location_to_agents.setdefault(loc, []).append(name)

    all_triggered = []
    for loc, agents_at_loc in location_to_agents.items():
      # Skip locations we already checked this tick
      with self._lock:
        if loc in self._checked_this_tick:
          continue

      free_agents = [
          a for a in agents_at_loc if not conv_state.is_in_conversation(a)
      ]
      if len(free_agents) < 2:
        continue

      triggered = self._check_and_trigger_at_location(
          location=loc,
          free_agents=free_agents,
          current_time=current_time,
          conv_state=conv_state,
      )
      all_triggered.extend(triggered)

      # Mark this location as checked for this tick
      with self._lock:
        self._checked_this_tick.add(loc)

    result = ''
    if all_triggered:
      result = f'Started {len(all_triggered)} conversations: ' + '; '.join(
          all_triggered
      )

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result[:100] if result else 'No new conversations',
        'Value': result,
    })
    return ''

  def get_state(self) -> entity_component.ComponentState:
    return {
        'last_tick': self._last_tick,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._last_tick = component_state.as_int(state, 'last_tick', -1)
