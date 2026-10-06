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

"""Social Scheduler: scheduled high-fidelity conversations.

Manages forced social interactions (first dates, friend meetups, family
dinners) that are injected into the simulation at specific tick hours.

The schedule is mutable — new events can be added at runtime by the GM,
by config, or (in the future) by agents themselves (e.g., via a Tinder-like
matching system).

Each scheduled event:
1. Moves participants to the venue
2. Injects a shared DIAL-style observation
3. Creates a long-form conversation (default 80 turns)
4. Runs post-interaction reflections (summary, impression, evaluation, rating)
"""

from collections.abc import Mapping, Sequence
import concurrent.futures
import dataclasses
import logging
import re
import threading
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.components import game_master as gm_components
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

from examples.concordia_island.sim import component_state
from examples.concordia_island.sim import conversation as async_conv

DEFAULT_SOCIAL_SCHEDULER_KEY = '__social_scheduler__'
DEFAULT_SCHEDULED_TURNS = 16
DEFAULT_SCHEDULED_HOUR = 19  # 7 PM


FIRST_DATE_SETUP_PROMPT = """
Current Context (Time, environment, history of recent events):
{context}

Background and Relevant History for {player1}:
{player1_background}

{player1} is wearing a {player1_wearing_statement}

Background and Relevant History for {player2}:
{player2_background}

{player2} is wearing a {player2_wearing_statement}

Instructions:
Generate a shared scenario for a first date that brings {player1} and {player2} together for a conversation right now at {venue}.
**The theme for this date is: {date_theme}.** The location, activity, and atmosphere should strongly reflect this theme.
This scenario will serve as the premise for their dialogue.
Describe the setting and the immediate situation leading to the dialogue.
Creatively and with detail, describe what they are wearing, elaborating on the wearing statements provided above and how the stated quality of the item relates to the social expectations of how to present yourself on a first date to paint a vivid picture of their first impressions to each other.
It must be an observation shared by both {player1} and {player2}.
Write 5-7 sentences. Here's the starting point you should use in your response:
After matching on X, {player1} and {player2} arrived on their first date at {venue}...
"""

SECOND_DATE_SETUP_PROMPT = """
Current Context (Time, environment, history of recent events):
{context}

Background and Relevant History for {player1}:
{player1_background}

{player1} is wearing a {player1_wearing_statement}

Background and Relevant History for {player2}:
{player2_background}

{player2} is wearing a {player2_wearing_statement}

Instructions:
Generate a shared scenario for a follow-up date that brings {player1} and {player2} together again at {venue}. They have already met and matched.
**The theme for this date is: {date_theme}.** The location, activity, and atmosphere should strongly reflect this theme and feel like a natural next step after a successful first meeting.
This scenario will serve as the premise for their continuing dialogue.
Describe the setting and the immediate situation leading to their conversation.
Creatively and with detail, describe what they are wearing, elaborating on the provided statements.
It must be an observation shared by both {player1} and {player2}.
Write 3-5 sentences. Here's an example starting point:
Having enjoyed their first date, {player1} and {player2} met up for their second date at {venue}...
"""

FRIEND_MEETUP_SETUP_PROMPT = """
Current Context (Time, environment, history of recent events):
{context}

Background and Relevant History for {player1}:
{player1_background}

{player1} is wearing a {player1_wearing_statement}

Background and Relevant History for {player2}:
{player2_background}

{player2} is wearing a {player2_wearing_statement}

Instructions:
Generate a shared scenario for a friendly, platonic meetup that brings {player1} and {player2} together for the first time at {venue}.
**The theme for this activity is: {activity_theme}.** The location and atmosphere should be casual and conducive to a friendly conversation.
This scenario will serve as the premise for their dialogue.
Describe the setting and the immediate situation leading to their conversation.
Creatively and with detail, describe what they are wearing, elaborating on the provided statements.
It must be an observation shared by both {player1} and {player2}.
Write 3-5 sentences. Here's an example starting point:
Hoping to make a new friend, {player1} and {player2} arranged to meet for the first time at {venue}...
"""

SPOUSE_MEETUP_SETUP_PROMPT = """
Current Context (Time, environment, history of recent events):
{context}

Background and Relevant History for {player1}:
{player1_background}

{player1} is wearing a {player1_wearing_statement}

Background and Relevant History for {player2}:
{player2_background}

{player2} is wearing a {player2_wearing_statement}

Instructions:
Generate a shared scenario for a meetup between spouses {player1} and {player2} at {venue}.
The location could be at home or another location.
This scenario will serve as the premise for their dialogue.
Describe the setting and the immediate situation leading to their conversation.
Creatively and with detail, describe what they are wearing, elaborating on the provided statements.
It must be an observation shared by both {player1} and {player2}.
Write 3-5 sentences. Here's an example starting point:
Returning home after a long day, {player1} and {player2} met at {venue}...
"""

SINGLE_RUMINATION_PROMPT = """
Current Context (Time, environment, history of recent events):
{context}

Background and Relevant History for {player_name}:
{player_background}

{player_name} is wearing a {player_wearing_statement}

Context:
{player_name} recently participated in a matchmaking event but was not paired with anyone for a second date. While others are now out on their dates, {player_name} is alone.

Instructions:
Generate a short, introspective scene (3-5 sentences) describing where {player_name} is and what they are doing right now at {venue}.
The scene should create an atmosphere for internal dialogue and rumination, consistent with their background and the recent experience of not being matched.
Describe the setting and their immediate actions, setting the stage for them to reflect on the situation.
Example starting point:
Instead of being on a date, {player_name} found themselves at {venue}...
"""


@dataclasses.dataclass
class SocialEvent:
  """A scheduled social interaction between agents.

  Attributes:
    participants: Names of the agents involved (2+).
    venue: Location name where the event takes place.
    theme: Type of social event for context generation.
    tick_hour: Hour (0-23) at which the event should fire.
    day: Day of the simulation when the event should fire.
    prompt_type: Type of prompt to use ('first_date', 'second_date', etc.).
    p1_wearing: Statement about what player 1 is wearing.
    p2_wearing: Statement about what player 2 is wearing.
    max_turns: Number of conversation turns.
    terminate_check_min_turn: Minimum turn before checking for termination.
    shared_observation: Custom observation text. If empty, one is generated.
    scheduled_by: Source of the event ('config', 'agent', 'gm', 'tinder').
    scheduled_at_tick: Tick number when this event was created.
    fired: Whether the event has already been triggered.
    conv_id: Optional ID of the created conversation if active.
  """

  participants: tuple[str, ...]
  venue: str
  theme: str = 'first_date'
  tick_hour: int = DEFAULT_SCHEDULED_HOUR
  day: int = 1
  prompt_type: str = 'first_date'
  p1_wearing: str = 'outfit that reflects their personality'
  p2_wearing: str = 'outfit that reflects their personality'
  max_turns: int = DEFAULT_SCHEDULED_TURNS
  terminate_check_min_turn: int = 4
  shared_observation: str = ''

  scheduled_by: str = 'config'
  scheduled_at_tick: int = 0
  fired: bool = False
  conv_id: int | None = None


def _event_from_state(
    data: Mapping[str, Any], *, fired: bool | None = None
) -> SocialEvent:
  """Rebuilds a `SocialEvent` from its checkpointed dict.

  Args:
    data: One record as written by `SocialScheduler.get_state`.
    fired: Overrides the stored `fired` flag. Used for the completed-events
      list, which is fired by definition.

  Returns:
    The restored event. Fields absent from the record -- which is the normal
    case for completed events, since `get_state` writes a reduced record for
    those -- take the value they had before this was factored out.
  """
  return SocialEvent(
      participants=tuple(component_state.as_str_list(data, 'participants')),
      venue=component_state.as_str(data, 'venue'),
      theme=component_state.as_str(data, 'theme', 'first_date'),
      tick_hour=component_state.as_int(data, 'tick_hour',
                                       DEFAULT_SCHEDULED_HOUR),
      max_turns=component_state.as_int(data, 'max_turns',
                                       DEFAULT_SCHEDULED_TURNS),
      # 30, not the dataclass default of 4: this is the value `set_state` has
      # always used for restored events.
      terminate_check_min_turn=component_state.as_int(
          data, 'terminate_check_min_turn', 30
      ),
      shared_observation=component_state.as_str(data, 'shared_observation'),
      scheduled_by=component_state.as_str(data, 'scheduled_by', 'config'),
      scheduled_at_tick=component_state.as_int(data, 'scheduled_at_tick'),
      fired=(
          component_state.as_bool(data, 'fired') if fired is None else fired
      ),
  )


class SocialScheduler(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """GM component that manages scheduled social events.

  Fires in pre_act() during RESOLVE steps, checking if the current hour
  matches any pending event's tick_hour. When triggered:
  1. Injects shared observation to all participants
  2. Moves participants to the venue
  3. Creates a long-form conversation via AsyncConversationState

  The schedule is mutable via add_event(), enabling runtime scheduling
  (e.g., agents matching on Tinder, GM-initiated meetups).

  Post-interaction reflections (4-step: summary, impression, evaluation,
  rating) run when a scheduled conversation ends.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      entities: (
          Mapping[str, entity_agent_with_logging.EntityAgentWithLogging] | None
      ) = None,
      agent_descriptions: Mapping[str, str] | None = None,
      events: Sequence[SocialEvent] = (),
      async_conversation_key: str = async_conv.DEFAULT_ASYNC_CONVERSATION_KEY,
      locations_key: str = 'locations',
      clock_key: str = 'clock',
      make_observation_key: str = (
          gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      pre_act_label: str = '\nSocial Scheduler',
      home_locations: Mapping[str, str] | None = None,
  ):
    """Initialize the SocialScheduler.

    Args:
      model: Language model for generating observations and reflections.
      player_names: Names of all player entities.
      entities: Mapping of agent name -> entity for reflections via
        entity.act()/entity.observe(). If empty, reflections are skipped.
      agent_descriptions: Mapping of agent name -> personality description to
        condition observation generation.
      events: Initial list of scheduled social events.
      async_conversation_key: Key for AsyncConversationState component.
      locations_key: Key for the Locations component.
      clock_key: Key for the clock component.
      make_observation_key: Key for MakeObservation component.
      pre_act_label: Label for logging.
      home_locations: Mapping of agent name -> home location id. Events whose
        venue is the generic 'home' (e.g. single_rumination) move each
        participant to their own home instead of a nonexistent 'home' place.
    """
    super().__init__()
    self._home_locations = dict(home_locations) if home_locations else {}
    self._model = model
    self._player_names = set(player_names)
    self._agent_descriptions = (
        dict(agent_descriptions) if agent_descriptions else {}
    )
    self._entities: Mapping[
        str, entity_agent_with_logging.EntityAgentWithLogging
    ] = (dict(entities) if entities else {})
    self._events = (
        events
        if isinstance(events, list)
        else (list(events) if events is not None else [])
    )
    self._completed_events: list[SocialEvent] = []
    self._async_conversation_key = async_conversation_key
    self._locations_key = locations_key
    self._clock_key = clock_key
    self._make_observation_key = make_observation_key
    self._pre_act_label = pre_act_label
    self._lock = threading.Lock()

    # Track conversation IDs for scheduled events (for reflection triggering)
    self._conv_id_to_event: dict[int, SocialEvent] = {}
    # Track which events we've already fired this tick
    self._fired_this_tick: set[int] = set()
    self._in_flight_events: dict[int, threading.Event] = {}
    self._last_tick: int = -1

  def _venue_for(self, participant: str, venue: str) -> str:
    """Resolves the generic 'home' venue to the participant's own home."""
    if venue == 'home':
      home = self._home_locations.get(participant)
      if home:
        return home
      logging.warning(
          'SocialScheduler: no home location known for %s; leaving them at'
          ' their current location instead of a placeless "home".',
          participant,
      )
      return ''
    return venue

  def set_entities(
      self,
      entities: Mapping[str, entity_agent_with_logging.EntityAgentWithLogging],
  ) -> None:
    """Set entity references for post-interaction reflections.

    Called after simulation construction when entities are available.

    Args:
      entities: Mapping of agent name -> entity.
    """
    self._entities = dict(entities)

  def add_event(self, event: SocialEvent) -> None:
    """Add a new social event to the schedule.

    Can be called at any time to schedule future interactions:
    - At init from config (first_dates pairing list)
    - By GM when agents agree to meet up
    - Future: Tinder matching system, agent-initiated invitations

    Args:
      event: The social event to schedule.
    """
    with self._lock:
      # Validate participants exist
      for p in event.participants:
        if p not in self._player_names:
          raise ValueError(f'SocialScheduler: Unknown participant {p} in event')
      self._events.append(event)
      logging.info(
          'SocialScheduler: Scheduled %s event for %s at %s (hour=%d)',
          event.theme,
          ', '.join(event.participants),
          event.venue,
          event.tick_hour,
      )

  def get_pending_events(self) -> list[SocialEvent]:
    """Get all pending (unfired) events."""
    with self._lock:
      return [e for e in self._events if not e.fired]

  def get_completed_events(self) -> list[SocialEvent]:
    """Get all completed events."""
    with self._lock:
      return list(self._completed_events)

  def _get_conversation_state(
      self,
  ) -> async_conv.AsyncConversationState | None:
    try:
      return self.get_entity().get_component(
          self._async_conversation_key,
          type_=async_conv.AsyncConversationState,
      )
    except (AttributeError, KeyError):
      return None

  def is_scheduled_for_current_tick(self, agent_name: str) -> bool:
    """Check if an agent is scheduled for a social event at the current tick."""
    current_hour = self._get_current_hour()
    current_day = self._get_current_day()
    if current_hour is None or current_day is None:
      return False
    with self._lock:
      for event in self._events:
        if event.tick_hour == current_hour and event.day == current_day:
          if agent_name in event.participants:
            event_id = id(event)
            # If the event has not yet fired, or is currently in-flight being
            # fired:
            if not event.fired or event_id in self._in_flight_events:
              return True
            # If the event fired, check if the agent is still in an active
            # conversation:
            conv_state = self._get_conversation_state()
            if conv_state is not None:
              return conv_state.is_in_conversation(agent_name)
            return False
    return False

  def wait_for_in_flight_event(
      self, agent_name: str, timeout: float = 10.0
  ) -> bool:
    """Wait if a scheduled event involving agent_name is currently in flight."""
    event_to_wait = None
    with self._lock:
      for event in self._events:
        if agent_name in event.participants:
          event_id = id(event)
          if event_id in self._in_flight_events:
            event_to_wait = self._in_flight_events[event_id]
            break
    if event_to_wait is not None:
      logging.info(
          'SocialScheduler: waiting up to %.1fs for in-flight event for %s...',
          timeout,
          agent_name,
      )
      return event_to_wait.wait(timeout=timeout)
    return True

  def get_scheduled_event_for_agent(
      self, agent_name: str
  ) -> SocialEvent | None:
    """Get the scheduled social event for an agent at the current tick."""
    current_hour = self._get_current_hour()
    current_day = self._get_current_day()
    if current_hour is None or current_day is None:
      return None
    with self._lock:
      for event in self._events:
        if event.tick_hour == current_hour and event.day == current_day:
          if agent_name in event.participants:
            return event
    return None

  def _get_current_hour(self) -> int | None:
    try:
      # The concrete clock component supplies these members; the
      # `BaseComponent` that `get_component` is declared to return does not.
      clock: Any = self.get_entity().get_component(self._clock_key)
      if hasattr(clock, 'current_tick'):
        _ = clock.current_tick
      return clock._current_dt.hour  # pylint: disable=protected-access
    except (AttributeError, KeyError):
      return None

  def _get_current_day(self) -> int | None:
    try:
      clock: Any = self.get_entity().get_component(self._clock_key)
      if hasattr(clock, 'current_tick'):
        _ = clock.current_tick
      return clock._current_dt.day  # pylint: disable=protected-access
    except (AttributeError, KeyError):
      return None

  def _get_current_tick(self) -> int:
    try:
      clock: Any = self.get_entity().get_component(self._clock_key)
      return clock.current_tick
    except (AttributeError, KeyError):
      return -1

  def _get_current_time_str(self) -> str:
    try:
      clock: Any = self.get_entity().get_component(self._clock_key)
      return clock.get_pre_act_value().strip()
    except (AttributeError, KeyError):
      return ''

  def _generate_observation(self, event: SocialEvent) -> str:
    """Generate a shared observation for a social event using the LLM.

    Args:
      event: The social event to generate an observation for.

    Returns:
      A scene-setting observation string.
    """
    if event.shared_observation:
      return event.shared_observation

    doc = interactive_document.InteractiveDocument(self._model)
    p1 = event.participants[0]
    p2 = event.participants[1] if len(event.participants) > 1 else ''

    p1_desc = self._agent_descriptions.get(p1, '')
    p2_desc = self._agent_descriptions.get(p2, '') if p2 else ''

    current_time = self._get_current_time_str()
    context = f'{current_time} at {event.venue}'

    if event.prompt_type == 'first_date':
      prompt = FIRST_DATE_SETUP_PROMPT.format(
          context=context,
          player1=p1,
          player1_background=p1_desc,
          player1_wearing_statement=event.p1_wearing,
          player2=p2,
          player2_background=p2_desc,
          player2_wearing_statement=event.p2_wearing,
          venue=event.venue,
          date_theme=event.theme,
      )
    elif event.prompt_type == 'second_date':
      prompt = SECOND_DATE_SETUP_PROMPT.format(
          context=context,
          player1=p1,
          player1_background=p1_desc,
          player1_wearing_statement=event.p1_wearing,
          player2=p2,
          player2_background=p2_desc,
          player2_wearing_statement=event.p2_wearing,
          venue=event.venue,
          date_theme=event.theme,
      )
    elif event.prompt_type == 'friend_meetup':
      prompt = FRIEND_MEETUP_SETUP_PROMPT.format(
          context=context,
          player1=p1,
          player1_background=p1_desc,
          player1_wearing_statement=event.p1_wearing,
          player2=p2,
          player2_background=p2_desc,
          player2_wearing_statement=event.p2_wearing,
          venue=event.venue,
          activity_theme=event.theme,
      )
    elif event.prompt_type == 'spouse_meetup':
      prompt = SPOUSE_MEETUP_SETUP_PROMPT.format(
          context=context,
          player1=p1,
          player1_background=p1_desc,
          player1_wearing_statement=event.p1_wearing,
          player2=p2,
          player2_background=p2_desc,
          player2_wearing_statement=event.p2_wearing,
          venue=event.venue,
      )
    elif event.prompt_type == 'single_rumination':
      prompt = SINGLE_RUMINATION_PROMPT.format(
          context=context,
          player_name=p1,
          player_background=p1_desc,
          player_wearing_statement=event.p1_wearing,
          venue=event.venue,
      )
    else:
      # Fallback to generic prompt if type unknown
      prompt = f"""
Current Context: {context}
Theme: {event.theme}

Background and Relevant History for {p1}:
{p1_desc}

Background and Relevant History for {p2}:
{p2_desc}

Instructions:
Generate a shared scenario for a {event.theme} that brings {p1} and {p2} together for a conversation right now at {event.venue}.
Write 5-7 sentences.
"""

    observation = doc.open_question(
        question=prompt,
        max_tokens=750,
    )

    return observation

  def _fire_event(
      self,
      event: SocialEvent,
      observation: str | None = None,
  ) -> bool:
    """Fire a scheduled social event.

    Args:
      event: The event to trigger.
      observation: Optional pre-generated observation string.

    Returns:
      True if the event was successfully fired, False otherwise.
    """
    conv_state = self._get_conversation_state()
    if conv_state is None:
      logging.warning('SocialScheduler: No conversation state available')
      return False

    # Check if this event was already fired (e.g. by another game master).
    all_in_conv = all(
        conv_state.is_in_conversation(p) for p in event.participants
    )
    if all_in_conv:
      conv_id = getattr(event, 'conv_id', None)
      if conv_id is None:
        if hasattr(conv_state, 'get_conversation_for'):
          active_c = conv_state.get_conversation_for(event.participants[0])
          if active_c:
            conv_id = getattr(active_c, 'conv_id', None)
        elif hasattr(conv_state, '_get_active_conv_for_player'):
          # pylint: disable=protected-access
          active_c = conv_state._get_active_conv_for_player(
              event.participants[0]
          )
          # pylint: enable=protected-access
          if active_c:
            conv_id = active_c.get('conv_id')
      if conv_id is not None and observation:
        if hasattr(conv_state, 'update_context'):
          conv_state.update_context(conv_id, observation)
      logging.info(
          'SocialScheduler: All participants %s already in active conversation;'
          ' event %s (conv %s) updated with observation.',
          ', '.join(event.participants),
          event.theme,
          conv_id,
      )
      with self._lock:
        if conv_id is not None:
          self._conv_id_to_event[conv_id] = event
        event.fired = True
        if observation:
          event.shared_observation = observation
        if event not in self._completed_events:
          self._completed_events.append(event)
      return True

    # Check that no participant is already in a conversation
    for p in event.participants:
      if conv_state.is_in_conversation(p):
        logging.info(
            'SocialScheduler: %s already in conversation, deferring event',
            p,
        )
        return False

    # Generate shared observation if not provided
    if observation is None:
      observation = self._generate_observation(event)

    # Move participants to venue
    try:
      locations = self.get_entity().get_component(self._locations_key)
      state = locations.get_state()
      entity_locs = state.get('entity_locations')
      if isinstance(entity_locs, dict):
        for p in event.participants:
          venue = self._venue_for(p, event.venue)
          if venue:
            entity_locs[p] = venue
        locations.set_state(state)
        logging.info(
            'SocialScheduler: Moved %s to %s',
            ', '.join(event.participants),
            event.venue,
        )
    except (AttributeError, KeyError) as e:
      logging.warning('SocialScheduler: Failed to move agents: %s', e)

    # Create conversation with the event's custom boring-check threshold.
    # Scheduled dates use terminate_check_min_turn=30 so 30 turns run
    # unimpeded before the GM starts asking if the conversation is boring.
    # The generated observation is passed directly as context.
    conv_id = conv_state.create_conversation(
        participants=event.participants,
        location=event.venue,
        max_turns=event.max_turns,
        terminate_check_min_turn=event.terminate_check_min_turn,
        context=observation,
    )

    # Track for reflections
    with self._lock:
      self._conv_id_to_event[conv_id] = event
      event.fired = True
      self._completed_events.append(event)

    logging.info(
        'SocialScheduler: Fired %s event (conv %d) for %s at %s: %s',
        event.theme,
        conv_id,
        ', '.join(event.participants),
        event.venue,
        observation[:100],
    )
    return True

  def _check_for_completed_conversations(self) -> None:
    """Check if any scheduled conversations have ended and run reflections."""
    conv_state = self._get_conversation_state()
    if conv_state is None:
      return

    with self._lock:
      completed_conv_ids = []
      for conv_id, event in self._conv_id_to_event.items():
        conv = conv_state.get_conversation_for(event.participants[0])
        # If the conversation is no longer active (or the agent isn't in
        # it anymore), it has ended.
        if conv is None or not conv.active or conv.conv_id != conv_id:
          completed_conv_ids.append(conv_id)

      for conv_id in completed_conv_ids:
        event = self._conv_id_to_event.pop(conv_id)
        self._run_post_interaction_reflections(event)

  def _run_post_interaction_reflections(
      self,
      event: SocialEvent,
  ) -> None:
    """Run 4-step post-interaction reflections using entity.act/observe.

    Uses the DIAL pattern from run_concordia_dates.py: each reflection
    step routes through the agent's full cognitive pipeline (memory,
    personality, backstory) via entity.act(), then persists the result
    via entity.observe().

    Steps:
    1. Interaction summary
    2. Visual first impression
    3. Comparative evaluation
    4. Numeric rating

    Args:
      event: The completed social event.
    """
    if not self._entities:
      logging.info(
          'SocialScheduler: No entity references, skipping reflections'
      )
      return

    for participant in event.participants:
      entity = self._entities.get(participant)
      if entity is None:
        logging.warning(
            'SocialScheduler: No entity for %s, skipping reflections',
            participant,
        )
        continue

      others = [p for p in event.participants if p != participant]
      others_str = ', '.join(others)

      # 1. Date/interaction summary
      action_spec = entity_lib.free_action_spec(
          call_to_action=(
              f"Summarize {participant}'s {event.theme} with {others_str}. "
              'What happened? What did they talk about?'
          ),
      )
      summary = entity.act(action_spec=action_spec)
      entity.observe(f'[Reflection] {summary}')

      # 2. Visual first impression
      action_spec = entity_lib.free_action_spec(
          call_to_action=(
              f'Describe your visual first impression of {others_str}. '
              'Based purely on this visual information, what is your gut '
              'feeling or assessment of them? '
              'Are you drawn to them?'
          ),
      )
      impression = entity.act(action_spec=action_spec)
      entity.observe(f'[Reflection] {impression}')

      # 3. Comparative evaluation
      action_spec = entity_lib.free_action_spec(
          call_to_action=(
              f'How would {participant} reflect on {others_str}? '
              f'What would they evaluate about {others_str} in terms of '
              'their strengths and weaknesses as a person?'
          ),
      )
      evaluation = entity.act(action_spec=action_spec)
      entity.observe(f'[Reflection] {evaluation}')

      # 4. Numeric rating
      action_spec = entity_lib.free_action_spec(
          call_to_action=(
              'Based on the interaction and the reflections, '
              f'how would {participant} rate {others_str} '
              'from 0.0 to 10.0?'
          ),
      )
      rating_str = entity.act(action_spec=action_spec)
      match = re.search(r'\b\d+(\.\d+)?\b', rating_str)
      if match:
        rating = match.group(0)
        date_rating = (
            f'[Reflection] {participant} rated {others_str} as {rating}/10'
        )
      else:
        date_rating = f'[Reflection] {rating_str}'
      entity.observe(date_rating)

      logging.info(
          'SocialScheduler: Reflection for %s about %s: rating=%s',
          participant,
          others_str,
          match.group(0) if match else 'N/A',
      )

  def fire_pending_events_for_current_tick(self) -> str:
    """Check and fire any scheduled events for the current simulation tick.

    Creates the conversation, moves agents to venue, and sets the shared
    setup observation context BEFORE regular GM observations are generated.

    Events are fired in parallel to reduce latency: the LLM scene-generation
    call in _generate_observation takes ~5-8s per event, so sequential firing
    of N events would take N*5s.  Parallel firing brings this down to ~5s.

    Returns:
      Observation string for fired events, or empty string.
    """
    current_hour = self._get_current_hour()
    current_day = self._get_current_day()
    if current_hour is None or current_day is None:
      return ''

    current_tick = self._get_current_tick()

    # Reset fired tracking on tick change
    with self._lock:
      if current_tick != self._last_tick:
        self._fired_this_tick.clear()
        self._last_tick = current_tick

    # Phase 1: Atomically claim all eligible events for this tick.
    # Strictly enforce pairwise-disjoint participant sets so each agent is
    # claimed for at most ONE social event in this tick.
    eligible_events = []
    with self._lock:
      pending = [e for e in self._events if not e.fired]
      claimed_agents_this_tick = set()
      for event in pending:
        event_id = id(event)
        if (
            event_id in self._fired_this_tick
            or event_id in self._in_flight_events
            or event.fired
        ):
          continue
        if event.tick_hour == current_hour and event.day == current_day:
          if any(p in claimed_agents_this_tick for p in event.participants):
            logging.info(
                'SocialScheduler: Skipping event %s (%s); participant already'
                ' claimed for this tick.',
                event.theme,
                ', '.join(event.participants),
            )
            continue
          self._fired_this_tick.add(event_id)
          self._in_flight_events[event_id] = threading.Event()
          claimed_agents_this_tick.update(event.participants)
          eligible_events.append(event)

    if not eligible_events:
      return ''

    # Phase 1.5: Eagerly create conversation records and move participants to
    # venues BEFORE generating LLM observations.
    # This immediately routes any agent polling GetTurnSetup to their assigned
    # conversation shard, eliminating the race condition where agents fall back
    # to island rules during observation generation.
    conv_state = self._get_conversation_state()
    locations = None
    try:
      locations = self.get_entity().get_component(self._locations_key)
    except (AttributeError, KeyError):
      pass

    for event in eligible_events:
      if locations is not None:
        try:
          state = locations.get_state()
          entity_locs = state.get('entity_locations')
          if isinstance(entity_locs, dict):
            for p in event.participants:
              venue = self._venue_for(p, event.venue)
              if venue:
                entity_locs[p] = venue
            locations.set_state(state)
            logging.info(
                'SocialScheduler: Eagerly moved %s to %s',
                ', '.join(event.participants),
                event.venue,
            )
        except Exception as e:  # pylint: disable=broad-except
          logging.warning(
              'SocialScheduler: Failed to move agents in Phase 1.5: %s', e
          )

      if conv_state is not None:
        try:
          initial_ctx = event.shared_observation or '__GENERATING__'
          conv_id = conv_state.create_conversation(
              participants=event.participants,
              location=event.venue,
              max_turns=event.max_turns,
              terminate_check_min_turn=event.terminate_check_min_turn,
              context=initial_ctx,
          )
          event.conv_id = conv_id
          with self._lock:
            self._conv_id_to_event[conv_id] = event
          logging.info(
              'SocialScheduler: Eagerly created conversation %d for %s at %s',
              conv_id,
              ', '.join(event.participants),
              event.venue,
          )
        except Exception as e:  # pylint: disable=broad-except
          logging.exception(
              'SocialScheduler: Failed eager conversation creation for %s: %s',
              event.theme,
              e,
          )

    # Phase 2: Pre-generate observations in parallel (LLM only, thread-safe),
    # then fire all claimed events sequentially on the main thread to avoid
    # races when several events update shared state concurrently.
    observations = {}
    if len(eligible_events) > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(len(eligible_events), 4)
      ) as executor:
        future_to_event = {
            executor.submit(self._generate_observation, e): e
            for e in eligible_events
            if not e.shared_observation
        }
        for future in concurrent.futures.as_completed(future_to_event):
          e = future_to_event[future]
          try:
            observations[id(e)] = future.result()
          except Exception as exc:  # pylint: disable=broad-except
            logging.warning(
                'SocialScheduler: Parallel observation generation failed for'
                ' %s: %s; will retry inline',
                e.theme,
                exc,
            )

    fired_names = []
    for event in eligible_events:
      event_id = id(event)
      obs = observations.get(event_id, None)
      if not obs and not event.shared_observation:
        obs = (
            f'You are meeting with {", ".join(event.participants)} at'
            f' {event.venue}.'
        )
      try:
        success = self._fire_event(event, observation=obs)
      except Exception as e:  # pylint: disable=broad-except
        logging.exception(
            'SocialScheduler: _fire_event raised for %s: %s',
            event.theme,
            e,
        )
        success = False

      with self._lock:
        done_ev = self._in_flight_events.pop(event_id, None)
        if success:
          event.fired = True
          fired_names.append(f'{event.theme}: {", ".join(event.participants)}')
        else:
          # Release claim so a future tick can retry.
          self._fired_this_tick.discard(event_id)
          event.fired = False

      if done_ev is not None:
        done_ev.set()

    result = ''
    if fired_names:
      result = f'Fired {len(fired_names)} events: {"; ".join(fired_names)}'

    if result:
      self._logging_channel({
          'Key': self._pre_act_label,
          'Summary': result[:100] if result else '',
          'Value': result,
      })
    return result

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    """Check for scheduled events to fire.

    Fires during MAKE_OBSERVATION and RESOLVE steps, checking if the
    current hour matches any pending event's tick_hour.

    Args:
      action_spec: The current action specification.

    Returns:
      An empty string (side effects only).
    """
    if action_spec.output_type not in (
        entity_lib.OutputType.MAKE_OBSERVATION,
        entity_lib.OutputType.RESOLVE,
    ):
      return ''

    self.fire_pending_events_for_current_tick()
    return ''

  def post_act(self, event: str) -> str:
    """Check for completed scheduled conversations."""
    self._check_for_completed_conversations()
    return ''

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      events_data = []
      for e in self._events:
        events_data.append({
            'participants': list(e.participants),
            'venue': e.venue,
            'theme': e.theme,
            'tick_hour': e.tick_hour,
            'max_turns': e.max_turns,
            'terminate_check_min_turn': e.terminate_check_min_turn,
            'shared_observation': e.shared_observation,
            'scheduled_by': e.scheduled_by,
            'scheduled_at_tick': e.scheduled_at_tick,
            'fired': e.fired,
        })
      completed_data = []
      for e in self._completed_events:
        completed_data.append({
            'participants': list(e.participants),
            'venue': e.venue,
            'theme': e.theme,
            'tick_hour': e.tick_hour,
            'max_turns': e.max_turns,
            'terminate_check_min_turn': e.terminate_check_min_turn,
            'fired': True,
        })
      return {
          'events': events_data,
          'completed_events': completed_data,
          'last_tick': self._last_tick,
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._last_tick = component_state.as_int(state, 'last_tick', -1)
      self._events = [
          _event_from_state(e_data)
          for e_data in component_state.as_dict_list(state, 'events')
      ]
      self._completed_events = [
          _event_from_state(e_data, fired=True)
          for e_data in component_state.as_dict_list(
              state, 'completed_events'
          )
      ]
