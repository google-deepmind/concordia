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

"""A prefab for a game master for games set in a specific location."""

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import threading
from typing import Any

# from absl import logging # Uncomment if you need specific logging
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.components.game_master import next_acting
from concordia.components.game_master import terminate as terminate_components
from concordia.contrib.components.game_master import gm_working_memory
from concordia.contrib.components.game_master import location_based_filter
from concordia.contrib.components.game_master import narrative_event_resolution
from concordia.contrib.components.game_master import npc_event_generator
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib


SimultaneousNarrativeEventResolution = (
    narrative_event_resolution.SimultaneousNarrativeEventResolution
)
NarrativeHistoryManager = narrative_event_resolution.NarrativeHistoryManager


class _FixedIntervalClock:
  """A simple fixed-interval clock for tracking simulation time."""

  def __init__(
      self,
      start: datetime.datetime,
      step_size: datetime.timedelta,
  ):
    self._start = start
    self._step_size = step_size
    self._step = 0
    self._step_lock = threading.Lock()

  def advance(self):
    with self._step_lock:
      self._step += 1

  def now(self) -> datetime.datetime:
    with self._step_lock:
      return self._start + self._step * self._step_size


class FixedIncrementClock(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that advances time by a fixed increment."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      start_time: str,
      increment_minutes: int = 3,
      pre_act_label: str = '\nCurrent time',
      use_variable_increments: bool = True,
      variable_increment_rules: dict[int, int] | None = None,
  ):
    """Initializes the component.

    Args:
      model: The language model to use (not used for time increment, but for
        logging).
      start_time: The initial time of the clock in a format like "Monday, March
        3, 2026 at 8:30 AM".
      increment_minutes: The base time unit in minutes for clock advancement.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
      use_variable_increments: If True, use rules to advance time in post_act.
      variable_increment_rules: A dictionary mapping start hour (0-23) to
        increment in minutes for variable time advancement. If None, default
        rules are used: 15 min for 8am-8pm, 60 min for 8pm-11pm, 180 min for
        11pm-8am.
    """
    self._pre_act_label = pre_act_label
    self._model = model
    self._increment_minutes = increment_minutes
    self._use_variable_increments = use_variable_increments

    default_rules = {0: 180, 8: 15, 20: 30, 23: 45}
    rules_dict = (
        variable_increment_rules
        if variable_increment_rules is not None
        else default_rules
    )
    self._increment_rules = sorted(rules_dict.items())

    parsed_start_time = self._parse_time(start_time)
    step_timedelta = datetime.timedelta(minutes=self._increment_minutes)
    self._game_clock = _FixedIntervalClock(
        start=parsed_start_time, step_size=step_timedelta
    )

  def _parse_time(self, time_str: str) -> datetime.datetime:
    """Parses a time string into a datetime object."""
    # Example format: "Monday, March 3, 2026 at 8:30 AM"
    try:
      return datetime.datetime.strptime(time_str, '%A, %B %d, %Y at %I:%M %p')
    except ValueError:
      # Fallback for other formats, or if parsing fails.
      default_time = datetime.datetime(2026, 3, 3, 8, 30)
      self._log(
          f"Warning: Could not parse time string '{time_str}'. Using default"
          f' time: {self._format_time(default_time)}'
      )
      return default_time

  def _format_time(self, dt: datetime.datetime) -> str:
    """Formats a datetime object into a time string."""
    return dt.strftime('%A, %B %d, %Y at %I:%M %p')

  def get_pre_act_label(self) -> str:
    """Returns the key used as a prefix in the string returned by `pre_act`."""
    return self._pre_act_label

  def get_pre_act_value(self) -> str:
    """Returns the current world state."""
    return self._format_time(self._game_clock.now()) + '\n'

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = self.get_pre_act_value()
    if self._logging_channel:
      self._logging_channel({
          'Key': self._pre_act_label,
          'Summary': result,
          'Value': result,
      })
    return result

  def post_act(
      self,
      event: str,
  ) -> str:
    return ''

  def advance_by_minutes(self, minutes: int):
    """Advances the clock by the specified number of minutes."""
    if self._use_variable_increments:
      now = self._game_clock.now()
      hour = now.hour
      minutes_to_advance = self._increment_minutes
      for start_hour, increment in reversed(self._increment_rules):
        if hour >= start_hour:
          minutes_to_advance = increment
          break
      self._advance_by_minutes_internal(minutes_to_advance)
    else:
      self._advance_by_minutes_internal(minutes)

  def _advance_by_minutes_internal(self, minutes: int):
    """Internal helper to advance the clock."""
    if minutes <= 0:
      return

    # Calculate how many steps to advance.
    steps_to_advance = minutes // self._increment_minutes
    if steps_to_advance == 0 and minutes > 0:
      # Ensure at least one step if minutes > 0, even if less than increment
      steps_to_advance = 1

    old_time = self._game_clock.now()
    for _ in range(steps_to_advance):
      self._game_clock.advance()
    new_time = self._game_clock.now()

    old_time_str = self._format_time(old_time)
    new_time_str = self._format_time(new_time)
    self._log(
        f'Clock advanced by {minutes} minutes. Changed from'
        f' {old_time_str} to {new_time_str}'
    )

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'start_time': self._format_time(self._game_clock._start),  # pylint: disable=protected-access
        'step': self._game_clock._step,  # pylint: disable=protected-access
        'increment_minutes': self._increment_minutes,
        'use_variable_increments': self._use_variable_increments,
        'increment_rules': self._increment_rules,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    start_time_str = state.get('start_time')
    step = state.get('step')
    increment_minutes = state.get('increment_minutes')
    increment_rules = state.get('increment_rules')

    if (
        start_time_str is not None
        and step is not None
        and increment_minutes is not None
        and increment_rules is not None
    ):
      parsed_start_time = self._parse_time(start_time_str)
      step_timedelta = datetime.timedelta(minutes=increment_minutes)
      self._game_clock = _FixedIntervalClock(
          start=parsed_start_time, step_size=step_timedelta
      )
      self._game_clock._step = step  # pylint: disable=protected-access
      self._increment_minutes = increment_minutes
      self._use_variable_increments = state.get('use_variable_increments', True)
      self._increment_rules = increment_rules
    else:
      self._log('Warning: Could not fully restore FixedIncrementClock state.')

  def _log(self, message: str):
    """Logs a message to the logging channel."""
    if self._logging_channel:
      self._logging_channel({
          'Key': self._pre_act_label,
          'Summary': message,
          'Value': message,
      })

# ==============================================================================


_DEFAULT_CLOCK_DESCRIPTION = (
    'The passing of time can be conveyed using any convenient '
    'feature of the environment, e.g. a physical clock, the angle '
    "of the sun, extent of a candle's melting, phase of the moon, "
    'agricultural season, elapsed time since an event, etc. Whenever '
    'possible, try to track the day and year as well as the time '
    'within the day. To determine the passing of time, try to make '
    'reasonable inferences about the amount of time that would '
    'most likely have elapsed between the previous event and the '
    'latest event, taking into account the number of simulation '
    'steps taken.'
)


@dataclasses.dataclass
class GameMasterSimultaneous(prefab_lib.Prefab):
  """A prefab entity implementing a game master for games with simultaneous event resolution."""

  description: str = (
      'A general game master for games with simultaneous event resolution.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'default rules',
          'clock_description': _DEFAULT_CLOCK_DESCRIPTION,
          'start_time': None,
          'locations': '',
          'game_rules': '',
          'allow_early_termination': False,
          'time_period_minutes': 15,
          'extra_components': {},
          'extra_components_index': {},
          'initial_causal_states': {},
          'ambient_location_details': {},
          'use_gm_working_memory': True,
          'use_location_based_filter': True,
          'use_narrative_history_manager': True,
          'use_npc_events': True,
          'npc_scenario_context': 'A generic setting.',
          'npc_event_probability': 0.15,
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the simultaneous game master entity."""

    # 1. Unpack basic parameters
    clock_description = self.params.get(
        'clock_description', _DEFAULT_CLOCK_DESCRIPTION
    )
    start_time = self.params.get('start_time', None)
    game_rules = self.params.get('game_rules', '')
    time_period_minutes = self.params.get('time_period_minutes', 15)
    name = self.params.get('name', 'default rules')
    extra_components = self.params.get('extra_components', {})
    extra_components_index = self.params.get('extra_components_index', {})
    use_gm_working_memory = self.params.get('use_gm_working_memory', True)
    use_location_based_filter = self.params.get(
        'use_location_based_filter', True
    )
    use_narrative_history_manager = self.params.get(
        'use_narrative_history_manager', True
    )
    use_npc_events = self.params.get('use_npc_events', True)

    # 2. Core GM memory and instruction components
    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    instructions_key = 'instructions'
    instructions = gm_components.instructions.Instructions()

    examples_synchronous_key = 'examples'
    examples_synchronous = gm_components.instructions.ExamplesSynchronous()

    player_names = [entity.name for entity in self.entities]
    player_characters_key = 'player_characters'
    player_characters = gm_components.instructions.PlayerCharacters(
        player_characters=player_names,
    )

    # 3. Observation processing components
    observation_to_memory_key = 'observation_to_memory'
    observation_to_memory = actor_components.observation.ObservationToMemory()

    observation_component_key = (
        actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = actor_components.observation.LastNObservations(
        history_length=1000,
    )

    # 4. World State, Location, and Rule tracking
    display_events_key = 'display_events'
    display_events = gm_components.event_resolution.DisplayEvents(
        model=model,
        pre_act_label=(
            'Story so far (ordered from oldest to most recent events)'
        ),
    )

    relevant_memories_key = 'relevant_memories'
    relevant_memories = (
        actor_components.all_similar_memories.AllSimilarMemories(
            model=model,
            components=[display_events_key],
            num_memories_to_retrieve=150,
            pre_act_label='Background info',
        )
    )

    locations_constant_key = 'locations_constant'
    locations_constant = actor_components.constant.Constant(
        self.params.get('locations'), pre_act_label='Locations'
    )

    clock_constant_key = 'clock_constant'
    clock_constant = actor_components.constant.Constant(
        clock_description, pre_act_label='Clock description'
    )

    game_rules_key = 'game_rules'
    game_rules_constant = actor_components.constant.Constant(
        game_rules, pre_act_label='Game Rules'
    )

    locations_key = 'locations'
    entity_locations = gm_components.world_state.Locations(
        model=model,
        entity_names=player_names,
        prompt=self.params.get('locations', ''),
        components=[
            instructions_key,
            locations_constant_key,
            game_rules_key,
            player_characters_key,
            relevant_memories_key,
            display_events_key,
        ],
    )

    world_state_key = 'world_state'
    world_state = gm_components.world_state.WorldState(
        model=model,
        components=[
            instructions_key,
            locations_constant_key,
            locations_key,
            game_rules_key,
            player_characters_key,
            relevant_memories_key,
            display_events_key,
        ],
        pre_act_label='World state',
    )

    # 5. Time and Pacing
    dynamic_pacing_key = 'dynamic_pacing'
    dynamic_pacing = actor_components.constant.Constant(
        f'{time_period_minutes} minutes', pre_act_label='Time Increment'
    )

    generative_clock_key = 'generative_clock'
    generative_clock = FixedIncrementClock(
        model=model,
        start_time=start_time,
        increment_minutes=time_period_minutes,
        pre_act_label='\nCurrent time',
    )

    # 5d. Location-Based Partial Observability Filter
    location_filter_key = 'location_filter'
    location_filter_component = None
    if use_location_based_filter:
      location_filter_component = location_based_filter.LocationBasedFilter(
          model=model,
          entity_names=player_names,
          pre_act_label='Partial Observability Enforcement',
      )

    # 5e. Narrative History Manager (for checkpointing and debugging)
    narrative_history_key = 'narrative_history'
    narrative_history_component = None
    if use_narrative_history_manager:
      narrative_history_component = (
          narrative_event_resolution.NarrativeHistoryManager()
      )

    # 5f. Game Master Working Memory
    gm_working_memory_key = 'gm_working_memory'
    gm_working_memory_component = None
    if use_gm_working_memory:
      gm_working_memory_component = gm_working_memory.GMWorkingMemory(
          model=model,
          memory_component_key=memory_component_key,
          components=[relevant_memories_key],
          verbose=True,
      )

    # 5g. NPC Event Generator
    npc_event_key = 'npc_event'
    npc_event_component = None
    if use_npc_events:
      npc_event_component = npc_event_generator.NpcEventGenerator(
          model=model,
          clock=generative_clock,
          scenario_context=self.params.get(
              'npc_scenario_context', 'A generic setting.'
          ),
          event_probability=self.params.get('npc_event_probability', 0.15),
          verbose=True,
      )

    # 6. Action Specification (How players should structure their turns)
    call_to_action_text = (
        'IMPORTANT CONTEXT: This simulation step represents'
        f' {time_period_minutes} minutes of real time. '
        'This means your plan must contain enough specific detail to fill a'
        f' full {time_period_minutes}-minute '
        f'period of activity - not a single quick action, but a realistic'
        f' sequence of events, movements, '
        'and interactions that would naturally take'
        f' {time_period_minutes} minutes to complete. Your plan '
        f"will be combined with other agents' plans to create a detailed"
        f' narrative of exactly what happens '
        f'in these {time_period_minutes} minutes.\\n\\n'
        f'What would {{name}} do in the next {time_period_minutes} minutes?'
        ' Write out a detailed plan with '
        f'high-level objectives that {{name}} will try to complete. '
        'Also clearly state what success looks like for this plan. '
        f'Your plan should be sufficiently detailed '
        f'that it can be expanded into a rich {time_period_minutes}-minute'
        ' narrative showing specific '
        f'actions, movements, dialogue, and observable behaviors. Write at'
        f' least 10 sentences describing '
        'what {name} will do throughout this full'
        f' {time_period_minutes}-minute period. Prioritize the '
        f'objectives in numbered form. Avoid passive or repetitive actions -'
        f' the plan should move the plot '
        f"forward. Check {{name}}'s previous actions and ONLY repeat activities"
        f' if it makes sense for '
        '{name} to be doing those things for more than'
        f' {time_period_minutes} minutes continuously. '
        f'IMPORTANT: Never use the first person ("I", "me", "my") or any'
        f' pronouns ("he", "she", "they", '
        f'"him", "her", "them", etc.). ALWAYS use {{name}}\'s specific name.'
    )

    default_action_spec = entity_lib.ActionSpec(
        call_to_action=call_to_action_text,
        output_type=entity_lib.OutputType.FREE,
        options=(),
        tag='action',
    )

    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    next_action_spec = gm_components.next_acting.FixedActionSpec(
        action_spec=default_action_spec
    )

    next_actor_key = next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_actor = next_acting.NextActingAllEntities(
        player_names=player_names,
    )

    # 7. Observation Generation for Players
    make_observation_key = (
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )
    make_observation = gm_components.make_observation.MakeObservation(
        model=model,
        player_names=player_names,
        components=[
            generative_clock_key,
            instructions_key,
            player_characters_key,
            relevant_memories_key,
            display_events_key,
            world_state_key,
            game_rules_key,
        ],
        reformat_observations_in_specified_style='',
    )

    event_resolution_prompt_key = 'event_resolution_prompt'
    event_resolution_prompt = actor_components.constant.Constant(
        'When summarizing events, begin with the time and date, followed by '
        f' In the last {time_period_minutes} minutes, and then describe what'
        ' happened. Write out everything that happened in the last'
        f' {time_period_minutes} minutes that you observed. Do not only write'
        ' about what others did and the consequences of their actions, but'
        ' also critically what you did. Be sure to note all of your actions'
        ' and the consequences of your actions. For the actions that you took,'
        ' be sure to structure them as {{your name}} did behavior, and'
        ' this was the consequence: {{consequence}}. When possible, do'
        ' this for others as well. Write out everything that was observed. '
    )

    # 8. Event Resolution (The Core Narrative Engine)
    # The narrator needs to see ALL these things to make a good ruling:
    event_resolution_components = [
        instructions_key,
        player_characters_key,
        relevant_memories_key,
        display_events_key,
        locations_key,
        game_rules_key,
        generative_clock_key,
        event_resolution_prompt_key,
    ]
    if use_gm_working_memory:
      event_resolution_components.append(gm_working_memory_key)
    if use_location_based_filter:
      event_resolution_components.append(location_filter_key)
    if use_npc_events:
      event_resolution_components.append(npc_event_key)

    event_resolution_key = (
        gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
    )
    # Instantiating your new custom component:
    narrative_event_resolution_component = SimultaneousNarrativeEventResolution(
        model=model,
        player_names=player_names,
        memory_component_key=memory_component_key,
        make_observation_component_key=make_observation_key,
        world_state_key=world_state_key,
        generative_clock_key=generative_clock_key,
        location_filter_key=location_filter_key
        if use_location_based_filter
        else '',
        narrative_history_key=narrative_history_key
        if use_narrative_history_manager
        else '',
        gm_working_memory_key=gm_working_memory_key
        if use_gm_working_memory
        else '',
        components=event_resolution_components,
        verbose=True,
        time_period_minutes=time_period_minutes,
    )

    # 9. Termination Logic
    allow_early_termination = self.params.get('allow_early_termination', False)
    terminate_key = terminate_components.DEFAULT_TERMINATE_COMPONENT_KEY
    if allow_early_termination:
      terminate_component = terminate_components.Terminate()
    else:
      terminate_component = terminate_components.NeverTerminate()

    # 10. Assemble the Game Master entity
    components_of_game_master = {
        terminate_key: terminate_component,
        dynamic_pacing_key: dynamic_pacing,
        instructions_key: instructions,
        examples_synchronous_key: examples_synchronous,
        player_characters_key: player_characters,
        relevant_memories_key: relevant_memories,
        observation_component_key: observation,
        observation_to_memory_key: observation_to_memory,
        display_events_key: display_events,
        clock_constant_key: clock_constant,
        game_rules_key: game_rules_constant,
        generative_clock_key: generative_clock,
        locations_constant_key: locations_constant,
        event_resolution_prompt_key: event_resolution_prompt,
        locations_key: entity_locations,
        world_state_key: world_state,
        memory_component_key: memory_component,
        make_observation_key: make_observation,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        event_resolution_key: narrative_event_resolution_component,
    }
    if use_narrative_history_manager:
      components_of_game_master[narrative_history_key] = (
          narrative_history_component
      )
    if use_location_based_filter:
      components_of_game_master[location_filter_key] = location_filter_component
    if use_gm_working_memory:
      components_of_game_master[gm_working_memory_key] = (
          gm_working_memory_component
      )
    if use_npc_events:
      components_of_game_master[npc_event_key] = npc_event_component

    # 11. Define Execution Order
    component_order = list(components_of_game_master.keys())
    ordered_keys = [
        dynamic_pacing_key,
        next_actor_key,
        next_action_spec_key,
        event_resolution_key,
        world_state_key,
    ]
    # Ensure ordered keys are at the end (before terminate)
    for key in ordered_keys:
      if key in component_order:
        component_order.remove(key)
    if terminate_key in component_order:
      component_order.remove(terminate_key)

    component_order.extend(ordered_keys)
    component_order.append(terminate_key)

    # 12. Insert extra components if provided in params
    if extra_components:
      components_of_game_master.update(extra_components)
      if extra_components_index:
        for component_name in extra_components.keys():
          component_order.insert(
              extra_components_index[component_name],
              component_name,
          )
      else:
        component_order.extend(extra_components.keys())

    # 13. Create the Act Component (the "Brain" of the GM)
    act_component = gm_components.switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    # 14. Return the final entity
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components_of_game_master,
    )
