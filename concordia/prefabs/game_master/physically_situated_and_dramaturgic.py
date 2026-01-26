# Copyright 2025 DeepMind Technologies Limited.
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

"""A prefab game master combining physical time/place simulation with scenes.

This game master combines the full functionality of situated_in_time_and_place
(clock, locations, world state, physical action resolution) with scene support
from dialogic_and_dramaturgic (SceneTracker, scene-based actor selection).

Use this for scenarios where agents need to take physical actions while
following structured scene progressions.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains as thought_chains_lib
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib

DEFAULT_NAME = 'physical action rules'

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
class GameMaster(prefab_lib.Prefab):
  """A prefab game master for physical time/place simulations with scenes.

  This game master combines:
  - Full physical simulation from situated_in_time_and_place:
    - Clock tracking (generative clock)
    - Location tracking for all entities
    - World state representation
    - Physical action resolution via AccountForAgencyOfOthers
  - Scene support from dialogic_and_dramaturgic:
    - SceneTracker for scene progression
    - Scene-based actor selection (NextActingFromSceneSpec)
    - Scene-based action spec (NextActionSpecFromSceneSpec)

  Use this for scenarios where agents may take physical actions like moving,
  fighting, or interacting with objects, while following structured scenes.
  """

  description: str = (
      'A game master for physical time/place simulations with scene support. '
      'Combines full world simulation with structured scene progressions.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': DEFAULT_NAME,
          'scenes': (),
          'next_game_master_name': None,
          'extra_event_resolution_steps': '',
          'clock_description': _DEFAULT_CLOCK_DESCRIPTION,
          'start_time': '',
          'locations': '',
          'extra_components': {},
          'extra_components_index': {},
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: list[str] | associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a game master for physical time/place simulations with scenes.

    Args:
      model: The language model to use for game master.
      memory_bank: Provide a memory_bank.

    Returns:
      A game master entity.
    """
    name = self.params.get('name', DEFAULT_NAME)

    # Get clock and location parameters
    clock_description = self.params.get(
        'clock_description', _DEFAULT_CLOCK_DESCRIPTION
    )
    assert isinstance(clock_description, str)
    start_time = self.params.get('start_time', '')
    assert isinstance(start_time, str)
    location_descriptions = self.params.get('locations', '')
    assert isinstance(location_descriptions, str)

    extra_components = self.params.get('extra_components', {})
    extra_components_index = self.params.get('extra_components_index', {})

    if extra_components_index and extra_components:
      if extra_components_index.keys() != extra_components.keys():
        raise ValueError(
            'extra_components_index must have the same keys as'
            ' extra_components.'
        )

    player_names = [entity.name for entity in self.entities]

    # Get scenes for scene tracking
    scenes = self.params.get('scenes', ())
    assert isinstance(scenes, Sequence), 'scenes must be a sequence.'
    if scenes:
      assert isinstance(
          scenes[0], scene_lib.SceneSpec
      ), 'scenes must be a sequence of SceneSpecs.'

    # Get extra event resolution steps
    extra_event_resolution_steps = self.params.get(
        'extra_event_resolution_steps', ''
    )
    assert isinstance(extra_event_resolution_steps, str)
    if ',' in extra_event_resolution_steps:
      extra_event_resolution_steps_list = [
          step.strip()
          for step in extra_event_resolution_steps.split(',')
          if step
      ]
    else:
      extra_event_resolution_steps_list = (
          [extra_event_resolution_steps] if extra_event_resolution_steps else []
      )

    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    instructions_key = 'instructions'
    instructions = gm_components.instructions.Instructions()

    examples_synchronous_key = 'examples'
    examples_synchronous = gm_components.instructions.ExamplesSynchronous()

    player_characters_key = 'player_characters'
    player_characters = gm_components.instructions.PlayerCharacters(
        player_characters=player_names,
    )

    observation_to_memory_key = 'observation_to_memory'
    observation_to_memory = actor_components.observation.ObservationToMemory()

    observation_component_key = (
        actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = actor_components.observation.LastNObservations(
        history_length=1000,
    )

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
            components=[
                display_events_key,
            ],
            num_memories_to_retrieve=25,
            pre_act_label='Background info',
        )
    )

    locations_constant_key = 'locations_constant'
    locations_constant = actor_components.constant.Constant(
        location_descriptions, pre_act_label='Locations'
    )

    clock_constant_key = 'clock_constant'
    clock_constant = actor_components.constant.Constant(
        clock_description, pre_act_label='Clock description'
    )

    generative_clock_key = 'generative_clock'
    generative_clock = gm_components.world_state.GenerativeClock(
        model=model,
        prompt=clock_description,
        start_time=start_time,
        components=[
            instructions_key,
            clock_constant_key,
            display_events_key,
        ],
        pre_act_label='\nCurrent time',
    )

    locations_key = 'locations'
    entity_locations = gm_components.world_state.Locations(
        model=model,
        entity_names=player_names,
        prompt=location_descriptions,
        components=[
            instructions_key,
            locations_constant_key,
            clock_constant_key,
            player_characters_key,
            relevant_memories_key,
            display_events_key,
            generative_clock_key,
        ],
        pre_act_label='\nCurrent locations',
    )

    world_state_key = 'world_state'
    world_state = gm_components.world_state.WorldState(
        model=model,
        components=[
            instructions_key,
            player_characters_key,
            locations_constant_key,
            clock_constant_key,
            locations_key,
            generative_clock_key,
            relevant_memories_key,
            display_events_key,
        ],
        pre_act_label='\nCurrent state',
    )

    make_observation_key = (
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )
    make_observation = gm_components.make_observation.MakeObservation(
        model=model,
        player_names=player_names,
        components=[
            instructions_key,
            player_characters_key,
            locations_constant_key,
            clock_constant_key,
            relevant_memories_key,
            display_events_key,
            generative_clock_key,
            locations_key,
            world_state_key,
        ],
        reformat_observations_in_specified_style=(
            'The format to use when describing the '
            'current situation to a player is: '
            '"// date or time // situation description".'
        ),
    )

    send_events_to_players_key = (
        gm_components.event_resolution.DEFAULT_SEND_PRE_ACT_VALUES_TO_PLAYERS_PRE_ACT_LABEL
    )
    send_events_to_players = (
        gm_components.event_resolution.SendEventToRelevantPlayers(
            model=model,
            player_names=player_names,
            make_observation_component_key=make_observation_key,
        )
    )

    scene_tracker_key = (
        gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
    )
    scene_tracker = gm_components.scene_tracker.SceneTracker(
        model=model,
        scenes=scenes,
        observation_component_key=(
            gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
        ),
    )

    # Use scene-based actor selection
    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_actor = gm_components.next_acting.NextActingFromSceneSpec(
        scene_tracker_component_key=scene_tracker_key,
    )

    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    next_action_spec = gm_components.next_acting.NextActionSpecFromSceneSpec(
        scene_tracker_component_key=scene_tracker_key,
    )

    account_for_agency_of_others = thought_chains_lib.AccountForAgencyOfOthers(
        model=model, players=self.entities, verbose=False
    )

    event_resolution_steps = [
        account_for_agency_of_others,
        thought_chains_lib.result_to_who_what_where,
    ]
    for step in extra_event_resolution_steps_list:
      if step:
        event_resolution_steps.append(getattr(thought_chains_lib, step))

    event_resolution_components = [
        instructions_key,
        player_characters_key,
        locations_constant_key,
        clock_constant_key,
        relevant_memories_key,
        display_events_key,
        generative_clock_key,
        locations_key,
        world_state_key,
    ]

    event_resolution_key = (
        gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
    )
    event_resolution = gm_components.event_resolution.EventResolution(
        model=model,
        event_resolution_steps=event_resolution_steps,
        components=event_resolution_components,
        notify_observers=True,
    )

    terminator_key = gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
    terminator = gm_components.terminate.SceneBasedTerminator(
        scene_tracker_component_key=scene_tracker_key
    )

    components_of_game_master = {
        terminator_key: terminator,
        instructions_key: instructions,
        examples_synchronous_key: examples_synchronous,
        player_characters_key: player_characters,
        locations_constant_key: locations_constant,
        clock_constant_key: clock_constant,
        relevant_memories_key: relevant_memories,
        observation_component_key: observation,
        observation_to_memory_key: observation_to_memory,
        display_events_key: display_events,
        generative_clock_key: generative_clock,
        locations_key: entity_locations,
        world_state_key: world_state,
        memory_component_key: memory_component,
        send_events_to_players_key: send_events_to_players,
        make_observation_key: make_observation,
        scene_tracker_key: scene_tracker,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        event_resolution_key: event_resolution,
    }

    component_order = list(components_of_game_master.keys())

    if extra_components:
      components_of_game_master.update(extra_components)
      if extra_components_index:
        for component_name in extra_components.keys():
          component_order.insert(
              extra_components_index[component_name],
              component_name,
          )
      else:
        component_order = list(components_of_game_master.keys())

    act_component = gm_components.switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    game_master = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components_of_game_master,
    )

    return game_master
