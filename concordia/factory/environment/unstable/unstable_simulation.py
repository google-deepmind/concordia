# Copyright 2023 DeepMind Technologies Limited.
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

"""A generic factory to configure simulations."""

from collections.abc import Callable, Mapping, Sequence

from concordia.agents.unstable import entity_agent_with_logging
from concordia.associative_memory.unstable import basic_associative_memory as associative_memory
from concordia.clocks import game_clock
from concordia.components.agent import unstable as agent_components
from concordia.components.agent.unstable import action_spec_ignored
from concordia.components.agent.unstable import memory as memory_component
from concordia.components.agent.unstable import observation as observation_component
from concordia.components.game_master import unstable as gm_components_lib
from concordia.contrib.components.agent.unstable import situation_representation_via_narrative
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.thought_chains.unstable import thought_chains as thought_chains_lib
from concordia.typing.unstable import entity_component as entity_component_lib
from concordia.typing.unstable import scene as scene_lib
from concordia.utils import measurements as measurements_lib
import numpy as np


_MEMORY_COMPONENT_KEY = memory_component.DEFAULT_MEMORY_COMPONENT_KEY
_OBSERVATION_COMPONENT_KEY = (
    observation_component.DEFAULT_OBSERVATION_COMPONENT_KEY
)


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build_simulation(
    model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
    clock: game_clock.MultiIntervalClock,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    shared_memories: Sequence[str],
    game_master_name: str = 'Main_Game_Master',
    memory: associative_memory.AssociativeMemoryBank | None = None,
    nonplayer_entities: Sequence[
        entity_component_lib.EntityWithComponents
    ] = tuple([]),
    event_resolution_steps: (
        Sequence[
            Callable[[interactive_document.InteractiveDocument, str, str], str]
        ]
        | None
    ) = None,
    additional_context_components: (
        Mapping[str, entity_component_lib.ContextComponent] | None
    ) = None,
    measurements: measurements_lib.Measurements | None = None,
) -> tuple[
    associative_memory.AssociativeMemoryBank,
    entity_agent_with_logging.EntityAgentWithLogging,
]:
  """Build a simulation (i.e., an environment).

  Args:
    model: The language model to use for game master.
    embedder: The embedder to use for similarity retrieval of memories.
    clock: The simulation clock.
    players: The players.
    shared_memories: Sequence of memories to be observed by all players.
    game_master_name: The name of the game master.
    memory: optionally provide a prebuilt memory, otherwise build it here.
    nonplayer_entities: The non-player entities.
    event_resolution_steps: thinking steps for the event resolution component to
      use whenever it converts putative events like action suggestions into real
      events in the simulation.
    additional_context_components: Additional context components to add to the
      game master.
    measurements: The measurements to use for the game master.

  Returns:
    A tuple consisting of a game master and its memory.
  """
  if measurements is None:
    measurements = measurements_lib.Measurements()

  if memory is not None:
    game_master_memory = memory
  else:
    game_master_memory = associative_memory.AssociativeMemoryBank(
        sentence_embedder=embedder,
    )

  instructions = gm_components_lib.instructions.Instructions()

  examples_synchronous = gm_components_lib.instructions.ExamplesSynchronous()

  player_names = [player.name for player in players]
  player_characters = gm_components_lib.instructions.PlayerCharacters(
      player_characters=player_names,
  )

  scenario_knowledge = agent_components.constant.Constant(
      state='\n'.join(shared_memories),
      pre_act_label='\nBackground:\n',
  )

  nonplayer_entities_list_key = '\nNon-player characters: '
  nonplayer_entities_list = agent_components.constant.Constant(
      state='\n'.join([entity.name for entity in nonplayer_entities]),
      pre_act_label=nonplayer_entities_list_key,
  )

  observation_to_memory = observation_component.ObservationToMemory()
  observation = observation_component.LastNObservations(
      history_length=100,
  )

  situation_representation_label = '\nSituation description'
  situation_representation = (
      situation_representation_via_narrative.SituationRepresentation(
          model=model,
          clock_now=clock.now,
          observation_component_key=_OBSERVATION_COMPONENT_KEY,
          components={
              _get_class_name(instructions): instructions.get_pre_act_label(),
              _get_class_name(scenario_knowledge): (
                  scenario_knowledge.get_pre_act_label()
              ),
              _get_class_name(player_characters): (
                  player_characters.get_pre_act_label()
              ),
              nonplayer_entities_list_key: nonplayer_entities_list_key,
          },
          declare_entity_as_protagonist=False,
          pre_act_label=situation_representation_label,
          logging_channel=measurements.get_channel(
              'SituationRepresentation'
          ).on_next,
      )
  )

  display_events = gm_components_lib.event_resolution.DisplayEvents(
      model=model,
  )

  components_of_game_master = {
      _get_class_name(instructions): instructions,
      _get_class_name(examples_synchronous): examples_synchronous,
      _get_class_name(player_characters): player_characters,
      _get_class_name(scenario_knowledge): scenario_knowledge,
      nonplayer_entities_list_key: nonplayer_entities_list,
      _OBSERVATION_COMPONENT_KEY: observation,
      _get_class_name(situation_representation): situation_representation,
      _get_class_name(observation_to_memory): observation_to_memory,
      _get_class_name(display_events): display_events,
      _MEMORY_COMPONENT_KEY: memory_component.ListMemory(
          memory_bank=game_master_memory
      ),
  }
  if additional_context_components is not None:
    for key, component in additional_context_components.items():
      if key in components_of_game_master:
        raise ValueError(f'Key {key} already exists default game master.')
      components_of_game_master[key] = component

  make_observation = gm_components_lib.make_observation.MakeObservation(
      model=model,
      components={
          _get_class_name(instructions): instructions.get_pre_act_label(),
          _get_class_name(
              examples_synchronous
          ): examples_synchronous.get_pre_act_label(),
          _get_class_name(
              player_characters
          ): player_characters.get_pre_act_label(),
          _get_class_name(
              scenario_knowledge
          ): scenario_knowledge.get_pre_act_label(),
          nonplayer_entities_list_key: nonplayer_entities_list_key,
          _OBSERVATION_COMPONENT_KEY: observation.get_pre_act_label(),
          _get_class_name(
              situation_representation
          ): situation_representation.get_pre_act_label(),
          _get_class_name(display_events): display_events.get_pre_act_label(),
      },
      logging_channel=measurements.get_channel('MakeObservation').on_next,
  )

  components_of_game_master[
      gm_components_lib.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
  ] = make_observation

  next_acting_kwargs = dict(
      model=model,
      components={
          _get_class_name(instructions): instructions.get_pre_act_label(),
          _get_class_name(
              examples_synchronous
          ): examples_synchronous.get_pre_act_label(),
          _get_class_name(
              player_characters
          ): player_characters.get_pre_act_label(),
          _get_class_name(
              scenario_knowledge
          ): scenario_knowledge.get_pre_act_label(),
          nonplayer_entities_list_key: nonplayer_entities_list_key,
          _OBSERVATION_COMPONENT_KEY: observation.get_pre_act_label(),
          _get_class_name(
              situation_representation
          ): situation_representation.get_pre_act_label(),
          _get_class_name(display_events): display_events.get_pre_act_label(),
      },
      logging_channel=measurements.get_channel('NextActing').on_next,
  )

  next_actor = gm_components_lib.next_acting.NextActing(
      **next_acting_kwargs,
      player_names=player_names,
  )

  components_of_game_master[
      gm_components_lib.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
  ] = next_actor

  # Define thinking steps for the event resolution component to use whenever it
  # converts putative events like action suggestions into real events in the
  # simulation.
  account_for_agency_of_others = thought_chains_lib.AccountForAgencyOfOthers(
      model=model, players=players, verbose=False
  )
  event_resolution_steps = event_resolution_steps or [
      thought_chains_lib.extract_direct_quote,
      thought_chains_lib.attempt_to_most_likely_outcome,
      thought_chains_lib.result_to_effect_caused_by_active_player,
      account_for_agency_of_others,
      thought_chains_lib.restore_direct_quote,
  ]

  event_resolution_components = {
      _get_class_name(instructions): instructions.get_pre_act_label(),
      _get_class_name(
          examples_synchronous
      ): examples_synchronous.get_pre_act_label(),
      _get_class_name(player_characters): player_characters.get_pre_act_label(),
      _get_class_name(
          scenario_knowledge
      ): scenario_knowledge.get_pre_act_label(),
      nonplayer_entities_list_key: nonplayer_entities_list_key,
      _OBSERVATION_COMPONENT_KEY: observation.get_pre_act_label(),
      _get_class_name(
          situation_representation
      ): situation_representation.get_pre_act_label(),
  }

  if additional_context_components is not None:
    for key, component in additional_context_components.items():
      if key in event_resolution_components:
        raise ValueError(f'Key {key} already exists default game master.')
      if isinstance(component, action_spec_ignored.ActionSpecIgnored):
        assert hasattr(component, 'get_pre_act_label')  # Assertion for pytype
        event_resolution_components[key] = component.get_pre_act_label()

  event_resolution = gm_components_lib.event_resolution.EventResolution(
      model=model,
      event_resolution_steps=event_resolution_steps,
      components=event_resolution_components,
      logging_channel=measurements.get_channel('EventResolution').on_next,
  )

  components_of_game_master[
      gm_components_lib.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
  ] = event_resolution

  component_order = list(components_of_game_master.keys())

  act_component = gm_components_lib.switch_act.SwitchAct(
      model=model,
      entity_names=player_names,
      component_order=component_order,
      logging_channel=measurements.get_channel('SwitchAct').on_next,
  )

  game_master = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=game_master_name,
      act_component=act_component,
      context_components=components_of_game_master,
      component_logging=measurements,
  )

  return game_master_memory, game_master


def build_simulation_with_scenes(
    model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
    clock: game_clock.MultiIntervalClock,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    shared_memories: Sequence[str],
    scenes: Sequence[scene_lib.ExperimentalSceneSpec],
    globabl_scene_counter: gm_components_lib.scene_tracker.ThreadSafeCounter,
    game_master_name: str = 'Main_Game_Master',
    memory: associative_memory.AssociativeMemoryBank | None = None,
    nonplayer_entities: Sequence[
        entity_component_lib.EntityWithComponents
    ] = tuple([]),
    event_resolution_steps: (
        Sequence[
            Callable[[interactive_document.InteractiveDocument, str, str], str]
        ]
        | None
    ) = None,
    additional_context_components: (
        Mapping[str, entity_component_lib.ContextComponent] | None
    ) = None,
    measurements: measurements_lib.Measurements | None = None,
) -> tuple[
    associative_memory.AssociativeMemoryBank,
    entity_agent_with_logging.EntityAgentWithLogging,
]:
  """Build a simulation (i.e., an environment).

  Args:
    model: The language model to use for game master.
    embedder: The embedder to use for similarity retrieval of memories.
    clock: The simulation clock.
    players: The players.
    shared_memories: Sequence of memories to be observed by all players.
    scenes: The scenes to use for the game master.
    globabl_scene_counter: The global scene counter.
    game_master_name: The name of the game master.
    memory: The memory to use for the game master.
    nonplayer_entities: The non-player entities.
    event_resolution_steps: thinking steps for the event resolution component to
      use whenever it converts putative events like action suggestions into real
      events in the simulation.
    additional_context_components: Additional context components to add to the
      game master.
    measurements: The measurements to use for the game master.

  Returns:
    A tuple consisting of a game master and its memory.
  """
  if measurements is None:
    measurements = measurements_lib.Measurements()

  if memory is not None:
    game_master_memory = memory
  else:
    game_master_memory = associative_memory.AssociativeMemoryBank(
        sentence_embedder=embedder,
    )

  instructions = gm_components_lib.instructions.Instructions()

  examples_synchronous = gm_components_lib.instructions.ExamplesSynchronous()

  player_names = [player.name for player in players]
  player_characters = gm_components_lib.instructions.PlayerCharacters(
      player_characters=player_names,
  )

  scenario_knowledge = agent_components.constant.Constant(
      state='\n'.join(shared_memories),
      pre_act_label='\nScenario knowledge:\n',
  )

  nonplayer_entities_list_key = '\nNon-player characters: '
  nonplayer_entities_list = agent_components.constant.Constant(
      state='\n'.join([entity.name for entity in nonplayer_entities]),
      pre_act_label=nonplayer_entities_list_key,
  )

  observation_to_memory = observation_component.ObservationToMemory()
  observation = observation_component.LastNObservations(
      history_length=100,
  )

  situation_representation_label = '\nSituation description'
  situation_representation = (
      situation_representation_via_narrative.SituationRepresentation(
          model=model,
          clock_now=clock.now,
          observation_component_key=_OBSERVATION_COMPONENT_KEY,
          components={
              _get_class_name(instructions): instructions.get_pre_act_label(),
              _get_class_name(scenario_knowledge): (
                  scenario_knowledge.get_pre_act_label()
              ),
              _get_class_name(player_characters): (
                  player_characters.get_pre_act_label()
              ),
              nonplayer_entities_list_key: nonplayer_entities_list_key,
          },
          declare_entity_as_protagonist=False,
          pre_act_label=situation_representation_label,
          logging_channel=measurements.get_channel(
              'SituationRepresentation'
          ).on_next,
      )
  )

  display_events = gm_components_lib.event_resolution.DisplayEvents(
      model=model,
  )

  components_of_game_master = {
      _get_class_name(instructions): instructions,
      _get_class_name(examples_synchronous): examples_synchronous,
      _get_class_name(player_characters): player_characters,
      _get_class_name(scenario_knowledge): scenario_knowledge,
      nonplayer_entities_list_key: nonplayer_entities_list,
      _OBSERVATION_COMPONENT_KEY: observation,
      _get_class_name(situation_representation): situation_representation,
      _get_class_name(observation_to_memory): observation_to_memory,
      _get_class_name(display_events): display_events,
      _MEMORY_COMPONENT_KEY: memory_component.ListMemory(
          memory_bank=game_master_memory
      ),
  }
  if additional_context_components is not None:
    for key, component in additional_context_components.items():
      if key in components_of_game_master:
        raise ValueError(f'Key {key} already exists default game master.')
      components_of_game_master[key] = component

  make_observation = gm_components_lib.make_observation.MakeObservation(
      model=model,
      components={
          _get_class_name(instructions): instructions.get_pre_act_label(),
          _get_class_name(
              examples_synchronous
          ): examples_synchronous.get_pre_act_label(),
          _get_class_name(
              player_characters
          ): player_characters.get_pre_act_label(),
          _get_class_name(
              scenario_knowledge
          ): scenario_knowledge.get_pre_act_label(),
          nonplayer_entities_list_key: nonplayer_entities_list_key,
          _OBSERVATION_COMPONENT_KEY: observation.get_pre_act_label(),
          # _get_class_name(
          #     situation_representation
          # ): situation_representation.get_pre_act_label(),
          _get_class_name(display_events): display_events.get_pre_act_label(),
      },
      logging_channel=measurements.get_channel('MakeObservation').on_next,
  )

  components_of_game_master[
      gm_components_lib.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
  ] = make_observation

  next_acting_kwargs = dict(
      model=model,
      components={
          _get_class_name(instructions): instructions.get_pre_act_label(),
          _get_class_name(
              examples_synchronous
          ): examples_synchronous.get_pre_act_label(),
          _get_class_name(
              player_characters
          ): player_characters.get_pre_act_label(),
          _get_class_name(
              scenario_knowledge
          ): scenario_knowledge.get_pre_act_label(),
          nonplayer_entities_list_key: nonplayer_entities_list_key,
          _OBSERVATION_COMPONENT_KEY: observation.get_pre_act_label(),
          _get_class_name(
              situation_representation
          ): situation_representation.get_pre_act_label(),
          _get_class_name(display_events): display_events.get_pre_act_label(),
      },
      logging_channel=measurements.get_channel('NextActing').on_next,
  )

  next_actor = gm_components_lib.next_acting.NextActingFromSceneSpec(
      **next_acting_kwargs,
  )
  next_game_master = gm_components_lib.next_game_master.NextGameMasterFromSceneSpec(
      model=model,
      scene_tracker_component_key=gm_components_lib.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY,
      pre_act_label=gm_components_lib.next_game_master.DEFAULT_NEXT_GAME_MASTER_PRE_ACT_KEY,
      logging_channel=measurements.get_channel('NextGameMaster').on_next,
  )
  components_of_game_master[
      gm_components_lib.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
  ] = next_game_master
  scene_tracker = gm_components_lib.scene_tracker.SceneTracker(
      model=model,
      scenes=scenes,
      step_counter=globabl_scene_counter,
      observation_component_key=(
          gm_components_lib.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
  )

  components_of_game_master[
      gm_components_lib.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY
  ] = scene_tracker
  next_action_spec = gm_components_lib.next_acting.NextActionSpecFromSceneSpec(
      scenes=scenes,
      memory_component_key=_MEMORY_COMPONENT_KEY,
      logging_channel=measurements.get_channel('NextActionSpec').on_next,
  )
  components_of_game_master[
      gm_components_lib.next_acting.DEFAULT_NEXT_ACTION_SPEC_PRE_ACT_KEY
  ] = next_action_spec

  components_of_game_master[
      gm_components_lib.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
  ] = next_actor

  # Define thinking steps for the event resolution component to use whenever it
  # converts putative events like action suggestions into real events in the
  # simulation.
  account_for_agency_of_others = thought_chains_lib.AccountForAgencyOfOthers(
      model=model, players=players, verbose=False
  )
  event_resolution_steps = event_resolution_steps or [
      thought_chains_lib.extract_direct_quote,
      thought_chains_lib.attempt_to_most_likely_outcome,
      thought_chains_lib.result_to_effect_caused_by_active_player,
      account_for_agency_of_others,
      thought_chains_lib.restore_direct_quote,
  ]

  event_resolution_components = {
      _get_class_name(instructions): instructions.get_pre_act_label(),
      _get_class_name(
          examples_synchronous
      ): examples_synchronous.get_pre_act_label(),
      _get_class_name(player_characters): player_characters.get_pre_act_label(),
      _get_class_name(
          scenario_knowledge
      ): scenario_knowledge.get_pre_act_label(),
      nonplayer_entities_list_key: nonplayer_entities_list_key,
      _OBSERVATION_COMPONENT_KEY: observation.get_pre_act_label(),
      _get_class_name(
          situation_representation
      ): situation_representation.get_pre_act_label(),
  }

  if additional_context_components is not None:
    for key, component in additional_context_components.items():
      if key in event_resolution_components:
        raise ValueError(f'Key {key} already exists default game master.')
      if isinstance(component, action_spec_ignored.ActionSpecIgnored):
        assert hasattr(component, 'get_pre_act_label')  # Assertion for pytype
        event_resolution_components[key] = component.get_pre_act_label()

  event_resolution = gm_components_lib.event_resolution.EventResolution(
      model=model,
      event_resolution_steps=event_resolution_steps,
      components=event_resolution_components,
      logging_channel=measurements.get_channel('EventResolution').on_next,
  )

  components_of_game_master[
      gm_components_lib.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
  ] = event_resolution

  component_order = list(components_of_game_master.keys())

  act_component = gm_components_lib.switch_act.SwitchAct(
      model=model,
      entity_names=player_names,
      component_order=component_order,
      logging_channel=measurements.get_channel('SwitchAct').on_next,
  )

  game_master = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=game_master_name,
      act_component=act_component,
      context_components=components_of_game_master,
      component_logging=measurements,
  )

  return game_master_memory, game_master


def build_decision_scene_game_master(
    model: language_model.LanguageModel,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    globabl_scene_counter: gm_components_lib.scene_tracker.ThreadSafeCounter,
    memory: associative_memory.AssociativeMemoryBank,
    scenes: Sequence[scene_lib.ExperimentalSceneSpec] | None = None,
    measurements: measurements_lib.Measurements | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build a decision game master for decision scenes."""

  if measurements is None:
    measurements = measurements_lib.Measurements()

  game_master_memory = memory

  instructions = gm_components_lib.instructions.Instructions()

  examples_synchronous = gm_components_lib.instructions.ExamplesSynchronous()

  player_names = [player.name for player in players]
  player_characters = gm_components_lib.instructions.PlayerCharacters(
      player_characters=player_names,
      logging_channel=measurements.get_channel('PlayerCharacters').on_next,
  )

  observation_to_memory = observation_component.ObservationToMemory()
  observation = observation_component.LastNObservations(
      history_length=100,
      logging_channel=measurements.get_channel('Observation').on_next,
  )

  display_events = gm_components_lib.event_resolution.DisplayEvents(
      model=model,
      logging_channel=measurements.get_channel('DisplayEvents').on_next,
  )

  components_of_game_master = {
      _get_class_name(instructions): instructions,
      _get_class_name(examples_synchronous): examples_synchronous,
      _get_class_name(player_characters): player_characters,
      _get_class_name(observation_to_memory): observation_to_memory,
      _get_class_name(display_events): display_events,
      _OBSERVATION_COMPONENT_KEY: observation,
      _MEMORY_COMPONENT_KEY: memory_component.ListMemory(
          memory_bank=game_master_memory
      ),
  }

  make_observation = (
      gm_components_lib.make_observation.MakeObservationFromQueueOnly(
          model=model,
          logging_channel=measurements.get_channel('MakeObservation').on_next,
      )
  )

  components_of_game_master[
      gm_components_lib.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
  ] = make_observation

  next_acting_kwargs = dict(
      model=model,
      components={
          _get_class_name(instructions): instructions.get_pre_act_label(),
          _get_class_name(
              examples_synchronous
          ): examples_synchronous.get_pre_act_label(),
          _get_class_name(
              player_characters
          ): player_characters.get_pre_act_label(),
          _get_class_name(display_events): display_events.get_pre_act_label(),
      },
      logging_channel=measurements.get_channel('NextActing').on_next,
  )

  next_actor = gm_components_lib.next_acting.NextActingFromSceneSpec(
      **next_acting_kwargs,
  )
  next_action_spec = gm_components_lib.next_acting.NextActionSpecFromSceneSpec(
      scenes=scenes,
      memory_component_key=_MEMORY_COMPONENT_KEY,
      logging_channel=measurements.get_channel('NextActionSpec').on_next,
  )
  components_of_game_master[
      gm_components_lib.next_acting.DEFAULT_NEXT_ACTION_SPEC_PRE_ACT_KEY
  ] = next_action_spec

  components_of_game_master[
      gm_components_lib.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
  ] = next_actor

  next_game_master = gm_components_lib.next_game_master.NextGameMasterFromSceneSpec(
      model=model,
      scene_tracker_component_key=gm_components_lib.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY,
      pre_act_label=gm_components_lib.next_game_master.DEFAULT_NEXT_GAME_MASTER_PRE_ACT_KEY,
      logging_channel=measurements.get_channel('NextGameMaster').on_next,
  )
  components_of_game_master[
      gm_components_lib.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
  ] = next_game_master

  event_resolution_steps = [
      thought_chains_lib.identity,
  ]

  event_resolution_components = {
      _get_class_name(instructions): instructions.get_pre_act_label(),
      _get_class_name(
          examples_synchronous
      ): examples_synchronous.get_pre_act_label(),
      _get_class_name(player_characters): player_characters.get_pre_act_label(),
      _OBSERVATION_COMPONENT_KEY: observation.get_pre_act_label(),
  }

  event_resolution = gm_components_lib.event_resolution.EventResolution(
      model=model,
      event_resolution_steps=event_resolution_steps,
      components=event_resolution_components,
      logging_channel=measurements.get_channel('EventResolution').on_next,
  )
  components_of_game_master[
      gm_components_lib.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
  ] = event_resolution

  scene_tracker = gm_components_lib.scene_tracker.SceneTracker(
      model=model,
      scenes=scenes,
      step_counter=globabl_scene_counter,
      observation_component_key=(
          gm_components_lib.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
  )
  components_of_game_master[
      gm_components_lib.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY
  ] = scene_tracker

  component_order = list(components_of_game_master.keys())

  act_component = gm_components_lib.switch_act.SwitchAct(
      model=model,
      entity_names=player_names,
      component_order=component_order,
      logging_channel=measurements.get_channel('SwitchAct').on_next,
  )

  game_master = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name='Decision_Game_Master',
      act_component=act_component,
      context_components=components_of_game_master,
      component_logging=measurements,
  )
  return game_master
