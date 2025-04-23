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
import copy

from concordia.agents.unstable import entity_agent_with_logging
from concordia.associative_memory.unstable import basic_associative_memory as associative_memory
from concordia.components.agent import unstable as actor_components
from concordia.components.agent.unstable import observation as observation_component
from concordia.components.game_master import unstable as gm_components
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.thought_chains.unstable import thought_chains as thought_chains_lib
from concordia.typing.unstable import entity_component as entity_component_lib
from concordia.typing.unstable import scene as scene_lib
import numpy as np

_MEMORY_COMPONENT_KEY = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
_OBSERVATION_COMPONENT_KEY = (
    observation_component.DEFAULT_OBSERVATION_COMPONENT_KEY
)


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build_simulation(
    model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    shared_memories: Sequence[str],
    game_master_name: str = 'Main_Game_Master',
    memory: associative_memory.AssociativeMemoryBank | None = None,
    event_resolution_steps: (
        Sequence[
            Callable[[interactive_document.InteractiveDocument, str, str], str]
        ]
        | None
    ) = None,
    additional_context_components: (
        Mapping[str, entity_component_lib.ContextComponent] | None
    ) = None,
) -> tuple[
    associative_memory.AssociativeMemoryBank,
    entity_agent_with_logging.EntityAgentWithLogging,
]:
  """Build a simulation (i.e., an environment).

  Args:
    model: The language model to use for game master.
    embedder: The embedder to use for similarity retrieval of memories.
    players: The players.
    shared_memories: Sequence of memories to be observed by all players.
    game_master_name: The name of the game master.
    memory: optionally provide a prebuilt memory, otherwise build it here.
    event_resolution_steps: thinking steps for the event resolution component to
      use whenever it converts putative events like action suggestions into real
      events in the simulation.
    additional_context_components: Additional context components to add to the
      game master.

  Returns:
    A tuple consisting of a game master and its memory.
  """

  if memory is not None:
    game_master_memory = memory
  else:
    game_master_memory = associative_memory.AssociativeMemoryBank(
        sentence_embedder=embedder,
    )

  instructions_key = 'instructions'
  instructions = gm_components.instructions.Instructions()

  examples_synchronous_key = 'examples'
  examples_synchronous = gm_components.instructions.ExamplesSynchronous()

  player_names = [player.name for player in players]
  player_characters_key = 'player_characters'
  player_characters = gm_components.instructions.PlayerCharacters(
      player_characters=player_names,
  )

  failed_actions_key = 'failed_actions'
  failed_actions = actor_components.constant.Constant(
      state=(
          'There will typically be negative consequences whenever a '
          'character attempts to do something but fails to achieve '
          'their goal. The game master is responsible for determining '
          'exactly what the consequences are.'
      ),
      pre_act_label='Failure has consequences',
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
      pre_act_label='Story so far (ordered from oldest to most recent events)',
  )

  memory_component_key = _MEMORY_COMPONENT_KEY
  memory_component = actor_components.memory.AssociativeMemory(
      memory_bank=game_master_memory
  )

  relevant_memories_key = 'relevant_memories'
  relevant_memories = actor_components.all_similar_memories.AllSimilarMemories(
      model=model,
      components={
          display_events_key: display_events.get_pre_act_label(),
      },
      num_memories_to_retrieve=25,
      pre_act_label='Background info',
  )

  world_state_key = 'world_state'
  world_state = gm_components.world_state.WorldState(
      model=model,
      components={
          instructions_key: instructions.get_pre_act_label(),
          player_characters_key: player_characters.get_pre_act_label(),
          failed_actions_key: failed_actions.get_pre_act_label(),
          relevant_memories_key: relevant_memories.get_pre_act_label(),
          display_events_key: display_events.get_pre_act_label(),
      },
  )

  make_observation_key = (
      gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
  )
  make_observation = gm_components.make_observation.MakeObservation(
      model=model,
      components={
          instructions_key: instructions.get_pre_act_label(),
          player_characters_key: player_characters.get_pre_act_label(),
          relevant_memories_key: relevant_memories.get_pre_act_label(),
          display_events_key: display_events.get_pre_act_label(),
          world_state_key: world_state.get_pre_act_label(),
      },
      reformat_observations_in_specified_style=(
          'The format to use when describing the '
          'current situation to a player is: '
          '"//date or time//situation description".'
      ),
  )

  next_acting_kwargs = dict(
      model=model,
      components={
          instructions_key: instructions.get_pre_act_label(),
          player_characters_key: player_characters.get_pre_act_label(),
          relevant_memories_key: relevant_memories.get_pre_act_label(),
          display_events_key: display_events.get_pre_act_label(),
          world_state_key: world_state.get_pre_act_label(),
      },
  )
  next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
  next_actor = gm_components.next_acting.NextActingInRandomOrder(
      player_names=player_names,
      replace=False,
  )
  next_action_spec_kwargs = copy.copy(next_acting_kwargs)
  next_action_spec_key = (
      gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
  )
  next_action_spec = gm_components.next_acting.NextActionSpec(
      **next_action_spec_kwargs,
      player_names=player_names,
  )

  # Define thinking steps for the event resolution component to use whenever
  # it converts putative events like action suggestions into real events in
  # the simulation.
  account_for_agency_of_others = thought_chains_lib.AccountForAgencyOfOthers(
      model=model, players=players, verbose=False
  )

  if not event_resolution_steps:
    event_resolution_steps = [
        thought_chains_lib.attempt_to_most_likely_outcome,
        thought_chains_lib.result_to_effect_caused_by_active_player,
        account_for_agency_of_others,
    ]

  event_resolution_components = {
      instructions_key: instructions.get_pre_act_label(),
      failed_actions_key: failed_actions.get_pre_act_label(),
      player_characters_key: player_characters.get_pre_act_label(),
      relevant_memories_key: relevant_memories.get_pre_act_label(),
      display_events_key: display_events.get_pre_act_label(),
      world_state_key: world_state.get_pre_act_label(),
  }

  event_resolution_key = (
      gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
  )
  event_resolution = gm_components.event_resolution.EventResolution(
      model=model,
      event_resolution_steps=event_resolution_steps,
      components=event_resolution_components,
      notify_observers=True,
  )

  components_of_game_master = {
      instructions_key: instructions,
      examples_synchronous_key: examples_synchronous,
      player_characters_key: player_characters,
      failed_actions_key: failed_actions,
      relevant_memories_key: relevant_memories,
      observation_component_key: observation,
      observation_to_memory_key: observation_to_memory,
      display_events_key: display_events,
      world_state_key: world_state,
      memory_component_key: memory_component,
      make_observation_key: make_observation,
      next_actor_key: next_actor,
      next_action_spec_key: next_action_spec,
      event_resolution_key: event_resolution,
  }

  if additional_context_components:
    for key, component in additional_context_components.items():
      if key in components_of_game_master:
        raise ValueError(f'Key {key} already exists default game master.')
      components_of_game_master[key] = component

  component_order = list(components_of_game_master.keys())

  act_component = gm_components.switch_act.SwitchAct(
      model=model,
      entity_names=player_names,
      component_order=component_order,
  )

  game_master = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=game_master_name,
      act_component=act_component,
      context_components=components_of_game_master,
  )

  for mem in shared_memories:
    game_master.observe(mem)

  return game_master_memory, game_master


def build_simulation_with_scenes(
    model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    shared_memories: Sequence[str],
    scenes: Sequence[scene_lib.ExperimentalSceneSpec],
    observation_queue: Mapping[str, list[str]],
    globabl_scene_counter: gm_components.scene_tracker.ThreadSafeCounter,
    game_master_name: str = 'Main_Game_Master',
    memory: associative_memory.AssociativeMemoryBank | None = None,
    event_resolution_steps: (
        Sequence[
            Callable[[interactive_document.InteractiveDocument, str, str], str]
        ]
        | None
    ) = None,
    additional_context_components: (
        Mapping[str, entity_component_lib.ContextComponent] | None
    ) = None,
) -> tuple[
    associative_memory.AssociativeMemoryBank,
    entity_agent_with_logging.EntityAgentWithLogging,
]:
  """Build a simulation (i.e., an environment).

  Args:
    model: The language model to use for game master.
    embedder: The embedder to use for similarity retrieval of memories.
    players: The players.
    shared_memories: Sequence of memories to be observed by all players.
    scenes: The scenes to use for the game master.
    observation_queue: The observation queue to use for the game master.
    globabl_scene_counter: The global scene counter to use for the game master.
    game_master_name: The name of the game master.
    memory: optionally provide a prebuilt memory, otherwise build it here.
    event_resolution_steps: thinking steps for the event resolution component to
      use whenever it converts putative events like action suggestions into real
      events in the simulation.
    additional_context_components: Additional context components to add to the
      game master.

  Returns:
    A tuple consisting of a game master and its memory.
  """

  if memory is not None:
    game_master_memory = memory
  else:
    game_master_memory = associative_memory.AssociativeMemoryBank(
        sentence_embedder=embedder,
    )

  instructions_key = 'instructions'
  instructions = gm_components.instructions.Instructions()

  examples_synchronous_key = 'examples'
  examples_synchronous = gm_components.instructions.ExamplesSynchronous()

  player_names = [player.name for player in players]
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

  memory_component_key = _MEMORY_COMPONENT_KEY
  memory = actor_components.memory.AssociativeMemory(
      memory_bank=game_master_memory
  )

  make_observation_key = (
      gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
  )
  make_observation = (
      gm_components.make_observation.MakeObservationFromQueueOnly(
          model=model,
          queue=observation_queue,
      )
  )

  next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
  next_action_spec_key = (
      gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
  )

  next_actor = gm_components.next_acting.NextActingFromSceneSpec(
      model=model,
      components=[],
      memory_component_key=_MEMORY_COMPONENT_KEY,
      scene_tracker_component_key=gm_components.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY,
  )

  next_game_master_key = (
      gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
  )
  next_game_master = gm_components.next_game_master.NextGameMasterFromSceneSpec(
      model=model,
      scene_tracker_component_key=gm_components.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY,
      pre_act_label=gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_PRE_ACT_LABEL,
  )

  scene_tracker_key = (
      gm_components.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY
  )
  scene_tracker = gm_components.scene_tracker.SceneTracker(
      model=model,
      scenes=scenes,
      step_counter=globabl_scene_counter,
      observation_component_key=(
          gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
  )

  next_action_spec = gm_components.next_acting.NextActionSpecFromSceneSpec(
      scenes=scenes,
  )

  # Define thinking steps for the event resolution component to use whenever
  # it converts putative events like action suggestions into real events in
  # the simulation.

  if not event_resolution_steps:
    event_resolution_steps = [
        thought_chains_lib.identity,
    ]

  event_resolution_components = {
      instructions_key: instructions.get_pre_act_label(),
      player_characters_key: player_characters.get_pre_act_label(),
  }

  event_resolution_key = (
      gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
  )
  event_resolution = gm_components.event_resolution.EventResolution(
      model=model,
      event_resolution_steps=event_resolution_steps,
      components=event_resolution_components,
      notify_observers=True,
  )

  components_of_game_master = {
      instructions_key: instructions,
      scene_tracker_key: scene_tracker,
      examples_synchronous_key: examples_synchronous,
      player_characters_key: player_characters,
      observation_component_key: observation,
      observation_to_memory_key: observation_to_memory,
      memory_component_key: memory,
      make_observation_key: make_observation,
      next_actor_key: next_actor,
      next_action_spec_key: next_action_spec,
      next_game_master_key: next_game_master,
      event_resolution_key: event_resolution,
  }

  if additional_context_components:
    for key, component in additional_context_components.items():
      if key in components_of_game_master:
        raise ValueError(f'Key {key} already exists default game master.')
      components_of_game_master[key] = component

  component_order = list(components_of_game_master.keys())

  act_component = gm_components.switch_act.SwitchAct(
      model=model,
      entity_names=player_names,
      component_order=component_order,
  )

  game_master = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=game_master_name,
      act_component=act_component,
      context_components=components_of_game_master,
  )
  for mem in shared_memories:
    game_master.observe(mem)

  return game_master_memory, game_master


def build_decision_scene_game_master(
    model: language_model.LanguageModel,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    globabl_scene_counter: gm_components.scene_tracker.ThreadSafeCounter,
    scenes: Sequence[scene_lib.ExperimentalSceneSpec],
    observation_queue: Mapping[str, list[str]],
    memory: associative_memory.AssociativeMemoryBank,
    additional_context_components: (
        Mapping[str, entity_component_lib.ContextComponent] | None
    ) = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build a decision game master for decision scenes."""

  game_master_memory = memory

  instructions = gm_components.instructions.Instructions()

  examples_synchronous = gm_components.instructions.ExamplesSynchronous()

  player_names = [player.name for player in players]
  player_characters = gm_components.instructions.PlayerCharacters(
      player_characters=player_names,
  )

  observation_to_memory = observation_component.ObservationToMemory()
  observation = observation_component.LastNObservations(
      history_length=100,
  )

  display_events_key = 'display_events'

  display_events = gm_components.event_resolution.DisplayEvents(
      model=model,
  )

  make_observation = (
      gm_components.make_observation.MakeObservationFromQueueOnly(
          model=model,
          queue=observation_queue,
      )
  )

  next_actor = gm_components.next_acting.NextActingFromSceneSpec(
      model=model,
      components=[],
      memory_component_key=_MEMORY_COMPONENT_KEY,
      scene_tracker_component_key=gm_components.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY,
  )

  next_action_spec = gm_components.next_acting.NextActionSpecFromSceneSpec(
      scenes=scenes,
      memory_component_key=_MEMORY_COMPONENT_KEY,
  )

  next_game_master = gm_components.next_game_master.NextGameMasterFromSceneSpec(
      model=model,
      scene_tracker_component_key=gm_components.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY,
      pre_act_label=gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_PRE_ACT_LABEL,
  )

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

  event_resolution = gm_components.event_resolution.EventResolution(
      model=model,
      event_resolution_steps=event_resolution_steps,
      components=event_resolution_components,
  )
  scene_tracker = gm_components.scene_tracker.SceneTracker(
      model=model,
      scenes=scenes,
      step_counter=globabl_scene_counter,
      observation_component_key=(
          gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
  )
  components_of_game_master = {
      _get_class_name(instructions): instructions,
      _get_class_name(examples_synchronous): examples_synchronous,
      _get_class_name(player_characters): player_characters,
      gm_components.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY: (
          scene_tracker
      ),
      _get_class_name(observation_to_memory): observation_to_memory,
      display_events_key: display_events,
      _OBSERVATION_COMPONENT_KEY: observation,
      _MEMORY_COMPONENT_KEY: actor_components.memory.AssociativeMemory(
          memory_bank=game_master_memory
      ),
      gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: (
          event_resolution
      ),
      gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
          make_observation
      ),
      gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_PRE_ACT_LABEL: (
          next_action_spec
      ),
      gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY: next_actor,
      gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY: (
          next_game_master
      ),
  }

  if additional_context_components is not None:
    for key, component in additional_context_components.items():
      if key in components_of_game_master:
        raise ValueError(f'Key {key} already exists default game master.')
      components_of_game_master[key] = component

  component_order = list(components_of_game_master.keys())

  act_component = gm_components.switch_act.SwitchAct(
      model=model,
      entity_names=player_names,
      component_order=component_order,
  )

  game_master = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name='Decision_Game_Master',
      act_component=act_component,
      context_components=components_of_game_master,
  )

  return game_master
