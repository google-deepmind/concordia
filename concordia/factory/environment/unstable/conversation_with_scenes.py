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

"""A factory to configure conversation game masters with scenes."""

from collections.abc import Mapping, Sequence

from concordia.agents.unstable import entity_agent_with_logging
from concordia.associative_memory.unstable import basic_associative_memory as associative_memory
from concordia.components.agent import unstable as actor_components
from concordia.components.game_master import unstable as gm_components
from concordia.language_model import language_model
from concordia.thought_chains.unstable import thought_chains as thought_chains_lib
from concordia.typing.unstable import scene as scene_lib


def build(
    model: language_model.LanguageModel,
    memory_bank: list[str] | associative_memory.AssociativeMemoryBank,
    player_names: Sequence[str],
    scenes: Sequence[scene_lib.ExperimentalSceneSpec],
    observation_queue: Mapping[str, list[str]],
    global_scene_counter: gm_components.scene_tracker.ThreadSafeCounter,
    name: str = 'conversation rules',
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build a game master with settings specialized for conversation and scenes.

  Args:
    model: The language model to use for game master.
    memory_bank: provide a memory_bank.
    player_names: The names of the players.
    scenes: The scenes to use for the game master.
    observation_queue: The observation queue to use for the game master.
    global_scene_counter: The global scene counter to use for the game master.
    name: The name of the game master to build.

  Returns:
    A conversation game master.
  """
  instructions_key = 'instructions'
  instructions = gm_components.instructions.Instructions()

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

  memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
  memory = actor_components.memory.AssociativeMemory(
      memory_bank=memory_bank
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
      memory_component_key=actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
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
      step_counter=global_scene_counter,
      observation_component_key=(
          gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
  )

  terminator_key = (
      gm_components.switch_act.DEFAULT_TERMINATE_COMPONENT_KEY
  )
  terminator = gm_components.scene_tracker.SceneTerminator(
      scene_tracker_component_key=gm_components.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY,
  )

  next_action_spec = gm_components.next_acting.NextActionSpecFromSceneSpec(
      scenes=scenes,
  )

  # Define thinking steps for the event resolution component to use whenever
  # it converts putative events like action suggestions into real events in
  # the simulation.

  identity_without_prefix = thought_chains_lib.RemoveSpecificText(
      substring_to_remove='Putative event to resolve:  '
  )

  event_resolution_key = (
      gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY)
  event_resolution = gm_components.event_resolution.EventResolution(
      model=model,
      event_resolution_steps=(identity_without_prefix,),
      notify_observers=False,
  )

  components_of_game_master = {
      instructions_key: instructions,
      scene_tracker_key: scene_tracker,
      terminator_key: terminator,
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
