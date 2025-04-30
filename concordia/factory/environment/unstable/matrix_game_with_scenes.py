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

from collections.abc import Mapping, Sequence

from concordia.agents.unstable import entity_agent_with_logging
from concordia.associative_memory.unstable import basic_associative_memory as associative_memory
from concordia.components.agent import unstable as actor_components
from concordia.components.game_master import unstable as gm_components
from concordia.language_model import language_model
from concordia.thought_chains.unstable import thought_chains as thought_chains_lib
from concordia.typing.unstable import scene as scene_lib


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build(
    model: language_model.LanguageModel,
    memory_bank: list[str] | associative_memory.AssociativeMemoryBank,
    player_names: Sequence[str],
    scenes: Sequence[scene_lib.ExperimentalSceneSpec],
    payoff_matrix_component: gm_components.payoff_matrix.PayoffMatrix,
    observation_queue: Mapping[str, list[str]],
    global_scene_counter: gm_components.scene_tracker.ThreadSafeCounter,
    name: str = 'decision rules',
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build a decision game master for decision scenes (matrix game stages).

  Args:
    model: The language model to use for game master.
    memory_bank: provide a memory_bank.
    player_names: The names of the players.
    scenes: The scenes to use for the game master.
    payoff_matrix_component: The payoff matrix component to use to track player
      decisions and compute payoffs.
    observation_queue: The observation queue to use for the game master.
    global_scene_counter: The global scene counter to use for the game master.
    name: The name of the game master to build.

  Returns:
    A game master specialized for decisions in a matrix game.
  """
  instructions = gm_components.instructions.Instructions()

  examples_synchronous = gm_components.instructions.ExamplesSynchronous()

  player_characters = gm_components.instructions.PlayerCharacters(
      player_characters=player_names,
  )

  observation_to_memory = actor_components.observation.ObservationToMemory()
  observation = actor_components.observation.LastNObservations(
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
      memory_component_key=actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
      scene_tracker_component_key=(
          gm_components.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY
      ),
  )

  next_action_spec = gm_components.next_acting.NextActionSpecFromSceneSpec(
      scenes=scenes,
      memory_component_key=actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
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
      actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: (
          observation.get_pre_act_label()),
  }

  event_resolution = gm_components.event_resolution.EventResolution(
      model=model,
      event_resolution_steps=event_resolution_steps,
      components=event_resolution_components,
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

  components_of_game_master = {
      _get_class_name(instructions): instructions,
      _get_class_name(examples_synchronous): examples_synchronous,
      _get_class_name(player_characters): player_characters,
      gm_components.scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY: (
          scene_tracker
      ),
      terminator_key: terminator,
      _get_class_name(observation_to_memory): observation_to_memory,
      display_events_key: display_events,
      actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: (
          observation
      ),
      actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: (
          actor_components.memory.AssociativeMemory(memory_bank=memory_bank)
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
      _get_class_name(payoff_matrix_component): payoff_matrix_component,
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
