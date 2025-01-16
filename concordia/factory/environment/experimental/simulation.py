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

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import agent as components_lib
from concordia.components.game_master import experimental as gm_components_lib
from concordia.environment.experimental import engine as engine_lib
from concordia.environment.experimental.engines import synchronous
from concordia.environment.scenes.experimental import runner
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component as entity_component_lib
from concordia.typing import scene as scene_lib
import numpy as np


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build_simulation(
    model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
    importance_model: importance_function.ImportanceModel,
    clock: game_clock.MultiIntervalClock,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    shared_memories: Sequence[str],
    memory: associative_memory.AssociativeMemory | None = None,
    supporting_players_at_fixed_locations: Sequence[str] | None = None,
    nonplayer_entities: Sequence[
        entity_component_lib.EntityWithComponents] = tuple([]),
) -> tuple[engine_lib.Engine,
           associative_memory.AssociativeMemory,
           entity_agent_with_logging.EntityAgentWithLogging]:
  """Build a simulation (i.e., an environment).

  Args:
    model: The language model to use for game master.
    embedder: The embedder to use for similarity retrieval of memories.
    importance_model: The importance model to use for game master memories.
    clock: The simulation clock.
    players: The players.
    shared_memories: Sequence of memories to be observed by all players.
    memory: optionally provide a prebuilt memory, otherwise build it here.
    supporting_players_at_fixed_locations: The locations where supporting
      characters who never move are located.
    nonplayer_entities: The non-player entities.

  Returns:
    A tuple consisting of a game master and its memory.
  """
  if memory is not None:
    game_master_memory = memory
  else:
    game_master_memory = associative_memory.AssociativeMemory(
        sentence_embedder=embedder,
        importance=importance_model.importance,
        clock=clock.now,
    )

  gm_memory_bank = legacy_associative_memory.AssociativeMemoryBank(
      game_master_memory)

  instructions = gm_components_lib.instructions.Instructions()

  player_names = [player.name for player in players]
  player_characters = gm_components_lib.instructions.PlayerCharacters(
      player_characters=player_names
  )

  scenario_knowledge = components_lib.constant.Constant(
      state='\n'.join(shared_memories),
      pre_act_key='\nBackground:\n',
  )

  nonplayer_entities_list = components_lib.constant.Constant(
      state='\n'.join([entity.name for entity in nonplayer_entities]),
      pre_act_key='\nNon-player entities:\n',
  )

  if supporting_players_at_fixed_locations is not None:
    supporting_character_locations_if_any = components_lib.constant.Constant(
        state='\n'.join(supporting_players_at_fixed_locations),
        pre_act_key='\nNotes:\n',
    )
  else:
    supporting_character_locations_if_any = components_lib.constant.Constant(
        state='',
        pre_act_key='\nNotes:\n',
    )

  entity_components = (
      instructions,
      player_characters,
      scenario_knowledge,
      supporting_character_locations_if_any,
      nonplayer_entities_list,
  )
  components_of_game_master = {_get_class_name(component): component
                               for component in entity_components}
  components_of_game_master[
      components_lib.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = (
          components_lib.memory_component.MemoryComponent(gm_memory_bank))
  component_order = list(components_of_game_master.keys())

  act_component = gm_components_lib.switch_act.SwitchAct(
      model=model,
      clock=clock,
      component_order=component_order,
  )

  game_master = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name='GM',
      act_component=act_component,
      context_components=components_of_game_master,
  )

  # Create the game master object
  env = synchronous.Synchronous()

  return env, game_master_memory, game_master


def run_simulation(
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    clock: game_clock.MultiIntervalClock,
    scenes: Sequence[scene_lib.ExperimentalSceneSpec],
    verbose: bool = False,
    compute_metrics: Callable[[Mapping[str, str]], None] | None = None,
) -> None:
  """Run a simulation.

  Args:
    players: The players.
    clock: The clock of the run.
    scenes: Sequence of scenes to simulate.
    verbose: Whether or not to print verbose debug information.
    compute_metrics: Optionally, a function to compute metrics.

  Returns:
    None
  """
  # Run the simulation.
  runner.run_scenes(
      scenes=scenes,
      players=players,
      clock=clock,
      verbose=verbose,
      compute_metrics=compute_metrics,
  )
