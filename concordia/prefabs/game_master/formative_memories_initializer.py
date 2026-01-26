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

"""A prefab containing an initializer game master to set initial memories."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib


def build_components(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    player_names: Sequence[str],
    next_game_master_name: str,
    shared_memories: Sequence[str],
    player_specific_memories: Mapping[str, Sequence[str]],
    player_specific_context: Mapping[str, str],
) -> dict[str, entity_component.ContextComponent]:
  """Build the components for a formative memories initializer game master.

  This function is separated from the prefab to allow reuse by extending
  prefabs that need to add additional components.

  Args:
    model: The language model to use.
    memory_bank: The memory bank to use.
    player_names: Names of all player entities.
    next_game_master_name: Name of the game master to transition to.
    shared_memories: Memories shared by all players.
    player_specific_memories: Memories specific to each player.
    player_specific_context: Context specific to each player.

  Returns:
    A dictionary mapping component keys to components.
  """
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

  next_game_master_key = (
      gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
  )
  next_game_master = (
      gm_components.formative_memories_initializer.FormativeMemoriesInitializer(
          model=model,
          next_game_master_name=next_game_master_name,
          player_names=player_names,
          shared_memories=shared_memories,
          player_specific_memories=player_specific_memories,
          player_specific_context=player_specific_context,
      )
  )

  make_observation_key = (
      gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
  )
  make_observation = gm_components.make_observation.MakeObservation(
      model=model,
      player_names=player_names,
  )

  skip_next_action_spec_key = (
      gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
  )
  skip_next_action_spec = gm_components.next_acting.FixedActionSpec(
      action_spec=entity_lib.skip_this_step_action_spec(),
  )

  terminate_key = gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
  terminate = gm_components.terminate.NeverTerminate()

  return {
      instructions_key: instructions,
      examples_synchronous_key: examples_synchronous,
      player_characters_key: player_characters,
      observation_component_key: observation,
      observation_to_memory_key: observation_to_memory,
      memory_component_key: memory_component,
      next_game_master_key: next_game_master,
      make_observation_key: make_observation,
      skip_next_action_spec_key: skip_next_action_spec,
      terminate_key: terminate,
  }


def build_game_master(
    model: language_model.LanguageModel,
    name: str,
    player_names: Sequence[str],
    components: dict[str, entity_component.ContextComponent],
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build the game master entity from components.

  Args:
    model: The language model to use.
    name: Name of the game master.
    player_names: Names of all player entities.
    components: Dictionary of components to use.

  Returns:
    The constructed game master entity.
  """
  component_order = list(components.keys())

  act_component = gm_components.switch_act.SwitchAct(
      model=model,
      entity_names=player_names,
      component_order=component_order,
  )

  game_master = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=name,
      act_component=act_component,
      context_components=components,
  )

  return game_master


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """A prefab entity implementing a formative memories initializer game master."""

  description: str = (
      'An initializer for all entities that '
      'generates formative memories from their childhood.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'initial setup rules',
          'next_game_master_name': 'default rules',
          # Provide a comma-separated list of shared memories to pass verbatim
          # to all entities and game masters.
          'shared_memories': [],
          'player_specific_context': {},
          'player_specific_memories': {},
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a game master (i.e. a kind of entity).

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      An entity.
    """
    name = self.params.get('name', 'initial setup rules')
    next_game_master_name = self.params.get(
        'next_game_master_name', 'default rules'
    )
    shared_memories = self.params.get('shared_memories', [])
    player_names = [entity.name for entity in self.entities]

    components = build_components(
        model=model,
        memory_bank=memory_bank,
        player_names=player_names,
        next_game_master_name=next_game_master_name,
        shared_memories=shared_memories,
        player_specific_memories=self.params.get(
            'player_specific_memories', {}
        ),
        player_specific_context=self.params.get('player_specific_context', {}),
    )

    return build_game_master(
        model=model,
        name=name,
        player_names=player_names,
        components=components,
    )
