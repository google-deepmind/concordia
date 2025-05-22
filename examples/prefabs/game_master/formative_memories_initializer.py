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

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """A prefab entity implementing a formative memories initializer game master.
  """

  description: str = ('An initializer for all entities that '
                      'generates formative memories from their childhood.')
  params: Mapping[str, str] = dataclasses.field(
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
  entities: (
      Sequence[entity_agent_with_logging.EntityAgentWithLogging]
  ) = ()

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
    next_game_master_name = self.params.get('next_game_master_name',
                                            'default rules')
    shared_memories = self.params.get('shared_memories', [])

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

    observation_to_memory_key = 'observation_to_memory'
    observation_to_memory = actor_components.observation.ObservationToMemory()

    observation_component_key = (
        actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = actor_components.observation.LastNObservations(
        history_length=1000,
    )

    next_game_master_key = (
        gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY)
    next_game_master = (
        gm_components.next_game_master.FormativeMemoriesInitializer(
            model=model,
            next_game_master_name=next_game_master_name,
            player_names=player_names,
            shared_memories=shared_memories,
            player_specific_memories=self.params.get(
                'player_specific_memories', {}
            ),
            player_specific_context=self.params.get(
                'player_specific_context', {}
            ),
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

    components_of_game_master = {
        instructions_key: instructions,
        examples_synchronous_key: examples_synchronous,
        player_characters_key: player_characters,
        observation_component_key: observation,
        observation_to_memory_key: observation_to_memory,
        memory_component_key: memory_component,
        next_game_master_key: next_game_master,
        make_observation_key: make_observation,
        skip_next_action_spec_key: skip_next_action_spec,
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
