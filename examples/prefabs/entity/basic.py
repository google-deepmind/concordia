# Copyright 2024 DeepMind Technologies Limited.
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

"""A prefab containing the three key questions actor."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """A prefab implementing a basic actor entity."""

  description: str = (
      'An entity that makes decisions by asking '
      '"What situation am I in right now?", "What kind of person am I?", and '
      '"What would a person like me do in a situation like this?"')
  params: Mapping[str, str] = dataclasses.field(default_factory=lambda: {
      'name': 'Alice',
      'goal': '',
  })

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an entity.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      An entity.
    """
    entity_name = self.params.get('name', 'Alice')
    entity_goal = self.params.get('goal', '')

    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

    instructions_key = 'Instructions'
    instructions = agent_components.instructions.Instructions(
        agent_name=entity_name,
        pre_act_label='\nInstructions',
    )

    observation_to_memory_key = 'Observation'
    observation_to_memory = agent_components.observation.ObservationToMemory()

    observation_key = (
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY)
    observation = agent_components.observation.LastNObservations(
        history_length=100,
        pre_act_label=(
            '\nEvents so far (ordered from least recent to most recent)'
        ),
    )

    situation_perception_key = 'SituationPerception'
    situation_perception = (
        agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            pre_act_label=(
                f'\nQuestion: What situation is {entity_name} in right now?'
                '\nAnswer'
            ),
        )
    )
    self_perception_key = 'SelfPerception'
    self_perception = (
        agent_components.question_of_recent_memories.SelfPerception(
            model=model,
            pre_act_label=(
                f'\nQuestion: What kind of person is {entity_name}?\nAnswer'
            ),
        )
    )

    person_by_situation_key = 'PersonBySituation'
    person_by_situation = (
        agent_components.question_of_recent_memories.PersonBySituation(
            model=model,
            components=[
                self_perception_key,
                situation_perception_key,
            ],
            pre_act_label=(
                f'\nQuestion: What would a person like {entity_name} do in '
                'a situation like this?\nAnswer'),
        )
    )
    relevant_memories_key = 'RelevantMemories'
    relevant_memories = (
        agent_components.all_similar_memories.AllSimilarMemories(
            model=model,
            components=[
                situation_perception_key,
            ],
            num_memories_to_retrieve=10,
            pre_act_label='\nRecalled memories and observations',
        )
    )

    if entity_goal:
      goal_key = 'Goal'
      overarching_goal = agent_components.constant.Constant(
          state=entity_goal, pre_act_label='\nGoal'
      )
    else:
      goal_key = None
      overarching_goal = None

    components_of_agent = {
        instructions_key: instructions,
        observation_to_memory_key: observation_to_memory,
        relevant_memories_key: relevant_memories,
        self_perception_key: self_perception,
        situation_perception_key: situation_perception,
        person_by_situation_key: person_by_situation,
        observation_key: observation,
        memory_key: memory,
    }

    component_order = list(components_of_agent.keys())

    if overarching_goal is not None:
      components_of_agent[goal_key] = overarching_goal
      # Place goal after the instructions.
      component_order.insert(1, goal_key)

    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        component_order=component_order,
    )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=act_component,
        context_components=components_of_agent,
    )

    return agent
