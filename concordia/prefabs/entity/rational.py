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

"""A prefab for a 'rational' agent."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """A prefab for a 'rational' agent, ported from the deprecated factory."""

  description: str = 'A rational agent that optimizes for its goal.'
  params: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Rational Agent',
          'goal': '',
          'randomize_choices': True,
          'prefix_entity_name': True,
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the rational agent.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      A rational entity agent.
    """
    entity_name = self.params.get('name', 'Rational Agent')
    entity_goal = self.params.get('goal', '')
    randomize_choices = self.params.get('randomize_choices', True)
    prefix_entity_name = self.params.get('prefix_entity_name', True)

    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

    instructions_key = 'Instructions'
    instructions = agent_components.instructions.Instructions(
        agent_name=entity_name,
        pre_act_label='\nInstructions',
    )

    observation_to_memory_key = 'ObservationToMemory'
    observation_to_memory = agent_components.observation.ObservationToMemory()

    observation_key = 'Observation'
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

    relevant_memories_key = 'RelevantMemories'
    relevant_memories = (
        agent_components.all_similar_memories.AllSimilarMemories(
            model=model,
            components=[situation_perception_key],
            num_memories_to_retrieve=10,
            pre_act_label='\nRecalled memories and observations',
        )
    )

    options_perception_key = 'AvailableOptionsPerception'
    options_perception = agent_components.question_of_recent_memories.AvailableOptionsPerception(
        model=model,
        components=[
            observation_key,
            relevant_memories_key,
            situation_perception_key,
        ],
        pre_act_label=(
            f'\nQuestion: Which options are available to {entity_name} '
            'right now?\nAnswer'
        ),
    )

    best_option_perception_key = 'BestOptionPerception'

    if entity_goal:
      goal_key = 'Goal'
      overarching_goal = agent_components.constant.Constant(
          state=entity_goal, pre_act_label='\nOverarching goal'
      )
      best_option_label = (
          f'\nQuestion: Of the options available to {entity_name}, and '
          'given their goal, which choice of action or strategy is '
          f'best for {entity_name} to take right now?\nAnswer'
      )
    else:
      goal_key = None
      overarching_goal = None
      best_option_label = (
          f'\nQuestion: Of the options available to {entity_name}, '
          'which choice of action or strategy is '
          f'best for {entity_name} to take right now?\nAnswer'
      )

    best_option_components = [
        observation_key,
        relevant_memories_key,
        situation_perception_key,
        options_perception_key,
    ]
    if overarching_goal:
      best_option_components.append(goal_key)

    best_option_perception = (
        agent_components.question_of_recent_memories.BestOptionPerception(
            model=model,
            components=best_option_components,
            pre_act_label=best_option_label,
        )
    )

    components_of_agent = {
        memory_key: memory,
        instructions_key: instructions,
        observation_to_memory_key: observation_to_memory,
        observation_key: observation,
        situation_perception_key: situation_perception,
        relevant_memories_key: relevant_memories,
        options_perception_key: options_perception,
        best_option_perception_key: best_option_perception,
    }

    component_order = [
        instructions_key,
        observation_key,
        situation_perception_key,
        relevant_memories_key,
        options_perception_key,
        best_option_perception_key,
    ]

    if overarching_goal is not None:
      components_of_agent[goal_key] = overarching_goal
      component_order.insert(1, goal_key)

    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        component_order=component_order,
        randomize_choices=randomize_choices,
        prefix_entity_name=prefix_entity_name,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=act_component,
        context_components=components_of_agent,
    )
