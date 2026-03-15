# Copyright 2026 DeepMind Technologies Limited.
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

"""A consumer agent for the marketplace."""

import dataclasses
from typing import List, Mapping

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.agent import question_of_recent_memories
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib


class ImportantMemories(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A simple component to receive observations."""

  def __init__(
      self,
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      last_n_observations: int = 20,
      pre_act_label: str = 'Important Memories',
  ):
    super().__init__(pre_act_label)
    self._memory_component_key = memory_component_key
    self._last_n_observations = last_n_observations

  def _make_pre_act_value(self) -> str:
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )

    all_memories = memory.retrieve_recent(limit=5000)
    last_n_observations = list(all_memories[-self._last_n_observations :])

    try:
      first_observation_index = next(
          i
          for i, memory in enumerate(all_memories)
          if '[observation]' in memory
      )
      formative_memories = all_memories[:first_observation_index]
    except StopIteration:
      formative_memories = all_memories[:]

    reflection_memories = [
        memory for memory in all_memories if '[Reflection]' in memory
    ]

    event_tags = ['[Daily Personal Event ', '[Daily Shared Setup]']

    event_memories = [
        memory
        for memory in all_memories
        if any(tag in memory for tag in event_tags)
    ]

    final_context = (
        list(formative_memories)
        + reflection_memories
        + event_memories
        + last_n_observations
    )

    result = '\n'.join(final_context) + '\n'
    self._logging_channel(
        {'Key': self.get_pre_act_label(), 'Value': result.splitlines()}
    )

    return result

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass


class ConsumerEvaluation(question_of_recent_memories.QuestionOfRecentMemories):
  """Evaluates market options based on utility and goals."""

  def __init__(
      self,
      agent_name: str,
      components: List[str],
      **kwargs,
  ):
    question = (
        'As {agent_name}, review the current market situation based on the'
        ' recent observations. What item do they want the most in this'
        ' situation? Evaluate the options, considering their prices and'
        ' quality. State which item they want most the absolute maximum price'
        ' they are willing to pay for it, and the quantity they may want given'
        ' the use case of the product for their life'
    )
    default_pre_act_label = '\n--- Consumer Market Evaluation ---\n{question}'

    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label

    super().__init__(
        question=question,
        answer_prefix=f'{agent_name} ',
        add_to_memory=False,
        memory_tag='[consumer evaluation]',
        components=components,
        **kwargs,
    )


@dataclasses.dataclass
class Consumer(prefab_lib.Prefab):
  """A prefab implementing a consumer entity for the marketplace."""

  description: str = 'An entity that is a consumer in a marketplace.'
  params: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Alice',
          'goal': '',
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
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
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = ImportantMemories(
        memory_component_key=memory_key,
        pre_act_label='\n--- Important Events ---\n',
    )

    if entity_goal:
      goal_key = 'Goal'
      overarching_goal = agent_components.constant.Constant(
          state=entity_goal, pre_act_label='\nContext'
      )
    else:
      goal_key = None
      overarching_goal = None

    situation_perception_key = 'SituationPerception'
    situation_perception = (
        agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            components=[goal_key] if goal_key else [],
            pre_act_label=(
                f'\nQuestion: What situation is {entity_name} in right now?'
                '\nAnswer'
            ),
        )
    )
    self_perception_key = 'SelfPerception'
    self_perception = agent_components.question_of_recent_memories.SelfPerception(
        model=model,
        pre_act_label=(
            f'\nQuestion: What kind of person is {entity_name} and what are'
            ' their preferences?\nAnswer'
        ),
        num_memories_to_retrieve=100,
    )

    consumer_evaluation_key = 'ConsumerEvaluation'
    evaluation_components = [
        observation_key,
        situation_perception_key,
        self_perception_key,
    ]
    if goal_key:
      evaluation_components.insert(1, goal_key)
    consumer_evaluation = ConsumerEvaluation(
        model=model,
        agent_name=entity_name,
        components=evaluation_components,
    )

    relevant_memories_key = 'RelevantMemories'
    relevant_memories_components = [situation_perception_key]
    if goal_key:
      relevant_memories_components.append(goal_key)
    relevant_memories = (
        agent_components.all_similar_memories.AllSimilarMemories(
            model=model,
            components=relevant_memories_components,
            num_memories_to_retrieve=10,
            pre_act_label='\nRecalled memories and observations',
        )
    )

    components_of_agent = {
        instructions_key: instructions,
        observation_to_memory_key: observation_to_memory,
        relevant_memories_key: relevant_memories,
        self_perception_key: self_perception,
        situation_perception_key: situation_perception,
        consumer_evaluation_key: consumer_evaluation,
        observation_key: observation,
        memory_key: memory,
    }

    component_order = list(components_of_agent.keys())

    if overarching_goal is not None:
      components_of_agent[goal_key] = overarching_goal
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
