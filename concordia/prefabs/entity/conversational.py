# Copyright 2025 DeepMind Technologies Limited.
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

"""A conversational agent designed to produce engaging dynamics."""

import dataclasses
from typing import Mapping

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import question_of_recent_memories
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


CONVERSATION_DYNAMICS_QUESTION = (
    'As {agent_name}, your goal is to maintain an engaging conversation.'
    ' This means balancing stability (staying on topic) with flexibility'
    ' (introducing new, related ideas). Review the recent conversation.'
    ' Has the immediate micro-topic become interesting or repetitive?'
    ' Based on this, choose a strategy for what to say next:\nA.'
    ' **Converge:** Stay on the micro-topic to deepen the conversation for'
    ' several turns. Choose this if the topic has more to explore.\nB.'
    ' **Diverge:** Broaden the topic by connecting it to a more abstract'
    ' theme, a related personal anecdote, or a question about them. Choose'
    ' this if the current micro-topic is becoming repetitive after several'
    " turns.\n Don't diverge too much, and don't introduce too many new"
    ' micro-topics. You should aim to stay on the current micro-topic for'
    ' a few turns, and then diverge.'
)


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """A prefab for a conversational agent aiming for engaging dynamics."""

  description: str = (
      'An entity that participates in conversations, aiming to create a '
      'dynamically balanced and engaging dialogue.'
  )
  params: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Debra',
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build the conversational agent.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      An entity agent.
    """
    entity_name = self.params.get('name', 'Debra')
    conversation_style = self.params.get('conversation_style', '')

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
    last_sentence_key = 'LastSentence'
    last_sentence = question_of_recent_memories.QuestionOfRecentMemories(
        model=model,
        pre_act_label=(
            '\nQuestion: Is there something in the last'
            f' sentence in the conversation that {entity_name} could respond'
            ' to to move the conversation forward?\nAnswer'
        ),
        num_memories_to_retrieve=2,
        question=(
            'Is there something in the last sentence in the conversation that'
            f' {entity_name} could respond to to move the conversation forward?'
        ),
        answer_prefix='',
        add_to_memory=False,
    )

    relevant_memories_key = 'RelevantMemories'
    relevant_memories_components = [situation_perception_key]
    relevant_memories = (
        agent_components.all_similar_memories.AllSimilarMemories(
            model=model,
            components=relevant_memories_components,
            num_memories_to_retrieve=5,
            pre_act_label='\nRecalled relevant memories and observations',
        )
    )

    if conversation_style:
      convo_style_key = 'ConversationStyle'
      conversation_style = agent_components.constant.Constant(
          state=conversation_style,
          pre_act_label='\nConversation Style',
      )
    else:
      convo_style_key = None
      conversation_style = None

    convo_components = [
        situation_perception_key,
        self_perception_key,
        last_sentence_key,
    ]
    if convo_style_key:
      convo_components.insert(2, convo_style_key)
    conversation_dynamics_key = 'ConversationDynamics'
    conversation_dynamics = (
        question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f'\n{CONVERSATION_DYNAMICS_QUESTION}',
            question=CONVERSATION_DYNAMICS_QUESTION,
            components=convo_components,
            num_memories_to_retrieve=100,
            answer_prefix='',
            add_to_memory=False,
            memory_tag='[conversation dynamics]',
        )
    )

    components_of_agent = {
        instructions_key: instructions,
        observation_to_memory_key: observation_to_memory,
        relevant_memories_key: relevant_memories,
        observation_key: observation,
        self_perception_key: self_perception,
        situation_perception_key: situation_perception,
        last_sentence_key: last_sentence,
        conversation_dynamics_key: conversation_dynamics,
        memory_key: memory,
    }

    component_order = list(components_of_agent.keys())

    if convo_style_key:
      components_of_agent[convo_style_key] = conversation_style
      component_order.insert(5, convo_style_key)

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
