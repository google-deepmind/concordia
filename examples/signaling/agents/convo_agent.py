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

"""A conversational agent designed to produce 'pink noise' dynamics."""

import dataclasses
from typing import List, Mapping

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import concat_act_component
from concordia.components.agent import memory as memory_component
from concordia.components.agent import question_of_recent_memories
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging as logging_lib
from concordia.typing import prefab as prefab_lib


DEFAULT_OBSERVATION_COMPONENT_KEY = '__observation__'


class ImportantMemories(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A simple component to receive observations."""

  def __init__(
      self,
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      pre_act_label: str = 'Important Memories',
  ):
    """Initializes the observation component.

    Args:
      memory_component_key: Name of the memory component to add observations to
        in `pre_observe` and to retrieve observations from in `pre_act`.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
    """
    super().__init__(pre_act_label)
    self._memory_component_key = memory_component_key

  def _make_pre_act_value(self) -> str:
    """Returns the latest observations to preact."""
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )

    all_memories = memory.retrieve_recent(limit=5000)

    # 1. Get Formative Memories
    try:
      first_observation_index = next(
          i
          for i, memory in enumerate(all_memories)
          if '[observation]' in memory
      )
      formative_memories = all_memories[:first_observation_index]
      other_memories = all_memories[first_observation_index:]
    except StopIteration:
      formative_memories = all_memories[:]
      other_memories = []

    keep_tags = [
        '[Daily Personal Event',
        '[Daily Shared Setup]',
        '[Reflection]',
    ]
    important_other_memories = [
        memory
        for memory in other_memories
        if any(tag in memory for tag in keep_tags)
    ]

    # Get the Last Dialogue Block (Unchanged)
    recent_dialogue_reversed = []
    for memory in reversed(other_memories):
      if 'Event:' in memory:
        recent_dialogue_reversed.append(memory)
      elif recent_dialogue_reversed:
        break
    recent_dialogue = recent_dialogue_reversed[::-1]

    memories_to_include = set(important_other_memories) | set(recent_dialogue)
    other_memories_in_order = [
        mem for mem in other_memories if mem in memories_to_include
    ]
    final_context = list(formative_memories) + other_memories_in_order

    result = '\n'.join(final_context) + '\n'
    self._logging_channel(
        {'Key': self.get_pre_act_label(), 'Value': result.splitlines()}
    )

    return result

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to JSON data."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the component state from JSON data."""


class PinkNoiseStrategy(question_of_recent_memories.QuestionOfRecentMemories):
  """Decides conversational strategy based on pink noise principles."""

  def __init__(
      self,
      agent_name: str,
      model: language_model.LanguageModel,
      components: List[str],
      **kwargs,
  ):
    """Initializes the component.

    Args:
      agent_name: The name of the agent.
      model: The language model to use.
      components: A list of component names that this component depends on.
      **kwargs: Any arguments to pass to the superclass.
    """
    question = (
        f'As {agent_name}, your goal is to maintain an engaging conversation.'
        ' This means balancing stability (staying on topic) with flexibility'
        ' (introducing new, related ideas). Review the recent conversation.'
        ' Has the immediate micro-topic become interesting or repetitive?'
        ' Based on this, choose a strategy for what to say next:\nA.'
        ' **Converge:** Stay on the micro-topic to deepen the conversation for'
        ' several turns. Choose this if the topic is has more to explore.\nB.'
        ' **Diverge:** Broaden the topic by connecting it to a more abstract'
        ' theme, a related personal anecdote, or a question about them. Choose'
        ' this if the current micro-topic is becoming repetitive after several'
        " turns.\n Don't diverge too much, and don't introduce too many new"
        ' micro-topics. You should aim to stay on the current micro-topic for'
        ' a few turns, and then diverge.'
    )
    default_pre_act_label = '\n--- Conversational Strategy ---\n{question}'

    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label

    super().__init__(
        question=question,
        model=model,
        answer_prefix='',
        add_to_memory=False,
        memory_tag='[pink noise strategy]',
        components=components,
        **kwargs,
    )


class CurrentQuestion(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Displays the current question and options from ActionSpec."""

  def __init__(
      self,
      pre_act_label: str = '\n--- Current Question ---\n',
  ):
    """Initializes the component."""
    super().__init__()
    self._pre_act_label = pre_act_label
    self._action_spec: entity_lib.ActionSpec | None = None

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Cache the action spec."""
    self._action_spec = action_spec
    return ''

  def get_pre_act_label(self) -> str:
    """Returns the pre-act label."""
    return self._pre_act_label

  def get_pre_act_value(self) -> str:
    """Returns the formatted question and options."""
    if not self._action_spec:
      return 'No question available.'

    parts = [f'The question is: {self._action_spec.call_to_action}']
    if (
        self._action_spec.output_type != entity_lib.OutputType.FREE
        and self._action_spec.options
    ):
      parts.append('The options are:')
      for option in self._action_spec.options:
        parts.append(f'- {option}')

    result = '\n'.join(parts) + '\n'
    self._logging_channel(
        {'Key': self.get_pre_act_label(), 'Value': result.splitlines()}
    )
    return result

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""


class SwitchingActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
  """Switches acting components based on ActionSpec output type."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      basic_component_order: List[str],
      convo_component_order: List[str],
      randomize_choices: bool,
  ):
    super().__init__()
    self._basic_component_order = basic_component_order
    self._convo_component_order = convo_component_order
    self._basic_act = concat_act_component.ConcatActComponent(
        model=model,
        component_order=basic_component_order,
        randomize_choices=randomize_choices,
    )
    self._convo_act = concat_act_component.ConcatActComponent(
        model=model,
        component_order=convo_component_order,
        randomize_choices=randomize_choices,
    )

  def set_entity(
      self, entity: entity_agent_with_logging.EntityAgentWithLogging
  ):
    super().set_entity(entity)
    self._basic_act.set_entity(entity)
    self._convo_act.set_entity(entity)

  def set_logging_channel(self, logging_channel: logging_lib.LoggingChannel):
    """Sets the logging channel for the component."""
    super().set_logging_channel(logging_channel)
    self._basic_act.set_logging_channel(logging_channel)
    self._convo_act.set_logging_channel(logging_channel)

  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    if action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
      delegate = self._basic_act
      component_order = self._basic_component_order
    else:
      delegate = self._convo_act
      component_order = self._convo_component_order

    filtered_contexts = {
        key: contexts[key] for key in component_order if key in contexts
    }

    result = delegate.get_action_attempt(filtered_contexts, action_spec)
    self._logging_channel({'Summary': f'Action: {result}', 'Value': result})
    return result

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""


@dataclasses.dataclass
class ConversationalAgent(prefab_lib.Prefab):
  """A prefab for a conversational agent aiming for pink noise dynamics."""

  description: str = (
      'An entity that participates in conversations, aiming to create a '
      'dynamically balanced and engaging dialogue.'
  )
  params: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Debra',
          'randomize_choices': True,
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
    randomize_choices = self.params.get('randomize_choices', True)

    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

    instructions_key = 'Instructions'
    instructions = agent_components.instructions.Instructions(
        agent_name=entity_name,
        pre_act_label='\nInstructions',
    )

    observation_to_memory_key = 'Observation'
    observation_to_memory = agent_components.observation.ObservationToMemory()

    observation_key = DEFAULT_OBSERVATION_COMPONENT_KEY
    # to change to custom memory
    observation = ImportantMemories(
        memory_component_key=memory_key,
        pre_act_label='\n--- Important Events ---\n',
    )

    situation_perception_key = 'SituationPerception'
    situation_perception = question_of_recent_memories.SituationPerception(
        model=model,
        components=[observation_key],
        pre_act_label=(
            '\n--- Situational Awareness ---\nQuestion: What kind of '
            'situation is {entity_name} in right now? Response using 1-5 '
            'sentences.\nAnswer'
        ),
        num_memories_to_retrieve=1,
    )
    self_perception_key = 'SelfPerception'
    self_perception = question_of_recent_memories.SelfPerception(
        model=model,
        components=[observation_key],
        pre_act_label=(
            '\n--- Self Perception ---\nQuestion: What kind of person is '
            f'{entity_name}? What are their values and conservational '
            'style?\nAnswer'
        ),
        num_memories_to_retrieve=1,
    )
    person_by_situation_key = 'PersonBySituation'
    person_by_situation = agent_components.question_of_recent_memories.PersonBySituation(
        model=model,
        components=[
            observation_key,
            self_perception_key,
            situation_perception_key,
        ],
        pre_act_label=(
            '\n--- Person by Situation ---\nQuestion: What would a person like'
            f' {entity_name} do in a situation like this?\nAnswer'
        ),
        num_memories_to_retrieve=1,
    )
    last_sentence_key = 'LastSentence'
    last_sentence = question_of_recent_memories.QuestionOfRecentMemories(
        model=model,
        pre_act_label=(
            '\n--- Last Sentence ---\nQuestion: Is there something in the last'
            f' sentence in the conversation that {entity_name} could respond'
            ' to to move the conversation forward?\nAnswer'
        ),
        num_memories_to_retrieve=1,
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

    strategy_components = [
        situation_perception_key,
        self_perception_key,
        observation_key,
        last_sentence_key,
    ]
    pink_noise_strategy_key = 'PinkNoiseStrategy'
    pink_noise_strategy = PinkNoiseStrategy(
        model=model,
        agent_name=entity_name,
        components=strategy_components,
        num_memories_to_retrieve=1,
    )

    components_of_agent = {
        instructions_key: instructions,
        observation_to_memory_key: observation_to_memory,
        relevant_memories_key: relevant_memories,
        observation_key: observation,
        self_perception_key: self_perception,
        situation_perception_key: situation_perception,
        person_by_situation_key: person_by_situation,
        last_sentence_key: last_sentence,
        pink_noise_strategy_key: pink_noise_strategy,
        memory_key: memory,
    }

    basic_component_order = [
        instructions_key,
        observation_key,
        relevant_memories_key,
        self_perception_key,
        situation_perception_key,
        person_by_situation_key,
    ]
    convo_component_order = [
        instructions_key,
        observation_key,
        relevant_memories_key,
        self_perception_key,
        situation_perception_key,
        person_by_situation_key,
        last_sentence_key,
        pink_noise_strategy_key,
    ]

    act_component = SwitchingActComponent(
        model=model,
        basic_component_order=basic_component_order,
        convo_component_order=convo_component_order,
        randomize_choices=randomize_choices,
    )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=act_component,
        context_components=components_of_agent,
    )

    return agent
