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

"""An agent reflects on how they are currently feeling."""

from collections.abc import Mapping
import types

from concordia.clocks import game_clock
from concordia.components.agent.deprecated import action_spec_ignored
from concordia.components.agent.deprecated import memory_component
from concordia.deprecated.memory_bank import legacy_associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import entity_component
from concordia.typing.deprecated import logging


DEFAULT_PRE_ACT_KEY = '\nAffective reflections'

_ASSOCIATIVE_RETRIEVAL = legacy_associative_memory.RetrieveAssociative()


class AffectReflection(action_spec_ignored.ActionSpecIgnored):
  """Implements a reflection component taking into account the agent's affect.

  This component recalls memories based salient recent feelings, concepts, and
  events. It then tries to infer high-level insights based on the memories it
  retrieved. This makes its output depend both on recent events and on the
  agent's past experience in life.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      clock: game_clock.MultiIntervalClock,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      num_salient_to_retrieve: int = 20,
      num_questions_to_consider: int = 3,
      num_to_retrieve_per_question: int = 10,
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Generates affect reflection based on recent and salient memories.

    Args:
      model: a language model
      clock: the game clock is needed to know when is the current time
      memory_component_name: The name of the memory component from which to
        retrieve recent memories.
      components: The components to consider when reflecting. This is a mapping
        of the component name to a label to use in the prompt.
      num_salient_to_retrieve: retrieve this many salient memories.
      num_questions_to_consider: how many questions to ask self.
      num_to_retrieve_per_question: how many memories to retrieve per question.
      pre_act_key: Prefix to add to the output of the component when called in
        `pre_act`.
      logging_channel: The channel to use for debug logging.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._clock = clock
    self._num_salient_to_retrieve = num_salient_to_retrieve
    self._num_questions_to_consider = num_questions_to_consider
    self._num_to_retrieve_per_question = num_to_retrieve_per_question
    self._logging_channel = logging_channel
    self._previous_pre_act_value = ''

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    context = '\n'.join([
        f"{agent_name}'s"
        f' {prefix}:\n{self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    salience_chain_of_thought = interactive_document.InteractiveDocument(
        self._model
    )

    query = f'salient event, period, feeling, or concept for {agent_name}'
    timed_query = f'[{self._clock.now()}] {query}'

    memory = self.get_entity().get_component(
        self._memory_component_name, type_=memory_component.MemoryComponent
    )
    mem_retrieved = '\n'.join([
        mem.text
        for mem in memory.retrieve(
            query=timed_query,
            scoring_fn=legacy_associative_memory.RetrieveAssociative(
                use_recency=True, add_time=True
            ),
            limit=self._num_salient_to_retrieve,
        )
    ])

    question_list = []

    questions = salience_chain_of_thought.open_question(
        (
            f'Recent feelings: {self._previous_pre_act_value} \n'
            + f"{agent_name}'s relevant memory:\n"
            + f'{mem_retrieved}\n'
            + f'Current time: {self._clock.now()}\n'
            + '\nGiven the thoughts and beliefs above, what are the '
            + f'{self._num_questions_to_consider} most salient high-level '
            + f'questions that can be answered about what {agent_name} '
            + 'might be feeling about the current moment?'
        ),
        answer_prefix='- ',
        max_tokens=3000,
        terminators=(),
    ).split('\n')

    question_related_mems = []
    for question in questions:
      question_list.append(question)
      question_related_mems = [
          mem.text
          for mem in memory.retrieve(
              query=agent_name,
              scoring_fn=legacy_associative_memory.RetrieveAssociative(
                  use_recency=False, add_time=True
              ),
              limit=self._num_to_retrieve_per_question,
          )
      ]
    insights = []
    question_related_mems = '\n'.join(question_related_mems)

    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    insight = chain_of_thought.open_question(
        f'Selected memories:\n{question_related_mems}\n'
        + f'Recent feelings: {self._previous_pre_act_value} \n\n'
        + 'New context:\n'
        + context
        + '\n'
        + f'Current time: {self._clock.now()}\n'
        + 'What high-level insight can be inferred from the above '
        + f'statements about what {agent_name} might be feeling '
        + 'in the current moment?',
        max_tokens=2000,
        terminators=(),
    )
    insights.append(insight)

    result = '\n'.join(insights)

    self._previous_pre_act_value = result

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Salience chain of thought': (
            salience_chain_of_thought.view().text().splitlines()
        ),
        'Chain of thought': chain_of_thought.view().text().splitlines(),
    })

    return result

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to a dictionary."""
    return {
        'previous_pre_act_value': self._previous_pre_act_value,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:

    self._previous_pre_act_value = str(state['previous_pre_act_value'])
