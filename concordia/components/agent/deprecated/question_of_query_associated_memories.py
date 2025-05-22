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

"""Agent question by query component."""

from collections.abc import Callable, Sequence
import datetime
import functools

from concordia.components.agent.deprecated import action_spec_ignored
from concordia.components.agent.deprecated import memory_component
from concordia.deprecated.memory_bank import legacy_associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import entity as entity_lib
from concordia.typing.deprecated import entity_component
from concordia.typing.deprecated import logging
from concordia.utils import concurrency


_ASSOCIATIVE_RETRIEVAL = legacy_associative_memory.RetrieveAssociative()


class QuestionOfQueryAssociatedMemories(action_spec_ignored.ActionSpecIgnored):
  """QuestionOfQueryAssociatedMemories component queries the memory and asks a question."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      queries: Sequence[str],
      question: str,
      pre_act_key: str,
      add_to_memory: bool = False,
      memory_tag: str = '',
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      clock_now: Callable[[], datetime.datetime] | None = None,
      summarization_question: str | None = None,
      num_memories_to_retrieve: int = 25,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initialize a QuestionOfQueryAssociatedMemories component.

    Args:
      model: a language model
      queries: strings to use as queries to the associative memory
      question: The question to ask the model of the retrieved memories. The
        question should be a format string with the following fields:
        - agent_name: the name of the agent
        - query: the query used to retrieve the memories
      pre_act_key: Prefix to add to the output of the component when called in
        `pre_act`.
      add_to_memory: Whether to add the answer to the memory.
      memory_tag: The tag to use when adding the answer to the memory.
      memory_component_name: The name of the memory component from which to
        retrieve related memories.
      clock_now: Function that returns the current time. If None, the current
        time will not be included in the question.
      summarization_question: if not None, the resulting state will be a one
        sentence summary, otherwise state it would be a concatenation of
        separate characteristics. The summary question should be a format string
        with the following fields:
        - agent_name: the name of the agent
        - results_str: the concatenation of the results of the queries.
      num_memories_to_retrieve: how many related memories to retrieve per query
      logging_channel: The channel to use for debug logging.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._clock_now = clock_now
    self._memory_component_name = memory_component_name
    self._add_to_memory = add_to_memory
    self._memory_tag = memory_tag

    self._queries = queries
    self._summarization_question = summarization_question
    self._num_memories_to_retrieve = num_memories_to_retrieve

    self._question = question

    self._logging_channel = logging_channel

  def _query_memory(self, query: str) -> str:
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_name, type_=memory_component.MemoryComponent
    )

    full_query = f"{agent_name}'s {query}"
    if self._clock_now:
      full_query = f'[{self._clock_now()}] {full_query}'

    mems = '\n'.join([
        mem.text
        for mem in memory.retrieve(
            query=full_query,
            scoring_fn=_ASSOCIATIVE_RETRIEVAL,
            limit=self._num_memories_to_retrieve,
        )
    ])
    prompt = interactive_document.InteractiveDocument(self._model)
    if self._clock_now:
      prompt.statement(f'Current time: {self._clock_now()}. ')

    question = self._question.format(query=query, agent_name=agent_name)

    result = prompt.open_question(
        '\n'.join([question, f'Statements:\n{mems}']),
        max_tokens=1000,
        answer_prefix=f'{agent_name} is ',
    )
    return result

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    results = concurrency.run_tasks({
        query: functools.partial(self._query_memory, query)
        for query in self._queries
    })
    results_str = '\n'.join(
        [f'{query}: {result}' for query, result in results.items()]
    )
    if self._summarization_question is not None:
      prompt = self._summarization_question.format(
          agent_name=agent_name,
          results_str=results_str,
      )
      output = f'{agent_name} is ' + self._model.sample_text(
          f'{prompt}\n {agent_name} is ', max_tokens=500
      )
    else:
      output = results_str

    if self._add_to_memory:
      memory = self.get_entity().get_component(
          self._memory_component_name, type_=memory_component.MemoryComponent
      )
      memory.add(f'{self._memory_tag} {output}', metadata={})

    all_queries = ', '.join(self._queries)
    log = {
        'Key': self.get_pre_act_key(),
        'Queries': f'{all_queries}',
        'State': output,
    }

    self._logging_channel(log)

    return output


class QuestionOfQueryAssociatedMemoriesWithoutPreAct(
    action_spec_ignored.ActionSpecIgnored
):
  """A QuestionOfQueryAssociatedMemories component that does not output its state to pre_act."""

  def __init__(self, *args, **kwargs):
    self._component = QuestionOfQueryAssociatedMemories(*args, **kwargs)

  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    self._component.set_entity(entity)

  def _make_pre_act_value(self) -> str:
    return ''

  def get_pre_act_value(self) -> str:
    return self._component.get_pre_act_value()

  def pre_act(
      self,
      unused_action_spec: entity_lib.ActionSpec,
  ) -> str:
    del unused_action_spec
    return ''

  def update(self) -> None:
    self._component.update()

  def get_state(self) -> entity_component.ComponentState:
    return self._component.get_state()

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._component.set_state(state)


class Identity(QuestionOfQueryAssociatedMemories):
  """Identity component containing a few characteristics.

  Identity is built out of individual characteristic queries to memory. For
  example, they could be:
  1. 'core characteristics',
  2. 'current daily occupation',
  3. 'feeling about recent progress in life',
  """

  def __init__(self, **kwargs):
    super().__init__(
        **kwargs,
        queries=[
            'core characteristics',
            'current daily occupation',
            'feeling about recent progress in life',
        ],
        question=(
            "How would one describe {agent_name}'s {query} given the "
            'following statements? '
        ),
        summarization_question=None,
    )


class IdentityWithoutPreAct(QuestionOfQueryAssociatedMemoriesWithoutPreAct):
  """An identity component that does not output its state to pre_act."""

  def __init__(self, *args, **kwargs):
    super().__init__(
        **kwargs,
        queries=[
            'core characteristics',
            'current daily occupation',
            'feeling about recent progress in life',
        ],
        question=(
            "How would one describe {agent_name}'s {query} given the "
            'following statements? '
        ),
        summarization_question=None,
    )


class SomaticState(QuestionOfQueryAssociatedMemories):
  """SomaticState component containing a few characteristics.

  Somatic state is comprised of hunger, thirst, fatigue, pain and loneliness.
  """

  def __init__(self, **kwargs):
    super().__init__(
        **kwargs,
        queries=[
            'level of hunger',
            'level of thirst',
            'level of fatigue',
            'level of pain',
            'level of loneliness',
        ],
        question=(
            "How would one describe {agent_name}'s {query} given the "
            'following statements? '
        ),
        summarization_question=(
            'Summarize the somatic state of {agent_name} in one sentence given'
            ' the readings below. Only mention readings that deviate from the'
            ' norm, for example if {agent_name} is not hungry then do not'
            ' mention hunger at all.\nReadings:\n{results_str}'
        ),
    )


class SomaticStateWithoutPreAct(QuestionOfQueryAssociatedMemoriesWithoutPreAct):
  """A SomaticState component that does not output its state to pre_act."""

  def __init__(self, *args, **kwargs):
    super().__init__(
        **kwargs,
        queries=[
            'level of hunger',
            'level of thirst',
            'level of fatigue',
            'level of pain',
            'level of loneliness',
        ],
        question=(
            "How would one describe {agent_name}'s {query} given the "
            'following statements? '
        ),
        summarization_question=(
            'Summarize the somatic state of {agent_name} in one sentence given'
            ' the readings below. Only mention readings that deviate from the'
            ' norm, for example if {agent_name} is not hungry then do not'
            ' mention hunger at all.\nReadings:\n{results_str}'
        ),
    )
