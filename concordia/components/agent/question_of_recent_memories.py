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

"""Agent component for self perception."""

from collections.abc import Callable, Mapping
import datetime
import types

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.typing import logging


class QuestionOfRecentMemories(action_spec_ignored.ActionSpecIgnored):
  """A question that conditions the agent's behavior.

  The default question is 'What would a person like {agent_name} do in a
  situation like this?' and the default answer prefix is '{agent_name} would '.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      pre_act_key: str,
      question: str,
      answer_prefix: str,
      add_to_memory: bool,
      memory_tag: str = '',
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the QuestionOfRecentMemories component.

    Args:
      model: The language model to use.
      pre_act_key: Prefix to add to the value of the component when called in
        `pre_act`.
      question: The question to ask.
      answer_prefix: The prefix to add to the answer.
      add_to_memory: Whether to add the answer to the memory.
      memory_tag: The tag to use when adding the answer to the memory.
      memory_component_name: The name of the memory component from which to
        retrieve recent memories.
      components: The components to condition the answer on. This is a mapping
        of the component name to a label to use in the prompt.
      clock_now: time callback to use.
      num_memories_to_retrieve: The number of recent memories to retrieve.
      logging_channel: channel to use for debug logging.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._question = question
    self._answer_prefix = answer_prefix
    self._add_to_memory = add_to_memory
    self._memory_tag = memory_tag

    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name

    memory = self.get_entity().get_component(
        self._memory_component_name, type_=memory_component.MemoryComponent
    )
    recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
    mems = '\n'.join([
        mem.text
        for mem in memory.retrieve(
            scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve
        )
    ])

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Recent observations of {agent_name}:\n{mems}')

    if self._clock_now is not None:
      prompt.statement(f'Current time: {self._clock_now()}.\n')

    component_states = '\n'.join([
        f"{agent_name}'s"
        f' {prefix}:\n{self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    prompt.statement(component_states)

    question = self._question.format(agent_name=agent_name)
    result = prompt.open_question(
        question,
        answer_prefix=self._answer_prefix.format(agent_name=agent_name),
        max_tokens=1000,
    )
    result = self._answer_prefix.format(agent_name=agent_name) + result

    if self._add_to_memory:
      memory.add(f'{self._memory_tag} {result}', metadata={})

    log = {
        'Key': self.get_pre_act_key(),
        'Summary': question,
        'State': result,
        'Chain of thought': prompt.view().text().splitlines(),
    }

    if self._clock_now is not None:
      log['Time'] = self._clock_now()

    self._logging_channel(log)

    return result


class SelfPerception(QuestionOfRecentMemories):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      **kwargs,
  ):
    super().__init__(
        question='Given the above, what kind of person is {agent_name}?',
        answer_prefix='{agent_name} is ',
        add_to_memory=True,
        memory_tag='[self reflection]',
        **kwargs,
    )


class SituationPerception(QuestionOfRecentMemories):
  """This component answers the question 'what kind of situation is it?'."""

  def __init__(
      self,
      **kwargs,
  ):
    super().__init__(
        question=(
            'Given the statements above, what kind of situation is'
            ' {agent_name} in right now?'
        ),
        answer_prefix='{agent_name} is currently ',
        add_to_memory=False,
        **kwargs,
    )


class PersonBySituation(QuestionOfRecentMemories):
  """What would a person like the agent do in a situation like this?"""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            'What would a person like {agent_name} do in a situation like this?'
        ),
        answer_prefix='{agent_name} would ',
        add_to_memory=True,
        memory_tag='[intent reflection]',
        **kwargs,
    )


class AvailableOptionsPerception(QuestionOfRecentMemories):
  """This component answers the question 'what actions are available to me?'."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            'Given the statements above, what actions are available to '
            '{agent_name} right now?'
        ),
        answer_prefix='',
        add_to_memory=False,
        **kwargs,
    )


class BestOptionPerception(QuestionOfRecentMemories):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "Given the statements above, which of {agent_name}'s options "
            'has the highest likelihood of causing {agent_name} to achieve '
            'their goal? If multiple options have the same likelihood, select '
            'the option that {agent_name} thinks will most quickly and most '
            'surely achieve their goal.'
        ),
        answer_prefix="{agent_name}'s best course of action is ",
        add_to_memory=False,
        **kwargs,
    )
