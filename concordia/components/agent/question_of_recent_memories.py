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

"""Agent component for asking questions about the agent's recent memories."""

from collections.abc import Callable, Collection, Sequence
import datetime

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


SELF_PERCEPTION_QUESTION = (
    'What kind of person is {agent_name}? Respond using 1-5 sentences.')

SITUATION_PERCEPTION_QUESTION = (
    'What kind of situation is {agent_name} in right now? Respond using 1-5 '
    'sentences.'
)
PERSON_BY_SITUATION_QUESTION = (
    'What would a person like {agent_name} do in a situation like this? '
    'Respond using 1-5 sentences.'
)
AVAILABLE_OPTIONS_QUESTION = (
    'What actions are available to {agent_name} right now?'
)
BEST_OPTION_PERCEPTION_QUESTION = (
    "Which of {agent_name}'s options "
    'has the highest likelihood of causing {agent_name} to achieve '
    'their goal? If multiple options have the same likelihood, select '
    'the option that {agent_name} thinks will most quickly and most '
    'surely achieve their goal.'
)


class QuestionOfRecentMemories(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A question that conditions the agent's behavior.

  The default question is 'What would a person like {agent_name} do in a
  situation like this?' and the default answer prefix is '{agent_name} would '.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      pre_act_label: str,
      question: str,
      answer_prefix: str,
      add_to_memory: bool,
      memory_tag: str = '',
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      components: Sequence[str] = (),
      terminators: Collection[str] = ('\n',),
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
  ):
    """Initializes the QuestionOfRecentMemories component.

    Args:
      model: The language model to use.
      pre_act_label: Prefix to add to the value of the component when called in
        `pre_act`.
      question: The question to ask.
      answer_prefix: The prefix to add to the answer.
      add_to_memory: Whether to add the answer to the memory.
      memory_tag: The tag to use when adding the answer to the memory.
      memory_component_key: The name of the memory component from which to
        retrieve recent memories.
      components: Keys of components to condition the answer on.
      terminators: strings that must not be present in the model's response. If
        emitted by the model the response will be truncated before them.
      clock_now: time callback to use.
      num_memories_to_retrieve: The number of recent memories to retrieve.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._components = components
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._question = question
    self._terminators = terminators
    self._answer_prefix = answer_prefix
    self._add_to_memory = add_to_memory
    self._memory_tag = memory_tag

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'  {self.get_component_pre_act_label(key)}: '
        f'{self.get_named_component_pre_act_value(key)}')

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    mems = '\n'.join([
        mem
        for mem in memory.retrieve_recent(limit=self._num_memories_to_retrieve)
    ])

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Recent observations of {agent_name}:\n{mems}')

    if self._clock_now is not None:
      prompt.statement(f'Current time: {self._clock_now()}.\n')

    component_states = '\n'.join(
        [self._component_pre_act_display(key) for key in self._components]
    )
    prompt.statement(component_states)

    question = self._question.format(agent_name=agent_name)
    result = prompt.open_question(
        question,
        answer_prefix=self._answer_prefix.format(agent_name=agent_name),
        max_tokens=1000,
        terminators=self._terminators,
    )
    result = self._answer_prefix.format(agent_name=agent_name) + result

    if self._add_to_memory:
      memory.add(f'{self._memory_tag} {result}')

    log = {
        'Key': self.get_pre_act_label(),
        'Summary': question,
        'State': result,
        'Chain of thought': prompt.view().text().splitlines(),
    }

    if self._clock_now is not None:
      log['Time'] = self._clock_now()

    self._logging_channel(log)

    return result

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to JSON data."""
    with self._lock:
      return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the component state from JSON data."""


class QuestionOfRecentMemoriesWithoutPreAct(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """QuestionOfRecentMemories component that does not output to pre_act.
  """

  def __init__(self, *args, **kwargs):
    self._component = QuestionOfRecentMemories(*args, **kwargs)

  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    self._component.set_entity(entity)

  def _make_pre_act_value(self) -> str:
    return ''

  def get_pre_act_value(self) -> str:
    return self._component.get_pre_act_value()

  def get_pre_act_label(self) -> str:
    return self._component.get_pre_act_label()

  def pre_act(
      self,
      unused_action_spec: entity_lib.ActionSpec,
  ) -> str:
    del unused_action_spec
    return ''

  def update(self) -> None:
    self._component.update()


class SelfPerception(QuestionOfRecentMemories):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      **kwargs,
  ):
    default_pre_act_label = f'\n{SELF_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=SELF_PERCEPTION_QUESTION,
        answer_prefix='{agent_name} is ',
        add_to_memory=False,
        memory_tag='[self reflection]',
        **kwargs,
    )


class SelfPerceptionWithoutPreAct(QuestionOfRecentMemoriesWithoutPreAct):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      **kwargs,
  ):
    default_pre_act_label = f'\n{SELF_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=SELF_PERCEPTION_QUESTION,
        answer_prefix='{agent_name} is ',
        add_to_memory=False,
        memory_tag='[self reflection]',
        **kwargs,
    )


class SituationPerception(QuestionOfRecentMemories):
  """This component answers the question 'what kind of situation is it?'."""

  def __init__(
      self,
      **kwargs,
  ):
    default_pre_act_label = f'\n{SITUATION_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=SITUATION_PERCEPTION_QUESTION,
        answer_prefix='{agent_name} is currently ',
        add_to_memory=False,
        **kwargs,
    )


class SituationPerceptionWithoutPreAct(QuestionOfRecentMemoriesWithoutPreAct):
  """This component answers the question 'what kind of situation is it?'."""

  def __init__(
      self,
      **kwargs,
  ):
    default_pre_act_label = f'\n{SITUATION_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=SITUATION_PERCEPTION_QUESTION,
        answer_prefix='{agent_name} is currently ',
        add_to_memory=False,
        **kwargs,
    )


class PersonBySituation(QuestionOfRecentMemories):
  """What would a person like the agent do in a situation like this?"""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{PERSON_BY_SITUATION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=PERSON_BY_SITUATION_QUESTION,
        answer_prefix='{agent_name} would ',
        add_to_memory=False,
        memory_tag='[intent reflection]',
        **kwargs,
    )


class PersonBySituationWithoutPreAct(QuestionOfRecentMemoriesWithoutPreAct):
  """What would a person like the agent do in a situation like this?"""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{PERSON_BY_SITUATION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=PERSON_BY_SITUATION_QUESTION,
        answer_prefix='{agent_name} would ',
        add_to_memory=False,
        memory_tag='[intent reflection]',
        **kwargs,
    )


class AvailableOptionsPerception(QuestionOfRecentMemories):
  """This component answers the question 'what actions are available to me?'."""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{AVAILABLE_OPTIONS_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=AVAILABLE_OPTIONS_QUESTION,
        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=False,
        **kwargs,
    )


class AvailableOptionsPerceptionsWithoutPreAct(
    QuestionOfRecentMemoriesWithoutPreAct):
  """This component answers the question 'what actions are available to me?'."""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{AVAILABLE_OPTIONS_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=AVAILABLE_OPTIONS_QUESTION,
        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=False,
        **kwargs,
    )


class BestOptionPerception(QuestionOfRecentMemories):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{BEST_OPTION_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=BEST_OPTION_PERCEPTION_QUESTION,
        answer_prefix="{agent_name}'s best course of action is ",
        add_to_memory=False,
        **kwargs,
    )


class BestOptionPerceptionWithoutPreAct(QuestionOfRecentMemoriesWithoutPreAct):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{BEST_OPTION_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=BEST_OPTION_PERCEPTION_QUESTION,
        answer_prefix="{agent_name}'s best course of action is ",
        add_to_memory=False,
        **kwargs,
    )
