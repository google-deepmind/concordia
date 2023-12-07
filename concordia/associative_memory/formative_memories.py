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


"""This is a factory for generating memories for concordia agents."""

from collections.abc import Callable, Iterable, Sequence
import dataclasses
import datetime
import re
from typing import Any
from concordia.associative_memory import associative_memory
from concordia.associative_memory import importance_function
from concordia.document import interactive_document
from concordia.language_model import language_model
from dateutil.relativedelta import relativedelta  # pylint: disable=g-importing-member


DEFAULT_DOB = datetime.datetime(year=1984, month=7, day=3, hour=0, minute=0)
DEFAULT_FORMATIVE_AGES = (3, 7, 12, 16, 21)
DEFAULT_IMPORTANT_MODEL = importance_function.ConstantImportanceModel()


@dataclasses.dataclass(frozen=True, kw_only=True)
class AgentConfig:
  """A card that describes a player.

  Attributes:
    name: name of the agent.
    gender: the gender of the agent.
    traits: any traits to use while generating formative memories. For example,
      big five.
    context: agent formative memories will be generated with this context
    specific_memories: inject these specific memories. Split memories at newline
      characters. Can be left blank if not used.
    goal: defines agents goal. Can be left blank if not used.
    date_of_birth: the date of birth for the agent.
    formative_ages: ages at which the formative episodes will be created
    formative_memory_importance: the importance value of formative memories.
    extras: a field for the user to keep any experiment specific data they need
      to define an agent
  """

  name: str
  gender: str
  traits: str
  context: str = ''
  specific_memories: str = ''
  goal: str = ''
  date_of_birth: datetime.datetime = DEFAULT_DOB
  formative_ages: Iterable[int] = DEFAULT_FORMATIVE_AGES
  formative_memory_importance: float = 1.0
  extras: dict[str, Any] = dataclasses.field(default_factory=dict)


class FormativeMemoryFactory:
  """Generator of formative memories."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      shared_memories: Sequence[str] = (),
      blank_memory_factory_call: Callable[
          [], associative_memory.AssociativeMemory
      ],
  ):
    self._model = model
    self._blank_memory_factory_call = blank_memory_factory_call
    self._shared_memories = shared_memories

  def make_backstory(
      self, name: str, gender: str, traits_description: str, context: str | None
  ) -> str:
    """Creates a backstory of an agent based on traits.

    Args:
      name: name of the agent
      gender: gender of the agent
      traits_description: descriptive traits of an agent, for example big five
      context: any context to add to the generation, i.e. genre

    Returns:
      Descriptive text about the agent
    """
    prompt = interactive_document.InteractiveDocument(self._model)

    if context:
      prompt.statement(context)
    question = (
        f'Given the following trats:\n{str(traits_description)}'
        f'\n create a backstory about a {gender} character called {name}.'
        ' Write a summary of the person:'
        ' what their job is, what a typical day is is like, what are their'
        ' goals, desires, hopes, dreams, and aspirations. Also write about'
        ' their duties, responsibilities, and obligations. What gives them joy'
        ' and what are they afraid of. Write about their friends and what they'
        ' like to do. Also write about their current concerns.'
    )
    if context:
      question += f'Take into account the following context: {context}'
    result = prompt.open_question(
        question,
        max_characters=2500,
        max_tokens=2500,
        terminators=[],
    )
    result = re.sub(r'\.\s', '.\n', result)

    query = '\n'.join([
        (
            'Replace all the pronounce in the following text with the name'
            f' {name}.'
        ),
        'The text:',
        result,
    ])

    description = self._model.sample_text(query)
    description = re.sub(r'\.\s', '.\n', description)

    return description

  def make_memories(
      self,
      agent_config: AgentConfig,
  ) -> associative_memory.AssociativeMemory:
    """Creates agent memory from the agent card."""

    mem = self._blank_memory_factory_call()
    # All players share generic memories.
    for item in self._shared_memories:
      mem.add(item)

    context = agent_config.context
    if agent_config.goal:
      context += '\n' + agent_config.goal

    self.add_memories(memory=mem, agent_config=agent_config)

    if context:
      context_items = context.split('\n')
      for item in context_items:
        if item:
          mem.add(item, importance=1.0)

    if agent_config.specific_memories:
      specific_memories = agent_config.specific_memories.split('\n')
      for item in specific_memories:
        if item:
          mem.add(item, importance=1.0)

    return mem

  def add_memories(
      self,
      memory: associative_memory.AssociativeMemory,
      agent_config: AgentConfig,
  ) -> None:
    """Creates formative memories of the agent at specific ages based on traits.

    First, a series of descriptive statements will be generated and based on
    them the formative episodes. There is an option to add description to memory
    as well.
    Args:
      memory: the memory structure to fill
      agent_config: the card describing the agent properties
    """
    description = self.make_backstory(
        agent_config.name,
        agent_config.gender,
        agent_config.traits,
        agent_config.context,
    )
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement('Context: ' + description)

    for episode_age in agent_config.formative_ages:
      question = (
          'Given the context above, come up with a formative episode at the '
          + f'age of {episode_age}, which is consistent with'
          f" {agent_config.name}'s "
          + f"personality. Describe the episode from {agent_config.name}'s"
          ' perspective '
          + 'using third-person limited point of view. Mention their age at '
          + 'the time. Use past tense. Write no more than three sentences.'
      )
      if agent_config.context:
        question += (
            '\nThe generated episode should be specifically related to some'
            f' aspect of the following context: "{agent_config.context}"'
        )

      episode = prompt.open_question(
          question,
          max_characters=2000,
          max_tokens=2000,
          terminators=[],
      )
      memory.add(
          episode,
          tags=['episode'],
          timestamp=agent_config.date_of_birth
          + relativedelta(years=episode_age),
          importance=agent_config.formative_memory_importance,
      )
