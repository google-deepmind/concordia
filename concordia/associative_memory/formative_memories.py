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
import logging
import re
from typing import Any
from concordia.associative_memory import associative_memory
from concordia.associative_memory import importance_function
from concordia.document import interactive_document
from concordia.language_model import language_model
from dateutil.relativedelta import relativedelta  # pylint: disable=g-importing-member

logger = logging.getLogger(__name__)

DEFAULT_DOB = datetime.datetime(year=1984, month=7, day=3, hour=0, minute=0)
DEFAULT_FORMATIVE_AGES = (6, 9, 13, 16, 21)
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

    question = (
        'Creative Writing Master Class\n'
        f'Write a fictional life story for a {gender} character called {name} '
        f'with the following traits:\n{str(traits_description)}.\nBegin the '
        f'story when {name} is very young and end it when they are quite old. '
        'The story should be no more than three paragraphs in total. '
        'The story may include details such as (but not limited to) any of the '
        'following: what their job is, what their typical day is like, what '
        'their goals, desires, hopes, dreams, and aspirations are, as well as '
        'their drives, duties, responsibilities, and obligations. It should '
        'clarify what gives them joy and what are they afraid of. It may '
        'include their friends and family. It should be a complete life story '
        'but it should not specify how or when their life begins or ends. The '
        f'reader should come away with a profound understanding of {name}.'
    )
    if context:
      question += f' Incorporate the following context: {context}'
    result = prompt.open_question(
        question,
        max_characters=5000,
        max_tokens=4500,
        terminators=[],
    )
    result = re.sub(r'\.\s', '.\n', result)
    return result

  def make_memories(
      self,
      agent_config: AgentConfig,
  ) -> associative_memory.AssociativeMemory:
    """Creates agent memory from the agent config."""

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
          mem.add(item, importance=agent_config.formative_memory_importance)

    if agent_config.specific_memories:
      specific_memories = agent_config.specific_memories.split('\n')
      for item in specific_memories:
        if item:
          mem.add(item, importance=agent_config.formative_memory_importance)

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
    prompt.statement('Creative Writing Master Class\n')
    prompt.statement('Character background story:\n\n' + description)

    question = (
        'Given the life story above, invent formative episodes from '
        f'the life of {agent_config.name} which could have taken '
        f'place at the following ages: {agent_config.formative_ages}. '
        'The episodes should be age appropriate and believeable. '
        f'They should be memorable events for {agent_config.name} and '
        'important for establishing who they are as a person. They should '
        f'be consistent with {agent_config.name}\'s personality. '
        f'Describe each episode from {agent_config.name}\'s perspective '
        'and use third-person limited point of view. Each episode must '
        'mention their age at the time the event occurred using language such '
        f'as "When {agent_config.name} was 5 years old, they experienced..." . '
        'Use past tense. Write no more than three sentences per episode. '
        'Separate episodes from one another by the delimiter "\n\n\n". Do not '
        'apply any other special formatting besides these delimiters.'
    )
    if agent_config.traits:
      question += (
          '\nTaken as a whole, these formative episodes from the life of '
          f'{agent_config.name} should explain their personality, which has '
          f'been described as: "{agent_config.traits}".')
    if agent_config.context:
      question += (
          'Make a few of the episodes relate to the '
          f'following context: "{agent_config.context}".'
      )

    aggregated_result = prompt.open_question(
        question=question,
        max_characters=8000,
        max_tokens=6000,
        terminators=[],
    )

    episodes = aggregated_result.split('\n\n\n')

    if len(episodes) != len(list(agent_config.formative_ages)):
      logger.warning(
          f'Number of generated formative episodes ({len(episodes)}) does ' +
          'not match number of formative ages ' +
          f'({len(list(agent_config.formative_ages))}).')

    for episode_age, episode in zip(agent_config.formative_ages, episodes):
      memory.add(
          episode,
          tags=['episode'],
          timestamp=(
              agent_config.date_of_birth + relativedelta(years=episode_age)),
          importance=agent_config.formative_memory_importance,
      )
