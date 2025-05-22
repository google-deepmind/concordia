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

from collections.abc import Callable, Collection, Sequence
import dataclasses
import datetime
import logging
import re
from typing import Any

from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from dateutil.relativedelta import relativedelta  # pylint: disable=g-importing-member
import numpy as np


logger = logging.getLogger(__name__)

DEFAULT_DOB = datetime.datetime(year=1984, month=7, day=3, hour=0, minute=0)
DEFAULT_FORMATIVE_AGES = (6, 9, 13, 16, 19, 21, 23)


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
    extras: a field for the user to keep any experiment specific data they need
      to define an agent
  """

  name: str
  gender: str = ''
  traits: str = ''
  context: str = ''
  specific_memories: str = ''
  goal: str = ''
  date_of_birth: datetime.datetime | None = None
  formative_ages: Collection[int] = DEFAULT_FORMATIVE_AGES
  extras: dict[str, Any] = dataclasses.field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    """Converts the AgentConfig to a dictionary."""
    result = dataclasses.asdict(self)
    if self.date_of_birth is not None:
      result['date_of_birth'] = self.date_of_birth.isoformat()
    return result

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'AgentConfig':
    """Initializes an AgentConfig from a dictionary."""
    if 'date_of_birth' in data and data['date_of_birth'] is not None:
      date_of_birth = datetime.datetime.fromisoformat(
          data['date_of_birth']
      )
      data = data | {'date_of_birth': date_of_birth}
    return cls(**data)


class MemoryFactory:
  """Generator of formative memories."""

  def __init__(
      self,
      embedder: Callable[[str], np.ndarray],
  ):
    """Initializes the memory factory.

    Args:
      embedder: The text embedder to use
    """
    self._embedder = embedder

  def make_blank_memory(
      self,
  ) -> associative_memory.AssociativeMemoryBank:
    """Creates a blank memory.

    Returns a blank memory

    Returns:
      An empty memory structure
    """

    return associative_memory.AssociativeMemoryBank(
        sentence_embedder=self._embedder,
    )


class FormativeMemoryFactory:
  """Generator of formative memories."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      embedder: Callable[[str], np.ndarray],
      shared_memories: Sequence[str] = (),
      delimiter_symbol: str = '***',
      current_date: datetime.datetime | None = None,
  ):
    """Initializes the formative memory factory.

    Args:
      model: the language model to use for generating memories
      embedder: The text embedder to use
      shared_memories: memories to be added to all agents
      delimiter_symbol: the delimiter to use when splitting the generated
        episodes
      current_date: (optional) the date of the simulation, used to calculate
        the age of each individual at the time of the simulation.
    """
    self._model = model
    self._delimiter_symbol = delimiter_symbol
    self._blank_memory_factory_call = MemoryFactory(
        embedder=embedder).make_blank_memory
    self._shared_memories = shared_memories
    self._current_date = current_date

  def make_backstory(self, agent_config: AgentConfig) -> str:
    """Creates a backstory for an agent based on the data provided.

    Args:
      agent_config: structured description of an agent

    Returns:
      Descriptive text about the agent
    """
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement('----- Creative Writing Master Class -----\n')
    prompt.statement('Question: What is the protagonist\'s name?')
    prompt.statement(f'Answer: {agent_config.name}\n')
    prompt.statement('Question: Describe the setting or background.')
    shared_memories = '\n'.join(self._shared_memories)
    prompt.statement(f'Answer: {shared_memories}\n')

    question = (
        f'Write a life story for a {agent_config.gender} character '
        f'named {agent_config.name} ')
    if agent_config.date_of_birth is not None:
      question += (
          f'who was born in the year {str(agent_config.date_of_birth.year)} ')
    if agent_config.traits:
      question += (
          f'with the following traits: {agent_config.traits}. ')
    question += (
        f'Begin the story when {agent_config.name} is very young and end it '
        'when they are quite old. The story should be no more than four '
        'paragraphs in total. The story may include details such as (but not '
        'limited to) any of the following: what their job is or was, what '
        'their typical day was or is like, what their goals, desires, '
        'hopes, dreams, and aspirations are, and have been, as well as their '
        'drives, duties, responsibilities, and obligations. It should clarify '
        'what gives them joy and what are they afraid of. It may include their '
        'friends and family, as well as antagonists. It should be a complete '
        'life story for a complete person but it should not specify how '
        'their life ends. The reader should be left with a profound '
        f'understanding of {agent_config.name}.'
    )
    if agent_config.context:
      question += ('Incorporate the following context into the '
                   f'story: {agent_config.context}')
    result = prompt.open_question(
        question,
        max_tokens=4500,
        terminators=['\nQuestion', '-----'],
    )
    result = re.sub(r'\.\s', '.\n', result)
    return result

  def add_memories(
      self,
      memory: associative_memory.AssociativeMemoryBank,
      agent_config: AgentConfig,
  ) -> None:
    """Creates formative memories of the agent at specific ages based on traits.

    First, a series of descriptive statements will be generated and based on
    them the formative episodes. There is an option to add description to memory
    as well.
    Args:
      memory: the memory structure to fill
      agent_config: structured description of an agent
    """
    description = self.make_backstory(agent_config)
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
        f'be consistent with {agent_config.name}\'s personality and '
        f'circumstances. Describe each episode from {agent_config.name}\'s '
        'perspective and use third-person limited point of view. Each episode '
        'must mention their age at the time the event occurred using language '
        f'such as "When {agent_config.name} was 5 years old, they '
        'experienced..." . Use past tense. Write no more than three sentences '
        'per episode. Separate episodes from one another by the delimiter '
        f'"{self._delimiter_symbol}". Do not apply any other '
        'special formatting besides these delimiters.'
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
        max_tokens=6000,
        terminators=[],
    )

    episodes = list(aggregated_result.split(self._delimiter_symbol))

    # If some episodes are still missing then try to regenerate them.
    formative_ages_list = list(agent_config.formative_ages)
    if len(episodes) != len(formative_ages_list):
      num_missing = len(formative_ages_list) - len(episodes)
      if num_missing > 0:
        for age in list(formative_ages_list[len(episodes):]):
          episode = prompt.open_question(
              question=(
                  f"What is {agent_config.name}'s formative memory from "
                  f'age {age}?'
              ),
              max_tokens=1000,
              terminators=(self._delimiter_symbol, '.\n', '\nQuestion:'),
          )
          episodes.append(episode)

    if len(episodes) != len(formative_ages_list):
      logger.warning(
          'Warning: Number of generated formative episodes ' +
          f'({len(episodes)}) does not match number of formative ages ' +
          f'({len(formative_ages_list)}). This is just a warning and '
          'probably not problematic.')

    for episode_age, episode in zip(agent_config.formative_ages, episodes):
      if agent_config.date_of_birth is not None:
        timestamp = (
            agent_config.date_of_birth + relativedelta(years=episode_age))
        memory_to_add = f'[{timestamp}] {episode}'
      else:
        memory_to_add = episode
      memory.add(memory_to_add)

    if self._current_date and agent_config.date_of_birth is not None:
      age = relativedelta(self._current_date, agent_config.date_of_birth).years
      timestamp = self._current_date
      memory.add(
          f'[{timestamp}] {agent_config.name} is {age} years old.',
      )

  def make_memories(
      self,
      agent_config: AgentConfig,
  ) -> associative_memory.AssociativeMemoryBank:
    """Creates agent memory from the agent config."""

    mem = self._blank_memory_factory_call()
    # All players share generic memories.
    for item in self._shared_memories:
      mem.add(item)

    context = agent_config.context
    if agent_config.goal:
      context += '\n' + f'{agent_config.name}\'s goal is: {agent_config.goal}'

    self.add_memories(memory=mem, agent_config=agent_config)

    if context:
      context_items = context.split('\n')
      for item in context_items:
        if item:
          mem.add(item)

    if agent_config.specific_memories:
      specific_memories = agent_config.specific_memories.split('\n')
      for item in specific_memories:
        if item:
          mem.add(item)

    return mem
