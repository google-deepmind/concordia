# Copyright 2024 DeepMind Technologies Limited.
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

"""Generates facts about an imagined world and highlights them when needed."""

import datetime
import logging
import random
from typing import Callable, Sequence

from concordia.agents.deprecated import deprecated_agent
from concordia.agents.deprecated import entity_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import component
import termcolor


logger = logging.getLogger(__name__)


class WorldBackgroundAndRelevance(component.Component):
  """Create factoids on a richly-imagined world and surface them as needed."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      world_building_elements: Sequence[str],
      players: Sequence[deprecated_agent.BasicAgent | entity_agent.EntityAgent],
      components: Sequence[component.Component] | None = None,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_factoids: int = 30,
      delimiter_symbol: str = ' *** ',
      num_memories_to_retrieve: int = 25,
      verbose: bool = False,
  ):
    self._name = name
    self._model = model
    self._memory = memory
    self._world_building_elements = world_building_elements
    self._players = players
    self._components = components

    self.num_factoids = num_factoids
    self._delimiter_symbol = delimiter_symbol
    self._num_memories_to_retrieve = num_memories_to_retrieve

    self._clock_now = clock_now
    if clock_now is None:
      self._clock_now = lambda: ''

    self._verbose = verbose
    self._history = []

    self.reset()

  def reset(self):
    self._state = ''
    self._partial_states = {player.name: '' for player in self._players}
    self._build_world(elements=self._world_building_elements)

  def _build_world(self, elements: Sequence[str]):
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement('World building\n')
    chain_of_thought.statement(
        'In order to create a rich and immersive setting, weave together '
        'all the following elicitations and considerations into a '
        'multi-scale mosaic of booming and buzzing color and context.')
    world_building_prompt_questions = [
        (
            'What is the physical landscape of the world? Are there forests, '
            'oceans, mountains, or deserts? How do these features '
            'influence cultures and societies?'
        ),
        (
            'What is history of the world? Were there ancient civilizations, '
            'significant events, major wars or upheavals in the past? What is '
            'the level of technology? Does technology shape daily life? What '
            'about warfare? What are the primary economic activities? What '
            'resources are valuable to people and why are they valuable? How '
            'wealthy are people generally? Is there substantial inequality? '
            'And, critically, how did historical events give rise to present '
            'conditions?'
        ),
        (
            'What cultures exist in the world? What are their customs, '
            'beliefs, and values? How are they different from one another and '
            'in what ways are they the same? What conflicts arise from the '
            'differences? Do religions or other similar belief systems exist? '
            'How prominent are they? Is there religious conflict or tension?'
        ),
        (
            'What political systems are in place? Are there kingdoms, empires, '
            'chiefdom, city-states, nomadic bands, or villages? How do they '
            'govern themselves, and what power dynamics exist between them? '
            'How much coercion exists? How hierarchical are the societies? '
            'Is there slavery? Are societies tight or loose? Are there '
            'complex kinship structures like clans? What role do they play in '
            'politics and governance?'
        ),
        (
            'What are the main sources of conflict in the world? Are there '
            'ongoing political struggles and wars? What about class conflict '
            'and ideological conflict? Which groups dislike which other '
            'groups? Why is this so?'
        ),
    ]
    permuted_world_building_prompt_questions = random.sample(
        world_building_prompt_questions, len(world_building_prompt_questions))
    # Ask the first question before inserting the user-provided elements.
    _ = chain_of_thought.open_question(
        question=permuted_world_building_prompt_questions[0],
        max_tokens=1000,
    )
    # Insert the user-provided elements.
    initial_user_elements = ', '.join(elements[:-1])
    final_user_element = elements[-1]
    chain_of_thought.statement(
        'Within the tapestry of elements making up the world, the following '
        'elements are some of the most foundational and important:\n'
        f'{initial_user_elements} and {final_user_element}.')
    for question in permuted_world_building_prompt_questions[1:]:
      _ = chain_of_thought.open_question(
          question=question,
          max_tokens=1000,
      )
    # Generate life stories for important people in the world.
    _ = chain_of_thought.open_question(
        question=(
            'Generate life stories for three or more important people in the '
            'history and present day of the world. Their lives should '
            'either be typical of their time and place or unusually pivotal. '
            'Emphasize their lived experience.'
            'At least one of the chosen lives should intersect with some of '
            'the themes embodied in the foundational elements:\n'
            f'{initial_user_elements} and {final_user_element}.'
        ),
        max_tokens=2000,
    )
    # Generate discrete factoids based on all of the above.
    example_elements = random.sample(elements, 3)
    factoids_str = chain_of_thought.open_question(
        question=(
            'Now comes the critical step in this berzerk form of world '
            'building: chop up all the aforementioned comments, '
            'considerations, contexts, ideas, knowledge, worldviews, beliefs, '
            'systems, cosmic forces, historical data, truths, falsehoods, '
            'invented pasts, and lived traditions. Yes, chop them all up. Put '
            'them in the blender. Distill them to their essence without losing '
            'anything of the full complexity they embody. The way to do this '
            f'is to turn them into a set of {self.num_factoids} "factoids" to '
            'be written down, saved for posterity, and used for everything '
            'that will come next. Here are the "rules" of the "game": 1) taken '
            f'as a whole, the set of {self.num_factoids} factoids must provide '
            'a wholly complete and accurate representation of the world, '
            'though decidedly an impressionistic one. 2) Each factoid should '
            'be no more than three sentences in length. 3) Separate each '
            'factoid from the others using the delimiter symbol '
            f'{self._delimiter_symbol}. For instance:\n'
            f'"{example_elements[0]}{self._delimiter_symbol}'
            f'{example_elements[1]}{self._delimiter_symbol}'
            f'{example_elements[2]}{self._delimiter_symbol}...". Do '
            'not apply any other special formatting besides these delimiters.'
        ),
        max_tokens=8000,
        terminators=(),
    )
    factoids = factoids_str.split(self._delimiter_symbol)
    if len(factoids) != self.num_factoids:
      logger.warning(
          'Number of generated facts (%d) does not match number requested %d.',
          len(factoids),
          self.num_factoids,
      )

    for factoid in factoids:
      self._memory.add(factoid, tags=['world fact'])

    if self._verbose:
      print(termcolor.colored(chain_of_thought.view().text(), 'red'))

  def name(self) -> str:
    return self._name

  def get_history(self):
    return self._history.copy()

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def get_components(self):
    return self._components

  def state(self) -> str:
    return self._state

  def partial_state(
      self,
      player_name: str,
  ) -> str:
    """Return a player-specific view of the component's state."""
    return self._partial_states[player_name]

  def update(self) -> None:
    prompt = interactive_document.InteractiveDocument(self._model)

    component_states = '\n'.join([
        f'{comp.name()}:\n{comp.state()}'
        for comp in self._components
    ])
    prompt.statement(f'Statements:\n{component_states}\n')
    prompt_summary = prompt.open_question(
        question='Summarize the statements above in a few sentences.',
        max_tokens=1500,
    )

    query = f'{prompt_summary}'
    if self._clock_now is not None:
      query = f'[{self._clock_now()}] {prompt_summary}'

    # Retrieve from the full set of game master memories, including the world
    # building background as well as recent events from the simulation itself.
    mems = '\n'.join(
        self._memory.retrieve_associative(
            query, self._num_memories_to_retrieve, add_time=True
        )
    )

    # Create a new thought chain doc to filter the set of results.
    new_prompt = prompt.new()
    new_prompt.statement(
        f'Statements about the world and its current state right now:\n{mems}'
    )

    question = (
        'Select the subset of the following set of statements about the world '
        'that it is most important for the game master to consider right now. '
        'Repeat all the selected statements verbatim. Do not summarize. '
        'When in doubt, err on the side of including more. Select statements '
        'in such a way as to encourage proactivity and diversity in the '
        'behavior of the player characters.'
    )
    if self._clock_now is not None:
      question = f'The current date/time is: {self._clock_now()}.\n{question}'

    self._state = new_prompt.open_question(
        question=f'{question}',
        max_tokens=2000,
        terminators=('\nQuestion',),
    )
    for player in self._players:
      self._partial_states[player.name] = new_prompt.open_question(
          question=(
              f'What part of the information above is salient to {player.name} '
              'right now? If none, then leave the answer blank.'
          ),
          max_tokens=2000,
          terminators=('\nQuestion',),
      )

    if self._verbose:
      print(termcolor.colored(prompt.view().text(), 'red'), end='')
      print(termcolor.colored(f'Query: {query}\n', 'red'), end='')
      print(termcolor.colored(new_prompt.view().text(), 'red'), end='')
      print(termcolor.colored(self._state, 'red'), end='')

    update_log = {
        'date': self._clock_now(),
        'Summary': self._name,
        'State': self.state(),
        'Initial chain of thought': prompt.view().text().splitlines(),
        'Query': f'{query}',
        'Final chain of thought': new_prompt.view().text().splitlines(),
    }
    self._history.append(update_log)

  def update_after_event(self, event_statement: str) -> None:
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement(self.state() + '\n')
    chain_of_thought.statement(
        'In light of the above, the following event may have great '
        'significance.\n')
    chain_of_thought.statement(f'Event: {event_statement}\n')
    significance = chain_of_thought.open_question(
        question="What is the event's significance?",
        answer_prefix='Because of it, ',
        max_tokens=1000,
        terminators=('\nQuestion',),
    )
    self._memory.add(significance, tags=['event significance'])
