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

from collections.abc import Callable, Sequence
import datetime
from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


class AffectReflection(component.Component):
  """Implements a reflection component taking into account the agent's affect.

  This component recalls memories based salient recent feelings, concepts, and 
  events. It then tries to infer high-level insights based on the memories it
  retrieved. This makes its output depend both on recent events and on the
  agent's past experience in life.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      clock_now: Callable[[], datetime.datetime],
      components: Sequence[component.Component],
      name: str = 'reflections on feelings',
      num_salient_to_retrieve: int = 20,
      num_questions_to_consider: int = 3,
      num_to_retrieve_per_question: int = 10,
      verbose: bool = False,
  ):
    self._model = model
    self._memory = memory
    self._components = components
    self._state = ''
    self._agent_name = agent_name
    self._clock_now = clock_now
    self._name = name
    self._num_salient_to_retrieve = num_salient_to_retrieve
    self._num_questions_to_consider = num_questions_to_consider
    self._num_to_retrieve_per_question = num_to_retrieve_per_question
    self._verbose = verbose
    self._history = []

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def update(self) -> None:
    context = '\n'.join([
        f"{self._agent_name}'s " + (comp.name() + ': ' + comp.state())
        for comp in self._components
    ])
    salience_chain_of_thought = interactive_document.InteractiveDocument(
        self._model)

    query = f'salient event, period, feeling, or concept for {self._agent_name}'
    timed_query = f'[{self._clock_now()}] {query}'

    mem_retrieved = self._memory.retrieve_associative(
        timed_query,
        k=self._num_salient_to_retrieve,
        use_recency=True,
        add_time=True)
    mem_retrieved = '\n'.join(mem_retrieved)
    question_list = []

    questions = salience_chain_of_thought.open_question(
        (
            f'Recent feelings: {self.state()} \n' +
            f"{self._agent_name}'s relevant memory:\n" +
            f'{mem_retrieved}\n' +
            f'Current time: {self._clock_now()}\n' +
            '\nGiven the thoughts and beliefs above, what are the ' +
            f'{self._num_questions_to_consider} most salient high-level '+
            f'questions that can be answered about what {self._agent_name} ' +
            'might be feeling about the current moment?'),
        answer_prefix='- ',
        max_tokens=3000,
        terminators=(),
    ).split('\n')

    question_related_mems = []
    for question in questions:
      question_list.append(question)
      question_related_mems = self._memory.retrieve_associative(
          question,
          self._num_to_retrieve_per_question,
          use_recency=False,
          add_time=True)
    insights = []
    question_related_mems = '\n'.join(question_related_mems)

    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    insight = chain_of_thought.open_question(
        f'Selected memories:\n{question_related_mems}\n' +
        f'Recent feelings: {self.state()} \n' +
        'New context:\n' + context + '\n' +
        f'Current time: {self._clock_now()}\n' +
        'What high-level insight can be inferred from the above ' +
        f'statements about what {self._agent_name} might be feeling ' +
        'in the current moment?',
        max_tokens=2000,
        terminators=(),
    )
    insights.append(insight)

    self._state = '\n'.join(insights)
    if self._verbose:
      print(termcolor.colored(chain_of_thought.view().text(), 'green'), end='')

    update_log = {
        'Date': self._clock_now(),
        'Summary': 'affect reflection and insights',
        'State': self._state,
        'Questions prompt': (
            salience_chain_of_thought.view().text().splitlines()
        ),
        'Insights prompt': chain_of_thought.view().text().splitlines(),
    }
    self._history.append(update_log)
