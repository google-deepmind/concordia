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

"""An agent represents their illness."""

from collections.abc import Callable, Sequence
import datetime
from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


class IllnessRepresentation(component.Component):
  """The agent represents their illness."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      components: Sequence[component.Component] | None = None,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
      verbose: bool = False,
  ):
    """Initializes the IllnessRepresentation component.

    Args:
      name: The name of the component.
      model: The language model to use.
      memory: The memory to use.
      agent_name: The name of the agent.
      components: The components to condition the answer on.
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: The number of memories to retrieve.
      verbose: Whether to print the state of the component.
    """

    self._verbose = verbose
    self._model = model
    self._memory = memory
    self._state = ''
    self._components = components or []
    self._agent_name = agent_name
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._name = name
    self._last_update = datetime.datetime.min
    self._history = []

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def get_components(self) -> Sequence[component.Component]:
    return self._components

  def update(self) -> None:
    if self._clock_now() == self._last_update:
      return
    self._last_update = self._clock_now()

    prompt = interactive_document.InteractiveDocument(self._model)

    component_states = '\n'.join([
        f"{self._agent_name}'s {comp.name()}:\n{comp.state()}"
        for comp in self._components
    ])
    prompt.statement(f'Statements:\n{component_states}\n')
    prompt_summary = prompt.open_question('Summarize the statements above.')

    query = f'{self._agent_name}, {prompt_summary}'
    if self._clock_now is not None:
      query = f'[{self._clock_now()}] {query}'

    mems = '\n'.join(
        self._memory.retrieve_associative(
            query, self._num_memories_to_retrieve, add_time=True
        )
    )

    illness_chain = prompt.new()
    illness_chain.statement(component_states)
    illness_chain.statement(f'Memories of {self._agent_name}:\n{mems}')
    delimiter = ', '
    question = (
        f'What symptoms does {self._agent_name} experience? Separate them with '
        f'{delimiter}, for example: "headache{delimiter}pain{delimiter}nausea".'
    )
    symptoms_str = illness_chain.open_question(question, max_tokens=750)
    symptoms = symptoms_str.split(delimiter)
    relevance_to_illness = {}
    for symptom in symptoms:
      relevance_to_illness[symptom] = illness_chain.yes_no_question(
          (f'Does {self._agent_name} believe their {symptom} is related to ' +
           'an illness?'))

    illness_identity = illness_chain.open_question(
        (
            f'Does {self._agent_name} believe they have an '
            'illness? If so, which illness? Why?'
        ),
        max_tokens=900,
    )
    illness_consequences = illness_chain.open_question(
        (
            f'Does {self._agent_name} believe there are the consequences of the'
            ' above? If so, what are they?'
        ),
        max_tokens=900,
    )
    illness_to_do = illness_chain.open_question(
        (
            f'Given the above, does {self._agent_name} believe they should do '
            'anything in particular? What?'
        ),
        max_tokens=900,
    )

    summarization_prompt = prompt.new()
    summarization_prompt.statement(illness_identity)
    summarization_prompt.statement(illness_consequences)
    summarization_prompt.statement(illness_to_do)
    for symptom, relevant in relevance_to_illness.items():
      if relevant:
        summarization_prompt.statement(
            f'{self._agent_name} experiences {symptom}.')

    summary = summarization_prompt.open_question(
        f'How would {self._agent_name} summarize all the information above?',
        max_tokens=1500,
        answer_prefix=f'{self._agent_name} believes that ',
    )
    self._state = f'{self._agent_name} believes that {summary}'
    self._memory.add(f'[idea] {self._state}')

    if self._verbose:
      print(termcolor.colored(prompt.view().text(), 'green'), end='')
      print(termcolor.colored(
          summarization_prompt.view().text(), 'green'), end='')
      print(termcolor.colored(self._state, 'green'), end='')

    update_log = {
        'date': self._clock_now(),
        'Summary': self._name,
        'State': self._state,
        'Initial chain of thought': prompt.view().text().splitlines(),
        'Illness chain of thought': illness_chain.view().text().splitlines(),
        'Final chain of thought': (
            summarization_prompt.view().text().splitlines()),
    }
    self._history.append(update_log)
