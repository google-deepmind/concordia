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
import datetime
from typing import Callable
from typing import Sequence

from concordia.associative_memory.deprecated import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import component
import termcolor


class PersonBySituation(component.Component):
  """What would a person like the agent do in a situation like this?"""

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
    """Initializes the PersonBySituation component.

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

  def update(self) -> None:
    if self._clock_now() == self._last_update:
      return
    self._last_update = self._clock_now()

    prompt = interactive_document.InteractiveDocument(self._model)

    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._num_memories_to_retrieve, add_time=True
        )
    )

    prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

    component_states = '\n'.join([
        f"{self._agent_name}'s "
        + (construct.name() + ':\n' + construct.state())
        for construct in self._components
    ])

    prompt.statement(
        f'***\nCurrent time: {self._clock_now()}\n' + component_states)
    question = (
        f'What would a person like {self._agent_name} do in a situation like'
        ' this?'
    )
    if self._clock_now is not None:
      question = f'{question}'

    old_state = self._state
    self._state = prompt.open_question(
        question,
        answer_prefix=f'{self._agent_name} would ',
        max_tokens=1000,
    )

    self._state = f'{self._agent_name} would {self._state}'

    if old_state != self._state:
      self._memory.add(f'[intent reflection] {self._state}')

    self._last_chain = prompt
    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

    update_log = {
        'date': self._clock_now(),
        'Summary': question,
        'State': self._state,
        'Chain of thought': prompt.view().text().splitlines(),
    }
    self._history.append(update_log)
