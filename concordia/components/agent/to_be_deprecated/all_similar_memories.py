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

"""Return all memories similar to a prompt and filter them for relevance.
"""

from collections.abc import Callable, Sequence
import datetime
from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


class AllSimilarMemories(component.Component):
  """Get all memories similar to the state of the components and filter them."""

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
    """Initialize a component to report relevant memories (similar to a prompt).

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
    prompt_summary = prompt.open_question(
        'Summarize the statements above.', max_tokens=750
    )

    query = f'{self._agent_name}, {prompt_summary}'
    if self._clock_now is not None:
      query = f'[{self._clock_now()}] {query}'

    mems = '\n'.join(
        self._memory.retrieve_associative(
            query, self._num_memories_to_retrieve, add_time=True
        )
    )

    question = (
        'Select the subset of the following set of statements that is most '
        f'important for {self._agent_name} to consider right now. Whenever two '
        'or more statements are not mutally consistent with each other '
        'select whichever statement is more recent. Repeat all the '
        'selected statements verbatim. Do not summarize. Include timestamps. '
        'When in doubt, err on the side of including more, especially for '
        'recent events. As long as they are not inconsistent, revent events '
        'are usually important to consider.'
    )
    if self._clock_now is not None:
      question = f'The current date/time is: {self._clock_now()}.\n{question}'
    new_prompt = prompt.new()
    self._state = new_prompt.open_question(
        f'{question}\nStatements:\n{mems}',
        max_tokens=2000,
        terminators=(),
    )

    if self._verbose:
      print(termcolor.colored(prompt.view().text(), 'green'), end='')
      print(termcolor.colored(f'Query: {query}\n', 'green'), end='')
      print(termcolor.colored(new_prompt.view().text(), 'green'), end='')
      print(termcolor.colored(self._state, 'green'), end='')

    update_log = {
        'date': self._clock_now(),
        'Summary': self._name,
        'State': self._state,
        'Initial chain of thought': prompt.view().text().splitlines(),
        'Query': f'{query}',
        'Final chain of thought': new_prompt.view().text().splitlines(),
    }
    self._history.append(update_log)
