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

"""Agent component for situation perception."""

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock
from concordia.typing import component
import termcolor


class SituationPerception(component.Component):
  """This component answers the question 'what kind of situation is it?'."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      state_clock: clock.GameClock | None = None,
      num_memories_to_retrieve: int = 25,
      verbose: bool = False,
  ):
    """Initializes the component.

    Args:
      name: The name of the component.
      model: The language model to use.
      memory: The memory to use.
      agent_name: The name of the agent.
      state_clock: The clock to use.
      num_memories_to_retrieve: The number of memories to retrieve.
      verbose: Whether to print the last chain.
    """
    self._verbose = verbose
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name
    self._clock = state_clock
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._name = name

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state

  def update(self) -> None:
    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._num_memories_to_retrieve, add_time=True
        )
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

    if self._clock is not None:
      prompt.statement(f'Current time: {self._clock.now()}.\n')

    question = (
        'Given the memories above, what kind of situation is'
        f' {self._agent_name} in?'
    )

    self._state = prompt.open_question(
        question,
        answer_prefix=f'{self._agent_name} is currently ',
        max_characters=3000,
        max_tokens=1000,
    )
    self._state = f'{self._agent_name} is currently {self._state}'

    self._last_chain = prompt
    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')
