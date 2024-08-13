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


"""Agent characteristic component."""
import datetime
from typing import Callable

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


class Characteristic(component.Component):
  """Implements a simple characteristic component.

  For example, "current daily occupation", "core characteristic" or "hunger".
  The component queries the memory for the agent's haracteristic and then
  summarises it.

  In psychology it is common to distinguish between `state` characteristics and
  `trait` characteristics. A `state` is temporary, like being hungry or afraid,
  but a `trait` endures over a long period of time, e.g. being neurotic or
  extroverted.

  When the characteristic is a `state` (as opposed to a `trait`) then time is
  used in the query for memory retrieval and the instruction for summarization.
  When the characteristic is a `trait` then time is not used.

  When you pass a `state_clock` while creating a characteristic then you create
  a `state` characteristic. When you do not pass a `state_clock` then you create
  a `trait` characteristic.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      characteristic_name: str,
      state_clock_now: Callable[[], datetime.datetime] | None = None,
      extra_instructions: str = '',
      num_memories_to_retrieve: int = 25,
      verbose: bool = False,
  ):
    """Represents a characteristic of an agent (a trait or a state).

    Args:
      model: a language model
      memory: an associative memory
      agent_name: the name of the agent
      characteristic_name: the string to use in similarity search of memory
      state_clock_now: if None then consider this component as representing a
        `trait`. If a clock is used then consider this component to represent a
        `state`. A state is temporary whereas a trait is meant to endure.
      extra_instructions: append additional instructions when asking the model
        to assess the characteristic.
      num_memories_to_retrieve: how many memories to retrieve during the update
      verbose: whether or not to print intermediate reasoning steps.
    """
    self._verbose = verbose
    self._model = model
    self._memory = memory
    self._cache = ''
    self._characteristic_name = characteristic_name
    self._agent_name = agent_name
    self._extra_instructions = extra_instructions
    self._clock_now = state_clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._history = []

  def name(self) -> str:
    return self._characteristic_name

  def state(self) -> str:
    return self._cache

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def update(self) -> None:
    query = f"{self._agent_name}'s {self._characteristic_name}"
    if self._clock_now is not None:
      query = f'[{self._clock_now()}] {query}'

    mems = '\n'.join(
        self._memory.retrieve_associative(
            query, self._num_memories_to_retrieve, add_time=True
        )
    )

    prompt = interactive_document.InteractiveDocument(self._model)

    question = (
        f"How would one describe {self._agent_name}'s"
        f' {self._characteristic_name} given the following statements? '
        f'{self._extra_instructions}'
    )
    if self._clock_now is not None:
      question = f'Current time: {self._clock_now()}.\n{question}'

    self._cache = prompt.open_question(
        '\n'.join([question, f'Statements:\n{mems}']),
        max_tokens=1000,
        answer_prefix=f'{self._agent_name} is ',
    )

    self._last_chain = prompt
    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'red'), end='')

    update_log = {
        'Summary': question,
        'State': self._cache,
        'Chain of thought': prompt.view().text().splitlines(),
    }
    self._history.append(update_log)
