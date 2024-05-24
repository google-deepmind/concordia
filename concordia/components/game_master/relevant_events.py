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


"""This component retrieves relevant events from the memory."""

from collections.abc import Callable
import datetime

from concordia.associative_memory import associative_memory
from concordia.language_model import language_model
from concordia.typing import component


class RelevantEvents(component.Component):
  """Tracks the status of players."""

  def __init__(
      self,
      clock_now: Callable[[], datetime.datetime],
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      name: str = 'Relevant events',
      num_memories_retrieved_for_update: int = 10,
      add_time: bool = True,
      use_recency: bool = True,
  ):
    """Initializes the component.

    Args:
      clock_now: Function that returns the current time.
      model: Language model.
      memory: Associative memory.
      name: Name of the component.
      num_memories_retrieved_for_update: Number of memories to retrieve when
        updating the state.
      add_time: Whether to add the time to the retrieved memories.
      use_recency: Whether to use recency in memory retrieval or not.
    """
    self._memory = memory
    self._model = model
    self._state = ''
    self._history = []
    self._clock_now = clock_now
    self._name = name
    self._num_memories_retrieved_for_update = num_memories_retrieved_for_update
    self._add_time = add_time
    self._use_recency = use_recency

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state

  def get_history(self):
    return self._history.copy()

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def update_before_event(self, action_attempt: str) -> None:
    mem_retrieved = self._memory.retrieve_associative(
        action_attempt,
        use_recency=self._use_recency,
        add_time=self._add_time,
        k=self._num_memories_retrieved_for_update,
    )

    mems = '\n'.join(mem_retrieved)
    self._state = mems

    update_log = {
        'date': self._clock_now(),
        'state': self._state,
        'action_attempt': action_attempt,
    }
    self._history.append(update_log)
