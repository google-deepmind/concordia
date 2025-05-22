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

"""A component backed by a memory bank."""

from collections.abc import Mapping, Sequence
import threading
from typing import Any

from concordia.typing.deprecated import entity_component
from concordia.typing.deprecated import memory as memory_lib
import pandas as pd


DEFAULT_MEMORY_COMPONENT_NAME = '__memory__'


def _default_scorer(query: str, text: str, **metadata: Any) -> float:
  """A memory scorer that always returns a default value of 1.0."""
  del query, text, metadata  # Unused.
  return 1.0


class MemoryComponent(entity_component.ContextComponent):
  """A component backed by a memory bank.

  This component caches additions to the memory bank issued within an `act` or
  `observe` call. The new memories are committed to the memory bank during the
  `UPDATE` phase.
  """

  def __init__(
      self,
      memory: memory_lib.MemoryBank,
  ):
    """Initializes the agent.

    Args:
      memory: The memory bank to use.
    """
    self._memory = memory
    self._lock = threading.Lock()
    self._buffer = []

  def _check_phase(self) -> None:
    if self.get_entity().get_phase() == entity_component.Phase.UPDATE:
      raise ValueError(
          'You can only access the memory outside of the `UPDATE` phase.'
      )

  def get_state(self) -> Mapping[str, Any]:
    with self._lock:
      return self._memory.get_state()

  def set_state(self, state: Mapping[str, Any]) -> None:
    with self._lock:
      self._memory.set_state(state)

  def retrieve(
      self,
      query: str = '',
      scoring_fn: memory_lib.MemoryScorer = _default_scorer,
      limit: int = -1,
  ) -> Sequence[memory_lib.MemoryResult]:
    """Retrieves memories from the memory bank using the given scoring function.

    Args:
      query: The query to use for retrieval.
      scoring_fn: The scoring function to use for retrieval.
      limit: The number of memories to retrieve.

    Returns:
      A list of memory results.
    """
    self._check_phase()
    with self._lock:
      return self._memory.retrieve(query, scoring_fn, limit)

  def add(
      self,
      text: str,
      metadata: Mapping[str, Any],
  ) -> None:
    self._check_phase()
    with self._lock:
      self._buffer.append({'text': text, 'metadata': metadata})

  def extend(
      self,
      texts: Sequence[str],
      metadata: Mapping[str, Any],
  ) -> None:
    self._check_phase()
    for text in texts:
      self.add(text, metadata)

  def update(
      self,
  ) -> None:
    with self._lock:
      for mem in self._buffer:
        self._memory.add(mem['text'], mem['metadata'])
      self._buffer = []

  def get_raw_memory(self) -> pd.DataFrame:
    """Returns the raw memory as a pandas dataframe."""
    self._check_phase()
    with self._lock:
      return self._memory.get_data_frame()

  def get_all_memories_as_text(
      self,
      add_time: bool = True,
      sort_by_time: bool = True,
  ) -> Sequence[str]:
    """Returns all memories in the memory bank as a sequence of strings."""
    self._check_phase()
    with self._lock:
      texts = self._memory.get_all_memories_as_text(
          add_time=add_time,
          sort_by_time=sort_by_time)
      return texts
