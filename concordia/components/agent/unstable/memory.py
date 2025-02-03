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

from collections.abc import Callable, Mapping, Sequence
import threading
from typing import Any

from concordia.associative_memory.unstable import basic_associative_memory
from concordia.typing import entity_component
import pandas as pd


DEFAULT_MEMORY_COMPONENT_NAME = '__memory__'


class Memory(entity_component.ContextComponent):
  """A component backed by a memory bank.

  This component caches additions to the memory bank issued within an `act` or
  `observe` call. The new memories are committed to the memory bank during the
  `UPDATE` phase.
  """

  def __init__(
      self,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ):
    """Initializes the agent.

    Args:
      memory_bank: The memory bank to use.
    """
    self._memory_bank = memory_bank
    self._lock = threading.Lock()
    self._buffer = []

  def _check_phase(self) -> None:
    if self.get_entity().get_phase() == entity_component.Phase.UPDATE:
      raise ValueError(
          'You can only access the memory outside of the `UPDATE` phase.'
      )

  def get_state(self) -> Mapping[str, Any]:
    with self._lock:
      return self._memory_bank.get_state()

  def set_state(self, state: Mapping[str, Any]) -> None:
    with self._lock:
      self._memory_bank.set_state(state)

  def retrieve_associative(
      self,
      query: str,
      limit: int = 1,
  ) -> Sequence[str]:
    """Retrieves memories most closely matching a query.

    Args:
      query: The query to use for retrieval.
      limit: The number of memories to retrieve.

    Returns:
      A list of memory results, sorted by their cosine similarity to the query.
    """
    self._check_phase()

    with self._lock:
      return self._memory_bank.retrieve_associative(query, limit)

  def retrieve_recent(
      self,
      limit: int = 1,
  ) -> Sequence[str]:
    """Retrieves most recent memories.

    Args:
      limit: The number of memories to retrieve.

    Returns:
      A list of memory results, sorted by their recency.
    """

    self._check_phase()

    with self._lock:
      return self._memory_bank.retrieve_recent(limit)

  def scan(
      self,
      selector_fn: Callable[[str], bool],
  ) -> Sequence[str]:
    """Retrieves selected memories.

    Args:
      selector_fn: The selector function that returns True for memories to
        retrieve.

    Returns:
      A list of memory results, sorted by their recency.
    """
    self._check_phase()
    with self._lock:
      return self._memory_bank.scan(selector_fn)

  def add(
      self,
      text: str,
  ) -> None:
    self._check_phase()
    with self._lock:
      self._buffer.append(text)

  def extend(
      self,
      texts: Sequence[str],
  ) -> None:
    self._check_phase()
    for text in texts:
      self.add(text)

  def update(
      self,
  ) -> None:
    with self._lock:
      for mem in self._buffer:
        self._memory_bank.add(mem)
      self._buffer = []

  def get_raw_memory(self) -> pd.DataFrame:
    """Returns the raw memory as a pandas dataframe."""
    self._check_phase()
    with self._lock:
      return self._memory_bank.get_data_frame()

  def get_all_memories_as_text(self) -> Sequence[str]:
    """Returns all memories in the memory bank as a sequence of strings."""
    self._check_phase()
    with self._lock:
      texts = self._memory_bank.get_all_memories_as_text()
      return texts
