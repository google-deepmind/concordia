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

from concordia.typing import entity_component
from concordia.typing import memory as memory_lib

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
