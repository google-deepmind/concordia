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

"""The abstract class for a memory."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Protocol


class MemoryScorer(Protocol):
  """Typing definition for a memory scorer function."""

  def __call__(self, query: str, text: str, **metadata: Any) -> float:
    """Returns a score for a memory (text and metadata) given the query.

    Args:
      query: The query to use for retrieval.
      text: The text of the memory.
      **metadata: The metadata of the memory.
    """


@dataclasses.dataclass(frozen=True, kw_only=True)
class MemoryResult:
  """The result item of a memory bank retrieval.

  Attributes:
    text: The text of the memory.
    score: The score of the memory.
  """

  text: str
  score: float


class MemoryBank(metaclass=abc.ABCMeta):
  """Base class for memory banks."""

  @abc.abstractmethod
  def add(self, text: str, metadata: Mapping[str, Any]) -> None:
    """Adds a memory (in the form of text) to the memory bank.

    The memory bank might add extra metadata to the memory.

    Args:
      text: The text to add to the memory bank.
      metadata: The metadata associated with the memory.
    """
    raise NotImplementedError()

  def extend(self, texts: Sequence[str], metadata: Mapping[str, Any]) -> None:
    """Adds a sequence of memories (in the form of text) to the memory bank.

    All memories will be added with the same metadata. The memory bank might add
    extra metadata to the memories.

    Args:
      texts: The texts to add to the memory bank.
      metadata: The metadata associated with all the memories.
    """
    for text in texts:
      self.add(text, metadata)

  @abc.abstractmethod
  def retrieve(
      self,
      query: str,
      scoring_fn: MemoryScorer,
      limit: int,
  ) -> Sequence[MemoryResult]:
    """Retrieves memories from the memory bank using the given scoring function.

    This function retrieves the memories from the memory bank that are most
    relevant to the given query, according to the scoring function. The scoring
    function is a function that takes the query, a memory (in the form of text),
    and a dictionary of metadata and returns a score for the memory. The higher
    the score, the more relevant the memory is to the query.

    Args:
      query: The query to use for retrieval.
      scoring_fn: The scoring function to use.
      limit: The maximum number of memories to retrieve. If negative, all
        memories will be retrieved.

    Returns:
      A list of memories (in the form of text) and their scores that are most
      relevant to the `query`. This list will be of at most `limit` elements,
      unless `limit` is negative, in which case all memories will be returned.
    """
    raise NotImplementedError()
