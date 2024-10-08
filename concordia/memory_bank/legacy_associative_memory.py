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

"""A memory bank that uses AssociativeMemory to store and retrieve memories."""


from collections.abc import Mapping, Sequence
import dataclasses
import datetime
from typing import Any

from concordia.associative_memory import associative_memory
from concordia.typing import memory as memory_lib


# These are dummy scoring functions that will be used to know the appropriate
# method to call in AssociativeMemory.
@dataclasses.dataclass(frozen=True)
class RetrieveAssociative(memory_lib.MemoryScorer):
  """A memory scorer that uses associative retrieval."""

  use_recency: bool = True
  use_importance: bool = True
  add_time: bool = True
  sort_by_time: bool = True

  def __call__(self, query: str, text: str, **metadata: Any) -> float:
    return 0.0


@dataclasses.dataclass(frozen=True)
class RetrieveAssociativeWithoutRecencyOrImportance(memory_lib.MemoryScorer):
  """A memory scorer that uses associative retrieval."""

  use_recency: bool = False
  use_importance: bool = False
  add_time: bool = True
  sort_by_time: bool = True

  def __call__(self, query: str, text: str, **metadata: Any) -> float:
    return 0.0


@dataclasses.dataclass(frozen=True)
class RetrieveRegex(memory_lib.MemoryScorer):
  """A memory scorer that uses regex matching."""

  add_time: bool = True
  sort_by_time: bool = True

  def __call__(self, query: str, text: str, **metadata: Any) -> float:
    return 0.0


@dataclasses.dataclass(frozen=True)
class RetrieveTimeInterval(memory_lib.MemoryScorer):
  """A memory scorer that uses time interval matching."""

  time_from: datetime.datetime
  time_until: datetime.datetime
  add_time: bool = False

  def __call__(self, query: str, text: str, **metadata: Any) -> float:
    return 0.0


@dataclasses.dataclass(frozen=True)
class RetrieveRecent(memory_lib.MemoryScorer):
  """A memory scorer that uses recency."""

  add_time: bool = False

  def __call__(self, query: str, text: str, **metadata: Any) -> float:
    return 0.0


@dataclasses.dataclass(frozen=True)
class RetrieveRecentWithImportance(memory_lib.MemoryScorer):
  """A memory scorer that uses recency and importance."""

  add_time: bool = False

  def __call__(self, query: str, text: str, **metadata: Any) -> float:
    return 0.0


class AssociativeMemoryBank(memory_lib.MemoryBank):
  """A wrapper over AssociativeMemory."""

  def __init__(self, memory: associative_memory.AssociativeMemory):
    self._memory = memory

  def add(self, text: str, metadata: Mapping[str, Any]) -> None:
    self._memory.add(text, **metadata)

  def _texts_with_constant_score(
      self, texts: Sequence[str]) -> Sequence[memory_lib.MemoryResult]:
    return [memory_lib.MemoryResult(text=t, score=0.0) for t in texts]

  def retrieve(
      self,
      query: str,
      scoring_fn: memory_lib.MemoryScorer,
      limit: int,
  ) -> Sequence[memory_lib.MemoryResult]:
    if isinstance(scoring_fn, RetrieveAssociative):
      return self._texts_with_constant_score(self._memory.retrieve_associative(
          query=query,
          k=limit,
          use_recency=scoring_fn.use_recency,
          use_importance=scoring_fn.use_importance,
          add_time=scoring_fn.add_time,
          sort_by_time=scoring_fn.sort_by_time,
      ))
    elif isinstance(scoring_fn, RetrieveAssociativeWithoutRecencyOrImportance):
      return self._texts_with_constant_score(self._memory.retrieve_associative(
          query=query,
          k=limit,
          use_recency=scoring_fn.use_recency,
          use_importance=scoring_fn.use_importance,
          add_time=scoring_fn.add_time,
          sort_by_time=scoring_fn.sort_by_time,
      ))
    elif isinstance(scoring_fn, RetrieveRegex):
      del limit
      return self._texts_with_constant_score(self._memory.retrieve_by_regex(
          regex=query,
          add_time=scoring_fn.add_time,
          sort_by_time=scoring_fn.sort_by_time,
      ))
    elif isinstance(scoring_fn, RetrieveTimeInterval):
      del query, limit
      return self._texts_with_constant_score(
          self._memory.retrieve_time_interval(
              time_from=scoring_fn.time_from,
              time_until=scoring_fn.time_until,
              add_time=scoring_fn.add_time,
          ))
    elif isinstance(scoring_fn, RetrieveRecent):
      del query
      return self._texts_with_constant_score(self._memory.retrieve_recent(
          k=limit,
          add_time=scoring_fn.add_time,
      ))
    elif isinstance(scoring_fn, RetrieveRecentWithImportance):
      del query
      return [
          memory_lib.MemoryResult(text=t, score=s) for t, s in zip(
              *self._memory.retrieve_recent_with_importance(
                  k=limit, add_time=scoring_fn.add_time))]
    else:
      raise ValueError(
          'Unknown scoring function. Only instances of RetrieveAssociative, '
          'RetrieveAssociativeWithoutRecencyOrImportance, '
          'RetrieveRegex, RetrieveTimeInterval, RetrieveRecent, and '
          'RetrieveRecentWithImportance are supported.')
