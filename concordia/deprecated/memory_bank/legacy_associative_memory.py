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

from collections.abc import Sequence
import dataclasses
import datetime
from typing import Any, Mapping

from concordia.associative_memory.deprecated import associative_memory
from concordia.typing.deprecated import entity_component
from concordia.typing.deprecated import memory as memory_lib
import pandas as pd
from typing_extensions import override


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

  def get_data_frame(self) -> pd.DataFrame:
    """Returns the memory bank as a pandas dataframe."""
    return self._memory.get_data_frame()

  def get_all_memories_as_text(
      self,
      add_time: bool = True,
      sort_by_time: bool = True,
  ) -> Sequence[str]:
    """Returns all memories in the memory bank as a sequence of strings."""
    return self._memory.get_all_memories_as_text(add_time=add_time,
                                                 sort_by_time=sort_by_time)

  @override
  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the memory bank.
    
    See `set_state` for details. The default implementation returns an empty
    dictionary.
    """
    return self._memory.get_state()

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the memory bank.
    
    This is used to restore the state of the memory bank. The state is assumed 
    to be the one returned by `get_state`.
    The state does not need to contain any information that is passed in the 
    initialization of the memory bank (e.g. embedder, clock, imporance etc.)
    It is assumed that set_state is called on the memory bank after it was 
    initialized with the same parameters as the one used to restore it.
    The default implementation does nothing, which implies that the memory bank
    does not have any state.

    Example (Creating a copy):
      obj1 = MemoryBank(**kwargs)
      state = obj.get_state()
      obj2 = MemoryBank(**kwargs)
      obj2.set_state(state)
      # obj1 and obj2 will behave identically.

    Example (Restoring previous behavior):
      obj = MemoryBank(**kwargs)
      state = obj.get_state()
      # do more with obj
      obj.set_state(state)
      # obj will now behave the same as it did before.
    
    Note that the state does not need to contain any information that is passed
    in __init__ (e.g. the embedder, clock, imporance etc.)

    Args:
      state: The state of the memory bank.
    """

    self._memory.set_state(state)

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
