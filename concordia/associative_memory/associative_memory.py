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


"""An associative memory similar to the one in the following paper.

Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P. and Bernstein,
M.S., 2023. Generative agents: Interactive simulacra of human behavior. arXiv
preprint arXiv:2304.03442.
"""

from collections.abc import Callable, Iterable
import datetime
import threading

from concordia.associative_memory import importance_function
import numpy as np
import pandas as pd


def _check_date_in_range(timestamp: datetime.datetime) -> None:
  if timestamp < pd.Timestamp.min:
    min_date = pd.Timestamp.min
    raise ValueError(f'timestamp {timestamp} < pd.Timestamp.min {min_date}')
  if timestamp > pd.Timestamp.max:
    max_date = pd.Timestamp.max
    raise ValueError(f'timestamp {timestamp} > pd.Timestamp.max {max_date}')


class AssociativeMemory:
  """Class that implements associative memory."""

  def __init__(
      self,
      sentence_embedder: Callable[[str], np.ndarray],
      importance: Callable[[str], float] | None = None,
      clock: Callable[[], datetime.datetime] = datetime.datetime.now,
      clock_step_size: datetime.timedelta | None = None,
  ):
    """Constructor.

    Args:
      sentence_embedder: text embedding model
      importance: maps a sentence into [0, 1] scale of importance, if None then
        use a constant importance model that sets all memories to importance 1.0
      clock: a callable to get time when adding memories
      clock_step_size: sets the step size of the clock. If None, assumes precise
        time
    """
    self._memory_bank_lock = threading.Lock()
    self._embedder = sentence_embedder
    self._importance = (
        importance or importance_function.ConstantImportanceModel().importance)

    self._memory_bank = pd.DataFrame(
        columns=['text', 'time', 'tags', 'embedding', 'importance']
    )
    self._clock_now = clock
    self._interval = clock_step_size
    self._stored_hashes = set()

  def add(
      self,
      text: str,
      *,
      timestamp: datetime.datetime | None = None,
      tags: Iterable[str] = (),
      importance: float | None = None,
  ):
    """Adds nonduplicated entries (time, text, tags, importance) to the memory.

    Args:
      text: what goes into the memory
      timestamp: the time of the memory
      tags: optional tags
      importance: optionally set the importance of the memory.
    """
    if importance is None:
      importance = self._importance(text)

    if timestamp is None:
      timestamp = self._clock_now()

    _check_date_in_range(timestamp)

    # Remove all newline characters from memories.
    text = text.replace('\n', ' ')

    contents = {
        'text': text,
        'time': timestamp,
        'tags': tuple(tags),
        'importance': importance,
    }
    hashed_contents = hash(tuple(contents.values()))
    derived = {'embedding': self._embedder(text)}
    new_df = pd.Series(contents | derived).to_frame().T.infer_objects()

    with self._memory_bank_lock:
      if hashed_contents in self._stored_hashes:
        return
      self._memory_bank = pd.concat(
          [self._memory_bank, new_df], ignore_index=True
      )
      self._stored_hashes.add(hashed_contents)

  def extend(
      self,
      texts: Iterable[str],
      **kwargs,
  ):
    """Adds the texts to the memory.

    Args:
      texts: list of strings to add to the memory
      **kwargs: arguments to pass on to .add
    """
    for text in texts:
      self.add(text, **kwargs)

  def get_data_frame(self):
    with self._memory_bank_lock:
      return self._memory_bank.copy()

  def _get_top_k_cosine(self, x: np.ndarray, k: int):
    """Returns the top k most cosine similar rows to an input vector x.

    Args:
      x: The input vector.
      k: The number of rows to return.

    Returns:
      Rows, sorted by cosine similarity in descending order.
    """
    with self._memory_bank_lock:
      cosine_similarities = self._memory_bank['embedding'].apply(
          lambda y: np.dot(x, y)
      )

      # Sort the cosine similarities in descending order.
      cosine_similarities.sort_values(ascending=False, inplace=True)

      # Return the top k rows.
      return self._memory_bank.iloc[cosine_similarities.head(k).index]

  def _get_top_k_similar_rows(
      self, x, k: int, use_recency: bool = True, use_importance: bool = True
  ):
    """Returns the top k most similar rows to an input vector x.

    Args:
      x: The input vector.
      k: The number of rows to return.
      use_recency: if true then weight similarity by recency
      use_importance: if true then weight similarity by importance

    Returns:
      Rows, sorted by cosine similarity in descending order.
    """
    with self._memory_bank_lock:
      cosine_similarities = self._memory_bank['embedding'].apply(
          lambda y: np.dot(x, y)
      )

      similarity_score = cosine_similarities

      if use_recency:
        max_time = self._memory_bank['time'].max()
        discounted_time = self._memory_bank['time'].apply(
            lambda y: 0.99 ** ((max_time - y) / datetime.timedelta(minutes=1))
        )
        similarity_score += discounted_time

      if use_importance:
        importance = self._memory_bank['importance']
        similarity_score += importance

      # Sort the similarities in descending order.
      similarity_score.sort_values(ascending=False, inplace=True)

      # Return the top k rows.
      return self._memory_bank.iloc[similarity_score.head(k).index]

  def _get_k_recent(self, k: int):
    with self._memory_bank_lock:
      recency = self._memory_bank['time'].sort_values(ascending=False)
      return self._memory_bank.iloc[recency.head(k).index]

  def _pd_to_text(
      self,
      data: pd.DataFrame,
      add_time: bool = False,
      sort_by_time: bool = True,
  ):
    """Formats a dataframe into list of strings.

    Args:
      data: the dataframe to process
      add_time: whether to add time
      sort_by_time: whether to sort by time

    Returns:
      A list of strings, one for each memory
    """
    if sort_by_time:
      data = data.sort_values('time', ascending=True)

    if add_time and not data.empty:
      if self._interval:
        this_time = data['time']
        next_time = data['time'] + self._interval

        interval = this_time.dt.strftime(
            '%d %b %Y [%H:%M:%S  '
        ) + next_time.dt.strftime('- %H:%M:%S]: ')
        output = interval + data['text']
      else:
        output = data['time'].dt.strftime('[%d %b %Y %H:%M:%S] ') + data['text']
    else:
      output = data['text']

    return output.tolist()

  def retrieve_associative(
      self,
      query: str,
      k: int = 1,
      use_recency: bool = True,
      use_importance: bool = True,
      add_time: bool = True,
      sort_by_time: bool = True,
  ):
    """Retrieve memories associatively.

    Args:
      query: a string to use for retrieval
      k: how many memories to retrieve
      use_recency: whether to use timestamps to weight by recency or not
      use_importance: whether to use importance for retrieval
      add_time: whether to add time stamp to the output
      sort_by_time: whether to sort the result by time

    Returns:
      List of strings corresponding to memories
    """
    query_embedding = self._embedder(query)

    data = self._get_top_k_similar_rows(
        query_embedding,
        k,
        use_recency=use_recency,
        use_importance=use_importance,
    )

    return self._pd_to_text(data, add_time=add_time, sort_by_time=sort_by_time)

  def retrieve_by_regex(
      self,
      regex: str,
      add_time: bool = True,
      sort_by_time: bool = True,
  ):
    """Retrieve memories matching a regex.

    Args:
      regex: a regex to match
      add_time: whether to add time stamp to the output
      sort_by_time: whether to sort the result by time

    Returns:
      List of strings corresponding to memories
    """
    with self._memory_bank_lock:
      data = self._memory_bank[self._memory_bank['text'].str.contains(regex)]

    return self._pd_to_text(data, add_time=add_time, sort_by_time=sort_by_time)

  def retrieve_time_interval(
      self,
      time_from: datetime.datetime,
      time_until: datetime.datetime,
      add_time: bool = False,
  ):
    """Retrieve memories within a time interval.

    Args:
      time_from: the start time of the interval
      time_until: the end time of the interval
      add_time: whether to add time stamp to the output

    Returns:
      List of strings corresponding to memories
    """

    with self._memory_bank_lock:
      data = self._memory_bank[
          (self._memory_bank['time'] >= time_from)
          & (self._memory_bank['time'] <= time_until)
      ]

    return self._pd_to_text(data, add_time=add_time, sort_by_time=True)

  def retrieve_recent(
      self,
      k: int = 1,
      add_time: bool = False,
  ):
    """Retrieve memories by recency.

    Args:
      k: number of entries to retrieve
      add_time: whether to add time stamp to the output

    Returns:
      List of strings corresponding to memories
    """
    data = self._get_k_recent(k)

    return self._pd_to_text(data, add_time=add_time, sort_by_time=True)

  def retrieve_recent_with_importance(
      self,
      k: int = 1,
      add_time: bool = False,
  ):
    """Retrieve memories by recency and return importance alongside.

    Args:
      k: number of entries to retrieve
      add_time: whether to add time stamp to the output

    Returns:
      List of strings corresponding to memories
    """
    data = self._get_k_recent(k)

    return (
        self._pd_to_text(data, add_time=add_time, sort_by_time=True),
        list(data['importance']),
    )

  def __len__(self):
    """Returns the number of entries in the memory bank.

    Since memories cannot be deleted, the length cannot decrease, and can be
    used to check if the contents of the memory bank have changed.
    """
    with self._memory_bank_lock:
      return len(self._memory_bank)
