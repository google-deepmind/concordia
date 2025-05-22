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


"""An associative memory with basic retrieval methods."""

from collections.abc import Callable, Iterable, Sequence
import threading

from concordia.typing import entity_component
import numpy as np
import pandas as pd


class AssociativeMemoryBank:
  """Class that implements associative memory."""

  def __init__(
      self,
      sentence_embedder: Callable[[str], np.ndarray] | None = None,
  ):
    """Constructor.

    Args:
      sentence_embedder: text embedding model, if None then skip setting the
        embedder on initialization of the object. It still must be set before
        calling `add` or `retrieve` methods.
    """
    self._memory_bank_lock = threading.Lock()
    self._embedder = sentence_embedder

    self._memory_bank = pd.DataFrame(columns=['text', 'embedding'])
    self._stored_hashes = set()

  def get_state(self) -> entity_component.ComponentState:
    """Converts the AssociativeMemory to a dictionary."""

    with self._memory_bank_lock:
      output = {
          'stored_hashes': list(self._stored_hashes),
          'memory_bank': self._memory_bank.to_json(),
      }
    return output

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the AssociativeMemory from a dictionary."""

    with self._memory_bank_lock:
      self._stored_hashes = set(state['stored_hashes'])
      self._memory_bank = pd.read_json(state['memory_bank'])

  def add(
      self,
      text: str,
  ) -> None:
    """Adds nonduplicated entries (time, text, tags, importance) to the memory.

    Args:
      text: what goes into the memory
    """
    if not self._embedder:
      raise ValueError('Embedder must be set before calling `add` method.')

    # Remove all newline characters from memories.
    text = text.replace('\n', ' ')

    contents = {
        'text': text,
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
  ) -> None:
    """Adds the texts to the memory.

    Args:
      texts: list of strings to add to the memory
    """
    for text in texts:
      self.add(text)

  def get_data_frame(self) -> pd.DataFrame:
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

  def _pd_to_text(
      self,
      data: pd.DataFrame,
  ) -> Sequence[str]:
    """Formats a dataframe into list of strings.

    Args:
      data: the dataframe to process

    Returns:
      A list of strings, one for each memory
    """
    if data.empty:
      return []
    output = data['text']

    return output.tolist()

  def retrieve_associative(
      self,
      query: str,
      k: int = 1,
  ) -> Sequence[str]:
    """Retrieve memories associatively.

    Args:
      query: a string to use for retrieval
      k: how many memories to retrieve

    Returns:
      List of strings corresponding to memories, sorted by cosine similarity
    """
    if not self._embedder:
      raise ValueError('Embedder must be set before calling the '
                       '`retrieve_associative` method.')

    if k <= 0:
      raise ValueError('Limit must be positive.')

    query_embedding = self._embedder(query)

    data = self._get_top_k_cosine(query_embedding, k)

    return self._pd_to_text(data)

  def scan(self, selector_fn: Callable[[str], bool]):
    """Retrieve memories that match the selector function.

    Args:
      selector_fn: a function that takes a string and returns a boolean
        indicating whether the string matches the selector

    Returns:
      List of strings corresponding to memories, sorted by recency
    """
    with self._memory_bank_lock:
      if self._memory_bank.empty:
        return []
      is_selected = self._memory_bank['text'].apply(selector_fn)
      data = self._memory_bank[is_selected]
    return self._pd_to_text(data)

  def retrieve_recent(
      self,
      k: int = 1,
  ) -> Sequence[str]:
    """Retrieve memories by recency.

    Args:
      k: number of entries to retrieve

    Returns:
      List of strings corresponding to memories, sorted by recency
    """
    if k <= 0:
      raise ValueError('Limit must be positive.')

    with self._memory_bank_lock:
      if self._memory_bank.empty:
        return []
      return self._pd_to_text(self._memory_bank.iloc[-k:])

  def __len__(self):
    """Returns the number of entries in the memory bank.

    Since memories cannot be deleted, the length cannot decrease, and can be
    used to check if the contents of the memory bank have changed.
    """
    with self._memory_bank_lock:
      return len(self._memory_bank)

  def get_all_memories_as_text(
      self,
  ) -> Sequence[str]:
    """Returns all memories in the memory bank as a sequence of strings."""
    memories_data_frame = self.get_data_frame()
    texts = self._pd_to_text(memories_data_frame)
    return texts

  def set_embedder(self, embedder: Callable[[str], np.ndarray]):
    """Sets the embedder for the memory bank."""
    self._embedder = embedder
