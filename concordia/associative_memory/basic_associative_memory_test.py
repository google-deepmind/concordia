# Copyright 2026 DeepMind Technologies Limited.
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

"""Tests for basic associative memory."""

from collections.abc import Callable, Mapping, Sequence

from absl.testing import absltest
from concordia.associative_memory import basic_associative_memory
import numpy as np


def _make_embedder(
    embeddings: Mapping[str, Sequence[float]],
) -> Callable[[str], np.ndarray]:
  """Returns a deterministic embedder backed by a local mapping."""

  def embed(text: str) -> np.ndarray:
    return np.asarray(embeddings[text], dtype=np.float64)

  return embed


class AssociativeMemoryBankTest(absltest.TestCase):

  def test_add_suppresses_duplicates_by_default(self):
    embedder = _make_embedder({'same memory': [1.0, 0.0]})
    memory = basic_associative_memory.AssociativeMemoryBank(embedder)

    memory.add('same memory')
    memory.add('same memory')

    self.assertLen(memory, 1)
    self.assertEqual(memory.get_all_memories_as_text(), ['same memory'])

  def test_add_retains_duplicates_when_allowed(self):
    embedder = _make_embedder({'same memory': [1.0, 0.0]})
    memory = basic_associative_memory.AssociativeMemoryBank(
        embedder, allow_duplicates=True
    )

    memory.add('same memory')
    memory.add('same memory')

    self.assertLen(memory, 2)
    self.assertEqual(
        memory.get_all_memories_as_text(), ['same memory', 'same memory']
    )

  def test_add_normalizes_newlines(self):
    embedder = _make_embedder({'line one line two line three': [1.0, 0.0]})
    memory = basic_associative_memory.AssociativeMemoryBank(embedder)

    memory.add('line one\nline two\nline three')

    self.assertEqual(
        memory.get_all_memories_as_text(),
        ['line one line two line three'],
    )

  def test_retrieve_associative_orders_by_similarity(self):
    embedder = _make_embedder({
        'find northern memories': [1.0, 0.0],
        'north market': [0.9, 0.1],
        'east library': [0.2, 0.8],
        'south garden': [0.7, 0.3],
    })
    memory = basic_associative_memory.AssociativeMemoryBank(embedder)
    memory.extend(['east library', 'north market', 'south garden'])

    self.assertEqual(
        memory.retrieve_associative('find northern memories', k=2),
        ['north market', 'south garden'],
    )

  def test_retrieve_recent_returns_most_recent_window_in_insertion_order(self):
    embedder = _make_embedder({
        'first': [1.0, 0.0],
        'second': [0.0, 1.0],
        'third': [1.0, 1.0],
    })
    memory = basic_associative_memory.AssociativeMemoryBank(embedder)
    memory.extend(['first', 'second', 'third'])

    self.assertEqual(memory.retrieve_recent(k=2), ['second', 'third'])

  def test_retrieve_validates_limit(self):
    embedder = _make_embedder({'query': [1.0, 0.0]})
    memory = basic_associative_memory.AssociativeMemoryBank(embedder)

    with self.assertRaisesRegex(ValueError, 'Limit must be positive.'):
      memory.retrieve_recent(k=0)

    with self.assertRaisesRegex(ValueError, 'Limit must be positive.'):
      memory.retrieve_associative('query', k=0)

  def test_get_state_and_set_state_preserve_pending_memories(self):
    embedder = _make_embedder({
        'first': [1.0, 0.0],
        'second': [0.0, 1.0],
    })
    memory = basic_associative_memory.AssociativeMemoryBank(embedder)
    memory.extend(['first', 'second'])

    restored = basic_associative_memory.AssociativeMemoryBank(embedder)
    restored.set_state(memory.get_state())

    self.assertLen(restored, 2)
    self.assertEqual(restored.get_all_memories_as_text(), ['first', 'second'])
    self.assertEqual(restored.retrieve_recent(k=2), ['first', 'second'])
    restored.add('first')
    self.assertEqual(restored.get_all_memories_as_text(), ['first', 'second'])


if __name__ == '__main__':
  absltest.main()
