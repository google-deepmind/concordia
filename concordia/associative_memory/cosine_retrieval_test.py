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

"""Memory retrieval ranks directions independently of embedding magnitude."""

from absl.testing import absltest
from absl.testing import parameterized
from concordia.associative_memory import basic_associative_memory
import numpy as np


def make_bank(vectors):
  bank = basic_associative_memory.AssociativeMemoryBank(
      lambda text: np.asarray(vectors[text], dtype=float)
  )
  bank.extend(key for key in vectors if key != 'query')
  return bank


class CosineRetrievalTest(parameterized.TestCase):

  def test_exact_match_beats_a_larger_off_axis_embedding(self):
    bank = make_bank(
        {'query': [1.0, 0.0], 'exact': [1.0, 0.0], 'large': [100.0, 100.0]}
    )
    self.assertEqual(bank.retrieve_associative('query', k=1), ['exact'])

  @parameterized.parameters(
      (1.0, 1.0, 1.0), (100.0, 0.01, 20.0), (0.1, 30.0, 0.001)
  )
  def test_independent_positive_rescaling_does_not_change_ranking(
      self, a, b, c
  ):
    vectors = {
        'query': np.array([1.0, 2.0]) * a,
        'near': np.array([1.0, 2.0]) * b,
        'partial': np.array([2.0, 1.0]) * c,
        'opposite': np.array([-1.0, -2.0]) * 1000,
    }
    self.assertEqual(
        make_bank(vectors).retrieve_associative('query', k=3),
        ['near', 'partial', 'opposite'],
    )

  def test_random_embeddings_match_explicit_normalized_dot_products(self):
    rng = np.random.default_rng(3)
    query = rng.normal(size=7)
    values = rng.normal(size=(20, 7)) * np.geomspace(0.001, 1000, 20)[:, None]
    vectors = {'query': query} | {str(i): x for i, x in enumerate(values)}
    scores = (values / np.linalg.norm(values, axis=1, keepdims=True)) @ (
        query / np.linalg.norm(query)
    )
    expected = [str(i) for i in np.argsort(-scores)[:8]]
    self.assertEqual(
        make_bank(vectors).retrieve_associative('query', k=8), expected
    )

  def test_restore_uses_stored_embeddings_without_mutating_them(self):
    vectors = {
        'query': [1.0, 0.0],
        'exact': [1.0, 0.0],
        'large': [100.0, 100.0],
    }
    original = make_bank(vectors)
    state = original.get_state()
    restored = basic_associative_memory.AssociativeMemoryBank(
        lambda text: np.asarray(vectors[text])
    )
    restored.set_state(state)
    before = restored.get_state()
    self.assertEqual(restored.retrieve_associative('query'), ['exact'])
    self.assertEqual(restored.get_state(), before)
    self.assertEqual(restored.retrieve_recent(2), ['exact', 'large'])

  def test_zero_vectors_have_zero_similarity_without_warnings(self):
    bank = make_bank({
        'query': [1.0, 0.0],
        'opposite': [-1.0, 0.0],
        'zero': [0.0, 0.0],
        'same': [2.0, 0.0],
    })
    with np.errstate(all='raise'):
      self.assertEqual(
          bank.retrieve_associative('query', k=3), ['same', 'zero', 'opposite']
      )
    zero_query = make_bank(
        {'query': [0.0, 0.0], 'a': [1.0, 0.0], 'b': [3.0, 2.0]}
    )
    with np.errstate(all='raise'):
      self.assertCountEqual(
          zero_query.retrieve_associative('query', k=2), ['a', 'b']
      )

  def test_empty_bank_and_unit_vectors_keep_existing_behavior(self):
    vectors = {
        'query': [1.0, 0.0],
        'north': [1.0, 0.0],
        'east': [0.0, 1.0],
        'south': [-1.0, 0.0],
    }
    bank = basic_associative_memory.AssociativeMemoryBank(
        lambda text: np.asarray(vectors[text])
    )
    self.assertEqual(bank.retrieve_associative('query'), [])
    bank.extend(['east', 'south', 'north'])
    self.assertEqual(
        bank.retrieve_associative('query', k=4), ['north', 'east', 'south']
    )


if __name__ == '__main__':
  absltest.main()
