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

"""Tests for the no-language-model implementations."""

import random

from absl.testing import absltest
from concordia.language_model import no_language_model


class RandomChoiceLanguageModelTest(absltest.TestCase):

  def test_seeded_call_does_not_change_global_random_state(self):
    random.seed(9182)
    state_before = random.getstate()

    no_language_model.RandomChoiceLanguageModel().sample_choice(
        'prompt', ['a', 'b', 'c'], seed=7
    )

    self.assertEqual(random.getstate(), state_before)

  def test_same_seed_and_responses_produce_same_result(self):
    model = no_language_model.RandomChoiceLanguageModel()

    first = model.sample_choice('prompt', ['a', 'b', 'c'], seed=7)
    second = model.sample_choice('prompt', ['a', 'b', 'c'], seed=7)

    self.assertEqual(first, second)

  def test_seed_seven_preserves_selected_index_and_response(self):
    result = no_language_model.RandomChoiceLanguageModel().sample_choice(
        'prompt', ['a', 'b', 'c'], seed=7
    )

    self.assertEqual(result, (1, 'b', {}))

  def test_unseeded_call_consumes_global_random_sequence(self):
    random.seed(9182)
    expected_index = random.randint(0, 2)
    expected_state = random.getstate()

    random.seed(9182)
    result = no_language_model.RandomChoiceLanguageModel().sample_choice(
        'prompt', ['a', 'b', 'c']
    )

    self.assertEqual(
        result, (expected_index, ['a', 'b', 'c'][expected_index], {})
    )
    self.assertEqual(random.getstate(), expected_state)

  def test_empty_responses_do_not_change_global_random_state(self):
    random.seed(9182)
    state_before = random.getstate()

    result = no_language_model.RandomChoiceLanguageModel().sample_choice(
        'prompt', [], seed=7
    )

    self.assertEqual(result, (0, '', {}))
    self.assertEqual(random.getstate(), state_before)


if __name__ == '__main__':
  absltest.main()
