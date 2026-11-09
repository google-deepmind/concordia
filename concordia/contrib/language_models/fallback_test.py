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

"""Tests for the fallback language model wrapper."""

from collections.abc import Sequence

from absl.testing import absltest
from concordia.contrib.language_models import fallback
from concordia.language_model import no_language_model


class _Fixed(no_language_model.NoLanguageModel):
  """Answers with fixed text and choice, or raises; counts its calls."""

  def __init__(self, text: str, choice: int, error: Exception | None = None):
    super().__init__()
    self.text, self.choice, self.error = text, choice, error
    self.calls = 0

  def sample_text(self, prompt: str, **kwargs) -> str:
    self.calls += 1
    if self.error:
      raise self.error
    return self.text

  def sample_choice(self, prompt: str, responses: Sequence[str], **kwargs):
    self.calls += 1
    if self.error:
      raise self.error
    return self.choice, responses[self.choice], {}


class FallbackLanguageModelTest(absltest.TestCase):

  def test_the_primary_answers_while_it_can(self):
    primary, spare = _Fixed('primary', 1), _Fixed('spare', 0)
    model = fallback.FallbackLanguageModel(primary, spare)
    self.assertEqual(model.sample_text('Q?'), 'primary')
    self.assertEqual(model.sample_choice('Pick.', ['a', 'b'])[1], 'b')
    self.assertEqual((model.primary_calls, model.fallbacks), (2, 0))
    self.assertEqual(spare.calls, 0)
    self.assertIsNone(model.last_error)

  def test_an_error_sends_the_call_to_the_fallback(self):
    primary = _Fixed('primary', 1, error=ConnectionError('503'))
    model = fallback.FallbackLanguageModel(primary, _Fixed('spare', 0))
    self.assertEqual(model.sample_text('Q?'), 'spare')
    self.assertEqual(model.sample_choice('Pick.', ['a', 'b'])[1], 'a')
    self.assertEqual(model.fallbacks, 2)
    self.assertEqual(model.last_error, 'ConnectionError: 503')

  def test_errors_outside_the_given_kinds_propagate(self):
    primary = _Fixed('primary', 1, error=ValueError('bad prompt'))
    model = fallback.FallbackLanguageModel(
        primary, _Fixed('spare', 0), errors=(ConnectionError,)
    )
    with self.assertRaises(ValueError):
      model.sample_text('Q?')

  def test_past_the_budget_the_fallback_answers(self):
    primary, spare = _Fixed('primary', 1), _Fixed('spare', 0)
    model = fallback.FallbackLanguageModel(primary, spare, max_primary_calls=2)
    answers = [model.sample_text('Q?') for _ in range(4)]
    self.assertEqual(answers, ['primary', 'primary', 'spare', 'spare'])
    self.assertEqual((primary.calls, spare.calls), (2, 2))
    self.assertEqual((model.primary_calls, model.fallbacks), (2, 2))
    self.assertIsNone(model.last_error)


if __name__ == '__main__':
  absltest.main()
