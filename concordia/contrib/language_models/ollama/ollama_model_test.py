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

"""Tests for OllamaLanguageModel.sample_choice with a fake client."""

import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.contrib.language_models.ollama import ollama_model
from concordia.language_model import language_model


def _model_with_responses(*payloads):
  """Returns a model whose client returns each JSON payload in turn."""
  with mock.patch.object(ollama_model.ollama, 'Client') as client_cls:
    model = ollama_model.OllamaLanguageModel('fake-model')
  client_cls.return_value.generate.side_effect = [
      {'response': json.dumps(payload)} for payload in payloads
  ]
  return model


class SampleChoiceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('choice_key', {'choice': ' (b) '}),
      ('first_value_when_choice_missing', {'answer': 'b)'}),
      ('top_level_string', '(b)'),
  )
  def test_extracts_choice(self, payload):
    model = _model_with_responses(payload)
    idx, response, _ = model.sample_choice('Pick one.', ['a', 'b'])
    self.assertEqual((idx, response), (1, 'b'))

  @parameterized.named_parameters(
      ('integer_choice', {'choice': 1}),
      ('null_choice', {'choice': None}),
      ('list_response', ['b']),
      ('number_response', 2),
      ('empty_object', {}),
  )
  def test_retries_after_unusable_response(self, payload):
    model = _model_with_responses(payload, {'choice': '(a)'})
    idx, response, _ = model.sample_choice('Pick one.', ['a', 'b'])
    self.assertEqual((idx, response), (0, 'a'))

  def test_raises_invalid_response_after_max_attempts(self):
    model = _model_with_responses(
        *[['b']] * ollama_model._MAX_MULTIPLE_CHOICE_ATTEMPTS
    )
    with self.assertRaises(language_model.InvalidResponseError):
      model.sample_choice('Pick one.', ['a', 'b'])


if __name__ == '__main__':
  absltest.main()
