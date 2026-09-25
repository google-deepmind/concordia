# Copyright 2024 DeepMind Technologies Limited.
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

"""Choice-contract regression tests; no live Ollama calls."""

import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.contrib.language_models.ollama import ollama_model
from concordia.language_model import language_model


class OllamaChoiceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    client_patch = mock.patch.object(ollama_model.ollama, 'Client')
    self.client = client_patch.start().return_value
    self.addCleanup(client_patch.stop)
    self.model = ollama_model.OllamaLanguageModel('test-model')

  def test_schema_uses_response_keys_not_labels_in_prompt(self):
    self.client.generate.return_value = {'response': '{"choice":"b"}'}
    result = self.model.sample_choice('(a) ACCEPT (b) DECLINE', ['a', 'b'])
    self.assertEqual(result[:2], (1, 'b'))
    args = self.client.generate.call_args.kwargs
    self.assertEqual(args['format']['properties']['choice']['enum'], ['a', 'b'])
    self.assertEqual(args['format']['required'], ['choice'])
    self.assertIn('["a", "b"]', args['prompt'])
    self.client.generate.assert_called_once()

  @parameterized.parameters('ACCEPT', 'one quiet song', 'қабылдау')
  def test_arbitrary_exact_response_strings(self, value):
    self.client.generate.return_value = {
        'response': json.dumps({'choice': value})
    }
    self.assertEqual(
        self.model.sample_choice('choose', ['other', value])[:2], (1, value)
    )

  @parameterized.parameters('a)', '(a)', 'Answer: (a)')
  def test_legacy_decorated_letters_use_existing_extractor(self, value):
    self.client.generate.return_value = {
        'response': json.dumps({'choice': value})
    }
    self.assertEqual(
        self.model.sample_choice('choose', ['a', 'b'])[:2], (0, 'a')
    )

  @parameterized.parameters(
      'null',
      '[]',
      '"a"',
      '{}',
      '{"choice":1}',
      '{"choice":[]}',
      '{"choice":"ACCEPT"}',
      'not json',
  )
  def test_invalid_response_retries_without_guessing(self, invalid):
    self.client.generate.side_effect = [
        {'response': invalid},
        {'response': '{"choice":"b"}'},
    ]
    self.assertEqual(
        self.model.sample_choice('(a) ACCEPT (b) DECLINE', ['a', 'b'])[:2],
        (1, 'b'),
    )
    self.assertEqual(self.client.generate.call_count, 2)

  def test_exhaustion_keeps_failure_instead_of_defaulting(self):
    self.client.generate.return_value = {'response': '{"choice":"ACCEPT"}'}
    with self.assertRaises(language_model.InvalidResponseError):
      self.model.sample_choice('(a) ACCEPT (b) DECLINE', ['a', 'b'])
    self.assertEqual(self.client.generate.call_count, 20)

  @parameterized.parameters(None, False, True)
  def test_optional_thinking_control_applies_to_text_and_choice(self, think):
    model = ollama_model.OllamaLanguageModel('test-model', think=think)
    self.client.generate.return_value = {
        'response': 'Hello',
        'thinking': 'private reasoning',
    }
    self.assertEqual(model.sample_text('speak'), 'Hello')
    text_args = self.client.generate.call_args.kwargs
    self.client.generate.return_value = {
        'response': '{"choice":"a"}',
        'thinking': 'private reasoning',
    }
    self.assertEqual(model.sample_choice('choose', ['a', 'b'])[:2], (0, 'a'))
    choice_args = self.client.generate.call_args.kwargs
    for args in (text_args, choice_args):
      if think is None:
        self.assertNotIn('think', args)
      else:
        self.assertIs(args['think'], think)

  @parameterized.parameters({'value': 1}, {'value': 'false'}, {'value': []})
  def test_invalid_thinking_control_is_rejected(self, value):
    with self.assertRaisesRegex(ValueError, 'think must'):
      ollama_model.OllamaLanguageModel('test-model', think=value)

  def test_empty_choices_fail_before_model_request(self):
    with self.assertRaises(ValueError):
      self.model.sample_choice('choose', [])
    self.client.generate.assert_not_called()


if __name__ == '__main__':
  absltest.main()
