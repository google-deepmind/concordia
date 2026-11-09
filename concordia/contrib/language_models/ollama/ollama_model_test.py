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

"""Tests for the Ollama language model wrapper (no server needed)."""

from unittest import mock

from absl.testing import absltest
from concordia.contrib.language_models.ollama import ollama_model


def _model(**kwargs) -> tuple[ollama_model.OllamaLanguageModel, mock.Mock]:
  model, client_cls = _model_and_class(**kwargs)
  return model, client_cls.return_value


def _model_and_class(
    **kwargs,
) -> tuple[ollama_model.OllamaLanguageModel, mock.Mock]:
  with mock.patch.object(ollama_model.ollama, 'Client') as client_cls:
    model = ollama_model.OllamaLanguageModel('qwen3:8b', **kwargs)
  return model, client_cls


class OllamaLanguageModelTest(absltest.TestCase):

  def test_think_is_left_to_the_model_by_default(self):
    model, client = _model()
    client.generate.return_value = {'response': 'a reply'}
    self.assertEqual(model.sample_text('Q?', terminators=('\n',)), 'a reply')
    _, kwargs = client.generate.call_args
    self.assertIsNone(kwargs['think'])  # The client then omits it.
    self.assertEqual(kwargs['options']['stop'], ['\n'])

  def test_think_false_is_sent_with_text_and_choice_requests(self):
    model, client = _model(think=False)
    client.generate.return_value = {'response': '{"choice": "b"}'}
    model.sample_text('Q?')
    self.assertIs(client.generate.call_args.kwargs['think'], False)
    idx, answer, _ = model.sample_choice('Pick one.', ['a', 'b'])
    self.assertEqual((idx, answer), (1, 'b'))
    self.assertIs(client.generate.call_args.kwargs['think'], False)

  def test_a_thinking_model_stops_its_answer_not_its_thinking(self):
    model, client = _model(think=True)
    client.generate.return_value = {
        'response': '"everyone else build"\nAnd more.',
        'thinking': 'Ash is not Bea.\nSo...',
    }
    reply = model.sample_text('Which clause?', terminators=('\n',))
    self.assertEqual(reply, '"everyone else build"')
    _, kwargs = client.generate.call_args
    self.assertEqual(kwargs['options']['stop'], [])
    self.assertIs(kwargs['think'], True)

  def test_max_tokens_bounds_the_reply_unless_thinking_is_kept_apart(self):
    for think in (None, False):
      model, client = _model(think=think)
      client.generate.return_value = {'response': 'a reply'}
      model.sample_text('Q?', max_tokens=64)
      options = client.generate.call_args.kwargs['options']
      self.assertEqual(options['num_predict'], 64, think)
    model, client = _model(think=True)
    client.generate.return_value = {'response': 'a reply', 'thinking': '...'}
    model.sample_text('Q?', max_tokens=64)
    self.assertNotIn('num_predict', client.generate.call_args.kwargs['options'])

  def test_timeout_reaches_the_client_only_when_set(self):
    _, client_cls = _model_and_class()
    client_cls.assert_called_once_with()
    _, client_cls = _model_and_class(timeout=120.0)
    client_cls.assert_called_once_with(timeout=120.0)


if __name__ == '__main__':
  absltest.main()
