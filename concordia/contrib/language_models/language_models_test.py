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

"""Tests for the registry and choice contracts, without provider calls."""

import importlib.util
import json
import sys
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.contrib import language_models
from concordia.language_model import language_model
import numpy as np


def _adapter(module, name, sdk_modules):
  # Import real adapter code with inert SDKs, then bypass client/model loading.
  sdk_modules = {
      '.'.join(parts[:i])
      for sdk in sdk_modules.split()
      for parts in [sdk.split('.')]
      for i in range(1, len(parts) + 1)
  }
  with mock.patch.dict(
      sys.modules, {sdk: mock.MagicMock() for sdk in sdk_modules}
  ):
    cls = getattr(
        importlib.import_module(f'concordia.contrib.language_models.{module}'),
        name,
    )
  model = object.__new__(cls)
  model._measurements = None
  model._system_message = ''
  model._verbosity = 'low'
  return model


_PARSING_ADAPTERS = (
    ('amazon.amazon_bedrock_model', 'AmazonBedrockLanguageModel', 'boto3'),
    ('google.gemini_model', 'GeminiModel', 'google.genai google.genai.types'),
    (
        'google.gemini_model_multimodal',
        'GeminiModelVision',
        'google.genai google.genai.types',
    ),
    (
        'google.google_cloud_custom_model',
        'VertexAI',
        'google.cloud google.cloud.aiplatform',
    ),
    (
        'langchain.langchain_ollama_model',
        'LangchainOllamaLanguageModel',
        'langchain_community langchain_community.llms',
    ),
)


class ChoiceAdaptersTest(parameterized.TestCase):

  @parameterized.product(
      adapter=_PARSING_ADAPTERS, label=('b', 'aa', 'ba', 'aaa')
  )
  def test_parser_adapters(self, adapter, label):
    model = _adapter(*adapter)
    with mock.patch.object(model, 'sample_text', return_value=f'({label})'):
      self.assertEqual(
          model.sample_choice('Choose?', ['a', label])[:2], (1, label)
      )

  @parameterized.product(
      adapter=_PARSING_ADAPTERS,
      invalid=(
          '(a) or (aa)',
          'aaah)',
          'I have a preference.',
          'a a',
      ),
  )
  def test_parser_adapters_retry_without_changing_temperature(
      self, adapter, invalid
  ):
    model = _adapter(*adapter)
    with mock.patch.object(
        model, 'sample_text', side_effect=[invalid, '(aa)']
    ) as sample:
      self.assertEqual(
          model.sample_choice('Choose?', ['a', 'aa'])[:2], (1, 'aa')
      )
    self.assertEqual(sample.call_count, 2)
    self.assertEqual(
        [c.kwargs['temperature'] for c in sample.call_args_list], [0.0, 0.0]
    )

  def test_ollama_json_choice(self):
    model = _adapter('ollama.ollama_model', 'OllamaLanguageModel', 'ollama')
    model._client = mock.Mock()
    model._model_name = 'fixture'
    model._client.generate.side_effect = [
        {'response': json.dumps({'choice': '(a) or (aa)'})},
        {'response': json.dumps({'choice': 'aa'})},
    ]
    self.assertEqual(model.sample_choice('Choose?', ['a', 'aa'])[:2], (1, 'aa'))
    self.assertEqual(model._client.generate.call_count, 2)

  @parameterized.product(
      completion=(False, True), label=('b', 'aa', 'aaa', 'aaaa')
  )
  def test_mistral_choice_and_budget(self, completion, label):
    model = _adapter(
        'mistral.mistral_model',
        'MistralLanguageModel',
        'mistralai mistralai.models',
    )
    model._completion_for_choice = completion
    method = '_complete_text' if completion else '_chat_text'
    with mock.patch.object(
        model, method, side_effect=['(a) or (' + label + ')', label]
    ) as sample:
      self.assertEqual(
          model.sample_choice('Choose?', ['a', label])[:2], (1, label)
      )
    budget = sample.call_args.kwargs['max_tokens']
    self.assertGreaterEqual(budget, len(label) + 2)
    if label == 'b':
      self.assertEqual(budget, 256 if completion else 3)

  @parameterized.parameters(
      ('openai.base_gpt_model', 'BaseGPTModel', 'openai', '_sample_text'),
      (
          'openai.gpt_model_multimodal',
          'GptVisionModel',
          'openai',
          '_chat_sample_text',
      ),
      ('groq.groq_model', 'GroqModel', 'groq', '_sample_text'),
  )
  def test_exact_match_adapters_already_support_long_labels(
      self, module, name, sdk, method
  ):
    model = _adapter(module, name, sdk)
    with mock.patch.object(
        model, method, side_effect=['a or aa', 'aa']
    ) as sample:
      self.assertEqual(
          model.sample_choice('Choose?', ['a', 'aa'])[:2], (1, 'aa')
      )
    self.assertEqual(sample.call_count, 2)

  @parameterized.parameters(('b', 1), ('aa', 16), ('aaa', 16))
  def test_gemma_generation_budget_and_decoding(self, label, budget):
    model = _adapter(
        'huggingface.pytorch_gemma_model',
        'PyTorchGemmaLanguageModel',
        'transformers',
    )
    model._device = 'cpu'
    model._tokenizer = mock.Mock()
    model._tokenizer.encode.side_effect = lambda value, **_: list(
        range(len(value))
    )
    model._tokenizer.return_value.input_ids.shape = (1, 2)
    # Decode a one-token greedy score for legacy choices, or the entire newly
    # generated sequence for longer labels. Prompt token IDs must be excluded.
    model._tokenizer.batch_decode.side_effect = lambda values, **_: [
        ''.join(
            {0: 'a', 1: 'b'}[int(i)] for i in np.asarray(values).reshape(-1)
        )
    ]
    score = mock.Mock()
    score.cpu.return_value = np.array([0.0, 1.0])
    model._model = mock.Mock()
    model._model.generate.return_value = types.SimpleNamespace(
        scores=[[score]], sequences=np.array([[99, 98] + [0] * len(label)])
    )
    self.assertEqual(
        model.sample_choice('Choose?', ['a', label])[:2], (1, label)
    )
    kwargs = model._model.generate.call_args.kwargs
    self.assertEqual(kwargs['max_new_tokens'], budget)
    if budget == 1:
      self.assertNotIn('do_sample', kwargs)
    else:
      self.assertFalse(kwargs['do_sample'])

  def test_gemma_budget_uses_tokenizer_and_rejects_ambiguous_output(self):
    model = _adapter(
        'huggingface.pytorch_gemma_model',
        'PyTorchGemmaLanguageModel',
        'transformers',
    )
    model._device = 'cpu'
    model._tokenizer = mock.Mock()
    model._tokenizer.encode.side_effect = [[0], list(range(20))]
    model._tokenizer.return_value.input_ids.shape = (1, 2)
    model._tokenizer.batch_decode.return_value = ['(a) or (aa)']
    model._model = mock.Mock()
    model._model.generate.return_value.sequences = np.array(
        [[99, 98, 0, 1, 2, 3]]
    )
    with self.assertRaises(language_model.InvalidResponseError):
      model.sample_choice('Choose?', ['a', 'aa'])
    self.assertEqual(
        model._model.generate.call_args.kwargs['max_new_tokens'], 22
    )

  @parameterized.parameters('aa)', ' aa)', '(aa)', ' (aa)')
  def test_gemma_preserves_formatted_labels_with_realistic_token_budget(
      self, continuation
  ):
    model = self._gemma_with_budgeted_continuation(continuation)
    self.assertEqual(
        model.sample_choice('Choose? Answer: (', ['a', 'aa'])[:2], (1, 'aa')
    )

  def test_gemma_rejects_a_valid_prefix_at_the_generation_limit(self):
    model = self._gemma_with_budgeted_continuation(' ' * 15 + 'aa)')
    with self.assertRaises(language_model.InvalidResponseError):
      model.sample_choice('Choose? Answer: (', ['a', 'aa'])

  @parameterized.parameters((3,), ([0, 3],))
  def test_gemma_accepts_eos_at_the_generation_limit(self, eos_ids):
    model = self._gemma_with_budgeted_continuation(' ' * 13 + 'aa\x03')
    model._model.generation_config.eos_token_id = eos_ids
    self.assertEqual(
        model.sample_choice('Choose? Answer: (', ['a', 'aa'])[:2], (1, 'aa')
    )

  def _gemma_with_budgeted_continuation(self, continuation):
    model = _adapter(
        'huggingface.pytorch_gemma_model',
        'PyTorchGemmaLanguageModel',
        'transformers',
    )
    model._device = 'cpu'
    model._tokenizer = mock.Mock()
    model._tokenizer.encode.side_effect = lambda text, **_: list(map(ord, text))
    model._tokenizer.return_value.input_ids.shape = (1, 2)
    model._tokenizer.batch_decode.side_effect = lambda tokens, **_: [
        ''.join(
            chr(int(token))
            for token in np.asarray(tokens)[0]
            if token not in (0, 3)
        )
    ]
    model._model = mock.Mock()

    def generate(_inputs, *, max_new_tokens, **_kwargs):
      emitted = list(map(ord, continuation))[:max_new_tokens]
      return types.SimpleNamespace(
          sequences=np.array([[99, 98, *emitted]]), scores=()
      )

    model._model.generate.side_effect = generate
    model._model.generation_config.eos_token_id = None
    return model

  def test_huggingface_scores_complete_choices(self):
    model = _adapter(
        'huggingface.huggingface_model',
        'HuggingFaceLanguageModel',
        'torch transformers',
    )
    with mock.patch.object(
        model, '_compute_log_probability', side_effect=[-5.0, -2.0]
    ) as score:
      self.assertEqual(
          model.sample_choice('Choose?', ['a', 'aa'])[:2], (1, 'aa')
      )
    self.assertEqual(
        score.call_args_list,
        [mock.call('Choose?', 'a'), mock.call('Choose?', 'aa')],
    )

  def test_vllm_scores_all_response_tokens(self):
    model = _adapter(
        'vllm.vllm_model',
        'VLLMLanguageModel',
        'vllm vllm.lora vllm.lora.request',
    )
    model._llm = mock.Mock()
    model._llm.get_tokenizer.return_value.encode.side_effect = (
        lambda value: list(map(ord, value))
    )
    # a wins only if both of aa's token log probabilities are included.
    model._llm.generate.return_value = [
        types.SimpleNamespace(
            prompt_logprobs=[None, {97: types.SimpleNamespace(logprob=-1.5)}]
        ),
        types.SimpleNamespace(
            prompt_logprobs=[
                None,
                {97: types.SimpleNamespace(logprob=-1.0)},
                {97: types.SimpleNamespace(logprob=-1.0)},
            ]
        ),
    ]
    self.assertEqual(model.sample_choice('?', ['a', 'aa'])[:2], (0, 'a'))
    self.assertEqual(model._llm.generate.call_args.args[0], ['?a', '?aa'])


class LanguageModelsRegistryTest(absltest.TestCase):
  """Tests that every registry entry resolves to an importable module."""

  def test_all_registry_entries_resolve_to_modules(self):
    for api_type, model_path in language_models._REGISTRY.items():
      module_path, _ = model_path.rsplit('.', 1)
      full_module = f'concordia.contrib.language_models.{module_path}'
      self.assertIsNotNone(
          importlib.util.find_spec(full_module),
          msg=(
              f'api_type {api_type!r} references module {full_module!r} which'
              ' does not exist.'
          ),
      )

  def test_google_aistudio_is_a_registered_api_type(self):
    # `google_aistudio` is the default api_type used by the example run
    # scripts and the persona generator CLI.
    self.assertIn('google_aistudio', language_models._REGISTRY)


if __name__ == '__main__':
  absltest.main()
