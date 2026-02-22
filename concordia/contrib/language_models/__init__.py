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

"""Utilities for loading language models."""

import importlib
import types

from concordia.language_model import language_model
from concordia.language_model import no_language_model


_REGISTRY = types.MappingProxyType({
    'amazon_bedrock': 'amazon.amazon_bedrock_model.AmazonBedrockLanguageModel',
    'gemini': 'google.gemini_model.GeminiModel',
    'google_cloud_custom_model': 'google.google_cloud_custom_model.VertexAI',
    'groq': 'groq.groq_model.GroqModel',
    'huggingface': 'huggingface.huggingface_model.HuggingFaceLanguageModel',
    'langchain_ollama': (
        'langchain.langchain_ollama_model.LangchainOllamaLanguageModel'
    ),
    'mistral': 'mistral.mistral_model.MistralLanguageModel',
    'ollama': 'ollama.ollama_model.OllamaLanguageModel',
    'openai': 'openai.gpt_model.GptLanguageModel',
    'pytorch_gemma': (
        'huggingface.pytorch_gemma_model.PyTorchGemmaLanguageModel'
    ),
    'together_ai': 'together.together_ai.Base',
    'vllm': 'vllm.vllm_model.VLLMLanguageModel',
})


def _import_model(model_path: str) -> type(language_model.LanguageModel):
  """Imports a model from this package."""
  module_path, class_name = f'{__name__}.{model_path}'.rsplit('.', 1)
  try:
    module = importlib.import_module(module_path)
  except ImportError as error:
    required_dependency, _ = model_path.split('.', 1)
    raise ImportError(
        f'Failed to import {module_path}. Please ensure you have installed the '
        'necessary dependencies: pip install '
        f'gdm-concordia[{required_dependency}].'
    ) from error
  return getattr(module, class_name)


def language_model_setup(
    *,
    api_type: str,
    model_name: str,
    api_key: str | None = None,
    device: str | None = None,
    disable_language_model: bool = False,
) -> language_model.LanguageModel:
  """Get the wrapped language model.

  Args:
    api_type: The type of API to use.
    model_name: The name of the specific model to use.
    api_key: The API key to use (if supported).
    device: The device to use for model processing (if supported).
    disable_language_model: If True then disable the language model. This uses a
      model that returns an empty string whenever asked for a free text response
      and a randome option when asked for a choice.

  Returns:
    The wrapped language model.
  """
  if disable_language_model:
    return no_language_model.NoLanguageModel()

  kwargs = {'model_name': model_name}
  if api_key is not None:
    kwargs['api_key'] = api_key
  if device is not None:
    kwargs['device'] = device

  try:
    model_path = _REGISTRY[api_type]
  except KeyError as error:
    raise ValueError(f'Unrecognized api_type: {api_type}') from error
  cls = _import_model(model_path)
  return cls(**kwargs)  # pytype: disable=wrong-keyword-args
