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

from concordia.language_model import amazon_bedrock_model
from concordia.language_model import google_aistudio_model
from concordia.language_model import google_cloud_custom_model
from concordia.language_model import gpt_model
from concordia.language_model import langchain_ollama_model
from concordia.language_model import language_model
from concordia.language_model import mistral_model
from concordia.language_model import no_language_model
from concordia.language_model import ollama_model
from concordia.language_model import pytorch_gemma_model
from concordia.language_model import together_ai


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
    disable_language_model: If True then disable the language model.
      This uses a model that returns an empty string whenever asked for a free
      text response and a randome option when asked for a choice.

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

  if api_type == 'amazon_bedrock':
    cls = amazon_bedrock_model.AmazonBedrockLanguageModel
  elif api_type == 'google_aistudio_model':
    cls = google_aistudio_model.GoogleAIStudioLanguageModel
  elif api_type == 'google_cloud_custom_model':
    cls = google_cloud_custom_model.VertexAI
  elif api_type == 'langchain_ollama':
    cls = langchain_ollama_model.LangchainOllamaLanguageModel
  elif api_type == 'mistral':
    cls = mistral_model.MistralLanguageModel
  elif api_type == 'ollama':
    cls = ollama_model.OllamaLanguageModel
  elif api_type == 'openai':
    cls = gpt_model.GptLanguageModel
  elif api_type == 'pytorch_gemma':
    cls = pytorch_gemma_model.PyTorchGemmaLanguageModel
  elif api_type == 'together_ai':
    cls = together_ai.Gemma2
  else:
    raise ValueError(f'Unrecognized api type: {api_type}')

  return cls(**kwargs)  # pytype: disable=wrong-keyword-args
