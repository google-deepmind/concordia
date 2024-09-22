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
    disable_language_model: bool = False,
    device: str = 'cpu'
) -> language_model.LanguageModel:
  """Get the wrapped language model.

  Args:
    api_type: The type of API to use.
    model_name: The name of the specific model to use.
    api_key: Optional, the API key to use. If None, will use the environment
      variable for the specified API type.
    disable_language_model: Optional, if True then disable the language model.
      This uses a model that returns an empty string whenever asked for a free
      text response and a randome option when asked for a choice.
    device: Specifies whether to use cpu or cuda for model processing.

  Returns:
    The wrapped language model.
  """
  if disable_language_model:
    return no_language_model.NoLanguageModel()
  elif api_type == 'amazon_bedrock':
    if api_key is not None:
      raise ValueError(
          'Explicitly passing the API key is not supported for Amazon Bedrock '
          'models. Please use an environment variable instead.'
      )
    return amazon_bedrock_model.AmazonBedrockLanguageModel(model_name)
  elif api_type == 'google_aistudio_model':
    return google_aistudio_model.GoogleAIStudioLanguageModel(
        model_name=model_name, api_key=api_key
    )
  elif api_type == 'langchain_ollama':
    return langchain_ollama_model.LangchainOllamaLanguageModel(model_name)
  elif api_type == 'mistral':
    return mistral_model.MistralLanguageModel(model_name, api_key=api_key)
  elif api_type == 'ollama':
    return ollama_model.OllamaLanguageModel(model_name)
  elif api_type == 'openai':
    return gpt_model.GptLanguageModel(model_name, api_key=api_key)
  elif api_type == 'pytorch_gemma':
    return pytorch_gemma_model.PyTorchGemmaLanguageModel(model_name, device=device)
  elif api_type == 'together_ai':
    return together_ai.Gemma2(model_name, api_key=api_key)
  else:
    raise ValueError(f'Unrecognized api type: {api_type}')
