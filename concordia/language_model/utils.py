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
from concordia.language_model import mistral_model
from concordia.language_model import no_language_model
from concordia.language_model import ollama_model
from concordia.language_model import pytorch_gemma_model


def language_model_setup(args):
  """Get the wrapped language model."""
  if args.disable_language_model:
    return no_language_model.NoLanguageModel()
  elif args.api_type == 'amazon_bedrock':
    return amazon_bedrock_model.AmazonBedrockLanguageModel(args.model_name)
  elif args.api_type == 'google_aistudio_model':
    return google_aistudio_model.GoogleAIStudioLanguageModel(args.model_name)
  elif args.api_type == 'langchain_ollama':
    return langchain_ollama_model.LangchainOllamaLanguageModel(args.model_name)
  elif args.api_type == 'mistral':
    return mistral_model.MistralLanguageModel(args.model_name)
  elif args.api_type == 'ollama':
    return ollama_model.OllamaLanguageModel(args.model_name)
  elif args.api_type == 'openai':
    return gpt_model.GptLanguageModel(args.model_name)
  elif args.api_type == 'pytorch_gemma':
    return pytorch_gemma_model.PyTorchGemmaLanguageModel(args.model_name)
  else:
    raise ValueError(f'Unrecognized api type: {args.api_type}')
