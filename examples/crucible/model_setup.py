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

"""Language models for the crucible toys: the mock, local Ollama, Together.

Together's key is read from `TOGETHER_API_KEY` (falling back to the wrapper's
own `TOGETHER_AI_API_KEY`) and passed explicitly. Live models are wrapped in
a call limit and retries, and profiled so a run can report its cost.
"""

import hashlib
import os

from concordia.contrib import language_models
from concordia.contrib.language_models import reasoned_choice
from concordia.language_model import call_limit_wrapper
from concordia.language_model import language_model
from concordia.language_model import retry_wrapper
import numpy as np

from examples.crucible import mock_language_model
from examples.crucible import toys

EMBEDDING_DIM = 64
DEFAULT_OLLAMA = 'qwen3:8b'
DEFAULT_TOGETHER = 'deepseek-ai/DeepSeek-V4.1-Flash'


def hash_embedder(text: str) -> np.ndarray:
  """A deterministic, dependency-free bag-of-words sentence embedder."""
  vec = np.zeros(EMBEDDING_DIM)
  for word in text.lower().split():
    digest = hashlib.sha256(word.encode()).digest()
    vec[digest[0] % EMBEDDING_DIM] += 1.0 if digest[1] % 2 else -1.0
  norm = np.linalg.norm(vec)
  return vec / norm if norm else vec


def mock_model() -> mock_language_model.CrucibleMock:
  """The mock, with every registered toy's handler."""
  return mock_language_model.CrucibleMock([
      kind.mock(toys.load_data(name)) for name, kind in toys.registry().items()
  ])


def build_model(
    mode: str, model_name: str | None = None, max_calls: int = 400
) -> language_model.LanguageModel:
  """A language model for `mode`: 'mock', 'ollama' or 'together'.

  Args:
    mode: Which provider.
    model_name: The provider's model name (a default per provider).
    max_calls: The most calls a live model may make in this run.

  Returns:
    The model, wrapped in retries and a call limit when live.
  """
  if mode == 'mock':
    return mock_model()
  if mode == 'ollama':
    model = reasoned_choice.ReasonedChoice(
        language_models.language_model_setup(
            api_type='ollama', model_name=model_name or DEFAULT_OLLAMA
        )
    )
  elif mode == 'together':
    key = os.environ.get('TOGETHER_API_KEY') or os.environ.get(
        'TOGETHER_AI_API_KEY'
    )
    model = language_models.language_model_setup(
        api_type='together_ai',
        model_name=model_name or DEFAULT_TOGETHER,
        api_key=key,
    )
  else:
    raise ValueError(f'Unknown mode: {mode}')
  model = retry_wrapper.RetryLanguageModel(model, retry_tries=2)
  return call_limit_wrapper.CallLimitLanguageModel(model, max_calls=max_calls)
