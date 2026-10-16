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

"""Role-based language models: fast acts, slow minds, or the offline mock.

Live models come from `contrib.language_models.language_model_setup`, wrapped
in the standard `RetryLanguageModel`, `CallLimitLanguageModel` (the budget
governor: past the cap, calls degrade to empty text and first choices instead
of failing) and `ProfiledLanguageModel` (cost accounting).

The Together wrapper reads `TOGETHER_AI_API_KEY`; Deep Time also accepts
`TOGETHER_API_KEY` and always passes the key explicitly. Keys are never
logged.
"""

# pylint: disable=missing-function-docstring

import dataclasses
import hashlib
import importlib
import os
from typing import Any
import urllib.request

from absl import logging
from examples.deep_time import mock_language_model
from concordia.language_model import call_limit_wrapper
from concordia.language_model import language_model
from concordia.language_model import profiled_language_model
from concordia.language_model import retry_wrapper
from concordia.utils import profiler as profiler_lib
import numpy as np

# What a million tokens cost, by mode: a conservative upper bound for the
# Together models.
USD_PER_MILLION_TOKENS = {'mock': 0.0, 'together': 1.0, 'ollama': 0.0}
DEFAULT_FAST_MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'
DEFAULT_SLOW_MODEL = 'openai/gpt-oss-120b'
# Local models (the Ollama app on this machine): small for acts, larger for
# slow minds.
OLLAMA_FAST_MODEL = 'llama3.2:3b'
OLLAMA_SLOW_MODEL = 'llama3.2:3b'  # qwen3:8b thinks a minute a call.
OLLAMA_URL = 'http://127.0.0.1:11434'
EMBEDDING_DIM = 64


@dataclasses.dataclass
class Models:
  """Language models by role."""

  fast: language_model.LanguageModel
  slow: language_model.LanguageModel
  label: str
  profiler: profiler_lib.ProfilerContext
  mock: mock_language_model.DeepTimeMockModel | None = None


def usage(models: Models, mode: str) -> dict[str, Any]:
  """Calls, tokens and the estimated cost of the models so far.

  Args:
    models: Models made by `make_models`.
    mode: The mode they were made in (the price; a live mode that fell back to
      the mock costs nothing, and its label says so).

  Returns:
    {'calls', 'tokens', 'usd'}.
  """
  stats = models.profiler.get_stats()
  tokens = int(sum(stats.get('values', {}).get('llm_total_tokens', [])))
  calls = int(stats.get('counters', {}).get('llm_calls_total', 0))
  live = models.mock is None
  price = USD_PER_MILLION_TOKENS.get(mode, 1.0) if live else 0.0
  return {
      'calls': calls,
      'tokens': tokens,
      'usd': round(tokens * price / 1e6, 4),
  }


def ollama_reachable(timeout: float = 2.0) -> bool:
  """Whether the local Ollama server answers."""
  try:
    with urllib.request.urlopen(f'{OLLAMA_URL}/api/tags', timeout=timeout):
      return True
  except (OSError, ValueError):
    return False


def together_api_key() -> str | None:
  """Returns the Together key from the environment, never logging it."""
  return os.environ.get('TOGETHER_API_KEY') or os.environ.get(
      'TOGETHER_AI_API_KEY'
  )


def hash_embedder(text: str) -> np.ndarray:
  """A deterministic, dependency-free sentence embedder.

  Deep Time's entities rely on subscriptions rather than associative recall,
  so a cheap hashed bag of words is enough for memory retrieval and keeps runs
  offline and reproducible.

  Args:
    text: Text to embed.

  Returns:
    A unit vector of length `EMBEDDING_DIM`.
  """
  vector = np.zeros(EMBEDDING_DIM)
  for word in text.lower().split():
    digest = hashlib.md5(word.encode('utf-8')).digest()
    vector[digest[0] % EMBEDDING_DIM] += 1.0 if digest[1] % 2 else -1.0
  norm = np.linalg.norm(vector)
  return vector / norm if norm else vector


def _wrap(
    model: language_model.LanguageModel,
    name: str,
    max_calls: int,
    profiler: profiler_lib.ProfilerContext,
) -> language_model.LanguageModel:
  model = retry_wrapper.RetryLanguageModel(
      model,
      retry_on_exceptions=(TimeoutError, ConnectionError),
      retry_tries=2,
  )
  model = call_limit_wrapper.CallLimitLanguageModel(model, max_calls=max_calls)
  return profiled_language_model.ProfiledLanguageModel(
      model, model_name=name, profiler_instance=profiler
  )


def make_models(
    mode: str = 'mock',
    *,
    fast_model: str = DEFAULT_FAST_MODEL,
    slow_model: str = DEFAULT_SLOW_MODEL,
    max_calls: int = 300,
    mock_variant: str = 'default',
) -> Models:
  """Builds the models for a run.

  Args:
    mode: 'mock' (deterministic, offline), 'together' or 'ollama'.
    fast_model: Model for per-tick acts.
    slow_model: Model for background deliberation.
    max_calls: Call cap per model (the budget governor).
    mock_variant: Policy set of the mock model (see
      `mock_language_model.VARIANTS`).

  Returns:
    The models. A live mode without credentials falls back to the mock and
    says so in `label`, so provider trouble never stops a run.
  """
  profiler = profiler_lib.ProfilerContext()
  profiler.enable()
  if mode == 'mock':
    mock = mock_language_model.DeepTimeMockModel(
        mock_language_model.VARIANTS[mock_variant]
    )
    wrapped = profiled_language_model.ProfiledLanguageModel(
        mock, model_name='mock', profiler_instance=profiler
    )
    label = 'MOCK' if mock_variant == 'default' else f'MOCK ({mock_variant})'
    return Models(
        fast=wrapped, slow=wrapped, label=label, profiler=profiler, mock=mock
    )
  if mode == 'together':
    key = together_api_key()
    if not key:
      logging.warning('No Together key in the environment; using the mock.')
      fallback = make_models('mock')
      fallback.label = 'MOCK (fallback: no Together key)'
      return fallback
    language_models = importlib.import_module(
        'concordia.contrib.language_models'
    )
    fast = language_models.language_model_setup(
        api_type='together_ai', model_name=fast_model, api_key=key
    )
    slow = language_models.language_model_setup(
        api_type='together_ai', model_name=slow_model, api_key=key
    )
  elif mode == 'ollama':
    if fast_model == DEFAULT_FAST_MODEL:  # Together's names mean nothing here.
      fast_model, slow_model = OLLAMA_FAST_MODEL, OLLAMA_SLOW_MODEL
    if not ollama_reachable():
      logging.warning('Ollama is not reachable; using the mock.')
      fallback = make_models('mock')
      fallback.label = 'MOCK (fallback: Ollama unreachable)'
      return fallback
    language_models = importlib.import_module(
        'concordia.contrib.language_models'
    )
    fast = language_models.language_model_setup(
        api_type='ollama', model_name=fast_model
    )
    slow = language_models.language_model_setup(
        api_type='ollama', model_name=slow_model
    )
  else:
    raise ValueError(f'Unknown model mode {mode!r}')
  return Models(
      fast=_wrap(fast, fast_model, max_calls, profiler),
      slow=_wrap(slow, slow_model, max_calls, profiler),
      label=f'LIVE ({fast_model} / {slow_model})',
      profiler=profiler,
  )
