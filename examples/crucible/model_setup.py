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

from collections.abc import Mapping, Sequence
import hashlib
import os
import threading
from typing import Any, override

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


class UsageCounter(language_model.LanguageModel):
  """Counts a model's calls and estimates its tokens (characters / 4).

  Thread-safe; a toy may read with it from several threads at once
  (`parallel_safe`).
  """

  parallel_safe = True

  def __init__(self, model: language_model.LanguageModel):
    self._model = model
    self._lock = threading.Lock()
    self.calls = 0
    self.chars = 0

  def _count(self, chars: int) -> None:
    with self._lock:
      self.calls += 1
      self.chars += chars

  @property
  def tokens(self) -> int:
    return self.chars // 4

  @override
  def sample_text(self, prompt: str, **kwargs: Any) -> str:
    out = self._model.sample_text(prompt, **kwargs)
    self._count(len(prompt) + len(out))
    return out

  @override
  def sample_choice(
      self, prompt: str, responses: Sequence[str], **kwargs: Any
  ) -> tuple[int, str, Mapping[str, Any]]:
    result = self._model.sample_choice(prompt, responses, **kwargs)
    self._count(len(prompt) + 4)
    return result


def mock_model() -> mock_language_model.CrucibleMock:
  """The mock, with every registered toy's handler."""
  return mock_language_model.CrucibleMock([
      kind.mock(toys.load_data(name)) for name, kind in toys.registry().items()
  ])


# Seconds an Ollama reply may take before the retry wrapper tries again (a
# thinking reply has no bound; one golden request hung for good, cycle 17).
OLLAMA_TIMEOUT = 180.0


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
    # Thinking kept apart: left in the reply, a newline terminator cut it
    # off and open questions came back empty; turned off, qwen3:8b broke 6
    # of 17 hard golden cases (cycle 17).
    from concordia.contrib.language_models.ollama import ollama_model  # pylint: disable=g-import-not-at-top

    base = ollama_model.OllamaLanguageModel(
        model_name or DEFAULT_OLLAMA, think=True, timeout=OLLAMA_TIMEOUT
    )
  elif mode == 'together':
    key = os.environ.get('TOGETHER_API_KEY') or os.environ.get(
        'TOGETHER_AI_API_KEY'
    )
    base = language_models.language_model_setup(
        api_type='together_ai',
        model_name=model_name or DEFAULT_TOGETHER,
        api_key=key,
    )
  else:
    raise ValueError(f'Unknown mode: {mode}')
  # Judgements (multiple choice) go through a short reason, at temperature 0
  # so the same reading gives the same answer.
  model = reasoned_choice.ReasonedChoice(base, temperature=0.0)
  model = retry_wrapper.RetryLanguageModel(model, retry_tries=2)
  return call_limit_wrapper.CallLimitLanguageModel(model, max_calls=max_calls)
