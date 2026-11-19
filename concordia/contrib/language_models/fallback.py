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

"""A language model wrapper that falls back to a second model.

A live run that hands its judgements to a hosted model can lose a whole
game or episode to one provider error that outlasts its retries, and a
budget enforced by `CallLimitLanguageModel` answers every call past the
limit with an empty string or the first option, which a game then acts on
as if it were a judgement.

`FallbackLanguageModel` answers with a primary model and sends a call to a
fallback model instead when the primary raises, or once a budget of primary
calls is spent. The fallback can be a local model, or a deterministic mock
that answers by the same rules (as a game's CI already does). It counts the
calls the fallback answered and keeps the last error, so a run can report
how much of it was the primary's.

Example:

```python
from concordia.contrib.language_models import fallback
from concordia.language_model import retry_wrapper

model = fallback.FallbackLanguageModel(
    retry_wrapper.RetryLanguageModel(hosted_model, retry_tries=2),
    local_model,
    max_primary_calls=300,
)
...
print(model.fallbacks, model.last_error)
```
"""

from collections.abc import Collection, Mapping, Sequence
import threading
from typing import Any, override

from absl import logging
from concordia.language_model import language_model


class FallbackLanguageModel(language_model.LanguageModel):
  """Answers with a primary model, and with a fallback when it cannot."""

  def __init__(
      self,
      primary: language_model.LanguageModel,
      fallback: language_model.LanguageModel,
      *,
      max_primary_calls: int | None = None,
      errors: tuple[type[Exception], ...] = (Exception,),
  ) -> None:
    """Initializes the wrapper.

    Args:
      primary: The model that answers while it can.
      fallback: The model that answers a call the primary cannot.
      max_primary_calls: How many calls the primary may take; later calls go
        to the fallback. None for no limit.
      errors: The exceptions from the primary that send a call to the
        fallback; others propagate.
    """
    self._primary = primary
    self._fallback = fallback
    self._max_primary_calls = max_primary_calls
    self._errors = errors
    self._lock = threading.Lock()
    self._primary_calls = 0
    self._fallbacks = 0
    self._last_error: str | None = None

  @property
  def primary_calls(self) -> int:
    """Calls handed to the primary, including those that failed."""
    with self._lock:
      return self._primary_calls

  @property
  def fallbacks(self) -> int:
    """Calls the fallback answered."""
    with self._lock:
      return self._fallbacks

  @property
  def last_error(self) -> str | None:
    """The primary's most recent error that sent a call to the fallback."""
    with self._lock:
      return self._last_error

  def _use_primary(self) -> bool:
    with self._lock:
      if (
          self._max_primary_calls is not None
          and self._primary_calls >= self._max_primary_calls
      ):
        self._fallbacks += 1
        return False
      self._primary_calls += 1
      return True

  def _failed(self, error: Exception) -> None:
    logging.warning('Primary model failed; using the fallback: %r', error)
    with self._lock:
      self._fallbacks += 1
      self._last_error = f'{type(error).__name__}: {error}'

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    kwargs = dict(
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=timeout,
        seed=seed,
    )
    if self._use_primary():
      try:
        return self._primary.sample_text(prompt, **kwargs)
      except self._errors as error:
        self._failed(error)
    return self._fallback.sample_text(prompt, **kwargs)

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    if self._use_primary():
      try:
        return self._primary.sample_choice(prompt, responses, seed=seed)
      except self._errors as error:
        self._failed(error)
    return self._fallback.sample_choice(prompt, responses, seed=seed)
