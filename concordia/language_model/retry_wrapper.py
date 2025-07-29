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

"""Wrapper to retry calls to an underlying language model."""

from collections.abc import Collection, Sequence, Mapping
from typing import Any, Type

from concordia.language_model import language_model
import retry
from typing_extensions import override


class RetryLanguageModel(language_model.LanguageModel):
  """Wraps an underlying language model and retries calls to it."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      retry_on_exceptions: Collection[Type[Exception]] = (Exception,),
      retry_tries: int = 3,
      retry_delay: float = 2.0,
      jitter: tuple[float, float] = (0.0, 1.0),
      exponential_backoff: bool = True,
      backoff_factor: float = 2.0,
      max_delay: float = 300.0,
  ) -> None:
    """Wrap the underlying language model with retries on given exceptions.

    Args:
      model: A language model to wrap with retries.
      retry_on_exceptions: the exception exceptions to retry on.
      retry_tries: number of retries before failing.
      retry_delay: minimum delay between retries.
      jitter: tuple of minimum and maximum jitter to add to the retry.
      exponential_backoff: whether to enable exponential backoff.
      backoff_factor: The factor to use for exponential backoff.
      max_delay: The maximum delay between retries.
    """
    self._model = model
    self._retry_on_exceptions = tuple(retry_on_exceptions)
    self._retry_tries = retry_tries
    self._retry_delay = retry_delay
    self._jitter = jitter
    self._exponential_backoff = exponential_backoff
    self._backoff_factor = backoff_factor
    self._max_delay = max_delay

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    @retry.retry(
        self._retry_on_exceptions,
        tries=self._retry_tries,
        delay=self._retry_delay,
        backoff=self._backoff_factor if self._exponential_backoff else 1,
        max_delay=self._max_delay,
        jitter=self._jitter,
    )
    def _sample_text(
        model,
        prompt,
        *,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        seed=seed,
    ):
      return model.sample_text(
          prompt,
          max_tokens=max_tokens,
          terminators=terminators,
          temperature=temperature,
          seed=seed,
      )

    return _sample_text(
        self._model,
        prompt,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        seed=seed,
    )

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    @retry.retry(
        self._retry_on_exceptions,
        tries=self._retry_tries,
        delay=self._retry_delay,
        backoff=self._backoff_factor if self._exponential_backoff else 1,
        max_delay=self._max_delay,
        jitter=self._jitter,
    )
    def _sample_choice(model, prompt, responses, *, seed):
      return model.sample_choice(prompt, responses, seed=seed)

    return _sample_choice(self._model, prompt, responses, seed=seed)
