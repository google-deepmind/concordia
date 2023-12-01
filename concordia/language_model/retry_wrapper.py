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

from collections.abc import Collection, Sequence
import copy
from typing import Any, Mapping, Tuple, Type

from concordia.language_model import language_model
import retry


class RetryLanguageModel(language_model.LanguageModel):
  """Wraps an underlying language model and retries calls to it."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      retry_on_exceptions: Collection[Type[Exception]] = (Exception,),
      retry_tries: float = 3.,
      retry_delay: float = 2.,
      jitter: Tuple[float, float] = (0.0, 1.0),
  ) -> None:
    """Wrap the underlying language model with retries on given exceptions.

    Args:
      model: A language model to wrap with retries.
      retry_on_exceptions: the exception exceptions to retry on.
      retry_tries: number of retries before failing.
      retry_delay: minimum delay between retries.
      jitter: tuple of minimum and maximum jitter to add to the retry.
    """
    self._model = model
    self._retry_on_exceptions = copy.deepcopy(retry_on_exceptions)
    self._retry_tries = retry_tries
    self._retry_delay = retry_delay
    self._jitter = jitter

  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      max_characters: int = language_model.DEFAULT_MAX_CHARACTERS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      seed: int | None = None,
  ) -> str:
    """See base class."""
    @retry.retry(self._retry_on_exceptions, tries=self._retry_tries,
                 delay=self._retry_delay, jitter=self._jitter)
    def _sample_text(model, prompt, *, max_tokens=max_tokens,
                     max_characters=max_characters, terminators=terminators,
                     temperature=temperature, seed=seed):
      return model.sample_text(
          prompt, max_tokens=max_tokens, max_characters=max_characters,
          terminators=terminators, temperature=temperature, seed=seed)

    return _sample_text(self._model, prompt, max_tokens=max_tokens,
                        max_characters=max_characters, terminators=terminators,
                        temperature=temperature, seed=seed)

  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    """See base class."""
    @retry.retry(self._retry_on_exceptions, tries=self._retry_tries,
                 delay=self._retry_delay, jitter=self._jitter)
    def _sample_choice(model, prompt, responses, *, seed):
      return model.sample_choice(prompt, responses, seed=seed)

    return _sample_choice(self._model, prompt, responses, seed=seed)
