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

"""Resilient language model wrapper with retry and markdown stripping."""

from collections.abc import Collection, Mapping, Sequence
from typing import Any, Type, override

from concordia.language_model import language_model
from concordia.language_model import markdown_stripper
from concordia.language_model import retry_wrapper


class ResilientLanguageModel(retry_wrapper.RetryLanguageModel):
  """Wraps a language model with retry and markdown stripping."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      *,
      strip_markdown: bool = True,
      retry_on_exceptions: Collection[Type[Exception]] = (Exception,),
      retry_tries: int = 3,
      retry_delay: float = 2.0,
      jitter: tuple[float, float] = (0.0, 1.0),
      exponential_backoff: bool = True,
      backoff_factor: float = 2.0,
      max_delay: float = 300.0,
  ) -> None:
    """Wrap the underlying language model with retries and markdown stripping.

    Args:
      model: A language model to wrap.
      strip_markdown: Whether to strip markdown formatting from responses.
      retry_on_exceptions: The exception types to retry on.
      retry_tries: Number of retries before failing.
      retry_delay: Minimum delay between retries.
      jitter: Tuple of minimum and maximum jitter to add to the retry.
      exponential_backoff: Whether to enable exponential backoff.
      backoff_factor: The factor to use for exponential backoff.
      max_delay: The maximum delay between retries.
    """
    super().__init__(
        model=model,
        retry_on_exceptions=retry_on_exceptions,
        retry_tries=retry_tries,
        retry_delay=retry_delay,
        jitter=jitter,
        exponential_backoff=exponential_backoff,
        backoff_factor=backoff_factor,
        max_delay=max_delay,
    )
    self._strip_markdown = strip_markdown

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
    result = super().sample_text(
        prompt,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=timeout,
        seed=seed,
    )
    if self._strip_markdown:
      result = markdown_stripper.strip_markdown(result)
    return result

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    index, result, info = super().sample_choice(prompt, responses, seed=seed)
    if self._strip_markdown:
      result = markdown_stripper.strip_markdown(result)
    return index, result, info