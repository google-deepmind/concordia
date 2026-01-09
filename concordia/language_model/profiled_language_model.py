# Copyright 2024 DeepMind Technologies Limited.
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

"""Wrapper to profile calls to an underlying language model."""

from collections.abc import Callable, Collection, Mapping, Sequence
import time
from typing import Any, override

from concordia.language_model import language_model
from concordia.utils import profiler


def estimate_tokens(text: str) -> int:
  """Estimate the number of tokens in text.

  Uses a simple heuristic: 1 token ≈ 4 characters.
  This is a rough approximation that works reasonably well for English text.

  Args:
    text: The text to estimate tokens for.

  Returns:
    Estimated number of tokens.
  """
  return max(1, len(text) // 4)


class ProfiledLanguageModel(language_model.LanguageModel):
  """Wraps an underlying language model and profiles calls to it.

  This wrapper tracks:
  - Timing of LLM calls
  - Number of calls (sample_text vs sample_choice)
  - Estimated token usage (prompt and completion)
  - Success/failure status

  The profiling data is sent to the global profiler, which can be enabled
  or disabled independently.

  Usage:
    from concordia.language_model import profiled_language_model
    from concordia.utils import profiler

    # Wrap your language model
    profiled_model = profiled_language_model.ProfiledLanguageModel(
        base_model,
        model_name="gpt-4"  # optional
    )

    # Enable profiling
    profiler.enable()

    # Use the model normally
    response = profiled_model.sample_text("Hello")

    # View profiling report
    profiler.print_report()
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      model_name: str = 'unknown',
      token_calculator: Callable[[str], int] | None = None,
      profiler_instance: profiler.ProfilerContext | None = None,
  ) -> None:
    """Wrap the underlying language model with profiling.

    Args:
      model: A language model to wrap with profiling.
      model_name: Optional name for the model (for reporting purposes).
      token_calculator: Optional custom token counting function. If None,
        uses estimate_tokens (1 token ≈ 4 characters).
      profiler_instance: Optional profiler instance to use. If None, uses
        the global profiler. Useful for parallel simulations to avoid stat
        bleeding between simulations.
    """
    self._model = model
    self._model_name = model_name
    self._token_calculator = token_calculator or estimate_tokens
    self._profiler = profiler_instance or profiler._global_profiler

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
    """Sample text from the model with profiling.

    Args:
      prompt: the initial text to condition on.
      max_tokens: the maximum number of tokens in the response.
      terminators: the response will be terminated before any of these
        characters.
      temperature: temperature for the model.
      top_p: filters tokens based on cumulative probability.
      top_k: filters tokens by selecting the top_k most probable tokens.
      timeout: timeout for the request.
      seed: optional seed for the sampling.

    Returns:
      The sampled response.

    Raises:
      TimeoutError: if the operation times out.
    """
    if not self._profiler.is_enabled():
      # Fast path: profiling disabled, just call the model
      return self._model.sample_text(
          prompt,
          max_tokens=max_tokens,
          terminators=terminators,
          temperature=temperature,
          top_p=top_p,
          top_k=top_k,
          timeout=timeout,
          seed=seed,
      )

    # Track profiling data
    start_time = time.perf_counter()

    try:
      response = self._model.sample_text(
          prompt,
          max_tokens=max_tokens,
          terminators=terminators,
          temperature=temperature,
          top_p=top_p,
          top_k=top_k,
          timeout=timeout,
          seed=seed,
      )

      # Record successful call metrics
      duration = time.perf_counter() - start_time
      self._profiler.record_time('llm_sample_text', duration)
      self._profiler.increment_counter('llm_calls_total')
      self._profiler.increment_counter('llm_calls_sample_text')
      self._profiler.increment_counter('llm_calls_success')

      # Estimate tokens
      prompt_tokens = self._token_calculator(prompt)
      completion_tokens = self._token_calculator(response)
      total_tokens = prompt_tokens + completion_tokens

      self._profiler.record_value('llm_prompt_tokens', prompt_tokens)
      self._profiler.record_value('llm_completion_tokens', completion_tokens)
      self._profiler.record_value('llm_total_tokens', total_tokens)
      self._profiler.record_value('llm_latency_seconds', duration)

      return response

    except Exception as e:
      duration = time.perf_counter() - start_time

      # Record failed call
      self._profiler.record_time('llm_sample_text', duration)
      self._profiler.increment_counter('llm_calls_total')
      self._profiler.increment_counter('llm_calls_sample_text')
      self._profiler.increment_counter('llm_calls_failed')
      self._profiler.record_value('llm_latency_seconds', duration)

      # Re-raise the exception
      raise

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    """Sample a choice from the model with profiling.

    Args:
      prompt: the initial text to condition on.
      responses: the responses to score.
      seed: optional seed for the sampling.

    Returns:
      (index, response, info) tuple.

    Raises:
      InvalidResponseError: if unable to produce a valid choice.
    """
    if not self._profiler.is_enabled():
      # Fast path: profiling disabled, just call the model
      return self._model.sample_choice(prompt, responses, seed=seed)

    # Track profiling data
    start_time = time.perf_counter()

    try:
      index, response, info = self._model.sample_choice(
          prompt, responses, seed=seed
      )

      # Record successful call metrics
      duration = time.perf_counter() - start_time
      self._profiler.record_time('llm_sample_choice', duration)
      self._profiler.increment_counter('llm_calls_total')
      self._profiler.increment_counter('llm_calls_sample_choice')
      self._profiler.increment_counter('llm_calls_success')

      # Estimate tokens (prompt + all possible responses)
      prompt_tokens = self._token_calculator(prompt)
      responses_tokens = sum(self._token_calculator(r) for r in responses)
      total_tokens = prompt_tokens + responses_tokens

      self._profiler.record_value('llm_prompt_tokens', prompt_tokens)
      self._profiler.record_value('llm_total_tokens', total_tokens)
      self._profiler.record_value('llm_latency_seconds', duration)

      return index, response, info

    except Exception as e:
      duration = time.perf_counter() - start_time

      # Record failed call
      self._profiler.record_time('llm_sample_choice', duration)
      self._profiler.increment_counter('llm_calls_total')
      self._profiler.increment_counter('llm_calls_sample_choice')
      self._profiler.increment_counter('llm_calls_failed')
      self._profiler.record_value('llm_latency_seconds', duration)

      # Re-raise the exception
      raise
