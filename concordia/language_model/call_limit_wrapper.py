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

"""Wrapper to limit calls to an underlying language model."""

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from concordia.language_model import language_model
from typing_extensions import override


class CallLimitLanguageModel(language_model.LanguageModel):
  """Wraps an underlying language model and limits calls to it.

  Once the limit on calls is reached the model outputs an empty string on sample
  text and returns the first response on choice questions. If the underlying
  model is calling sample_text or sample_choice, it will result in counting
  towards the call limit.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      max_calls: int = 1000,
  ) -> None:
    """Wrap the underlying language model with a call limit.

    Args:
      model: A language model to wrap with a call limit.
      max_calls: the maximum number of calls to the underlying model.
    """
    self._model = model
    self._max_calls = max_calls
    self._calls = 0

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
    if self._calls >= self._max_calls:
      print(
          f'\n\n***** WARNING *****\nCall limit of {self._max_calls} reached.'
          ' All further sample_text calls will be replaced with empty strings'
          ' sample_choice calls with the first response\n\n'
      )

      return ''

    self._calls += 1
    return self._model.sample_text(
        prompt,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        timeout=timeout,
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
    if self._calls >= self._max_calls:
      print(
          f'\n\n***** WARNING *****\nCall limit of {self._max_calls} reached.'
          ' All further sample_text calls will be replaced with empty strings'
          ' sample_choice calls with the first response\n\n'
      )
      return 0, responses[0], {}

    self._calls += 1
    return self._model.sample_choice(prompt, responses, seed=seed)
