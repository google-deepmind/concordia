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

"""Language model that always returns empty strings and choice 0 (for debug)."""

from collections.abc import Collection, Mapping, Sequence
import random
from typing import Any

from concordia.language_model import language_model
import numpy as np
from typing_extensions import override


class NoLanguageModel(language_model.LanguageModel):
  """Debuging model that always returns empty strings and choice 0."""

  def __init__(
      self,
  ) -> None:
    """Debuging model that always returns empty strings and choice 0."""
  pass

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
    return ""

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    return 0, responses[0], {}


class RandomChoiceLanguageModel(NoLanguageModel):
  """A model that always returns a random choice in sample_choice."""

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    if not responses:
      return 0, "", {}
    if seed is not None:
      random.seed(seed)
    choice_index = random.randint(0, len(responses) - 1)
    return choice_index, responses[choice_index], {}


class BiasedMedianChoiceLanguageModel(NoLanguageModel):
  """A model that biases choices around the median in sample_choice."""

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    if not responses:
      return 0, "", {}

    if seed is not None:
      np.random.seed(seed)

    median_index = len(responses) // 2

    rand_val = np.random.rand()

    if rand_val < 0.8:
      choice_index = median_index
    elif rand_val < 0.9:
      choice_index = min(median_index + 1, len(responses) - 1)
    else:
      choice_index = max(median_index - 1, 0)

    return choice_index, responses[choice_index], {}
