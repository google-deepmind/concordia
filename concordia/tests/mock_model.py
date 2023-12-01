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
"""A mock Language Model."""

from collections.abc import Collection, Sequence
import sys

from concordia.language_model import language_model


class MockModel(language_model.LanguageModel):
  """Mock LLM with fixed responses."""

  def __init__(
      self, response: str = 'Quick brown fox jumps over a lazy dog'
  ) -> None:
    """Initializes the instance.

    Args:
      response: string that the model returns when sampling text
    """
    self._response = response

  def sample_text(
      self,
      prompt: str,
      *,
      timeout: float = 0,
      max_tokens: int = 0,
      max_characters: int = sys.maxsize,
      terminators: Collection[str] = (),
      temperature: float = 0.5,
      seed: int | None = None,
  ) -> str:
    """See base class."""
    del (
        prompt,
        timeout,
        max_tokens,
        max_characters,
        terminators,
        seed,
        temperature,
    )
    return self._response

  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    """See base class."""
    del prompt, seed
    return 0, responses[0], {}
