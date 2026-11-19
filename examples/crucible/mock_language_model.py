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

"""A deterministic, rule-based offline language model for the crucible toys.

Like `examples/concordia_island/mock_language_model.py`, it lets every toy run
without network access, but it plays sensibly: each toy contributes a
handler that reads its own prompts (a relation question, a move request) and
answers with small rules. Answers depend only on the prompt, so runs are
reproducible. This is the lookup-table floor of each toy's semantics; the
games are meant for real language models.
"""

from collections.abc import Collection, Mapping, Sequence
import threading
from typing import Any, Protocol, override

from concordia.language_model import language_model


class Handler(Protocol):
  """A toy's mock answers; None means "not my prompt"."""

  def choice(self, prompt: str, responses: Sequence[str]) -> int | None:
    ...

  def text(self, prompt: str) -> str | None:
    ...


class CrucibleMock(language_model.LanguageModel):
  """Routes each prompt to the first toy handler that recognises it."""

  def __init__(self, handlers: Sequence[Handler]):
    self._handlers = list(handlers)
    self._lock = threading.Lock()
    self.calls = 0

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
    del max_tokens, terminators, temperature, top_p, top_k, timeout, seed
    with self._lock:
      self.calls += 1
    for handler in self._handlers:
      answer = handler.text(prompt)
      if answer is not None:
        return answer
    return ''

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    del seed
    with self._lock:
      self.calls += 1
    for handler in self._handlers:
      idx = handler.choice(prompt, responses)
      if idx is not None:
        return idx, responses[idx], {}
    return 0, responses[0], {}
