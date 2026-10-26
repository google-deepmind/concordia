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

"""A deterministic, game-aware language model for Lifeways (MOCK).

It answers a person's question the way the person's prompt suggests: from
what the person observed of its body and place, and from what its customs
say (their texts are in the prompt, with their ids). Customs therefore change
what a mock person does, just as they change a model's answer. The same
prompt always gets the same answer.
"""

from collections.abc import Collection, Mapping, Sequence
import re
from typing import Any, override
import zlib

from concordia.language_model import language_model

_FOOD = re.compile(r'Food stored: (\d+(?:\.\d+)?)')
_AGE = re.compile(r'is (\d+) years old')
_LAND = re.compile(r'the land around holds (\d+(?:\.\d+)?) food')
_MOVES = re.compile(r'^- (.+)$', re.MULTILINE)
_LABEL = re.compile(r'(?:Question|Exercise): ')
_CUSTOM = re.compile(r'\(([a-z]+)\) \[(c\d+)\]: ([^\n]*)')


def _stable(text: str) -> float:
  """A number in [0, 1) fixed by the text."""
  return (zlib.crc32(text.encode()) % 10_000) / 10_000


class MockModel(language_model.LanguageModel):
  """Answers Lifeways' questions by rules of the prompt."""

  def __init__(self):
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
    self.calls += 1
    question = _LABEL.split(prompt)[-1]
    customs = _CUSTOM.findall(prompt)
    if 'pass on' in question and 'Which of' in question:
      return 'pass on ' + ' '.join(cid for _, cid, _ in customs)
    moves = _MOVES.findall(question)
    if not moves:
      return ''
    return _choose(prompt, moves, customs)

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    del seed
    self.calls += 1
    index = int(_stable(prompt) * len(responses))
    return index, responses[index], {}


def _choose(
    prompt: str, moves: Sequence[str], customs: Sequence[tuple[str, str, str]]
) -> str:
  """A person's move, with the custom it follows."""
  if moves == ['watch']:
    return 'watch'
  texts = ' '.join(text.lower() for _, _, text in customs)
  food_skill = next(
      (cid for domain, cid, text in customs if domain == 'food'), ''
  )
  food = float(_FOOD.findall(prompt)[-1]) if _FOOD.findall(prompt) else 0.0
  age = int(_AGE.findall(prompt)[-1]) if _AGE.findall(prompt) else 30
  draw = _stable(prompt)
  land = float(_LAND.findall(prompt)[-1]) if _LAND.findall(prompt) else 99.0

  def follow(move: str, cid: str = '') -> str:
    return f'{move} {cid}'.strip()

  if 'move to better land' in moves and land < 4.0 and draw > 0.4:
    return follow('move to better land')
  if 'play' in moves and age < 12:
    return follow('play') if food >= 0.5 else follow('gather food', food_skill)
  if food < 2.5 or ('lean' in texts and food < 4.0):
    return follow('gather food', food_skill)
  if 'look for a partner' in moves and age >= 18 and draw < 0.9:
    return follow('look for a partner')
  if 'have a child' in moves:
    kin = next((cid for d, cid, _ in customs if d == 'kinship'), '')
    careful = 'only when' in texts and food < 6.0
    if not careful and draw < 0.8:
      return follow('have a child', kin)
  if 'move to better land' in moves and draw > 0.92:
    return follow('move to better land')
  return follow('gather food', food_skill)
