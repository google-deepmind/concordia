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
_EXPECTED = re.compile(r'others expect of us \(([a-z]+)\) \[(c\d+)\]')
_SIGHT = re.compile(r'(\d+) other people are within sight')


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
    expected = {cid for _, cid in _EXPECTED.findall(prompt)}
    return _choose(prompt, moves, customs, expected)

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
    prompt: str,
    moves: Sequence[str],
    customs: Sequence[tuple[str, str, str]],
    expected: Collection[str],
) -> str:
  """A person's move, with the customs it follows.

  A norm of the person's own is kept whenever it applies; one merely
  expected of the person (`expected`), only when someone is within sight.

  Args:
    prompt: The person's prompt.
    moves: The moves offered.
    customs: The person's customs: (domain, id, text).
    expected: The ids of the norms only expected of the person.

  Returns:
    The answer.
  """
  if moves == ['watch']:
    return 'watch'
  texts = ' '.join(text.lower() for _, _, text in customs)
  food = float(_FOOD.findall(prompt)[-1]) if _FOOD.findall(prompt) else 0.0
  age = int(_AGE.findall(prompt)[-1]) if _AGE.findall(prompt) else 30
  draw = _stable(prompt)
  land = float(_LAND.findall(prompt)[-1]) if _LAND.findall(prompt) else 99.0
  watched = bool(_SIGHT.findall(prompt)) and int(_SIGHT.findall(prompt)[-1]) > 0
  # The sharing norms that call for sharing (not every one does).
  sharing = [
      cid
      for domain, cid, text in customs
      if domain == 'sharing'
      and 'share' in text.lower()
      and (cid not in expected or watched)
  ]
  share = [m for m in moves if m.startswith('share food with ')]
  shame = [m for m in moves if m.startswith('shame ')]

  def follow(move: str, *domains: str) -> str:
    """The move, naming the customs of its domains (all, for teaching)."""
    ids = [
        cid for domain, cid, _ in customs if not domains or domain in domains
    ]
    return ' '.join([move, *ids])

  if 'move to better land' in moves and land < 4.0 and draw > 0.4:
    return 'move to better land'
  if 'learn a craft' in moves and food >= 1.0 and draw < 0.6:
    return 'learn a craft'
  if 'play' in moves and age < 12:
    return 'play' if food >= 0.5 else follow('gather food', 'food')
  if food < 2.5 or ('lean' in texts and food < 4.0):
    return follow('gather food', 'food')
  if 'look for a partner' in moves and age >= 18 and draw < 0.9:
    return follow('look for a partner', 'kinship')
  if 'have a child' in moves:
    careful = 'only when' in texts and food < 6.0
    if not careful and draw < 0.8:
      return follow('have a child', 'kinship')
  if share and sharing and food >= 4.0:
    return ' '.join([share[0], *sharing])
  if shame and sharing and draw < 0.5:
    return ' '.join([shame[0], *sharing])
  if 'teach' in moves and food >= 4.0 and draw < 0.35:
    return follow('teach')
  if 'move to better land' in moves and draw > 0.92:
    return 'move to better land'
  return follow('gather food', 'food')
