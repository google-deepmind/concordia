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

"""The mock model's answers for the claims toy, read from prompts alone.

It judges relations with the lexicon (`lexicon.py`) and plays a side by
attacking the strongest opposing claim on the board it is shown, or shoring
up its thesis when that is close to falling.
"""

from collections.abc import Mapping, Sequence
import re
from typing import Any

from examples.crucible.toys.claims import bots
from examples.crucible.toys.claims import lexicon as lexicon_lib
from examples.crucible.toys.claims import toy as toy_lib
from examples.crucible.toys.claims import world as world_lib

_RELATION = re.compile(r'Claim A: (.*)\nClaim B: (.*)\n')
_OPTION = re.compile(r'^\s*\(([a-z])\) (.+)$', re.MULTILINE)
_LINE = re.compile(
    r'^\s*(c\d+) \[(\d+\.\d+)\] \(([^)]*?)( THESIS)?\) (.+)$', re.MULTILINE
)
_ACTOR = re.compile(re.escape(toy_lib.MOVE_MARKER) + r', ([^:]+):')
_FALL = re.compile(r'A claim falls below (\d+\.\d+)')
_INDEX = {
    world_lib.AGREE: 0,
    world_lib.CONFLICT: 1,
    world_lib.RESTATES: 2,
    world_lib.UNRELATED: 3,
}


def make_lexicon(data: Mapping[str, Any]) -> lexicon_lib.Lexicon:
  lex = data['lexicon']
  return lexicon_lib.Lexicon(lex['stopwords'], lex['negations'], lex['poles'])


class ClaimsMock:
  """Answers the claims toy's prompts deterministically."""

  def __init__(self, data: Mapping[str, Any]):
    self._lexicon = make_lexicon(data)

  def choice(self, prompt: str, responses: Sequence[str]) -> int | None:
    match = _RELATION.search(prompt)
    if not match or toy_lib.RELATION_QUESTION not in prompt:
      return None
    wanted = toy_lib.RELATION_OPTIONS[
        _INDEX[self._lexicon.relation(match.group(1), match.group(2))]
    ]
    # The options are lettered in the prompt (InteractiveDocument); answer
    # with the letter of the wanted one.
    letters = {text.strip(): key for key, text in _OPTION.findall(prompt)}
    letter = letters.get(wanted)
    return list(responses).index(letter) if letter in responses else None

  def text(self, prompt: str) -> str | None:
    actor_match = _ACTOR.search(prompt)
    if not actor_match:
      return None
    actor = actor_match.group(1).strip()
    fall_match = _FALL.search(prompt)
    fall = float(fall_match.group(1)) if fall_match else 0.35
    # Only the last rendering of the board counts (earlier turns repeat it).
    start = prompt.rfind('Standing claims')
    claims = _LINE.findall(prompt[start:] if start >= 0 else prompt)
    mine = [c for c in claims if c[2] == actor and c[3]]
    theirs = [c for c in claims if c[2] not in (actor, 'board')]
    if mine and float(mine[0][1]) < fall + 0.25:
      return bots.affirm(mine[0][4])
    if theirs:
      target = max(theirs, key=lambda c: (bool(c[3]), float(c[1])))
      return bots.negate(target[4])
    return 'pass'
