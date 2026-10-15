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

"""The mock model's answers for the orders toy, read from prompts alone.

It splits an order into clauses, treats a clause with "if", "when" (and
similar) as applying only when its condition holds, maps verbs to actions
with a word list, and then reads the result through each temperament: the
bold build unless told otherwise, the cautious repair and rest, the literal
do only what is written. This is the lookup-table floor of the toy.
"""

from collections.abc import Mapping, Sequence
import re
from typing import Any

from examples.crucible.toys.orders import toy as toy_lib

_STATEMENT = re.compile(
    r'(\w+) is a crew member who is (\w+):.*?\n'
    r'The standing order from the commander is: "(.*)"\n'
    r'This turn: (.*)\n'
    r'Stores: (\d+) supplies; hull damage (\d+); project (\d+) of (\d+)\.'
    r' \w+ is (worn out|fresh)\.',
)
_OPTION = re.compile(r'^\s*\(([a-z])\) (.+)$', re.MULTILINE)
_WORD = re.compile(r'[a-z]+')
_CLAUSE = re.compile(r'[.;]|\b(?:otherwise|else|then)\b')


class OrdersMock:
  """Answers the orders toy's prompts deterministically."""

  def __init__(self, data: Mapping[str, Any]):
    self._data = data
    lex = data['lexicon']
    self._verbs = {a: frozenset(lex[a]) for a in data['actions']}
    self._conditions = {k: frozenset(v) for k, v in lex['conditions'].items()}
    self._if = frozenset(lex['if_words'])

  def ordered(self, order: str, now: set[str]) -> list[str]:
    """The actions the order asks for now, in the order they are named."""
    conditional, default = [], []
    for clause in _CLAUSE.split(order.lower()):
      words = _WORD.findall(clause or '')
      if not words:
        continue
      named = [a for w in words for a, vs in self._verbs.items() if w in vs]
      if self._if & set(words):
        wants = {k for k, vs in self._conditions.items() if vs & set(words)}
        if wants & now:
          conditional.extend(named)
      else:
        default.extend(named)
    chosen = conditional or default
    return list(dict.fromkeys(chosen))

  def decide(
      self,
      temperament: str,
      order: str,
      situation: str,
      supplies: int,
      damage: int,
      worn: bool,
  ) -> str:
    now = set()
    if 'leak' in situation.lower():
      now.add('leak')
    if 'storm' in situation.lower():
      now.add('storm')
    if worn:
      now.add('tired')
    if supplies <= 1:
      now.add('low')
    wanted = self.ordered(order, now)
    if temperament == 'literal':
      return wanted[0] if wanted else 'rest'
    if temperament == 'cautious':
      if worn and 'rest' not in wanted:
        return 'rest'
      if 'leak' in now and 'repair' not in wanted and not wanted:
        return 'repair'
      if wanted:
        return wanted[0]
      return 'repair' if damage > 0 else 'gather'
    # Bold.
    if 'build' in wanted:
      return 'build' if supplies > 0 else 'gather'
    if wanted:
      return wanted[0]
    return 'build' if supplies > 0 else 'gather'

  def choice(self, prompt: str, responses: Sequence[str]) -> int | None:
    match = _STATEMENT.search(prompt)
    if not match or 'do this turn?' not in prompt:
      return None
    _, temperament, order, situation, supplies, damage, _, _, cond = (
        match.groups()
    )
    action = self.decide(
        temperament, order, situation, int(supplies), int(damage),
        cond == 'worn out',
    )  # fmt: skip
    wanted = self._data['actions'][action]['text']
    letters = {text.strip(): key for key, text in _OPTION.findall(prompt)}
    letter = letters.get(wanted)
    return list(responses).index(letter) if letter in responses else None

  def text(self, prompt: str) -> str | None:
    if toy_lib.ORDER_MARKER in prompt:
      return 'Build; if a leak opens, repair it.'
    return None
