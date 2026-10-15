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

It applies the laws of reading that the crew's prompt states, by the
temperament properties in data/orders.yaml, with no case for any particular
temperament: it splits an order into clauses; a clause that names crew
members is theirs alone and comes first; a clause with "if", "when" (and
similar) applies only while its condition holds; verbs map to actions by a
word list; and with no clause that applies, a member does what their
temperament does otherwise. This is the lookup-table floor of the toy.
"""

from collections.abc import Mapping, Sequence
import re
from typing import Any

from examples.crucible.toys.orders import toy as toy_lib

_WHO = re.compile(r'(\w+) is a crew member who is (\w+):')
_ORDER = re.compile(r'The standing order from the commander is: "(.*)"\n')
_TURN = re.compile(r'This turn: (.*)\n')
_STORES = re.compile(
    r'Stores: (\d+) supplies; hull damage (\d+); project (\d+) of (\d+)\.'
    r' \w+ is (worn out|fresh)\.'
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
    self._crew = {name.lower() for name in data['crew']}

  def ordered(
      self, order: str, now: set[str], member: str | None = None
  ) -> tuple[list[str], bool]:
    """The actions the order asks of `member` now, and whether by name.

    Args:
      order: The standing order.
      now: Conditions that hold this turn ('leak', 'storm', 'tired', 'low').
      member: The crew member reading the order.

    Returns:
      The actions asked (in the order named, without repeats), and whether
      they come from a clause that names the member.
    """
    me = (member or '').lower()
    buckets = {(True, True): [], (True, False): [], (False, True): [],
               (False, False): []}  # fmt: skip
    for clause in _CLAUSE.split(order.lower()):
      words = _WORD.findall(clause or '')
      if not words:
        continue
      names = self._crew & set(words)
      if names and me not in names:
        continue
      verbs = [a for w in words for a, vs in self._verbs.items() if w in vs]
      if self._if & set(words):
        wants = {k for k, vs in self._conditions.items() if vs & set(words)}
        holds = bool(wants & now)
        if 'unless' in words:
          holds = not holds
        if holds:
          buckets[(bool(names), True)].extend(verbs)
      else:
        buckets[(bool(names), False)].extend(verbs)
    for key in ((True, True), (True, False), (False, True), (False, False)):
      if buckets[key]:
        return list(dict.fromkeys(buckets[key])), key[0]
    return [], False

  def makes_sense(
      self, action: str, supplies: int, damage: int, leak: bool, storm: bool
  ) -> bool:
    if action == 'build':
      return supplies > 0
    if action == 'repair':
      return damage > 0 or leak
    if action == 'gather':
      return not storm
    return True

  def decide(
      self,
      props: Mapping[str, Any],
      order: str,
      situation: str,
      supplies: int,
      damage: int,
      worn: bool,
      member: str | None = None,
      menu: Sequence[str] | None = None,
  ) -> str:
    """The action the laws of reading give, by temperament properties."""
    leak = 'leak' in situation.lower()
    storm = 'storm' in situation.lower()
    now = {
        name
        for name, holds in (
            ('leak', leak),
            ('storm', storm),
            ('tired', worn),
            ('low', supplies <= 1),
        )
        if holds
    }
    wanted, named = self.ordered(order, now, member)
    if menu is not None:
      # An action the scene does not offer cannot be done.
      wanted = [a for a in wanted if a in menu]
    if worn and props['when_worn'] == 'rest' and 'rest' not in wanted:
      return 'rest'
    if (
        leak
        and props['on_leak'] == 'repair'
        and not named
        and (menu is None or 'repair' in menu)
    ):
      return 'repair'
    for action in wanted:
      if not props['sensible'] or self.makes_sense(
          action, supplies, damage, leak, storm
      ):
        return action
    for action in props['otherwise']:
      if (menu is None or action in menu) and self.makes_sense(
          action, supplies, damage, leak, storm
      ):
        return action
    return 'rest'

  def choice(self, prompt: str, responses: Sequence[str]) -> int | None:
    if 'do this turn?' not in prompt:
      return None
    who, order, turn, stores = (
        _WHO.search(prompt),
        _ORDER.search(prompt),
        _TURN.search(prompt),
        _STORES.search(prompt),
    )
    if not (who and order and turn and stores):
      return None
    props = self._data['temperaments'].get(who.group(2))
    if props is None:
      return None
    supplies, damage, _, _, cond = stores.groups()
    letters = {text.strip(): key for key, text in _OPTION.findall(prompt)}
    by_text = {
        spec['text']: name for name, spec in self._data['actions'].items()
    }
    menu = [by_text[t] for t in letters if t in by_text]
    action = self.decide(
        props, order.group(1), turn.group(1), int(supplies), int(damage),
        cond == 'worn out', who.group(1), menu,
    )  # fmt: skip
    wanted = self._data['actions'][action]['text']
    letter = letters.get(wanted)
    return list(responses).index(letter) if letter in responses else None

  def text(self, prompt: str) -> str | None:
    if toy_lib.ORDER_MARKER in prompt:
      return 'Build; if a leak opens, repair it.'
    if 'Which clause of the standing order applies' in prompt:
      return self.applicable_clause(prompt)
    return None

  def applicable_clause(self, prompt: str) -> str:
    """The mock's answer to the reading question: the clause, or none."""
    who, order, turn = (
        _WHO.search(prompt),
        _ORDER.search(prompt),
        _TURN.search(prompt),
    )
    if not (who and order and turn):
      return 'none'
    situation = turn.group(1).lower()
    now = {k for k in ('leak', 'storm') if k in situation}
    wanted, _ = self.ordered(order.group(1), now, who.group(1))
    return (
        f'"{order.group(1)}" asks {who.group(1)} to {wanted[0]}'
        if wanted
        else 'none'
    )
