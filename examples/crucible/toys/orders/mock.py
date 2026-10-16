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
_SOMEONE = re.compile(
    r'^(\w+) takes up the clause for someone this turn', re.MULTILINE
)
_OPTION = re.compile(r'^\s*\(([a-z])\) (.+)$', re.MULTILINE)
_WORD = re.compile(r'[a-z]+')
_SENTENCE = re.compile(r'[.;]')
_CLAUSE = re.compile(r'\b(?:otherwise|else|then)\b', re.IGNORECASE)


def _someone(prompt: str) -> str | None:
  """Who takes up a clause for "someone" this turn, as the reading says."""
  found = _SOMEONE.search(prompt)
  return found.group(1) if found else None


class OrdersMock:
  """Answers the orders toy's prompts deterministically."""

  def __init__(self, data: Mapping[str, Any]):
    self._data = data
    lex = data['lexicon']
    self._verbs = {a: frozenset(lex[a]) for a in data['actions']}
    self._conditions = {k: frozenset(v) for k, v in lex['conditions'].items()}
    self._if = frozenset(lex['if_words'])
    self._crew = {name.lower() for name in data['crew']}
    self._negations = frozenset(lex.get('negations', ()))
    self._someone = [
        re.compile(rf'\b{re.escape(w)}\b')
        for w in data.get('someone', {}).get('words', ())
    ]

  def read(
      self,
      order: str,
      now: set[str],
      member: str | None = None,
      someone: str | None = None,
  ) -> tuple[list[str], bool, list[str]]:
    """What the order asks of `member` now: wanted, by name, forbidden."""
    return self.read_clauses(order, now, member, someone)[:3]

  def read_clauses(
      self,
      order: str,
      now: set[str],
      member: str | None = None,
      someone: str | None = None,
  ) -> tuple[list[str], bool, list[str], list[str]]:
    """What the order asks of `member` now, and the clauses that ask it.

    Clauses that name crew members apply to them alone and come first; a
    conditional clause applies only while its condition holds; an action
    named after "not", "never" or "don't" is forbidden, not wanted.

    Args:
      order: The standing order.
      now: Conditions that hold this turn ('leak', 'storm', 'squall', 'tired').
      member: The crew member reading the order.
      someone: The member who takes up a clause for "someone" this turn, if
        anyone does.

    Returns:
      The actions asked (in the order named, without repeats), whether they
      come from a clause that names the member, the actions forbidden, and
      the clauses (as written) that asked or forbade them.
    """
    me = (member or '').lower()
    buckets = {(True, True): [], (True, False): [], (False, True): [],
               (False, False): []}  # fmt: skip
    forbidden = {key: [] for key in buckets}
    sources = {key: [] for key in buckets}
    parts = [
        (sentence, part)
        for sentence in _SENTENCE.split(order)
        for part in _CLAUSE.split(sentence)
    ]
    for written, part in parts:
      clause = part.lower()
      words = _WORD.findall(clause)
      if not words:
        continue
      names = self._crew & set(words)
      if not names and any(p.search(clause) for p in self._someone):
        # A clause for someone names the one member who takes it up.
        names = {someone.lower()} if someone else {''}
      if names and me not in names:
        continue
      conditional = bool(self._if & set(words))
      wants = {k for k, vs in self._conditions.items() if vs & set(words)}
      # In a conditional clause, the condition's own words are not orders:
      # "If tired, build" asks for building, not rest.
      said = (
          [w for w in words if not any(w in self._conditions[k] for k in wants)]
          if conditional
          else words
      )
      verbs, nots, negated = [], [], False
      for w in said:
        if w in self._negations:
          negated = True
          continue
        for a, vs in self._verbs.items():
          if w in vs:
            (nots if negated else verbs).append(a)
      key = None
      if conditional:
        holds = bool(wants & now)
        if 'unless' in words:
          holds = not holds
        if holds:
          key = (bool(names), True)
      else:
        key = (bool(names), False)
      if key is not None:
        buckets[key].extend(verbs)
        forbidden[key].extend(nots)
        if verbs or nots:
          sources[key].append(written.strip())
    for key in ((True, True), (True, False), (False, True), (False, False)):
      if buckets[key] or forbidden[key]:
        return (
            list(dict.fromkeys(buckets[key])),
            key[0],
            list(dict.fromkeys(forbidden[key])),
            list(dict.fromkeys(sources[key])),
        )
    return [], False, [], []

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

  def conditions_now(self, situation: str, worn: bool) -> set[str]:
    """The conditions an order's clauses can name that hold this turn."""
    text = situation.lower()
    now = {k for k in ('leak', 'storm', 'squall') if k in text}
    if worn:
      now.add('tired')
    return now

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
      someone: str | None = None,
  ) -> str:
    """The action the laws of reading give, by temperament properties."""
    leak = 'leak' in situation.lower()
    storm = 'storm' in situation.lower()
    now = self.conditions_now(situation, worn)
    wanted, named, forbidden = self.read(order, now, member, someone)
    if props.get('contrary'):
      # The opposite of what applies: the forbidden is done; work becomes
      # rest and rest becomes work.
      if forbidden:
        wanted = forbidden[:1]
      elif wanted:
        wanted = [props['works'] if 'rest' in wanted else 'rest']
      forbidden = []
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
      if (
          (menu is None or action in menu)
          and action not in forbidden
          and self.makes_sense(action, supplies, damage, leak, storm)
      ):
        return action
    return 'rest'

  def choice(self, prompt: str, responses: Sequence[str]) -> int | None:
    if toy_lib.CHOICE_QUESTION.split('{name}')[1] not in prompt:
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
        cond == 'worn out', who.group(1), menu, _someone(prompt),
    )  # fmt: skip
    wanted = self._data['actions'][action]['text']
    letter = letters.get(wanted)
    return list(responses).index(letter) if letter in responses else None

  def text(self, prompt: str) -> str | None:
    if toy_lib.ORDER_MARKER in prompt:
      return 'Build; if a leak opens, repair it.'
    if toy_lib.CLAUSE_QUESTION.split('{name}')[0] in prompt:
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
    stores = _STORES.search(prompt)
    worn = bool(stores and stores.group(5) == 'worn out')
    now = self.conditions_now(turn.group(1), worn)
    _, _, _, clauses = self.read_clauses(
        order.group(1), now, who.group(1), _someone(prompt)
    )
    return f'"{"; ".join(clauses)}"' if clauses else 'none'
