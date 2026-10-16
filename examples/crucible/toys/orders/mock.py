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
    r'Stores: (?P<supplies>\d+) supplies; hull damage (?P<damage>\d+);'
    r' project \d+[^.\n]*\. \w+ is (?P<cond>worn out|fresh)\.'
)
_AGE = re.compile(r"This is the order's turn (\d+)\.")
_SPAN = re.compile(r'\bfor\s+(\w+)\s+turns?\b', re.IGNORECASE)
_THEN = re.compile(r'\bthen\b', re.IGNORECASE)
_NUMBERS = {
    w: n
    for n, w in enumerate(
        'one two three four five six seven eight nine ten eleven twelve'.split(),
        1,
    )
} | {'a': 1}
_SOMEONE = re.compile(
    r'^(\w+) takes up the clause for someone this turn', re.MULTILINE
)
_RELAY = re.compile(r'\w+ is the bosun, who is (\w+):')
_COMMANDER_ORDER = re.compile(r'The commander\'s order: "(.*)"\n')
_OPTION = re.compile(r'^\s*\(([a-z])\) (.+)$', re.MULTILINE)
_WORD = re.compile(r'[a-z]+')
_TOKEN = re.compile(r'[a-z]+|,')
_SENTENCE = re.compile(r'[.;]')
_CLAUSE = re.compile(r'\b(?:otherwise|else|then)\b', re.IGNORECASE)


def _timed(order: str, prompt: str) -> str:
  """The part of a timed order that holds now, by the order's age.

  A span covers its own clause ("build for two turns", "for two turns,
  build") and the one it hands over to ("then repair", in the same clause or
  the next); other clauses hold throughout (cycle 81).

  Args:
    order: The order as the member heard it.
    prompt: The reading prompt, which gives the order's turn.

  Returns:
    The clauses that hold this turn, or `order` itself if it has no span.
  """
  found = _AGE.search(prompt)
  clauses = [c.strip() for c in _SENTENCE.split(order) if c.strip()]
  if not found or not any(_SPAN.search(c) for c in clauses):
    return order
  age = int(found.group(1))
  kept, i = [], 0
  while i < len(clauses):
    clause, i = clauses[i], i + 1
    span = _SPAN.search(clause)
    if not span:
      kept.append(clause)
      continue
    word = span.group(1).lower()
    turns = int(word) if word.isdigit() else _NUMBERS.get(word, 1)
    this, *then = _THEN.split(clause, 1)
    if not then and i < len(clauses) and _THEN.match(clauses[i]):
      then, i = _THEN.split(clauses[i], 1)[1:], i + 1
    this = _SPAN.sub('', this).strip(' ,')
    now = this if age <= turns else ''.join(then).strip(' ,')
    if now:
      kept.append(now)
  return '; '.join(kept) + '.' if kept else ''


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
    self._negation_ends = frozenset(lex.get('negation_ends', ()))
    self._condition_negations = self._negations | frozenset(
        lex.get('condition_negations', ())
    )
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
        (piece, part)
        for sentence in _SENTENCE.split(order)
        for piece in self._by_name(sentence)
        for part in _CLAUSE.split(piece)
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
      verbs, nots, condition = self._asks(clause)
      wants = {k for k, vs in self._conditions.items() if vs & set(condition)}
      key = None
      if conditional:
        holds = bool(wants & now)
        if 'unless' in condition:
          holds = not holds
        if self._condition_negations & set(condition):
          holds = not holds  # "If there is no leak, build."
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

  def _by_name(self, part: str) -> list[str]:
    """Splits "Ash builds, Bea keeps watch" at a comma before a name.

    Only where both sides ask for something: "Ash, keep watch" and "Ash,
    Bea, build" stay whole (cycle 84).

    Args:
      part: One sentence of an order.

    Returns:
      The clauses in it, in order.
    """
    names = '|'.join(re.escape(n) for n in sorted(self._crew))
    if not names:
      return [part]
    pieces = re.split(rf',\s*(?=(?:{names})\b)', part, flags=re.IGNORECASE)
    out = []
    for piece in pieces:
      if out and not any(self._asks(out[-1])[:2]):
        out[-1] += ', ' + piece  # the one before asked nothing: a vocative
      else:
        out.append(piece)
    if len(out) > 1 and not any(self._asks(out[-1])[:2]):
      out[-2:] = [out[-2] + ', ' + out[-1]]
    return out

  def _asks(self, clause: str) -> tuple[list[str], list[str], list[str]]:
    """What one clause asks for and forbids, and its condition's words.

    A negation forbids the actions after it up to a comma, "but" or
    "instead" ("Don't rest, build."); a condition's words are not orders.

    Args:
      clause: One clause of an order.

    Returns:
      The actions asked, the actions forbidden, and the condition's tokens
      (empty for a plain clause).
    """
    said = _TOKEN.findall(clause.lower())
    condition = []
    if self._if & set(said):
      said, condition = self._condition(said)
    verbs, nots, negated = [], [], False
    for w in said:
      if w == ',' or w in self._negation_ends:
        negated = False
      elif w in self._negations:
        negated = True
      else:
        for a, vs in self._verbs.items():
          if w in vs:
            (nots if negated else verbs).append(a)
    return verbs, nots, condition

  def _condition(self, tokens: list[str]) -> tuple[list[str], list[str]]:
    """Splits a conditional clause into (its orders, its condition).

    The condition runs from its if-word to the next comma ("If there is no
    leak, build"; "Build if there is a leak, quickly"), or without a comma to
    its last condition word ("Rest when tired"; "During storms build"). Its
    own words are not orders: "If tired, build" asks for building, not rest.

    Args:
      tokens: The clause's words and commas, lower case.

    Returns:
      The tokens outside the condition, and the condition's tokens.
    """
    start = next(i for i, w in enumerate(tokens) if w in self._if)
    if ',' in tokens[start:]:
      end = tokens.index(',', start)
    else:
      named = [
          i
          for i, w in enumerate(tokens)
          if i > start and any(w in vs for vs in self._conditions.values())
      ]
      end = named[-1] + 1 if named else start + 1
    return tokens[:start] + tokens[end:], tokens[start:end]

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

  def conditions_now(
      self, situation: str, worn: bool, damage: int = 0
  ) -> set[str]:
    """The conditions an order's clauses can name that hold this turn."""
    text = situation.lower()
    now = {k for k in ('leak', 'storm', 'squall') if k in text}
    if worn:
      now.add('tired')
    if damage > 0:
      now.add('damaged')
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
    now = self.conditions_now(situation, worn, damage)
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
        and 'repair' not in forbidden  # a forbidding clause wins (cycle 78)
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
    supplies, damage, cond = stores.group('supplies', 'damage', 'cond')
    letters = {text.strip(): key for key, text in _OPTION.findall(prompt)}
    by_text = {
        spec['text']: name for name, spec in self._data['actions'].items()
    }
    menu = [by_text[t] for t in letters if t in by_text]
    action = self.decide(
        props, _timed(order.group(1), prompt), turn.group(1), int(supplies), int(damage),
        cond == 'worn out', who.group(1), menu, _someone(prompt),
    )  # fmt: skip
    wanted = self._data['actions'][action]['text']
    letter = letters.get(wanted)
    return list(responses).index(letter) if letter in responses else None

  def relay(self, order: str, how: str) -> str:
    """How a bosun passes `order` on, by their temperament's `relays`."""
    if how == 'word_for_word':
      return order
    parts = [p.strip() for p in _SENTENCE.split(order) if p.strip()]

    def asks(part, action):
      return action in self._asks(part)[0]

    if how == 'drops_rest':
      parts = [p for p in parts if not (asks(p, 'rest') or asks(p, 'watch'))]
    elif how == 'adds_repair':
      # Unless a clause already says whether to repair (cycle 83).
      if not any('repair' in sum(self._asks(p)[:2], []) for p in parts):
        parts.append('repair leaks')
    elif how == 'negates':

      def negated(part):
        if self._negations & set(_WORD.findall(part.lower())):
          return part
        # After whom it addresses: "Cy, don't keep watch", "everyone else,
        # don't build".
        who = re.match(
            r'^((?:everyone(?: else)?|\w+(?:,? and \w+)*),?)\s+(.*)$', part
        )
        if who:
          names = set(_WORD.findall(who.group(1).lower()))
          if names <= self._crew | {'and', 'everyone', 'else'}:
            return f"{who.group(1).rstrip(',')}, don't {who.group(2)}"
        return "don't " + part[0].lower() + part[1:]

      parts = [negated(p) for p in parts]
    return '; '.join(parts) + '.' if parts else ''

  def text(self, prompt: str) -> str | None:
    found = _RELAY.search(prompt)
    if toy_lib.RELAY_MARKER in prompt and found:
      props = self._data['temperaments'].get(found.group(1), {})
      order = _COMMANDER_ORDER.search(prompt)
      return self.relay(
          order.group(1) if order else '',
          props.get('relays', 'word_for_word'),
      )
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
    worn = bool(stores and stores.group('cond') == 'worn out')
    damage = int(stores.group('damage')) if stores else 0
    now = self.conditions_now(turn.group(1), worn, damage)
    _, _, _, clauses = self.read_clauses(
        _timed(order.group(1), prompt), now, who.group(1), _someone(prompt)
    )
    return f'"{"; ".join(clauses)}"' if clauses else 'none'
