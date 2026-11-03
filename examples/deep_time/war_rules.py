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

"""Rules that play a people where conquest is played (stage 6, C8).

Each is a `rule_act.Rule`: a people's acting component (`RuleAct`) answers
its question by it, reading only the question and what the people's own
components say this turn (its sheet: its arms, its restless or courted
towns, the people named as holding much of the map, where every town
stands). The World GM resolves the answer like anyone's. They are bots for
measuring the game (a road to conquest each, a skilled mix, the others'
answer to a leader) and the computer's peoples in a mock game that asks for
war (`conquest_bots`); characters played by live minds never use them.

Between moves of war each falls back on the mock's skilled policies for a
people (feed it, then its industry), so a road is measured against the same
economy.
"""

from collections.abc import Mapping, Sequence
import re
from typing import Any

from examples.deep_time import actors as actors_lib
from examples.deep_time import mock_language_model as mock_lib
from concordia.typing import entity as entity_lib

_OPTION = re.compile(r'^\s*\(([a-z]+)\)\s(.+)$')
_STRENGTH = re.compile(r'Our arms: [^(]*\(strength (\d+(?:\.\d+)?)')
_RESTLESS = re.compile(r'holds to us at (\d+(?:\.\d+)?) \(settled at')
_COURTED_US = re.compile(r'court our town[^:]*: [^(]*\((\d+)% to go\)')
_TO_GO = re.compile(r'Our envoys court a town[^:]*: [^(]*\((\d+)% to go\)')
_LEADER = re.compile(r'The ([A-Z][\w ]*?) hold (\d+) of the (\d+) towns\.')
_US_LEADING = re.compile(r'We hold \d+ of the \d+ towns')
_THREAT = re.compile(r'An army of the ([\w ]+?) \([^)]*\) marches to')
_GRIEVANCE = re.compile(r'We hold grievances against the ([\w ]+?) \(')
_GRAIN = re.compile(r'\bgrain (\d+(?:\.\d+)?)')
_OTHERS = re.compile(r'towns of others: (.*?)\.(?:\s|$)')
_OURS = re.compile(r'Our towns stand at (.*?) \(row,col\)')
_TILE = re.compile(r'(\d+),(\d+)(?: \(([^)]*)\))?')
_SEEN_ARMS = re.compile(r'arms (\d+(?:\.\d+)?)')
# What a town's people count for with no arms, as a rule guesses it (the
# Great Acceleration's militia; a poorer world's is less, so it is wary).
_TOWN_GUARD = 1.5
_CAPITAL_GUARD = 3.0
# A town on a sheet: row, column, capital, the arms seen in it.
Town = tuple[int, int, bool, float]
# Arms, the strongest first (each needs its dual-use know-how).
_ARMS = (
    'forge rifles',
    'breed horses',
    'train archers',
    'muster axemen',
    'arm levies',
)


def _moves(spec: entity_lib.ActionSpec) -> list[str]:
  """The moves a question offers: its options, or a free one's list."""
  if spec.options:
    return list(spec.options)
  return [
      m.group(2).strip()
      for m in map(_OPTION.match, spec.call_to_action.splitlines())
      if m
  ]


def _find(moves: Sequence[str], start: str) -> str | None:
  return next((m for m in moves if m.startswith(start)), None)


def _number(pattern: re.Pattern[str], text: str) -> float | None:
  found = pattern.findall(text)
  return float(found[-1]) if found else None


def _tiles(text: str) -> list[Town]:
  """The towns a sheet lists: each tile, and what its note says."""
  towns = []
  for r, c, note in _TILE.findall(text):
    arms = _SEEN_ARMS.search(note)
    towns.append((
        int(r),
        int(c),
        'capital' in note,
        float(arms.group(1)) if arms else 0.0,
    ))
  return towns


def _others(sheet: str) -> dict[str, list[Town]]:
  """The other peoples' towns, from the sheet, by people."""
  found = _OTHERS.findall(sheet)
  peoples = {}
  for part in (found[-1] if found else '').split('; '):
    named = re.match(r'the (.+?) at (.*)$', part)
    if named:
      peoples[named.group(1)] = _tiles(named.group(2))
  return peoples


def _towns_of(sheet: str, people: str) -> list[Town]:
  """Another people's towns, from the sheet."""
  return _others(sheet).get(people, [])


def _own_towns(sheet: str) -> list[Town]:
  """Its own towns, from the sheet."""
  found = _OURS.findall(sheet)
  return _tiles(found[-1] if found else '')


def _economy(
    spec: entity_lib.ActionSpec,
    contexts: Mapping[str, str],
    moves: Sequence[str],
) -> str:
  """The mock's skilled choice for a people (feed it, then its industry)."""
  prompt = '\n'.join([*contexts.values(), spec.call_to_action])
  options = {chr(ord('a') + i): m for i, m in enumerate(moves[:26])}
  for policy in mock_lib.DEFAULT_POLICIES:
    if policy is mock_lib.found_policy:  # Its towns found (`_found`).
      continue
    choice = policy(prompt, options)
    if choice is not None and choice in moves:
      return mock_lib.with_words(prompt, choice)
  return _find(moves, actors_lib.KEEP_LABOR) or (moves[-1] if moves else '')


class _PeopleRule:
  """A people's rule: a move of war when it has one, else its economy."""

  def __init__(self, strike: float = 8.0, rest: int = 3):
    self._turn = 0
    self._armed_at = -rest
    self._strike = strike  # The strength it marches with.
    self._rest = rest  # Turns between shifts of labor into arms.

  def __call__(
      self, spec: entity_lib.ActionSpec, contexts: Mapping[str, str]
  ) -> str:
    self._turn += 1
    moves = _moves(spec)
    sheet = '\n'.join(contexts.values())
    people = self.war(sheet, moves) or _economy(spec, contexts, moves)
    # Its towns' own orders ride with it, a line each (stage 6, C6): a new
    # town settled from its capital whenever it can, then the rule's own.
    return '\n'.join(
        [people, *self._found(sheet, moves), *self.orders(sheet, moves)]
    )

  def _found(self, sheet: str, moves: Sequence[str]) -> list[str]:
    """A new town, settled from its capital, whenever it can found one."""
    found = _find(moves, actors_lib.FOUND_TOWN)
    capital = next((t for t in _own_towns(sheet) if t[2]), None)
    if found is None or capital is None:
      return []
    return [f'town {capital[0]},{capital[1]}: {found}']

  def war(self, sheet: str, moves: Sequence[str]) -> str | None:
    """Its move of war this turn, or None."""
    del sheet, moves
    return None

  def orders(self, sheet: str, moves: Sequence[str]) -> list[str]:
    """Its towns' orders this turn (`"town row,col: <move> @ row,col"`)."""
    del sheet, moves
    return []

  def _court(
      self, sheet: str, moves: Sequence[str], reserve: float
  ) -> list[str]:
    """A cart from each of its towns to one town, while grain lasts.

    The town courted is the first not a capital of the people it courts
    (kept until it joins), the people with the fewest towns first.
    """
    envoys = _find(moves, actors_lib.SEND_ENVOYS)
    grain = _number(_GRAIN, sheet) or 0.0
    reserve = min(reserve, grain / 2)  # A poorer people keeps half.
    if envoys is None or grain < reserve + 5.0:
      return []
    courtable = []
    for towns in _others(sheet).values():
      tiles = [(r, c) for r, c, cap, _ in towns if not cap]
      if tiles:
        courtable.append((len(towns), tiles))
    if not courtable:
      return []
    # A town hears a people's envoys once a turn: a cart to each of the
    # weakest people's towns, then the next people's.
    targets = [t for _, tiles in sorted(courtable) for t in tiles]
    carts = int((grain - reserve) // 5.0)
    return [
        f'town {tr},{tc}: {envoys} @ {r},{c}'
        for (tr, tc, *_), (r, c) in zip(_own_towns(sheet)[:carts], targets)
    ]

  # Moves of war, shared by the rules.
  def _feast(self, sheet: str, moves: Sequence[str]) -> str | None:
    """A feast in a taken town about to rise."""
    loyalty = _number(_RESTLESS, sheet)
    if loyalty is not None and loyalty < 0.3:
      return _find(moves, actors_lib.HOLD_FEAST)
    return None

  def _target(
      self, sheet: str, on: str | None = None
  ) -> tuple[tuple[int, int], float] | None:
    """The weakest town (of `on`, else of anyone) and the arms to take it.

    About twice the defence its map shows (its arms, and its people), and
    never less than the strength the rule strikes with.
    """
    peoples = _others(sheet)
    towns = (
        peoples.get(on, []) if on else [t for v in peoples.values() for t in v]
    )
    if not towns:
      return None

    def need(town: Town) -> float:
      guard = _CAPITAL_GUARD if town[2] else _TOWN_GUARD
      return max(self._strike, 2.0 * (town[3] + guard))

    r, c, _, _ = town = min(towns, key=lambda t: (need(t), t[2], t[0], t[1]))
    return (r, c), need(town)

  def _arm(
      self, sheet: str, moves: Sequence[str], on: str | None = None
  ) -> str | None:
    """More hands into arms, while weaker than its target needs."""
    strength = _number(_STRENGTH, sheet) or 0.0
    target = self._target(sheet, on)
    need = target[1] if target else self._strike
    if strength >= need or self._turn - self._armed_at < self._rest:
      return None
    for kind in _ARMS:
      move = next((m for m in moves if kind in m), None)
      if move is not None:
        self._armed_at = self._turn
        return move
    return None

  def _march(
      self, sheet: str, moves: Sequence[str], on: str | None = None
  ) -> str | None:
    """A march on the weakest town (of `on`, else of anyone), when strong.

    Strong enough is about twice the defence its map shows (`_target`).
    """
    strength = _number(_STRENGTH, sheet) or 0.0
    march = _find(moves, actors_lib.MARCH_TAKE)
    target = self._target(sheet, on)
    if march is None or target is None or strength < target[1]:
      return None
    (r, c), _ = target
    return f'{march} @ {r},{c}'


class Sword(_PeopleRule):
  """The sword: arms, marches when strong, feasts what is about to rise."""

  def war(self, sheet: str, moves: Sequence[str]) -> str | None:
    return (
        self._feast(sheet, moves)
        or self._march(sheet, moves)
        or self._arm(sheet, moves)
    )


class Bread(_PeopleRule):
  """Bread: a cart from every town to one town while the granary allows.

  Its people's own order is its economy (or a feast at home).
  """

  def war(self, sheet: str, moves: Sequence[str]) -> str | None:
    return self._feast(sheet, moves)

  def orders(self, sheet: str, moves: Sequence[str]) -> list[str]:
    return self._court(sheet, moves, reserve=30.0)


class Conqueror(_PeopleRule):
  """A skilled mix: the sword, and bread from a rich granary.

  Its people's order feasts what is about to rise, marches when strong,
  arms when not; meanwhile its towns court other towns with the grain it
  can spare (and, like every rule, found a town whenever it can).
  """

  def war(self, sheet: str, moves: Sequence[str]) -> str | None:
    return (
        self._feast(sheet, moves)
        or self._march(sheet, moves)
        or self._arm(sheet, moves)
    )

  def orders(self, sheet: str, moves: Sequence[str]) -> list[str]:
    return self._court(sheet, moves, reserve=60.0)


class Rival(_PeopleRule):
  """The others' answer: arm against a leader, an army coming, a wrong.

  When one people holds half the map, or marches on its towns, or wronged
  it (its sheet says so), it arms and marches on that people's towns.
  Otherwise its economy.
  """

  def __init__(self, strike: float = 8.0):
    super().__init__(strike=strike, rest=4)

  def war(self, sheet: str, moves: Sequence[str]) -> str | None:
    courted = [int(n) for n in _COURTED_US.findall(sheet)]
    if courted and min(courted) <= 50:  # A town of ours about to be won.
      feast = _find(moves, actors_lib.HOLD_FEAST)
      if feast is not None:
        return feast
    if _US_LEADING.search(sheet):
      return self._feast(sheet, moves)
    foes = [
        *(name for name, _, _ in _LEADER.findall(sheet)),
        *_THREAT.findall(sheet),
        *_GRIEVANCE.findall(sheet),
    ]
    if not foes:
      return self._feast(sheet, moves)
    return (
        self._feast(sheet, moves)
        or self._march(sheet, moves, foes[0])
        or self._arm(sheet, moves, foes[0])
    )


class Random(_PeopleRule):
  """Any move it has, by a fixed hash of what it reads (a baseline)."""

  def __call__(
      self, spec: entity_lib.ActionSpec, contexts: Mapping[str, str]
  ) -> str:
    moves = _moves(spec)
    text = '\n'.join([*contexts.values(), spec.call_to_action])
    return moves[mock_lib.stable_index('random bot\n' + text, len(moves))]


class Passive(_PeopleRule):
  """Never changes how its people work (a baseline)."""

  def __call__(
      self, spec: entity_lib.ActionSpec, contexts: Mapping[str, str]
  ) -> str:
    del contexts
    return _find(_moves(spec), actors_lib.KEEP_LABOR) or ''


def settings(
    scenario: Mapping[str, Any], rival: bool = False
) -> dict[str, float]:
  """The rules' settings in a world (`conquest_bots: {strike: ...}`).

  The strength a rule marches with suits the world's peoples (a poorer
  age raises fewer arms), and the others' answer may wait for more
  (`rival_strike`); `conquest_bots: true` keeps the defaults.

  Args:
    scenario: The world's data.
    rival: For the others' answer (`Rival`), else a people's own rule.

  Returns:
    Keyword arguments for the rule.
  """
  found = scenario.get('conquest_bots')
  if not isinstance(found, Mapping):
    return {}
  key = 'rival_strike' if rival and 'rival_strike' in found else 'strike'
  return {'strike': float(found[key])} if key in found else {}


def make(
    name: str, scenario: Mapping[str, Any], rival: bool = False
) -> '_PeopleRule':
  """A rule (`RULES[name]`, or the others' `Rival`) set for a world."""
  found = settings(scenario, rival=rival)
  rule = Rival if rival else RULES[name]
  return rule(strike=found['strike']) if 'strike' in found else rule()


# The bots of the people audit, by name.
RULES = {
    'conqueror': Conqueror,
    'sword': Sword,
    'bread': Bread,
    'economy': _PeopleRule,
    'random': Random,
    'passive': Passive,
}
