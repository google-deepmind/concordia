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

"""The orders toy's ledger and its rule components.

One input, the order, carries several pressures at once (GUIDE.md, Phase 2).
Each pressure is its own rule component with its own test, and they meet only
in the ledger (`Ship`):

| Component | Pressure | Owns |
|---|---|---|
| `CommandRule` | cost: speaking costs command points, by length | `points`, `order` |
| `HazardRule` | the world: leaks open and bleed, storms blow | `leak`, `damage` from leaks |
| `CrewRule` | directive: what each member does, as a model reads the order; who takes up a clause for "someone" | supplies, progress, repairs, fatigue |
| `WatchRule` | information: a member on watch reveals the turns ahead | `seen`, `watched` |

The rota (who is off duty on which turn) is scene data the toy reads.

The toy (`toy.py`) runs them in a fixed order each turn.
"""

from collections.abc import Callable, Collection, Mapping, Sequence
import dataclasses
from typing import Any


@dataclasses.dataclass
class Ship:
  """The ledger."""

  turn: int = 0
  supplies: int = 0
  damage: int = 0
  progress: int = 0
  leak: bool = False
  order: str = ''
  points: int = 0
  spent: int = 0  # command points spent on orders so far
  words_spoken: int = 0  # words in all the orders given
  fatigue: dict[str, int] = dataclasses.field(default_factory=dict)
  shift: int = 0
  # Events the commander has seen ahead, by turn (str keys: JSON-safe).
  seen: dict[str, str] = dataclasses.field(default_factory=dict)
  # When >= 0, the ship is imagined (a forecast): turns after this one use
  # only what was seen, quiet otherwise.
  imagine_after: int = -1
  watched: bool = False  # someone kept watch last turn
  # Members hurt by a squall: the last turn they are off duty (cycle 21).
  injured: dict[str, int] = dataclasses.field(default_factory=dict)
  log: list[dict[str, Any]] = dataclasses.field(default_factory=list)

  @property
  def foresight(self) -> int:
    """How many turns ahead the commander has seen."""
    return len(self.seen)

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> 'Ship':
    return cls(**{k: data[k] for k in data})


class CommandRule:
  """Cost: a new order costs points by its length; points come back."""

  def __init__(self, command: Mapping[str, Any], max_words: int):
    self._points = int(command['points'])
    self._regen = int(command['regen'])
    self._cap = int(command['cap'])
    self._per_words = int(command['per_words'])
    self._max_words = max_words

  def start(self) -> int:
    return self._points

  def cost(self, order: str) -> int:
    return 1 + len(order.split()) // self._per_words

  def issue(self, ship: Ship, order: str) -> str:
    """Tries to replace the standing order; returns what happened."""
    words = len(order.split())
    if words > self._max_words:
      return (
          f'The order is {words} words, over the limit of {self._max_words}:'
          ' the old order stands.'
      )
    price = self.cost(order)
    if price > ship.points:
      return (
          f'The order costs {price} command and you have {ship.points}: the'
          ' old order stands.'
      )
    ship.order = order
    ship.points -= price
    ship.spent += price
    ship.words_spoken += words
    return f'New order (cost {price}): "{order}".'

  def regenerate(self, ship: Ship) -> None:
    ship.points = min(self._cap, ship.points + self._regen)


class HazardRule:
  """The world: events by their data; an open leak bleeds.

  One law reads every event (data/orders.yaml `hazards`): it sets the
  flags the event `opens`, and does its `strikes_unwatched` damage when no
  one kept watch the turn before.
  """

  def __init__(self, effects: Mapping[str, Any], hazards: Mapping[str, Any]):
    self._leak_damage = int(effects['leak_damage'])
    self._hazards = hazards

  def begin(self, ship: Ship, kind: str) -> str:
    spec = self._hazards.get(kind, {})
    for flag in spec.get('opens', ()):
      setattr(ship, flag, True)
    strike = int(spec.get('strikes_unwatched', 0))
    if strike and not ship.watched:
      ship.damage += strike
      return f'Unwatched, it strikes: {strike} damage.'
    if strike:
      return 'The watch saw it coming: the ship weathers it.'
    return ''

  def strikes(self, ship: Ship, kind: str) -> bool:
    """Whether `kind` strikes this turn (it does damage when unwatched)."""
    spec = self._hazards.get(kind, {})
    return bool(int(spec.get('strikes_unwatched', 0))) and not ship.watched

  def injure(
      self,
      ship: Ship,
      kind: str,
      turn: int,
      on_duty: Sequence[str],
  ) -> str:
    """An event that `injures` hurts the most worn member on duty.

    The tired are the careless: the member with the highest fatigue (the
    first in crew order on a tie) is off duty this turn and for the rest of
    the `injures` turns.

    Args:
      ship: The ledger.
      kind: The event that struck.
      turn: This turn.
      on_duty: The members on duty, in crew order.

    Returns:
      What happened, or '' when the event hurts no one.
    """
    turns = int(self._hazards.get(kind, {}).get('injures', 0))
    if not turns or not on_duty:
      return ''
    hurt = max(on_duty, key=lambda m: ship.fatigue[m])
    ship.injured[hurt] = turn + turns - 1
    return f'{hurt} is hurt: off duty for {turns} turns.'

  def end(self, ship: Ship) -> str:
    if not ship.leak:
      return ''
    ship.damage += self._leak_damage
    return f'The open leak adds {self._leak_damage} damage.'


class CrewRule:
  """Directive: each member's action (a model's reading), then its effects.

  One law applies every action, from its data (data/orders.yaml `actions`):
  an action that lacks what it `needs` does nothing; an event in its
  `spoiled_by` voids its gains; a worn-out member's gains are scaled by
  `worn`; then `gives`, `fatigue` and `closes` apply. Effects apply in the
  fixed crew order, so two builds that compete for the last supply always
  resolve the same way.
  """

  def __init__(
      self,
      crew: Sequence[str],
      effects: Mapping[str, Any],
      actions: Mapping[str, Any],
  ):
    self._crew = list(crew)
    self._effects = effects
    self._actions = actions

  def worn(self, ship: Ship, member: str) -> bool:
    return ship.fatigue[member] >= int(self._effects['tired_at'])

  def _apply(self, ship: Ship, member: str, action: str, kind: str) -> None:
    spec = self._actions.get(action, {})
    needs = spec.get('needs', {})
    if all(getattr(ship, stock) >= amount for stock, amount in needs.items()):
      for stock, amount in needs.items():
        setattr(ship, stock, getattr(ship, stock) - amount)
      if kind not in spec.get('spoiled_by', ()):
        scale = float(spec.get('worn', 1.0)) if self.worn(ship, member) else 1.0
        for stock, amount in spec.get('gives', {}).items():
          setattr(
              ship, stock, max(0, getattr(ship, stock) + int(amount * scale))
          )
      for flag in spec.get('closes', ()):
        setattr(ship, flag, False)
    ship.fatigue[member] = max(
        0, ship.fatigue[member] + int(spec.get('fatigue', 0))
    )

  def volunteer(
      self,
      ship: Ship,
      takes: Mapping[str, str],
      excluded: Collection[str] = (),
  ) -> str | None:
    """Who takes up a clause for "someone": the freshest willing member.

    Args:
      ship: The ledger (fatigue).
      takes: Each member's willingness: 'always', 'fresh' (only when not
        worn out) or 'never'.
      excluded: Members who do not take it up: those named in another
        clause, and those off duty.

    Returns:
      The member with the lowest fatigue among the willing (the first in
      crew order on a tie), or None when no one is willing.
    """
    willing = [
        m
        for m in self._crew
        if m not in excluded
        and (
            takes.get(m) == 'always'
            or (takes.get(m) == 'fresh' and not self.worn(ship, m))
        )
    ]
    return min(willing, key=lambda m: ship.fatigue[m], default=None)

  def act(
      self,
      ship: Ship,
      kind: str,
      choose: Callable[[str], str],
  ) -> dict[str, str]:
    """Decides every member's action first, then applies them in order."""
    chosen = {member: choose(member) for member in self._crew}
    for member in self._crew:
      self._apply(ship, member, chosen[member], kind)
    return chosen


class WatchRule:
  """Information: if anyone kept watch, the next turns are revealed."""

  def __init__(self, sees: int):
    self._sees = sees

  def reveal(
      self,
      ship: Ship,
      chosen: Mapping[str, str],
      event: Callable[[int], str],
  ) -> str:
    watchers = [m for m, act in chosen.items() if act == 'watch']
    ship.watched = bool(watchers)
    if not watchers:
      return ''
    ahead = []
    for turn in range(ship.turn + 1, ship.turn + 1 + self._sees):
      kind = event(turn)
      ship.seen[str(turn)] = kind
      ahead.append(f'turn {turn}: {kind}')
    return f'{" and ".join(watchers)} on watch see ahead: {", ".join(ahead)}.'
