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
| `CrewRule` | directive: what each member does, as a model reads the order | supplies, progress, repairs, fatigue |
| `WatchRule` | information: a member on watch reveals the turns ahead | `seen` |

The toy (`toy.py`) runs them in a fixed order each turn.
"""

from collections.abc import Callable, Mapping, Sequence
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
  fatigue: dict[str, int] = dataclasses.field(default_factory=dict)
  shift: int = 0
  # Events the commander has seen ahead, by turn (str keys: JSON-safe).
  seen: dict[str, str] = dataclasses.field(default_factory=dict)
  # When >= 0, the ship is imagined (a forecast): turns after this one use
  # only what was seen, quiet otherwise.
  imagine_after: int = -1
  log: list[dict[str, Any]] = dataclasses.field(default_factory=list)

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
    return f'New order (cost {price}): "{order}".'

  def regenerate(self, ship: Ship) -> None:
    ship.points = min(self._cap, ship.points + self._regen)


class HazardRule:
  """The world: an event opens a leak or blows a storm; leaks bleed."""

  def __init__(self, effects: Mapping[str, Any]):
    self._leak_damage = int(effects['leak_damage'])

  def begin(self, ship: Ship, kind: str) -> None:
    if kind == 'leak':
      ship.leak = True

  def end(self, ship: Ship) -> str:
    if not ship.leak:
      return ''
    ship.damage += self._leak_damage
    return f'The open leak adds {self._leak_damage} damage.'


class CrewRule:
  """Directive: each member's action (a model's reading), then its effects.

  Effects apply in the fixed crew order, so two builds that compete for the
  last supply always resolve the same way.
  """

  def __init__(self, crew: Sequence[str], effects: Mapping[str, Any]):
    self._crew = list(crew)
    self._effects = effects

  def worn(self, ship: Ship, member: str) -> bool:
    return ship.fatigue[member] >= int(self._effects['tired_at'])

  def act(
      self,
      ship: Ship,
      kind: str,
      choose: Callable[[str], str],
  ) -> dict[str, str]:
    """Decides every member's action first, then applies them in order."""
    chosen = {member: choose(member) for member in self._crew}
    effects = self._effects
    for member in self._crew:
      act = chosen[member]
      worn = self.worn(ship, member)
      if act == 'repair':
        ship.damage = max(0, ship.damage - int(effects['repair_damage']))
        ship.leak = False
        ship.fatigue[member] += 1
      elif act == 'gather':
        if kind != 'storm':
          ship.supplies += int(effects['gather_supplies'])
        ship.fatigue[member] += 1
      elif act == 'build':
        if ship.supplies >= int(effects['build_cost']):
          ship.supplies -= int(effects['build_cost'])
          gain = int(effects['build_progress'])
          ship.progress += gain // 2 if worn else gain
        ship.fatigue[member] += 1
      elif act == 'rest':
        ship.fatigue[member] = max(
            0, ship.fatigue[member] - int(effects['rest_recovery'])
        )
      # 'watch' costs nothing here; WatchRule acts on it.
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
    if not watchers:
      return ''
    ahead = []
    for turn in range(ship.turn + 1, ship.turn + 1 + self._sees):
      kind = event(turn)
      ship.seen[str(turn)] = kind
      ahead.append(f'turn {turn}: {kind}')
    return f'{" and ".join(watchers)} on watch see ahead: {", ".join(ahead)}.'
