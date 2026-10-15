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

"""Toy C, "orders": one standing order, read anew by a crew each turn.

The commander (the player) gives one short standing order and may replace it
a few times. Each turn, a language model reads the order for each crew
member, against that member's temperament and what is happening this turn,
and picks the member's action from a fixed menu. The ledger applies the
actions: repairs close leaks, gathering fills the stores, building spends
supplies on progress, and work wears people out.

One order has to hold up across situations the commander did not list, as
three different temperaments understand it.
"""

from collections.abc import Mapping
import copy
import dataclasses
import random
from typing import Any

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from examples.crucible.toys import base

CHOICE_STATEMENT = (
    '{name} is a crew member who is {temperament}.\n'
    'The standing order from the commander is: "{order}"\n'
    'This turn: {situation}\n'
    'Stores: {supplies} supplies; hull damage {damage}; project {progress}'
    ' of {target}. {name} is {condition}.\n'
)
CHOICE_QUESTION = 'What does {name} do this turn?'
ORDER_MARKER = 'Give your standing order'
HOLD = 'hold'


@dataclasses.dataclass
class Ship:
  """The ledger."""

  turn: int = 0
  supplies: int = 0
  damage: int = 0
  progress: int = 0
  leak: bool = False
  order: str = ''
  revisions_left: int = 0
  fatigue: dict[str, int] = dataclasses.field(default_factory=dict)
  shift: int = 0
  log: list[dict[str, Any]] = dataclasses.field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> 'Ship':
    return cls(**{k: data[k] for k in data})


class OrdersToy(base.Toy):
  """A crew interprets one standing order, turn by turn."""

  name = 'orders'

  def __init__(
      self,
      model: language_model.LanguageModel,
      data: Mapping[str, Any],
      scenario: str = 'voyage',
      seed: int = 0,
  ):
    super().__init__()
    self._model = model
    if scenario not in data.get('scenarios', {}):
      scenario = data.get('default_scenario', 'voyage')
    self._data = {**data, **data.get('scenarios', {}).get(scenario, {})}
    data = self._data
    self.scenario = scenario
    self.free = bool(data.get('free', False))
    self.commander: str = data['commander']
    self.crew: list[str] = list(data['crew'])
    self.menu: list[str] = list(data['actions'])
    start = data['start']
    self.world = Ship(
        supplies=int(start['supplies']),
        damage=int(start['damage']),
        progress=int(start['progress']),
        revisions_left=int(data['revisions']),
        fatigue={c: 0 for c in self.crew},
        shift=random.Random(seed).randint(0, 2),
    )
    self._cache: dict[str, int] = {}
    self.tell(self.commander, str(data['premise']).strip())

  # ---- the ledger's view ------------------------------------------------

  def event(self, ship: Ship, turn: int) -> str:
    """The event of a turn (1-based), shifted by the seed."""
    events = {int(k): v for k, v in self._data['events'].items()}
    index = turn - ship.shift
    cycle = self._data.get('events_cycle')
    if cycle and index > 0:
      index = (index - 1) % int(cycle) + 1
    return events.get(index, 'quiet')

  def situation(self, ship: Ship, turn: int) -> str:
    kind = self.event(ship, turn)
    text = self._data['situations'][kind]
    if ship.leak and kind != 'leak':
      text += ' A leak is still open in the hull.'
    return text

  def condition(self, ship: Ship, member: str) -> str:
    tired = int(self._data['effects']['tired_at'])
    return 'worn out' if ship.fatigue[member] >= tired else 'fresh'

  def choose(self, ship: Ship, member: str, turn: int) -> int:
    """The model's reading of what `member` does under the order (cached)."""
    statement = CHOICE_STATEMENT.format(
        name=member,
        temperament=self._data['crew'][member]['temperament'],
        order=ship.order or '(no order given)',
        situation=self.situation(ship, turn),
        supplies=ship.supplies,
        damage=ship.damage,
        progress=ship.progress,
        target=self._data['target'],
        condition=self.condition(ship, member),
    )
    if statement not in self._cache:
      doc = interactive_document.InteractiveDocument(self._model)
      doc.statement(statement)
      self._cache[statement] = doc.multiple_choice_question(
          CHOICE_QUESTION.format(name=member),
          [self._data['actions'][a]['text'] for a in self.menu],
          randomize_choices=False,
      )
    return self._cache[statement]

  # ---- turn order and moves ---------------------------------------------

  def actors(self) -> list[str]:
    return [self.commander]

  def max_steps(self) -> int:
    return int(self._data['turns'])

  def action_spec(self, actor: str) -> entity_lib.ActionSpec:
    ship = self.world
    turn = ship.turn + 1
    stakes = (
        f'hull damage {ship.damage}; project {ship.progress}'
        if self.free
        else (
            f'hull damage {ship.damage} of {self._data["wreck_at"]}; '
            f'project {ship.progress} of {self._data["target"]}'
        )
    )
    call = (
        f'Turn {turn} of {self._data["turns"]}. {self.situation(ship, turn)}\n'
        f'Stores: {ship.supplies} supplies; {stakes}. Crew: '
        + ', '.join(f'{c} {self.condition(ship, c)}' for c in self.crew)
        + '.\n'
        f'Current order: "{ship.order or "(none)"}". New orders left: '
        f'{ship.revisions_left}.\n'
        f'{ORDER_MARKER}, {{name}}: a new order of at most '
        f'{self._data["max_words"]} words, or "{HOLD}" to keep the current '
        'one.'
    )
    return entity_lib.free_action_spec(call_to_action=call, tag='order')

  def apply(self, world: Ship, actor: str, action: str) -> base.Receipt:
    ship = world
    effects = self._data['effects']
    steps = []
    new_order = action.strip()
    if new_order and new_order.lower().rstrip('.') not in (HOLD, base.PASS):
      words = new_order.split()
      if len(words) > int(self._data['max_words']):
        steps.append(
            f'The order is {len(words)} words, over the limit of '
            f'{self._data["max_words"]}: the old order stands.'
        )
      elif ship.revisions_left <= 0:
        steps.append('No new orders left: the old order stands.')
      else:
        ship.order = new_order
        ship.revisions_left -= 1
        steps.append(f'New order: "{new_order}".')
    ship.turn += 1
    turn = ship.turn
    kind = self.event(ship, turn)
    if kind == 'leak':
      ship.leak = True
    steps.append(self.situation(ship, turn))
    chosen = {m: self.menu[self.choose(ship, m, turn)] for m in self.crew}
    before = (ship.progress, ship.damage, ship.supplies)
    for member in self.crew:
      act = chosen[member]
      worn = ship.fatigue[member] >= int(effects['tired_at'])
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
      else:
        ship.fatigue[member] = max(
            0, ship.fatigue[member] - int(effects['rest_recovery'])
        )
    steps.append(
        ' '.join(
            f'{m} {self._data["actions"][chosen[m]]["does"]}.'
            for m in self.crew
        )
    )
    if ship.leak:
      ship.damage += int(effects['leak_damage'])
      steps.append(f'The open leak adds {effects["leak_damage"]} damage.')
    steps.append(
        f'Project {ship.progress}/{self._data["target"]}, damage '
        f'{ship.damage}/{self._data["wreck_at"]}, supplies {ship.supplies}.'
    )
    ship.log.append({'turn': turn, 'event': kind, 'actions': chosen})
    delta = {
        'progress': float(ship.progress - before[0]),
        'damage': float(ship.damage - before[1]),
        'supplies': float(ship.supplies - before[2]),
    }
    return base.Receipt(actor, new_order or HOLD, ' '.join(steps), steps, delta)

  # ---- end of game ------------------------------------------------------

  def wrecked(self, ship: Ship) -> bool:
    return not self.free and ship.damage >= int(self._data['wreck_at'])

  def won(self, ship: Ship) -> bool:
    return (
        not self.free
        and not self.wrecked(ship)
        and ship.progress >= int(self._data['target'])
    )

  def play_out(self, order: str) -> Ship:
    """The ship at the end if `order` is given now and then held."""
    ship = copy.deepcopy(self.world)
    outbox = list(self._outbox)
    try:
      self.apply(ship, self.commander, order)
      while ship.turn < int(self._data['turns']) and not self.wrecked(ship):
        self.apply(ship, self.commander, HOLD)
    finally:
      self._outbox = outbox
    return ship

  def forecast(self, order: str) -> str:
    """Where `order` leads if held to the end, in one line."""
    if self.free:
      return ''
    end = self.play_out(order)
    verdict = (
        'wrecked'
        if self.wrecked(end)
        else ('the project done' if self.won(end) else 'the project unfinished')
    )
    return (
        f'If held to the end: turn {end.turn}, project {end.progress}/'
        f'{self._data["target"]}, damage {end.damage}/'
        f'{self._data["wreck_at"]}: {verdict}.'
    )

  def done(self) -> bool:
    ship = self.world
    return ship.turn >= int(self._data['turns']) or self.wrecked(ship)

  def outcome(self) -> dict[str, Any]:
    ship = self.world
    return {
        'winner': self.commander if self.won(ship) else None,
        'scenario': self.scenario,
        'score': float(ship.progress - 2 * ship.damage),
        'progress': ship.progress,
        'damage': ship.damage,
        'turns': ship.turn,
        'orders_given': int(self._data['revisions']) - ship.revisions_left,
    }

  # ---- state ------------------------------------------------------------

  def world_to_dict(self) -> Mapping[str, Any]:
    return self.world.to_dict()

  def world_from_dict(self, data: Mapping[str, Any]) -> None:
    self.world = Ship.from_dict(data)
