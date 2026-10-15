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
when command points allow. Each turn, a language model reads the order for
each crew member, against that member's temperament and what is happening
now, and picks the member's action from a fixed menu. The rule components in
`rules.py` apply the consequences in a fixed order: the order's price, the
world's hazards, the crew's actions and what a watch reveals.

One order carries several pressures at once: what to do (directive), what it
costs to say (cost), who keeps watch (information), when clauses apply
(timing), and what each temperament makes of words it was not given (side
effects).
"""

from collections.abc import Mapping
import copy
import random
from typing import Any

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from examples.crucible.toys import base
from examples.crucible.toys.orders import rules

Ship = rules.Ship

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
    self.command = rules.CommandRule(data['command'], int(data['max_words']))
    self.hazards = rules.HazardRule(data['effects'])
    self.crew_rule = rules.CrewRule(self.crew, data['effects'])
    self.watch = rules.WatchRule(int(data['watch']['sees']))
    start = data['start']
    self.world = Ship(
        supplies=int(start['supplies']),
        damage=int(start['damage']),
        progress=int(start['progress']),
        points=self.command.start(),
        fatigue={c: 0 for c in self.crew},
        shift=random.Random(seed).randint(0, 2),
    )
    self._cache: dict[str, int] = {}
    self.tell(self.commander, str(data['premise']).strip())

  # ---- what is known --------------------------------------------------------

  def event(self, ship: Ship, turn: int) -> str:
    """The event of a turn (1-based): the truth, or in a forecast the seen."""
    if 0 <= ship.imagine_after < turn - 1:
      return ship.seen.get(str(turn), 'quiet')
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

  def ahead(self, ship: Ship) -> str:
    """What the commander knows of the turns after this one."""
    turn = ship.turn + 1
    known = [
        f'turn {t}: {ship.seen[str(t)]}'
        for t in range(turn + 1, turn + 1 + int(self._data['watch']['sees']))
        if str(t) in ship.seen
    ]
    return '; '.join(known) if known else 'unknown (no one has kept watch)'

  def condition(self, ship: Ship, member: str) -> str:
    return 'worn out' if self.crew_rule.worn(ship, member) else 'fresh'

  def choose(self, ship: Ship, member: str, turn: int) -> str:
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
    return self.menu[self._cache[statement]]

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
    per_words = int(self._data['command']['per_words'])
    call = (
        f'Turn {turn} of {self._data["turns"]}. {self.situation(ship, turn)}\n'
        f'Ahead: {self.ahead(ship)}.\n'
        f'Stores: {ship.supplies} supplies; {stakes}. Crew: '
        + ', '.join(f'{c} {self.condition(ship, c)}' for c in self.crew)
        + '.\n'
        f'Current order: "{ship.order or "(none)"}". Command: {ship.points} '
        f'points (a new order costs 1, plus 1 per {per_words} words).\n'
        f'{ORDER_MARKER}, {{name}}: a new order of at most '
        f'{self._data["max_words"]} words, or "{HOLD}" to keep the current '
        'one.'
    )
    return entity_lib.free_action_spec(call_to_action=call, tag='order')

  def apply(self, world: Ship, actor: str, action: str) -> base.Receipt:
    ship = world
    steps = []
    new_order = action.strip()
    if new_order and new_order.lower().rstrip('.') not in (HOLD, base.PASS):
      steps.append(self.command.issue(ship, new_order))
    ship.turn += 1
    turn = ship.turn
    kind = self.event(ship, turn)
    self.hazards.begin(ship, kind)
    steps.append(self.situation(ship, turn))
    before = (ship.progress, ship.damage, ship.supplies)
    chosen = self.crew_rule.act(
        ship, kind, lambda member: self.choose(ship, member, turn)
    )
    steps.append(
        ' '.join(
            f'{m} {self._data["actions"][chosen[m]]["does"]}.'
            for m in self.crew
        )
    )
    seen = self.watch.reveal(ship, chosen, lambda t: self.event(ship, t))
    if seen:
      steps.append(seen)
    bleed = self.hazards.end(ship)
    if bleed:
      steps.append(bleed)
    self.command.regenerate(ship)
    steps.append(
        f'Project {ship.progress}/{self._data["target"]}, damage '
        f'{ship.damage}/{self._data["wreck_at"]}, supplies {ship.supplies}, '
        f'command {ship.points}.'
    )
    ship.log.append({'turn': turn, 'event': kind, 'actions': chosen})
    delta = {
        'progress': float(ship.progress - before[0]),
        'damage': float(ship.damage - before[1]),
        'supplies': float(ship.supplies - before[2]),
    }
    return base.Receipt(actor, new_order or HOLD, ' '.join(steps), steps, delta)

  # ---- end of game, and forecasts ---------------------------------------

  def wrecked(self, ship: Ship) -> bool:
    return not self.free and ship.damage >= int(self._data['wreck_at'])

  def won(self, ship: Ship) -> bool:
    return (
        not self.free
        and not self.wrecked(ship)
        and ship.progress >= int(self._data['target'])
    )

  def play_out(self, order: str) -> Ship:
    """Where the ship ends if `order` is given now and then held.

    A forecast knows only what the commander knows: this turn, and the
    turns someone has seen from the watch; the rest are taken as quiet.

    Args:
      order: The order to give now (or HOLD).

    Returns:
      The imagined ship at the end.
    """
    ship = copy.deepcopy(self.world)
    ship.imagine_after = ship.turn
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
        f'If held to the end (as far as you can see): turn {end.turn}, '
        f'project {end.progress}/{self._data["target"]}, damage '
        f'{end.damage}/{self._data["wreck_at"]}: {verdict}.'
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
    }

  # ---- state ------------------------------------------------------------

  def world_to_dict(self) -> Mapping[str, Any]:
    return self.world.to_dict()

  def world_from_dict(self, data: Mapping[str, Any]) -> None:
    self.world = Ship.from_dict(data)
