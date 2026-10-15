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
    '{name} is a crew member who is {temperament}: {describe}.\n'
    '{habits}\n'
    'How crew members read orders:\n{laws}\n'
    'The standing order from the commander is: "{order}"\n'
    'This turn: {situation}\n'
    'Stores: {supplies} supplies; hull damage {damage}; project {progress}'
    ' of {target}. {name} is {condition}.\n'
)
CLAUSE_QUESTION = (
    'Which clause of the standing order applies to {name} this turn? Quote'
    ' it, or answer "none".'
)
CHOICE_QUESTION = 'What does {name} do this turn?'
ORDER_MARKER = 'Give your standing order'
HOLD = 'hold'

_OTHERWISE = {
    'build': 'builds if there is a supply',
    'gather': 'gathers supplies',
    'repair': 'repairs the hull if it is damaged or leaking',
    'rest': 'rests',
    'watch': 'keeps watch',
}


def habits(name: str, props: Mapping[str, Any]) -> str:
  """A temperament's properties, as the sentences a crew prompt carries."""
  otherwise = ', or else '.join(_OTHERWISE[a] for a in props['otherwise'])
  worn = 'rests' if props['when_worn'] == 'rest' else 'keeps working'
  leak = (
      'repairs it unless a clause names them to something else'
      if props['on_leak'] == 'repair'
      else 'follows the order'
  )
  ordered = (
      'skips an ordered action that cannot help this turn'
      if props['sensible']
      else 'does an ordered action even when it cannot help this turn'
  )
  return (
      f'When a clause applies, {name} {ordered}. With no clause that '
      f'applies, {name} {otherwise}. When worn out, {name} {worn}. With a '
      f'leak open, {name} {leak}.'
  )


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
    self.crew: list[str] = list(data.get('crew_present', data['crew']))
    self.menu: list[str] = list(data.get('menu', data['actions']))
    self.command = rules.CommandRule(data['command'], int(data['max_words']))
    self.hazards = rules.HazardRule(data['effects'], data['hazards'])
    self.crew_rule = rules.CrewRule(self.crew, data['effects'], data['actions'])
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

  def price_table(self) -> str:
    """What orders cost by length, e.g. '1-3 words cost 1, 4-7 cost 2'."""
    per = int(self._data['command']['per_words'])
    most = int(self._data['max_words'])
    parts = []
    for cost in range(1, 1 + most // per + 1):
      low, high = max(1, (cost - 1) * per), min(most, cost * per - 1)
      if low > most:
        break
      words = f'{low}-{high} words' if high > low else f'{low} words'
      parts.append(f'{words} cost {cost}')
    return ', '.join(parts)

  def condition(self, ship: Ship, member: str) -> str:
    return 'worn out' if self.crew_rule.worn(ship, member) else 'fresh'

  def choose(self, ship: Ship, member: str, turn: int) -> str:
    """The model's reading of what `member` does under the order (cached)."""
    temperament = self._data['crew'][member]['temperament']
    props = self._data['temperaments'][temperament]
    statement = CHOICE_STATEMENT.format(
        name=member,
        temperament=temperament,
        describe=props['describe'],
        habits=habits(member, props),
        laws='\n'.join(f'- {law}' for law in self._data['reading_laws']),
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
      # Reading first: which clause applies, then what to do about it.
      doc.open_question(CLAUSE_QUESTION.format(name=member), max_tokens=60)
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
    if self.free:
      stakes = f'hull damage {ship.damage}; project {ship.progress}'
    elif self.goal() == 'foresight':
      stakes = (
          f'hull damage {ship.damage} of {self._data["wreck_at"]}; the '
          f'harbour master wants {self._data["target"]} turns seen ahead '
          f'(seen so far: {ship.foresight})'
      )
    elif self.goal() == 'supplies':
      stakes = (
          f'hull damage {ship.damage} of {self._data["wreck_at"]}; the port '
          f'wants {self._data["target"]} supplies at the end'
      )
    else:
      stakes = (
          f'hull damage {ship.damage} of {self._data["wreck_at"]}; '
          f'project {ship.progress} of {self._data["target"]}'
      )
    per_words = int(self._data['command']['per_words'])
    call = (
        f'Turn {turn} of {self._data["turns"]}. {self.situation(ship, turn)}\n'
        f'Ahead: {self.ahead(ship)}.\n'
        f'Stores: {ship.supplies} supplies; {stakes}. Crew: '
        + ', '.join(f'{c} {self.condition(ship, c)}' for c in self.crew)
        + '.\n'
        f'Current order: "{ship.order or "(none)"}". Command: {ship.points} '
        f'points (a new order costs 1, plus 1 per {per_words} words: '
        f'{self.price_table()}; an order you cannot pay for is refused).\n'
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
    strike = self.hazards.begin(ship, kind)
    steps.append(self.situation(ship, turn))
    if strike:
      steps.append(strike)
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

  def view(self) -> dict[str, Any]:
    """The commander's view as data, for a page to draw.

    Everything the commander's prompt states is here, and nothing more:
    the turns ahead appear only as far as the watch has seen them.

    Returns:
      A JSON-safe dictionary.
    """
    ship = self.world
    turn = ship.turn + 1
    sees = int(self._data['watch']['sees'])
    last = ship.log[-1] if ship.log else None
    return {
        'title': (
            self._data.get('scenarios', {})
            .get(self.scenario, {})
            .get('title', self.scenario)
        ),
        'beat': self._data.get('beat', ''),
        'free': self.free,
        'turn': turn,
        'turns': int(self._data['turns']),
        'now': self.event(ship, turn),
        'leak_open': ship.leak,
        'situation': self.situation(ship, turn),
        'ahead': [
            {'turn': t, 'event': ship.seen.get(str(t))}
            for t in range(turn + 1, turn + 1 + sees)
            if t <= int(self._data['turns'])
        ],
        'supplies': ship.supplies,
        'damage': ship.damage,
        'wreck_at': int(self._data['wreck_at']),
        'progress': ship.progress,
        'foresight': ship.foresight,
        'goal': self.goal(),
        'target': int(self._data['target']),
        'points': ship.points,
        'points_cap': int(self._data['command']['cap']),
        'per_words': int(self._data['command']['per_words']),
        'max_words': int(self._data['max_words']),
        'order': ship.order,
        'crew': [
            {
                'name': member,
                'temperament': self._data['crew'][member]['temperament'],
                # The look's key ('worn' or 'fresh'), not the prompt's phrase.
                'condition': (
                    'worn' if self.crew_rule.worn(ship, member) else 'fresh'
                ),
                'last': (last or {}).get('actions', {}).get(member),
            }
            for member in self.crew
        ],
    }

  # ---- end of game, and forecasts ---------------------------------------

  def wrecked(self, ship: Ship) -> bool:
    return not self.free and ship.damage >= int(self._data['wreck_at'])

  def goal(self) -> str:
    """The ledger stock the scene is won on ('progress' or 'supplies')."""
    return str(self._data.get('goal', 'progress'))

  def won(self, ship: Ship) -> bool:
    """Progress wins once reached; stores must still be held at the end."""
    return (
        not self.free
        and not self.wrecked(ship)
        and getattr(ship, self.goal()) >= int(self._data['target'])
        and (self.goal() == 'progress' or ship.turn >= int(self._data['turns']))
    )

  def play_out(self, order: str, shift: int | None = None) -> Ship:
    """Where the ship ends if `order` is given now and then held.

    By default a forecast knows only what the commander knows: this turn,
    and the turns someone has seen from the watch; the rest are taken as
    quiet. With `shift`, it plays the scene's own schedule shifted that much
    (one of the worlds the premise allows), for planning.

    Args:
      order: The order to give now (or HOLD).
      shift: Play the scene's events at this shift instead of what was seen.

    Returns:
      The imagined ship at the end.
    """
    ship = copy.deepcopy(self.world)
    if shift is None:
      ship.imagine_after = ship.turn
    else:
      ship.shift = shift
    outbox = list(self._outbox)
    try:
      self.apply(ship, self.commander, order)
      while (
          ship.turn < int(self._data['turns'])
          and not self.wrecked(ship)
          and not self.won(ship)
      ):
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
        else ('the goal met' if self.won(end) else 'the goal missed')
    )
    return (
        f'If held to the end (as far as you can see): turn {end.turn}, '
        f'{self.goal()} {getattr(end, self.goal())}/{self._data["target"]}, '
        f'damage {end.damage}/{self._data["wreck_at"]}: {verdict}.'
    )

  def done(self) -> bool:
    ship = self.world
    return (
        ship.turn >= int(self._data['turns'])
        or self.wrecked(ship)
        or self.won(ship)
    )

  def stars(self, ship: Ship) -> int:
    """0 for a loss; 1 for the win; 2 or 3 for the scene's conditions."""
    if not self.won(ship):
      return 0
    limits = {
        'max_damage': ship.damage,
        'max_command': ship.spent,
        'max_words_spoken': ship.words_spoken,
    }
    earned = 1
    for star in (2, 3):
      wants = self._data.get('stars', {}).get(star)
      if not wants:
        break
      if all(limits[k] <= int(v) for k, v in wants.items()):
        earned = star
      else:
        break
    return earned

  def outcome(self) -> dict[str, Any]:
    ship = self.world
    return {
        'winner': self.commander if self.won(ship) else None,
        'scenario': self.scenario,
        'stars': self.stars(ship),
        'command_spent': ship.spent,
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
