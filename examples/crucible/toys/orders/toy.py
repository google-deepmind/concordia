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
import re
from typing import Any

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.utils import concurrency
from examples.crucible.toys import base
from examples.crucible.toys.orders import readings
from examples.crucible.toys.orders import rules

Ship = rules.Ship

# The words live in readings.py; these names stay importable from here.
CHOICE_STATEMENT = readings.CHOICE_STATEMENT
TIMED = readings.TIMED
SOMEONE_YOU = readings.SOMEONE_YOU
SOMEONE_OTHER = readings.SOMEONE_OTHER
NO_ONE = readings.NO_ONE
CLAUSE_QUESTION = readings.CLAUSE_QUESTION
CHOICE_QUESTION = readings.CHOICE_QUESTION
ORDER_MARKER = readings.ORDER_MARKER
RELAY_STATEMENT = readings.RELAY_STATEMENT
RELAY_MARKER = readings.RELAY_MARKER
HOLD = readings.HOLD
OFF_DUTY = readings.OFF_DUTY
OFF_DUTY_ACTION = readings.OFF_DUTY_ACTION
clause_text = readings.clause_text
habits = readings.habits


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
    self.rota = rules.RotaRule(self.crew, data.get('off_duty', {}))
    start = data['start']
    self.world = Ship(
        supplies=int(start['supplies']),
        damage=int(start['damage']),
        progress=int(start['progress']),
        points=self.command.start(),
        fatigue={c: 0 for c in self.crew},
        shift=random.Random(seed).randint(0, 2),
        # A scene may open just after a watch (its first squall weathered).
        watched=bool(start.get('watched', False)),
    )
    self._cache: dict[str, tuple[int, str]] = {}
    self._relayed: dict[str, str] = {}
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
    off = self.rota.off_on(ship, turn)
    if off:
      text += (
          f' {" and ".join(off)} {"is" if len(off) == 1 else "are"} off duty.'
      )
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

  def asks_someone(self, order: str) -> bool:
    """Whether the order has a clause for "someone" (arc 4)."""
    text = order.lower()
    return any(
        re.search(rf'\b{re.escape(w)}\b', text)
        for w in self._data.get('someone', {}).get('words', ())
    )

  def volunteer(self, ship: Ship, turn: int) -> str | None:
    """Who takes up the order's clause for "someone" this turn (arc 4).

    Counting fatigue is the ledger's work, not the reader's: the toy finds
    the freshest willing member and every reading is told who it is.

    Args:
      ship: The ship.
      turn: The turn being read (who is off duty).

    Only members on duty whose heard order has a clause for someone, and who
    are not named in it, can take it up; one of them does, the same one for
    every reader (through a bosun the crew hear other words than the bosun,
    cycle 77).

    Returns:
      The member, or None when no one on duty is willing.
    """

    def out(member: str) -> bool:
      heard = self.rota.order_of(ship, member)
      return (
          self.rota.off_duty(ship, member, turn)
          or not self.asks_someone(heard)
          or member.lower() in set(re.findall(r'[a-z]+', heard.lower()))
      )

    excluded = [m for m in self.crew if out(m)]
    takes = {
        m: (
            self._data['temperaments'][
                self._data['crew'][m]['temperament']
            ].get('volunteers', 'never')
        )
        for m in self.crew
    }
    return self.crew_rule.volunteer(ship, takes, excluded)

  def bosun_of(self, ship: Ship) -> str | None:
    """The bosun: named by order (where allowed) or the scene's."""
    return ship.bosun or self._data.get('bosun')

  def appointed(self, order: str) -> str | None:
    """A bosun named in `order` ('Bea is bosun'), '' for 'no bosun', or None."""
    text = order.lower()
    if re.search(r'\bno bosun\b', text):
      return ''
    for member in self.crew:
      if re.search(rf'\b{member.lower()}\b[^;.]*\bbosun\b', text):
        return member
    return None

  def relay(self, order: str, bosun: str) -> str:
    """How `bosun` passes `order` on to the rest of the crew (cached)."""
    temperament = self._data['crew'][bosun]['temperament']
    props = self._data['temperaments'][temperament]
    law = self._data['relaying']['laws'][props.get('relays', 'word_for_word')]
    statement = RELAY_STATEMENT.format(
        name=bosun,
        temperament=temperament,
        describe=props['describe'],
        law=law,
        order=order,
    )
    if statement not in self._relayed:
      doc = interactive_document.InteractiveDocument(self._model)
      doc.statement(statement)
      words = doc.open_question(
          ' '.join(str(self._data['relaying']['question']).split()).format(
              name=bosun
          ),
          max_tokens=60,
          temperature=float(self._data.get('reading_temperature', 0.0)),
      )
      relayed = readings.clean_relay(bosun, words)
      # The bosun is held to a little more than the commander's limit.
      limit = int(self._data['max_words']) + 4
      self._relayed[statement] = ' '.join(relayed.split()[:limit])
    return self._relayed[statement]

  def garble(self, ship: Ship, turn: int) -> str:
    """An order given in a storm that garbles carries only its first clause.

    The wind takes the rest: everyone on duty (the bosun too) hears only the
    words before the first ';' or '.'.

    Args:
      ship: The ship; `ship.order` is the order just given.
      turn: The turn it is given for.

    Returns:
      What carried, as a sentence, or ''.
    """
    kind = self.event(ship, turn)
    if not self.hazards.garbles(kind):
      return ''
    first = re.split(r'[;.]', ship.order)[0].strip()
    if not first or first.rstrip('.') == ship.order.rstrip('.'):
      return ''
    carried = first + '.'
    ship.carried = carried
    for member in self.rota.on_duty(ship, turn):
      ship.heard[member] = carried
    return f'In the {kind} only "{carried}" carries.'

  def pass_on(self, ship: Ship, turn: int) -> str:
    """The bosun relays the order just given to the crew on duty (arc 5).

    Args:
      ship: The ship; `ship.order` is the order just given.
      turn: The turn it is given for.

    Returns:
      What the bosun said, as a sentence, or '' without an active bosun.
    """
    bosun = self.bosun_of(ship)
    if not bosun or bosun not in self.crew:
      return ''
    if self.rota.off_duty(ship, bosun, turn):
      ship.relayed = ''
      return f'{bosun} is off duty: the order goes out as you gave it.'
    relayed = self.relay(self.rota.order_of(ship, bosun), bosun)
    ship.relayed = relayed
    for member in self.rota.on_duty(ship, turn):
      if member != bosun:
        ship.heard[member] = relayed
    return f'{bosun} passes it on: "{relayed}".'

  def condition(self, ship: Ship, member: str) -> str:
    return 'worn out' if self.crew_rule.worn(ship, member) else 'fresh'

  def choose(self, ship: Ship, member: str, turn: int) -> str:
    """The model's reading of what `member` does under the order (cached)."""
    return self.reading(ship, member, turn)[0]

  def reading(self, ship: Ship, member: str, turn: int) -> tuple[str, str]:
    """What `member` does under the order, and the clause they acted on.

    The clause is the model's own answer to the reading question (quoted
    from the order, or 'none' when only their temperament applies).

    Args:
      ship: The ship.
      member: The crew member.
      turn: The turn being read.

    Returns:
      (action, clause), cached by the whole situation.
    """
    temperament = self._data['crew'][member]['temperament']
    props = self._data['temperaments'][temperament]
    order = self.rota.order_of(ship, member)
    someone = self.asks_someone(order)
    timed = bool(TIMED.search(order)) and 'timing' in self._data
    laws = list(self._data['reading_laws'])
    if timed:
      laws.append(' '.join(str(self._data['timing']['law']).split()))
    if someone:
      # Right after the plain-clause law, which it qualifies (cycle 32: a
      # model took "Someone repair the hull." for a plain clause).
      laws.insert(1, ' '.join(str(self._data['someone']['law']).split()))
    statement = CHOICE_STATEMENT.format(
        name=member,
        temperament=temperament,
        describe=props['describe'],
        habits=habits(member, props),
        laws='\n'.join(f'- {law}' for law in laws),
        order=order or '(no order given)',
        situation=self.situation(ship, turn),
        supplies=ship.supplies,
        damage=ship.damage,
        stakes=self.crew_stakes(ship),
        condition=self.condition(ship, member),
    )
    if timed:
      # Counted from 1, so "for N turns" has no off-by-one to argue about.
      age = self.rota.age(ship, member, turn)
      statement += f"This is the order's turn {age}.\n"
    if someone:
      who = self.volunteer(ship, turn)
      if who is None:
        statement += NO_ONE.format(name=member)
      elif who == member:
        statement += SOMEONE_YOU.format(who=who)
      else:
        statement += SOMEONE_OTHER.format(who=who, name=member)
    if statement not in self._cache:
      doc = interactive_document.InteractiveDocument(self._model)
      doc.statement(statement)
      # Reading first: which clause applies, then what to do about it.
      clause = doc.open_question(
          CLAUSE_QUESTION.format(name=member),
          max_tokens=60,
          temperature=float(self._data.get('reading_temperature', 0.0)),
      )
      idx = doc.multiple_choice_question(
          CHOICE_QUESTION.format(name=member),
          [self._data['actions'][a]['text'] for a in self.menu],
          randomize_choices=False,
      )
      self._cache[statement] = (idx, clause_text(clause))
    idx, clause = self._cache[statement]
    return self.menu[idx], clause

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
    else:
      stakes = (
          f'hull damage {ship.damage} of {self._data["wreck_at"]};'
          f' {self.aim(ship)}'
      )
    per_words = int(self._data['command']['per_words'])
    call = self.laws_for_commander(ship) + (
        f'Turn {turn} of {self._data["turns"]}. {self.situation(ship, turn)}\n'
        f'Ahead: {self.ahead(ship)}.\n'
        f'Stores: {ship.supplies} supplies; {stakes}. Crew: '
        + ', '.join(f'{c} {self.condition(ship, c)}' for c in self.crew)
        + '.\n'
        f'Current order: "{ship.order or "(none)"}"{self.span(ship, turn)}.'
        f'{self.earlier(ship)}'
        f' Command: {ship.points} '
        f'points (a new order costs 1, plus 1 per {per_words} words: '
        f'{self.price_table()}; an order you cannot pay for is refused).\n'
        f'{ORDER_MARKER}, {{name}}: a new order of at most '
        f'{self._data["max_words"]} words, or "{HOLD}" to keep the current '
        "one. Answer with the order's words alone."
    )
    return entity_lib.free_action_spec(call_to_action=call, tag='order')

  def laws_for_commander(self, ship: Ship) -> str:
    """The laws the crew read by, so the commander can plan with them."""
    laws = list(self._data['reading_laws'])
    for extra in ('someone', 'timing'):
      if extra in self._data:
        spec = self._data[extra]
        law = spec.get('commander_law', spec['law'])
        laws.append(' '.join(str(law).split()))
    bosun = self.bosun_of(ship)
    if bosun:
      temperament = self._data['crew'][bosun]['temperament']
      how = self._data['temperaments'][temperament].get(
          'relays', 'word_for_word'
      )
      laws.append(
          f'Your order goes to {bosun}, the bosun, who '
          f'{self._data["relaying"]["laws"][how]}; the rest of the crew hear'
          ' only that.'
      )
    return (
        'How your crew read orders:\n'
        + '\n'.join(f'- {law}' for law in laws)
        + '\n'
    )

  def span(self, ship: Ship, turn: int) -> str:
    """Which turn of a timed standing order `turn` would be, if kept."""
    if not (ship.order and TIMED.search(ship.order) and 'timing' in self._data):
      return ''
    return f' (if kept, its turn {turn - ship.order_turn + 1})'

  def earlier(self, ship: Ship) -> str:
    """Who still follows an earlier order, for the commander's prompt."""
    bosun = self.bosun_of(ship)
    relayed = ship.relayed if bosun else None
    behind = [
        f'{m} ("{self.rota.order_of(ship, m) or "(none)"}")'
        for m in self.crew
        if self.rota.order_of(ship, m)
        not in (ship.order, relayed, ship.carried)
    ]
    text = f' Still on an earlier order: {", ".join(behind)}.' if behind else ''
    if ship.carried:
      text += f' In the storm only "{ship.carried}" carried.'
    if relayed and ship.order:
      text += f' {bosun} passed it on as: "{relayed}".'
    return text

  def apply(self, world: Ship, actor: str, action: str) -> base.Receipt:
    ship = world
    steps = []
    new_order = action.strip()
    # "hold!", "Hold on" and bare punctuation keep the order too (cycle 88).
    said = ' '.join(re.findall(r'[a-z]+', new_order.lower()))
    if said and said not in (HOLD, base.PASS, f'{HOLD} on'):
      previous, spent, since = ship.order, ship.spent, ship.order_turn
      steps.append(self.command.issue(ship, new_order))
      if ship.spent > spent:  # given: it reaches whoever is on duty
        ship.order_turn = ship.turn + 1
        ship.carried = ''
        if self._data.get('bosun_by_order'):
          named = self.appointed(ship.order)
          if named is not None and named != ship.bosun:
            ship.bosun = named
            if not named:
              ship.relayed = ''
            steps.append(f'{named} is bosun now.' if named else 'No bosun now.')
        missed = self.rota.hear(ship, previous, ship.turn + 1, since)
        if missed:
          steps.append(missed)
        carried = self.garble(ship, ship.turn + 1)
        if carried:
          steps.append(carried)
        relayed = self.pass_on(ship, ship.turn + 1)
        if relayed:
          steps.append(relayed)
    ship.turn += 1
    turn = ship.turn
    kind = self.event(ship, turn)
    struck = self.hazards.strikes(ship, kind)
    strike = self.hazards.begin(ship, kind)
    hurt = (
        self.hazards.injure(ship, kind, turn, self.rota.on_duty(ship, turn))
        if struck
        else ''
    )
    steps.append(self.situation(ship, turn))
    steps.extend(text for text in (strike, hurt) if text)
    before = (ship.progress, ship.damage, ship.supplies)
    # With a live model, every member reads the order at once (the readings
    # are independent); the rule then applies them in the fixed crew order.
    on = self.rota.on_duty(ship, turn)
    if getattr(self._model, 'parallel_safe', False):
      readings = dict(
          concurrency.run_tasks({
              member: lambda m=member: self.reading(ship, m, turn)
              for member in on
          })
      )
    else:
      readings = {m: self.reading(ship, m, turn) for m in on}
    for member in self.crew:
      if member not in on:  # Off duty: they rest and read nothing.
        readings[member] = (OFF_DUTY_ACTION, OFF_DUTY)
    chosen = self.crew_rule.act(ship, kind, lambda member: readings[member][0])
    # The situation already says who is off duty.
    steps.append(
        ' '.join(f'{m} {self._data["actions"][chosen[m]]["does"]}.' for m in on)
    )
    seen = self.watch.reveal(
        ship,
        chosen,
        lambda t: self.event(ship, t),
        self.hazards.sight(kind, self.watch.sees),
    )
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
    ship.log.append({
        'turn': turn,
        'event': kind,
        'actions': chosen,
        'why': {m: readings[m][1] for m in self.crew},
    })
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
        'scenario': self.scenario,
        # The scene's task in plain words, for the player (cycle 93).
        'premise': ' '.join(str(self._data.get('premise', '')).split()),
        # What the scene teaches, shown once it is over.
        'lesson': self._data.get('lesson') if self.done() else None,
        'bosun': self.bosun_of(ship),  # who passes orders on (arc 5)
        # For an order with a span: which of its turns the next one is.
        'order_turn': (
            turn - ship.order_turn + 1
            if ship.order and TIMED.search(ship.order)
            else None
        ),
        'free': self.free,
        'turn': turn,
        'turns': int(self._data['turns']),
        'now': self.event(ship, turn),
        'leak_open': ship.leak,
        'situation': self.situation(ship, turn),
        'off': self.rota.off_on(ship, turn),
        'ahead': [
            {
                'turn': t,
                'event': ship.seen.get(str(t)),
                'off': self.rota.off_on(ship, t),
            }
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
        # What each star above the first asks, as limits (cycle 89).
        'star_rules': {
            str(star): {k: int(v) for k, v in wants.items()}
            for star, wants in sorted(self._data.get('stars', {}).items())
        },
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
                'off': self.rota.off_duty(ship, member, turn),
                # The earlier order they still follow, if they missed one.
                'heard': (
                    self.rota.order_of(ship, member)
                    if self.rota.order_of(ship, member) != ship.order
                    else None
                ),
                'carried': bool(
                    ship.carried
                    and self.rota.order_of(ship, member) == ship.carried
                ),
                'via_bosun': bool(
                    self.bosun_of(ship)
                    and self.rota.order_of(ship, member) == ship.relayed
                    and member != self.bosun_of(ship)
                ),
                'last': (last or {}).get('actions', {}).get(member),
                'why': (last or {}).get('why', {}).get(member),
            }
            for member in self.crew
        ],
    }

  # ---- end of game, and forecasts ---------------------------------------

  def wrecked(self, ship: Ship) -> bool:
    return not self.free and ship.damage >= int(self._data['wreck_at'])

  def aim(self, ship: Ship) -> str:
    """The scene's goal and how far along it is, as told to everyone."""
    target = self._data['target']
    if self.goal() == 'foresight':
      return (
          f'the harbour master wants {target} turns seen ahead'
          f' (seen so far: {ship.foresight})'
      )
    if self.goal() == 'supplies':
      return f'the port wants {target} supplies at the end'
    return f'project {ship.progress} of {target}'

  def crew_stakes(self, ship: Ship) -> str:
    """The crew's line on the project and the goal (cycle 80: in scenes
    whose goal is not the project, they were told "project 0 of 10")."""
    if self.free:
      return f'project {ship.progress}'
    if self.goal() == 'progress':
      return self.aim(ship)
    return f'project {ship.progress}; {self.aim(ship)}'

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
        'words_spoken': ship.words_spoken,
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
