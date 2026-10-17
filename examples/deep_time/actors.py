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

"""Game rules for each kind of acting entity.

The World GM asks the actor registry for an entity's action spec and hands
each chosen action back to the same actor to turn into world changes. Entities
themselves are ordinary Concordia agents built by prefabs; an `Actor` is the
game-side rule book for one of them.
"""

# pylint: disable=g-complex-comprehension,g-doc-return-or-yield,missing-function-docstring

import abc
from collections.abc import Sequence

from examples.deep_time import attention as attention_lib
from examples.deep_time import crises as crises_lib
from examples.deep_time.world import automation as automation_lib
from examples.deep_time.world import civ as civ_lib
from examples.deep_time.world import planet as planet_lib
from concordia.typing import entity as entity_lib

SPECIES_CALL_TO_ACTION = (
    'You are {name}, a living species. Read what you can sense of the climate'
    ' and the census, then choose how your populations behave this tick.'
)

POLITY_CALL_TO_ACTION = (
    'You are {name}, a people of {age}. Read your storehouse,'
    ' the census of the peoples and the weather, then choose where to put'
    ' more of your labor this generation.'
)
KEEP_LABOR = 'keep the current balance of labor'
# Offered to a people a person plays, when it has towns: the next town rises
# where the person taps the map.
PLAN_TOWN = 'plan where our next town will stand'
MISSION_CALL_TO_ACTION = (
    'You are {name}, at home, four light-years from the colony. Everything'
    ' you hear from it is four years old, and anything you send arrives four'
    ' years from now. Choose one transmission this year, or listen.'
)
LISTEN = 'listen and wait'
COLONY_CALL_TO_ACTION = (
    'You are {name}, colonists on a cold world light-years from home. Read'
    ' your stores, what you need each generation and the sky, then choose'
    ' where to put more of your labor this generation.'
)
# Menu sizes: 10 labor + 6 learn + 5 trade + 4 reflex + keep = 26, plus up
# to two treaty offers and a broadsheet; a menu never passes 26 options (as
# many as a multiple-choice question can letter): the last trades and
# reflexes make room.
MAX_OPTIONS = 26
MAX_FOCUS = 10
# Labor slots kept for the newest recipes a people has never worked.
NEW_RECIPE_SLOTS = 3
MAX_LEARN = 6  # As many as the sheet lists within reach.
MAX_TRADES = 5
MAX_REFLEXES = 4
PRINT_BROADSHEET = 'print a broadsheet in the Gazette'
PAY_PAMPHLETS = 'pay the coal guilds for pamphlets'
MAX_PLANS = 3
MIN_ORDERS = 3  # Trade and reflex options always kept on a crowded menu.
# A target with less of the labor than this may be planned (again).
PLANNED_SHARE = 0.05
# Generations before the same plan may be adopted again.
PLAN_REST = 8

GRAZER_CALL_TO_ACTION = (
    'You are {name}, a clade of grazing animals. Read what you sense of the'
    ' climate and the meadows, then choose how your herds behave this tick.'
)

CHRONICLER_CALL_TO_ACTION = '{name} keeps watch over the planet.'
CHRONICLER_RESPONSE = 'keeps watch.'

STEWARD_CALL_TO_ACTION = (
    'You are {name}, Steward of this planet. Read your instruments, then'
    ' choose one intervention for this tick.'
)
# Where the Steward may speak an omen, it answers in its own words: one of
# the interventions as written, or a sign and what it is taken to say.
STEWARD_SPEAKS = (
    'You are {name}, Steward of this planet. Read your instruments, then'
    ' make one intervention for this tick: answer with one of the'
    ' interventions below exactly as written, or speak an omen by answering'
    ' with a sign every people will see ({signs}), a colon, and up to'
    ' {words} words the sign will be taken to say.'
)


class Actor(metaclass=abc.ABCMeta):
  """Rules for one acting entity."""

  def __init__(self, name: str, outcome_topic: str | None = None):
    """Initializes the actor.

    Args:
      name: Entity name.
      outcome_topic: Topic on which resolved outcomes are announced, if any.
    """
    self.name = name
    self.outcome_topic = outcome_topic
    self.acts = 0  # Actions resolved so far.
    # What each resolved action did, as (tick, action, outcome), newest last.
    self.log: list[tuple[int, str, str]] = []
    # Extra (topic, text) messages to publish after resolution.
    self.outbox: list[tuple[str, str]] = []

  def drain_outbox(self) -> list[tuple[str, str]]:
    messages, self.outbox = self.outbox, []
    return messages

  def is_active(self) -> bool:
    """Whether the entity acts this tick."""
    return True

  @abc.abstractmethod
  def action_spec(self) -> entity_lib.ActionSpec:
    """The action spec for this tick."""

  @abc.abstractmethod
  def resolve(self, action: str) -> str:
    """Applies a chosen action to the world; returns what happened."""

  def aimable(self, option: str) -> bool:
    """Whether a person may aim this option at a place on the map."""
    del option
    return False


class StewardActor(Actor):
  """The planet's Steward, choosing one Ω intervention per tick.

  With limited attention it may instead spend the tick refocusing: watching
  one instrument in place of another costs no Ω.
  """

  def __init__(
      self,
      name: str,
      planet: planet_lib.Planet,
      attention: attention_lib.Attention | None = None,
  ):
    super().__init__(name, outcome_topic='gaia.omens')
    self._planet = planet
    self._attention = attention
    # The world's omen, if the Steward may speak one here.
    self.omen = next(
        (i for i in planet.interventions() if i.effect['kind'] == 'omen'),
        None,
    )

  def options(self) -> list[str]:
    # Interventions still recharging, or costing more omega than there is,
    # are not offered: choosing one would waste the turn.
    options = [
        i.option
        for i in self._planet.interventions()
        if self._planet.ready(i) and i.cost <= self._planet.omega
    ]
    if self._attention is not None:
      options.extend(self._attention.options())
    return options

  def out_of_reach(self) -> list[str]:
    """Moves off their cooldown that cost more omega than there is yet.

    Not offered this turn, but shown (and plannable for a later turn).
    """
    return [
        i.option
        for i in self._planet.interventions()
        if self._planet.ready(i) and i.cost > self._planet.omega
    ]

  def action_spec(self) -> entity_lib.ActionSpec:
    if self.omen is None:
      return entity_lib.choice_action_spec(
          call_to_action=STEWARD_CALL_TO_ACTION,
          options=self.options(),
          tag='steward',
      )
    # Concordia's free action: the moves are listed as a choice lists them.
    letters = 'abcdefghijklmnopqrstuvwxyz'
    moves = '\n'.join(
        f'({letters[i]}) {option}' for i, option in enumerate(self.options())
    )
    effect = self.omen.effect
    return entity_lib.free_action_spec(
        call_to_action=STEWARD_SPEAKS.replace(
            '{signs}', ', '.join(effect['signs'])
        ).replace('{words}', str(effect['max_words']))
        + '\n'
        + moves,
        tag='steward',
    )

  def resolve(self, action: str) -> str:
    text = action.strip()
    if text.startswith(f'{self.name} '):  # A model's free answer.
      text = text[len(self.name) + 1 :].strip()
    if self._attention is not None:
      swap = self._attention.options().get(text.lower())
      if swap is not None:
        self._attention.swap(*swap)
        return (
            f'{self.name} now watches the {swap[0].lower()} instead of the'
            f' {swap[1].lower()}.'
        )
    intervention = self._planet.intervention_for_option(text)
    if intervention is None and self.omen is not None:
      # Words that are no move are an omen's.
      return self._planet.apply_intervention(self.name, self.omen, text)
    if intervention is None:
      return f'{self.name} hesitated; "{text}" is not an option.'
    return self._planet.apply_intervention(self.name, intervention)

  def aimable(self, option: str) -> bool:
    found = self._planet.intervention_for_option(option)
    return found is not None and found.effect['kind'] in planet_lib.AIMABLE


class SpeciesActor(Actor):
  """A species (daisies, mats, ...) choosing a behaviour each tick."""

  def __init__(self, name: str, species: str, planet: planet_lib.Planet):
    super().__init__(name, outcome_topic=f'bio.species.{species}.behaviour')
    self.species = species
    self._planet = planet

  def _evolution(self):
    evolution = self._planet.evolution
    if evolution is None:
      raise ValueError('Species need a planet with evolution rules.')
    return evolution

  def _pact_partners(self) -> dict[str, str]:
    """Option text -> partner key for every pact this species could offer."""
    symbiosis = self._planet.symbiosis
    if symbiosis is None:
      return {}
    partners = {}
    for other in self._planet.daisies.species():
      if (
          other == self.species
          or self._planet.is_extinct(other)
          or symbiosis.has_pact(self.species, other)
          or symbiosis.pending_for(other) is not None
      ):
        continue
      name = self._planet.species_name(other)
      partners[f'propose a pact to the {name}'] = other
    return partners

  def options(self) -> list[str]:
    return [b.label for b in self._evolution().behaviours] + list(
        self._pact_partners()
    )

  def is_active(self) -> bool:
    return not self._planet.is_extinct(self.species)

  def action_spec(self) -> entity_lib.ActionSpec:
    return entity_lib.choice_action_spec(
        call_to_action=SPECIES_CALL_TO_ACTION,
        options=self.options(),
        tag='species',
    )

  def resolve(self, action: str) -> str:
    partner = self._pact_partners().get(action.strip().lower()) or next(
        (
            key
            for text, key in self._pact_partners().items()
            if action.strip().lower() == text.lower()
        ),
        None,
    )
    if partner is not None:
      text = self._planet.propose_pact(self.species, partner)
      if text is None:
        return f'{self.name} could not offer a pact right now.'
      self.outbox.append((f'diplomacy.{partner}.{self.species}', text))
      return text
    behaviour = self._evolution().behaviour_for_option(action)
    if behaviour is None:
      return f'{self.name} hesitated; "{action.strip()}" is not a behaviour.'
    return self._planet.set_behaviour(
        self.species, behaviour.key, actor=self.name
    )

  def aimable(self, option: str) -> bool:
    found = self._evolution().behaviour_for_option(option)
    return found is not None and found.key == 'spread'


class PolityActor(Actor):
  """A people deciding where to put more of its labor each tick."""

  def __init__(
      self,
      name: str,
      polity: str,
      planet: planet_lib.Planet,
      share: float,
      automation: automation_lib.Automation | None = None,
      answers_offers: bool = False,
      age: str = 'the early iron age',
      coached: bool = True,
  ):
    super().__init__(name, outcome_topic=f'civ.{polity}.decrees')
    self.age = age  # Named in the call to action ("a people of …").
    # Whether its question ends with what matters most and its menu always
    # keeps the tech worth learning: a playbook, off when a live mind plays
    # the people as a character (it decides from who the people are).
    self.coached = coached
    self.polity = polity
    self._planet = planet
    self._share = share
    self._automation = automation
    # Answer treaty offers on the menu (a person plays this people; otherwise
    # its treaty mind answers in the background).
    self.answers_offers = answers_offers
    # How often each kind of decision was made: labor, learn, trade,
    # reflex, keep.
    self.decisions: dict[str, int] = {}

  def _tally(self, kind: str) -> None:
    self.decisions[kind] = self.decisions.get(kind, 0) + 1

  def _civ(self):
    civ = self._planet.civ
    if civ is None:
      raise ValueError('Polities need a planet with a civilization.')
    return civ

  def _focus(self) -> dict[str, str]:
    """Option text -> recipe key, for the recipes worth offering now.

    Farming, the recipes in use, those that make what last tick ran short
    of, and recipes never worked; when they do not all fit, a few slots are
    kept for the most recently unlocked ones, so a fresh unlock always
    shows. At most `MAX_FOCUS`, in the data's order: a short menu for the
    model.

    Returns:
      Option text -> recipe key.
    """
    civ = self._civ()
    people = civ.polities[self.polity]
    short = set(people.shortages.values())
    recipes = civ.recipes_for(self.polity)
    wanted = recipes
    if len(recipes) > MAX_FOCUS:
      fresh = [r for r in recipes if r.key not in people.allocation]
      core = [
          r
          for r in recipes
          if r not in fresh
          and (
              r.key == 'farm'
              or people.allocation.get(r.key, 0.0) > 0
              or set(r.outputs) & short
          )
      ]
      if len(core) + len(fresh) > MAX_FOCUS:
        # Keep makers of what ran short, then the most recently unlocked
        # (by the order techs were learned).
        learned = {t: i for i, t in enumerate(people.techs)}
        fresh = sorted(
            fresh,
            key=lambda r: (
                bool(set(r.outputs) & short),
                learned.get(r.requires or '', -1),
            ),
        )
        fresh = fresh[-NEW_RECIPE_SLOTS:]
        # Producers of what ran short first, then farming, then the
        # recipes worked most.
        core = sorted(
            core,
            key=lambda r: (
                not set(r.outputs) & short,
                r.key != 'farm',
                -people.allocation.get(r.key, 0.0),
            ),
        )[: MAX_FOCUS - len(fresh)]
      wanted = [r for r in recipes if r in core or r in fresh]
    return {
        f'put more labor into: {r.label}': r.key for r in wanted[:MAX_FOCUS]
    }

  def pushes(self, options: Sequence[str]) -> dict[str, str]:
    """What this people's labor and learning moves do.

    A labor move's push is worked out from the people's own land, tools
    and recipes (`Civilization.labor_shift`); a tech's from its effects.

    Args:
      options: This people's options this turn.

    Returns:
      Option -> push, for the options that have one.
    """
    civ = self._civ()
    focus, learn = self._focus(), self._learnable()
    shifts = civ.labor_shifts(
        self.polity,
        sorted({
            focus[o.strip().lower()]
            for o in options
            if o.strip().lower() in focus
        }),
        self._share,
    )
    found = {}
    for option in options:
      text = option.strip().lower()
      if text in focus:
        found[option] = civ_lib.describe_change(
            shifts[focus[text]],
            civ.pollutant_words,
            made=civ.recipe(focus[text]).outputs,
        )
      elif text in learn:
        found[option] = civ.describe_tech(learn[text])
    return found

  def _learnable(self) -> dict[str, str]:
    """Option text -> tech key for the techs the polity can afford."""
    civ = self._civ()
    research = civ.stock(self.polity, 'research')
    affordable = sorted(
        (
            t
            for t in civ.available_techs(self.polity)
            if t.cost <= research + 1e-9
        ),
        key=lambda t: (t.cost, t.key),
    )
    if len(affordable) > MAX_LEARN:
      # The tech that most helps the goal's chain, the cheapest
      # half, and the most advanced of the rest (the frontier would
      # otherwise hide behind cheap old techs).
      good = crises_lib.goal_good(self._planet) if self.coached else None
      best = civ.best_tech(self.polity, good) if good else None
      keep = [t for t in affordable if best is not None and t.key == best.key]
      others = [t for t in affordable if t not in keep]
      cheap = others[: MAX_LEARN // 2]
      rest = sorted(
          others[MAX_LEARN // 2 :], key=lambda t: (-t.tier, -t.cost, t.key)
      )
      room = MAX_LEARN - len(keep) - len(cheap)
      chosen = {t.key for t in keep + cheap + rest[:room]}
      affordable = [t for t in affordable if t.key in chosen]
    return {f'learn {t.label} ({t.cost:g} research)': t.key for t in affordable}

  def _trades(self) -> dict[str, tuple[str, str]]:
    """Option text -> (good, side) for every order the polity could place."""
    trade = self._planet.trade
    if trade is None:
      return {}
    civ = self._civ()
    sellable = trade.sellable(self.polity)
    # The biggest surpluses first, and asks for what idled a workshop
    # first: the menu is capped, so the pressing trades must lead.
    offers = sorted(
        sellable,
        key=lambda g: -civ.stock(self.polity, g)
        / max(1e-9, trade.good(g).surplus),
    )
    short = set(civ.polities[self.polity].shortages.values())
    asks = sorted(
        (g for g in trade.buyable(self.polity) if g not in sellable),
        key=lambda g: g not in short,
    )
    # Interleave offers and asks so a trimmed menu keeps both kinds.
    options = {}
    for i in range(max(len(offers), len(asks))):
      if i < len(offers):
        options[f'offer surplus {offers[i]} for sale'] = (offers[i], 'ask')
      if i < len(asks):
        options[f'ask to buy {asks[i]}'] = (asks[i], 'bid')
    return dict(list(options.items())[:MAX_TRADES])

  def _reflexes(self) -> dict[str, tuple[str, ...]]:
    """Option text -> ('cancel', name) or ('adopt', side, good).

    Cancellations come first, so a people can always drop an order.

    Returns:
      Option text -> what it does.
    """
    if self._automation is None:
      return {}
    options: dict[str, tuple[str, ...]] = {
        text: ('cancel', name)
        for text, name in self._automation.cancels(self.polity).items()
    }
    for text, order in self._automation.offers(self.polity).items():
      options[text] = ('adopt', *order)
    return dict(list(options.items())[:MAX_REFLEXES])

  def _treaty_partners(self) -> dict[str, tuple[str, str]]:
    """Option text -> (partner key, kind) for treaties the people could offer.

    Every kind needs what its data lists on both sides: diplomacy for coal
    treaties, fishing boats too for quotas, chemistry for runoff pacts.

    Returns:
      Option text -> (partner, kind).
    """
    planet = self._planet
    civ, treaties = self._civ(), planet.treaties
    if treaties is None or not civ.unlocked(self.polity, 'treaties'):
      return {}
    options = {}

    def has(people: str, need: str) -> bool:
      return civ.unlocked(people, need) or need in civ.polities[people].techs

    for kind, spec in treaties.kinds.items():
      for key, p in sorted(civ.polities.items()):
        if (
            key != self.polity
            and all(
                has(k, n) for n in spec.requires for k in (key, self.polity)
            )
            and treaties.can_propose(self.polity, key, kind, planet.tick)
        ):
          options[f'propose a {spec.name} to the {p.name}'] = (key, kind)
    return options

  def _plans(self) -> dict[str, str]:
    """Option text -> target recipe, for balanced chains worth planning."""
    civ = self._civ()
    people = civ.polities[self.polity]
    learned = {t: i for i, t in enumerate(people.techs)}
    # The most recently unlocked targets not yet worked come last (and stay
    # on the menu when it is trimmed).
    targets = sorted(
        (
            spec
            for spec in civ.plannable(self.polity)
            if people.allocation.get(spec.key, 0.0) < PLANNED_SHARE
            and self._planet.tick - people.planned.get(spec.key, -PLAN_REST)
            >= PLAN_REST
        ),
        key=lambda spec: learned.get(spec.requires or '', -1),
    )
    return {
        f'plan a balanced chain to {spec.label}': spec.key for spec in targets
    }

  def _can_print(self) -> bool:
    gazette, civ = self._planet.gazette, self._civ()
    return (
        gazette is not None
        and civ.unlocked(self.polity, 'broadsheets')
        and gazette.can_print(
            self.polity, self._planet.tick, civ.stock(self.polity, 'paper')
        )
    )

  def _can_pamphlet(self) -> bool:
    gazette, civ = self._planet.gazette, self._civ()
    return (
        gazette is not None
        and civ.polities[self.polity].emissions.get('coal', 0.0) > 0
        and gazette.can_pamphlet(
            self.polity, self._planet.tick, civ.stock(self.polity, 'silver')
        )
    )

  def _answers(self) -> dict[str, tuple[str, bool]]:
    """Option text -> (reply, accepted) for an offer awaiting our answer."""
    treaties, civ = self._planet.treaties, self._civ()
    if not self.answers_offers or treaties is None:
      return {}
    offer = treaties.pending_for(self.polity)
    if offer is None:
      return {}
    proposer = civ.polities[offer.proposer].name
    what = treaties.kinds[offer.kind].name
    return {
        f'{verb} the {what} offered by the {proposer}': (
            f'Reply to {proposer} ({what}): {verb}.',
            verb == 'accept',
        )
        for verb in ('accept', 'decline')
    }

  def options(self) -> list[str]:
    # An offer awaiting our answer comes first and is never trimmed.
    fixed = [*self._answers(), *self._focus(), *self._learnable()]
    # One offer of each kind of treaty first, so offers to every partner do
    # not push the press and the plans off a trimmed menu.
    first, more = [], []
    for text, (_, kind) in self._treaty_partners().items():
      (more if any(k == kind for _, k in first) else first).append((text, kind))
    extras = [
        *(text for text, _ in first),
        *([PRINT_BROADSHEET] if self._can_print() else []),
        *([PAY_PAMPHLETS] if self._can_pamphlet() else []),
        *list(self._plans())[-MAX_PLANS:],
        *(text for text, _ in more),
    ]
    # Diplomacy, the press and plans may not crowd out every market order.
    extras = extras[: max(0, MAX_OPTIONS - len(fixed) - 1 - MIN_ORDERS)]
    orders = [*self._trades(), *self._reflexes()]
    room = max(0, MAX_OPTIONS - len(fixed) - len(extras) - 1)
    town = (
        [PLAN_TOWN] if self.answers_offers and self._planet.settlements else []
    )
    return [*fixed, *orders[:room], *extras, *town, KEEP_LABOR]

  def is_active(self) -> bool:
    return self.polity not in self._civ().collapsed(0.05)

  def action_spec(self) -> entity_lib.ActionSpec:
    options = self.options()
    # What matters most this turn ends the question, where a mind reads
    # last: the crisis a move answers, and the weakest link.
    found = (
        crises_lib.pressing(self._planet, self.polity, options, {})
        if self.coached
        else []
    )
    return entity_lib.choice_action_spec(
        # Concordia fills in {name}; the age is the scenario's.
        call_to_action=POLITY_CALL_TO_ACTION.replace('{age}', self.age)
        + crises_lib.pressing_words(found),
        options=options,
        tag='polity',
    )

  def resolve(self, action: str) -> str:
    text = action.strip().lower()
    answers = {k.lower(): v for k, v in self._answers().items()}
    if text in answers:
      reply, accepted = answers[text]
      self._tally('treaty')
      settled = self._planet.answer_treaty(self.polity, reply, accepted)
      if settled is None:
        return f'{self.name} found no offer left to answer.'
      proposer, news = settled
      # Announced like a treaty mind's reply: to the proposer, and as news.
      self.outbox.append((f'diplomacy.{proposer}.{self.polity}', news))
      self.outbox.append((planet_lib.CIV_EVENTS, news))
      return news
    tech = self._learnable().get(text)
    if tech is not None:
      self._tally('learn')
      return self._civ().learn(self.polity, tech)
    partners = {k.lower(): v for k, v in self._treaty_partners().items()}
    chosen = partners.get(text)
    if chosen is not None:
      partner, kind = chosen
      self._tally('treaty')
      offer = self._planet.propose_treaty(self.polity, partner, kind)
      if offer is None:
        return f'{self.name} could not offer a treaty right now.'
      self.outbox.append((f'diplomacy.{partner}.{self.polity}', offer))
      return offer
    target = self._plans().get(text)
    if target is not None:
      self._tally('plan')
      civ = self._civ()
      shares = civ.adopt_plan(self.polity, target)
      civ.polities[self.polity].planned[target] = self._planet.tick
      parts = ', '.join(
          f'{civ.recipe(k).label} {v:.0%}'
          for k, v in sorted(shares.items(), key=lambda kv: -kv[1])
      )
      return (
          f'{self.name} plan a balanced chain to {civ.recipe(target).label}:'
          f' {parts}.'
      )
    if text == PAY_PAMPHLETS and self._can_pamphlet():
      self._tally('press')
      news = self._planet.commission_pamphlet(self.polity)
      return news or f'{self.name} could not pay for pamphlets.'
    if text == PRINT_BROADSHEET.lower() and self._can_print():
      self._tally('press')
      news = self._planet.commission_broadsheet(self.polity)
      return news or f'{self.name} could not print a broadsheet.'
    reflex = self._reflexes().get(text)
    if reflex is not None and self._automation is not None:
      self._tally('reflex')
      if reflex[0] == 'adopt':
        return self._automation.adopt(self.polity, reflex[1], reflex[2])
      return self._automation.cancel(self.polity, reflex[1])
    order = self._trades().get(text)
    if order is not None and self._planet.trade is not None:
      self._tally('trade')
      self._planet.trade.intend(self.polity, *order)
      if self._automation is not None:
        self._automation.note_turn_order()
      verb = 'offer' if order[1] == 'ask' else 'ask to buy'
      return f'{self.name} {verb} {order[0]} at market.'
    if text == PLAN_TOWN and self._planet.settlements is not None:
      self._tally('town')
      return self._planet.plan_town(self.polity, self.name)
    recipe = self._focus().get(text)
    if recipe is None:
      self._tally('keep' if text == KEEP_LABOR else 'unmatched')
      return f'{self.name} keep their labor as it is.'
    self._tally('labor')
    self._civ().shift_labor(self.polity, recipe, self._share)
    where = ''
    if recipe == 'woodcut':  # Felling may be aimed at chosen woods.
      where = self._planet.aim_felling(self.polity, self.name)
    return (
        f'{self.name} put more of their labor into {text.split(": ")[1]}'
        f'{where}.'
    )

  def aimable(self, option: str) -> bool:
    text = option.strip().lower()
    return text == PLAN_TOWN or self._focus().get(text) == 'woodcut'


class ColonyActor(Actor):
  """Colonists deciding where to put more of their labor each generation."""

  def __init__(
      self,
      name: str,
      planet: planet_lib.Planet,
      share: float,
      plan_share: float,
  ):
    super().__init__(name, outcome_topic='colony.decrees')
    self._planet = planet
    self._share = share
    self._plan_share = plan_share
    self.decisions: dict[str, int] = {}

  def _colony(self):
    colony = self._planet.colony
    if colony is None:
      raise ValueError('Colonists need a colony in the scenario.')
    return colony

  def _focus(self) -> dict[str, str]:
    return {
        f'put more labor into: {r.label}': r.key for r in self._colony().known()
    }

  def _plans(self) -> dict[str, str]:
    colony = self._colony()
    return {
        f'plan a balanced chain to {r.label}': r.key
        for r in colony.known()
        if r.inputs
    }

  def options(self) -> list[str]:
    return [*self._focus(), *self._plans(), KEEP_LABOR]

  def is_active(self) -> bool:
    return self._colony().population > 0.5

  def action_spec(self) -> entity_lib.ActionSpec:
    return entity_lib.choice_action_spec(
        call_to_action=COLONY_CALL_TO_ACTION,
        options=self.options(),
        tag='colony',
    )

  def resolve(self, action: str) -> str:
    text = action.strip().lower()
    colony = self._colony()
    target = self._plans().get(text)
    if target is not None:
      self.decisions['plan'] = self.decisions.get('plan', 0) + 1
      shares = colony.adopt_plan(target, self._plan_share)
      labels = {r.key: r.label for r in colony.recipes}
      parts = ', '.join(
          f'{labels[k]} {v:.0%}'
          for k, v in sorted(shares.items(), key=lambda kv: -kv[1])
      )
      return f'{self.name} plan a balanced chain: {parts}.'
    recipe = self._focus().get(text)
    if recipe is None:
      kind = 'keep' if text == KEEP_LABOR else 'unmatched'
      self.decisions[kind] = self.decisions.get(kind, 0) + 1
      return f'{self.name} keep their labor as it is.'
    self.decisions['labor'] = self.decisions.get('labor', 0) + 1
    colony.shift_labor(recipe, self._share)
    where = self._planet.aim_mine(self.name) if recipe == 'ice_mine' else ''
    return (
        f'{self.name} put more of their labor into {text.split(": ")[1]}'
        f'{where}.'
    )

  def aimable(self, option: str) -> bool:
    return self._focus().get(option.strip().lower()) == 'ice_mine'


class MissionActor(Actor):
  """Mission Control at home: designs and directives over light delay."""

  def __init__(self, name: str, planet: planet_lib.Planet):
    super().__init__(name, outcome_topic='mission.log')
    self._planet = planet
    self.decisions: dict[str, int] = {}

  def _colony(self):
    colony = self._planet.colony
    if colony is None:
      raise ValueError('Mission Control needs a colony in the scenario.')
    return colony

  def _designs(self) -> dict[str, str]:
    colony = self._colony()
    in_flight = {w for _, k, w in colony.incoming if k == 'design'}
    return {
        f'transmit the design for {label}': key
        for key, label in colony.design_labels.items()
        if key not in colony.designs and key not in in_flight
    }

  def _directives(self) -> dict[str, str]:
    return {
        f'tell the colonists: put more labor into {r.label}': (
            f'put more labor into {r.label}.'
        )
        for r in self._colony().known()
    }

  def options(self) -> list[str]:
    return [*self._designs(), *self._directives(), LISTEN]

  def action_spec(self) -> entity_lib.ActionSpec:
    return entity_lib.choice_action_spec(
        call_to_action=MISSION_CALL_TO_ACTION,
        options=self.options(),
        tag='mission',
    )

  def resolve(self, action: str) -> str:
    text = action.strip().lower()
    colony = self._colony()
    design = self._designs().get(text)
    if design is not None:
      arrival = colony.transmit('design', design)
      label = colony.design_labels[design]
      message = f'The design for {label}, sent in year {colony.generation}.'
      self.outbox.append(('mission.uplink', message))
      self.decisions['design'] = self.decisions.get('design', 0) + 1
      return (
          f'{self.name} transmit the design for {label}; it reaches'
          f' {colony.name} in year {arrival}.'
      )
    directive = self._directives().get(text)
    if directive is not None:
      arrival = colony.transmit('directive', directive)
      if 'mine ice' in directive:  # A survey of where, if one was marked.
        where = self._planet.aim_mine(self.name, by_wire=True)
        if where:
          directive = directive.rstrip('.') + where + '.'
      self.outbox.append(('mission.uplink', f'Directive: {directive}'))
      self.decisions['directive'] = self.decisions.get('directive', 0) + 1
      return (
          f'{self.name} send a directive ({directive}); it reaches'
          f' {colony.name} in year {arrival}.'
      )
    self.decisions['listen'] = self.decisions.get('listen', 0) + 1
    return f'{self.name} listen and wait.'

  def aimable(self, option: str) -> bool:
    found = self._directives().get(option.strip().lower())
    return found is not None and 'mine ice' in found


class ChroniclerActor(Actor):
  """The Chronicler: acts every tick (no language model) so its slow mind runs.

  Its act is a fixed puppet response; the chapters come from a background
  `pubsub.Publisher` that the act triggers. Acting has no effect on the world.
  """

  def __init__(self, name: str):
    super().__init__(name)

  def action_spec(self) -> entity_lib.ActionSpec:
    return entity_lib.free_action_spec(
        call_to_action=CHRONICLER_CALL_TO_ACTION, tag='chronicler'
    )

  def resolve(self, action: str) -> str:
    del action
    return ''


class GrazerActor(Actor):
  """The grazer clade choosing how its herds behave each tick."""

  def __init__(self, name: str, planet: planet_lib.Planet):
    super().__init__(name, outcome_topic='bio.species.grazers.behaviour')
    self._planet = planet

  def _grazers(self):
    grazers = self._planet.grazers
    if grazers is None:
      raise ValueError('This planet has no grazers.')
    return grazers

  def is_active(self) -> bool:
    return not self._grazers().is_extinct()

  def action_spec(self) -> entity_lib.ActionSpec:
    return entity_lib.choice_action_spec(
        call_to_action=GRAZER_CALL_TO_ACTION,
        options=[b.label for b in self._grazers().behaviours],
        tag='grazers',
    )

  def resolve(self, action: str) -> str:
    grazers = self._grazers()
    behaviour = grazers.behaviour_for_option(action)
    if behaviour is None:
      return f'{self.name} hesitated; "{action.strip()}" is not a behaviour.'
    return self._planet.set_behaviour(
        grazers.key, behaviour.key, actor=self.name
    )

  def aimable(self, option: str) -> bool:
    found = self._grazers().behaviour_for_option(option)
    return found is not None and found.key == 'wander'
