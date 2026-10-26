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

"""The world of Lifeways: land, people's bodies and lives, and the year.

This is the deterministic substrate the World GM owns. People's minds are
their own entities; here the world knows only what the GM could know: their
bodies (age, food, place, kin), what they did (their answers to the GM's
questions) and what the GM gave them (inheritance at birth, offers). All
randomness is drawn from the world seed, the year and what it is for, so a
year's outcome depends only on what was done in it, never on the order in
which people's threads happened to act.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import math
import re
from typing import Any
import zlib

from examples.lifeways.world import culture as culture_lib
from examples.lifeways.world import land as land_lib
import numpy as np

EXPERIMENTER = 'Experimenter'

# The moves of a year.
GATHER = 'gather food'
REST = 'rest'
CHILD = 'have a child'
COURT = 'look for a partner'
MOVE = 'move to better land'
PLAY = 'play'
WATCH = 'watch'
PASS_ON = 'pass on'

ADULT_AGE = 16
CHILD_ACTS_AGE = 5
ELDER_AGE = 56
FERTILE = (16, 45)
CUSTOM_ID = re.compile(r'\bc(\d+)\b')
_SYLLABLES = (
    'ka',
    'ri',
    'to',
    'ma',
    'el',
    'an',
    'so',
    'vi',
    'lu',
    'den',
    'ar',
    'ne',
    'bo',
    'tem',
    'sa',
    'yo',
    'mi',
    'ra',
    'gun',
    'li',
    'ta',
    'or',
    'wen',
    'ha',
)


def _seed_of(*keys: Any) -> list[int]:
  """Stable integers for a random stream's seed."""
  return [
      key if isinstance(key, int) else zlib.crc32(str(key).encode())
      for key in keys
  ]


@dataclasses.dataclass
class Person:
  """A person's body and life, as the world knows them."""

  name: str
  born: int
  sex: str  # 'f', 'm', or '' in asexual mode.
  tile: tuple[int, int]
  food: float
  parents: list[str] = dataclasses.field(default_factory=list)
  partner: str = ''
  children: list[str] = dataclasses.field(default_factory=list)
  starving: int = 0
  # Customs the GM gave this person (at birth, by offers): ids.
  given: list[str] = dataclasses.field(default_factory=list)

  def age(self, year: int) -> int:
    return year - self.born

  def to_dict(self) -> dict[str, Any]:
    data = dataclasses.asdict(self)
    data['tile'] = list(self.tile)
    return data

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> 'Person':
    return cls(
        name=str(data['name']),
        born=int(data['born']),
        sex=str(data['sex']),
        tile=(int(data['tile'][0]), int(data['tile'][1])),
        food=float(data['food']),
        parents=[str(p) for p in data.get('parents', ())],
        partner=str(data.get('partner', '')),
        children=[str(c) for c in data.get('children', ())],
        starving=int(data.get('starving', 0)),
        given=[str(g) for g in data.get('given', ())],
    )


@dataclasses.dataclass
class YearOutcome:
  """What closing a year changed beyond the world itself."""

  # Prefab params of each person born, to add to the simulation.
  births: list[dict[str, Any]]
  # Names of the people who died, to remove from it.
  deaths: list[str]


class World:
  """The land, the people's lives and the record of customs."""

  def __init__(self, settings: Mapping[str, Any]):
    self.settings = dict(settings)
    self.seed = int(settings.get('seed', 0))
    self.mode = str(settings.get('mode', 'sexual'))
    self.year = 0
    self.land = land_lib.Land(
        int(settings.get('rows', 16)),
        int(settings.get('cols', 24)),
        self.seed,
        richness=float(settings.get('richness', 0.45)),
    )
    self.culture = culture_lib.Culture()
    self.people: dict[str, Person] = {}
    self.dead: list[str] = []
    self.used_names: set[str] = set()
    # This year's answers, by name; pending births, by first parent.
    self.actions: dict[str, str] = {}
    self.expecting: dict[str, list[str]] = {}
    # Observations to deliver at the start of next year, by name.
    self.queued: dict[str, list[str]] = {}
    # Rolling census, a row a year (the long history goes to disk).
    self.history: list[dict[str, Any]] = []
    self._founding: list[dict[str, Any]] = []

  # --- Setup -------------------------------------------------------------

  def found(self) -> list[dict[str, Any]]:
    """Creates the founders; returns each one's prefab params."""
    customs = [
        self.culture.add(
            c['kind'], c['domain'], c['text'], c.get('params'), year=0
        )
        for c in self.settings.get('founding_customs', ())
    ]
    count = int(self.settings.get('founders', 12))
    rng = np.random.default_rng(_seed_of(self.seed, 'founders'))
    rows, cols = self.land.rows, self.land.cols
    home = (rows // 2, cols // 3)
    founders = []
    for i in range(count):
      tile = self.land.wrap(
          home[0] + int(rng.integers(-2, 3)), home[1] + int(rng.integers(-2, 3))
      )
      sex = '' if self.mode == 'asexual' else ('f' if i % 2 == 0 else 'm')
      person = Person(
          name=self._new_name(rng),
          born=-int(rng.integers(18, 40)),
          sex=sex,
          tile=tile,
          food=8.0,
          given=[c.id for c in customs],
      )
      self.people[person.name] = person
      founders.append(self._params(person))
      self._tell(person.name, self._birth_story(person, founder=True))
    return founders

  def _params(self, person: Person) -> dict[str, Any]:
    """The prefab params of a person's entity."""
    return {
        'name': person.name,
        'born': person.born,
        'sex': person.sex,
        'parents': list(person.parents),
        'seed': self.seed,
        'customs': [self.culture.get(c).spec() for c in person.given],
    }

  def _new_name(self, rng: np.random.Generator) -> str:
    for _ in range(50):
      parts = [
          _SYLLABLES[int(rng.integers(len(_SYLLABLES)))]
          for _ in range(int(rng.integers(2, 4)))
      ]
      name = ''.join(parts).capitalize()
      if name not in self.used_names:
        self.used_names.add(name)
        return name
    n = len(self.used_names)
    name = f'Child {n}'
    self.used_names.add(name)
    return name

  # --- Who acts ----------------------------------------------------------

  def living(self) -> list[str]:
    return sorted(self.people)

  def participants(self) -> list[str]:
    """Who must answer before the year closes."""
    return [
        name
        for name in self.living()
        if self.people[name].age(self.year) >= CHILD_ACTS_AGE
    ] + [EXPERIMENTER]

  def may_act(self, name: str) -> bool:
    return name in self.participants() and name not in self.actions

  def record_action(self, name: str, action: str) -> None:
    if name in self.participants() and name not in self.actions:
      self.actions[name] = action

  def year_complete(self) -> bool:
    return all(name in self.actions for name in self.participants())

  # --- What each is asked ------------------------------------------------

  def moves(self, name: str) -> list[str]:
    """A person's moves this year (the Experimenter only watches)."""
    if name == EXPERIMENTER:
      return [WATCH]
    person = self.people[name]
    age = person.age(self.year)
    if age < ADULT_AGE:
      return [PLAY, GATHER]
    found = [GATHER, REST, MOVE]
    if self._may_have_child(person):
      found.append(CHILD)
    if self.mode == 'sexual' and not person.partner:
      found.append(COURT)
    return found

  def asked_to_pass_on(self, name: str) -> bool:
    return any(name in parents for parents in self.expecting.values())

  def _may_have_child(self, person: Person) -> bool:
    age = person.age(self.year)
    if not FERTILE[0] <= age <= FERTILE[1] or self.asked_to_pass_on(
        person.name
    ):
      return False
    if self.mode == 'sexual' and (person.sex != 'f' or not person.partner):
      return False
    return person.food >= float(self.settings.get('child_cost', 2.0)) + 1.0

  def status(self, name: str) -> str:
    """What a person knows of its own body and place this year."""
    if name == EXPERIMENTER:
      return f'Year {self.year}. {len(self.people)} people live in the valley.'
    person = self.people[name]
    partner = person.partner or 'no partner'
    kids = ', '.join(c for c in person.children if c in self.people)
    return (
        f'Year {self.year}. {name} is {person.age(self.year)} years old and'
        f' lives on {self.land.kind(person.tile)} land at'
        f' {person.tile[0]},{person.tile[1]}; the land around holds'
        f' {self.land.around_food(person.tile):.1f} food. Food stored:'
        f' {person.food:.1f} (a year needs 1.0). Partner: {partner}.'
        f' Children: {kids or "none"}.'
    )

  # --- Closing the year --------------------------------------------------

  def close_year(self) -> YearOutcome:
    """Resolves the year's answers in a fixed order and moves on a year."""
    births = self._births()
    self._pair_partners()
    for name in sorted(self.actions):
      if name in self.people:
        self._resolve(name, self.actions[name])
    deaths = self._eat_and_die()
    self.land.regrow()
    self.history.append(self._census(len(births), len(deaths)))
    del self.history[: -int(self.settings.get('history_kept', 200))]
    self.actions = {}
    self.year += 1
    for name in self.participants():
      self._tell(name, self.status(name))
    return YearOutcome(births=births, deaths=deaths)

  def _rng(self, *keys: Any) -> np.random.Generator:
    return np.random.default_rng(_seed_of(self.seed, self.year, *keys))

  def _resolve(self, name: str, action: str) -> None:
    person = self.people[name]
    move = action.strip().lower()
    cited = [f'c{m}' for m in CUSTOM_ID.findall(move)]
    if move.startswith(GATHER):
      effort = 2.0 if person.age(self.year) >= ADULT_AGE else 1.0
      effort *= self._skill(person, cited, 'food')
      got = self.land.gather(person.tile, effort)
      person.food += got
    elif move.startswith(MOVE):
      person.tile = self.land.best_near(person.tile)
    elif move.startswith(CHILD) and self._may_have_child(person):
      parents = [name] + ([person.partner] if person.partner else [])
      self.expecting[name] = parents
      for parent in parents:
        self._tell(parent, f'{name} is expecting a child.')
    # REST, PLAY, WATCH and anything not understood change nothing.

  def _skill(self, person: Person, cited: Sequence[str], domain: str) -> float:
    """How much better a cited skill makes the work (1.0: none).

    A skill counts only if the GM once gave it to this person: the world
    credits know-how it delivered, never what a mind might claim.

    Args:
      person: Who works.
      cited: Custom ids the answer names.
      domain: The work's domain.

    Returns:
      The effort multiplier.
    """
    best = 1.0
    for custom_id in cited:
      if custom_id in person.given and custom_id in self.culture:
        custom = self.culture.get(custom_id)
        if custom.kind == 'skill' and custom.domain == domain:
          level = int(custom.params.get('level', 1))
          best = max(best, 1.0 + 0.25 * level)
    return best

  def _pair_partners(self) -> None:
    """Pairs those who looked for a partner, nearest first, in name order."""
    looking = [
        n
        for n in sorted(self.actions)
        if n in self.people
        and self.actions[n].strip().lower().startswith(COURT)
        and not self.people[n].partner
    ]
    taken: set[str] = set()
    for name in looking:
      if name in taken:
        continue
      person = self.people[name]
      matches = [
          other
          for other in looking
          if other not in taken
          and other != name
          and self.people[other].sex not in ('', person.sex)
          and other not in person.parents
          and name not in self.people[other].parents
      ]
      if not matches:
        continue
      other = min(matches, key=lambda o: (self._distance(person, o), o))
      taken.update((name, other))
      person.partner, self.people[other].partner = other, name
      self._tell(name, f'{name} and {other} are now partners.')
      self._tell(other, f'{other} and {name} are now partners.')

  def _distance(self, person: Person, other: str) -> int:
    (r1, c1), (r2, c2) = person.tile, self.people[other].tile
    dr = min(abs(r1 - r2), self.land.rows - abs(r1 - r2))
    dc = min(abs(c1 - c2), self.land.cols - abs(c1 - c2))
    return dr + dc

  def _births(self) -> list[dict[str, Any]]:
    """Children of last year's expecting parents, with what they passed on."""
    births = []
    mutation = float(self.settings.get('mutation', 0.05))
    cost = float(self.settings.get('child_cost', 2.0))
    for first in sorted(self.expecting):
      parents = [p for p in self.expecting[first] if p in self.people]
      if first not in self.people:
        continue
      passed = [self._passed_on(p) for p in parents]
      rng = self._rng('birth', first)
      if len(passed) == 2:
        ids = self.culture.recombine(passed[0], passed[1], rng)
      else:
        ids = list(passed[0]) if passed else []
      ids = [self.culture.copy(i, rng, mutation, self.year).id for i in ids]
      mother = self.people[first]
      child = Person(
          name=self._new_name(rng),
          born=self.year,
          sex=''
          if self.mode == 'asexual'
          else ('f' if rng.random() < 0.5 else 'm'),
          tile=mother.tile,
          food=min(cost, mother.food),
          parents=list(parents),
          given=list(dict.fromkeys(ids)),
      )
      mother.food -= child.food
      for parent in parents:
        self.people[parent].children.append(child.name)
        self._tell(parent, f'{child.name} is born to {" and ".join(parents)}.')
      self.people[child.name] = child
      self._tell(child.name, self._birth_story(child), newborn=True)
      births.append(self._params(child))
    self.expecting = {}
    return births

  def _passed_on(self, parent: str) -> list[str]:
    """The ids a parent named when asked what it passes on (if it answered)."""
    answer = self.actions.get(parent, '')
    if not answer.strip().lower().startswith(PASS_ON):
      return []
    return [
        f'c{m}'
        for m in dict.fromkeys(CUSTOM_ID.findall(answer.lower()))
        if f'c{m}' in self.culture
    ]

  def _birth_story(self, person: Person, founder: bool = False) -> str:
    if founder:
      return f'{person.name} is one of the first people to settle the valley.'
    return (
        f'{person.name} is born in year {self.year} to'
        f' {" and ".join(person.parents)}.'
    )

  def _eat_and_die(self) -> list[str]:
    """Everyone eats; the old, the sick and the starving may die."""
    deaths = []
    for name in self.living():
      person = self.people[name]
      age = person.age(self.year)
      need = 0.5 if age < ADULT_AGE else 1.0
      if person.food < need and age < ADULT_AGE:
        for parent in person.parents:  # Children are fed by their parents.
          giver = self.people.get(parent)
          if giver is not None and giver.food >= need + 1.0:
            giver.food -= need
            person.food += need
            break
      person.food -= need
      person.starving = person.starving + 1 if person.food < 0 else 0
      person.food = max(0.0, person.food)
      hazard = 0.0003 * math.exp(0.085 * max(0, age)) + (
          0.02 if age < 2 else 0.0
      )
      if person.starving >= 2 or self._rng('death', name).random() < hazard:
        deaths.append(name)
    for name in deaths:
      self._die(name)
    return deaths

  def _die(self, name: str) -> None:
    person = self.people.pop(name)
    self.dead.append(name)
    if person.partner in self.people:
      self.people[person.partner].partner = ''
    for kin in {*person.parents, *person.children, person.partner}:
      if kin in self.people:
        self._tell(kin, f'{name} has died, aged {person.age(self.year)}.')
    for parents in list(self.expecting.values()):
      if name in parents:
        parents.remove(name)
    self.queued.pop(name, None)

  def _census(self, births: int, deaths: int) -> dict[str, Any]:
    ages = [p.age(self.year) for p in self.people.values()]
    return {
        'year': self.year,
        'population': len(self.people),
        'births': births,
        'deaths': deaths,
        'mean_age': round(sum(ages) / len(ages), 2) if ages else 0.0,
        'customs_recorded': len(self.culture),
        'food': round(float(self.land.food.sum()), 3),
    }

  def _tell(self, name: str, text: str, *, newborn: bool = False) -> None:
    """Queues an observation (a baby keeps only the story of its birth)."""
    person = self.people.get(name)
    if (
        person is not None
        and person.age(self.year) < CHILD_ACTS_AGE
        and not newborn
    ):
      return
    self.queued.setdefault(name, []).append(text)

  def take_observations(self) -> dict[str, list[str]]:
    """The observations queued for delivery, emptied."""
    queued, self.queued = self.queued, {}
    return queued

  # --- State -------------------------------------------------------------

  def get_state(self) -> dict[str, Any]:
    return {
        'year': self.year,
        'land': self.land.get_state(),
        'culture': self.culture.get_state(),
        'people': [p.to_dict() for p in self.people.values()],
        'dead': list(self.dead[-500:]),
        'used_names': sorted(self.used_names),
        'actions': dict(self.actions),
        'expecting': {k: list(v) for k, v in self.expecting.items()},
        'queued': {k: list(v) for k, v in self.queued.items()},
        'history': list(self.history),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.year = int(state['year'])
    self.land.set_state(state['land'])
    self.culture.set_state(state['culture'])
    self.people = {
        str(p['name']): Person.from_dict(p) for p in state.get('people', ())
    }
    self.dead = [str(d) for d in state.get('dead', ())]
    self.used_names = {str(n) for n in state.get('used_names', ())}
    self.actions = {str(k): str(v) for k, v in state.get('actions', {}).items()}
    self.expecting = {
        str(k): [str(p) for p in v]
        for k, v in state.get('expecting', {}).items()
    }
    self.queued = {
        str(k): [str(t) for t in v] for k, v in state.get('queued', {}).items()
    }
    self.history = [dict(r) for r in state.get('history', ())]
