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

import collections
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
TEACH = 'teach'
APPRENTICE = 'learn a craft'
SHARE = 'share food with'  # ... a name.
SHAME = 'shame'  # ... a name.

ADULT_AGE = 16
APPRENTICE_AGE = 12
CHILD_ACTS_AGE = 5
ELDER_AGE = 56
FERTILE = (16, 45)
CUSTOM_ID = re.compile(r'\bc(\d+)\b')
# Names of the dead the world keeps (their lives go to the run's archive).
DEAD_KEPT = 500

# How customs pass between the living, by channel. A scenario's `channels`
# setting replaces these; a channel it leaves out does not exist.
# - upbringing: a parent teaches its children (the `teach` move), and children
#   see what their parents do;
# - observation: anyone within `radius` sees what the `witnessed` nearest
#   people do, and the customs they follow in doing it;
# - apprenticeship: a youth learns a craft beside the nearest adult (within
#   `radius`, at most `per_master` each) who worked by a skill, and may come
#   to know it a level better (`improve`);
# - sanction: the one shamed is shown the norm the shamer shames by;
# - practice: keeping a norm one knows makes it one's own.
# A channel's `form` is the form a norm comes in by it: what children are
# raised in and what one practises is internalized; what one sees others
# keep, or is shamed for breaking, is known (expected).
DEFAULT_CHANNELS: Mapping[str, Mapping[str, float | str]] = {
    'upbringing': {'mutation': 0.05, 'form': culture_lib.INTERNALIZED},
    'observation': {
        'radius': 2,
        'witnessed': 3,
        'mutation': 0.05,
        'form': culture_lib.KNOWN,
    },
    'apprenticeship': {
        'radius': 4,
        'per_master': 2,
        'mutation': 0.02,
        'improve': 0.05,
    },
    'sanction': {'mutation': 0.02, 'form': culture_lib.KNOWN},
    'practice': {'mutation': 0.0, 'form': culture_lib.INTERNALIZED},
}
# Names are syllables: the n-th name is n's digits in base len(_SYLLABLES),
# scattered by a seeded affine map, so every name is new.
_NAME_STRIDE = 7919
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
  # Variants nobody living holds any more, for the run's long record.
  extinct: list[dict[str, Any]] = dataclasses.field(default_factory=list)


class World:
  """The land, the people's lives and the record of customs."""

  def __init__(self, settings: Mapping[str, Any]):
    self.settings = dict(settings)
    self.seed = int(settings.get('seed', 0))
    self.mode = str(settings.get('mode', 'sexual'))
    self.channels: dict[str, dict[str, Any]] = {
        str(k): dict(v)
        for k, v in settings.get('channels', DEFAULT_CHANNELS).items()
    }
    # How far a person sees (who is watched, who is near); the stores past
    # which keeping them while a neighbour goes hungry is seen; how much a
    # person gives when sharing.
    self.sight = int(settings.get('sight', 2))
    self.hoard = float(settings.get('hoard', 6.0))
    self.share = float(settings.get('share', 1.5))
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
    self.named = 0  # Names given so far.
    # This year's answers, by name; pending births, by first parent.
    self.actions: dict[str, str] = {}
    self.expecting: dict[str, list[str]] = {}
    # Observations to deliver at the start of next year, by name.
    self.queued: dict[str, list[str]] = {}
    # Rolling census, a row a year (the long history goes to disk).
    self.history: list[dict[str, Any]] = []
    # Who kept full stores last year and gave nothing to the needy within
    # sight, and those needy: what people may shame this year.
    self.breaches: dict[str, list[str]] = {}
    # Who is looking for a partner (until paired).
    self.seeking: list[str] = []
    # Offers made while closing the year, by channel (for the census).
    self._offers: collections.Counter[str] = collections.Counter()

  # --- Setup -------------------------------------------------------------

  def found(self) -> list[dict[str, Any]]:
    """Creates the founders; returns each one's prefab params.

    A founding custom is held by every founder, or by the share of them its
    `held_by` says (drawn from the seed).

    Returns:
      Each founder's prefab params.
    """
    count = int(self.settings.get('founders', 12))
    holders: dict[str, set[int]] = {}
    for i, spec in enumerate(self.settings.get('founding_customs', ())):
      custom = self.culture.add(
          spec['kind'], spec['domain'], spec['text'], spec.get('params')
      )
      share = float(spec.get('held_by', 1.0))
      order = np.random.default_rng(_seed_of(self.seed, 'held', i))
      holders[custom.id] = {
          int(f) for f in order.permutation(count)[: round(share * count)]
      }
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
          name=self._new_name(),
          born=-int(rng.integers(18, 40)),
          sex=sex,
          tile=tile,
          food=8.0,
          given=[c for c, held in holders.items() if i in held],
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

  def _new_name(self) -> str:
    """A name nobody in this world has had: two syllables, then three..."""
    number, base, size = self.named, len(_SYLLABLES), 2
    self.named += 1
    while number >= base**size:
      number -= base**size
      size += 1
    index = (number * _NAME_STRIDE + self.seed) % base**size
    parts = []
    for _ in range(size):
      index, digit = divmod(index, base)
      parts.append(_SYLLABLES[digit])
    return ''.join(parts).capitalize()

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
      if age >= APPRENTICE_AGE and 'apprenticeship' in self.channels:
        return [PLAY, GATHER, APPRENTICE]
      return [PLAY, GATHER]
    found = [GATHER, REST, MOVE]
    if self._may_have_child(person):
      found.append(CHILD)
    if self.mode == 'sexual' and not person.partner:
      found.append(COURT)
    if 'upbringing' in self.channels and self._pupils(person):
      found.append(TEACH)
    if person.food >= self.share + 1.0:
      needy = sorted(
          (self.people[other].food, other)
          for other in self._in_sight(person)
          if self.people[other].food < 1.0
      )
      found.extend(f'{SHARE} {other}' for _, other in needy[:2])
    if 'sanction' in self.channels:
      found.extend(
          f'{SHAME} {other}'
          for other in self._in_sight(person)
          if other in self.breaches
      )
    return found

  def _in_sight(self, person: Person) -> list[str]:
    """The others living within sight of a person, by name."""
    return [
        other
        for other in self.living()
        if other != person.name and self._distance(person, other) <= self.sight
    ]

  def _pupils(self, person: Person) -> list[str]:
    """A person's children old enough to be taught and not yet grown."""
    return [
        child
        for child in person.children
        if child in self.people
        and CHILD_ACTS_AGE <= self.people[child].age(self.year) < ADULT_AGE
    ]

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
        f' {len(self._in_sight(person))} other people are within sight.'
    )

  # --- Closing the year --------------------------------------------------

  def close_year(self) -> YearOutcome:
    """Resolves the year's answers and moves on a year.

    Births come first, then partners pair. Then each answer takes effect, in
    an order drawn from the year's seed (the commons is shared fairly), and
    what people did passes on what they did it by: to their children, to
    apprentices and to those who saw them. Then everyone eats, some die, the
    land regrows and the variants nobody holds are forgotten.

    Returns:
      The born, the dead and the forgotten variants.
    """
    created = self.culture.created()
    self._offers.clear()
    # The moves each was offered, as the year began.
    allowed = {
        name: [PASS_ON] if self.asked_to_pass_on(name) else self.moves(name)
        for name in self.actions
    }
    births = self._births()
    self._pair_partners()
    order = self._rng('order').permutation(sorted(self.actions))
    did: dict[str, tuple[str, list[str]]] = {}
    for name in (str(n) for n in order):
      if name in self.people:
        did[name] = self._resolve(name, self.actions[name], allowed[name])
    self._apprentice(did)
    self._transmit(did)
    worked = [self._level(e, 'food') for m, e in did.values() if m == GATHER]
    deaths = self._eat_and_die()
    self._see_breaches(did, allowed)
    self.land.regrow()
    extinct = self.culture.prune(
        {c for person in self.people.values() for c in person.given}
    )
    self.history.append(
        self._census(
            births=len(births),
            deaths=len(deaths),
            did=did,
            worked=worked,
            created=self.culture.created() - created,
            extinct=len(extinct),
        )
    )
    del self.history[: -int(self.settings.get('history_kept', 200))]
    self.actions = {}
    self.year += 1
    for name in self.participants():
      self._tell(name, self.status(name))
    return YearOutcome(
        births=births,
        deaths=deaths,
        extinct=[custom.to_dict() for custom in extinct],
    )

  def _rng(self, *keys: Any) -> np.random.Generator:
    return np.random.default_rng(_seed_of(self.seed, self.year, *keys))

  def _resolve(
      self, name: str, action: str, allowed: Sequence[str]
  ) -> tuple[str, list[str]]:
    """One answer's effect; returns the move and the customs it enacted.

    The customs a move enacts are those it names that the GM once gave the
    person: what others can see the person do it by.

    Args:
      name: Who answered.
      action: The answer.
      allowed: The moves the person was offered.

    Returns:
      The move made ('' if none was understood) and the ids it enacted.
    """
    person = self.people[name]
    answer = action.strip().lower()
    move = next(
        (
            m
            for m in sorted(allowed, key=len, reverse=True)
            if answer.startswith(m.lower())
        ),
        '',
    )
    enacted = [
        custom_id
        for custom_id in dict.fromkeys(
            f'c{m}' for m in CUSTOM_ID.findall(answer)
        )
        if custom_id in person.given and custom_id in self.culture
    ]
    if move in (GATHER, APPRENTICE):
      effort = 2.0 if person.age(self.year) >= ADULT_AGE else 1.0
      if move == GATHER:
        effort *= self._skill(enacted, 'food')
      person.food += self.land.gather(person.tile, effort)
    elif move == MOVE:
      person.tile = self.land.best_near(person.tile)
    elif move == CHILD and self._may_have_child(person):
      parents = [name] + ([person.partner] if person.partner else [])
      self.expecting[name] = parents
      for parent in parents:
        self._tell(parent, f'{name} is expecting a child.')
    elif move.startswith(f'{SHARE} '):
      other = self.people.get(move.removeprefix(f'{SHARE} '))
      if other is not None and person.food >= self.share + 1.0:
        person.food -= self.share
        other.food += self.share
        self._tell(other.name, f'{name} shares food with {other.name}.')
    elif move.startswith(f'{SHAME} '):
      other = move.removeprefix(f'{SHAME} ')
      if other in self.people:
        needy = ' or '.join(self.breaches.get(other, ()))
        for custom_id in enacted:
          self._offer(
              other,
              name,
              custom_id,
              'sanction',
              f'{name} shames {other} for giving nothing to {needy}',
          )
    elif move == TEACH:
      pupils = self._pupils(person)
      for pupil in pupils:
        for custom_id in enacted:
          self._offer(pupil, name, custom_id, 'upbringing', f'{name} teaches')
      if pupils and enacted:
        self._tell(name, f'{name} teaches {", ".join(pupils)}.')
    # REST, PLAY, WATCH and anything not understood change nothing.
    return move, enacted

  def _skill(self, enacted: Sequence[str], domain: str) -> float:
    """How much better an enacted skill makes the work (1.0: none).

    A skill counts only if the GM once gave it to this person (`enacted`):
    the world credits know-how it delivered, never what a mind might claim.

    Args:
      enacted: The customs the work was done by.
      domain: The work's domain.

    Returns:
      The effort multiplier.
    """
    return 1.0 + 0.25 * self._level(enacted, domain)

  def _level(self, enacted: Sequence[str], domain: str) -> int:
    """The best level of the skills of a domain among `enacted` (0: none)."""
    return max(
        (
            int(self.culture.get(c).params.get('level', 1))
            for c in enacted
            if self.culture.get(c).kind == 'skill'
            and self.culture.get(c).domain == domain
        ),
        default=0,
    )

  def _offer(
      self, receiver: str, source: str, custom_id: str, via: str, scene: str
  ) -> None:
    """Shows `receiver` a custom `source` enacts, copied as the channel copies.

    The receiver is given the copy (an offer line in next year's
    observation); whether it takes the custom up is its own Learner's choice.

    Args:
      receiver: Who is offered the custom.
      source: Whose custom it is.
      custom_id: The variant enacted.
      via: The channel (a key of `channels`).
      scene: What the receiver sees, e.g. 'Ada teaches'.
    """
    channel = self.channels[via]
    if channel.get('form'):
      custom_id = self.culture.in_form(
          custom_id, str(channel['form']), self.year
      ).id
    custom = self.culture.copy(
        custom_id,
        self._rng('offer', receiver, source, custom_id, via),
        float(channel.get('mutation', 0.0)),
        self.year,
        improve=float(channel.get('improve', 0.0)),
    )
    person = self.people[receiver]
    if custom.id not in person.given:
      person.given.append(custom.id)
    self._offers[via] += 1
    line = culture_lib.offer_line(
        custom.spec(), via, source, self.people[source].food
    )
    self._tell(receiver, f'{scene}: {line}')

  def _apprentice(self, did: Mapping[str, tuple[str, list[str]]]) -> None:
    """Pairs each youth learning a craft with a skilled worker nearby."""
    channel = self.channels.get('apprenticeship')
    if channel is None:
      return
    masters = {
        name: [c for c in enacted if self.culture.get(c).kind == 'skill']
        for name, (move, enacted) in sorted(did.items())
        if move == GATHER and self.people[name].age(self.year) >= ADULT_AGE
    }
    masters = {name: skills for name, skills in masters.items() if skills}
    taken: collections.Counter[str] = collections.Counter()
    for youth in sorted(
        n for n, (move, _) in did.items() if move == APPRENTICE
    ):
      person = self.people[youth]
      near = [
          (self._distance(person, master), master)
          for master in masters
          if taken[master] < int(channel.get('per_master', 2))
          and self._distance(person, master) <= int(channel.get('radius', 4))
      ]
      if not near:
        self._tell(youth, f'{youth} finds nobody nearby to learn a craft from.')
        continue
      master = min(near)[1]
      taken[master] += 1
      self._tell(master, f'{youth} works beside {master} to learn the craft.')
      for custom_id in masters[master]:
        self._offer(
            youth, master, custom_id, 'apprenticeship', f'{master} shows'
        )

  def _transmit(self, did: Mapping[str, tuple[str, list[str]]]) -> None:
    """What people did is seen by their children and by those near them."""
    seen = self.channels.get('observation')
    radius = int(seen.get('radius', 2)) if seen else -1
    witnessed = int(seen.get('witnessed', 3)) if seen else 0
    raised = 'upbringing' in self.channels
    actors = [
        name
        for name, (move, enacted) in sorted(did.items())
        if enacted and move not in (TEACH, '')
    ]
    for name in self.living():
      person = self.people[name]
      age = person.age(self.year)
      if age < CHILD_ACTS_AGE:
        continue
      # A child sees its parents at home, wherever they work.
      parents = [
          actor
          for actor in actors
          if raised and age < ADULT_AGE and actor in person.parents
      ]
      others = sorted(
          (self._distance(person, actor), actor)
          for actor in actors
          if actor != name
          and actor not in parents
          and self._distance(person, actor) <= radius
      )[:witnessed]
      for actor, via in [(p, 'upbringing') for p in parents] + [
          (a, 'observation') for _, a in others
      ]:
        move, enacted = did[actor]
        for custom_id in enacted:
          self._offer(
              name, actor, custom_id, via, f'{name} sees {actor} {move}'
          )
    if 'practice' not in self.channels:
      return
    for actor in actors:
      for custom_id in did[actor][1]:
        custom = self.culture.get(custom_id)
        if culture_lib.form_of(custom.spec()) == culture_lib.KNOWN:
          self._offer(actor, actor, custom_id, 'practice', f'{actor} keeps')

  def _see_breaches(
      self,
      did: Mapping[str, tuple[str, list[str]]],
      allowed: Mapping[str, Sequence[str]],
  ) -> None:
    """Who kept full stores and gave nothing to the needy within sight.

    It is what those near saw, not a judgment: whether it was wrong is for
    each of them to decide (to shame or not, next year).

    Args:
      did: This year's moves.
      allowed: The moves each was offered this year.
    """
    self.breaches = {}
    for name in self.living():
      person = self.people[name]
      needy = [
          m.removeprefix(f'{SHARE} ')
          for m in allowed.get(name, ())
          if m.startswith(f'{SHARE} ')
      ]
      if (
          not needy
          or person.food < self.hoard
          or did.get(name, ('', []))[0].startswith(SHARE)
      ):
        continue
      self.breaches[name] = needy
      for witness in self._in_sight(person):
        if witness not in needy:
          self._tell(
              witness,
              f'{name} kept full stores and gave nothing to'
              f' {" or ".join(needy)}, who had none.',
          )

  def _pair_partners(self) -> None:
    """Pairs those looking for a partner, nearest first, in name order.

    Whoever looked for a partner this year or before, and has none, is
    looking until paired. Parents and children, and siblings, never pair.
    """
    self.seeking = sorted(
        name
        for name in {
            *self.seeking,
            *(
                n
                for n, answer in self.actions.items()
                if answer.strip().lower().startswith(COURT)
            ),
        }
        if name in self.people and not self.people[name].partner
    )
    taken: set[str] = set()
    for name in self.seeking:
      if name in taken:
        continue
      person = self.people[name]
      matches = [
          other
          for other in self.seeking
          if other not in taken
          and other != name
          and self.people[other].sex not in ('', person.sex)
          and not self._kin(person, self.people[other])
      ]
      if not matches:
        continue
      other = min(matches, key=lambda o: (self._distance(person, o), o))
      taken.update((name, other))
      person.partner, self.people[other].partner = other, name
      self._tell(name, f'{name} and {other} are now partners.')
      self._tell(other, f'{other} and {name} are now partners.')
    self.seeking = [name for name in self.seeking if name not in taken]

  def _kin(self, person: Person, other: Person) -> bool:
    """Whether two people are parent and child, or siblings."""
    return (
        other.name in person.parents
        or person.name in other.parents
        or bool(set(person.parents) & set(other.parents))
    )

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
          name=self._new_name(),
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
    del self.dead[:-DEAD_KEPT]
    if person.partner in self.people:
      self.people[person.partner].partner = ''
    for kin in {*person.parents, *person.children, person.partner}:
      if kin in self.people:
        self._tell(kin, f'{name} has died, aged {person.age(self.year)}.')
    for parents in list(self.expecting.values()):
      if name in parents:
        parents.remove(name)
    self.queued.pop(name, None)

  def _census(
      self,
      births: int,
      deaths: int,
      did: Mapping[str, tuple[str, list[str]]],
      worked: Sequence[int],
      created: int,
      extinct: int,
  ) -> dict[str, Any]:
    """The year in numbers, from what the GM saw (never from minds)."""
    ages = [p.age(self.year) for p in self.people.values()]
    return {
        'year': self.year,
        'population': len(self.people),
        'births': births,
        'deaths': deaths,
        'mean_age': round(sum(ages) / len(ages), 2) if ages else 0.0,
        'food': round(float(self.land.food.sum()), 3),
        'moves': dict(
            sorted(
                collections.Counter(
                    _move_kind(m) for m, _ in did.values()
                ).items()
            )
        ),
        'breaches': len(self.breaches),
        'offers': dict(sorted(self._offers.items())),
        'variants': len(self.culture),
        'new_variants': created,
        'lost_variants': extinct,
        'skill_worked': round(sum(worked) / len(worked), 2) if worked else 0.0,
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
        'dead': list(self.dead),
        'named': self.named,
        'actions': dict(self.actions),
        'expecting': {k: list(v) for k, v in self.expecting.items()},
        'queued': {k: list(v) for k, v in self.queued.items()},
        'history': list(self.history),
        'breaches': {k: list(v) for k, v in self.breaches.items()},
        'seeking': list(self.seeking),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.year = int(state['year'])
    self.land.set_state(state['land'])
    self.culture.set_state(state['culture'])
    self.people = {
        str(p['name']): Person.from_dict(p) for p in state.get('people', ())
    }
    self.dead = [str(d) for d in state.get('dead', ())]
    self.named = int(state.get('named', 0))
    self.actions = {str(k): str(v) for k, v in state.get('actions', {}).items()}
    self.expecting = {
        str(k): [str(p) for p in v]
        for k, v in state.get('expecting', {}).items()
    }
    self.queued = {
        str(k): [str(t) for t in v] for k, v in state.get('queued', {}).items()
    }
    self.history = [dict(r) for r in state.get('history', ())]
    self.breaches = {
        str(k): [str(n) for n in v]
        for k, v in state.get('breaches', {}).items()
    }
    self.seeking = [str(n) for n in state.get('seeking', ())]


def _move_kind(move: str) -> str:
  """A move without the one it is made towards (for counting)."""
  for kind in (SHARE, SHAME):
    if move.startswith(f'{kind} '):
      return kind
  return move
