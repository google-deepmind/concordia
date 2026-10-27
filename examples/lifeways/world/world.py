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
from examples.lifeways.world import places as places_lib
import numpy as np

EXPERIMENTER = 'Experimenter'
# The Experimenter's yearly status, and what reads the year from it.
EXPERIMENTER_STATUS_FORM = 'Year {year}. {people} people live in the valley.'
EXPERIMENTER_STATUS = re.compile(
    r'\bYear (\d+)\. \d+ people live in the valley\.'
)
SEXUAL = 'sexual'
ASEXUAL = 'asexual'
# Whom a child learns to speak from (the `speech_from` setting): its mother,
# or one of the grown around her, drawn (children pick up the local tongue).
MOTHER = 'mother'
COMMUNITY = 'community'

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
FOUND_LIBRARY = 'found a library'
WRITE = 'write at the library'
READ = 'read at the library'
FOUND_SCHOOL = 'found a school'
TEACH_AT_SCHOOL = 'teach at the school'
GO_TO_SCHOOL = 'go to school'
BUILD_SHRINE = 'build a shrine'
LEAD_RITE = 'lead the rite'
JOIN_RITE = 'join the rite'
TELL = 'tell a story'
PREACH = 'preach'
COURT_ONE = 'court'  # ... a name: look for that partner only.
SOW = 'sow a field'
# Gathering that leaves alone the fields others tend (offered where there
# are some within a day's walk): taking from them is no breach of the
# World's, but a people's norms may make it one.
SPARE = 'gather food, sparing the fields of others'
# The Experimenter's: end the run (the only way a Sandbox ends).
STOP = 'stop the world'
# The kinds of question the World GM asks (what an answer answers).
INTERVIEW = 'interview'
PASSING_ON = 'pass_on'
MOVES = 'move'
# The Experimenter's: put a question to a person, who answers it next year.
ASK_FORM = 'ask <name>: <question>'
_ASK = re.compile(r'^ask (?P<name>[A-Za-z]+): (?P<question>.+)$')
TRANSCRIPTS_KEPT = 50
# The Experimenter's: release a custom it designed into the world.
RELEASE = 'release'
RELEASE_FORM = (
    'release <kind> <domain> as a <text or song> at <row>,<col>: <what it says>'
)
# A skill may name its level; a marriage rule whom it allows (`within`
# tongue or descent; anyone if not said).
_RELEASE = re.compile(
    r'^release (?P<kind>[a-z]+) (?P<domain>[a-z]+)(?: level (?P<level>\d+))?'
    r'(?: within (?P<within>[a-z]+))?'
    r' as a (?P<how>text|song) at (?P<row>\d+),(?P<col>\d+): (?P<text>.+)$'
)
# Disasters: the world sends one by chance (the `disasters`
# setting), and the Experimenter may send one at a place. Each strikes the
# land and people within a few steps of its place.
DROUGHT, FLOOD, FIRE, PLAGUE, RAID = (
    'drought',
    'flood',
    'fire',
    'plague',
    'raid',
)
DISASTERS = (DROUGHT, FLOOD, FIRE, PLAGUE, RAID)
SEND = 'send'
DISASTER_FORM = 'send a <drought, flood, fire, plague or raid> at <row>,<col>'
_SEND = re.compile(
    r'^send an? (?P<kind>[a-z]+) at (?P<row>\d+)\s*,\s*(?P<col>\d+)\.?$'
)
DISASTER_RADIUS = 3
# A flood takes what grows on low land; a fire, what grows on grass and in
# forest, and the texts on a library's shelf.
_FLOODED = ('river', 'coast', 'marsh')
_BURNED = ('grass', 'forest')
# The Experimenter's: bring strangers of the arrivals' people to a place
# (strangers with no customs of their own if the world has no arrivals).
DEFAULT_ARRIVALS: Mapping[str, Any] = {
    'people': 'Strangers',
    'customs': (),
    'food': 6.0,
    'group': 4,
}
BRING = 'bring migrants'
# The Experimenter's: found an institution at a place, or close one.
FOUND_PLACE = 'found'
CLOSE_PLACE = 'close'
INSTITUTION_FORMS = (
    'found a <library, school or shrine> at <row>,<col>',
    'close the <library, school or shrine> at <row>,<col>',
)
_INSTITUTION = re.compile(
    r'^(?P<act>found|close) (?:a|the) (?P<kind>library|school|shrine) at'
    r' (?P<row>\d+)\s*,\s*(?P<col>\d+)\.?$'
)
MIGRANTS_FORM = 'bring migrants at <row>,<col>'
_BRING = re.compile(r'^bring migrants at (?P<row>\d+)\s*,\s*(?P<col>\d+)\.?$')

# The deeds others see a person do by its customs, and so may copy. Teaching,
# schooling, writing, reading and rites reach children, pupils, shelves and
# worshippers, not onlookers; passing on and joining are words, not deeds.
WITNESSED = (
    GATHER,
    REST,
    MOVE,
    CHILD,
    COURT,
    COURT_ONE,
    SHARE,
    SHAME,
    SOW,
    SPARE,
)

ADULT_AGE = 16
APPRENTICE_AGE = 12
CHILD_ACTS_AGE = 5
ELDER_AGE = 56
FERTILE = (16, 45)
CUSTOM_ID = re.compile(r'\bc(\d+)\b')
# The dead the world remembers, for kin (their lives go to the run's archive).
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
# - practice: keeping a norm one knows makes it one's own;
# - reading: a literate reader at a library reads `texts` of its shelf, exact;
# - school: pupils are taught the curriculum of a school within sight, in a
#   teacher's term;
# - rite: those who join a shrine's rite, in a year it falls and someone
#   leads it, are offered the customs it is led by;
# - story: an elder's story reaches all within sight, each retelling a
#   paraphrase (often a new variant);
# - preaching: a preacher's one custom reaches all within sight; learners
#   take up preached beliefs and values more readily than other kinds.
# A channel's `form` is the form a norm comes in by it: what children are
# raised in, what one practises and what one is shamed for breaking may
# become one's own (internalized); what one sees others keep is known
# (expected).
DEFAULT_CHANNELS: Mapping[str, Mapping[str, float | str]] = {
    culture_lib.UPBRINGING: {
        'mutation': 0.05,
        'form': culture_lib.INTERNALIZED,
    },
    culture_lib.OBSERVATION: {
        'radius': 2,
        'witnessed': 3,
        'mutation': 0.05,
        'form': culture_lib.KNOWN,
    },
    culture_lib.APPRENTICESHIP: {
        'radius': 4,
        'per_master': 2,
        'mutation': 0.02,
        'improve': 0.05,
    },
    culture_lib.SANCTION: {'mutation': 0.02, 'form': culture_lib.INTERNALIZED},
    culture_lib.PRACTICE: {'mutation': 0.0, 'form': culture_lib.INTERNALIZED},
    culture_lib.READING: {'mutation': 0.0, 'texts': 3},
    culture_lib.SCHOOL: {'mutation': 0.01, 'form': culture_lib.INTERNALIZED},
    culture_lib.RITE: {'mutation': 0.0, 'form': culture_lib.INTERNALIZED},
    culture_lib.STORY: {'mutation': 0.2},
    culture_lib.PREACHING: {'mutation': 0.05},
    culture_lib.RELEASE: {'mutation': 0.0},
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


def name_of(number: int, seed: int) -> str:
  """A world's `number`-th name: two syllables, then three; all distinct.

  Args:
    number: Which name (founders take the first ones).
    seed: The world seed, which scatters the syllables.

  Returns:
    The name.
  """
  base, size = len(_SYLLABLES), 2
  while number >= base**size:
    number -= base**size
    size += 1
  index = (number * _NAME_STRIDE + seed) % base**size
  parts = []
  for _ in range(size):
    index, digit = divmod(index, base)
    parts.append(_SYLLABLES[digit])
  return ''.join(parts).capitalize()


def seed_of(*keys: Any) -> list[int]:
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
  # The year the person came to the valley, if not born in it.
  arrived: int | None = None
  # The dialect the person speaks, as raised (its id): how others hear it.
  speech: str = ''
  # The people the person is of by descent (its mother's).
  descent: str = ''

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
        arrived=(None if data.get('arrived') is None else int(data['arrived'])),
        speech=str(data.get('speech', '')),
        descent=str(data.get('descent', '')),
    )


@dataclasses.dataclass
class YearOutcome:
  """What closing a year changed beyond the world itself."""

  # Prefab params of each person born or arrived, to add to the simulation.
  births: list[dict[str, Any]]
  # Names of the people who died, to remove from it.
  deaths: list[str]
  # Variants nobody living holds any more, for the run's long record.
  extinct: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  # Whether the Experimenter ended the run this year.
  stopped: bool = False


class World:
  """The land, the people's lives and the record of customs."""

  def __init__(self, settings: Mapping[str, Any]):
    self.settings = dict(settings)
    self.seed = int(settings.get('seed', 0))
    self.mode = str(settings.get('mode', SEXUAL))
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
    # Libraries, if the setting allows them: what founding one costs, how
    # far apart they stand, how many texts a shelf holds, the years unused
    # before one is a ruin, and how fast a ruin's texts crumble.
    library = settings.get('library')
    self.library: dict[str, Any] | None = (
        None if library is None else dict(library)
    )
    # Schools, if the setting allows them: what founding one costs, how far
    # apart they stand and the years a teacher's term lasts.
    school = settings.get('school')
    self.school: dict[str, Any] | None = (
        None if school is None else dict(school)
    )
    # Shrines, if the setting allows them: what building one costs, how far
    # apart they stand and every how many years their rite falls.
    shrine = settings.get('shrine')
    self.shrine: dict[str, Any] | None = (
        None if shrine is None else dict(shrine)
    )
    library = self.library or {}
    self.places = places_lib.Places(
        int(settings.get('rows', 16)),
        int(settings.get('cols', 24)),
        shelf=int(library.get('shelf', 12)),
        ruin_after=int(library.get('ruin_after', 30)),
        crumble=float(library.get('crumble', 0.1)),
    )
    self.year = 0
    self.land = land_lib.Land(
        int(settings.get('rows', 16)),
        int(settings.get('cols', 24)),
        self.seed,
        richness=float(settings.get('richness', 0.45)),
        regrowth=float(settings.get('regrowth', land_lib.REGROWTH)),
        seed_bank=float(settings.get('seed_bank', land_lib.SEED_BANK)),
        ridge=settings.get('ridge'),
        ridges=settings.get('ridges', ()),
        field_yield=float(
            (settings.get('farming') or {}).get('yield', land_lib.FIELD_YIELD)
        ),
    )
    # Farming (`yield`, `fallow`): one given a farming skill may sow the land
    # it lives on; None, and nobody may.
    self.farming: Mapping[str, Any] | None = settings.get('farming')
    # The deeds others see (`seen_deeds`, of `WITNESSED`; all by default),
    # whose customs onlookers may copy (observation) and children learn from
    # their parents (upbringing).
    self.seen_deeds = tuple(settings.get('seen_deeds', WITNESSED))
    unknown = set(self.seen_deeds) - set(WITNESSED)
    if unknown:
      raise ValueError(f'No deed can be seen of {sorted(unknown)}.')
    self.culture = culture_lib.Culture()
    self.people: dict[str, Person] = {}
    # The latest dead, as the valley remembers them (for kin): name, sex,
    # born, died, parents, children, descent.
    self.dead: list[dict[str, Any]] = []
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
    # Questions the Experimenter put, by person, to answer this year; and
    # the answers given (the latest few).
    self.interviews: dict[str, str] = {}
    self.transcripts: list[dict[str, Any]] = []
    # What each participant was last asked, and in which year.
    self.asked: dict[str, list[Any]] = {}
    # Who is looking for a partner (until paired), and for whom: a name, or
    # '' for anyone.
    self.courting: dict[str, str] = {}
    # Offers made while closing the year, by channel, and how many readings
    # brought back a custom nobody living had been given (for the census).
    self._offers: collections.Counter[str] = collections.Counter()
    self._rediscovered = 0
    self._released = 0
    self._sown = 0
    # This year's releases: id, how, where, and how many heard a song.
    self._releases: list[dict[str, Any]] = []
    # This year's disasters: kind, place, who sent it, what it took.
    self._disasters: list[dict[str, Any]] = []
    # What the World told the Experimenter of its moves this year.
    self._reports: list[str] = []

  # --- Setup -------------------------------------------------------------

  def found(self, founders: Sequence[Mapping[str, Any]]) -> None:
    """Settles the founders the scenario made (`scenarios.founders`).

    Args:
      founders: For each founder, its person prefab's `params` (name, birth
        year, sex, customs with their ids), its `tile`, its `food` and its
        `people`.
    """
    for founder in founders:
      params = founder['params']
      for spec in params.get('customs', ()):
        self.culture.adopt(spec)
      person = Person(
          name=str(params['name']),
          born=int(params['born']),
          sex=str(params.get('sex', '')),
          tile=(int(founder['tile'][0]), int(founder['tile'][1])),
          food=float(founder['food']),
          given=[str(spec['id']) for spec in params.get('customs', ())],
          speech=self._dialect(params.get('customs', ())),
          descent=str(founder.get('people', self.settings.get('people', ''))),
      )
      self.people[person.name] = person
      self._tell(person.name, self._birth_story(person, founder=True))
    self.named = max(self.named, len(founders))

  def _params(self, person: Person) -> dict[str, Any]:
    """The prefab params of a person's entity."""
    params = {
        'name': person.name,
        'born': person.born,
        'sex': person.sex,
        'parents': list(person.parents),
        'seed': self.seed,
        'customs': [self.culture.get(c).spec() for c in person.given],
    }
    if person.arrived is not None:
      params['arrived'] = person.arrived
    return params

  def _arrive(self) -> list[dict[str, Any]]:
    """Strangers who come to a nearly empty valley, with their own customs.

    The `arrivals` setting: while fewer than `below` people live, each year
    a `group` of strangers of a people (`people`) arrives with chance
    `chance`, at a place drawn from the seed, each holding that people's
    `customs` and `food` stores.

    Returns:
      The prefab params of each who arrived.
    """
    arrivals = self.settings.get('arrivals')
    if not arrivals or len(self.people) >= int(arrivals.get('below', 20)):
      return []
    rng = self._rng('arrivals')
    if rng.random() >= float(arrivals.get('chance', 0.25)):
      return []
    tile = (
        int(rng.integers(self.land.rows)),
        int(rng.integers(self.land.cols)),
    )
    return self._strangers(arrivals, tile, rng, int(arrivals.get('group', 2)))

  def _strangers(
      self,
      arrivals: Mapping[str, Any],
      tile: land_lib.Tile,
      rng: np.random.Generator,
      group: int,
  ) -> list[dict[str, Any]]:
    """A group of strangers of the arrivals' people comes to a place.

    Args:
      arrivals: The `arrivals` setting (their people, customs and stores).
      tile: Where they come.
      rng: The year's draw for their ages.
      group: How many come.

    Returns:
      The prefab params of each who came.
    """
    customs = [
        self.culture.add(
            c['kind'],
            c['domain'],
            c['text'],
            c.get('params'),
            origin=culture_lib.ARRIVAL,
            year=self.year,
        ).id
        for c in arrivals.get('customs', ())
    ]
    people = str(arrivals.get('people', 'strangers'))
    came = []
    for i in range(group):
      person = Person(
          name=self._new_name(),
          born=self.year - int(rng.integers(18, 31)),
          sex='' if self.mode == ASEXUAL else ('f' if i % 2 == 0 else 'm'),
          tile=tile,
          food=float(arrivals.get('food', 6.0)),
          given=list(customs),
          arrived=self.year,
          speech=self._dialect([self.culture.get(c).spec() for c in customs]),
          descent=people,
      )
      self.people[person.name] = person
      self._tell(
          person.name,
          f'{person.name} comes to the valley with the {people}.',
          newborn=True,
      )
      came.append(self._params(person))
    return came

  def _bring(self, answer: str) -> list[dict[str, Any]]:
    """The Experimenter brings strangers to a place (`MIGRANTS_FORM`).

    They are of the `arrivals` setting's people, with its customs and
    stores; at least four come.

    Args:
      answer: The Experimenter's answer this year.

    Returns:
      The prefab params of each who came.
    """
    answer = answer.strip()
    if not answer.lower().startswith(BRING):
      return []
    match = _BRING.match(answer.lower())
    arrivals = self.settings.get('arrivals') or DEFAULT_ARRIVALS
    if match is None:
      self._report('The migrants are not understood: ' + answer)
      return []
    tile = self.land.wrap(int(match.group('row')), int(match.group('col')))
    group = max(4, int(arrivals.get('group', 2)))
    came = self._strangers(arrivals, tile, self._rng('bring'), group)
    self._report(
        f'{len(came)} of the {arrivals.get("people", "strangers")} come to'
        f' {tile[0]},{tile[1]}: {", ".join(p["name"] for p in came)}.'
    )
    return came

  def _institution(self, answer: str) -> None:
    """The Experimenter founds or closes an institution (`INSTITUTION_FORMS`).

    It founds one where the world has that kind of place and none stands
    within its spacing; it closes the one of that kind at the place, or
    within a step of it, and what it held is gone.

    Args:
      answer: The Experimenter's answer this year.
    """
    answer = answer.strip()
    head = answer.lower().split(' ', 1)[0]
    if head not in (FOUND_PLACE, CLOSE_PLACE):
      return
    match = _INSTITUTION.match(answer.lower())
    if match is None:
      self._report('The institution is not understood: ' + answer)
      return
    kind = match.group('kind')
    tile = self.land.wrap(int(match.group('row')), int(match.group('col')))
    where = f'{tile[0]},{tile[1]}'
    if match.group('act') == FOUND_PLACE:
      setting = {
          places_lib.LIBRARY: self.library,
          places_lib.SCHOOL: self.school,
          places_lib.SHRINE: self.shrine,
      }[kind]
      if setting is None:
        self._report(f'This world has no {kind}s.')
        return
      spacing = int(setting.get('spacing', 5))
      if self.places.near(tile, spacing, kind):
        self._report(f'A {kind} stands within {spacing} steps of {where}.')
        return
      self.places.found(kind, tile, EXPERIMENTER, self.year)
      self._report(f'A {kind} is founded at {where}; it stands empty.')
      return
    near = self.places.near(tile, 1, kind)
    if not near:
      self._report(f'No {kind} stands at {where}.')
      return
    place = near[0]
    self.places.close(place)
    self._report(
        f'The {kind} at {place.tile[0]},{place.tile[1]} is closed;'
        f' {len(place.shelf)} customs it held are gone with it.'
    )

  def _send(self, answer: str) -> list[str]:
    """The Experimenter sends a disaster to a place (`DISASTER_FORM`).

    Args:
      answer: The Experimenter's answer this year.

    Returns:
      Who died of it.
    """
    answer = answer.strip()
    if not answer.lower().startswith(SEND):
      return []
    match = _SEND.match(answer.lower())
    if match is None or match.group('kind') not in DISASTERS:
      self._report('The disaster is not understood: ' + answer)
      return []
    tile = self.land.wrap(int(match.group('row')), int(match.group('col')))
    return self._strike(match.group('kind'), tile, EXPERIMENTER)

  def _disaster(self) -> list[str]:
    """By chance, a disaster the world sends (the `disasters` setting).

    Each year one strikes with chance `chance`, of a kind drawn from `kinds`,
    at a place drawn from the seed.

    Returns:
      Who died of it.
    """
    disasters = self.settings.get('disasters')
    if not disasters:
      return []
    rng = self._rng('disaster')
    if rng.random() >= float(disasters.get('chance', 0.0)):
      return []
    kinds = [k for k in disasters.get('kinds', DISASTERS) if k in DISASTERS]
    if not kinds:
      return []
    kind = kinds[int(rng.integers(len(kinds)))]
    tile = (
        int(rng.integers(self.land.rows)),
        int(rng.integers(self.land.cols)),
    )
    return self._strike(kind, tile, 'world')

  def _strike(self, kind: str, tile: land_lib.Tile, by: str) -> list[str]:
    """A disaster strikes the land, places and people near a tile.

    - drought: what grows within reach falls to a fifth;
    - flood: low land (river, coast, marsh) loses what grows, and its places
      their texts;
    - fire: grass and forest lose what grows, and libraries their texts;
    - plague: each within reach may die, the old and the young most;
    - raid: raiders take the stores of all within reach, and may kill.

    Args:
      kind: One of `DISASTERS`.
      tile: Where it strikes.
      by: Who sent it: the Experimenter, or 'world'.

    Returns:
      Who died of it.
    """
    radius = int(
        (self.settings.get('disasters') or {}).get('radius', DISASTER_RADIUS)
    )
    reach = [
        t
        for t in self.land.around(tile, radius)
        if self.places.distance(tile, t) <= radius
    ]
    near = [
        name
        for name in self.living()
        if self.places.distance(self.people[name].tile, tile) <= radius
    ]
    rng = self._rng('strike', kind, *tile)
    lost = 0.0
    texts = 0
    dead: list[str] = []
    if kind in (DROUGHT, FLOOD, FIRE):
      struck = {
          DROUGHT: reach,
          FLOOD: [t for t in reach if self.land.kind(t) in _FLOODED],
          FIRE: [t for t in reach if self.land.kind(t) in _BURNED],
      }[kind]
      keep = 0.2 if kind == DROUGHT else 0.0
      for t in struck:
        lost += float(self.land.food[t]) * (1.0 - keep)
        self.land.food[t] *= keep
      if kind in (FLOOD, FIRE):
        for place in self.places.all():
          if place.tile in struck and (
              kind == FLOOD or place.kind == places_lib.LIBRARY
          ):
            texts += len(place.shelf)
            place.shelf.clear()
    elif kind == PLAGUE:
      for name in near:
        age = self.people[name].age(self.year)
        chance = 0.5 if age < CHILD_ACTS_AGE or age >= 50 else 0.25
        if rng.random() < chance:
          dead.append(name)
    elif kind == RAID:
      for name in near:
        person = self.people[name]
        lost += person.food
        person.food = 0.0
        if person.age(self.year) >= ADULT_AGE and rng.random() < 0.1:
          dead.append(name)
    what = {
        DROUGHT: 'A drought withers the land around',
        FLOOD: 'A flood drowns the low land around',
        FIRE: 'A fire sweeps the land around',
        PLAGUE: 'A plague comes to',
        RAID: 'Raiders fall upon',
    }[kind]
    for name in near:
      fate = ' and takes many lives' if dead and kind == PLAGUE else ''
      self._tell(name, f'{what} {name}{fate}.')
    for name in dead:
      self._die(name)
    record = {
        'kind': kind,
        'tile': list(tile),
        'by': by,
        'near': len(near),
        'dead': len(dead),
        'food_lost': round(lost, 1),
        'texts_lost': texts,
    }
    self._disasters.append(record)
    told = (
        f'A {kind} strikes at {tile[0]},{tile[1]}: {len(near)} lived within'
        f' reach, {len(dead)} died'
        + (f', {round(lost, 1)} food was lost' if lost else '')
        + (f', {texts} texts were lost' if texts else '')
        + '.'
    )
    if by == EXPERIMENTER:
      self._report(told)
    else:
      self._tell(EXPERIMENTER, told)
    return dead

  def _new_name(self) -> str:
    """The world's next name (one the living do not bear)."""
    while True:
      name = name_of(self.named, self.seed)
      self.named += 1
      if name not in self.people:
        return name

  # --- Who acts ----------------------------------------------------------

  def _hears(self, person: Person, tile: land_lib.Tile) -> bool:
    """Whether a person hears a song sung at a tile (old enough, in sight)."""
    return (
        self.places.distance(person.tile, tile) <= self.sight
        and person.age(self.year) >= CHILD_ACTS_AGE
    )

  def song_reach(self) -> dict[land_lib.Tile, int]:
    """How many would hear a song sung at each tile where anyone would.

    The rule is a released song's (`_hears`); the Experimenter's page reads it
    to say who a song would reach.

    Returns:
      Each tile some would hear a song at, and how many.
    """
    reach: collections.Counter[land_lib.Tile] = collections.Counter()
    for name in self.living():
      person = self.people[name]
      near = {
          self.land.wrap(person.tile[0] + dr, person.tile[1] + dc)
          for dr in range(-self.sight, self.sight + 1)
          for dc in range(-self.sight, self.sight + 1)
      }
      reach.update(tile for tile in near if self._hears(person, tile))
    return dict(reach)

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
    """Whether a participant has a question to answer now.

    Args:
      name: The participant.

    Returns:
      True if it has not made this year's move, or owes an interview.
    """
    return name in self.participants() and (
        name not in self.actions or name in self.interviews
    )

  def mark_asked(self, name: str, question: str) -> None:
    """Notes which question a participant was just asked, and in which year.

    Args:
      name: The participant.
      question: `INTERVIEW`, `PASSING_ON` or `MOVES` (the kind of question).
    """
    self.asked[name] = [question, self.year]

  def record_action(self, name: str, action: str) -> None:
    """Files a participant's answer by the question it answers.

    An answer to a question of a year already closed (in free mode, a year
    closes at its deadline) is dropped: who answered late did nothing that
    year. With no question noted (a direct caller), an interview pending is
    answered first, then the move.

    Args:
      name: Who answered.
      action: The answer.
    """
    asked = self.asked.pop(name, None)
    if asked is not None:
      question, year = asked
      if int(year) != self.year:
        return
      if question == INTERVIEW:
        if name in self.interviews:
          self._answered(name, action)
        return
    elif name in self.interviews:
      self._answered(name, action)
      return
    if name in self.participants() and name not in self.actions:
      self.actions[name] = action

  def interview(self, name: str) -> str:
    """The question put to a person this year, unanswered.

    Args:
      name: The person.

    Returns:
      The question ('' if none).
    """
    return self.interviews.get(name, '')

  def _answered(self, name: str, answer: str) -> None:
    question = self.interviews.pop(name)
    self.transcripts.append({
        'year': self.year,
        'name': name,
        'question': question,
        'answer': answer,
    })
    del self.transcripts[:-TRANSCRIPTS_KEPT]
    self._report(f'{name}, asked "{question}", says: "{answer}"')

  def year_complete(self) -> bool:
    return all(name in self.actions for name in self.participants())

  # --- What each is asked ------------------------------------------------

  def moves(self, name: str) -> list[str]:
    """A person's moves this year (the Experimenter only watches)."""
    if name == EXPERIMENTER:
      return [
          WATCH,
          RELEASE_FORM,
          ASK_FORM,
          DISASTER_FORM,
          MIGRANTS_FORM,
          *INSTITUTION_FORMS,
          STOP,
      ]
    person = self.people[name]
    age = person.age(self.year)
    if age < ADULT_AGE:
      places = [
          *self._library_moves(person),
          *self._school_moves(person),
          *self._shrine_moves(person),
      ]
      spare = [SPARE] if self._fields_of_others(person) else []
      if age >= APPRENTICE_AGE and culture_lib.APPRENTICESHIP in self.channels:
        return [PLAY, GATHER, *spare, APPRENTICE, *places]
      return [PLAY, GATHER, *spare, *places]
    found = [
        GATHER,
        *([SPARE] if self._fields_of_others(person) else []),
        REST,
        MOVE,
        *self._library_moves(person),
        *self._school_moves(person),
        *self._shrine_moves(person),
    ]
    if (
        self.farming
        and self._knows(person, culture_lib.FARMING)
        and self.land.sowable(person.tile)
    ):
      found.append(SOW)
    if self._may_have_child(person):
      found.append(CHILD)
    if self.mode == SEXUAL and not person.partner:
      found.append(COURT)
      found.extend(f'{COURT_ONE} {other}' for other in self._suitors(person))
    if culture_lib.UPBRINGING in self.channels and self._pupils(person):
      found.append(TEACH)
    if (
        culture_lib.STORY in self.channels
        and person.age(self.year) >= ELDER_AGE
    ):
      found.append(TELL)
    if culture_lib.PREACHING in self.channels:
      found.append(PREACH)
    if person.food >= self.share + 1.0:
      needy = sorted(
          (self.people[other].food, other)
          for other in self._in_sight(person)
          if self.people[other].food < 1.0
      )
      found.extend(f'{SHARE} {other}' for _, other in needy[:2])
    if culture_lib.SANCTION in self.channels:
      found.extend(
          f'{SHAME} {other}'
          for other in self._in_sight(person)
          if other in self.breaches
      )
    return found

  def _library_moves(self, person: Person) -> list[str]:
    """Reading where a shelf is within sight; writing and founding (adults)."""
    if self.library is None:
      return []
    near = self.places.near(person.tile, self.sight)
    found = []
    if culture_lib.READING in self.channels and any(p.shelf for p in near):
      found.append(READ)
    if person.age(self.year) < ADULT_AGE:
      return found
    if any(not self.places.ruined(p, self.year) for p in near):
      found.append(WRITE)
    if not self.places.near(
        person.tile, int(self.library.get('spacing', 5))
    ) and person.food >= float(self.library.get('cost', 3.0)):
      found.append(FOUND_LIBRARY)
    return found

  def _school_moves(self, person: Person) -> list[str]:
    """Going to school (children); teaching at or founding one (adults)."""
    if self.school is None or culture_lib.SCHOOL not in self.channels:
      return []
    near = self._standing(person, places_lib.SCHOOL)
    if person.age(self.year) < ADULT_AGE:
      return [GO_TO_SCHOOL] if near else []
    if near:
      return [TEACH_AT_SCHOOL]
    if not self.places.near(
        person.tile, int(self.school.get('spacing', 5)), places_lib.SCHOOL
    ) and person.food >= float(self.school.get('cost', 3.0)):
      return [FOUND_SCHOOL]
    return []

  def _shrine_moves(self, person: Person) -> list[str]:
    """Joining or leading a rite that falls this year; building a shrine."""
    if self.shrine is None or culture_lib.RITE not in self.channels:
      return []
    near = [
        p
        for p in self._standing(person, places_lib.SHRINE)
        if self.rite_falls(p)
    ]
    if near:
      if person.age(self.year) < ADULT_AGE:
        return [JOIN_RITE]
      return [JOIN_RITE, LEAD_RITE]
    if (
        person.age(self.year) >= ADULT_AGE
        and not self.places.near(
            person.tile, int(self.shrine.get('spacing', 5)), places_lib.SHRINE
        )
        and person.food >= float(self.shrine.get('cost', 3.0))
    ):
      return [BUILD_SHRINE]
    return []

  def rite_falls(self, shrine: places_lib.Place) -> bool:
    """Whether a shrine's rite falls this year (on its calendar).

    Args:
      shrine: The shrine.

    Returns:
      True in a year of its rite (every `every` years from its founding).
    """
    every = int((self.shrine or {}).get('every', 4))
    return (self.year - shrine.founded) % every == 0 and (
        self.year > shrine.founded
    )

  def _standing(self, person: Person, kind: str) -> list[places_lib.Place]:
    """The places of a kind within sight that are not ruins, nearest first."""
    return [
        p
        for p in self.places.near(person.tile, self.sight, kind)
        if not self.places.ruined(p, self.year)
    ]

  def _suitors(self, person: Person, most: int = 3) -> list[str]:
    """The grown singles a person may court (not kin): first those courting
    the person, then the nearest others, `most` of them."""
    singles = [
        other
        for other in self.living()
        if other != person.name
        and not self.people[other].partner
        and self.people[other].age(self.year) >= ADULT_AGE
        and self.people[other].sex not in ('', person.sex)
        and not self._kin(person, self.people[other])
    ]
    wooing = [o for o in singles if self.courting.get(o) == person.name]
    nearest = sorted(
        (o for o in singles if o not in wooing),
        key=lambda o: (self._distance(person, o), o),
    )
    return wooing + nearest[:most]

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
    """Whether a person is a parent of a child born this year's close.

    Args:
      name: The person.

    Returns:
      True if the person is asked what it passes on.
    """
    return any(name in parents for parents in self.expecting.values())

  def heir(self, name: str, born: Sequence[str]) -> str:
    """Whom a person who died lives on as, among those just born.

    A person someone plays lives on after death as the first of the born who
    is of its line (a descendant: its child, if one is born the year it
    dies). If nobody of its line lives, it lives on as the first born of its
    people, and if nobody of its people lives either, as the first born of
    all.

    Args:
      name: The person who died.
      born: Those just born, in the order they were born.

    Returns:
      The heir, or '' if none of the born will do (the person waits for the
      next birth in its line, or of its people).
    """
    records = {record['name']: record for record in self.dead}

    def children(who: str) -> Sequence[str]:
      if who in self.people:
        return self.people[who].children
      return records.get(who, {}).get('children', ())

    line, kin = set(), [name]
    while kin:
      for child in children(kin.pop()):
        if child not in line:
          line.add(child)
          kin.append(child)
    of_line = [child for child in born if child in line]
    if of_line or any(who in self.people for who in line):
      return of_line[0] if of_line else ''
    descent = records.get(name, {}).get('descent', '')
    of_people = [
        child for child in born if self.people[child].descent == descent
    ]
    if of_people or any(p.descent == descent for p in self.people.values()):
      return of_people[0] if of_people else ''
    return born[0] if born else ''

  def _may_have_child(self, person: Person) -> bool:
    age = person.age(self.year)
    if not FERTILE[0] <= age <= FERTILE[1] or self.asked_to_pass_on(
        person.name
    ):
      return False
    if self.mode == SEXUAL and (person.sex != 'f' or not person.partner):
      return False
    return person.food >= float(self.settings.get('child_cost', 2.0)) + 1.0

  def status(self, name: str) -> str:
    """What a person knows of its own body and place this year.

    Args:
      name: A living person, or the Experimenter.

    Returns:
      The status line the person observes.
    """
    if name == EXPERIMENTER:
      return EXPERIMENTER_STATUS_FORM.format(
          year=self.year, people=len(self.people)
      )
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
        f' {name} is of the {person.descent or "valley"}.'
        + self._suitors_line(person)
        + (
            ' The rite falls this year.'
            if JOIN_RITE in self._shrine_moves(person)
            else ''
        )
        + ''.join(
            f' Within sight is {self._describe(place)}.'
            for place in self._nearest_places(person)
        )
    )

  def _suitors_line(self, person: Person) -> str:
    """The singles a person may court, as the person sees them."""
    if person.partner or person.age(self.year) < ADULT_AGE:
      return ''
    seen = []
    for other in self._suitors(person):
      speaks = (
          'speaks as you do'
          if self.understanding(person.name, other) >= culture_lib.SAME_TONGUE
          else 'speaks strangely'
      )
      wooing = (
          'courting you, ' if self.courting.get(other) == person.name else ''
      )
      seen.append(
          f'{other} ({wooing}{speaks}, of the {self.people[other].descent})'
      )
    if not seen:
      return ''
    return f' Unpartnered near you: {", ".join(seen)}.'

  def _nearest_places(self, person: Person) -> list[places_lib.Place]:
    """The nearest library, school and shrine within sight, if any."""
    found = []
    for kind in (places_lib.LIBRARY, places_lib.SCHOOL, places_lib.SHRINE):
      near = self.places.near(person.tile, self.sight, kind)
      if near:
        found.append(near[0])
    return found

  def _describe(self, place: places_lib.Place) -> str:
    return places_lib.describe(place, self.places.ruined(place, self.year))

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
    stopped = _is_move(self.actions.get(EXPERIMENTER, '').strip().lower(), STOP)
    self._offers.clear()
    self._rediscovered = 0
    self._released = 0
    self._sown = 0
    self._releases = []
    self._disasters = []
    # The moves each was offered, as the year began.
    allowed = {
        name: [PASS_ON] if self.asked_to_pass_on(name) else self.moves(name)
        for name in self.actions
    }
    births = self._births()
    self._pair_partners(allowed)
    order = self._rng('order').permutation(sorted(self.actions))
    did: dict[str, tuple[str, list[str]]] = {}
    for name in (str(n) for n in order):
      if name in self.people:
        did[name] = self._resolve(name, self.actions[name], allowed[name])
    self._release(self.actions.get(EXPERIMENTER, ''))
    self._ask(self.actions.get(EXPERIMENTER, ''))
    brought = self._bring(self.actions.get(EXPERIMENTER, ''))
    self._institution(self.actions.get(EXPERIMENTER, ''))
    self._forage(did)
    self._apprentice(did)
    self._schools(did)
    self._rites(did)
    self._transmit(did)
    worked = [
        self._level(e, culture_lib.FOOD)
        for m, e in did.values()
        if m in (GATHER, SPARE)
    ]
    # Disasters strike after the year's work, before everyone eats.
    deaths = self._send(self.actions.get(EXPERIMENTER, ''))
    deaths += self._disaster()
    deaths += self._eat_and_die()
    self._see_breaches(did, allowed)
    # Untended fields go wild whether or not anyone may sow now.
    self.land.fallow(
        self.year, int((self.farming or {}).get('fallow', land_lib.FALLOW))
    )
    self.land.regrow()
    self.places.decay(self._rng('decay'), self.year)
    arrived = brought + self._arrive()
    extinct = self.culture.prune(
        {c for person in self.people.values() for c in person.given}
        | self.places.texts()
    )
    self.history.append(
        self._census(
            births=len(births),
            arrivals=len(arrived),
            deaths=len(deaths),
            did=did,
            worked=worked,
            created=self.culture.created() - created,
            extinct=len(extinct),
        )
    )
    self._reports = []
    kept = int(self.settings.get('history_kept', 200))
    del self.history[: max(0, len(self.history) - kept)]
    self.actions = {}
    self.year += 1
    for name in self.participants():
      self._tell(name, self.status(name))
    return YearOutcome(
        births=births + arrived,
        deaths=deaths,
        extinct=[custom.to_dict() for custom in extinct],
        stopped=stopped,
    )

  def _rng(self, *keys: Any) -> np.random.Generator:
    return np.random.default_rng(seed_of(self.seed, self.year, *keys))

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
    move = first_move(answer, allowed)
    enacted = [
        custom_id
        for custom_id in dict.fromkeys(
            f'c{m}' for m in CUSTOM_ID.findall(answer)
        )
        if custom_id in person.given and custom_id in self.culture
    ]
    if move in (GATHER, SPARE, APPRENTICE):
      effort = 2.0 if person.age(self.year) >= ADULT_AGE else 1.0
      if move in (GATHER, SPARE):
        effort *= self._skill(enacted, culture_lib.FOOD)
      spare = self._fields_of_others(person) if move == SPARE else []
      person.food += self.land.gather(person.tile, effort, spare)
    elif move == SOW:
      # Know-how the GM gave, enacted, makes a field; else nothing grows.
      level = self._level(enacted, culture_lib.FARMING)
      if level:
        self.land.sow(person.tile, level, self.year, tender=name)
        self._sown += 1
      else:
        self._tell(name, 'Nothing took root where you sowed.')
    elif move == MOVE:
      person.tile = self.land.best_near(
          person.tile, int(self.settings.get('roam', 1))
      )
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
              culture_lib.SANCTION,
              f'{name} shames {other} for giving nothing to {needy}',
          )
    elif move in (FOUND_LIBRARY, WRITE, READ):
      self._use_library(person, move, enacted)
    elif move in (FOUND_SCHOOL, TEACH_AT_SCHOOL):
      self._teach_at_school(person, move, enacted)
    elif move in (BUILD_SHRINE, LEAD_RITE):
      self._lead_rite(person, move, enacted)
    elif move in (TELL, PREACH):
      self._speak(person, move, enacted)
    elif move == TEACH:
      pupils = self._pupils(person)
      for pupil in pupils:
        for custom_id in enacted:
          self._offer(
              pupil, name, custom_id, culture_lib.UPBRINGING, f'{name} teaches'
          )
      if pupils and enacted:
        self._tell(name, f'{name} teaches {", ".join(pupils)}.')
    # REST, PLAY, WATCH and anything not understood change nothing.
    return move, enacted

  def _use_library(
      self, person: Person, move: str, enacted: Sequence[str]
  ) -> None:
    """Founds a library, writes at one or reads at one, if literate.

    Reading and writing need the skill of reading, enacted (named in the
    answer and once given by the GM). A founder or writer puts exact copies
    of the other customs the answer names on the shelf; a reader is offered
    some of the shelf's texts (the `reading` channel).

    Args:
      person: Who acts.
      move: `FOUND_LIBRARY`, `WRITE` or `READ`.
      enacted: The customs the answer names.
    """
    library = self.library or {}
    customs = [self.culture.get(c) for c in enacted]
    if not places_lib.literate([(c.kind, c.domain) for c in customs]):
      self._tell(person.name, f'{person.name} cannot make out the marks.')
      return
    texts = [
        c.id for c in customs if not places_lib.literate([(c.kind, c.domain)])
    ]
    near = self.places.near(person.tile, self.sight)
    if move == READ:
      shelves = [p for p in near if p.shelf]
      if not shelves:
        return
      read = self.places.read(
          shelves[0],
          self._rng('read', person.name),
          int(self.channels[culture_lib.READING].get('texts', 3)),
          self.year,
      )
      for custom_id in read:
        self._offer(
            person.name,
            '',
            custom_id,
            culture_lib.READING,
            f'{person.name} reads',
        )
      return
    if move == FOUND_LIBRARY:
      cost = float(library.get('cost', 3.0))
      spacing = int(library.get('spacing', 5))
      if person.food < cost or self.places.near(person.tile, spacing):
        return
      person.food -= cost
      place = self.places.found(
          places_lib.LIBRARY, person.tile, person.name, self.year
      )
      for name in (person.name, *self._in_sight(person)):
        self._tell(
            name,
            f'{person.name} founds a library at'
            f' {place.tile[0]},{place.tile[1]}.',
        )
    else:
      standing = [p for p in near if not self.places.ruined(p, self.year)]
      if not standing:
        return
      place = standing[0]
    for custom_id in texts:
      self.places.write(place, custom_id, self.year)
    if texts:
      self._tell(
          person.name,
          f'{person.name} writes {len(texts)} text'
          f'{"" if len(texts) == 1 else "s"} at the library.',
      )

  def _teach_at_school(
      self, person: Person, move: str, enacted: Sequence[str]
  ) -> None:
    """Founds a school or teaches at one: the customs named are taught."""
    if move == FOUND_SCHOOL:
      school = self.school or {}
      cost = float(school.get('cost', 3.0))
      spacing = int(school.get('spacing', 5))
      if person.food < cost or self.places.near(
          person.tile, spacing, places_lib.SCHOOL
      ):
        return
      person.food -= cost
      place = self.places.found(
          places_lib.SCHOOL, person.tile, person.name, self.year
      )
      for name in (person.name, *self._in_sight(person)):
        self._tell(
            name,
            f'{person.name} founds a school at'
            f' {place.tile[0]},{place.tile[1]}.',
        )
    else:
      standing = self._standing(person, places_lib.SCHOOL)
      if not standing:
        return
      place = standing[0]
    if enacted:
      self.places.keep(place, person.name, enacted, self.year)

  def _ask(self, answer: str) -> None:
    """Puts the Experimenter's question to a person, for next year."""
    match = _ASK.match(' '.join(answer.split()))
    if match is None:
      if _is_move(answer.strip().lower(), 'ask'):
        self._report('The question is not understood: ' + answer)
      return
    name, question = match.group('name'), match.group('question')
    person = self.people.get(name)
    if person is None or person.age(self.year + 1) < CHILD_ACTS_AGE:
      self._report(f'Nobody of the name {name} can answer.')
      return
    self.interviews[name] = question
    self._report(f'{name} will be asked next year.')

  def _forage(self, did: Mapping[str, tuple[str, list[str]]]) -> None:
    """Everyone old enough forages a little, whatever else they did.

    Whatever a person does with its year, it finds some food on the way
    (`subsistence`: a grown person's, half of it for a child); gathering
    is the year's work on top of it.

    Args:
      did: This year's moves.
    """
    subsistence = float(self.settings.get('subsistence', 0.0))
    if subsistence <= 0:
      return
    for name in sorted(did):
      person = self.people[name]
      if did[name][0] in (GATHER, SPARE, APPRENTICE):
        continue
      share = 1.0 if person.age(self.year) >= ADULT_AGE else 0.5
      person.food += self.land.gather(person.tile, subsistence * share)

  def _release(self, answer: str) -> None:
    """Puts a custom the Experimenter designed into the world.

    The Experimenter never edits a mind: a released custom is a new variant
    in the world's record, either written as a text on a shelf at a place
    (the nearest library within sight of it, or a new one there) or sung as
    a song that all within sight of the place hear (the `release` channel).

    Args:
      answer: The Experimenter's answer this year (`RELEASE_FORM`).
    """
    match = _RELEASE.match(' '.join(answer.split()))
    if match is None:
      if _is_move(answer.strip().lower(), RELEASE):
        self._report('The release is not understood: ' + answer)
      return
    kind, domain = match.group('kind'), match.group('domain')
    tile = self.land.wrap(int(match.group('row')), int(match.group('col')))
    if kind not in culture_lib.KINDS or domain not in culture_lib.DOMAINS:
      self._report(f'No custom is of kind {kind}, domain {domain}.')
      return
    if domain not in culture_lib.domains_of(kind):
      self._report(
          f'A {kind} is of {" or ".join(culture_lib.domains_of(kind))},'
          f' not {domain}.'
      )
      return
    within = match.group('within')
    if within and (
        kind != culture_lib.MARRIAGE
        or within not in (culture_lib.TONGUE, culture_lib.DESCENT)
    ):
      self._report(
          f'Only a marriage rule allows within {culture_lib.TONGUE} or'
          f' {culture_lib.DESCENT}: {answer}'
      )
      return
    if '|params=' in answer or '[custom' in answer:
      # The text would read as the protocol of an offer line.
      self._report('The release is not understood: ' + answer)
      return
    if (
        match.group('how') == 'song'
        and culture_lib.RELEASE not in self.channels
    ):
      self._report('No song carries in this world.')
      return
    params = {}
    if kind == culture_lib.SKILL:
      level = int(match.group('level') or 1)
      params['level'] = min(culture_lib.MAX_LEVEL, max(1, level))
    if within:
      params['within'] = within
    custom = self.culture.add(
        kind,
        domain,
        match.group('text'),
        params,
        origin=culture_lib.RELEASE,
        year=self.year,
    )
    self._released += 1
    if match.group('how') == 'text':
      near = [
          p
          for p in self.places.near(tile, self.sight)
          if not self.places.ruined(p, self.year)
      ]
      place = (
          near[0]
          if near
          else self.places.found(
              places_lib.LIBRARY, tile, EXPERIMENTER, self.year
          )
      )
      self.places.write(place, custom.id, self.year)
      self._releases.append(
          {'id': custom.id, 'how': 'text', 'tile': list(place.tile), 'heard': 0}
      )
      self._report(
          f'Your release "{custom.text}" is written at the library at'
          f' {place.tile[0]},{place.tile[1]}.',
      )
      return
    listeners = [
        name for name in self.living() if self._hears(self.people[name], tile)
    ]
    for name in listeners:
      self._offer(name, '', custom.id, culture_lib.RELEASE, 'A song is heard')
    self._releases.append({
        'id': custom.id,
        'how': 'song',
        'tile': list(tile),
        'heard': len(listeners),
    })
    self._report(
        f'Your release "{custom.text}" is sung at {tile[0]},{tile[1]};'
        f' {len(listeners)} {"hears" if len(listeners) == 1 else "hear"} it.',
    )

  def _dialect(self, specs: Sequence[Mapping[str, Any]]) -> str:
    """The id of the first dialect among custom specs ('' if none)."""
    return next(
        (str(s['id']) for s in specs if s['kind'] == culture_lib.DIALECT), ''
    )

  def understanding(self, a: str, b: str) -> float:
    """How well two living people understand each other's speech.

    Args:
      a: A living person.
      b: Another.

    Returns:
      The share of their dialects' words that agree (1.0 if either has none).
    """
    speech_a, speech_b = self.people[a].speech, self.people[b].speech
    if speech_a not in self.culture or speech_b not in self.culture:
      return 1.0
    return culture_lib.intelligibility(
        self.culture.get(speech_a), self.culture.get(speech_b)
    )

  def _speak(self, person: Person, move: str, enacted: Sequence[str]) -> None:
    """An elder's story, or a preacher's sermon, to all within sight.

    A story carries every custom the answer names, retold with the `story`
    channel's errors; a sermon carries the first only.

    Args:
      person: Who speaks.
      move: `TELL` or `PREACH`.
      enacted: The customs the answer names.
    """
    if move == TELL:
      via, told, scene = culture_lib.STORY, enacted, 'tells a story'
    else:
      via, told, scene = culture_lib.PREACHING, enacted[:1], 'preaches'
    for listener in self._in_sight(person):
      if self.people[listener].age(self.year) >= CHILD_ACTS_AGE:
        for custom_id in told:
          self._offer(
              listener, person.name, custom_id, via, f'{person.name} {scene}'
          )

  def _lead_rite(
      self, person: Person, move: str, enacted: Sequence[str]
  ) -> None:
    """Builds a shrine, or leads its rite: the customs named are its rite.

    The first to lead a rite in a year sets what it is led by; anyone else
    who leads it that year takes part.

    Args:
      person: Who acts.
      move: `BUILD_SHRINE` or `LEAD_RITE`.
      enacted: The customs the answer names.
    """
    if move == BUILD_SHRINE:
      shrine = self.shrine or {}
      cost = float(shrine.get('cost', 3.0))
      if person.food < cost or self.places.near(
          person.tile, int(shrine.get('spacing', 5)), places_lib.SHRINE
      ):
        return
      person.food -= cost
      place = self.places.found(
          places_lib.SHRINE, person.tile, person.name, self.year
      )
      self.places.keep(place, person.name, enacted, self.year)
      for name in (person.name, *self._in_sight(person)):
        self._tell(
            name,
            f'{person.name} builds a shrine at'
            f' {place.tile[0]},{place.tile[1]}.',
        )
      return
    near = [
        p
        for p in self._standing(person, places_lib.SHRINE)
        if self.rite_falls(p)
    ]
    if near and enacted and near[0].kept != self.year:
      self.places.keep(near[0], person.name, enacted, self.year)

  def _rites(self, did: Mapping[str, tuple[str, list[str]]]) -> None:
    """Those who join a rite led this year are offered what it is led by."""
    for name in sorted(
        n for n, (m, _) in did.items() if m in (JOIN_RITE, LEAD_RITE)
    ):
      person = self.people[name]
      led = [
          p
          for p in self._standing(person, places_lib.SHRINE)
          if p.kept == self.year and self.rite_falls(p)
      ]
      if not led:
        self._tell(name, 'Nobody leads the rite this year.')
        continue
      shrine = led[0]
      for custom_id in shrine.shelf:
        if custom_id in self.culture and shrine.keeper != name:
          self._offer(
              name,
              shrine.keeper,
              custom_id,
              culture_lib.RITE,
              f'{shrine.keeper} leads',
          )

  def _schools(self, did: Mapping[str, tuple[str, list[str]]]) -> None:
    """Pupils are taught the curriculum of a school in a teacher's term."""
    term = int((self.school or {}).get('term', 5))
    for pupil in sorted(n for n, (m, _) in did.items() if m == GO_TO_SCHOOL):
      person = self.people[pupil]
      taught = [
          p
          for p in self._standing(person, places_lib.SCHOOL)
          if p.keeper in self.people and self.year - p.kept < term
      ]
      if not taught:
        self._tell(pupil, 'Nobody teaches at the school this year.')
        continue
      school, teacher = taught[0], taught[0].keeper
      self.places.use(school, self.year)
      for custom_id in school.shelf:
        if custom_id in self.culture:
          self._offer(
              pupil,
              teacher,
              custom_id,
              culture_lib.SCHOOL,
              f'{teacher} teaches',
          )

  def _fields_of_others(self, person: Person) -> list[land_lib.Tile]:
    """The fields around a person that others tend (none without farming)."""
    if not self.farming:
      return []
    # A field is another's while its tender lives; the dead's are nobody's.
    return [
        tile
        for tile in self.land.fields_of_others(person.tile, person.name)
        if self.land.fields[tile][2] in self.people
    ]

  def _knows(self, person: Person, domain: str) -> bool:
    """Whether the GM ever gave a person a skill of a domain (know-how)."""
    return any(
        custom_id in self.culture
        and self.culture.get(custom_id).kind == culture_lib.SKILL
        and self.culture.get(custom_id).domain == domain
        for custom_id in person.given
    )

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
            if self.culture.get(c).kind == culture_lib.SKILL
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
    channel = self.channels.get(via)
    if channel is None:
      return  # A world without the channel (its twin may lack one).
    mutation = float(channel.get('mutation', 0.0))
    if source in self.people:
      # Talk between dialects garbles what is passed on.
      mutation += (1.0 - self.understanding(receiver, source)) * float(
          self.settings.get('garble', 0.5)
      )
    if channel.get('form'):
      custom_id = self.culture.in_form(
          custom_id, str(channel['form']), self.year
      ).id
    custom = self.culture.copy(
        custom_id,
        self._rng('offer', receiver, source, custom_id, via),
        min(1.0, mutation),
        self.year,
        improve=float(channel.get('improve', 0.0)),
    )
    person = self.people[receiver]
    if via == culture_lib.READING and not any(
        custom.id in p.given for p in self.people.values()
    ):
      self._rediscovered += 1  # A text nobody living had been given.
    if custom.id not in person.given:
      person.given.append(custom.id)
    self._offers[via] += 1
    line = culture_lib.offer_line(
        custom.spec(),
        via,
        source,
        self.people[source].food if source in self.people else None,
    )
    self._tell(receiver, f'{scene}: {line}')

  def _apprentice(self, did: Mapping[str, tuple[str, list[str]]]) -> None:
    """Pairs each youth learning a craft with a skilled worker nearby."""
    channel = self.channels.get(culture_lib.APPRENTICESHIP)
    if channel is None:
      return
    masters = {
        name: [
            c for c in enacted if self.culture.get(c).kind == culture_lib.SKILL
        ]
        for name, (move, enacted) in sorted(did.items())
        if move in (GATHER, SPARE)
        and self.people[name].age(self.year) >= ADULT_AGE
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
            youth,
            master,
            custom_id,
            culture_lib.APPRENTICESHIP,
            f'{master} shows',
        )

  def _transmit(self, did: Mapping[str, tuple[str, list[str]]]) -> None:
    """What people did is seen by their children and by those near them."""
    seen = self.channels.get(culture_lib.OBSERVATION)
    radius = int(seen.get('radius', 2)) if seen else -1
    witnessed = int(seen.get('witnessed', 3)) if seen else 0
    raised = culture_lib.UPBRINGING in self.channels
    actors = [
        name
        for name, (move, enacted) in sorted(did.items())
        if enacted and _move_kind(move) in self.seen_deeds
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
      for actor, via in [(p, culture_lib.UPBRINGING) for p in parents] + [
          (a, culture_lib.OBSERVATION) for _, a in others
      ]:
        move, enacted = did[actor]
        for custom_id in enacted:
          self._offer(
              name, actor, custom_id, via, f'{name} sees {actor} {move}'
          )
    if culture_lib.PRACTICE not in self.channels:
      return
    for actor in actors:
      for custom_id in did[actor][1]:
        custom = self.culture.get(custom_id)
        if culture_lib.form_of(custom.spec()) == culture_lib.KNOWN:
          self._offer(
              actor, actor, custom_id, culture_lib.PRACTICE, f'{actor} keeps'
          )

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

  def _pair_partners(self, allowed: Mapping[str, Sequence[str]]) -> None:
    """Pairs those looking for a partner, nearest first, in name order.

    Whoever looked for a partner this year (when that was a move offered) or
    before, and has none, is looking until paired: for anyone (`COURT`), or
    for one person (`court <name>`). Two pair when each would have the
    other: both look for anyone, or one courts the other and the other looks
    for anyone or courts back. Parents and children, and siblings, never
    pair.

    Args:
      allowed: The moves each was offered this year.
    """
    for name, answer in sorted(self.actions.items()):
      move = first_move(answer, allowed.get(name, ()))
      if move == COURT:
        self.courting[name] = ''
      elif move.startswith(f'{COURT_ONE} '):
        self.courting[name] = move.removeprefix(f'{COURT_ONE} ')
    self.courting = {
        name: whom
        for name, whom in sorted(self.courting.items())
        if name in self.people and not self.people[name].partner
    }
    taken: set[str] = set()
    for name in list(self.courting):
      if name in taken:
        continue
      person = self.people[name]
      matches = [
          other
          for other in self.courting
          if other not in taken
          and other != name
          and self.people[other].sex not in ('', person.sex)
          and not self._kin(person, self.people[other])
          and self.courting[name] in ('', other)
          and self.courting[other] in ('', name)
      ]
      if not matches:
        continue
      other = min(matches, key=lambda o: (self._distance(person, o), o))
      taken.update((name, other))
      person.partner, self.people[other].partner = other, name
      self._tell(name, f'{name} and {other} are now partners.')
      self._tell(other, f'{other} and {name} are now partners.')
    self.courting = {
        name: whom for name, whom in self.courting.items() if name not in taken
    }

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
      # A child speaks as its mother does, or (`speech_from: community`) as
      # one of the grown around her does, with the drift of each generation.
      spoken = (
          self._spoken_around(mother, rng)
          if self.settings.get('speech_from', MOTHER) == COMMUNITY
          else mother.speech
      )
      speech = ''
      if spoken in self.culture:
        speech = self.culture.copy(
            spoken,
            rng,
            float(self.settings.get('dialect_drift', 0.02)),
            self.year,
        ).id
        ids = [
            i for i in ids if self.culture.get(i).kind != culture_lib.DIALECT
        ] + [speech]
      child = Person(
          name=self._new_name(),
          born=self.year,
          sex=''
          if self.mode == ASEXUAL
          else ('f' if rng.random() < 0.5 else 'm'),
          tile=mother.tile,
          food=min(cost, mother.food),
          parents=list(parents),
          given=list(dict.fromkeys(ids)),
          speech=speech,
          descent=mother.descent,
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

  def _spoken_around(self, person: Person, rng: np.random.Generator) -> str:
    """The speech a child picks up around a person: a grown speaker's, drawn.

    Each of the grown within sight of the person (the person included) is
    as likely to be the one the child learns from, so a variant spreads in
    proportion to how many speak it (neutral drift, not conformity).

    Args:
      person: The one the speakers are around (the mother).
      rng: The birth's draw.

    Returns:
      The speech drawn (the person's own if nobody is in sight).
    """
    near = sorted(
        p.speech
        for p in self.people.values()
        if p.speech
        and p.age(self.year) >= ADULT_AGE
        and self.places.distance(p.tile, person.tile) <= self.sight
    )
    if not near:
      return person.speech
    return near[int(rng.integers(len(near)))]

  def _passed_on(self, parent: str) -> list[str]:
    """The ids a parent named when asked what it passes on.

    Asked, a parent's answer names what it passes on, whatever words come
    with the ids. Only customs the GM once gave the parent pass on: the world
    passes on what it delivered, never what a mind might claim.

    Args:
      parent: A living parent.

    Returns:
      The ids passed on.
    """
    answer = self.actions.get(parent, '')
    given = self.people[parent].given
    return [
        f'c{m}'
        for m in dict.fromkeys(CUSTOM_ID.findall(answer.lower()))
        if f'c{m}' in self.culture and f'c{m}' in given
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
    self.dead.append({
        'name': name,
        'sex': person.sex,
        'born': person.born,
        'died': self.year,
        'parents': list(person.parents),
        'children': list(person.children),
        'descent': person.descent,
    })
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
    self.asked.pop(name, None)
    if self.interviews.pop(name, None) is not None:
      self._report(f'{name} died before answering.')

  def _census(
      self,
      births: int,
      arrivals: int,
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
        'arrivals': arrivals,
        'deaths': deaths,
        'mean_age': round(sum(ages) / len(ages), 2) if ages else 0.0,
        'food': round(float(self.land.food.sum()), 3),
        'fields': len(self.land.fields),
        'sown': self._sown,
        'moves': dict(
            sorted(
                collections.Counter(
                    _move_kind(m) for m, _ in did.values()
                ).items()
            )
        ),
        'breaches': len(self.breaches),
        'libraries': sum(
            1
            for p in self.places.all()
            if p.kind == places_lib.LIBRARY
            and not self.places.ruined(p, self.year)
        ),
        'ruins': sum(
            1 for p in self.places.all() if self.places.ruined(p, self.year)
        ),
        'texts': sum(
            len(p.shelf)
            for p in self.places.all()
            if p.kind == places_lib.LIBRARY
        ),
        'schools': sum(
            1
            for p in self.places.all()
            if p.kind == places_lib.SCHOOL
            and not self.places.ruined(p, self.year)
        ),
        'shrines': sum(
            1
            for p in self.places.all()
            if p.kind == places_lib.SHRINE
            and not self.places.ruined(p, self.year)
        ),
        'offers': dict(sorted(self._offers.items())),
        'rediscovered': self._rediscovered,
        'tongues': len({p.speech for p in self.people.values() if p.speech}),
        'understanding': self._mean_understanding(),
        'released': self._released,
        'releases': [dict(r) for r in self._releases],
        'disasters': [dict(d) for d in self._disasters],
        'reports': list(self._reports),
        'variants': len(self.culture),
        'new_variants': created,
        'lost_variants': extinct,
        'skill_worked': round(sum(worked) / len(worked), 2) if worked else 0.0,
    }

  def _mean_understanding(self) -> float:
    """The mean understanding between two of the living (1.0: one tongue)."""
    names = self.living()
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    if not pairs:
      return 1.0
    return round(
        sum(self.understanding(a, b) for a, b in pairs) / len(pairs), 3
    )

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

  def _report(self, text: str) -> None:
    """Tells the Experimenter how a move of its went (kept for the census)."""
    self._tell(EXPERIMENTER, text)
    self._reports.append(text)

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
        'dead': [dict(d) for d in self.dead],
        'named': self.named,
        'actions': dict(self.actions),
        'expecting': {k: list(v) for k, v in self.expecting.items()},
        'queued': {k: list(v) for k, v in self.queued.items()},
        'history': list(self.history),
        'breaches': {k: list(v) for k, v in self.breaches.items()},
        'courting': dict(self.courting),
        'interviews': dict(self.interviews),
        'asked': {k: list(v) for k, v in self.asked.items()},
        # What the World has told the Experimenter so far this year.
        'reports': list(self._reports),
        'transcripts': [dict(t) for t in self.transcripts],
        'places': self.places.get_state(),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.year = int(state['year'])
    self.land.set_state(state['land'])
    self.culture.set_state(state['culture'])
    self.people = {
        str(p['name']): Person.from_dict(p) for p in state.get('people', ())
    }
    self.dead = [
        dict(d) if isinstance(d, Mapping) else {'name': str(d)}
        for d in state.get('dead', ())
    ]
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
    self.courting = {
        str(k): str(v) for k, v in state.get('courting', {}).items()
    }
    self.places.set_state(state.get('places', {}))
    self.interviews = {
        str(k): str(v) for k, v in state.get('interviews', {}).items()
    }
    self.transcripts = [dict(t) for t in state.get('transcripts', ())]
    self.asked = {
        str(k): [str(v[0]), int(v[1])]
        for k, v in state.get('asked', {}).items()
    }
    self._reports = [str(r) for r in state.get('reports', ())]


def _move_kind(move: str) -> str:
  """A move without the one it is made towards (for counting)."""
  for kind in (SHARE, SHAME, COURT_ONE):
    if move.startswith(f'{kind} '):
      return kind
  return move


def _is_move(answer: str, move: str) -> bool:
  """Whether an answer makes a move (the move, then nothing or a space)."""
  return answer == move or answer.startswith(move + ' ')


def first_move(answer: str, allowed: Sequence[str]) -> str:
  """The move an answer makes: the offered move it names first.

  A mind may say more than the move ("This year I gather food, by c3."); the
  world reads the first move offered that the answer names as whole words
  (the longest, where two start at the same place).

  Args:
    answer: The answer.
    allowed: The moves offered.

  Returns:
    The move ('' if the answer names none).
  """
  answer = ' '.join(answer.lower().split())
  found = []
  for move in allowed:
    match = re.search(
        r'(?<![a-z])' + re.escape(move.lower()) + r'(?![a-z])', answer
    )
    if match:
      found.append((match.start(), -len(move), move))
  return min(found)[2] if found else ''
