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

"""The places people build: libraries, schools, shrines, courts, markets.

A library holds texts: exact copies of customs written by literate people,
which anyone literate who comes may read. Its shelf is a pool of culture in
the world itself, like plasmids in the environment: a custom can wait there
long after the last person who held it has died. A library nobody uses
falls to ruin; its texts can still be read in the ruins, but crumble one by
one.

A school holds a curriculum (on its shelf): the customs its teacher teaches
there. A teacher keeps the school for a term; children nearby who go to
school during it are taught the curriculum, whoever their parents are.

A shrine holds a rite (on its shelf): the customs its last leader led it
by. The rite falls on a calendar; those who join it may take them as their
own.

A court holds a law (on its shelf): the rules it last judged by. A market
holds the know-how last sold there.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from examples.lifeways.world import culture as culture_lib
import numpy as np

LIBRARY = 'library'
SCHOOL = 'school'
SHRINE = 'shrine'
COURT = 'court'
MARKET = 'market'

Tile = tuple[int, int]


@dataclasses.dataclass
class Place:
  """A building and what it holds."""

  id: str
  kind: str
  tile: Tile
  founded: int
  founder: str
  # The last year anyone used it.
  used: int
  # The customs written (taught) there, oldest first.
  shelf: list[str] = dataclasses.field(default_factory=list)
  # Who last kept it (a school's teacher), and when.
  keeper: str = ''
  kept: int = -1
  # A library's texts' scripts: the marks each was written in, by id.
  scripts: dict[str, list[str]] = dataclasses.field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    data = dataclasses.asdict(self)
    data['tile'] = list(self.tile)
    return data

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> 'Place':
    return cls(
        id=str(data['id']),
        kind=str(data['kind']),
        tile=(int(data['tile'][0]), int(data['tile'][1])),
        founded=int(data['founded']),
        founder=str(data['founder']),
        used=int(data['used']),
        shelf=[str(c) for c in data.get('shelf', ())],
        keeper=str(data.get('keeper', '')),
        kept=int(data.get('kept', -1)),
        scripts={
            str(k): [str(m) for m in v]
            for k, v in data.get('scripts', {}).items()
        },
    )


class Places:
  """The places of a toroidal map.

  Args:
    rows: The map's rows.
    cols: The map's columns.
    shelf: The most texts a library holds; past it, the oldest crumbles.
    ruin_after: The years unused after which a library is a ruin.
    crumble: The chance a year that a text in a ruin crumbles.
  """

  def __init__(
      self,
      rows: int,
      cols: int,
      shelf: int = 12,
      ruin_after: int = 30,
      crumble: float = 0.1,
  ):
    self._rows, self._cols = int(rows), int(cols)
    self._shelf = int(shelf)
    self._ruin_after = int(ruin_after)
    self._crumble = float(crumble)
    self._places: dict[str, Place] = {}
    self._next = 1

  def distance(self, a: Tile, b: Tile) -> int:
    dr = min(abs(a[0] - b[0]), self._rows - abs(a[0] - b[0]))
    dc = min(abs(a[1] - b[1]), self._cols - abs(a[1] - b[1]))
    return dr + dc

  def ruined(self, place: Place, year: int) -> bool:
    return year - place.used > self._ruin_after

  def all(self) -> list[Place]:
    return list(self._places.values())

  def near(self, tile: Tile, within: int, kind: str = LIBRARY) -> list[Place]:
    """The places of a kind near a tile.

    Args:
      tile: Where from.
      within: The most steps away.
      kind: The kind of place.

    Returns:
      The places, nearest first (ties by id).
    """
    return sorted(
        (
            p
            for p in self._places.values()
            if p.kind == kind and self.distance(tile, p.tile) <= within
        ),
        key=lambda p: (self.distance(tile, p.tile), p.id),
    )

  def found(self, kind: str, tile: Tile, founder: str, year: int) -> Place:
    place = Place(
        id=f'p{self._next}',
        kind=kind,
        tile=tile,
        founded=year,
        founder=founder,
        used=year,
    )
    self._next += 1
    self._places[place.id] = place
    return place

  def write(
      self,
      place: Place,
      custom_id: str,
      year: int,
      script: Sequence[str] = (),
  ) -> None:
    """Puts a text on the shelf (the newest), the oldest crumbling if full.

    Args:
      place: The library.
      custom_id: The custom written.
      year: This year.
      script: The marks it is written in (the writer's script).
    """
    if custom_id in place.shelf:
      place.shelf.remove(custom_id)
    place.shelf.append(custom_id)
    del place.shelf[: max(0, len(place.shelf) - self._shelf)]
    if script:
      place.scripts[custom_id] = list(script)
    for gone in set(place.scripts) - set(place.shelf):
      del place.scripts[gone]
    place.used = year

  def keep(
      self, place: Place, keeper: str, ids: Sequence[str], year: int
  ) -> None:
    """A keeper takes the place, with what it holds (a new curriculum)."""
    place.shelf = list(dict.fromkeys(ids))[: self._shelf]
    place.keeper, place.kept, place.used = keeper, year, year

  def use(self, place: Place, year: int) -> None:
    place.used = year

  def read(
      self, place: Place, rng: np.random.Generator, texts: int, year: int
  ) -> list[str]:
    """Some texts from the shelf, drawn by `rng`; a standing place is used."""
    if not self.ruined(place, year):
      place.used = year
    count = min(texts, len(place.shelf))
    picked = sorted(rng.choice(len(place.shelf), size=count, replace=False))
    return [place.shelf[int(i)] for i in picked]

  def close(self, place: Place) -> None:
    """Closes a place: it and what it holds are gone."""
    self._places.pop(place.id, None)

  def decay(self, rng: np.random.Generator, year: int) -> None:
    """Texts in ruins crumble; a ruin with nothing left is gone."""
    for place in list(self._places.values()):
      if not self.ruined(place, year):
        continue
      place.shelf = [c for c in place.shelf if rng.random() >= self._crumble]
      if not place.shelf:
        del self._places[place.id]

  def texts(self) -> set[str]:
    """Every custom on any shelf."""
    return {c for place in self._places.values() for c in place.shelf}

  def get_state(self) -> dict[str, Any]:
    return {
        'next': self._next,
        'places': [p.to_dict() for p in self._places.values()],
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self._next = int(state.get('next', 1))
    self._places = {
        str(p['id']): Place.from_dict(p) for p in state.get('places', ())
    }


def describe(place: Place, ruined: bool) -> str:
  """How a place looks to those who see it."""
  count = len(place.shelf)
  if place.kind == SCHOOL:
    held = f'{count} custom{"" if count == 1 else "s"} taught'
  elif place.kind == SHRINE:
    held = f'a rite of {count} custom{"" if count == 1 else "s"}'
  elif place.kind == COURT:
    held = f'a law of {count} rule{"" if count == 1 else "s"}'
  elif place.kind == MARKET:
    held = f'{count} kind{"" if count == 1 else "s"} of know-how last sold'
  else:
    held = f'{count} text{"" if count == 1 else "s"}'
  if ruined:
    return f'the ruins of a {place.kind}, with {held} left'
  return f'a {place.kind}, with {held}'


def literate(kinds_and_domains: Sequence[tuple[str, str]]) -> bool:
  """Whether customs (kind, domain) include the skill of reading."""
  return any(
      kind == culture_lib.SKILL and domain == culture_lib.LEARNING
      for kind, domain in kinds_and_domains
  )
