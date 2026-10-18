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

"""Peoples on the map: settlements, the land they work, roads and routes.

Each people has a capital on the best tile of its home region and a town
for every few thousand of its number, each on the best free tile of its
lands (good farmland, near the sea, off the heights) a few tiles from the
others. Towns are founded as a people grows and abandoned as it starves,
and both are news. A settlement works the tiles around it (its catchment):
how good that farmland and how thick that timber is now, against when the
settlement was first laid out, scales the people's harvests, so a fire in a
town's woods, ice on its fields or drought that turns them to scrub costs
it (each settlement is measured against its own land when it was laid
out, so spreading onto poorer land costs nothing). Roads join a people's
settlements
(each new one to the nearest already joined), and trade routes join the
capitals of peoples that trade, over land and across the sea, along the
cheapest tiles.
"""

# pylint: disable=g-complex-comprehension,g-doc-return-or-yield,missing-function-docstring

from collections.abc import Mapping, Sequence
import dataclasses
import heapq
from typing import Any

from examples.deep_time.world import biomes as biomes_lib
from examples.deep_time.world import news as news_lib
from examples.deep_time.world import tiles as tiles_lib
import numpy as np


@dataclasses.dataclass
class Settlement:
  id: int
  polity: str
  row: int  # Tile row, south first.
  col: int
  capital: bool = False


class Settlements:
  """Every people's settlements, catchments and routes."""

  def __init__(
      self,
      tile_map: tiles_lib.TileMap,
      biomes: biomes_lib.Biomes,
      lands: Mapping[str, Sequence[int]],
      homes: Mapping[str, int],
      names: Mapping[str, str],
      config: Mapping[str, Any],
      trade: bool = False,
  ):
    """Lays out the capitals.

    Args:
      tile_map: The tiles.
      biomes: Their biomes.
      lands: Each people's regions.
      homes: Each people's home region.
      names: Each people's name.
      config: `data/settlements.yaml`.
      trade: Whether the peoples trade (trade routes join their capitals).
    """
    self.tiles = tile_map
    self.biomes = biomes
    self.config = config
    self.trade = trade
    # A whisper of seeded noise breaks ties between equally good tiles
    # (otherwise the first column of the map wins them all).
    self._jitter = 0.05 * np.random.default_rng([tile_map.seed, 8081]).random(
        tile_map.land.shape
    )
    self.lands = {k: list(v) for k, v in lands.items()}
    self.names = dict(names)
    self.places: dict[str, list[Settlement]] = {k: [] for k in lands}
    self.next_id = 1
    # Each settlement's worked land when it was laid out, by id.
    self.start: dict[int, dict[str, float]] = {}
    self.routes: list[tuple[str, list[tuple[int, int]]]] = []
    region = tile_map.region
    for key, home in homes.items():
      score = np.where(region == home, self.siting(), -np.inf)
      if not np.isfinite(score).any():
        score = self._in_lands(key, self.siting())
      self._found(key, score, capital=True)
    self._route()

  def siting(self) -> np.ndarray:
    """How settlers rate each tile: farmland, sea nearby, low ground."""
    c = self.config
    land = self.tiles.land & ~self.tiles.ice
    coast = self.tiles.terrain == tiles_lib.COAST
    high = self.tiles.terrain == tiles_lib.HIGHLAND
    score = (
        self.farmland()
        + float(c['coast_bonus']) * coast
        - float(c['highland_penalty']) * high
        + self._jitter
    )
    return np.where(land, score, -np.inf)

  def farmland(self) -> np.ndarray:
    """How good each tile is for fields now (0-1)."""
    table = self.config['farmland']
    good = np.array(
        [float(table.get(b['key'], 0.0)) for b in self.biomes.specs]
    )
    return good[self.biomes.biome] * (self.tiles.land & ~self.tiles.ice)

  def _in_lands(self, key: str, score: np.ndarray) -> np.ndarray:
    mine = np.isin(self.tiles.region, self.lands[key])
    return np.where(mine, score, -np.inf)

  def _found(
      self, key: str, score: np.ndarray, capital: bool = False
  ) -> Settlement | None:
    spacing = int(self.config['spacing'])
    score = score.copy()
    for others in self.places.values():
      for s in others:
        for r, c in self._around(s.row, s.col, spacing - 1):
          score[r, c] = -np.inf
    if not np.isfinite(score).any():
      return None
    cell = int(np.argmax(np.where(np.isfinite(score), score, -1e9)))
    row, col = divmod(cell, self.tiles.cols)
    place = Settlement(self.next_id, key, row, col, capital)
    self.next_id += 1
    self.places[key].append(place)
    return place

  def _around(self, row: int, col: int, reach: int) -> list[tuple[int, int]]:
    rows, cols = self.tiles.rows, self.tiles.cols
    return [
        (r, (col + dc) % cols)
        for r in range(max(0, row - reach), min(rows, row + reach + 1))
        for dc in range(-reach, reach + 1)
    ]

  def where(self, place: Settlement) -> str:
    """A few words for where a settlement stands."""
    biome = self.biomes.specs[int(self.biomes.biome[place.row, place.col])]
    lat = float(self.biomes.lat[place.row])
    side = 'north' if lat > 20 else 'south' if lat < -20 else 'tropics'
    if self.tiles.terrain[place.row, place.col] == tiles_lib.COAST:
      return f'on the coast in the {side}'
    return f'in the {biome["name"]} of the {side}'

  def update(self, populations: Mapping[str, float]) -> list[str]:
    """Founds and abandons towns as each people grows and shrinks."""
    news = []
    per = float(self.config['town_population'])
    top = int(self.config['max_settlements'])
    changed = False
    for key, people in populations.items():
      if key not in self.places:
        continue
      want = int(np.clip(1 + people // per, 1, top))
      places = self.places[key]
      while len(places) > want:
        lost = places.pop()
        news.append(
            news_lib.located(
                f'{self._name(key)} abandon their town {self.where(lost)}.',
                (lost.row, lost.col),
            )
        )
        changed = True
      while len(places) < want:
        score = self._in_lands(key, self.siting())
        place = self._found(key, score)
        if place is None:
          break
        news.append(
            news_lib.located(
                f'{self._name(key)} found a town {self.where(place)}.',
                (place.row, place.col),
            )
        )
        changed = True
    if changed:
      self._route()
    return news

  def tiles_of(self, key: str) -> list[tuple[int, int]]:
    """Where a people's settlements stand."""
    return [(s.row, s.col) for s in self.places.get(key, [])]

  def _name(self, key: str) -> str:
    return self.names.get(key, key)

  def catchment(self, key: str) -> np.ndarray:
    """The tiles a people's settlements work."""
    out = np.zeros(self.tiles.land.shape, dtype=bool)
    for s in self.places.get(key, []):
      out |= self._works(s)
    return out

  def _works(self, place: Settlement) -> np.ndarray:
    """The tiles one settlement works."""
    out = np.zeros(self.tiles.land.shape, dtype=bool)
    for r, c in self._around(
        place.row, place.col, int(self.config['catchment'])
    ):
      out[r, c] = True
    return out & self.tiles.land

  def quality(
      self,
      key: str,
      covers: Mapping[str, np.ndarray],
      crop_loss: np.ndarray | None = None,
      area: np.ndarray | None = None,
  ) -> dict[str, float]:
    """How good worked land is now: farmland and timber.

    Args:
      key: The people.
      covers: Plant cover per species per tile.
      crop_loss: Farmland spoiled per tile (soot).
      area: The tiles to measure (default: all the people work).
    """
    area = self.catchment(key) if area is None else area
    if not area.any():
      return {'farming': 0.0, 'forest': 0.0}
    timber = np.zeros(self.tiles.land.shape)
    for species, cover in covers.items():
      if species.rstrip('0123456789') in self.config['timber']:
        timber += cover
    farmland = self.farmland()
    if crop_loss is not None:  # Soot spoils fields under the haze.
      farmland = farmland * (1.0 - crop_loss)
    return {
        'farming': float(farmland[area].mean()),
        'forest': float(timber[area].mean()),
    }

  def local(
      self,
      covers: Mapping[str, np.ndarray],
      crop_loss: np.ndarray | None = None,
  ) -> dict[str, dict[str, float]]:
    """Each people's local yield factors.

    Each settlement's land is measured against itself when the settlement
    was laid out, and a people's factor is the mean over its settlements:
    a town on poorer land works it as well as it ever could, so spreading
    out costs nothing, while fire, ice, drought or soot on worked land do.

    Args:
      covers: Plant cover per species per tile.
      crop_loss: Farmland spoiled per tile (soot).

    Returns:
      People -> kind ('farming', 'forest') -> factor.
    """
    lo, hi = (float(x) for x in self.config['local_range'])
    out = {}
    for key, places in self.places.items():
      ratios: dict[str, list[float]] = {'farming': [], 'forest': []}
      for place in places:
        now = self.quality(key, covers, crop_loss, self._works(place))
        start = self.start.setdefault(place.id, dict(now))
        for kind, value in now.items():
          ratios[kind].append(
              value / start[kind] if start[kind] > 1e-9 else 1.0
          )
      out[key] = {
          kind: float(np.clip(np.mean(r), lo, hi)) if r else 1.0
          for kind, r in ratios.items()
      }
    return out

  def _route(self) -> None:
    """Roads within each people; trade routes between their capitals."""
    routes = []
    for places in self.places.values():
      joined = places[:1]
      for place in places[1:]:
        near = min(
            joined,
            key=lambda s, p=place: self._steps(s.row, s.col, p.row, p.col),
        )
        path = self._path((near.row, near.col), (place.row, place.col), False)
        if path:
          routes.append(('road', path))
        joined.append(place)
    if self.trade:
      capitals = [p[0] for p in self.places.values() if p]
      for i, one in enumerate(capitals):
        for two in capitals[i + 1 :]:
          path = self._path((one.row, one.col), (two.row, two.col), True)
          if path:
            routes.append(('trade', path))
    self.routes = routes

  def _steps(self, r1: int, c1: int, r2: int, c2: int) -> int:
    cols = self.tiles.cols
    dc = min((c1 - c2) % cols, (c2 - c1) % cols)
    return max(abs(r1 - r2), dc)

  def _path(
      self, start: tuple[int, int], goal: tuple[int, int], by_sea: bool
  ) -> list[tuple[int, int]]:
    """The cheapest tiles from one place to another (A*, 8 neighbours)."""
    cost = self.config['route_cost']
    terrain = self.tiles.terrain
    by_kind = {
        tiles_lib.LOWLAND: float(cost['lowland']),
        tiles_lib.COAST: float(cost['coast']),
        tiles_lib.HIGHLAND: float(cost['highland']),
        tiles_lib.SHELF: float(cost['shelf']) if by_sea else np.inf,
        tiles_lib.DEEP_OCEAN: float(cost['deep']) if by_sea else np.inf,
    }
    rows, cols = self.tiles.rows, self.tiles.cols
    best = {start: 0.0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, start)]
    while queue:
      _, here = heapq.heappop(queue)
      if here == goal:
        path = [here]
        while path[-1] in came:
          path.append(came[path[-1]])
        return path[::-1]
      r, c = here
      for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
          if not (dr or dc) or not 0 <= r + dr < rows:
            continue
          there = (r + dr, (c + dc) % cols)
          step = by_kind[int(terrain[there])]
          if not np.isfinite(step):
            continue
          g = best[here] + step * (1.414 if dr and dc else 1.0)
          if g < best.get(there, np.inf):
            best[there] = g
            came[there] = here
            heapq.heappush(queue, (g + self._steps(*there, *goal), there))
    return []

  def payload(self, colours: Mapping[str, Any]) -> dict[str, Any]:
    """For the map (rows north first)."""
    keys = list(self.places)
    north = self.tiles.rows - 1
    grey = {'light': [90, 90, 90], 'dark': [200, 200, 200]}
    return {
        'polities': [
            {
                'key': k,
                'name': self._name(k),
                **{t: list(colours.get(k, grey)[t]) for t in ('light', 'dark')},
            }
            for k in keys
        ],
        'settlements': [
            [
                i,
                s.id,
                north - s.row,
                s.col,
                int(s.capital),
                int(self._port(s)),
            ]
            for i, k in enumerate(keys)
            for s in self.places[k]
        ],
        'routes': [
            [kind, [[north - r, c] for r, c in path]]
            for kind, path in self.routes
        ],
    }

  def _port(self, place: Settlement) -> bool:
    return any(
        self.tiles.terrain[r, c] == tiles_lib.SHELF
        for r, c in self._around(place.row, place.col, 1)
    )

  def get_state(self) -> dict[str, Any]:
    return {
        'next_id': self.next_id,
        'start': {str(k): dict(v) for k, v in self.start.items()},
        'places': {
            k: [[s.id, s.row, s.col, int(s.capital)] for s in v]
            for k, v in self.places.items()
        },
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.next_id = int(state['next_id'])
    self.start = {
        int(k): {kk: float(vv) for kk, vv in v.items()}
        for k, v in state['start'].items()
    }
    self.places = {
        k: [
            Settlement(int(i), k, int(r), int(c), bool(cap))
            for i, r, c, cap in v
        ]
        for k, v in state['places'].items()
    }
    self._route()
