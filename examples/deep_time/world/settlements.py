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

Where conquest is played (`civilization.conquest`), a town can change hands:
it is kept by the people that holds it, apart from the people that founded
it, with its loyalty, walls and garrison; a town's people are its holder's
population split across the holder's towns by the farmland each works, and
a change of hands moves that share.
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
  polity: str  # The people that founded it.
  row: int  # Tile row, south first.
  col: int
  capital: bool = False
  # Where conquest is played: how firmly the town holds to its people (0-1),
  # its walls, and the arms stationed there (units by kind, `data/arms.yaml`).
  loyalty: float = 1.0
  walls: int = 0
  garrison: dict[str, float] = dataclasses.field(default_factory=dict)
  # A won town warned it would rise (it does next turn, unless settled).
  warned: bool = False
  # Taken by arms and not yet settled: restless unless held (C4).
  taken: bool = False
  # Its regard for other peoples that court it with grain and envoys (C5).
  regard: dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Army:
  """A garrison on the march (where conquest is played, stage 6 C4)."""

  id: int
  polity: str  # The people it marches for.
  units: dict[str, float]  # Arms by kind.
  path: list[tuple[int, int]]  # Tiles from its town to its target.
  at: int  # How far along the path it is.
  target: int  # The town it marches on.
  intent: str  # 'take' or 'raze'.
  arrives: int  # The tick it reaches the town.
  per_turn: int  # Tiles it covers a turn.


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
      conquest: bool = False,
      arms: Mapping[str, Mapping[str, Any]] | None = None,
      war: Mapping[str, Any] | None = None,
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
      conquest: Whether towns can change hands.
      arms: The kinds of arms (`data/arms.yaml`), where conquest is played.
      war: How marches, fights and their aftermath go (`data/arms.yaml`).
    """
    self.tiles = tile_map
    self.biomes = biomes
    self.config = config
    self.trade = trade
    self.conquest = conquest
    self.arms = {k: dict(v) for k, v in (arms or {}).items()}
    self.war = dict(war or {})
    # Armies on the march, and the next army's id.
    self.armies: list[Army] = []
    self.next_army = 1
    # A whisper of seeded noise breaks ties between equally good tiles
    # (otherwise the first column of the map wins them all).
    self._jitter = 0.05 * np.random.default_rng([tile_map.seed, 8081]).random(
        tile_map.land.shape
    )
    self.lands = {k: list(v) for k, v in lands.items()}
    self.names = dict(names)
    # Each people's towns: those it holds (founded, or won where conquest is
    # played), capital first.
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

  def _free(self, score: np.ndarray) -> np.ndarray:
    """`score` with every tile too near a town ruled out."""
    spacing = int(self.config['spacing'])
    score = score.copy()
    for others in self.places.values():
      for s in others:
        for r, c in self._around(s.row, s.col, spacing - 1):
          score[r, c] = -np.inf
    return score

  def has_site(self, key: str) -> bool:
    """Whether a people's lands have a free site for a new town."""
    return bool(
        np.isfinite(self._free(self._in_lands(key, self.siting()))).any()
    )

  def _found(
      self, key: str, score: np.ndarray, capital: bool = False
  ) -> Settlement | None:
    score = self._free(score)
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

  def update(
      self, populations: Mapping[str, float], found: bool = True
  ) -> list[str]:
    """Founds and abandons towns as each people grows and shrinks.

    Args:
      populations: Each people's population.
      found: Whether growth founds towns (where conquest is played, only a
        people's move founds one).

    Returns:
      The news.
    """
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
        # The newest town it founded itself: a town it won is never
        # abandoned for want of people (nor is the capital).
        lost = next(
            (p for p in reversed(places) if p.polity == key and not p.capital),
            None,
        )
        if lost is None:
          break
        places.remove(lost)
        news.append(
            news_lib.located(
                f'{self._name(key)} abandon their town {self.where(lost)}.',
                (lost.row, lost.col),
            )
        )
        changed = True
      while found and len(places) < want:
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

  def founded(self, key: str) -> int:
    """How many of the towns a people holds it founded itself."""
    return sum(1 for p in self.places.get(key, []) if p.polity == key)

  def found_at(
      self, key: str, tile: tuple[int, int] | None
  ) -> tuple[Settlement | None, str]:
    """A people founds a town where asked, or on its best free site.

    Args:
      key: The people.
      tile: Where it asked (south first), or None for its best free site.

    Returns:
      The town, or None and why not.
    """
    score = self._in_lands(key, self.siting())
    if tile is not None:
      r, c = int(tile[0]), int(tile[1])
      if not self.tiles.land[r, c] or self.tiles.ice[r, c]:
        return None, 'there is no dry land free of ice there'
      if not np.isfinite(score[r, c]):
        return None, 'that is not their land'
      spacing = int(self.config['spacing'])
      for others in self.places.values():
        for s in others:
          if self._steps(r, c, s.row, s.col) < spacing:
            return None, 'it is too near another town'
      score = np.full(score.shape, -np.inf)
      score[r, c] = 1.0
    place = self._found(key, score)
    if place is None:
      return None, 'there is no free site left in their lands'
    self._route()
    return place, ''

  def find(self, town: int) -> tuple[Settlement, str] | None:
    """A town by its id, and the people that holds it."""
    for key, places in self.places.items():
      for place in places:
        if place.id == town:
          return place, key
    return None

  def shares(self, key: str) -> dict[int, float]:
    """Each of a people's towns' share of its people: the farmland it works."""
    farmland = self.farmland()
    land = {
        p.id: float(farmland[self._works(p)].sum()) for p in self.places[key]
    }
    total = sum(land.values())
    if total <= 1e-9:
      return {town: 1.0 / len(land) for town in land} if land else {}
    return {town: value / total for town, value in land.items()}

  def change_hands(
      self, town: int, taker: str, loyalty: float, how: str = 'take'
  ) -> str:
    """Moves a town to the people that took or won it; the news of it.

    Args:
      town: The town's id.
      taker: The people that holds it now.
      loyalty: Its loyalty to its new holder.
      how: 'take' (by arms), 'win' (won over) or 'rise' (a won town
        returning to its founder).

    Returns:
      The news.
    """
    found = self.find(town)
    if found is None:
      raise KeyError(town)
    place, holder = found
    if holder == taker:
      raise ValueError(f'{taker} already hold town {town}')
    self.places[holder].remove(place)
    self.places[taker].append(place)
    place.loyalty = float(loyalty)
    place.garrison = {}
    place.warned = False
    place.taken = how == 'take'
    place.regard = {}
    if place.capital:  # A people that lost its capital rules from its oldest.
      place.capital = False
      if self.places[holder]:
        self.places[holder][0].capital = True
    self._route()
    if how == 'rise':
      text = (
          f'A town {self.where(place)} rises against the'
          f' {self._name(holder)} and returns to the {self._name(taker)}.'
      )
    else:
      verb = 'take' if how == 'take' else 'win over'
      text = (
          f'{self._name(taker)} {verb} a town of the {self._name(holder)}'
          f' {self.where(place)}.'
      )
    return news_lib.located(text, (place.row, place.col))

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
                # A town won from another people: its founder.
                *([keys.index(s.polity)] if s.polity != k else []),
            ]
            for i, k in enumerate(keys)
            for s in self.places[k]
        ],
        'routes': [
            [kind, [[north - r, c] for r, c in path]]
            for kind, path in self.routes
        ],
        # Armies on the march: [people, row, col, strength, arms by kind,
        # target row, target col, arrival tick, 'take' or 'raze'].
        **(
            {
                'armies': [
                    [
                        keys.index(a.polity),
                        north - a.path[a.at][0],
                        a.path[a.at][1],
                        round(self.army_strength(a), 1),
                        {
                            self.arms.get(k, {}).get('label', k): round(n, 1)
                            for k, n in sorted(a.units.items())
                        },
                        north - a.path[-1][0],
                        a.path[-1][1],
                        a.arrives,
                        a.intent,
                    ]
                    for a in self.armies
                    if a.polity in keys
                ]
            }
            if self.conquest
            else {}
        ),
        # Each garrisoned town's strength and its arms by kind.
        **(
            {
                'garrisons': {
                    str(s.id): [
                        round(self.strength(s), 1),
                        {
                            self.arms.get(k, {}).get('label', k): round(n, 1)
                            for k, n in sorted(s.garrison.items())
                        },
                    ]
                    for v in self.places.values()
                    for s in v
                    if s.garrison
                },
                # Each courted town: the people it leans to most, its
                # regard for them and the regard it needs to join them.
                'leaning': {
                    str(s.id): [
                        keys.index(lean[0]),
                        round(lean[1], 2),
                        round(self.need(s), 2),
                    ]
                    for v in self.places.values()
                    for s in v
                    if (lean := self.courted_by(s)) is not None
                },
                # Each taken town not yet settled: its loyalty to its holder.
                'restless': {
                    str(s.id): round(s.loyalty, 2)
                    for v in self.places.values()
                    for s in v
                    if s.taken
                },
            }
            if self.conquest
            else {}
        ),
    }

  def strength(self, place: Settlement) -> float:
    """What a town's garrison counts for in a fight."""
    return sum(
        n * float(self.arms.get(k, {}).get('strength', 1.0))
        for k, n in place.garrison.items()
    )

  def arms_of(self, polity: str) -> dict[str, float]:
    """A people's arms, by kind, across the towns it holds."""
    found: dict[str, float] = {}
    for place in self.places.get(polity, ()):
      for kind, n in place.garrison.items():
        found[kind] = found.get(kind, 0.0) + n
    return found

  def describe_arms(self, polity: str) -> str:
    """A people's arms in words, for its own sheet."""
    held = {k: n for k, n in self.arms_of(polity).items() if n >= 0.05}
    if not held:  # (The moves that raise them say so.)
      return ''
    kinds = ' and '.join(
        f'{n:.1f} {self.arms.get(k, {}).get("label", k)}'
        for k, n in sorted(held.items())
    )
    strength = sum(self.strength(t) for t in self.places.get(polity, ()))
    garrisons = ', '.join(
        f'{self.strength(t):.1f} {self.where(t)}'
        for t in self.places.get(polity, ())
        if t.garrison
    )
    return (
        f'Our arms: {kinds} (strength {strength:.1f}; garrisons: {garrisons});'
        f' they eat {self.upkeep(polity):.1f} grain a turn.'
    )

  def describe_war(self, polity: str) -> str:
    """Armies a people has on the march, and those marching on its towns."""
    words = []
    for army in self.armies:
      found = self.find(army.target)
      if found is None:
        continue
      town, holder = found
      kinds = ' and '.join(
          f'{n:.1f} {self.arms.get(k, {}).get("label", k)}'
          for k, n in sorted(army.units.items())
      )
      if army.polity == polity:
        words.append(
            f'Our army of {kinds} marches to {army.intent} a town of the'
            f' {self._name(holder)} {self.where(town)}, arriving in turn'
            f' {army.arrives}.'
        )
      elif holder == polity:
        words.append(
            f'An army of the {self._name(army.polity)} ({kinds}) marches to'
            f' {army.intent} our town {self.where(town)}, arriving in turn'
            f' {army.arrives}; it stands {self.defence(town):.1f} strong'
            ' with its people and walls.'
        )
    for holder, towns in self.places.items():
      for town in towns:
        need = self.need(town)
        if holder != polity and town.regard.get(polity, 0.0) > 1e-6:
          words.append(
              f'Our envoys court a town of the {self._name(holder)}'
              f' {self.where(town)}: its regard for us is'
              f' {town.regard[polity]:.2f} of the {need:.2f} it needs'
              f' ({self.to_go(town, polity):.0%} to go).'
          )
        if holder != polity:
          continue
        for other, regard in sorted(town.regard.items()):
          words.append(
              f'The {self._name(other)} court our town {self.where(town)}:'
              f' its regard for them is {regard:.2f} of the {need:.2f} it'
              f' needs to leave us ({self.to_go(town, other):.0%} to go).'
          )
        if town.taken:
          words.append(
              f'Our town {self.where(town)}, taken from the'
              f' {self._name(town.polity)}, holds to us at'
              f' {town.loyalty:.2f} (settled at 1.00).'
          )
    return ' '.join(words)

  def capital(self, polity: str) -> Settlement | None:
    """A people's capital (its first town if it has none)."""
    towns = self.places.get(polity) or []
    return next((t for t in towns if t.capital), towns[0] if towns else None)

  def nearest_town(
      self, polity: str, tile: tuple[int, int] | None
  ) -> tuple[Settlement, str] | None:
    """Another people's town nearest a tile (or the people's capital)."""
    home = self.capital(polity)
    if tile is None and home is None:
      return None
    r, c = tile if tile is not None else (home.row, home.col)  # type: ignore
    others = [
        (self._steps(r, c, t.row, t.col), t.id, t, k)
        for k, towns in self.places.items()
        if k != polity
        for t in towns
    ]
    if not others:
      return None
    _, _, town, holder = min(others, key=lambda x: (x[0], x[1]))
    return town, holder

  def march(
      self, polity: str, town: int, intent: str, tick: int
  ) -> tuple[Army | None, str]:
    """A people sends its capital's garrison to take or raze a town.

    Args:
      polity: The people that marches.
      town: The town marched on (another people's).
      intent: 'take' or 'raze'.
      tick: Now.

    Returns:
      The army, or None and why not.
    """
    home = self.capital(polity)
    found = self.find(town)
    if home is None:
      return None, 'they hold no town to march from'
    if found is None or found[1] == polity:
      return None, 'there is no town of another people there'
    if not home.garrison:
      return None, 'there are no arms in their capital'
    target = found[0]
    path = self._path((home.row, home.col), (target.row, target.col), True)
    if not path:
      return None, 'there is no way there'
    speed = min(
        int(self.arms.get(k, {}).get('speed', 1)) for k in home.garrison
    )
    per_turn = max(1, int(self.war.get('march_tiles', 8)) * speed)
    turns = max(1, -(-(len(path) - 1) // per_turn))
    army = Army(
        id=self.next_army,
        polity=polity,
        units=dict(home.garrison),
        path=path,
        at=0,
        target=town,
        intent=intent,
        arrives=tick + turns,
        per_turn=per_turn,
    )
    self.next_army += 1
    home.garrison = {}
    self.armies.append(army)
    return army, ''

  def army_strength(self, army: Army) -> float:
    return sum(
        n * float(self.arms.get(k, {}).get('strength', 1.0))
        for k, n in army.units.items()
    )

  def defence(self, place: Settlement) -> float:
    """What a town fights with: its garrison and its people, behind walls."""
    militia = float(
        self.war.get('capital_militia' if place.capital else 'militia', 1.0)
    )
    walls = 1.0 + float(self.war.get('walls_bonus', 0.5)) * place.walls
    return (self.strength(place) + militia) * walls

  def march_on(self, tick: int) -> list[Army]:
    """Armies move a turn's way; those arriving now (taken off the march)."""
    arrived = []
    for army in list(self.armies):
      if tick >= army.arrives:
        army.at = len(army.path) - 1
        self.armies.remove(army)
        arrived.append(army)
      else:
        army.at = min(len(army.path) - 1, army.at + army.per_turn)
    return arrived

  def fight(self, army: Army, draw: float) -> dict[str, Any]:
    """An army at its target: the fight, and what is left of each side.

    Args:
      army: The army, arrived.
      draw: A seeded draw in [0, 1).

    Returns:
      `{'outcome': 'won'|'repelled'|'gone', 'town', 'holder', 'survivors'}`:
      'gone' if the town is no longer another people's to fight for.
    """
    found = self.find(army.target)
    if found is None or found[1] == army.polity:
      return {'outcome': 'gone', 'survivors': dict(army.units)}
    town, holder = found
    attack, defend = self.army_strength(army), self.defence(town)
    won = draw < attack**2 / max(1e-9, attack**2 + defend**2)
    if won:  # The attackers lose in proportion to the defence they met.
      kept = max(0.1, 1.0 - 0.5 * min(1.0, defend / max(attack, 1e-9)))
      town.garrison = {}
    else:  # Thrown back: the attackers lose most; the defenders some.
      kept = 0.4
      hurt = 1.0 - min(0.5, 0.3 * attack / max(defend, 1e-9))
      town.garrison = {k: n * hurt for k, n in town.garrison.items()}
    survivors = {k: n * kept for k, n in army.units.items() if n * kept > 1e-3}
    return {
        'outcome': 'won' if won else 'repelled',
        'town': town,
        'holder': holder,
        'survivors': survivors,
    }

  def come_home(self, polity: str, units: Mapping[str, float]) -> None:
    """Survivors join their people's capital (if it still stands)."""
    for kind, n in units.items():
      self.muster(polity, kind, n)

  def raze(self, town: int) -> tuple[str, float]:
    """A town razed: gone from its holder; the holder and its share."""
    found = self.find(town)
    if found is None:
      raise KeyError(town)
    place, holder = found
    share = self.shares(holder).get(place.id, 0.0)
    self.places[holder].remove(place)
    if place.capital and self.places[holder]:
      self.places[holder][0].capital = True
    self._route()
    return holder, share

  def unrest(self) -> list[tuple[Settlement, str, str]]:
    """Taken towns settle when held by arms, or grow restless and rise.

    A taken town held until its loyalty is whole is settled; a town won
    over grows content a step a turn.

    Returns:
      (town, holder, 'warn' or 'rise') for each town that warns or rises.
    """
    step = float(self.war.get('loyalty_step', 0.05))
    need = float(self.war.get('hold_need', 2.0))
    rise_at = float(self.war.get('rise_at', 0.2))
    found = []
    for holder, towns in self.places.items():
      for town in towns:
        if town.polity == holder:
          continue
        held = not town.taken or self.strength(town) >= need
        town.loyalty = min(
            1.0, max(0.0, town.loyalty + (step if held else -step))
        )
        if town.loyalty >= 1.0:
          town.taken = False
        if not town.taken:
          continue
        if town.loyalty >= rise_at:
          town.warned = False
        elif town.warned and self.places.get(town.polity):
          found.append((town, holder, 'rise'))
        elif not town.warned:
          town.warned = True
          found.append((town, holder, 'warn'))
    return found

  def need(self, place: Settlement) -> float:
    """The regard a town needs to join another people (C5).

    Its loyalty to its holder, and never less than the floor, so a fresh
    conquest does not change hands for one cart of grain.
    """
    return max(place.loyalty, float(self.war.get('win_floor', 0.5)))

  def to_go(self, place: Settlement, polity: str) -> float:
    """How far a town is from joining a people, as a share of its need."""
    need = self.need(place)
    return max(0.0, need - place.regard.get(polity, 0.0)) / need

  def courtable(self, polity: str) -> list[tuple[Settlement, str]]:
    """Other peoples' towns a people may court (capitals hold to their own)."""
    return [
        (t, k)
        for k, towns in self.places.items()
        if k != polity
        for t in towns
        if not t.capital
    ]

  def best_court(
      self, polity: str, tile: tuple[int, int] | None = None
  ) -> tuple[Settlement, str] | None:
    """The town a people courts: nearest a tile, or nearest to joining it."""
    towns = self.courtable(polity)
    if tile is not None:
      found = self.nearest_town(polity, tile)
      return found
    home = self.capital(polity)
    if not towns or home is None:
      return None
    return min(
        towns,
        key=lambda tk: (
            round(self.to_go(tk[0], polity), 6),
            self._steps(home.row, home.col, tk[0].row, tk[0].col),
            tk[0].id,
        ),
    )

  def court(self, town: int, polity: str, gain: float) -> float:
    """A people's grain and envoys reach a town; its regard for them now."""
    found = self.find(town)
    if found is None:
      raise KeyError(town)
    place = found[0]
    regard = min(1.0, place.regard.get(polity, 0.0) + gain)
    place.regard[polity] = regard
    return regard

  def courted_by(self, place: Settlement) -> tuple[str, float] | None:
    """The people a town leans to most, and its regard for them."""
    if not place.regard:
      return None
    key = max(sorted(place.regard), key=lambda k: place.regard[k])
    return key, place.regard[key]

  def unsettled(
      self, polity: str, tile: tuple[int, int] | None = None
  ) -> Settlement | None:
    """A people's own town to feast: nearest a tile, or its least settled.

    Least settled: courted nearest to leaving it, or taken and restless.
    """
    towns = self.places.get(polity) or []
    if tile is not None:
      r, c = tile
      return min(
          towns,
          key=lambda t: (self._steps(r, c, t.row, t.col), t.id),
          default=None,
      )

    def at_risk(place: Settlement) -> float:
      leaning = self.courted_by(place)
      courted = leaning[1] / self.need(place) if leaning else 0.0
      restless = 1.0 - place.loyalty if place.taken else 0.0
      return max(courted, restless)

    risky = [t for t in towns if at_risk(t) > 1e-9]
    return min(risky, key=lambda t: (-at_risk(t), t.id), default=None)

  def feast(self, town: int, gain: float) -> None:
    """A town feasted by its holder: loyalty up, others' regard down."""
    found = self.find(town)
    if found is None:
      raise KeyError(town)
    place = found[0]
    place.loyalty = min(1.0, place.loyalty + gain)
    if place.loyalty >= float(self.war.get('rise_at', 0.2)):
      place.warned = False
    place.regard = {
        k: v - gain for k, v in place.regard.items() if v - gain > 1e-6
    }

  def fade_regard(self) -> None:
    """Regard not renewed fades a little each turn."""
    fade = float(self.war.get('regard_fade', 0.02))
    for towns in self.places.values():
      for place in towns:
        place.regard = {
            k: v - fade for k, v in place.regard.items() if v - fade > 1e-6
        }

  def muster(self, polity: str, kind: str, count: float) -> None:
    """New arms join the garrison of the people's capital."""
    capital = self.capital(polity)
    if capital is None or count <= 0:
      return
    capital.garrison[kind] = capital.garrison.get(kind, 0.0) + float(count)

  def upkeep(self, polity: str) -> float:
    """The grain a people's arms eat in a turn (on the march too)."""
    held = dict(self.arms_of(polity))
    for army in self.armies:
      if army.polity == polity:
        for kind, n in army.units.items():
          held[kind] = held.get(kind, 0.0) + n
    return sum(
        n * float(self.arms.get(k, {}).get('upkeep', 0.0))
        for k, n in held.items()
    )

  def desert(self, polity: str, share: float) -> dict[str, float]:
    """A share of a people's arms leaves every town; what was lost, by kind."""
    lost: dict[str, float] = {}
    for place in self.places.get(polity, ()):
      for kind, n in list(place.garrison.items()):
        gone = n * share
        lost[kind] = lost.get(kind, 0.0) + gone
        place.garrison[kind] = n - gone
        if place.garrison[kind] < 1e-6:
          del place.garrison[kind]
    return lost

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
        # Where conquest is played: each town's founder, loyalty, walls
        # and garrison.
        **(
            {
                'towns': {
                    str(s.id): [
                        s.polity,
                        s.loyalty,
                        s.walls,
                        dict(s.garrison),
                        s.warned,
                        s.taken,
                        dict(s.regard),
                    ]
                    for v in self.places.values()
                    for s in v
                },
                'armies': [dataclasses.asdict(a) for a in self.armies],
                'next_army': self.next_army,
            }
            if self.conquest
            else {}
        ),
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
    for town, (founder, loyalty, walls, garrison, *rest) in state.get(
        'towns', {}
    ).items():
      found = self.find(int(town))
      if found is not None:
        place, holder = found
        place.polity, place.loyalty = str(founder), float(loyalty)
        place.warned = bool(rest[0]) if rest else False
        # Older saves: every town held by another people was taken.
        place.taken = bool(rest[1]) if len(rest) > 1 else place.polity != holder
        place.regard = (
            {str(k): float(v) for k, v in rest[2].items()}
            if len(rest) > 2
            else {}
        )
        place.walls = int(walls)
        if isinstance(garrison, Mapping):
          place.garrison = {str(k): float(v) for k, v in garrison.items()}
        else:  # Older saves: one number, before arms had kinds.
          place.garrison = {'levies': float(garrison)} if garrison else {}
    self.armies = [
        Army(
            id=int(a['id']),
            polity=str(a['polity']),
            units={str(k): float(v) for k, v in a['units'].items()},
            path=[(int(r), int(c)) for r, c in a['path']],
            at=int(a['at']),
            target=int(a['target']),
            intent=str(a['intent']),
            arrives=int(a['arrives']),
            per_turn=int(a['per_turn']),
        )
        for a in state.get('armies', ())
    ]
    self.next_army = int(state.get('next_army', 1))
    self._route()
