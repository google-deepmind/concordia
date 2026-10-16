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

"""Mobile animals on the tile map.

Animals live in herds and packs, each with a tile and a size. Everything
they do is local:

* **Grazing.** A grazer herd eats the plants on its own tile with a Holling
  type II response, cropping that tile's cover, and grows on what it eats,
  less mortality and crowding by herds of its kind nearby.
* **Predation.** A pack hunts the prey herds on its own and the eight
  neighbouring tiles, killing in proportion to its catch.
* **Movement.** Each tick a herd may move a few tiles: grazers towards food
  and a climate that suits them and away from packs (fear) and crowds;
  packs towards prey. Land animals keep off the sea and the ice.
* **Disease.** Two large herds of a kind side by side may start an
  outbreak that thins every herd of that kind nearby.
* **Fire.** A herd whose tile burns loses part of its number.
* **The seas.** Where peoples fish, schools of fish swim the shelves and
  reefs; their total follows the fisheries' shared stock, and they flee the
  dead zones (`world/pollution.py` places them).
* **Migration.** Flocks of geese fly to summer grounds poleward and back to
  winter grounds on alternate ticks, grazing lightly, and sow the seeds of
  what they ate where they land.

A herd grown large splits in two; one grown too small is gone. A species
lives on a world whose plants (or prey) it can eat. Draws are seeded per
tick, so a world's herds move the same way on every replay.
"""

# pylint: disable=g-complex-comprehension,g-doc-args,g-doc-return-or-yield,missing-function-docstring

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from examples.deep_time.world import biomes as biomes_lib
from examples.deep_time.world import news as news_lib
from examples.deep_time.world import tiles as tiles_lib
import numpy as np


@dataclasses.dataclass
class Herd:
  id: int
  row: int  # Tile row, south first, like every tile array.
  col: int
  size: float


# Kinds that walk the land (their tracks are kept), and how many tiles back.
WALKERS = ('grazer', 'predator')
TRAIL_LENGTH = 8
# Flocks are the region grazers (`world/fauna.py`) drawn on the map; they
# leave tracks too.
TRACKED = WALKERS + ('flock',)


def _weight(table: Mapping[str, float], key: str) -> float:
  """A table's value for a key, or for the key's parent (`grass2`)."""
  if key in table:
    return float(table[key])
  return float(table.get(key.rstrip('0123456789'), 0.0))


class Herds:
  """Every mobile species on the map, herd by herd."""

  def __init__(
      self,
      tile_map: tiles_lib.TileMap,
      biomes: biomes_lib.Biomes,
      covers: Mapping[str, np.ndarray],
      config: Mapping[str, Any],
      seed: int,
      fisheries: bool = False,
      flocks: bool = False,
  ):
    self.tiles = tile_map
    self.biomes = biomes
    self.config = config
    self.seed = int(seed)
    present: list[Mapping[str, Any]] = []
    for spec in config['species']:
      kind = spec['kind']
      if kind in ('grazer', 'migrant'):
        if any(_weight(spec['eats'], k) > 0 for k in covers):
          present.append(spec)
      elif kind == 'school':
        if fisheries:
          present.append(spec)
      elif kind == 'flock':
        if flocks:
          present.append(spec)
      elif any(p['key'] in spec['hunts'] for p in present):
        present.append(spec)
    self.dead = np.zeros(tile_map.land.shape, dtype=bool)
    self._tick = 0
    self.specs = {s['key']: s for s in present}
    self.herds: dict[str, list[Herd]] = {k: [] for k in self.specs}
    # Where each walking herd or pack has been (tiles, oldest first): the
    # map draws it as a fading track, so a migration reads at a glance.
    self.trails: dict[int, list[tuple[int, int]]] = {}
    # Tracks daughters take from their mothers, recorded with the next tiles.
    self._inherited: dict[int, list[tuple[int, int]]] = {}
    # The flocks' diet (share of white petals), which colours them.
    self.flock_diet = 0.5
    self.next_id = 1
    rng = np.random.default_rng([self.seed, 7071])
    temps = biomes.temperature
    for key, spec in self.specs.items():
      if spec['kind'] == 'flock':
        continue  # Placed from the region grazers (`place_flocks`).
      if spec['kind'] in ('grazer', 'migrant'):
        score = self.food(spec, covers) * self.suitability(spec, temps)
      elif spec['kind'] == 'school':
        score = self.waters(spec).copy()
      else:  # Packs start among the herds they hunt.
        prey = self._density(spec['hunts'])
        score = self._near(prey) * self.suitability(spec, temps)
      score = score * self.habitat(spec)
      count = spec['schools'] if spec['kind'] == 'school' else spec['herds']
      for _ in range(int(count)):
        if score.sum() <= 0:
          break
        flat = score.ravel() / score.sum()
        cell = int(rng.choice(flat.size, p=flat))
        row, col = divmod(cell, tile_map.cols)
        self._add(key, row, col, float(spec.get('start_size', 1.0)))
        # Keep the next herds a little apart.
        score[max(0, row - 1) : row + 2, :] *= np.where(
            np.abs(
                (
                    (np.arange(tile_map.cols) - col + tile_map.cols // 2)
                    % tile_map.cols
                )
                - tile_map.cols // 2
            )
            <= 1,
            0.15,
            1.0,
        )

  def _add(self, species: str, row: int, col: int, size: float) -> Herd:
    herd = Herd(self.next_id, int(row), int(col), float(size))
    self.next_id += 1
    self.herds[species].append(herd)
    return herd

  def habitat(self, spec: Mapping[str, Any] | None = None) -> np.ndarray:
    """Where a species may be: ice-free land, or open sea for schools.

    Flocks fly anywhere but land only on land.
    """
    if spec is not None and spec['kind'] == 'school':
      return ((self.waters(spec) > 0) & ~self.tiles.ice).astype(float)
    return (self.tiles.land & ~self.tiles.ice).astype(float)

  def waters(self, spec: Mapping[str, Any]) -> np.ndarray:
    """How much a school likes each tile: reefs, then shelf, then deep."""
    likes = spec['habitats']
    terrain = self.tiles.terrain
    out = np.where(
        terrain == tiles_lib.SHELF,
        float(likes['shelf']),
        np.where(terrain == tiles_lib.DEEP_OCEAN, float(likes['deep']), 0.0),
    )
    reef = self.biomes.index.get('reef')
    if reef is not None:
      out = np.where(self.biomes.biome == reef, float(likes['reef']), out)
    return out

  def suitability(
      self, spec: Mapping[str, Any], temps: np.ndarray
  ) -> np.ndarray:
    width, optimum = float(spec['width']), float(spec['optimum_c'])
    return np.clip(1.0 - width * (temps - optimum) ** 2, 0.0, 1.0)

  def food(
      self, spec: Mapping[str, Any], covers: Mapping[str, np.ndarray]
  ) -> np.ndarray:
    """Edible cover per tile for a grazer, 0 to 1."""
    total = np.zeros(self.tiles.land.shape)
    for key, cover in covers.items():
      total += _weight(spec['eats'], key) * cover
    return np.clip(total, 0.0, 1.0)

  def _density(self, species: Sequence[str]) -> np.ndarray:
    """Herd units of these species per tile."""
    out = np.zeros(self.tiles.land.shape)
    for key in species:
      for h in self.herds.get(key, []):
        out[h.row, h.col] += h.size
    return out

  def _near(self, values: np.ndarray) -> np.ndarray:
    """Sum over each tile and its eight neighbours (wrapping in longitude)."""
    out = np.zeros_like(values)
    rows = self.tiles.rows
    for dy in (-1, 0, 1):
      shifted = np.zeros_like(values)
      if dy == 0:
        shifted = values
      elif dy > 0:
        shifted[dy:] = values[: rows - dy]
      else:
        shifted[: rows + dy] = values[-dy:]
      for dx in (-1, 0, 1):
        out += np.roll(shifted, dx, axis=1)
    return out

  def _around(self, row: int, col: int, reach: int) -> list[tuple[int, int]]:
    rows, cols = self.tiles.rows, self.tiles.cols
    return [
        (r, (col + dc) % cols)
        for r in range(max(0, row - reach), min(rows, row + reach + 1))
        for dc in range(-reach, reach + 1)
    ]

  def thin(
      self, species: str, fraction: float, near: np.ndarray | None = None
  ) -> float:
    """Removes a share of a kind's herds (near a mask, or all); returns it."""
    lost = 0.0
    for herd in self.herds.get(species, []):
      if near is None or near[herd.row, herd.col]:
        cut = herd.size * min(1.0, max(0.0, fraction))
        herd.size -= cut
        lost += cut
    return lost

  def census(self) -> dict[str, dict[str, Any]]:
    """Each kind's herds: how many, their total size and where most are."""
    out = {}
    for key, spec in self.specs.items():
      if spec['kind'] == 'flock':
        continue  # The region grazers are in the census already.
      herds = self.herds[key]
      if not herds:
        out[spec['name']] = {'herds': 0, 'size': 0.0, 'lat': None}
        continue
      weight = sum(h.size for h in herds)
      lat = sum(float(self.biomes.lat[h.row]) * h.size for h in herds) / max(
          1e-9, weight
      )
      out[spec['name']] = {
          'herds': len(herds),
          'size': round(weight, 2),
          'lat': round(lat, 0),
      }
    return out

  def total(self, species: str) -> float:
    return sum(h.size for h in self.herds.get(species, []))

  def step(
      self,
      tick: int,
      covers: Mapping[str, np.ndarray],
      burned: np.ndarray,
      floor: np.ndarray,
      fish_stock: float | None = None,
      dead: np.ndarray | None = None,
  ) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """One tick of life on the move.

    Args:
      tick: The world's tick (seeds the draws; its parity is the season).
      covers: Plant cover per species per tile (not modified).
      burned: Tiles that burned this tick.
      floor: Cover no grazing takes a plant below, per tile.
      fish_stock: The fisheries' shared stock (0-1), if there are any.
      dead: The dead zones (`world/pollution.py`), if there are any.

    Returns:
      News, the cover each plant species lost to grazing per tile, and the
      cover flocks sowed per tile.
    """
    rng = np.random.default_rng([self.seed, 7171, int(tick)])
    temps = self.biomes.temperature
    news: list[str] = []
    self.dead = (
        np.zeros(self.tiles.land.shape, dtype=bool) if dead is None else dead
    )
    eaten = {k: np.zeros(self.tiles.land.shape) for k in covers}
    sown = {k: np.zeros(self.tiles.land.shape) for k in covers}
    self._tick = int(tick)
    loss = float(self.config['fire_loss'])
    for herds in self.herds.values():
      for h in herds:
        if burned[h.row, h.col]:
          h.size *= 1.0 - loss
    for key, spec in self.specs.items():
      kind = spec['kind']
      if kind == 'flock':
        continue  # Driven by the region grazers (`place_flocks`).
      if kind == 'school':
        self._shoal(key, spec, fish_stock or 0.0, rng)
        continue
      suit = self.suitability(spec, temps)
      crowd = self._density([key])
      if kind in ('grazer', 'migrant'):
        self._graze(spec, self.herds[key], covers, eaten, floor, suit, crowd)
      else:
        self._hunt(spec, self.herds[key], suit, crowd)
    news += self._outbreaks(rng)
    for key, spec in self.specs.items():
      if spec['kind'] == 'flock':
        continue
      self._move(key, spec, covers, eaten, sown, rng)
      if spec['kind'] != 'school':
        news += self._split_and_die(key, spec)
    self._record_trails(WALKERS)
    return news, eaten, sown

  def _record_trails(self, kinds: Sequence[str]) -> None:
    """Adds these kinds' tiles to their tracks; forgets the tracks of the lost.

    The map thread reads the tracks while the game moves: a new dict is built
    and swapped in whole, never changed in place.
    """
    trails = {**self.trails, **self._inherited}
    self._inherited = {}
    for key, spec in self.specs.items():
      if spec['kind'] in kinds:
        for herd in self.herds[key]:
          trail = trails.get(herd.id, []) + [(herd.row, herd.col)]
          trails[herd.id] = trail[-TRAIL_LENGTH:]
    live = {
        h.id
        for key, spec in self.specs.items()
        if spec['kind'] in TRACKED
        for h in self.herds[key]
    }
    self.trails = {i: t for i, t in trails.items() if i in live}

  def place_flocks(
      self,
      tick: int,
      grazers: Any,
      covers: Mapping[str, np.ndarray],
      record: bool = True,
  ) -> None:
    """Draws the region grazers as flocks on their regions' best pasture.

    The grazers live per region (`world/fauna.py`): here each region's
    biomass becomes whole flocks on its land, which keep their identity from
    tick to tick, so they glide and leave tracks. A region with too few takes
    the nearest spare flock (within `wander` tiles), as herds wander; then
    each flock steps toward the tile with most of the petals it prefers,
    where the climate suits it.

    Args:
      tick: The tick (seeds the tie-breaking).
      grazers: The region grazers (`density`, `diet_white`, `suitability`).
      covers: Plant cover per tile ('white' and 'black' petals).
      record: Whether to add this tick's tiles to the flocks' tracks.
    """
    key = next((k for k, v in self.specs.items() if v['kind'] == 'flock'), '')
    if not key:
      return
    spec, tiles = self.specs[key], self.tiles
    region, cols = tiles.region, tiles.cols
    land = tiles.land & ~tiles.ice
    diet = float(grazers.diet_white)
    self.flock_diet = diet
    zero = np.zeros(land.shape)
    food = diet * covers.get('white', zero) + (1.0 - diet) * covers.get(
        'black', zero
    )
    suit = grazers.suitability(self.biomes.temperature)
    rng = np.random.default_rng([self.seed, int(tick), 5151])
    # Near-equal pasture is told apart by chance, so flocks scatter over it
    # rather than lining up on the one best row of each region.
    score = (np.clip(food, 0.0, 1.0) * suit + 1e-3) * land
    score = score * (0.75 + 0.5 * rng.random(land.shape))
    regions = len(grazers.density)
    land_tiles = np.bincount(region[land], minlength=regions)
    mass = np.asarray(grazers.density, dtype=float) * land_tiles
    per = float(spec['per_herd'])
    # How many flocks in all (one per `per_herd` of biomass), shared among
    # the regions by largest remainder, so the count follows the biomass.
    count = min(int(spec['max_herds']), int(mass.sum() / per + 0.5))
    quota = mass / max(mass.sum(), 1e-12) * count
    want = np.floor(quota).astype(int)
    for k in np.argsort(-(quota - want), kind='stable')[: count - want.sum()]:
      want[k] += 1

    def steps(a: tuple[int, int], b: tuple[int, int]) -> int:
      dc = abs(a[1] - b[1])
      return max(abs(a[0] - b[0]), min(dc, cols - dc))

    by_region: dict[int, list[Herd]] = {k: [] for k in range(regions)}
    for flock in self.herds[key]:
      by_region[int(region[flock.row, flock.col])].append(flock)
    spare: list[Herd] = []
    for k in range(regions):
      have = sorted(by_region[k], key=lambda f: -score[f.row, f.col])
      spare += have[int(want[k]) :]
      by_region[k] = have[: int(want[k])]
    taken = {(f.row, f.col) for v in by_region.values() for f in v}
    wander = int(spec['wander'])
    for k in range(regions):
      while len(by_region[k]) < want[k]:
        cells = np.argwhere((region == k) & land)
        free = [tuple(c) for c in cells if tuple(c) not in taken] or [
            tuple(c) for c in cells
        ]
        best = max(free, key=lambda c: score[c])
        near = [f for f in spare if steps((f.row, f.col), best) <= wander]
        if near:
          flock = min(
              near, key=lambda f, b=best: (steps((f.row, f.col), b), f.id)
          )
          spare.remove(flock)
          flock.row, flock.col = int(best[0]), int(best[1])
        else:
          flock = Herd(self.next_id, int(best[0]), int(best[1]), 1.0)
          self.next_id += 1
        taken.add((flock.row, flock.col))
        by_region[k].append(flock)
    reach = int(spec['range'])
    kept = []
    for k in range(regions):
      for flock in sorted(by_region[k], key=lambda f: f.id):
        taken.discard((flock.row, flock.col))
        spots = [
            (r, c)
            for r, c in self._around(flock.row, flock.col, reach)
            if region[r, c] == k and land[r, c] and (r, c) not in taken
        ] or [(flock.row, flock.col)]
        flock.row, flock.col = max(spots, key=lambda rc: score[rc])
        taken.add((flock.row, flock.col))
        flock.size = float(mass.sum() / per / max(1, count))  # Equal shares.
        kept.append(flock)
    self.herds[key] = sorted(kept, key=lambda f: f.id)
    if record:
      self._record_trails(('flock',))

  def _shoal(self, key, spec, stock, rng) -> None:
    """Fish schools follow the fisheries' stock: fewer as it is fished."""
    target = max(0.0, stock) * float(spec['schools'])
    count = int(round(target))
    schools = self.herds[key]
    while len(schools) > count:  # The newest schools are lost first.
      schools.pop()
    if count:
      waters = self.waters(spec) * (1.0 - self.dead) * self.habitat(spec)
      taken = self._density([key])
      while len(schools) < count and waters.sum() > 0:
        score = waters * (taken == 0)
        if score.sum() <= 0:
          score = waters
        flat = score.ravel() / score.sum()
        row, col = divmod(int(rng.choice(flat.size, p=flat)), self.tiles.cols)
        schools.append(Herd(self.next_id, row, col, 1.0))
        self.next_id += 1
        taken[row, col] += 1.0
      for school in schools:
        school.size = target / len(schools) if schools else 0.0

  def _graze(self, spec, herds, covers, eaten, floor, suit, crowd) -> None:
    a, h_time = float(spec['attack']), float(spec['handling'])
    for herd in herds:
      r, c = herd.row, herd.col
      left = {
          k: max(0.0, float(v[r, c] - eaten[k][r, c] - floor[r, c]))
          for k, v in covers.items()
      }
      prefs = {k: _weight(spec['eats'], k) for k in left}
      food = min(1.0, sum(prefs[k] * left[k] for k in left))
      intake = a * food / (1.0 + a * h_time * food) * float(suit[r, c])
      crop = float(spec['bite']) * intake * herd.size
      weights = {k: prefs[k] * left[k] for k in left}
      total = sum(weights.values())
      if total > 0:
        for k, w in weights.items():
          eaten[k][r, c] += min(left[k], crop * w / total)
      growth = (
          float(spec['efficiency']) * intake
          - float(spec['mortality'])
          - float(spec['crowding']) * float(crowd[r, c])
      )
      herd.size = max(0.0, herd.size * (1.0 + growth))

  def _hunt(self, spec, packs, suit, crowd) -> None:
    a, h_time = float(spec['attack']), float(spec['handling'])
    for pack in packs:
      prey = [
          herd
          for key in spec['hunts']
          for herd in self.herds.get(key, [])
          if abs(herd.row - pack.row) <= 1
          and min(
              (herd.col - pack.col) % self.tiles.cols,
              (pack.col - herd.col) % self.tiles.cols,
          )
          <= 1
      ]
      catch = sum(p.size for p in prey)
      intake = a * catch / (1.0 + a * h_time * catch)
      intake *= float(suit[pack.row, pack.col])
      kill = min(catch, float(spec['kill']) * intake * pack.size)
      for p in prey:
        p.size -= kill * p.size / catch if catch > 0 else 0.0
      growth = (
          float(spec['efficiency']) * intake
          - float(spec['mortality'])
          - float(spec['crowding']) * float(crowd[pack.row, pack.col])
      )
      pack.size = max(0.0, pack.size * (1.0 + growth))

  def _outbreaks(self, rng: np.random.Generator) -> list[str]:
    c = self.config['disease']
    big, reach = float(c['min_size']), int(c['reach'])
    news = []
    for key, herds in self.herds.items():
      crowded = [h for h in herds if h.size >= big]
      for i, one in enumerate(crowded):
        for two in crowded[i + 1 :]:
          if (
              abs(one.row - two.row) > 1
              or min(
                  (one.col - two.col) % self.tiles.cols,
                  (two.col - one.col) % self.tiles.cols,
              )
              > 1
          ):
            continue
          if rng.random() >= float(c['chance']):
            continue
          struck = [
              h
              for h in herds
              if abs(h.row - one.row) <= reach
              and min(
                  (h.col - one.col) % self.tiles.cols,
                  (one.col - h.col) % self.tiles.cols,
              )
              <= reach
          ]
          for h in struck:
            h.size *= 1.0 - float(c['loss'])
          if len(struck) >= int(self.config['news_herds']):
            news.append(
                news_lib.located(
                    'Disease runs through the'
                    f' {self.specs[key]["name"].lower()}'
                    f' {self._where(one.row)}: {len(struck)} herds sicken.',
                    (one.row, one.col),
                )
            )
          break
    return news

  def _where(self, row: int) -> str:
    lat = float(self.biomes.lat[row])
    if lat > 50:
      return 'in the far north'
    if lat > 20:
      return 'in the north'
    if lat > -20:
      return 'in the tropics'
    if lat > -50:
      return 'in the south'
    return 'in the far south'

  def _move(self, key, spec, covers, eaten, sown, rng) -> None:
    temps = self.biomes.temperature
    kind = spec['kind']
    habitat = self.habitat(spec)
    if kind == 'school':
      draw = self.waters(spec) * (1.0 - self.dead)
    elif kind == 'migrant':
      suit = self.suitability(spec, temps)
      left = {k: covers[k] - eaten[k] for k in covers}
      draw = 0.8 * self.food(spec, left) * suit
    elif kind == 'grazer':
      suit = self.suitability(spec, temps)
      left = {k: covers[k] - eaten[k] for k in covers}
      draw = self.food(spec, left) * suit
      predators = [
          k
          for k, s in self.specs.items()
          if s['kind'] == 'predator' and key in s['hunts']
      ]
      danger = self._near(self._density(predators))
      draw = draw - float(self.config['fear']) * np.minimum(1.0, danger)
    else:
      suit = self.suitability(spec, temps)
      draw = self._near(self._density(spec['hunts'])) * suit
    crowd = self._density([key])
    reach = int(spec['range'])
    northern_summer = self._tick % 2 == 0
    for herd in self.herds[key]:
      crowd[herd.row, herd.col] -= herd.size
      pull = draw
      if kind == 'migrant':
        # Summer grounds poleward, winter grounds nearer the equator, in
        # the flock's own hemisphere.
        north = self.biomes.lat[herd.row] >= 0
        summer = northern_summer == north
        goal = float(spec['summer_lat'] if summer else spec['winter_lat'])
        goal = goal if north else -goal
        pull = draw - np.abs(self.biomes.lat - goal)[:, None] / 30.0
      best, best_score = (herd.row, herd.col), -np.inf
      for r, c in self._around(herd.row, herd.col, reach):
        if habitat[r, c] <= 0:
          continue
        score = (
            float(pull[r, c])
            - 0.15 * float(crowd[r, c])
            + 0.03 * float(rng.random())
        )
        if score > best_score:
          best, best_score = (r, c), score
      if kind == 'migrant' and best != (herd.row, herd.col):
        self._sow(spec, herd, best, covers, eaten, sown)
      herd.row, herd.col = best
      crowd[herd.row, herd.col] += herd.size

  def _sow(self, spec, herd, to, covers, eaten, sown) -> None:
    """A flock carries the seeds of what it ate to where it lands."""
    r, c = herd.row, herd.col
    tr, tc = to
    left = {k: float(covers[k][r, c] - eaten[k][r, c]) for k in covers}
    weights = {k: _weight(spec['eats'], k) * v for k, v in left.items()}
    total = sum(weights.values())
    if total <= 0:
      return
    bare = float(self.tiles.land[tr, tc]) - sum(
        float(covers[k][tr, tc] - eaten[k][tr, tc] + sown[k][tr, tc])
        for k in covers
    )
    seeds = min(max(0.0, bare), float(spec['sow']) * herd.size)
    for k, w in weights.items():
      sown[k][tr, tc] += seeds * w / total

  def _split_and_die(self, key: str, spec: Mapping[str, Any]) -> list[str]:
    herds = self.herds[key]
    had = bool(herds)
    alive = [h for h in herds if h.size >= float(spec['die_below'])]
    self.herds[key] = alive
    habitat = self.habitat()
    for herd in list(alive):
      if herd.size < float(spec['split_at']):
        continue
      if len(self.herds[key]) >= int(spec['max_herds']):
        herd.size = float(spec['split_at'])
        continue
      spots = [
          (r, c)
          for r, c in self._around(herd.row, herd.col, 1)
          if (r, c) != (herd.row, herd.col) and habitat[r, c] > 0
      ]
      if not spots:
        continue
      herd.size /= 2.0
      r, c = spots[(herd.id * 7 + len(alive)) % len(spots)]
      daughter = self._add(key, r, c, herd.size)
      self._inherited[daughter.id] = list(self.trails.get(herd.id, []))
    if had and not self.herds[key]:
      return [f'The last of the {spec["name"].lower()} are gone.']
    return []

  def payload(self) -> dict[str, Any]:
    """For the map: the species, and every herd (rows north first)."""
    keys = list(self.specs)
    return {
        'species': [self._legend(k) for k in keys],
        'herds': [
            [i, h.id, self.tiles.rows - 1 - h.row, h.col, round(h.size, 2)]
            for i, k in enumerate(keys)
            for h in self.herds[k]
        ],
        # Each walker's track, before the tile it stands on now.
        'trails': {
            str(i): [[self.tiles.rows - 1 - r, c] for r, c in t[:-1]]
            for i, t in self.trails.items()
            if len(t) > 1
        },
    }

  def _legend(self, key: str) -> dict[str, Any]:
    """How the map draws and names a kind (flocks by what they eat)."""
    spec = self.specs[key]
    out = {
        'key': key,
        'name': spec['name'],
        'kind': spec['kind'],
        'light': list(spec['light']),
        'dark': list(spec['dark']),
    }
    if spec['kind'] == 'flock':
      # Pale flocks crop white petals (darkening the world), dark ones black.
      w = self.flock_diet
      for theme in ('light', 'dark'):
        dark, pale = spec[theme]
        out[theme] = [round(d + (p - d) * w) for d, p in zip(dark, pale)]
      if w >= 0.6:
        out['name'] += ' (eat white petals)'
      elif w <= 0.4:
        out['name'] += ' (eat black petals)'
    return out

  def get_state(self) -> dict[str, Any]:
    return {
        'next_id': self.next_id,
        'flock_diet': self.flock_diet,
        'herds': {
            k: [[h.id, h.row, h.col, h.size] for h in v]
            for k, v in self.herds.items()
        },
        'trails': {
            str(i): [list(p) for p in t] for i, t in self.trails.items()
        },
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.next_id = int(state['next_id'])
    self.herds = {k: [] for k in self.specs}  # Older saves lack new kinds.
    for k, v in state['herds'].items():
      if k in self.specs:
        self.herds[k] = [
            Herd(int(i), int(r), int(c), float(s)) for i, r, c, s in v
        ]
    self.flock_diet = float(state.get('flock_diet', 0.5))
    self.trails = {  # Older saves have none: tracks start afresh.
        int(i): [(int(r), int(c)) for r, c in t]
        for i, t in state.get('trails', {}).items()
    }
