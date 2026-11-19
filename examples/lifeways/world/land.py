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

"""The land: a toroidal grid of terrain whose food regrows each year."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# Terrain kinds, and the food each tile carries at most and regrows by.
TERRAIN = ('grass', 'forest', 'river', 'coast', 'hills', 'marsh')
CAPACITY = {
    'grass': 6.0,
    'forest': 5.0,
    'river': 9.0,
    'coast': 8.0,
    'hills': 2.5,
    'marsh': 4.0,
}
# Defaults: the logistic rate food regrows by in a year, and the share of
# its capacity even a bare tile regrows.
REGROWTH = 0.6
SEED_BANK = 0.1
# Fields: land sown by one who knows how carries this much more food per
# level of the sower's farming skill (times the land's richness), while it is
# tended; untended for `FALLOW` years, it goes back to the wild. Hills are
# never sown.
FIELD_YIELD = 3.0
FALLOW = 3
UNSOWABLE = ('hills',)

Tile = tuple[int, int]


class Land:
  """Terrain and food on a toroidal grid."""

  def __init__(
      self,
      rows: int,
      cols: int,
      seed: int,
      richness: float = 1.0,
      regrowth: float = REGROWTH,
      seed_bank: float = SEED_BANK,
      ridge: Mapping[str, Any] | None = None,
      ridges: Sequence[Mapping[str, Any]] = (),
      field_yield: float = FIELD_YIELD,
  ):
    """Makes the land.

    Args:
      rows: The map's rows.
      cols: The map's columns.
      seed: The world seed (the terrain is drawn from it).
      richness: Scales every tile's food.
      regrowth: The logistic rate food regrows by in a year.
      seed_bank: The share of its capacity even a bare tile regrows.
      ridge: Optional. Hills from edge to edge at column `col` (`width`
        columns wide) but for a pass at rows `gap` (first and last; no pass
        if absent).
      ridges: More ridges, each as `ridge`. The map wraps east to west, so
        two ridges are needed to part two valleys altogether.
      field_yield: How much more food a field carries per level of its
        sower's skill (times the richness).
    """
    self.rows, self.cols = int(rows), int(cols)
    # Scales every tile's food; with the regrowth, how many the land feeds.
    self.richness = float(richness)
    self.regrowth = float(regrowth)
    self.seed_bank = float(seed_bank)
    self.field_yield = float(field_yield)
    # The fields: each sown tile's level, the year it was last tended and
    # its latest tender (whose field it is).
    self.fields: dict[Tile, tuple[int, int, str]] = {}
    rng = np.random.default_rng([seed, 1])
    # A smooth random field picks the terrain: rivers in its valleys, hills
    # on its heights, a coast along one edge.
    field = rng.random((self.rows, self.cols))
    for _ in range(3):
      field = (
          field
          + np.roll(field, 1, 0)
          + np.roll(field, -1, 0)
          + np.roll(field, 1, 1)
          + np.roll(field, -1, 1)
      ) / 5.0
    lo, hi = np.quantile(field, [0.12, 0.85])
    wet = rng.random((self.rows, self.cols))
    terrain = np.full((self.rows, self.cols), TERRAIN.index('grass'))
    terrain[field > hi] = TERRAIN.index('hills')
    terrain[field < lo] = TERRAIN.index('river')
    terrain[(field >= lo) & (field <= hi) & (wet < 0.25)] = TERRAIN.index(
        'forest'
    )
    terrain[(field >= lo) & (field <= hi) & (wet > 0.93)] = TERRAIN.index(
        'marsh'
    )
    terrain[:, 0] = TERRAIN.index('coast')
    for one in ([ridge] if ridge else []) + list(ridges):
      # A line of hills from edge to edge, but for a pass: two valleys.
      col, width = int(one['col']), int(one.get('width', 2))
      low, high = (int(r) for r in one.get('gap', (0, -1)))
      for row in range(self.rows):
        kind = 'grass' if low <= row <= high else 'hills'
        terrain[row, col : col + width] = TERRAIN.index(kind)
    self.terrain = terrain
    self.food = self._capacities()

  def _capacities(self) -> np.ndarray:
    capacity = np.array(
        [
            [CAPACITY[TERRAIN[t]] * self.richness for t in row]
            for row in self.terrain
        ],
        dtype=float,
    )
    for tile, (level, _, _) in self.fields.items():
      capacity[tile] += self.field_yield * level * self.richness
    return capacity

  def kind(self, tile: Tile) -> str:
    return TERRAIN[int(self.terrain[tile])]

  def capacity(self, tile: Tile) -> float:
    level = self.fields.get(tile, (0, 0, ''))[0]
    return (CAPACITY[self.kind(tile)] + self.field_yield * level) * (
        self.richness
    )

  def sowable(self, tile: Tile) -> bool:
    return self.kind(tile) not in UNSOWABLE

  def sow(self, tile: Tile, level: int, year: int, tender: str = '') -> None:
    """Sows or tends a field: its level is the best of those who sowed it.

    Args:
      tile: Where.
      level: The sower's farming skill.
      year: This year (the field is tended in it).
      tender: Who sows it; the field is its latest tender's.
    """
    if not self.sowable(tile) or level < 1:
      return
    best = max(level, self.fields.get(tile, (0, year, ''))[0])
    self.fields[tile] = (best, year, tender)

  def pass_on(self, tender: str, heir: str) -> list[Tile]:
    """Makes a tender's fields an heir's, as they stand; returns them."""
    passed = sorted(
        t for t, (_, _, who) in self.fields.items() if who == tender
    )
    for tile in passed:
      level, tended, _ = self.fields[tile]
      self.fields[tile] = (level, tended, heir)
    return passed

  def fields_of_others(self, tile: Tile, name: str) -> list[Tile]:
    """The fields within a day's walk of a tile that others tend."""
    return [
        t
        for t in self.around(tile)
        if t in self.fields and self.fields[t][2] not in ('', name)
    ]

  def fallow(self, year: int, years: int = FALLOW) -> list[Tile]:
    """Lets fields untended for `years` go wild; returns them.

    Their food falls to the wild land's at the next regrowth.

    Args:
      year: This year.
      years: How long a field lasts untended.

    Returns:
      The fields gone wild.
    """
    wild = sorted(
        t for t, (_, tended, _) in self.fields.items() if year - tended >= years
    )
    for tile in wild:
      del self.fields[tile]
    return wild

  def wrap(self, row: int, col: int) -> Tile:
    return (row % self.rows, col % self.cols)

  def around(self, tile: Tile, radius: int = 1) -> list[Tile]:
    """The tiles within `radius` steps (the tile itself first)."""
    r, c = tile
    found = [tile]
    for dr in range(-radius, radius + 1):
      for dc in range(-radius, radius + 1):
        if (dr, dc) != (0, 0):
          found.append(self.wrap(r + dr, c + dc))
    return found

  def best_near(self, tile: Tile, radius: int = 1) -> Tile:
    """The tile within `radius` steps whose surroundings hold most food.

    Args:
      tile: Where from.
      radius: How far a person looks for better land.

    Returns:
      The best tile (ties: the nearest in the order `around` gives).
    """
    return max(
        self.around(tile, radius),
        key=lambda t: (round(self.around_food(t), 6),),
    )

  def around_food(self, tile: Tile) -> float:
    """The food within a day's walk (the tile and its neighbours)."""
    return float(sum(self.food[t] for t in self.around(tile)))

  def gather(
      self, tile: Tile, effort: float, spare: Sequence[Tile] = ()
  ) -> float:
    """Forages food around a tile, richest tiles first.

    Args:
      tile: Where the forager lives.
      effort: The most food the forager can take this year.
      spare: Tiles the forager leaves alone (e.g. others' fields).

    Returns:
      The food taken.
    """
    wanted = max(0.0, effort)
    taken = 0.0
    for place in sorted(
        (t for t in self.around(tile) if t not in spare),
        key=lambda t: (-round(self.food[t], 6), t),
    ):
      if taken >= wanted:
        break
      got = float(min(self.food[place], wanted - taken))
      self.food[place] -= got
      taken += got
    return taken

  def regrow(self, season: float = 1.0) -> None:
    """Food grows back, `season` times as much as in a usual year."""
    capacity = self._capacities()
    self.food += season * (
        self.regrowth * self.food * (1.0 - self.food / capacity)
        + self.seed_bank * capacity
    )
    np.minimum(self.food, capacity, out=self.food)

  def get_state(self) -> dict[str, Any]:
    return {
        'rows': self.rows,
        'cols': self.cols,
        'richness': self.richness,
        'terrain': self.terrain.astype(int).tolist(),
        'food': self.food.tolist(),
        'fields': [
            [r, c, level, tended, tender]
            for (r, c), (level, tended, tender) in sorted(self.fields.items())
        ],
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Restores the terrain, food and fields; the settings stay this run's.

    Args:
      state: `get_state()` of a land of the same size.

    Raises:
      ValueError: If the saved land has another size.
    """
    if (int(state['rows']), int(state['cols'])) != (self.rows, self.cols):
      raise ValueError(
          f'A {state["rows"]}x{state["cols"]} land cannot be restored into a'
          f' {self.rows}x{self.cols} one.'
      )
    self.terrain = np.array(state['terrain'], dtype=int)
    self.food = np.array(state['food'], dtype=float)
    self.fields = {
        (int(r), int(c)): (
            int(level),
            int(tended),
            str(rest[0]) if rest else '',
        )
        for r, c, level, tended, *rest in state.get('fields', ())
    }
