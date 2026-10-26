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

from collections.abc import Mapping
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
  ):
    self.rows, self.cols = int(rows), int(cols)
    # Scales every tile's food; with the regrowth, how many the land feeds.
    self.richness = float(richness)
    self.regrowth = float(regrowth)
    self.seed_bank = float(seed_bank)
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
    self.terrain = terrain
    self.food = self._capacities()

  def _capacities(self) -> np.ndarray:
    return np.array(
        [
            [CAPACITY[TERRAIN[t]] * self.richness for t in row]
            for row in self.terrain
        ],
        dtype=float,
    )

  def kind(self, tile: Tile) -> str:
    return TERRAIN[int(self.terrain[tile])]

  def capacity(self, tile: Tile) -> float:
    return CAPACITY[self.kind(tile)] * self.richness

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

  def best_near(self, tile: Tile) -> Tile:
    """The neighbouring tile with the most food (ties: the first)."""
    return max(self.around(tile), key=lambda t: (round(self.food[t], 6),))

  def around_food(self, tile: Tile) -> float:
    """The food within a day's walk (the tile and its neighbours)."""
    return float(sum(self.food[t] for t in self.around(tile)))

  def gather(self, tile: Tile, effort: float) -> float:
    """Forages up to `effort` food around a tile, richest tiles first."""
    wanted = max(0.0, effort)
    taken = 0.0
    for place in sorted(
        self.around(tile), key=lambda t: (-round(self.food[t], 6), t)
    ):
      if taken >= wanted:
        break
      got = float(min(self.food[place], wanted - taken))
      self.food[place] -= got
      taken += got
    return taken

  def regrow(self) -> None:
    capacity = self._capacities()
    self.food += (
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
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.rows, self.cols = int(state['rows']), int(state['cols'])
    self.richness = float(state.get('richness', 1.0))
    self.terrain = np.array(state['terrain'], dtype=int)
    self.food = np.array(state['food'], dtype=float)
