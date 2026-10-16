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

"""The tile map: 1,200 tiles that refine the 48 regions.

Each region splits into 5 × 5 tiles. A region's land tiles are the ones
scoring highest on a smooth field (the regions' land shares, upsampled with
the sectors wrapping round the planet) plus seeded fractal noise, and there
are exactly `round(25 × land share)` of them, so coastlines run across region
borders while every region keeps the land share the calibrated physics uses.
Elevation rises inland and with noise ridges. Water next to land is shelf,
land next to water is coast. Ice is a dynamic layer: each region's frozen
share goes to its coldest, highest tiles.
"""

from collections.abc import Mapping
from typing import Any

from examples.deep_time.world import grid as grid_lib
import numpy as np

DEEP_OCEAN, SHELF, COAST, LOWLAND, HIGHLAND = range(5)
CODES = 'dsclh'  # One character per class, for compact payloads.


def value_noise(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    base_cells: int,
    octaves: int,
    persistence: float,
) -> np.ndarray:
  """Fractal value noise in [0, 1), periodic across columns (longitude)."""
  total = np.zeros((rows, cols))
  amplitude, norm = 1.0, 0.0
  for octave in range(octaves):
    cells = base_cells * 2**octave
    lattice_rows = max(2, int(round(cells * rows / cols)) + 1)
    lattice = rng.random((lattice_rows, cells))
    y = np.linspace(0, lattice_rows - 1, rows)
    x = np.arange(cols) * cells / cols
    y0 = np.floor(y).astype(int).clip(0, lattice_rows - 2)
    x0 = np.floor(x).astype(int) % cells
    x1 = (x0 + 1) % cells
    fy = (y - y0)[:, None]
    fx = (x - np.floor(x))[None, :]
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)  # Smoothstep.
    top = lattice[y0][:, x0] * (1 - fx) + lattice[y0][:, x1] * fx
    bottom = lattice[y0 + 1][:, x0] * (1 - fx) + lattice[y0 + 1][:, x1] * fx
    total += amplitude * (top * (1 - fy) + bottom * fy)
    norm += amplitude
    amplitude *= persistence
  return total / norm


class TileMap:
  """Terrain classes, elevation and ice for every tile of a region grid."""

  def __init__(
      self,
      grid: grid_lib.RegionGrid,
      land: np.ndarray,
      config: Mapping[str, Any],
      seed: int = 0,
  ):
    """Builds the terrain.

    Args:
      grid: The region grid the tiles refine.
      land: Land share per region (the calibrated geography).
      config: Contents of `data/tiles.yaml`.
      seed: Seed of the terrain (with the scenario's geography seed).
    """
    self.grid = grid
    self.per = int(config['per_region'])
    self.rows = grid.bands * self.per
    self.cols = grid.sectors * self.per
    self.seed = int(seed)
    rng = np.random.default_rng(self.seed + 2020)
    band = np.arange(self.rows) // self.per
    sector = np.arange(self.cols) // self.per
    # Tile (row 0 = south) -> region index.
    self.region = (band[:, None] * grid.sectors + sector[None, :]).astype(int)
    noise_cfg = config['noise']
    noise = value_noise(
        self.rows,
        self.cols,
        rng,
        int(noise_cfg['base_cells']),
        int(noise_cfg['octaves']),
        float(noise_cfg['persistence']),
    )
    field = self._upsample(np.asarray(land, dtype=float))
    score = field + float(config['coast_noise']) * (noise - 0.5)
    self.land = np.zeros((self.rows, self.cols), dtype=bool)
    cells = self.per * self.per
    for r in range(grid.size):
      mask = self.region == r
      count = int(round(cells * float(land[r])))
      if count:
        order = np.argsort(-score[mask], kind='stable')
        chosen = np.zeros(cells, dtype=bool)
        chosen[order[:count]] = True
        self.land[mask] = chosen
    elevation_cfg = config['elevation']
    inland = np.minimum(
        self._distance_to(~self.land),
        float(elevation_cfg.get('max_inland_tiles', self.rows)),
    )
    relief = value_noise(
        self.rows, self.cols, rng, int(noise_cfg['base_cells']) * 2, 3, 0.55
    )
    # A world with no sea has no 'inland': only relief, mostly lowland.
    rise = (
        float(elevation_cfg['inland_per_tile']) if (~self.land).any() else 0.0
    )
    land_height = rise * (inland - 1) + float(elevation_cfg['relief']) * (
        relief - 0.3
    )
    near_land = self._near(self.land)
    sea_depth = np.where(
        near_land,
        float(elevation_cfg['shelf_depth']),
        float(elevation_cfg['deep_ocean']) * (0.6 + 0.4 * relief),
    )
    self.elevation = np.where(
        self.land, np.maximum(5.0, land_height), sea_depth
    ).round()
    near_sea = self._near(~self.land)
    highland = float(elevation_cfg['highland'])
    self.terrain = np.select(
        [
            ~self.land & ~near_land,
            ~self.land,
            self.land & near_sea,
            self.land & (self.elevation < highland),
        ],
        [DEEP_OCEAN, SHELF, COAST, LOWLAND],
        HIGHLAND,
    ).astype(int)
    self.ice = np.zeros((self.rows, self.cols), dtype=bool)

  def _upsample(self, values: np.ndarray) -> np.ndarray:
    """Bilinear field from region centres to tiles (wrapping in longitude)."""
    regions = values.reshape(self.grid.bands, self.grid.sectors)
    y = (np.arange(self.rows) + 0.5) / self.per - 0.5
    x = (np.arange(self.cols) + 0.5) / self.per - 0.5
    y0 = np.floor(y).astype(int)
    fy = (y - y0)[:, None]
    y1 = np.clip(y0 + 1, 0, self.grid.bands - 1)
    y0 = np.clip(y0, 0, self.grid.bands - 1)
    x0 = np.floor(x).astype(int)
    fx = (x - x0)[None, :]
    x1 = (x0 + 1) % self.grid.sectors
    x0 = x0 % self.grid.sectors
    top = regions[y0][:, x0] * (1 - fx) + regions[y0][:, x1] * fx
    bottom = regions[y1][:, x0] * (1 - fx) + regions[y1][:, x1] * fx
    return top * (1 - fy) + bottom * fy

  def _near(self, mask: np.ndarray) -> np.ndarray:
    """Tiles with a neighbour (8 around, wrapping in longitude) in `mask`."""
    out = np.zeros_like(mask)
    for dy in (-1, 0, 1):
      for dx in (-1, 0, 1):
        if dy or dx:
          shifted = np.roll(mask, dx, axis=1)
          if dy > 0:
            shifted = np.vstack([np.zeros((1, self.cols), bool), shifted[:-1]])
          elif dy < 0:
            shifted = np.vstack([shifted[1:], np.zeros((1, self.cols), bool)])
          out |= shifted
    return out

  def _distance_to(self, mask: np.ndarray) -> np.ndarray:
    """Steps (8-neighbour) from each tile to the nearest tile in `mask`."""
    distance = np.where(mask, 0, self.rows + self.cols).astype(float)
    frontier = mask.copy()
    step = 0
    while frontier.any() and step < self.rows + self.cols:
      step += 1
      grown = self._near(frontier) & (distance > step)
      distance[grown] = step
      frontier = grown
    return distance

  @property
  def size(self) -> int:
    return self.rows * self.cols

  def region_land(self) -> np.ndarray:
    """Land share per region, aggregated from the tiles."""
    shares = np.zeros(self.grid.size)
    np.add.at(shares, self.region.ravel(), self.land.ravel().astype(float))
    return shares / (self.per * self.per)

  def update_ice(self, region_ice: np.ndarray) -> None:
    """Freezes each region's share of tiles: highest and most poleward first."""
    cells = self.per * self.per
    lat = np.abs(np.arange(self.rows) + 0.5 - self.rows / 2)[:, None]
    coldness = self.elevation / 1000.0 + lat + 0.01 * self.terrain
    ice = np.zeros_like(self.ice)
    for r in range(self.grid.size):
      count = int(round(cells * float(region_ice[r])))
      if count:
        mask = self.region == r
        order = np.argsort(-coldness[mask], kind='stable')
        chosen = np.zeros(cells, dtype=bool)
        chosen[order[:count]] = True
        ice[mask] = chosen
    self.ice = ice

  def payload(self) -> dict[str, Any]:
    """Compact terrain for the map: rows of class codes, north first."""
    north = slice(None, None, -1)
    return {
        'rows': self.rows,
        'cols': self.cols,
        'per_region': self.per,
        'terrain': [
            ''.join(CODES[c] for c in row) for row in self.terrain[north]
        ],
        'elevation': [[int(e) for e in row] for row in self.elevation[north]],
    }

  def heart(self, mask: np.ndarray) -> tuple[int, int]:
    """The tile of a mask with most of the mask around it (wrapping east-west)."""
    rows, dense = self.rows, np.zeros(mask.shape)
    for dy in (-1, 0, 1):
      moved = np.zeros(mask.shape)
      if dy > 0:
        moved[dy:] = mask[: rows - dy]
      elif dy < 0:
        moved[: rows + dy] = mask[-dy:]
      else:
        moved = mask.astype(float)
      for dx in (-1, 0, 1):
        dense += np.roll(moved, dx, axis=1)
    flat = int(np.argmax(dense * mask))
    return flat // self.cols, flat % self.cols

  def ice_rows(self) -> list[str]:
    """Ice per tile ('1' frozen), north first."""
    return [''.join('1' if i else '0' for i in row) for row in self.ice[::-1]]

  def get_state(self) -> dict[str, Any]:
    return {'ice': self.ice_rows()}

  def set_state(self, state: Mapping[str, Any]) -> None:
    rows = state.get('ice')
    if rows:
      self.ice = np.array([[c == '1' for c in row] for row in rows])[::-1]
