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

"""Seeded continents: which regions are land, sea or coast.

Scenarios describe geography either explicitly (`land: all`, a number, or 48
values) or with the generator (`generator: continents`), which grows a few
seeded continents until the requested land fraction is reached. Coastal
regions (next to both land and sea) are half land.
"""

from collections.abc import Mapping
from typing import Any

from examples.deep_time.world import grid as grid_lib
import numpy as np


def continents(
    grid: grid_lib.RegionGrid,
    land_fraction: float,  # pylint: disable=redefined-outer-name
    count: int,
    seed: int,
    *,
    width: float = 1.3,
    coast: float = 0.5,
) -> np.ndarray:
  """Grows `count` seeded continents to roughly `land_fraction` of the area.

  Args:
    grid: The planet grid.
    land_fraction: Target fraction of the planet's area that is land.
    count: Number of continents.
    seed: Random seed.
    width: Continent radius in grid cells.
    coast: Land fraction given to coastal regions.

  Returns:
    Land fraction per region.
  """
  rng = np.random.default_rng(seed)
  bands = np.repeat(np.arange(grid.bands), grid.sectors)
  sectors = np.tile(np.arange(grid.sectors), grid.bands)
  score = rng.normal(0.0, 0.15, grid.size)
  for _ in range(max(1, count)):
    band = rng.uniform(0.5, grid.bands - 1.5)
    sector = rng.uniform(0, grid.sectors)
    d_sector = np.abs(sectors - sector)
    d_sector = np.minimum(d_sector, grid.sectors - d_sector)
    distance2 = (bands - band) ** 2 + d_sector**2
    score += np.exp(-distance2 / (2 * width**2))
  weights = grid.area_weights()
  land = np.zeros(grid.size)
  for region in np.argsort(-score):
    if np.dot(weights, land) >= land_fraction:
      break
    land[region] = 1.0
  result = land.copy()
  for region in range(grid.size):
    if land[region] == 1.0 and any(
        land[n] == 0.0 for n in grid.neighbours(region)
    ):
      result[region] = coast
  return result


def islands(
    grid: grid_lib.RegionGrid,
    count: int,
    land: tuple[float, float],
    seed: int,
) -> np.ndarray:
  """A sea of islands: `count` regions of land, none touching another.

  Islands keep off the polar bands, and no two share a side or a corner, so
  at tile level too each is its own island. The first is the largest.

  Args:
    grid: The planet grid.
    count: How many islands (fewer if they do not fit).
    land: Range of each island region's land share.
    seed: Random seed.

  Returns:
    Land fraction per region.
  """
  rng = np.random.default_rng(seed)
  candidates = [
      r for r in range(grid.size) if 0 < grid.band_of(r) < grid.bands - 1
  ]
  rng.shuffle(candidates)
  chosen: list[int] = []
  for region in candidates:
    if len(chosen) >= count:
      break
    band, sector = grid.band_of(region), grid.sector_of(region)
    touching = any(
        abs(grid.band_of(o) - band) <= 1
        and min(
            (grid.sector_of(o) - sector) % grid.sectors,
            (sector - grid.sector_of(o)) % grid.sectors,
        )
        <= 1
        for o in chosen
    )
    if not touching:
      chosen.append(region)
  values = np.zeros(grid.size)
  low, high = land
  for i, region in enumerate(chosen):
    values[region] = high if i == 0 else float(rng.uniform(low, high))
  return values


def land_fraction(
    geography: Mapping[str, Any] | None,
    grid: grid_lib.RegionGrid,
    seed: int,
    config: Mapping[str, Any],
) -> np.ndarray:
  """Land fraction per region for a scenario's `geography` entry."""
  geography = dict(geography or {'land': 'all'})
  if geography.get('generator') == 'islands':
    low, high = geography.get('land', (0.45, 0.75))
    return islands(
        grid,
        int(geography.get('count', 8)),
        (float(low), float(high)),
        seed + int(geography.get('seed_offset', 0)),
    )
  if geography.get('generator') == 'continents':
    settings = config['continents']
    return continents(
        grid,
        float(geography.get('land', 0.3)),
        int(geography.get('continents', 3)),
        seed + int(geography.get('seed_offset', 0)),
        width=float(settings['blob_width_sectors']),
        coast=float(settings['coast_fraction']),
    )
  land = geography.get('land', 'all')
  if land == 'all':
    return np.ones(grid.size)
  if isinstance(land, (int, float)):
    return np.full(grid.size, float(land))
  values = np.asarray(land, dtype=float)
  if values.shape != (grid.size,):
    raise ValueError(f'geography.land needs {grid.size} values')
  return values
