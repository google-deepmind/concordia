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

"""The planet's regions: latitude bands × longitude sectors.

Region `r = band * sectors + sector`; band 0 is the far south. Neighbours are
east and west (wrapping round the planet) and north and south (no wrap at the
poles). The graph Laplacian drives heat diffusion, and later migration, trade
and stream latency.
"""

import dataclasses
import math

import numpy as np


@dataclasses.dataclass(frozen=True)
class RegionGrid:
  """Geometry of a bands × sectors planet."""

  bands: int = 6
  sectors: int = 8

  @property
  def size(self) -> int:
    return self.bands * self.sectors

  def region(self, band: int, sector: int) -> int:
    return band * self.sectors + sector % self.sectors

  def band_of(self, region: int) -> int:
    return region // self.sectors

  def sector_of(self, region: int) -> int:
    return region % self.sectors

  def latitudes(self) -> np.ndarray:
    """Centre latitude (degrees) of every region."""
    step = 180.0 / self.bands
    centres = -90.0 + step * (np.arange(self.bands) + 0.5)
    return np.repeat(centres, self.sectors)

  def band_latitudes(self) -> list[float]:
    step = 180.0 / self.bands
    return [-90.0 + step * (b + 0.5) for b in range(self.bands)]

  def area_weights(self) -> np.ndarray:
    """Fraction of the planet's surface in each region (sums to one)."""
    step = math.radians(180.0 / self.bands)
    lat = np.radians(self.latitudes())
    strip = np.sin(lat + step / 2) - np.sin(lat - step / 2)
    return strip / strip.sum()

  def neighbours(self, region: int) -> list[int]:
    band, sector = self.band_of(region), self.sector_of(region)
    result = [self.region(band, sector - 1), self.region(band, sector + 1)]
    if band > 0:
      result.append(self.region(band - 1, sector))
    if band < self.bands - 1:
      result.append(self.region(band + 1, sector))
    return result

  def diffusion_operator(
      self,
      sea_fraction: np.ndarray | None = None,
      current_boost: float = 0.0,
  ) -> np.ndarray:
    """Energy-conserving diffusion: (K @ T)[r] = sum_n k_rn (T_n - T_r) / A_r.

    Conductances k_rn = (A_r + A_n) / 2 use areas A normalised to mean one,
    so the area-weighted sum of K @ T is zero: diffusion moves heat between
    regions without creating or destroying it. Ocean currents multiply the
    conductance between two sea regions by `1 + current_boost * min(sea_r,
    sea_n)`.

    Args:
      sea_fraction: Sea fraction per region (no currents if None).
      current_boost: Extra conductance along ocean currents.

    Returns:
      A (size × size) matrix.
    """
    areas = self.area_weights() * self.size
    operator = np.zeros((self.size, self.size))
    for r in range(self.size):
      for n in self.neighbours(r):
        conductance = 0.5 * (areas[r] + areas[n])
        if sea_fraction is not None:
          conductance *= 1.0 + current_boost * min(
              sea_fraction[r], sea_fraction[n]
          )
        operator[r, n] += conductance / areas[r]
        operator[r, r] -= conductance / areas[r]
    return operator

  def band_name(self, band: int) -> str:
    lat = self.band_latitudes()[band]
    hemisphere = 'N' if lat > 0 else 'S'
    return f'{abs(lat):.0f}°{hemisphere}'

  def region_name(self, region: int) -> str:
    return f'{self.band_name(self.band_of(region))}/{self.sector_of(region)}'

  def band_means(self, values: np.ndarray) -> np.ndarray:
    """Area-weighted mean of a per-region array within each band."""
    weights = self.area_weights().reshape(self.bands, self.sectors)
    grid = np.asarray(values).reshape(self.bands, self.sectors)
    return (grid * weights).sum(axis=1) / weights.sum(axis=1)
