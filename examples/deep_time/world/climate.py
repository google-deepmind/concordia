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

"""Regional energy-balance climate (Budyko–Sellers).

Each tick is long (tens of millions of years in the evolution era), so the
climate is solved to equilibrium for the current surface albedo: the linear
system `(B·I − D·K) T = Q·L·s·(1 − α) − A + F` with the grid's
energy-conserving diffusion operator `K`.
The ice–albedo feedback makes `α` depend on `T`, so the solve is iterated a
few times. Greenhouse forcing `F` (W m⁻²) is the hook for the carbon cycle.
"""

from collections.abc import Mapping
from typing import Any

from examples.deep_time.world import grid as grid_lib
import numpy as np


def insolation_profile(latitudes: np.ndarray, p2: float) -> np.ndarray:
  """Annual-mean insolation s(lat) = 1 − p2·P2(sin lat), mean ≈ 1."""
  x = np.sin(np.radians(latitudes))
  return 1.0 - p2 * (3.0 * x**2 - 1.0) / 2.0


class EnergyBalanceClimate:
  """Equilibrium regional temperatures with diffusion and sea/land ice."""

  def __init__(
      self,
      config: Mapping[str, Any],
      grid: grid_lib.RegionGrid,
      land_fraction: np.ndarray | None = None,
  ):
    """Initializes the solver.

    Args:
      config: Contents of `data/climate.yaml`.
      grid: The planet grid.
      land_fraction: Land per region; sea regions conduct extra heat along ocean
        currents. None means no currents.
    """
    self.grid = grid
    self.solar_flux = float(config['solar_flux'])
    self.olr_a = float(config['olr_a'])
    self.olr_b = float(config['olr_b'])
    self.diffusion = float(config['diffusion'])
    self.ocean_albedo = float(config['ocean_albedo'])
    ice = config['ice']
    self.freeze_c = float(ice['freeze_c'])
    self.ice_ramp_c = float(ice['ramp_c'])
    self.ice_albedo = float(ice['albedo'])
    self.ice_iterations = int(config['ice_iterations'])
    self.local_heating_c = float(config['local_heating_c'])
    self.insolation = insolation_profile(
        grid.latitudes(), float(config['insolation_p2'])
    )
    self.weights = grid.area_weights()
    sea = None if land_fraction is None else 1.0 - np.asarray(land_fraction)
    operator = grid.diffusion_operator(
        sea, float(config.get('ocean_current_boost', 0.0))
    )
    system = self.olr_b * np.eye(grid.size) - self.diffusion * operator
    self._inverse = np.linalg.inv(system)

  def ice_fraction(self, temps_c: np.ndarray) -> np.ndarray:
    """Fraction of open (non-flower) surface frozen at these temperatures."""
    return np.clip((self.freeze_c - temps_c) / self.ice_ramp_c, 0.0, 1.0)

  def surface_albedo(
      self,
      open_albedo: np.ndarray,
      flower_albedo: np.ndarray,
      open_fraction: np.ndarray,
      temps_c: np.ndarray,
  ) -> np.ndarray:
    """Region albedo: flowers plus open ground or sea, partly iced over."""
    ice = self.ice_fraction(temps_c)
    open_part = (1.0 - ice) * open_albedo + ice * self.ice_albedo
    return flower_albedo + open_fraction * open_part

  def equilibrium(
      self,
      luminosity: float,
      albedo_fn,
      forcing: float = 0.0,
      initial_c: np.ndarray | None = None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Solves for temperatures given a function from temperatures to albedo.

    Args:
      luminosity: Solar luminosity relative to the scenario's reference.
      albedo_fn: Maps regional temperatures (C) to regional albedo, so that ice
        can respond to temperature.
      forcing: Greenhouse forcing in W m^-2 (positive warms).
      initial_c: Starting guess for the ice iteration.

    Returns:
      (temperatures in C, albedo) per region.
    """
    temps = (
        np.zeros(self.grid.size) if initial_c is None else np.array(initial_c)
    )
    albedo = albedo_fn(temps)
    for _ in range(max(1, self.ice_iterations)):
      absorbed = self.solar_flux * luminosity * self.insolation * (1.0 - albedo)
      temps = self._inverse @ (absorbed - self.olr_a + forcing)
      albedo = albedo_fn(temps)
    return temps, albedo

  def global_mean(self, values: np.ndarray) -> float:
    return float(np.dot(self.weights, values))
