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

"""Oceans: sea level, sea ice, acidity and dead zones.

Sea level falls as ice piles up on land (and rises as it melts), and ocean pH
falls as the ocean absorbs carbon. Fertiliser runoff loads the coastal seas
with nutrients that the currents flush away slowly; the load starves the water
of oxygen (more so when the seas are warm) and opens dead zones where fish
cannot live. Heat transport by currents is part of the climate's diffusion
operator (`ocean_current_boost`).
"""

# pylint: disable=g-doc-return-or-yield

from collections.abc import Mapping
import math
from typing import Any

import numpy as np


class Ocean:
  """Derived ocean state, recomputed each tick from climate and carbon."""

  def __init__(self, config: Mapping[str, Any], reference_ocean_carbon: float):
    self.sea_level_per_land_ice = float(config['sea_level_per_land_ice'])
    self.ph_reference = float(config['ph_reference'])
    self.ph_per_e_fold = float(config['ph_per_e_fold_carbon'])
    self.acid_alert_ph = float(config['acid_alert_ph'])
    self.reference_carbon = float(reference_ocean_carbon)
    self.sea_level_m = 0.0
    self.ph = self.ph_reference
    self.sea_ice = 0.0
    self.mean_sst_c = 0.0
    dead = config.get('dead_zones', {})
    self.flushing = float(dead.get('flushing', 1.0))
    self.half_load = float(dead.get('half_load', 1.0))
    self.max_dead_zone = float(dead.get('max_share', 0.0))
    self.warm_reference_c = float(dead.get('warm_reference_c', 15.0))
    self.warm_per_c = float(dead.get('warm_per_c', 0.0))
    self.dead_zone_alert = float(dead.get('alert_share', 1.0))
    self.nutrients = 0.0  # Runoff load in the coastal seas.
    # How much the seas' life weighs in the planet's biodiversity.
    self.marine_weight = float(config.get('marine_life', {}).get('weight', 0.0))
    self.dead_zone = 0.0  # Share of the coastal seas too starved of oxygen.

  def feed(self, runoff: float) -> None:
    """Adds fertiliser runoff that reached the seas this tick."""
    self.nutrients += max(0.0, float(runoff))

  def dead_zone_for(self, nutrients: float, sst_c: float) -> float:
    """The dead-zone share a nutrient load makes at a sea temperature."""
    if nutrients <= 0:
      return 0.0
    warmth = min(
        2.0,
        max(0.5, 1.0 + self.warm_per_c * (sst_c - self.warm_reference_c)),
    )
    share = self.max_dead_zone * nutrients / (nutrients + self.half_load)
    return min(self.max_dead_zone, share * warmth)

  def update(
      self,
      *,
      weights: np.ndarray,
      land: np.ndarray,
      ice: np.ndarray,
      temps_c: np.ndarray,
      ocean_carbon: float | None,
  ) -> list[str]:
    """Recomputes the ocean state; returns alert texts (if any).

    Args:
      weights: Area weights of the regions.
      land: Land fraction per region.
      ice: Fraction of each region's open surface frozen.
      temps_c: Regional temperatures.
      ocean_carbon: Carbon in the ocean reservoir, if a carbon cycle runs.
    """
    sea = 1.0 - land
    land_ice = float(np.dot(weights, land * ice))
    self.sea_level_m = self.sea_level_per_land_ice * land_ice
    sea_area = float(np.dot(weights, sea))
    self.sea_ice = (
        float(np.dot(weights, sea * ice)) / sea_area if sea_area > 0 else 0.0
    )
    self.mean_sst_c = (
        float(np.dot(weights, sea * temps_c)) / sea_area
        if sea_area > 0
        else 0.0
    )
    previous = self.ph
    if ocean_carbon is not None and ocean_carbon > 0:
      self.ph = self.ph_reference + self.ph_per_e_fold * math.log(
          ocean_carbon / self.reference_carbon
      )
    alerts = []
    if self.ph < self.acid_alert_ph <= previous:
      alerts.append(
          f'The seas are souring: ocean pH has fallen to {self.ph:.2f}.'
      )
    was_dead = self.dead_zone
    self.nutrients *= 1.0 - self.flushing
    self.dead_zone = self.dead_zone_for(self.nutrients, self.mean_sst_c)
    if self.dead_zone >= self.dead_zone_alert > was_dead:
      alerts.append(
          f'Dead zones spread along the coasts: {self.dead_zone:.0%} of the'
          ' coastal seas hold too little oxygen for fish.'
      )
    return alerts

  def payload(self, sea_area: float) -> dict[str, Any]:
    return {
        'sea_area': round(sea_area, 3),
        'sea_level_m': round(self.sea_level_m, 1),
        'sea_ice': round(self.sea_ice, 3),
        'mean_sst_c': round(self.mean_sst_c, 1),
        'ph': round(self.ph, 3),
        'nutrients': round(self.nutrients, 2),
        'dead_zone': round(self.dead_zone, 3),
    }

  def get_state(self) -> dict[str, Any]:
    return {
        'sea_level_m': self.sea_level_m,
        'ph': self.ph,
        'sea_ice': self.sea_ice,
        'mean_sst_c': self.mean_sst_c,
        'reference_carbon': self.reference_carbon,
        'nutrients': self.nutrients,
        'dead_zone': self.dead_zone,
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.sea_level_m = float(state['sea_level_m'])
    self.ph = float(state['ph'])
    self.sea_ice = float(state['sea_ice'])
    self.mean_sst_c = float(state['mean_sst_c'])
    self.reference_carbon = float(state['reference_carbon'])
    self.nutrients = float(state.get('nutrients', 0.0))
    self.dead_zone = float(state.get('dead_zone', 0.0))
