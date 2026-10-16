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

"""Gaia's backlash: pollution, pressure and the land striking back.

Like Factorio's biters. A people's industry (soot, coal smoke, slag)
pollutes the regions it lives in; pollution drifts to neighbours, disperses,
and is absorbed by plants, forests most of all. Pollution that lingers raises
the region's pressure; at the threshold the land strikes the people living
there with pests (a few poor harvests) or plague (deaths), then calms a
little. Forests are the defence, and the Steward can send cleansing rains.
"""

# pylint: disable=g-complex-comprehension

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from examples.deep_time.world import grid as grid_lib
import numpy as np


@dataclasses.dataclass(frozen=True)
class Attack:
  """The land striking a people."""

  polity: str
  kind: str
  region: int
  text: str
  harvest_loss: float = 0.0
  ticks: int = 0
  deaths: float = 0.0


class Backlash:
  """Regional pollution and pressure, and the attacks they bring."""

  def __init__(
      self,
      config: Mapping[str, Any],
      grid: grid_lib.RegionGrid,
      seed: int = 0,
  ):
    pollution = config['pollution']
    self.per = {
        'soot': float(pollution['per_soot']),
        'fossil_co2': float(pollution['per_fossil_co2']),
        'slag': float(pollution['per_slag']),
    }
    self.decay = float(pollution['decay'])
    self.spread = float(pollution['spread'])
    self.absorb_forest = float(pollution['absorb_forest'])
    self.absorb_grass = float(pollution['absorb_grass'])
    pressure = config['pressure']
    self.per_pollution = float(pressure['per_pollution'])
    self.relief = float(pressure['relief'])
    self.threshold = float(pressure['threshold'])
    self.after_attack = float(pressure['after_attack'])
    self.attacks = [dict(a) for a in config['attacks']]
    self._grid = grid
    self._links = [
        (r, n) for r in range(grid.size) for n in grid.neighbours(r) if r < n
    ]
    self.pollution = np.zeros(grid.size)
    self.pressure = np.zeros(grid.size)
    self.count = 0
    self._rng = np.random.default_rng(seed + 31337)

  def emit(
      self, regions: Sequence[int], emissions: Mapping[str, float]
  ) -> None:
    """Spreads a people's emissions over the regions it lives in."""
    amount = sum(self.per.get(k, 0.0) * v for k, v in emissions.items())
    if amount <= 0 or not regions:
      return
    self.pollution[list(regions)] += amount / len(regions)

  def cleanse(self, fraction: float) -> None:
    """Cleansing rains wash out pollution and calm the land everywhere."""
    keep = 1.0 - min(1.0, max(0.0, fraction))
    self.pollution *= keep
    self.pressure *= keep

  def step(
      self,
      forest: np.ndarray,
      grass: np.ndarray,
      homes: Mapping[str, Sequence[int]],
      names: Mapping[str, str],
  ) -> list[Attack]:
    """Drifts, absorbs and disperses pollution; returns this tick's attacks.

    Args:
      forest: Forest cover per region (share of the region).
      grass: Grass cover per region.
      homes: Polity key -> the regions it lives in.
      names: Polity key -> display name.

    Returns:
      Attacks, at most one per people.
    """
    drift = np.zeros_like(self.pollution)
    for r, n in self._links:
      flow = self.spread * 0.5 * (self.pollution[r] - self.pollution[n])
      drift[r] -= flow
      drift[n] += flow
    self.pollution = np.maximum(0.0, self.pollution + drift)
    share = np.clip(
        self.absorb_forest * forest + self.absorb_grass * grass, 0.0, 1.0
    )
    absorbed = self.pollution * share
    self.pollution = (self.pollution - absorbed) * (1.0 - self.decay)
    self.pressure = np.maximum(
        0.0, self.pressure + self.per_pollution * self.pollution - self.relief
    )
    attacks = []
    weights = np.array([float(a['weight']) for a in self.attacks])
    weights = weights / weights.sum()
    for key in sorted(homes):
      regions = list(homes[key])
      if not regions:
        continue
      worst = max(regions, key=lambda r: (self.pressure[r], -r))
      if self.pressure[worst] < self.threshold:
        continue
      spec = self.attacks[int(self._rng.choice(len(self.attacks), p=weights))]
      self.pressure[worst] = self.after_attack * self.threshold
      self.count += 1
      attacks.append(
          Attack(
              polity=key,
              kind=str(spec['kind']),
              region=worst,
              text=str(spec['text']).format(name=names[key]),
              harvest_loss=float(spec.get('harvest_loss', 0.0)),
              ticks=int(spec.get('ticks', 0)),
              deaths=float(spec.get('deaths', 0.0)),
          )
      )
    return attacks

  def restless(self, homes: Mapping[str, Sequence[int]]) -> dict[str, float]:
    """Each people's worst pressure, as a share of the threshold."""
    return {
        key: round(
            float(max((self.pressure[r] for r in regions), default=0.0))
            / self.threshold,
            3,
        )
        for key, regions in sorted(homes.items())
    }

  def get_state(self) -> dict[str, Any]:
    return {
        'pollution': self.pollution.tolist(),
        'pressure': self.pressure.tolist(),
        'count': self.count,
        'rng': self._rng.bit_generator.state,
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.pollution = np.array(state['pollution'], dtype=float)
    self.pressure = np.array(state['pressure'], dtype=float)
    self.count = int(state['count'])
    self._rng.bit_generator.state = dict(state['rng'])
