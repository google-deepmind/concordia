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

"""New Ararat on tiles: the Light Delay colony's own world.

The colony keeps one temperature, its sky's (`world/colony.py`). Here it is
spread over the same kind of tile map as home (`world/tiles.py`), warmer at
the equator and colder at the poles, with the same lapse to the heights and
the same climate layers and biomes (`world/biomes.py`): a white world at
landfall that thaws to tundra, and further, as the colonists warm it.

The colonists live in domes around their landing site, one more for every
few colonists. They mine ice where they choose: a person playing them (or
Mission Control, whose word arrives years late) may tap the ice field, and
the ice around that site against the ice around the landing scales the
mine's yield. Everything here follows from the colony's state each tick, so
the world needs nothing of its own in a checkpoint.
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import collections
from collections.abc import Mapping
from typing import Any

from examples.deep_time.world import biomes as biomes_lib
from examples.deep_time.world import colony as colony_lib
from examples.deep_time.world import geography
from examples.deep_time.world import grid as grid_lib
from examples.deep_time.world import tiles as tiles_lib
import numpy as np


class ColonyWorld:
  """New Ararat's tiles, climate layers, ice, domes and mining site."""

  def __init__(
      self,
      colony: colony_lib.Colony,
      config: Mapping[str, Any],
      climate_config: Mapping[str, Any],
      ocean_config: Mapping[str, Any],
      tiles_config: Mapping[str, Any],
      biomes_config: Mapping[str, Any],
      seed: int,
  ):
    self.colony = colony
    self.config = config
    self.grid = grid_lib.RegionGrid(**climate_config['grid'])
    land = geography.land_fraction(
        config['geography'], self.grid, seed, ocean_config
    )
    world_seed = seed + int(config['geography'].get('seed_offset', 0))
    self.tiles = tiles_lib.TileMap(self.grid, land, tiles_config, world_seed)
    self.biomes = biomes_lib.Biomes(self.tiles, biomes_config)
    lat = np.radians(self.grid.latitudes())
    weights = self.grid.area_weights()
    shape = np.cos(lat)
    self._offset = float(config['gradient_c']) * (
        shape - float(np.dot(weights, shape) / weights.sum())
    )
    # The map as it was each of the last few generations (newest last), so
    # home can see the colony as its light shows it: years late.
    self.frames: collections.deque[dict[str, Any]] = collections.deque(
        maxlen=colony.light_delay + 1
    )
    self.landing: tuple[int, int] | None = None
    self.update()
    # The landing: the warmest low ground by the sea.
    warm = np.where(
        (self.tiles.terrain == tiles_lib.COAST)
        | (self.tiles.terrain == tiles_lib.LOWLAND),
        self.biomes.temperature,
        -np.inf,
    )
    row, col = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
        int(np.argmax(warm)), warm.shape
    )
    self.landing = (int(row), int(col))
    self._snapshot()

  def region_temps(self) -> np.ndarray:
    """Each region's temperature: the colony's sky, warmer at the equator."""
    return self.colony.temp_c + self._offset

  def update(self) -> list[str]:
    """Ice and biomes for the colony's current sky; news of big thaws."""
    temps = self.region_temps()
    freeze, ramp = float(self.config['freeze_c']), float(self.config['ramp_c'])
    self.tiles.update_ice(np.clip((freeze - temps) / ramp, 0.0, 1.0))
    news = [
        f'On {self.colony.name}: {text}' for text in self.biomes.update(temps)
    ]
    if self.landing is not None:
      self._snapshot()
    return news

  def _snapshot(self) -> None:
    """Keeps this generation's map for those who see it years late."""
    self.frames.append({
        'ice': self.tiles.ice_rows(),
        'biomes': self.biomes.rows(),
        'domes': self.domes(),
        'generation': self.colony.generation,
    })

  def ice_near(self, site: tuple[int, int]) -> float:
    """The share of ice within the mine's reach of a tile."""
    reach = int(self.config['mine_reach'])
    rows, cols = self.tiles.rows, self.tiles.cols
    r, c = site
    rr = slice(max(0, r - reach), min(rows, r + reach + 1))
    cc = [(c + d) % cols for d in range(-reach, reach + 1)]
    return float(self.tiles.ice[rr][:, cc].mean())

  def mine_factor(self, site: tuple[int, int] | None = None) -> float:
    """The ice mine's yield at a site (default: the colony's), against the

    landing's.
    """
    site = self.colony.site if site is None else site
    if site is None or self.landing is None:
      return 1.0
    lo, hi = (float(x) for x in self.config['mine_factor'])
    base = self.ice_near(self.landing)
    here = self.ice_near(site)
    if base <= 1e-9:
      return hi if here > 0 else 1.0
    return float(np.clip(here / base, lo, hi))

  def domes(self) -> list[tuple[int, int]]:
    """Where the domes stand: the landing, then rings around it."""
    count = int(
        np.clip(
            1
            + self.colony.population
            // float(self.config['colonists_per_dome']),
            1,
            int(self.config['max_domes']),
        )
    )
    assert self.landing is not None
    r0, c0 = self.landing
    rows, cols = self.tiles.rows, self.tiles.cols
    spots = [(r0, c0)]
    ring = 2  # Two tiles apart, so each dome has its own ground.
    while len(spots) < count and ring < 9:
      for dr, dc in (
          (-ring, 0),
          (0, ring),
          (ring, 0),
          (0, -ring),
          (-ring, ring),
          (ring, ring),
          (ring, -ring),
          (-ring, -ring),
      ):
        r, c = r0 + dr, (c0 + dc) % cols
        if 0 <= r < rows and self.tiles.land[r, c] and len(spots) < count:
          spots.append((r, c))
      ring += 2
    return spots

  def get_state(self) -> dict[str, Any]:
    return {'biomes': self.biomes.get_state()}

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.update()  # The layers for the restored sky, then what was announced.
    self.biomes.set_state(state['biomes'])
