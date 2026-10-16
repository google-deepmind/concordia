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

"""Sessile life on tiles: plants, mats and forests hold ground.

The daisy growth law runs on every tile: each species grows on bare ground
at a rate set by its flowers' temperature on that tile (the tile's
temperature plus its region's flower-warming offset), by the tile's biome
and by the region's other multipliers (behaviour, soil, air, genome, pacts,
ultraviolet), and dies at its usual rate. A share of the seeds a tile
receives comes from its four neighbours, so on uniform ground the law is the
regional one, and where cover differs, species spread across the map,
across region borders too, where the ground suits them. Biomes weigh growth
relative to the biomes each region started with, so a region grows as fast
as the regional model on day one and biomes decide where within it; when
biomes shift, growth shifts with them. Regions aggregate tiles: each
species' region cover is the mean of its tiles, which is what the energy
balance sees.

Everything that still changes cover per region between ticks (seeding,
grazers, speciation, seed banks, blights) keeps working: at every substep
the tiles reconcile with the region covers, adding the difference to
suitable bare tiles or scaling cover down.
"""

# pylint: disable=g-doc-args

from collections.abc import Mapping
from typing import Any

from examples.deep_time.world import biomes as biomes_lib
from examples.deep_time.world import daisyworld
from examples.deep_time.world import tiles as tiles_lib
import numpy as np


class TileLife:
  """Cover per species per tile, grown locally and aggregated to regions."""

  def __init__(
      self,
      tile_map: tiles_lib.TileMap,
      biomes: biomes_lib.Biomes,
      daisies: daisyworld.Daisyworld,
      config: Mapping[str, Any],
  ):
    self.tiles = tile_map
    self.biomes = biomes
    self.config = config
    self.spread = float(config['spread'])
    suitability = config['suitability']
    self._suit_by_biome = np.array(
        [float(suitability.get(b['key'], 0.0)) for b in biomes.specs]
    )
    self._land = tile_map.land.astype(float)
    self._cells = tile_map.per * tile_map.per
    self.cells = self._cells  # Tiles per region.
    # Suitability is relative to each region's starting biomes, so a region
    # grows as much as the calibrated regional model at the start, while
    # biomes still decide where within it, and shifting biomes still matter.
    start = self._suit_by_biome[biomes.biome] * self._land
    land_tiles = np.bincount(
        tile_map.region.ravel(),
        weights=self._land.ravel(),
        minlength=tile_map.grid.size,
    )
    mean = np.bincount(
        tile_map.region.ravel(),
        weights=start.ravel(),
        minlength=tile_map.grid.size,
    ) / np.maximum(land_tiles, 1e-9)
    self._region_norm = np.where(mean > 1e-9, 1.0 / np.maximum(mean, 1e-9), 1.0)
    self._min_cover = daisies.params.min_cover
    # Extra growth per tile (ash on burned ground, from world/fire.py).
    self.growth_boost: np.ndarray | None = None
    self.covers: dict[str, np.ndarray] = {}
    self.reconcile(daisies)
    # Land that plants have reached: only there does a seed bank wait (on
    # a world that starts green everywhere, that is all of it).
    total = np.sum(list(self.covers.values()), axis=0) if self.covers else 0.0
    self.reached = (np.asarray(total) > 0) & tile_map.land

  def region_mean(self, values: np.ndarray) -> np.ndarray:
    """Mean over each region's tiles (tile area units -> region area)."""
    total = np.bincount(
        self.tiles.region.ravel(),
        weights=values.ravel(),
        minlength=self.tiles.grid.size,
    )
    return total / self._cells

  def _bare(self) -> np.ndarray:
    covered = sum(self.covers.values()) if self.covers else 0.0
    return np.clip(self._land - covered, 0.0, None)

  def _suitability(
      self, daisies: daisyworld.Daisyworld, species: str
  ) -> np.ndarray:
    """Growth rate per tile, before bare ground and the region factors."""
    temps = self.biomes.tile_temperatures(daisies.temperatures_c())
    optimum, width = daisies.niche(species)
    local = temps + daisies.local_offset_c(species)[self.tiles.region]
    base = np.clip(1.0 - width * (optimum - local) ** 2, 0.0, None)
    suit = self._suit_by_biome[self.biomes.biome]
    return base * suit * self._region_norm[self.tiles.region] * self._land

  def reconcile(self, daisies: daisyworld.Daisyworld) -> None:
    """Brings the tiles back in line with each species' region cover.

    Every species shrinks first and only then do any grow, so a species can
    grow into room another just gave up, whatever their order. The new
    covers are swapped in at once (the web server reads them from its own
    thread).
    """
    region = self.tiles.region
    covers: dict[str, np.ndarray] = {}
    targets = {}
    for name in daisies.species():
      target = daisies.cover_map(name)
      current = self.covers.get(name)
      if current is None:
        current = np.zeros_like(self._land)
      have = self.region_mean(current)
      shrink = target < have - 1e-12
      if shrink.any():
        ratio = np.where(shrink, target / np.maximum(have, 1e-12), 1.0)
        current = current * ratio[region]
      covers[name] = current
      targets[name] = target
    for name, target in targets.items():
      current = covers[name]
      if not (target > self.region_mean(current) + 1e-12).any():
        continue
      self.covers = covers  # Suitability and bare ground read the new map.
      weight = self._bare() * (0.05 + self._suitability(daisies, name))
      for _ in range(3):  # Leftovers go round again where tiles filled up.
        need = np.clip(target - self.region_mean(current), 0.0, None)
        if not (need > 1e-12).any():
          break
        room = np.clip(self._land - sum(covers.values()), 0.0, None)
        w = np.where(room > 0, weight, 0.0)
        per_region = np.bincount(
            region.ravel(), weights=w.ravel(), minlength=self.tiles.grid.size
        )
        share = np.where(
            per_region[region] > 0,
            w / np.maximum(per_region[region], 1e-12),
            0.0,
        )
        added = np.minimum(room, share * need[region] * self._cells)
        current = current + added
        covers[name] = current
    self.covers = covers

  def grow(
      self, daisies: daisyworld.Daisyworld, dt: float
  ) -> dict[str, np.ndarray]:
    """One substep of local growth; returns each species' region cover."""
    self.reconcile(daisies)
    region = self.tiles.region
    bare = self._bare()
    floor = self._min_cover * self._land * self.reached
    new = {}
    for name, cover in self.covers.items():
      rate = (
          self._suitability(daisies, name)
          * daisies.growth_factors(name)[region]
      )
      if self.growth_boost is not None:
        rate = rate * self.growth_boost
      death = daisies.death_rate_of(name)
      neighbours = (
          np.roll(cover, 1, axis=1)
          + np.roll(cover, -1, axis=1)
          + np.vstack([cover[1:], cover[-1:]])
          + np.vstack([cover[:1], cover[:-1]])
      ) / 4.0
      # A share of the seeds that land on a tile come from its neighbours:
      # on uniform ground this is the regional law exactly, and where cover
      # differs, species spread outward (across region borders too).
      seeds = (1.0 - self.spread) * cover + self.spread * neighbours
      change = seeds * bare * rate - cover * death
      new[name] = np.maximum(floor, cover + dt * change)
    total = sum(new.values())
    over = total > self._land
    if np.any(over):
      scale = np.where(over, self._land / np.maximum(total, 1e-12), 1.0)
      new = {k: v * scale for k, v in new.items()}
    self.covers = new
    total = np.sum(list(new.values()), axis=0)
    self.reached = self.reached | (total > 2.0 * self._min_cover * self._land)
    return {k: self.region_mean(v) for k, v in new.items()}

  def dominant(
      self, covers: Mapping[str, np.ndarray] | None = None
  ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(species keys, index of the commonest per tile, total cover per tile)."""
    covers = self.covers if covers is None else covers
    keys = list(covers)
    if not keys:
      zero = np.zeros_like(self._land)
      return keys, zero.astype(int), zero
    stack = np.stack([covers[k] for k in keys])
    return keys, stack.argmax(axis=0), stack.sum(axis=0)

  def get_state(self) -> dict[str, Any]:
    # Full precision: later ticks depend on every digit.
    return {
        'covers': {k: v.tolist() for k, v in self.covers.items()},
        'reached': self.reached.astype(int).tolist(),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    covers = state['covers'] if 'covers' in state else state  # Older saves.
    self.covers = {k: np.array(v, dtype=float) for k, v in covers.items()}
    if 'reached' in state:
      self.reached = np.array(state['reached'], dtype=bool)
