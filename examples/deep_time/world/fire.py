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

"""Wildfire on the tile map.

A tile's fire risk is its dryness (warm and short of rain, from the climate
layers) times its fuel (flammable cover: forests, grass). Each tick
lightning strikes at random, more often where the risk is high; a strike on
risky ground starts a fire, and the fire front steps to the four neighbours,
catching each with a chance that grows with its own risk, so fires run
through dry country and stop at rivers of rainforest, deserts and the sea.
A burning tile loses part of its flammable cover; the carbon goes to the
air, and ash speeds regrowth there for a few ticks. Draws are seeded per
tick, so a world burns the same way every time it is replayed.
"""

# pylint: disable=g-long-ternary

from collections.abc import Mapping
from typing import Any

from examples.deep_time.world import biomes as biomes_lib
from examples.deep_time.world import tiles as tiles_lib
import numpy as np

# Burn scars: ticks they stay on the map (this tick's fires are the newest).
SCAR_TICKS = 3


class Wildfire:
  """Fire risk, fires and burn scars per tile."""

  def __init__(
      self,
      tile_map: tiles_lib.TileMap,
      biomes: biomes_lib.Biomes,
      config: Mapping[str, Any],
      seed: int,
  ):
    self.tiles = tile_map
    self.biomes = biomes
    self.config = config
    self.seed = int(seed)
    self.fuel_weights = {k: float(v) for k, v in config['fuel'].items()}
    shape = tile_map.land.shape
    self.scar = np.zeros(shape, dtype=int)  # SCAR_TICKS on the tick it burns.
    self.ash = np.zeros(shape)
    self.burned_last = np.zeros(shape, dtype=bool)
    # Ground soaked by summoned rain (0-1), drying out tick by tick.
    self.wet = np.zeros(shape)

  def flammability(self, species: str) -> float:
    """How readily a species burns (a daughter burns like its parent)."""
    return self.fuel_weights.get(
        species, self.fuel_weights.get(species.rstrip('0123456789'), 0.0)
    )

  def fuel(self, covers: Mapping[str, np.ndarray]) -> np.ndarray:
    """Flammable cover per tile, 0 (none) to 1 (fully fuelled)."""
    total = np.zeros(self.tiles.land.shape)
    for species, cover in covers.items():
      total += self.flammability(species) * cover
    return np.clip(total / float(self.config['fuel_full']), 0.0, 1.0)

  def dryness(self) -> np.ndarray:
    """0 (wet or cold) to 1 (hot and rainless), on ice-free land only."""
    heat = self.config['heat']
    warm = np.clip(
        (self.biomes.temperature - float(heat['from_c']))
        / float(heat['span_c']),
        0.0,
        1.0,
    )
    dry = np.clip(
        1.0 - self.biomes.rainfall / float(self.config['wet_mm']), 0.0, 1.0
    )
    soaked = 1.0 - np.clip(self.wet, 0.0, 1.0)
    return warm * dry * soaked * (self.tiles.land & ~self.tiles.ice)

  def soak(self, amount: float, near: np.ndarray | None = None) -> None:
    """Rain: wets the ground (closeness 0-1 per tile, or everywhere)."""
    spread = np.ones_like(self.wet) if near is None else near
    self.wet = np.clip(np.maximum(self.wet, amount * spread), 0.0, 1.0)

  def risk(self, covers: Mapping[str, np.ndarray]) -> np.ndarray:
    return self.dryness() * self.fuel(covers)

  def _neighbours(self, mask: np.ndarray) -> np.ndarray:
    """Tiles beside (4 around, wrapping in longitude) a tile in `mask`."""
    rows = self.tiles.rows
    out = np.roll(mask, 1, axis=1) | np.roll(mask, -1, axis=1)
    out[1:] |= mask[:-1]
    out[: rows - 1] |= mask[1:]
    return out

  def spread(self, tick: int, risk: np.ndarray) -> np.ndarray:
    """This tick's burned tiles: strikes, then fronts stepping outward."""
    c = self.config
    rng = np.random.default_rng([self.seed, 6061, int(tick)])
    total = float(risk.sum())
    strikes = rng.poisson(float(c['ignition_rate']) * total) if total > 0 else 0
    burning = np.zeros(risk.shape, dtype=bool)
    if strikes:
      flat = risk.ravel() / total
      hit = rng.choice(flat.size, size=strikes, p=flat)
      burning.ravel()[hit] = True
      burning &= risk >= float(c['min_risk'])
    front = burning.copy()
    # Soaked ground stops a front even where a dry one would always cross.
    chance = np.clip(float(c['spread']) * risk, 0.0, float(c['max_spread']))
    chance = chance * (1.0 - np.clip(self.wet, 0.0, 1.0))
    for _ in range(int(c['waves'])):
      if not front.any():
        break
      reach = self._neighbours(front) & ~burning
      front = reach & (rng.random(risk.shape) < chance)
      burning |= front
    return burning

  def step(
      self,
      tick: int,
      covers: Mapping[str, np.ndarray],
      oxygen_pal: float | None = None,
  ) -> np.ndarray:
    """Ages the scars and ash, then burns; returns this tick's burned tiles."""
    self.scar = np.maximum(0, self.scar - 1)
    self.ash *= float(self.config['ash']['decay'])
    if oxygen_pal is not None and oxygen_pal < float(
        self.config['min_oxygen_pal']
    ):
      burned = np.zeros(self.tiles.land.shape, dtype=bool)
    else:
      burned = self.spread(tick, self.risk(covers))
    self.scar[burned] = SCAR_TICKS
    self.ash[burned] = float(self.config['ash']['boost'])
    self.burned_last = burned
    # Soaked ground dries after the season's fires, not before them.
    self.wet = self.wet * float(self.config.get('wet_decay', 0.6))
    return burned

  def losses(
      self,
      covers: Mapping[str, np.ndarray],
      burned: np.ndarray,
      floor: np.ndarray,
  ) -> dict[str, np.ndarray]:
    """Cover each species loses per tile to this tick's fires."""
    fraction = float(self.config['burn_fraction'])
    lost = {}
    for species, cover in covers.items():
      weight = self.flammability(species)
      if weight <= 0 or not burned.any():
        continue
      loss = np.where(burned, cover * fraction * min(1.0, weight), 0.0)
      lost[species] = np.clip(loss, 0.0, np.maximum(0.0, cover - floor))
    return lost

  def fires(self, burned: np.ndarray) -> list[np.ndarray]:
    """Connected burned patches (4 around, wrapping), largest first."""
    left = burned.copy()
    patches = []
    while left.any():
      seed = np.zeros_like(left)
      seed.ravel()[int(np.flatnonzero(left)[0])] = True
      patch = seed
      while True:
        grown = (patch | self._neighbours(patch)) & burned
        if (grown == patch).all():
          break
        patch = grown
      patches.append(patch)
      left &= ~patch
    return sorted(patches, key=lambda p: -int(p.sum()))

  def describe(self, patch: np.ndarray) -> str:
    """News of one fire: its size, the country it ran through and where."""
    biomes = self.biomes
    counts = np.bincount(biomes.biome[patch], minlength=len(biomes.specs))
    country = biomes.specs[int(counts.argmax())]['name']
    lat = float(np.mean(biomes.lat[np.nonzero(patch)[0]]))
    where = (
        'in the far north'
        if lat > 50
        else 'in the north'
        if lat > 20
        else 'in the tropics'
        if lat > -20
        else 'in the south'
        if lat > -50
        else 'in the far south'
    )
    return f'Wildfire sweeps {int(patch.sum())} tiles of {country} {where}.'

  def get_state(self) -> dict[str, Any]:
    return {
        'scar': self.scar.tolist(),
        'ash': self.ash.tolist(),  # Full precision: regrowth reads it.
        'wet': self.wet.tolist(),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.scar = np.array(state['scar'], dtype=int)
    self.ash = np.array(state['ash'], dtype=float)
    self.wet = np.array(state.get('wet', np.zeros_like(self.ash)), dtype=float)
