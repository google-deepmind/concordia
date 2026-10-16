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

"""Local pollution on the tile map.

A people's soot (coal, steel, cement) settles as a haze over the tiles
around its towns, decaying from tick to tick, and damages the fields it
covers. Its fertiliser runoff washes into the shelf within a few tiles of
its coastal towns as a plume. The ocean's dead-zone share (`world/ocean.py`
sets how much) opens where the plumes are thickest, in the warmest water
first, so dead zones lie off the coasts of the peoples that farm hardest;
without plumes they open in the warmest shallows. A people whose own coast
is choked catches fewer fish, one with clean water more.
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield

from collections.abc import Mapping, Sequence
from typing import Any

from examples.deep_time.world import biomes as biomes_lib
from examples.deep_time.world import tiles as tiles_lib
import numpy as np


class Pollution:
  """Soot haze over land, runoff plumes and dead zones at sea."""

  def __init__(
      self,
      tile_map: tiles_lib.TileMap,
      biomes: biomes_lib.Biomes,
      config: Mapping[str, Any],
  ):
    self.tiles = tile_map
    self.biomes = biomes
    self.config = config
    shape = tile_map.land.shape
    self.soot = np.zeros(shape)
    self.plume = np.zeros(shape)
    self.dead = np.zeros(shape, dtype=bool)
    self.share = 0.0  # The dead-zone share `dead` was placed with.
    self.shelf = tile_map.terrain == tiles_lib.SHELF

  def _near(self, places: Sequence[tuple[int, int]], reach: int) -> np.ndarray:
    """Closeness (1 at a place, falling to 0 beyond `reach`) to any place."""
    rows, cols = self.tiles.rows, self.tiles.cols
    r = np.arange(rows)[:, None]
    c = np.arange(cols)[None, :]
    out = np.zeros((rows, cols))
    for pr, pc in places:
      dc = np.minimum((c - pc) % cols, (pc - c) % cols)
      steps = np.maximum(np.abs(r - pr), dc)
      out = np.maximum(out, np.clip(1.0 - steps / (reach + 1.0), 0.0, 1.0))
    return out

  def step(
      self,
      peoples: Mapping[
          str, tuple[Sequence[tuple[int, int]], Mapping[str, float]]
      ],
      dead_share: float,
  ) -> None:
    """Settles this tick's soot and runoff and places the dead zones.

    Args:
      peoples: Each people's settlement tiles and this tick's emissions.
      dead_share: The share of the coastal seas starved of oxygen.
    """
    c = self.config
    self.soot *= float(c['soot']['decay'])
    self.plume *= float(c['runoff']['decay'])
    land, sea = self.tiles.land, self.shelf
    for places, emissions in peoples.values():
      if not places:
        continue
      soot = float(emissions.get('soot', 0.0))
      if soot > 0:
        near = self._near(places, int(c['soot']['reach'])) * land
        if near.sum() > 0:
          self.soot += float(c['soot']['per_unit']) * soot * near / near.max()
      runoff = float(emissions.get('runoff', 0.0))
      if runoff > 0:
        near = self._near(places, int(c['runoff']['reach'])) * sea
        if near.sum() > 0:
          share = near / near.sum() * float(sea.sum())  # Mean 1 over shelf.
          self.plume += float(c['runoff']['per_unit']) * runoff * share / 10.0
    self.soot = np.clip(self.soot, 0.0, 1.0)
    self.share = max(0.0, min(1.0, float(dead_share)))
    count = int(round(self.share * int(sea.sum())))
    self.dead = np.zeros(sea.shape, dtype=bool)
    if count:
      # Thickest plume first; warmth breaks ties (and decides alone when
      # there is no plume yet).
      warmth = (self.biomes.temperature - self.biomes.temperature.min()) / (
          np.ptp(self.biomes.temperature) + 1e-9
      )
      rank = np.where(sea, self.plume + 1e-3 * warmth, -np.inf)
      order = np.argsort(-rank.ravel(), kind='stable')[:count]
      self.dead.ravel()[order] = True

  def crop_loss(self) -> np.ndarray:
    """Share of each tile's farmland the haze spoils."""
    return float(self.config['soot']['crop_damage']) * self.soot

  def fishing(self, places: Sequence[tuple[int, int]]) -> float:
    """A people's local fishing factor: its own water against the seas'.

    Both sides are the dead zones as last placed, so the factor only moves
    catch between peoples and never cancels the seas' own losses.
    """
    dead_share = self.share
    lo, hi = (float(x) for x in self.config['fish_range'])
    near = (self._near(places, int(self.config['runoff']['reach'])) > 0) & (
        self.shelf
    )
    if not near.any():
      return 1.0
    local = float(self.dead[near].mean())
    return float(np.clip((1.0 - local) / max(1e-9, 1.0 - dead_share), lo, hi))

  def payload(self) -> dict[str, Any]:
    """For the map (rows north first): haze and plume in tenths, dead zones."""

    def tenths(values: np.ndarray) -> list[str]:
      return [
          ''.join(str(min(9, int(v * 10))) for v in row) for row in values[::-1]
      ]

    return {
        'soot': tenths(self.soot),
        'plume': tenths(np.clip(self.plume, 0.0, 1.0)),
        'dead_zones': (
            [''.join('1' if d else '0' for d in row) for row in self.dead[::-1]]
            if self.dead.any()
            else None
        ),
    }

  def get_state(self) -> dict[str, Any]:
    return {
        'soot': self.soot.tolist(),
        'plume': self.plume.tolist(),
        'dead': self.dead.astype(int).tolist(),
        'share': self.share,
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.soot = np.array(state['soot'], dtype=float)
    self.plume = np.array(state['plume'], dtype=float)
    self.dead = np.array(state['dead'], dtype=bool)
    self.share = float(state.get('share', 0.0))
