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

"""Climate layers and biomes on the tile map.

The energy balance stays at region level. Each tile's temperature is its
region's (interpolated between region centres) less a lapse rate with
elevation, measured from the mean height of the region's land, so a
region's tiles average to the temperature the energy balance computed:
highlands are colder and lowlands warmer than the region, not all of it
colder. Rainfall starts from latitude belts (wet tropics, dry subtropics,
wet storm tracks, dry poles), falls off inland, rises on windward slopes and
drops in the rain shadow of highlands (the winds blow by band: trades,
westerlies, polar easterlies), and grows with warmth. Seasonality is an
annual temperature range: larger poleward and inland. A Whittaker-style
lookup turns them into biomes, recomputed every tick, and big changes are
news.
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield,missing-function-docstring

from collections.abc import Mapping
from typing import Any

from examples.deep_time.world import news as news_lib
from examples.deep_time.world import tiles as tiles_lib
import numpy as np


class Biomes:
  """Per-tile temperature, rainfall, seasonality and biome."""

  def __init__(self, tile_map: tiles_lib.TileMap, config: Mapping[str, Any]):
    self.tiles = tile_map
    self.config = config
    self.specs = list(config['biomes'])
    self.index = {b['key']: i for i, b in enumerate(self.specs)}
    self.codes = ''.join(b['code'] for b in self.specs)
    rows, cols = tile_map.rows, tile_map.cols
    # Height above the mean of each region's land (zero at sea).
    height = np.maximum(0.0, tile_map.elevation) * tile_map.land
    region = tile_map.region.ravel()
    land_tiles = np.bincount(
        region,
        weights=tile_map.land.ravel().astype(float),
        minlength=tile_map.grid.size,
    )
    mean_height = np.bincount(
        region, weights=height.ravel(), minlength=tile_map.grid.size
    ) / np.maximum(land_tiles, 1.0)
    self._relief = (height - mean_height[tile_map.region]) * tile_map.land
    # Latitude of each tile row (row 0 = south), degrees.
    self.lat = -90.0 + 180.0 * (np.arange(rows) + 0.5) / rows
    rain = config['rain']
    self._sea_distance = tile_map._distance_to(~tile_map.land)  # pylint: disable=protected-access
    belts = np.array(rain['belts'], dtype=float)
    # Each tile reads its belt at a jittered latitude, so belt edges meander.
    jitter = tiles_lib.value_noise(
        rows, cols, np.random.default_rng(tile_map.seed + 4040), 5, 3, 0.5
    )
    wobble = float(rain.get('belt_wobble_deg', 0.0)) * 2.0 * (jitter - 0.5)
    lat_read = np.clip(np.abs(self.lat)[:, None] + wobble, 0.0, 89.9)
    base = belts[np.searchsorted(belts[:, 0], lat_read), 1]  # By belt.
    continental = float(rain['continental_floor']) + (
        1.0 - float(rain['continental_floor'])
    ) * np.exp(-self._sea_distance / float(rain['continental_tiles']))
    if tile_map.land.all():  # No sea: moisture is recycled over the land.
      continental = np.ones_like(continental)
    winds = np.array(rain['winds'], dtype=float)
    direction = winds[np.searchsorted(winds[:, 0], np.abs(self.lat)), 1]
    elevation = np.maximum(0.0, tile_map.elevation)
    upwind = np.stack(
        [np.roll(elevation[r], int(direction[r])) for r in range(rows)]
    )
    rise = elevation - upwind  # Positive where the land climbs along the wind.
    shadow = np.zeros((rows, cols), dtype=bool)
    high = tile_map.terrain == tiles_lib.HIGHLAND
    for r in range(rows):
      d = int(direction[r])
      for step in (1, 2):
        shadow[r] |= np.roll(high[r], d * step) & ~high[r]
    orographic = 1.0 + float(rain['windward']) * np.maximum(0.0, rise)
    orographic = np.where(shadow, float(rain['rain_shadow']), orographic)
    wobble = tiles_lib.value_noise(
        rows, cols, np.random.default_rng(tile_map.seed + 3030), 6, 3, 0.5
    )
    variation = 1.0 + float(rain.get('noise', 0.0)) * (wobble - 0.5)
    self._dry_rain = base * continental * orographic * variation
    season = config['seasonality']
    self.seasonality = (
        float(season['base'])
        + float(season['per_degree_latitude'])
        * np.abs(self.lat)[:, None]
        * (
            1.0
            + float(season['continental'])
            * (1.0 - np.exp(-self._sea_distance / 4.0))
        )
    ) * np.where(tile_map.land, 1.0, 0.4)
    self.temperature = np.zeros((rows, cols))
    self.rainfall = np.zeros((rows, cols))
    self.biome = np.zeros((rows, cols), dtype=int)
    self._announced: dict[str, int] | None = None

  def tile_temperatures(self, region_temps: np.ndarray) -> np.ndarray:
    """Region temperatures, interpolated between centres, plus relief."""
    field = self.tiles._upsample(np.asarray(region_temps, dtype=float))  # pylint: disable=protected-access
    lapse = float(self.config['lapse_c_per_km'])
    return field - lapse * self._relief / 1000.0

  def update(self, region_temps: np.ndarray) -> list[str]:
    """Recomputes the layers and biomes; returns news of big changes."""
    t = self.tile_temperatures(region_temps)
    warmth = float(self.config['rain']['warmth_per_c'])
    self.temperature = t
    self.rainfall = self._dry_rain * np.clip(
        1.0 + warmth * (t - 15.0), 0.3, 2.0
    )
    before = getattr(self, 'biome', None)
    self.biome = self._classify(t, self.rainfall)
    return self._news(before)

  def _classify(self, t: np.ndarray, rain: np.ndarray) -> np.ndarray:
    th = self.config['thresholds']
    i = self.index
    terrain, ice = self.tiles.terrain, self.tiles.ice
    sea = ~self.tiles.land
    temperate = np.select(
        [
            rain < float(th['desert_below_mm']),
            rain < float(th['grassland_below_mm']),
        ],
        [i['desert'], i['grassland']],
        i['temperate_forest'],
    )
    tropical = np.select(
        [
            rain < float(th['desert_below_mm']),
            rain < float(th['savanna_below_mm']),
        ],
        [i['desert'], i['savanna']],
        i['rainforest'],
    )
    land = np.select(
        [
            ice,
            t < float(th['tundra_below_c']),
            t < float(th['taiga_below_c']),
            (rain >= float(th['wetland_from_mm']))
            & (self.tiles.elevation < float(th['wetland_below_m']))
            & (terrain != tiles_lib.HIGHLAND)
            & (t >= float(th['taiga_below_c'])),
            t < float(th['tropical_from_c']),
        ],
        [i['ice'], i['tundra'], i['taiga'], i['wetland'], temperate],
        tropical,
    )
    water = np.select(
        [
            ice,
            (terrain == tiles_lib.SHELF) & (t >= float(th['reef_from_c'])),
            terrain == tiles_lib.SHELF,
        ],
        [i['sea_ice'], i['reef'], i['shelf']],
        i['deep_ocean'],
    )
    return np.where(sea, water, land).astype(int)

  def counts(self) -> dict[str, int]:
    found = np.bincount(self.biome.ravel(), minlength=len(self.specs))
    return {b['key']: int(found[k]) for k, b in enumerate(self.specs)}

  def _news(self, previous: np.ndarray | None) -> list[str]:
    """News of biomes that grew or shrank far since they were last news.

    Each is placed (`news.located`) at the heart of where that biome came or
    went this tick, so it ripples over the map from there.
    """
    now = self.counts()
    if self._announced is None:
      self._announced = now
      return []
    step = int(self.config['announce_change_tiles'])
    news = []
    for key, count in now.items():
      before = self._announced.get(key, 0)
      if abs(count - before) >= step:
        name = self.specs[self.index[key]]['name']
        verb = 'spreads' if count > before else 'retreats'
        size = self.tiles.size
        if not before:
          text = f'The {name} appears on {count} of the {size} tiles.'
        elif not count:
          text = (
              f'The {name} is gone: none of the {size} tiles (it held'
              f' {before}).'
          )
        else:
          text = (
              f'The {name} {verb} from {before} to {count} of the {size} tiles.'
          )
        code = self.index[key]
        changed = (
            (previous == code) != (self.biome == code)
            if previous is not None
            else np.zeros(self.biome.shape, dtype=bool)
        )
        if changed.any() and changed.sum() < 0.5 * changed.size:
          text = news_lib.located(text, self.tiles.heart(changed))
        news.append(text)
        self._announced[key] = count
    return news

  def rows(self) -> list[str]:
    """Biome codes per tile, north first."""
    return [''.join(self.codes[b] for b in row) for row in self.biome[::-1]]

  def legend(self) -> list[dict[str, Any]]:
    return [
        {
            'code': b['code'],
            'name': b['name'],
            'light': list(b['light']),
            'dark': list(b['dark']),
        }
        for b in self.specs
    ]

  def get_state(self) -> dict[str, Any]:
    return {'announced': dict(self._announced or {})}

  def set_state(self, state: Mapping[str, Any]) -> None:
    announced = state.get('announced')
    self._announced = dict(announced) if announced else None
