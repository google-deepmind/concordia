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

"""Tests for climate layers and biomes on the tile map."""

import re

from absl.testing import absltest
from examples.deep_time import game as game_lib
from examples.deep_time.world import biomes as biomes_lib
from examples.deep_time.world import data
from examples.deep_time.world import news as news_lib
from examples.deep_time.world import tiles as tiles_lib
import numpy as np

# A biome tally's words: 'The desert spreads from 40 to 60 of the 1200 tiles.'
TALLY = re.compile(r' of the \d+ tiles')


def _planet(scenario: str = 'great_oxidation'):
  return game_lib.new_game(scenario).planet


class BiomesTest(absltest.TestCase):

  def test_thirteen_biomes_with_unique_codes_and_colours(self):
    specs = data.load('biomes.yaml')['biomes']
    self.assertGreaterEqual(len(specs), 10)
    codes = [b['code'] for b in specs]
    self.assertLen(set(codes), len(codes))
    self.assertTrue(
        all(len(b['light']) == 3 and len(b['dark']) == 3 for b in specs)
    )

  def test_highlands_are_colder_and_land_averages_to_its_region(self):
    planet = _planet()
    b = planet.biomes
    tiles = planet.tiles
    field = tiles._upsample(planet.daisies.temperatures_c())  # pylint: disable=protected-access
    land = tiles.land
    region = tiles.region[land]
    count = np.bincount(region, minlength=tiles.grid.size)
    has_land = count > 0
    mean_t = np.bincount(region, b.temperature[land], tiles.grid.size)
    mean_f = np.bincount(region, field[land], tiles.grid.size)
    np.testing.assert_allclose(mean_t[has_land], mean_f[has_land], atol=1e-9)
    high = tiles.terrain == tiles_lib.HIGHLAND
    low = (tiles.terrain == tiles_lib.LOWLAND) | (
        tiles.terrain == tiles_lib.COAST
    )
    for r in np.unique(tiles.region[high]):
      here = tiles.region == r
      if (low & here).any():
        offset = b.temperature - field
        self.assertLess(offset[high & here].mean(), offset[low & here].mean())
    sea = ~land
    np.testing.assert_allclose(b.temperature[sea], field[sea])

  def test_the_tropics_are_wetter_than_the_subtropics(self):
    b = _planet().biomes
    lat = np.abs(b.lat)
    tropics = b.rainfall[lat < 8].mean()
    subtropics = b.rainfall[(lat > 24) & (lat < 32)].mean()
    self.assertGreater(tropics, 2 * subtropics)

  def test_classification_follows_temperature_rain_and_ice(self):
    planet = _planet()
    b = planet.biomes
    i = b.index
    land = planet.tiles.land
    rows, cols = b.biome.shape
    cold = np.full((rows, cols), -5.0)
    wet_hot = np.full((rows, cols), 27.0)
    dry = np.full((rows, cols), 100.0)
    wet = np.full((rows, cols), 2500.0)
    planet.tiles.ice[:] = False
    self.assertTrue(np.all(b._classify(cold, wet)[land] == i['tundra']))  # pylint: disable=protected-access
    hot_wet = b._classify(wet_hot, wet)  # pylint: disable=protected-access
    self.assertTrue(
        np.all(np.isin(hot_wet[land], [i['rainforest'], i['wetland']]))
    )
    self.assertTrue(np.all(b._classify(wet_hot, dry)[land] == i['desert']))  # pylint: disable=protected-access
    shelf = planet.tiles.terrain == tiles_lib.SHELF
    self.assertTrue(np.all(hot_wet[shelf] == i['reef']))
    planet.tiles.ice[:] = True
    frozen = b._classify(wet_hot, wet)  # pylint: disable=protected-access
    self.assertTrue(np.all(frozen[land] == i['ice']))
    self.assertTrue(np.all(frozen[~land] == i['sea_ice']))

  def test_biomes_shift_with_the_climate_and_say_so(self):
    planet = _planet('great_acceleration')
    b = planet.biomes
    before = b.counts()
    news = b.update(planet.daisies.temperatures_c() + 12.0)
    after = b.counts()
    cold = ('tundra', 'taiga')
    self.assertLess(sum(after[k] for k in cold), sum(before[k] for k in cold))
    self.assertTrue(any('retreat' in n for n in news))
    self.assertTrue(any('spread' in n for n in news))
    self.assertEqual(b.update(planet.daisies.temperatures_c() + 12.0), [])

  def test_biome_news_is_placed_where_the_biome_changed(self):
    planet = _planet('great_acceleration')
    b = planet.biomes
    taiga = b.index['taiga']
    before = b.biome.copy()
    news = b.update(planet.daisies.temperatures_c() + 12.0)
    item = next(n for n in news if n.startswith('The Taiga') or 'aiga' in n)
    where = news_lib.place_of(item)
    assert where is not None
    changed = (before == taiga) != (b.biome == taiga)
    self.assertTrue(changed[where])  # The heart of the retreat.
    # The planet's events carry the place, so the map ripples from there,
    # but a tally of the whole planet is not local news.
    for _ in range(12):
      report = planet.advance()
      tallies = [e for e in report.events if TALLY.search(e.text)]
      for event in tallies:
        self.assertFalse(event.local)
      if any(e.where is not None for e in tallies):
        break
    else:
      self.fail('no placed biome news in 12 ticks')

  def test_an_all_land_world_has_no_seas_or_continental_drought(self):
    b = _planet('daisyworld').biomes
    sea = ('deep_ocean', 'shelf', 'reef', 'sea_ice')
    self.assertEqual(sum(b.counts()[k] for k in sea), 0)
    self.assertLess(b.counts()['desert'], 0.4 * b.tiles.size)

  def test_rows_legend_and_checkpoint(self):
    context = game_lib.new_game('great_oxidation')
    planet = context.planet
    rows = planet.biomes.rows()
    self.assertLen(rows, 30)
    self.assertLessEqual(set(''.join(rows)), set(planet.biomes.codes))
    self.assertEqual(
        [b['code'] for b in planet.biomes.legend()], list(planet.biomes.codes)
    )
    planet.advance()
    other = game_lib.new_game('great_oxidation')
    other.set_state(context.get_state())
    np.testing.assert_array_equal(
        other.planet.biomes.biome, planet.biomes.biome
    )
    self.assertEqual(other.planet.biomes.counts(), planet.biomes.counts())
    self.assertIsInstance(planet.biomes, biomes_lib.Biomes)


if __name__ == '__main__':
  absltest.main()
