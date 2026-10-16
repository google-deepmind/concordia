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

"""Tests for the tile map."""

from absl.testing import absltest
from absl.testing import parameterized
from examples.deep_time import game as game_lib
from examples.deep_time import scenarios
from examples.deep_time.world import data
from examples.deep_time.world import geography
from examples.deep_time.world import grid as grid_lib
from examples.deep_time.world import tiles
import numpy as np

GRID = grid_lib.RegionGrid(6, 8)
CONFIG = data.load('tiles.yaml')


def _map(scenario: str = 'great_acceleration', seed: int = 0) -> tiles.TileMap:
  land = geography.land_fraction(
      data.scenario(scenario).get('geography'),
      GRID,
      seed,
      data.load('ocean.yaml'),
  )
  return tiles.TileMap(GRID, land, CONFIG, seed)


class TileMapTest(parameterized.TestCase):

  @parameterized.parameters(
      *[(s, seed) for s in sorted(scenarios.SCENARIOS) for seed in (0, 3)]  # pylint: disable=g-complex-comprehension
  )
  def test_regions_aggregate_exactly(self, scenario, seed):
    land = geography.land_fraction(
        data.scenario(scenario).get('geography'),
        GRID,
        seed,
        data.load('ocean.yaml'),
    )
    tile_map = tiles.TileMap(GRID, land, CONFIG, seed)
    self.assertEqual((tile_map.rows, tile_map.cols), (30, 40))
    self.assertEqual(tile_map.size, 1200)
    np.testing.assert_allclose(tile_map.region_land(), np.round(land * 25) / 25)

  def test_terrain_is_deterministic_and_seeded(self):
    a, b, c = _map(seed=0), _map(seed=0), _map(seed=5)
    np.testing.assert_array_equal(a.terrain, b.terrain)
    np.testing.assert_array_equal(a.elevation, b.elevation)
    self.assertFalse(np.array_equal(a.terrain, c.terrain))

  def test_classes_follow_the_coast(self):
    tile_map = _map()
    land = tile_map.land
    near_land = tile_map._near(land)  # pylint: disable=protected-access
    near_sea = tile_map._near(~land)  # pylint: disable=protected-access
    t = tile_map.terrain
    self.assertTrue(np.all(near_land[t == tiles.SHELF]))
    self.assertFalse(np.any(near_land[t == tiles.DEEP_OCEAN]))
    self.assertTrue(np.all(near_sea[t == tiles.COAST]))
    self.assertTrue(np.all(land[t >= tiles.COAST]))
    self.assertTrue(np.all(tile_map.elevation[land] > 0))
    self.assertTrue(np.all(tile_map.elevation[~land] < 0))
    self.assertGreater(np.count_nonzero(t == tiles.HIGHLAND), 0)

  def test_an_all_land_world_is_rolling_lowland_without_coasts(self):
    tile_map = _map('daisyworld')
    self.assertTrue(tile_map.land.all())
    self.assertFalse(np.isin(tile_map.terrain, [0, 1, 2]).any())
    self.assertLess(tile_map.elevation.max(), 2000)

  def test_each_region_freezes_its_share_coldest_first(self):
    tile_map = _map()
    shares = np.linspace(0.0, 1.0, GRID.size)
    tile_map.update_ice(shares)
    frozen = np.zeros(GRID.size)
    np.add.at(frozen, tile_map.region.ravel(), tile_map.ice.ravel())
    np.testing.assert_allclose(frozen / 25, np.round(shares * 25) / 25)
    # In a region half frozen, the frozen tiles are the poleward ones.
    region = GRID.region(5, 2)  # The northernmost band.
    half = np.zeros(GRID.size)
    half[region] = 0.4
    tile_map.update_ice(half)
    rows = np.nonzero(tile_map.ice)[0]
    self.assertGreaterEqual(rows.mean(), 27)

  def test_the_heart_of_a_patch_is_its_most_surrounded_tile(self):
    m = _map()
    mask = np.zeros((m.rows, m.cols), dtype=bool)
    mask[10:13, 0:3] = True  # A 3 x 3 patch: its middle is the heart.
    mask[20, 30] = True  # A lone tile elsewhere.
    self.assertEqual(m.heart(mask), (11, 1))
    mask[:] = False
    mask[5, m.cols - 1] = mask[5, 0] = mask[5, 1] = True  # Across the seam.
    self.assertEqual(m.heart(mask), (5, 0))

  def test_payload_is_north_first_and_compact(self):
    tile_map = _map()
    payload = tile_map.payload()
    self.assertLen(payload['terrain'], 30)
    self.assertTrue(all(len(row) == 40 for row in payload['terrain']))
    self.assertEqual(
        payload['terrain'][0],
        ''.join(tiles.CODES[c] for c in tile_map.terrain[-1]),
    )
    self.assertLessEqual(set(''.join(payload['terrain'])), set(tiles.CODES))

  def test_the_planet_keeps_tiles_consistent_and_checkpointed(self):
    context = game_lib.new_game('snowball')
    planet = context.planet
    np.testing.assert_allclose(
        planet.tiles.region_land(), planet.daisies.land_fraction(), atol=0.02
    )
    self.assertTrue(planet.tiles.ice.any())  # Snowball starts frozen.
    planet.advance()
    state = context.get_state()
    other = game_lib.new_game('snowball')
    other.set_state(state)
    np.testing.assert_array_equal(other.planet.tiles.ice, planet.tiles.ice)


if __name__ == '__main__':
  absltest.main()
