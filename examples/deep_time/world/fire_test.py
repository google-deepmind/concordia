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

"""Tests for wildfire on the tile map."""

from absl.testing import absltest
from examples.deep_time import game as game_lib
from examples.deep_time.world import data
import numpy as np


def _planet(scenario: str = 'great_acceleration'):
  return game_lib.new_game(scenario).planet


class WildfireTest(absltest.TestCase):

  def test_fires_need_dry_ground_and_fuel(self):
    planet = _planet()
    fire = planet.fire
    covers = planet.tile_life.covers
    risk = fire.risk(covers)
    land = planet.tiles.land
    self.assertTrue(np.all(risk[~land] == 0))  # The sea never burns.
    wet = planet.biomes.rainfall >= float(data.load('fire.yaml')['wet_mm'])
    self.assertTrue(np.all(risk[wet] == 0))
    bare = {k: np.zeros_like(v) for k, v in covers.items()}
    self.assertEqual(float(fire.risk(bare).sum()), 0.0)
    self.assertEqual(fire.flammability('grass7'), fire.flammability('grass'))
    self.assertEqual(fire.flammability('white'), 0.0)  # Daisies don't burn.

  def test_fires_spread_from_strikes_through_risky_tiles(self):
    fire = _planet().fire
    risk = np.zeros(fire.tiles.land.shape)
    risk[10:14, 5:25] = 0.5  # A dry strip; everything else is safe.
    burned = fire.spread(3, risk)
    self.assertTrue(burned.any())
    self.assertTrue(np.all(risk[burned] > 0))  # Fire stays on fuel.
    np.testing.assert_array_equal(burned, fire.spread(3, risk))  # Seeded.

  def test_burning_costs_cover_feeds_the_air_and_makes_news(self):
    planet = _planet()
    air = planet.carbon.amount('atmosphere')
    grass = planet.daisies.cover('grass') + planet.daisies.cover('forests')
    covers = planet.tile_life.covers
    burned = planet.fire.risk(covers) > 0.05
    planet.fire.spread = lambda tick, risk: burned  # Force a big fire.
    events = planet._burn()  # pylint: disable=protected-access
    self.assertGreater(planet.burned_tiles, 20)
    after = planet.daisies.cover('grass') + planet.daisies.cover('forests')
    self.assertLess(after, grass)
    self.assertGreater(planet.carbon.amount('atmosphere'), air)
    self.assertTrue(any('Wildfire sweeps' in e.text for e in events))
    self.assertTrue(np.all(planet.fire.scar[burned] == 3))
    self.assertTrue(np.all(planet.tile_life.growth_boost[burned] > 1.0))
    # The tiles and the regions still agree after the fire.
    for species in planet.daisies.species():
      np.testing.assert_allclose(
          planet.tile_life.region_mean(planet.tile_life.covers[species]),
          planet.daisies.cover_map(species),
          atol=1e-6,
      )

  def test_no_fire_without_oxygen(self):
    fire = _planet().fire
    covers = _planet().tile_life.covers
    burned = fire.step(5, covers, oxygen_pal=0.0)
    self.assertFalse(burned.any())

  def test_checkpoint_round_trip(self):
    planet = _planet()
    for _ in range(6):
      planet.advance()
    state = planet.fire.get_state()
    other = _planet()
    other.fire.set_state(state)
    np.testing.assert_array_equal(other.fire.scar, planet.fire.scar)


if __name__ == '__main__':
  absltest.main()
