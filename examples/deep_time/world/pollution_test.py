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

"""Tests for local pollution on the tile map."""

from absl.testing import absltest
from examples.deep_time import game as game_lib
import numpy as np


def _planet(scenario: str = 'great_acceleration'):
  return game_lib.new_game(scenario).planet


class PollutionTest(absltest.TestCase):

  def test_soot_settles_around_the_towns_that_make_it(self):
    planet = _planet()
    pollution, towns = planet.pollution, planet.settlements
    keys = list(towns.places)
    smoky = keys[0]
    peoples = {k: (towns.tiles_of(k), {}) for k in keys}
    peoples[smoky] = (towns.tiles_of(smoky), {'soot': 2.0})
    pollution.step(peoples, 0.0)
    r, c = towns.tiles_of(smoky)[0]
    self.assertGreater(pollution.soot[r, c], 0.5)
    for other in keys[1:]:
      orr, oc = towns.tiles_of(other)[0]
      self.assertEqual(pollution.soot[orr, oc], 0.0)
    self.assertGreater(pollution.crop_loss()[r, c], 0.0)
    self.assertTrue(np.all(pollution.soot[~planet.tiles.land] == 0))

  def test_dead_zones_open_off_the_coasts_that_farm_hardest(self):
    planet = _planet()
    pollution, towns = planet.pollution, planet.settlements
    keys = list(towns.places)
    farmer = keys[-1]
    peoples = {k: (towns.tiles_of(k), {}) for k in keys}
    peoples[farmer] = (towns.tiles_of(farmer), {'runoff': 1.0})
    pollution.step(peoples, 0.1)
    shelf = pollution.shelf
    self.assertEqual(int(pollution.dead.sum()), int(round(0.1 * shelf.sum())))
    self.assertTrue(np.all(shelf[pollution.dead]))
    choked = pollution.fishing(towns.tiles_of(farmer))
    clean = [pollution.fishing(towns.tiles_of(k)) for k in keys[:-1]]
    self.assertLess(choked, 1.0)
    self.assertTrue(all(f >= 1.0 for f in clean))

  def test_without_runoff_dead_zones_open_in_the_warmest_shallows(self):
    planet = _planet()
    pollution = planet.pollution
    pollution.step({}, 0.2)
    warm = planet.biomes.temperature
    shelf = pollution.shelf
    self.assertGreaterEqual(
        warm[pollution.dead].min(), np.median(warm[shelf & ~pollution.dead])
    )

  def test_checkpoint_round_trip(self):
    planet = _planet()
    for _ in range(8):
      planet.advance()
    state = planet.pollution.get_state()
    other = _planet()
    other.pollution.set_state(state)
    self.assertEqual(other.pollution.get_state(), state)


if __name__ == '__main__':
  absltest.main()
