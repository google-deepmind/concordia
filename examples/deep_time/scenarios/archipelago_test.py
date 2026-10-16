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

"""Tests for the Archipelago scenario."""

from absl.testing import absltest
from examples.deep_time import game as game_lib
from examples.deep_time import sweep
import numpy as np


class ArchipelagoTest(absltest.TestCase):

  def test_a_sea_of_islands_with_life_on_one(self):
    planet = game_lib.new_game('archipelago').planet
    self.assertGreaterEqual(len(planet.islands), 6)
    for one, two in zip(planet.islands, planet.islands[1:]):
      self.assertFalse((one & two).any())
    total = np.sum(list(planet.tile_life.covers.values()), axis=0)
    self.assertGreater(total[planet.islands[0]].mean(), 0.3)
    for island in planet.islands[1:]:
      self.assertEqual(float(total[island].sum()), 0.0)  # No seed bank yet.
    self.assertEqual(planet.green_islands(), (1, len(planet.islands)))
    self.assertFalse(planet.is_lifeless())
    planet.advance()
    total = np.sum(list(planet.tile_life.covers.values()), axis=0)
    self.assertEqual(float(total[planet.islands[-1]].sum()), 0.0)

  def test_sowing_greens_islands_and_a_passive_steward_loses(self):
    sown = sweep.run_one('archipelago', 'default', 0)
    self.assertEqual(sown['outcome'], 'victory', sown['reason'])
    idle = sweep.run_one('archipelago', 'passive', 0)
    self.assertEqual(idle['outcome'], 'defeat')
    self.assertIn('islands are green', idle['reason'])

  def test_unaimed_sowing_greens_only_the_nearest_bare_island(self):
    planet = game_lib.new_game('archipelago').planet
    planet.omega = 20.0
    sow = next(i for i in planet.interventions() if i.key == 'sow_grass')
    nearest = planet._nearest_bare_island()  # pylint: disable=protected-access
    assert nearest is not None
    planet.apply_intervention('Gaia', sow)
    total = np.sum(list(planet.tile_life.covers.values()), axis=0)
    self.assertGreater(float(total[nearest].mean()), 0.3)
    self.assertTrue(planet.tile_life.reached[nearest].all())
    others = [i for i in planet.islands[1:] if not np.array_equal(i, nearest)]
    for island in others:
      self.assertEqual(float(total[island].sum()), 0.0)
    self.assertEqual(planet.green_islands()[0], 2)

  def test_aimed_sowing_stays_on_the_island_it_lands_on(self):
    planet = game_lib.new_game('archipelago').planet
    planet.omega = 20.0
    sow = next(i for i in planet.interventions() if i.key == 'sow_grass')
    small = planet.islands[-1]
    row, col = (int(v) for v in np.argwhere(small)[0])
    planet.set_target('Gaia', row, col)
    planet.apply_intervention('Gaia', sow)
    total = np.sum(list(planet.tile_life.covers.values()), axis=0)
    self.assertGreater(float(total[small].sum()), 0.0)
    for island in planet.islands[1:-1]:  # Not the start, not the target.
      self.assertEqual(float(total[island].sum()), 0.0)

  def test_seed_banks_sprout_only_where_life_has_been(self):
    planet = game_lib.new_game('archipelago').planet
    banks = planet.seed_banks
    assert banks is not None
    planted = planet._sow('grass', 0.01)  # pylint: disable=protected-access
    self.assertGreater(planted, 0.0)
    total = np.sum(list(planet.tile_life.covers.values()), axis=0)
    for island in planet.islands[1:]:
      self.assertEqual(float(total[island].sum()), 0.0)

  def test_extinction_is_judged_against_the_land_life_holds(self):
    planet = game_lib.new_game('archipelago').planet
    floor = planet._extinct_floor()  # pylint: disable=protected-access
    bank = planet.daisies.params.min_cover * float(
        planet.climate.global_mean(
            planet.tile_life.region_mean(
                (planet.tiles.land & planet.tile_life.reached).astype(float)
            )
        )
    )
    self.assertAlmostEqual(floor, 1.5 * bank)
    self.assertFalse(planet.is_extinct('grass'))  # The start island is green.

  def test_the_hud_shows_band_strain(self):
    planet = game_lib.new_game('archipelago').planet
    labels = [i['label'] for i in planet.progress()]
    self.assertEqual(labels, ['turn', 'islands', 'climate', 'strain'])


if __name__ == '__main__':
  absltest.main()
