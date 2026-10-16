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

"""Tests for the Savanna Fire scenario and summoned rain."""

from absl.testing import absltest
from examples.deep_time import game as game_lib
from examples.deep_time import sweep
import numpy as np


def _rain(planet):
  return next(i for i in planet.interventions() if i.key == 'summon_rain')


class SavannaFireTest(absltest.TestCase):

  def test_aimed_rain_damps_fire_around_the_tap_for_a_few_ticks(self):
    planet = game_lib.new_game('savanna_fire').planet
    planet.omega = 20.0
    covers = planet.tile_life.covers
    before = planet.fire.risk(covers)
    row, col = (int(v) for v in np.argwhere(before == before.max())[0])
    target = (row, col)
    planet.set_target('Gaia', *target)
    text = planet.apply_intervention('Gaia', _rain(planet))
    self.assertIn('around', text)
    after = planet.fire.risk(covers)
    self.assertLess(after[target], 0.2 * before[target])
    far = planet._near_target(target) > 8  # pylint: disable=protected-access
    np.testing.assert_allclose(after[far], before[far])
    planet.fire.step(1, covers)
    self.assertGreater(planet.fire.wet[target], 0.0)  # Drying, not dry.
    self.assertLess(planet.fire.wet[target], 0.9)

  def test_unaimed_rain_is_a_light_wet_season_everywhere(self):
    planet = game_lib.new_game('savanna_fire').planet
    planet.omega = 20.0
    planet.apply_intervention('Gaia', _rain(planet))
    land = planet.tiles.land
    np.testing.assert_allclose(planet.fire.wet[land], 0.27)

  def test_rain_keeps_the_woods_and_a_passive_steward_loses_them(self):
    kept = sweep.run_one('savanna_fire', 'default', 0)
    self.assertEqual(kept['outcome'], 'victory', kept['reason'])
    lost = sweep.run_one('savanna_fire', 'passive', 0)
    self.assertEqual(lost['outcome'], 'defeat')
    self.assertIn('kept only', lost['reason'])

  def test_soaked_ground_does_not_burn_and_dries_after_the_fires(self):
    planet = game_lib.new_game('savanna_fire').planet
    covers = planet.tile_life.covers
    fire = planet.fire
    fire.soak(1.0)
    self.assertFalse(fire.step(1, covers).any())  # Rain on the fire season.
    self.assertAlmostEqual(float(fire.wet.max()), 0.6)  # Then it dries.

  def test_damp_ground_burns_far_less_than_dry(self):
    burned = {}
    for wet in (0.0, 0.7):
      planet = game_lib.new_game('savanna_fire').planet
      covers = planet.tile_life.covers
      total = 0
      for tick in range(1, 9):
        planet.fire.wet[:] = wet  # Fronts as well as strikes are damped.
        total += int(planet.fire.step(tick, covers).sum())
      burned[wet] = total
    self.assertGreater(burned[0.0], 0)
    self.assertLess(burned[0.7], burned[0.0] / 4)


if __name__ == '__main__':
  absltest.main()
