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

"""Tests for the fauna and grazer dynamics in Deep Time."""

from absl.testing import absltest
from examples.deep_time.world import climate as climate_lib
from examples.deep_time.world import daisyworld
from examples.deep_time.world import data
from examples.deep_time.world import fauna
from examples.deep_time.world import grid as grid_lib
import numpy as np

GRID = grid_lib.RegionGrid(6, 8)


def _world(land=None) -> daisyworld.Daisyworld:
  land = np.ones(GRID.size) if land is None else land
  return daisyworld.from_config(
      data.load('daisyworld.yaml'),
      {'white': 0.3, 'black': 0.3},
      climate_lib.EnergyBalanceClimate(data.load('climate.yaml'), GRID, land),
      land,
      1.0,
  )


def _grazers(land=None, **overrides) -> fauna.Grazers:
  config = data.load('fauna.yaml')
  config['grazers'].update(overrides)
  return fauna.Grazers(
      config, GRID, np.ones(GRID.size) if land is None else land
  )


class GrazersTest(absltest.TestCase):

  def test_grazing_removes_preferred_petals(self):
    white_eaters, black_eaters = _world(), _world()
    _grazers(diet_white=0.9).step(white_eaters)
    _grazers(diet_white=0.1).step(black_eaters)
    self.assertLess(white_eaters.cover('white'), black_eaters.cover('white'))
    self.assertLess(black_eaters.cover('black'), white_eaters.cover('black'))

  def test_diet_sets_the_climate(self):
    warm, cool = _world(), _world()
    warm_herds = _grazers(diet_white=0.85, initial_density=0.15)
    cool_herds = _grazers(diet_white=0.15, initial_density=0.15)
    for _ in range(4):
      warm_herds.step(warm)
      warm.step(1.0)
      cool_herds.step(cool)
      cool.step(1.0)
    self.assertGreater(warm.global_temperature_c(), cool.global_temperature_c())

  def test_herds_starve_without_food_and_respect_the_floor(self):
    world = _world()
    for species in world.species():
      world.kill(species, 1.0)
    herds = _grazers()
    for _ in range(40):
      herds.step(world)
    self.assertTrue(herds.is_extinct())
    self.assertTrue(np.all(herds.density >= herds._floor - 1e-12))

  def test_crowding_caps_booms(self):
    loose, crowded = _grazers(crowding=0.0), _grazers(crowding=3.0)
    w1, w2 = _world(), _world()
    for _ in range(6):
      loose.step(w1)
      crowded.step(w2)
    self.assertLess(crowded.total(), loose.total())

  def test_migration_stays_on_land_and_conserves(self):
    land = np.zeros(GRID.size)
    land[:24] = 1.0
    herds = _grazers(land=land)
    herds.density = herds._floor.copy()
    herds.density[GRID.region(1, 3)] = 0.8
    before = herds.total()
    herds._migrate(0.1)
    self.assertTrue(np.all(herds.density[land == 0] == 0))
    self.assertAlmostEqual(herds.total(), before)
    self.assertGreater(herds.density[GRID.region(1, 4)], 0.0)

  def test_cull_behaviour_and_adaptation(self):
    herds = _grazers()
    lost = herds.cull(0.4)
    self.assertGreater(lost, 0.0)
    self.assertIn('wander', herds.set_behaviour('wander'))
    browse = herds.behaviour_for_option('Browse lightly')
    assert browse is not None
    self.assertEqual(browse.key, 'browse')
    self.assertIn(
        'prefer white petals',
        herds.queue_adaptation('Prefer white petals: they are everywhere.'),
    )
    self.assertIn('stay as they are', herds.queue_adaptation('hmm'))
    ((label, left),) = herds.evolving()  # On its way, for the page.
    self.assertIn('white petals', label)
    self.assertEqual(left, herds.phase_in_ticks)
    world = _world()
    events = []
    for _ in range(2):
      events += herds.step(world)
    self.assertEqual(herds.evolving(), [])
    self.assertAlmostEqual(herds.diet_white, 0.65)
    self.assertTrue(any('have evolved' in e for e in events))
    self.assertIn('prefers white petals 65%', herds.genome_text())

  def test_diet_never_leaves_its_range(self):
    herds = _grazers()
    world = _world()
    for _ in range(10):
      herds.queue_adaptation('prefer white petals')
      herds.step(world)
    self.assertLessEqual(herds.diet_white, herds.diet_range[1] + 1e-9)

  def test_state_round_trip(self):
    herds = _grazers()
    world = _world()
    herds.queue_adaptation('warmer optimum')
    herds.step(world)
    restored = _grazers()
    restored.set_state(herds.get_state())
    w1, w2 = _world(), _world()
    herds.step(w1)
    restored.step(w2)
    np.testing.assert_allclose(herds.density, restored.density)
    self.assertEqual(herds.payload(), restored.payload())


if __name__ == '__main__':
  absltest.main()
