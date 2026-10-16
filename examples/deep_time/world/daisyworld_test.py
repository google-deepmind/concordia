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

"""Tests for the regional grid, climate and Daisyworld physics."""

from absl.testing import absltest
from absl.testing import parameterized
from examples.deep_time.world import climate as climate_lib
from examples.deep_time.world import daisyworld
from examples.deep_time.world import data
from examples.deep_time.world import grid as grid_lib
import numpy as np

GRID = grid_lib.RegionGrid(6, 8)


def _climate(**overrides) -> climate_lib.EnergyBalanceClimate:
  config = data.load('climate.yaml')
  config.update(overrides)
  return climate_lib.EnergyBalanceClimate(config, GRID)


def _world(
    white: float = 0.2, black: float = 0.2, land=None, luminosity: float = 1.0
) -> daisyworld.Daisyworld:
  return daisyworld.from_config(
      data.load('daisyworld.yaml'),
      {'white': white, 'black': black},
      _climate(),
      np.ones(GRID.size) if land is None else land,
      luminosity,
  )


class GridTest(absltest.TestCase):

  def test_geometry(self):
    self.assertEqual(GRID.size, 48)
    self.assertAlmostEqual(GRID.area_weights().sum(), 1.0)
    self.assertGreater(
        GRID.area_weights()[GRID.region(2, 0)],
        GRID.area_weights()[GRID.region(0, 0)],
    )
    self.assertCountEqual(GRID.neighbours(GRID.region(0, 0)), [7, 1, 8])
    self.assertLen(GRID.neighbours(GRID.region(3, 4)), 4)
    self.assertEqual(GRID.band_name(5), '75°N')
    self.assertEqual(GRID.region_name(GRID.region(1, 3)), '45°S/3')

  def test_diffusion_conserves_energy(self):
    temps = np.random.default_rng(0).normal(20, 15, GRID.size)
    flux = GRID.diffusion_operator() @ temps
    self.assertAlmostEqual(float(np.dot(GRID.area_weights(), flux)), 0.0)


class ClimateTest(parameterized.TestCase):

  @parameterized.parameters(1.6, 6.0, 12.0)
  def test_global_mean_matches_classic_daisyworld(self, diffusion):
    climate = _climate(diffusion=diffusion)
    temps, _ = climate.equilibrium(1.15, lambda t: np.full(GRID.size, 0.5))
    # Classic Daisyworld bare planet at L=1.15 runs at 37.4 C.
    self.assertAlmostEqual(climate.global_mean(temps), 37.44, places=1)
    bands = GRID.band_means(temps)
    self.assertGreater(bands[2], bands[1])
    self.assertGreater(bands[1], bands[0])

  def test_ice_albedo_feedback_cools_further(self):
    climate = _climate()
    open_albedo = np.full(GRID.size, 0.5)

    def no_ice(t):
      del t
      return open_albedo

    def with_ice(t):
      return climate.surface_albedo(
          open_albedo, np.zeros(GRID.size), np.ones(GRID.size), t
      )

    cold_no_ice, _ = climate.equilibrium(0.55, no_ice)
    cold_ice, albedo = climate.equilibrium(0.55, with_ice)
    self.assertLess(
        climate.global_mean(cold_ice), climate.global_mean(cold_no_ice)
    )
    self.assertGreater(albedo.max(), 0.5)

  def test_greenhouse_forcing_warms(self):
    climate = _climate()
    flat = lambda t: np.full(GRID.size, 0.5)
    base, _ = climate.equilibrium(1.0, flat)
    warm, _ = climate.equilibrium(1.0, flat, forcing=12.24)
    self.assertAlmostEqual(
        climate.global_mean(warm) - climate.global_mean(base), 2.0, places=3
    )


class DaisyworldTest(parameterized.TestCase):

  @parameterized.parameters(0.9, 1.0, 1.2, 1.3)
  def test_daisies_regulate_along_a_slow_ramp(self, luminosity):
    world = _world(luminosity=0.9)
    level = 0.9
    while level < luminosity:
      for _ in range(3):
        world.step(level)
      level = min(luminosity, level + 0.05)
    for _ in range(5):
      world.step(luminosity)
    self.assertBetween(world.global_temperature_c(), 15.0, 32.0)

  def test_bare_planet_is_hot_and_ocean_cannot_grow_daisies(self):
    land = np.where(np.arange(GRID.size) % 2 == 0, 1.0, 0.0)
    world = _world(white=0.3, black=0.3, land=land)
    for _ in range(3):
      world.step(1.0)
    for species in world.species():
      self.assertTrue(np.all(world.cover_map(species)[land == 0] == 0))
    self.assertAlmostEqual(world.land(), 0.5, places=1)

  def test_white_daisies_cool_black_daisies_warm(self):
    white = _world(white=0.6, black=0.0)
    black = _world(white=0.0, black=0.6)
    self.assertLess(white.global_temperature_c(), black.global_temperature_c())
    self.assertTrue(
        np.all(
            white.local_temperature_c('white')
            < white.local_temperature_c('black')
        )
    )

  def test_cover_stays_within_land(self):
    world = _world(white=0.49, black=0.49)
    for luminosity in (0.6, 1.0, 1.7, 1.0):
      for _ in range(3):
        world.step(luminosity)
        self.assertTrue(np.all(world.bare_map() >= -1e-9))
        for species in world.species():
          self.assertTrue(
              np.all(world.cover_map(species) >= world.params.min_cover - 1e-12)
          )

  def test_seed_and_kill_are_bounded(self):
    world = _world(white=0.5, black=0.45)
    self.assertAlmostEqual(world.seed('white', 0.2), 0.05)
    self.assertAlmostEqual(world.bare(), 0.0)
    self.assertAlmostEqual(world.kill('black', 0.5), 0.225)
    self.assertAlmostEqual(world.kill('black', 1.0), 0.225 - 0.01)
    self.assertAlmostEqual(world.cover('black'), world.params.min_cover)

  def test_albedo_pulse_and_fertility_decay(self):
    world = _world()
    base = world.planetary_albedo()
    world.add_albedo_pulse(0.1)
    world.add_fertility(0.2, decay=0.5)
    world.step(1.0)
    self.assertGreater(world.cloud_albedo(), 0.0)
    self.assertAlmostEqual(world.fertility(), 1.1)
    for _ in range(20):
      world.step(1.0)
    self.assertEqual(world.cloud_albedo(), 0.0)
    self.assertEqual(world.fertility(), 1.0)
    del base

  def test_band_summary(self):
    summary = _world().band_summary()
    self.assertLen(summary, 6)
    self.assertEqual(summary[0]['band'], '75°S')
    self.assertEqual(set(summary[0]['cover']), {'white', 'black'})

  def test_state_round_trip_is_deterministic(self):
    world = _world()
    world.step(1.1)
    world.set_greenhouse_forcing(3.0)
    copy = _world(white=0.0, black=0.0)
    copy.set_state(world.get_state())
    world.step(1.2)
    copy.step(1.2)
    np.testing.assert_allclose(world.temperatures_c(), copy.temperatures_c())
    self.assertEqual(world.get_state(), copy.get_state())

  def test_traits_change_growth(self):
    world = _world()
    before = world.growth_rate('white').sum()
    traits = world.traits('white')
    world.set_traits(
        'white',
        daisyworld.SpeciesTraits(
            albedo=traits.albedo,
            optimum_c=traits.optimum_c + 10,
            growth_width=traits.growth_width,
        ),
    )
    self.assertNotAlmostEqual(world.growth_rate('white').sum(), before)


if __name__ == '__main__':
  absltest.main()
