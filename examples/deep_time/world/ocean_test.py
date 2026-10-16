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

"""Tests for geography, ocean currents, sea level and pH."""

import json

from absl.testing import absltest
from concordia.contrib.pubsub import topics
from examples.deep_time.world import climate as climate_lib
from examples.deep_time.world import data
from examples.deep_time.world import geography
from examples.deep_time.world import geology
from examples.deep_time.world import grid as grid_lib
from examples.deep_time.world import ocean as ocean_lib
from examples.deep_time.world import planet as planet_lib
from examples.deep_time.world import publishing
import numpy as np

GRID = grid_lib.RegionGrid(6, 8)
OCEAN = data.load('ocean.yaml')


class GeographyTest(absltest.TestCase):

  def test_continents_are_seeded_and_hit_the_target(self):
    first = geography.continents(GRID, 0.45, 3, seed=4)
    again = geography.continents(GRID, 0.45, 3, seed=4)
    other = geography.continents(GRID, 0.45, 3, seed=5)
    np.testing.assert_array_equal(first, again)
    self.assertFalse(np.array_equal(first, other))
    area = float(np.dot(GRID.area_weights(), first))
    self.assertBetween(area, 0.2, 0.5)
    self.assertLessEqual(set(np.unique(first)), {0.0, 0.5, 1.0})
    self.assertIn(0.5, first)

  def test_explicit_geographies(self):
    np.testing.assert_array_equal(
        geography.land_fraction({'land': 'all'}, GRID, 0, OCEAN),
        np.ones(48),
    )
    self.assertEqual(
        geography.land_fraction({'land': 0.25}, GRID, 0, OCEAN)[0], 0.25
    )
    with self.assertRaises(ValueError):
      geography.land_fraction({'land': [1, 0]}, GRID, 0, OCEAN)


class CurrentsTest(absltest.TestCase):

  def test_currents_flatten_the_gradient_over_sea(self):
    config = data.load('climate.yaml')
    sea_world = np.zeros(GRID.size)
    still = climate_lib.EnergyBalanceClimate(
        dict(config, ocean_current_boost=0.0), GRID, sea_world
    )
    flowing = climate_lib.EnergyBalanceClimate(
        dict(config, ocean_current_boost=2.0), GRID, sea_world
    )
    flat = lambda t: np.full(GRID.size, 0.4)
    t_still, _ = still.equilibrium(1.0, flat)
    t_flow, _ = flowing.equilibrium(1.0, flat)
    spread = lambda t: np.ptp(GRID.band_means(t))
    self.assertLess(spread(t_flow), spread(t_still))
    self.assertAlmostEqual(
        still.global_mean(t_still), flowing.global_mean(t_flow)
    )


class OceanTest(absltest.TestCase):

  def test_sea_level_ph_and_alert(self):
    ocean = ocean_lib.Ocean(OCEAN, reference_ocean_carbon=2000.0)
    weights = GRID.area_weights()
    land = np.full(GRID.size, 0.5)
    alerts = ocean.update(
        weights=weights,
        land=land,
        ice=np.ones(GRID.size),
        temps_c=np.full(GRID.size, -20.0),
        ocean_carbon=2000.0,
    )
    self.assertEqual(alerts, [])
    self.assertAlmostEqual(ocean.sea_level_m, -60.0)
    self.assertAlmostEqual(ocean.sea_ice, 1.0)
    self.assertAlmostEqual(ocean.ph, 8.2)
    alerts = ocean.update(
        weights=weights,
        land=land,
        ice=np.zeros(GRID.size),
        temps_c=np.full(GRID.size, 25.0),
        ocean_carbon=2000.0 * np.e**1.5,
    )
    self.assertLen(alerts, 1)
    self.assertLess(ocean.ph, 7.8)
    self.assertEqual(ocean.sea_level_m, 0.0)
    payload = ocean.payload(0.5)
    self.assertEqual(payload['sea_area'], 0.5)
    restored = ocean_lib.Ocean(OCEAN, 1.0)
    restored.set_state(ocean.get_state())
    self.assertEqual(restored.payload(0.5), payload)


class SnowballTest(absltest.TestCase):

  def _planet(self, seed: int = 0) -> planet_lib.Planet:
    return planet_lib.Planet(
        data.scenario('snowball'),
        data.load('daisyworld.yaml'),
        planet_lib.interventions_from_config(data.load('interventions.yaml')),
        lithosphere=geology.Lithosphere(data.load('geology.yaml'), seed=seed),
        carbon_config=data.load('carbon.yaml'),
        seed=seed,
    )

  def test_starts_frozen_and_stalls_weathering(self):
    planet = self._planet()
    self.assertLess(planet.temperature_c(), -10.0)
    self.assertGreater(planet.ice_cover(), 0.9)
    self.assertLess(planet.ice_free_land(), 0.01)
    self.assertLess(planet.ocean.sea_level_m, -10.0)
    planet.advance()
    assert planet.carbon is not None
    self.assertLess(planet.carbon.last_fluxes['weathering'], 0.1)
    self.assertGreater(planet.carbon.relative_co2(), 1.0)
    self.assertIsNone(planet.outcome)

  def _play(self, planet, haze_from: float | None) -> None:
    """Rouses the volcanoes; once the air holds `haze_from`, hazes."""
    moves = {i.key: i for i in planet.interventions()}
    rouse, haze = moves['rouse_volcanoes'], moves['dark_haze']
    assert planet.carbon is not None
    while planet.outcome is None:
      co2 = planet.carbon.relative_co2()
      if haze_from is not None and co2 >= haze_from:
        if planet.ready(haze) and planet.omega >= haze.cost:
          planet.apply_intervention('Gaia', haze)
      elif planet.ready(rouse) and planet.omega >= rouse.cost:
        self.assertIn(
            'rose into the air', planet.apply_intervention('Gaia', rouse)
        )
      planet.advance()

  def test_carbon_then_a_haze_breaks_the_ice(self):
    for seed in (0, 1, 2):
      planet = self._planet(seed)
      self._play(planet, haze_from=2.0)
      self.assertEqual(planet.outcome, planet_lib.VICTORY)
      self.assertEqual(planet.outcome_reason, 'the ice broke')
      self.assertLess(planet.tick, planet.max_ticks)
      self.assertGreater(planet.temperature_c(), 0.0)

  def test_under_a_faint_sun_carbon_alone_does_not(self):
    planet = self._planet()
    self._play(planet, haze_from=None)
    self.assertEqual(planet.outcome, planet_lib.DEFEAT)
    assert planet.carbon is not None
    self.assertGreater(planet.carbon.relative_co2(), 2.0)  # Plenty to tip.

  def test_a_haze_too_soon_only_warms_a_turn(self):
    planet = self._planet()
    (haze,) = [i for i in planet.interventions() if i.key == 'dark_haze']
    planet.advance()
    planet.apply_intervention('Gaia', haze)
    for _ in range(3):
      planet.advance()
    self.assertIsNone(planet.outcome)
    self.assertGreater(planet.ice_cover(), 0.9)

  def test_the_air_gauge_shows_where_a_haze_tips_the_ice(self):
    payload = self._planet().cycles_payload()
    self.assertEqual(payload['tipping_co2'], [1.3, 2.0])
    text = publishing.render_carbon(
        topics.Message(
            topic='cycles.carbon', payload=json.dumps(payload), seq=1
        )
    )
    self.assertIn('A dark haze may tip the ice from x1.3', text)

  def test_frozen_planet_loses_at_the_deadline(self):
    scenario = data.scenario('snowball')
    scenario['max_ticks'] = 3
    planet = planet_lib.Planet(
        scenario,
        data.load('daisyworld.yaml'),
        planet_lib.interventions_from_config(data.load('interventions.yaml')),
        carbon_config=data.load('carbon.yaml'),
    )
    for _ in range(3):
      planet.advance()
    self.assertEqual(planet.outcome, planet_lib.DEFEAT)
    self.assertEqual(planet.outcome_reason, 'the planet stayed frozen')


if __name__ == '__main__':
  absltest.main()
