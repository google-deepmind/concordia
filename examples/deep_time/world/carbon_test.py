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

"""Tests for the carbon cycle."""

from absl.testing import absltest
from absl.testing import parameterized
from examples.deep_time.world import carbon
from examples.deep_time.world import data
from examples.deep_time.world import planet as planet_lib


def _cycle(**overrides) -> carbon.CarbonCycle:
  config = data.load('carbon.yaml')
  config.update(overrides)
  return carbon.CarbonCycle(
      config, reference_temp_c=25.0, initial_cover=0.5, land=1.0
  )


class CarbonCycleTest(parameterized.TestCase):

  def test_starts_in_steady_state_and_conserves(self):
    cycle = _cycle()
    start = cycle.reservoirs()
    for _ in range(20):
      cycle.step(temp_c=25.0, cover=0.5, land=1.0)
    for reservoir, amount in cycle.reservoirs().items():
      self.assertAlmostEqual(amount, start[reservoir], delta=1.0)
    self.assertAlmostEqual(cycle.forcing(), 0.0, delta=0.1)
    self.assertTrue(cycle.ledger.is_conserved())

  def test_weathering_thermostat_draws_down_a_hot_planet(self):
    hot, cold = _cycle(), _cycle()
    for _ in range(30):
      hot.step(temp_c=40.0, cover=0.5, land=1.0)
      cold.step(temp_c=10.0, cover=0.5, land=1.0)
    self.assertLess(hot.relative_co2(), cold.relative_co2())
    self.assertGreater(
        hot.weathering_rate(40.0, 0.5, 1.0),
        cold.weathering_rate(10.0, 0.5, 1.0),
    )

  def test_life_is_a_carbon_pump(self):
    cycle = _cycle()
    before = cycle.amount(carbon.BIOMASS)
    cycle.step(temp_c=25.0, cover=0.9, land=1.0)
    self.assertLess(cycle.relative_co2(), 1.0)
    self.assertGreater(cycle.amount(carbon.BIOMASS), before)

  def test_co2_starvation_limits_life(self):
    cycle = _cycle()
    self.assertAlmostEqual(cycle.co2_limitation(), 1.0)
    cycle.drawdown(480.0, 'test')
    self.assertLess(cycle.co2_limitation(), 0.5)
    for _ in range(20):
      cycle.step(temp_c=25.0, cover=1.0, land=1.0)
    self.assertGreater(cycle.relative_co2(), 0.2)

  def test_eruptions_outgas_and_ocean_absorbs_over_time(self):
    cycle = _cycle()
    cycle.step(temp_c=25.0, cover=0.5, land=1.0, eruption_magnitude=3)
    spike = cycle.relative_co2()
    self.assertGreater(spike, 1.15)
    for _ in range(15):
      cycle.step(temp_c=25.0, cover=0.5, land=1.0)
    self.assertLess(cycle.relative_co2(), spike)
    self.assertGreater(cycle.forcing(), 0.0)

  def test_drawdown_moves_carbon_to_sediments(self):
    cycle = _cycle()
    before = cycle.amount(carbon.SEDIMENTS)
    self.assertEqual(cycle.drawdown(150.0, 'basalt'), 150.0)
    self.assertAlmostEqual(cycle.relative_co2(), 0.75)
    self.assertLess(cycle.forcing(), -10.0)
    self.assertAlmostEqual(cycle.amount(carbon.SEDIMENTS), before + 150.0)
    self.assertEqual(cycle.ledger.flows()[-1].reason, 'basalt')

  def test_fluxes_are_audited_per_tick(self):
    cycle = _cycle()
    fluxes = cycle.step(temp_c=25.0, cover=0.5, land=1.0)
    self.assertEqual(
        set(fluxes),
        {
            'outgassing',
            'weathering',
            'ocean_uptake',
            'photosynthesis',
            'litterfall',
            'decay',
            'burial',
            'subduction',
        },
    )
    reasons = {f.reason for f in cycle.ledger.flows(tick=0)}
    self.assertIn('weathering', reasons)
    self.assertEqual(cycle.payload()['fluxes'], fluxes)

  def test_state_round_trip(self):
    cycle = _cycle()
    cycle.step(temp_c=30.0, cover=0.6, land=1.0, eruption_magnitude=2)
    restored = _cycle()
    restored.set_state(cycle.get_state())
    a = cycle.step(temp_c=31.0, cover=0.6, land=1.0)
    b = restored.step(temp_c=31.0, cover=0.6, land=1.0)
    self.assertEqual(a, b)
    self.assertEqual(cycle.reservoirs(), restored.reservoirs())


class PlanetCarbonTest(absltest.TestCase):

  def test_planet_couples_carbon_to_climate_and_basalt(self):
    planet = planet_lib.Planet(
        data.scenario('daisyworld'),
        data.load('daisyworld.yaml'),
        planet_lib.interventions_from_config(data.load('interventions.yaml')),
        carbon_config=data.load('carbon.yaml'),
    )
    planet.omega = 10.0
    basalt = {i.key: i for i in planet.interventions()}['basalt_dust']
    self.assertIn('Gt of carbon', planet.apply_intervention('Gaia', basalt))
    planet.advance()
    assert planet.carbon is not None
    self.assertLess(planet.history[-1]['co2'], 1.0)
    self.assertAlmostEqual(
        planet.daisies.greenhouse_forcing(), planet.carbon.forcing()
    )
    self.assertIsNotNone(planet.climate_payload()['co2'])


if __name__ == '__main__':
  absltest.main()
