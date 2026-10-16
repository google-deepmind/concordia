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

"""Tests for Gaia's backlash."""

from absl.testing import absltest
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
from examples.deep_time.world import backlash as backlash_lib
from examples.deep_time.world import data
from examples.deep_time.world import grid as grid_lib
import numpy as np

GRID = grid_lib.RegionGrid(bands=6, sectors=8)
HOMES = {'amber': [9, 10], 'basalt': [30]}
NAMES = {'amber': 'Amber folk', 'basalt': 'Basalt clans'}


def _backlash() -> backlash_lib.Backlash:
  return backlash_lib.Backlash(data.load('backlash.yaml'), GRID, seed=1)


class BacklashTest(absltest.TestCase):

  def test_lingering_pollution_strikes_and_forests_shield(self):
    bare, wooded = _backlash(), _backlash()
    forest = np.full(GRID.size, 0.8)
    attacks = []
    for _ in range(15):
      for b in (bare, wooded):
        b.emit(HOMES['amber'], {'soot': 6.0, 'fossil_co2': 6.0})
      attacks += bare.step(
          np.zeros(GRID.size), np.zeros(GRID.size), HOMES, NAMES
      )
      wooded.step(forest, np.zeros(GRID.size), HOMES, NAMES)
    self.assertGreater(bare.pollution.sum(), wooded.pollution.sum())
    self.assertGreater(bare.count, wooded.count)
    self.assertNotEmpty(attacks)
    self.assertEqual({a.polity for a in attacks}, {'amber'})
    self.assertIn(attacks[0].kind, ('pests', 'plague'))
    self.assertIn('Amber folk', attacks[0].text)

  def test_pollution_drifts_to_neighbours_and_rains_cleanse(self):
    b = _backlash()
    b.emit([9], {'soot': 10.0})
    b.step(np.zeros(GRID.size), np.zeros(GRID.size), HOMES, NAMES)
    self.assertGreater(b.pollution[GRID.neighbours(9)[0]], 0.0)
    b.pressure[9] = 0.9
    b.cleanse(0.5)
    self.assertAlmostEqual(b.pressure[9], 0.45)
    self.assertEqual(b.restless(HOMES)['amber'], 0.45)

  def test_state_round_trip(self):
    b = _backlash()
    b.emit([9], {'soot': 10.0})
    b.step(np.full(GRID.size, 0.5), np.zeros(GRID.size), HOMES, NAMES)
    other = _backlash()
    other.set_state(b.get_state())
    self.assertTrue(np.array_equal(other.pressure, b.pressure))


class StrikeTest(absltest.TestCase):

  def test_pests_blight_harvests_and_plague_kills(self):
    context = game_lib.new_game('fire_and_iron')
    planet = context.planet
    civ = planet.civ
    assert civ is not None and planet.backlash is not None
    people = civ.polities['amber']
    before = people.population
    civ.strike('amber', harvest_loss=0.0, ticks=0, deaths=0.1)
    self.assertAlmostEqual(people.population, 0.9 * before)
    civ.strike('amber', harvest_loss=0.5, ticks=2, deaths=0.0)
    temps = np.full(GRID.size, 15.0)
    forest = np.zeros(GRID.size)
    blighted = civ._yield('farming', people, temps, 1.0, forest)  # pylint: disable=protected-access
    people.blight = (0.0, 0)
    healthy = civ._yield('farming', people, temps, 1.0, forest)  # pylint: disable=protected-access
    self.assertAlmostEqual(blighted, 0.5 * healthy)

  def test_steward_sees_restless_lands_and_can_send_rain(self):
    context = game_lib.new_game('great_acceleration')
    planet = context.planet
    assert planet.backlash is not None and planet.civ is not None
    planet.backlash.pressure[planet.civ.polities['basalt'].regions[0]] = 0.9
    self.assertEqual(planet.indices_payload()['restless']['Basalt clans'], 0.9)
    planet.omega = 10.0
    rains = {i.key: i for i in planet.interventions()}['cleansing_rains']
    planet.apply_intervention('Gaia', rains)
    self.assertAlmostEqual(
        planet.indices_payload()['restless']['Basalt clans'], 0.45
    )
    options = {'a': 'wait and save omega', 'b': rains.option}
    prompt = (
        'Global mean temperature 15.0 C (habitable band 5 to 25 C); sunlight'
        ' 0.72. Omega 9.0; biodiversity 0.4; stability 0.9. Restless lands'
        ' (pressure, share of the breaking point): Basalt clans 90%.'
    )
    self.assertEqual(
        mock_language_model.steward_policy(prompt, options), rains.option
    )


if __name__ == '__main__':
  absltest.main()
