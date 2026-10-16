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

"""Tests for rocketry: rockets, launches and weather satellites."""

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
import numpy as np

SPACE_AGE = [
    'coal_mining',
    'steam_power',
    'steelmaking',
    'metallurgy',
    'electricity',
    'glassmaking',
    'chemistry',
    'rocketry',
]


def _game():
  context = game_lib.new_game('great_acceleration')
  planet = context.planet
  civ = planet.civ
  assert civ is not None
  civ.polities['basalt'].techs += SPACE_AGE
  return context, planet, civ


class SpaceTest(absltest.TestCase):

  def test_satellites_sharpen_harvest_forecasts(self):
    _, _, civ = _game()
    people = civ.polities['basalt']
    temps = np.full(48, 15.0)
    forest = np.zeros(48)
    before = civ._yield('farming', people, temps, 1.0, forest)  # pylint: disable=protected-access
    civ.ledger.deposit('basalt', 'satellites', 100.0, 'test')
    self.assertEqual(civ.coverage(people), 1.0)
    after = civ._yield('farming', people, temps, 1.0, forest)  # pylint: disable=protected-access
    self.assertAlmostEqual(after, before * (1.0 + civ.forecast_bonus))

  def test_a_launch_puts_a_satellite_in_orbit_and_feeds_gaia(self):
    _, planet, civ = _game()
    tier = planet.instrument_tier
    civ.ledger.deposit('basalt', 'rockets', 1.0, 'test')
    civ.set_allocation('basalt', {'farm': 0.5, 'launch_pad': 0.5})
    report = planet.advance()
    texts = ' '.join(e.text for e in report.events)
    self.assertIn('Basalt clans launch their first satellite into orbit', texts)
    self.assertIn('Weather satellites circle the planet', texts)
    self.assertGreater(civ.in_orbit(), 0.9)  # One, less this tick's wear.
    self.assertGreater(planet.instrument_tier, tier)
    self.assertIn('fuel', civ.sheet('basalt'))
    self.assertIn('rockets 0.0, satellites 0.9', civ.sheet('basalt'))

  def test_rockets_are_planned_as_a_whole_chain(self):
    _, planet, civ = _game()
    shares, _ = civ.plan('basalt', 'rocket_works')
    for key in ('rocket_works', 'fuel_plant', 'steelworks', 'wire_mill'):
      self.assertGreater(shares[key], 0.0)
    basalt = actors_lib.PolityActor('Basalt clans', 'basalt', planet, 0.15)
    self.assertIn('plan a balanced chain to build rockets', basalt.options())

  def test_mock_launches_what_it_builds(self):
    sheet = (
        '10.0 people, fed 100% last generation. Storehouse: grain 20.0, wood'
        ' 5.0, charcoal 5.0, ore 5.0, iron 5.0, tools 5.0, fuel 0.0, rockets'
        ' 1.0, satellites 0.0. '
    )
    options = {
        'a': 'put more labor into: launch satellites',
        'b': 'plan a balanced chain to build rockets',
        'c': actors_lib.KEEP_LABOR,
    }
    policy = mock_language_model.polity_policy
    self.assertEqual(policy(sheet, options), options['a'])
    empty = sheet.replace('rockets 1.0', 'rockets 0.0')
    self.assertEqual(policy(empty, options), options['b'])


if __name__ == '__main__':
  absltest.main()
