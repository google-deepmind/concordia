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

"""Tests for fisheries: a shared fish stock that sours with the seas."""

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
import numpy as np


def _game():
  context = game_lib.new_game('fire_and_iron')
  civ = context.planet.civ
  assert civ is not None
  civ.polities['cedar'].techs.append('fishing_boats')
  return context, context.planet, civ


class FisheriesTest(absltest.TestCase):

  def test_yield_follows_stock_coast_ice_and_acid(self):
    _, _, civ = _game()
    cedar = civ.polities['cedar']
    temps, forest = np.full(48, 15.0), np.zeros(48)

    def fish():
      return civ._yield('fish', cedar, temps, 1.0, forest)  # pylint: disable=protected-access

    full = fish()
    self.assertAlmostEqual(full, civ.coastal['cedar'])
    civ.fish_stock = 0.5
    self.assertAlmostEqual(fish(), 0.5 * full)
    civ.sea_ice = 0.5
    self.assertAlmostEqual(fish(), 0.25 * full)
    civ.sea_ice = 0.0
    civ.ocean_ph = civ.ph_reference - 0.1
    self.assertAlmostEqual(fish(), 0.5 * full * (1 - civ.ph_penalty * 0.1))

  def test_the_catch_depletes_and_the_seas_regrow(self):
    _, planet, civ = _game()
    civ.set_allocation('cedar', {'fishing': 1.0})
    planet.advance()
    self.assertLess(civ.fish_stock, 1.0)
    low = civ.fish_stock
    civ.set_allocation('cedar', {'farm': 1.0})
    planet.advance()
    self.assertGreater(civ.fish_stock, low)
    self.assertIn('fish stocks stand at', civ.sheet('cedar'))

  def test_fish_stocks_survive_a_checkpoint(self):
    context, planet, civ = _game()
    civ.set_allocation('cedar', {'fishing': 1.0})
    planet.advance()
    state = context.get_state()
    other, _, other_civ = _game()
    other.set_state(state)
    self.assertEqual(other_civ.fish_stock, civ.fish_stock)

  def test_hungry_mock_fishes_full_seas(self):
    sheet = (
        '10.0 people, fed 80% last generation. Storehouse: grain 2.0, wood'
        " 5.0, charcoal 5.0, ore 5.0, iron 5.0, tools 5.0. The seas' fish"
        ' stocks stand at 90% of their plenty; 60% of our land is coast.'
    )
    options = {
        'a': 'put more labor into: farm the land',
        'b': 'put more labor into: fish the coastal seas',
        'c': actors_lib.KEEP_LABOR,
    }
    policy = mock_language_model.polity_policy
    self.assertEqual(policy(sheet, options), options['b'])
    emptied = sheet.replace('90% of their plenty', '20% of their plenty')
    self.assertEqual(policy(emptied, options), options['a'])


if __name__ == '__main__':
  absltest.main()
