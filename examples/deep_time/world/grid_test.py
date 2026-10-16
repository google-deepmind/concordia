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

"""Tests for the power grid: energy as a good, sent by wire with losses."""

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model

GRID = [
    'coal_mining',
    'steam_power',
    'steelmaking',
    'metallurgy',
    'electricity',
    'railways',
    'power_grid',
]


def _game():
  context = game_lib.new_game('great_acceleration')
  planet = context.planet
  civ, trade = planet.civ, planet.trade
  assert civ is not None and trade is not None
  return context, planet, civ, trade


class GridTest(absltest.TestCase):

  def test_energy_needs_the_grid_on_both_ends(self):
    _, _, civ, trade = _game()
    civ.ledger.deposit('amber', 'energy', 30.0, 'test')
    self.assertNotIn('energy', trade.sellable('amber'))
    civ.polities['amber'].techs += GRID
    self.assertIn('energy', trade.sellable('amber'))
    self.assertNotIn('energy', trade.buyable('basalt'))
    civ.polities['basalt'].techs += GRID
    self.assertIn('energy', trade.buyable('basalt'))

  def test_energy_arrives_at_once_less_line_losses(self):
    _, _, civ, trade = _game()
    for key in ('amber', 'basalt'):
      civ.polities[key].techs += GRID
    civ.ledger.deposit('amber', 'energy', 40.0, 'test')
    before = civ.stock('basalt', 'energy')
    trade.intend('amber', 'energy', 'ask')
    trade.intend('basalt', 'energy', 'bid')
    texts = trade.step()
    sold = [t for t in texts if 'over the power grid' in t]
    self.assertLen(sold, 1, texts)
    qty = float(sold[0].split(' sell ')[1].split(' ')[0])
    self.assertAlmostEqual(
        civ.stock('basalt', 'energy') - before, qty * (1 - trade.grid_loss)
    )

  def test_the_trade_menu_leads_with_the_biggest_surplus(self):
    _, planet, civ, _ = _game()
    civ.polities['amber'].techs += GRID
    civ.ledger.deposit('amber', 'energy', 60.0, 'test')
    amber = actors_lib.PolityActor('Amber folk', 'amber', planet, 0.15)
    offers = [o for o in amber.options() if o.startswith('offer surplus')]
    self.assertEqual(offers[0], 'offer surplus energy for sale')

  def test_mock_near_its_coal_cap_buys_clean_power(self):
    sheet = (
        '10.0 people, fed 100% last generation. Storehouse: grain 20.0, wood'
        ' 5.0, charcoal 5.0, ore 5.0, iron 5.0, tools 5.0. Last generation we'
        ' burned 3.9 coal. Silver 30.0. Standing orders: none. We are bound by'
        ' coal treaties with the Amber folk: at most 4 coal a generation.'
    )
    options = {
        'a': 'put more labor into: farm the land',
        'b': 'ask to buy energy',
        'c': actors_lib.KEEP_LABOR,
    }
    self.assertEqual(
        mock_language_model.polity_policy(sheet, options), options['b']
    )


if __name__ == '__main__':
  absltest.main()
