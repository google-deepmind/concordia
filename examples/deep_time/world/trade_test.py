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

"""Tests for trade on the MarketPlace and shipping on ledger links."""

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time.world import trade as trade_lib


def _world():
  context = game_lib.new_game('fire_and_iron')
  planet = context.planet
  assert planet.civ is not None and planet.trade is not None
  return planet, planet.civ, planet.trade


class TradeTest(absltest.TestCase):

  def test_a_sale_ships_goods_and_pays_silver(self):
    _, civ, trade = _world()
    civ.ledger.deposit('amber', 'grain', 30.0, 'test')
    grain_before = civ.stock('amber', 'grain')
    trade.intend('amber', 'grain', 'ask')
    trade.intend('basalt', 'grain', 'bid')
    events = trade.step()
    self.assertLen(events, 1)
    self.assertIn('Amber folk sell 3 grain to Basalt clans', events[0])
    self.assertIn('arrives in 2 ticks by porters', events[0])
    self.assertAlmostEqual(civ.stock('amber', 'grain'), grain_before - 3.0)
    self.assertAlmostEqual(civ.stock('amber', trade_lib.SILVER), 23.0)
    self.assertAlmostEqual(civ.stock('basalt', trade_lib.SILVER), 17.0)
    self.assertAlmostEqual(civ.ledger.in_transit('grain'), 3.0)
    # Basalt bid to refill its store, more than Amber offered: grain dearer.
    self.assertGreater(trade.prices['grain'], 1.0)
    self.assertEqual(trade.trades, 1)
    self.assertTrue(civ.ledger.is_conserved())

  def test_prices_follow_the_balance_of_orders_which_stand(self):
    _, civ, trade = _world()
    civ.ledger.deposit('amber', 'ore', 10.0, 'test')
    trade.intend('amber', 'ore', 'ask')
    trade.intend('basalt', 'tools', 'bid')
    self.assertIn('offer ore (3 more)', trade.standing('amber'))
    trade.step()
    self.assertLess(trade.prices['ore'], 2.0)  # Unsold supply.
    self.assertGreater(trade.prices['tools'], 5.0)  # Unmet demand.
    self.assertIn('offer ore (2 more)', trade.standing('amber'))
    trade.step()
    trade.step()
    self.assertIn('Standing orders: none', trade.standing('amber'))

  def test_a_new_order_on_a_good_replaces_the_old_one(self):
    _, civ, trade = _world()
    civ.ledger.deposit('amber', 'grain', 30.0, 'test')
    trade.intend('amber', 'grain', 'ask')
    trade.intend('amber', 'grain', 'bid')
    self.assertEqual(trade.intents['amber'], [('grain', 'bid', 3)])
    self.assertEqual(trade.step(), [])  # No trade with itself.

  def test_roads_carry_more_and_faster(self):
    _, civ, trade = _world()
    civ.polities['amber'].techs = ['granaries', 'writing', 'roads']
    civ.ledger.deposit('amber', 'grain', 30.0, 'test')
    civ.ledger.deposit('basalt', trade_lib.SILVER, 30.0, 'test')
    trade.intend('amber', 'grain', 'ask')
    trade.intend('basalt', 'grain', 'bid')
    events = trade.step()
    self.assertIn('by roads', events[0])
    self.assertIn('arrives in 1 ticks', events[0])

  def test_rail_beats_roads(self):
    _, civ, trade = _world()
    civ.polities['amber'].techs = ['granaries', 'writing', 'roads', 'railways']
    civ.ledger.deposit('amber', 'grain', 40.0, 'test')
    civ.ledger.deposit('basalt', trade_lib.SILVER, 30.0, 'test')
    trade.intend('amber', 'grain', 'ask')
    trade.intend('basalt', 'grain', 'bid')
    events = trade.step()
    self.assertIn('by rail', events[0])

  def test_polity_offers_surpluses_and_buys_what_it_can_afford(self):
    planet, civ, trade = _world()
    civ.ledger.deposit('cedar', 'charcoal', 5.0, 'test')
    actor = actors_lib.PolityActor('Cedar league', 'cedar', planet, 0.15)
    options = actor.options()
    self.assertIn('offer surplus charcoal for sale', options)
    self.assertNotIn('ask to buy charcoal', options)
    self.assertIn('ask to buy iron', options)
    self.assertIn(
        'offer charcoal at market',
        actor.resolve('offer surplus charcoal for sale'),
    )
    self.assertIn('offer charcoal', trade.standing('cedar'))

  def test_buyers_refill_and_tallies_count_goods_on_the_road(self):
    _, civ, trade = _world()
    civ.ledger.deposit('amber', 'charcoal', 30.0, 'test')
    trade.intend('amber', 'charcoal', 'ask')
    trade.intend('basalt', 'charcoal', 'bid')
    trade.step()  # 3 charcoal leave by porters; they arrive in 2 ticks.
    self.assertAlmostEqual(civ.incoming('basalt')['charcoal'], 3.0)
    # Without tallies Basalt counts only its store and orders the full 8.
    self.assertEqual(trade._wanted('basalt', 'charcoal'), 8)  # pylint: disable=protected-access
    civ.polities['basalt'].techs = ['granaries', 'writing']
    self.assertEqual(trade._wanted('basalt', 'charcoal'), 5)  # pylint: disable=protected-access
    census = civ.payload()['polities']['basalt']
    self.assertEqual(census['position']['charcoal'], 3.0)
    trade.intend('basalt', 'charcoal', 'bid')
    trade.step()
    self.assertEqual(trade.blind_reorders, 1)
    self.assertIsInstance(trade.bullwhip(skip=0), dict)

  def test_state_round_trip(self):
    planet, civ, trade = _world()
    civ.ledger.deposit('amber', 'grain', 10.0, 'test')
    trade.intend('amber', 'grain', 'ask')
    trade.intend('basalt', 'grain', 'bid')
    trade.step()
    state = planet.get_state()
    other, _, other_trade = _world()
    other.set_state(state)
    self.assertEqual(other_trade.payload(), trade.payload())
    self.assertEqual(other_trade.standing('amber'), trade.standing('amber'))


class PriceNewsTest(absltest.TestCase):

  def test_couriers_bring_old_prices_and_the_telegraph_fresh_ones(self):
    context = game_lib.new_game('fire_and_iron')
    civ, trade = context.planet.civ, context.planet.trade
    assert civ is not None and trade is not None
    self.assertEqual(trade.news_lag('amber'), trade.news_latency)
    start = trade.prices['ore']
    for _ in range(3):
      trade.prices['ore'] *= 2.0
      trade.history = (trade.history + [dict(trade.prices)])[-3:]
    self.assertAlmostEqual(
        trade.believed_price('amber', 'ore'), start * 2.0
    )  # Two clearings old.
    civ.polities['amber'].techs.append('telegraph')
    self.assertEqual(trade.news_lag('amber'), trade.wire_latency)
    self.assertAlmostEqual(
        trade.believed_price('amber', 'ore'), trade.prices['ore']
    )

  def test_orders_on_stale_prices_are_counted_and_checkpointed(self):
    context = game_lib.new_game('fire_and_iron')
    civ, trade = context.planet.civ, context.planet.trade
    assert civ is not None and trade is not None
    trade.history = [{**trade.prices, 'ore': trade.prices['ore'] * 3}] * 3
    civ.ledger.deposit('basalt', 'ore', 20.0, 'test')
    trade.intend('basalt', 'ore', 'ask')
    trade.step()
    self.assertGreater(trade.stale_orders, 0)
    other = game_lib.new_game('fire_and_iron')
    assert other.planet.trade is not None
    other.planet.trade.set_state(trade.get_state())
    self.assertEqual(other.planet.trade.history, trade.history)
    self.assertEqual(other.planet.trade.stale_orders, trade.stale_orders)


if __name__ == '__main__':
  absltest.main()
