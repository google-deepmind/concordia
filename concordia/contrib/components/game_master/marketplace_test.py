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

"""Tests for the MarketPlace's programmatic order and clearing API."""

import math

from absl.testing import absltest
from concordia.contrib.components.game_master import marketplace


def _market(market_type: str = 'clearing_house') -> marketplace.MarketPlace:
  return marketplace.MarketPlace(
      acting_player_names=['Ada', 'Bo'],
      agents=[
          marketplace.MarketplaceAgent('Ada', 'buyer', 100.0, {}, []),
          marketplace.MarketplaceAgent('Bo', 'seller', 0.0, {'iron': 5}, []),
      ],
      goods=[
          marketplace.Good('metal', 'plain', 'iron', price=7.0),
          marketplace.Good('food', 'plain', 'grain', price=2.0),
      ],
      market_type=market_type,
  )


class MarketPlaceApiTest(absltest.TestCase):

  def test_orders_clear_at_the_midpoint_and_move_goods_and_cash(self):
    market = _market()
    market.place_order('Ada', 'iron', 'bid', 10.0, 2)
    market.place_order('Bo', 'iron', 'ask', 6.0, 3)
    prices, completed, sold = market.clear()
    self.assertTrue(sold)
    self.assertEqual(prices['iron'], 8.0)
    self.assertLen(completed, 1)
    self.assertEqual(market.agent('Ada').inventory['iron'], 2)
    self.assertEqual(market.agent('Bo').inventory['iron'], 3)
    self.assertEqual(market.agent('Ada').cash, 84.0)
    self.assertEqual(market.agent('Bo').cash, 16.0)
    self.assertEqual(
        market.drain_outcomes('Ada'),
        ['You successfully BOUGHT 2 iron at $8.00 each.'],
    )
    self.assertEqual(market.drain_outcomes('Ada'), [])
    self.assertEqual(market.history, [prices])

  def test_books_empty_after_clearing_and_prices_persist(self):
    market = _market()
    market.place_order('Ada', 'iron', 'bid', 9.0, 1)
    market.place_order('Bo', 'iron', 'ask', 7.0, 1)
    first, _, _ = market.clear()
    second, completed, sold = market.clear()
    self.assertFalse(sold)
    self.assertEqual(completed, [])
    self.assertEqual(second['iron'], first['iron'])
    self.assertTrue(math.isnan(second['grain']))

  def test_unaffordable_bids_fail(self):
    market = _market()
    market.agent('Ada').cash = 1.0
    market.place_order('Ada', 'iron', 'bid', 10.0, 2)
    market.place_order('Bo', 'iron', 'ask', 6.0, 3)
    _, completed, _ = market.clear()
    self.assertEqual(completed, [])
    self.assertEqual(market.agent('Bo').inventory['iron'], 5)
    self.assertIn('not have enough cash', market.drain_outcomes('Ada')[0])

  def test_fixed_prices_sell_from_the_market_stock(self):
    market = _market('fixed_prices')
    for good in ('iron', 'grain'):
      market._goods[good].inventory = 4  # pylint: disable=protected-access
    market.place_order('Ada', 'iron', 'bid', 7.0, 3)
    prices, completed, sold = market.clear()
    self.assertTrue(sold)
    self.assertEqual(prices['iron'], 7.0)
    self.assertLen(completed, 1)
    self.assertEqual(market.agent('Ada').inventory['iron'], 3)
    self.assertEqual(market.agent('Ada').cash, 79.0)

  def test_validation(self):
    market = _market()
    with self.assertRaises(KeyError):
      market.place_order('Cy', 'iron', 'bid', 1.0, 1)
    with self.assertRaises(KeyError):
      market.place_order('Ada', 'gold', 'bid', 1.0, 1)
    with self.assertRaises(ValueError):
      market.place_order('Ada', 'iron', 'swap', 1.0, 1)
    with self.assertRaises(ValueError):
      market.place_order('Ada', 'iron', 'bid', 1.0, -1)
    with self.assertRaises(ValueError):
      market.place_order('Ada', 'iron', 'bid', 1.0, 0.5)  # pyrefly: ignore[bad-argument-type]
    with self.assertRaises(ValueError):
      market.place_order('Ada', 'iron', 'bid', float('nan'), 1)


if __name__ == '__main__':
  absltest.main()
