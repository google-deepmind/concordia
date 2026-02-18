# Copyright 2024 DeepMind Technologies Limited.
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

"""Unit tests for the logic in haggling_multi_item simulation."""

import random

from absl.testing import absltest
from absl.testing import parameterized
from examples.games.haggling_multi_item import simulation as haggling_multi_item


class ParseItemAndPriceTest(absltest.TestCase):

  def test_parse_valid_action(self):
    item, price = haggling_multi_item.parse_item_and_price("apple for 3 coins")
    self.assertEqual(item, "apple")
    self.assertEqual(price, 3.0)

  def test_parse_banana(self):
    item, price = haggling_multi_item.parse_item_and_price("banana for 5 coins")
    self.assertEqual(item, "banana")
    self.assertEqual(price, 5.0)

  def test_parse_case_insensitive(self):
    item, price = haggling_multi_item.parse_item_and_price("PEAR for 2 COINS")
    self.assertEqual(item, "pear")
    self.assertEqual(price, 2.0)

  def test_parse_invalid_action(self):
    item, price = haggling_multi_item.parse_item_and_price("invalid action")
    self.assertIsNone(item)
    self.assertEqual(price, 0.0)

  def test_parse_empty_string(self):
    item, price = haggling_multi_item.parse_item_and_price("")
    self.assertIsNone(item)
    self.assertEqual(price, 0.0)


class MultiItemHagglingPayoffTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "accept_apple_profitable",
          {"Alice": "apple for 3 coins", "Bob": "accept"},
          {"apple": 5.0, "banana": 6.0},
          {"apple": 1.0, "banana": 2.0},
          {"Alice": 2.0, "Bob": 2.0},
      ),
      (
          "accept_banana_profitable",
          {"Alice": "banana for 4 coins", "Bob": "accept"},
          {"apple": 5.0, "banana": 6.0},
          {"apple": 1.0, "banana": 2.0},
          {"Alice": 2.0, "Bob": 2.0},
      ),
      (
          "reject_no_score",
          {"Alice": "apple for 3 coins", "Bob": "reject"},
          {"apple": 5.0, "banana": 6.0},
          {"apple": 1.0, "banana": 2.0},
          {"Alice": 0.0, "Bob": 0.0},
      ),
      (
          "buyer_loses_money",
          {"Alice": "apple for 6 coins", "Bob": "accept"},
          {"apple": 5.0, "banana": 6.0},
          {"apple": 1.0, "banana": 2.0},
          {"Alice": -1.0, "Bob": 5.0},
      ),
      (
          "seller_loses_money",
          {"Alice": "banana for 1 coins", "Bob": "accept"},
          {"apple": 5.0, "banana": 6.0},
          {"apple": 1.0, "banana": 2.0},
          {"Alice": 5.0, "Bob": -1.0},
      ),
      (
          "break_even",
          {"Alice": "apple for 3 coins", "Bob": "accept"},
          {"apple": 3.0, "banana": 4.0},
          {"apple": 3.0, "banana": 4.0},
          {"Alice": 0.0, "Bob": 0.0},
      ),
  )
  def test_action_to_scores(
      self, joint_action, buyer_rewards, seller_costs, expected_scores
  ):
    payoff = haggling_multi_item.MultiItemHagglingPayoff(
        buyer_name="Alice",
        seller_name="Bob",
        buyer_base_reward_per_item=buyer_rewards,
        seller_base_reward_per_item=seller_costs,
    )
    actual_scores = payoff.action_to_scores(joint_action)
    for player, expected in expected_scores.items():
      self.assertAlmostEqual(actual_scores[player], expected)

  def test_missing_player_returns_zero(self):
    payoff = haggling_multi_item.MultiItemHagglingPayoff(
        buyer_name="Alice",
        seller_name="Bob",
        buyer_base_reward_per_item={"apple": 5.0},
        seller_base_reward_per_item={"apple": 1.0},
    )
    scores = payoff.action_to_scores({"Alice": "apple for 3 coins"})
    self.assertEqual(scores["Alice"], 0.0)
    self.assertEqual(scores["Bob"], 0.0)

  def test_unknown_item_returns_zero(self):
    payoff = haggling_multi_item.MultiItemHagglingPayoff(
        buyer_name="Alice",
        seller_name="Bob",
        buyer_base_reward_per_item={"apple": 5.0},
        seller_base_reward_per_item={"apple": 1.0},
    )
    scores = payoff.action_to_scores({
        "Alice": "orange for 3 coins",
        "Bob": "accept",
    })
    self.assertEqual(scores["Alice"], -3.0)
    self.assertEqual(scores["Bob"], 3.0)


class GeneratePriceOptionsTest(absltest.TestCase):

  def test_generate_options(self):
    items = ("apple", "banana")
    prices = (1, 2, 3)
    options = haggling_multi_item.generate_price_options(items, prices)
    expected = (
        "apple for 1 coins",
        "apple for 2 coins",
        "apple for 3 coins",
        "banana for 1 coins",
        "banana for 2 coins",
        "banana for 3 coins",
    )
    self.assertEqual(options, expected)


class CreatePlayerPairsTest(absltest.TestCase):

  def test_create_pairs(self):
    rng = random.Random(42)
    players = ["A", "B", "C"]
    pairs = haggling_multi_item.create_player_pairs(players, rng)
    self.assertLen(pairs, 3)
    for pair in pairs:
      self.assertLen(pair, 2)
      self.assertNotEqual(pair[0], pair[1])


if __name__ == "__main__":
  absltest.main()
