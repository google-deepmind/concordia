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

"""Tests for the scoring logic in haggling simulation."""

import random
from absl.testing import absltest
from absl.testing import parameterized
from examples.games.haggling import simulation as haggling


class HagglingLogicTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "accept_at_3_coins",
          "Alice",
          "Bob",
          5.0,
          1.0,
          {"Alice": "3 coins", "Bob": "accept"},
          {"Alice": 2.0, "Bob": 2.0},
      ),
      (
          "accept_at_1_coin",
          "Alice",
          "Bob",
          5.0,
          1.0,
          {"Alice": "1 coin", "Bob": "accept"},
          {"Alice": 4.0, "Bob": 0.0},
      ),
      (
          "accept_at_5_coins",
          "Alice",
          "Bob",
          5.0,
          1.0,
          {"Alice": "5 coins", "Bob": "accept"},
          {"Alice": 0.0, "Bob": 4.0},
      ),
      (
          "reject_deal",
          "Alice",
          "Bob",
          5.0,
          1.0,
          {"Alice": "3 coins", "Bob": "reject"},
          {"Alice": 0.0, "Bob": 0.0},
      ),
      (
          "high_cost_seller",
          "Alice",
          "Bob",
          6.0,
          2.0,
          {"Alice": "4 coins", "Bob": "accept"},
          {"Alice": 2.0, "Bob": 2.0},
      ),
      (
          "buyer_loss",
          "Alice",
          "Bob",
          4.0,
          1.0,
          {"Alice": "5 coins", "Bob": "accept"},
          {"Alice": -1.0, "Bob": 4.0},
      ),
  )
  def test_action_to_scores(
      self,
      buyer_name,
      seller_name,
      buyer_base_reward,
      seller_base_cost,
      joint_action,
      expected_scores,
  ):
    payoff = haggling.HagglingPayoff(
        buyer_name=buyer_name,
        seller_name=seller_name,
        buyer_base_reward=buyer_base_reward,
        seller_base_reward=seller_base_cost,
    )
    actual_scores = payoff.action_to_scores(joint_action)

    for player, expected in expected_scores.items():
      self.assertAlmostEqual(
          actual_scores[player],
          expected,
          msg=f"Score mismatch for {player}",
      )

  def test_scores_to_observation_accept(self):
    payoff = haggling.HagglingPayoff(
        buyer_name="Alice",
        seller_name="Bob",
        buyer_base_reward=5.0,
        seller_base_reward=1.0,
    )
    joint_action = {"Alice": "3 coins", "Bob": "accept"}
    scores = payoff.action_to_scores(joint_action)
    observations = payoff.scores_to_observation(scores)

    self.assertIn("Alice", observations)
    self.assertIn("Bob", observations)
    self.assertIn("profit", observations["Alice"])

  def test_scores_to_observation_reject(self):
    payoff = haggling.HagglingPayoff(
        buyer_name="Alice",
        seller_name="Bob",
        buyer_base_reward=5.0,
        seller_base_reward=1.0,
    )
    joint_action = {"Alice": "3 coins", "Bob": "reject"}
    scores = payoff.action_to_scores(joint_action)
    observations = payoff.scores_to_observation(scores)

    self.assertIn("Alice", observations)
    self.assertIn("Bob", observations)
    self.assertIn("fell through", observations["Alice"])

  def test_latest_joint_action_property(self):
    payoff = haggling.HagglingPayoff(
        buyer_name="Alice",
        seller_name="Bob",
        buyer_base_reward=5.0,
        seller_base_reward=1.0,
    )
    self.assertEqual(payoff.latest_joint_action, {})

    joint_action = {"Alice": "2 coins", "Bob": "accept"}
    payoff.action_to_scores(joint_action)

    self.assertEqual(payoff.latest_joint_action, joint_action)

  def test_cumulative_payoff(self):
    cumulative = haggling.CumulativePayoff()

    payoff1 = haggling.HagglingPayoff("Alice", "Bob", 5.0, 1.0)
    scores1 = payoff1.action_to_scores({"Alice": "3 coins", "Bob": "accept"})
    cumulative.add_game_payoff(payoff1)
    cumulative.update_scores(scores1)

    payoff2 = haggling.HagglingPayoff("Alice", "Bob", 5.0, 1.0)
    scores2 = payoff2.action_to_scores({"Alice": "2 coins", "Bob": "accept"})
    cumulative.add_game_payoff(payoff2)
    cumulative.update_scores(scores2)

    totals = cumulative.get_total_scores()
    self.assertAlmostEqual(totals["Alice"], 5.0)
    self.assertAlmostEqual(totals["Bob"], 3.0)

  def test_sample_parameters(self):
    male_names = ["M1", "M2", "M3"]
    female_names = ["F1", "F2"]
    people, rng = haggling.sample_parameters(
        male_names=male_names,
        female_names=female_names,
        num_people=3,
        seed=42,
    )
    self.assertLen(people, 3)
    for p in people:
      self.assertIn(p, male_names + female_names)
    self.assertIsNotNone(rng)

  def test_create_player_pairs(self):
    rng = random.Random(42)
    players = ["Alice", "Bob", "Charlie"]
    pairs = haggling.create_player_pairs(players, rng)
    self.assertLen(pairs, 3)
    for buyer, seller in pairs:
      self.assertIn(buyer, players)
      self.assertIn(seller, players)
      self.assertNotEqual(buyer, seller)


if __name__ == "__main__":
  absltest.main()
