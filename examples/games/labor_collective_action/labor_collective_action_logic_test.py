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

"""Unit tests for the scoring logic in labor_collective_action."""

import datetime

from absl.testing import absltest
from absl.testing import parameterized
from concordia.components.game_master import inventory as inventory_lib
from examples.games.labor_collective_action import simulation as labor_collective_action
from concordia.language_model import no_language_model


class LaborCollectiveActionLogicTest(parameterized.TestCase):
  """Tests for the LaborStrikePayoff class."""

  def _create_payoff(
      self,
      player_names=("Alice", "Bob", "Charlie"),
      initial_wage=2.0,
      daily_expenses=0.0,
  ):
    """Create a payoff calculator for testing."""
    return labor_collective_action.LaborStrikePayoff(
        player_names=player_names,
        strike_option="join the strike",
        work_option="go to work",
        initial_wage=initial_wage,
        daily_expenses=daily_expenses,
    )

  def test_all_work(self):
    """All workers work, everyone earns the wage."""
    payoff = self._create_payoff(initial_wage=2.0)
    scores = payoff.action_to_scores({
        "Alice": "go to work",
        "Bob": "go to work",
        "Charlie": "go to work",
    })
    self.assertAlmostEqual(scores["Alice"], 2.0)
    self.assertAlmostEqual(scores["Bob"], 2.0)
    self.assertAlmostEqual(scores["Charlie"], 2.0)
    self.assertAlmostEqual(payoff.get_strike_pressure(), 0.0)

  def test_all_strike(self):
    """All workers strike, no one earns anything."""
    payoff = self._create_payoff(initial_wage=2.0)
    scores = payoff.action_to_scores({
        "Alice": "join the strike",
        "Bob": "join the strike",
        "Charlie": "join the strike",
    })
    self.assertAlmostEqual(scores["Alice"], 0.0)
    self.assertAlmostEqual(scores["Bob"], 0.0)
    self.assertAlmostEqual(scores["Charlie"], 0.0)
    self.assertAlmostEqual(payoff.get_strike_pressure(), 1.0)

  @parameterized.named_parameters(
      (
          "two_strike_one_works",
          {
              "Alice": "join the strike",
              "Bob": "go to work",
              "Charlie": "join the strike",
          },
          {"Alice": 0.0, "Bob": 2.0, "Charlie": 0.0},
          2.0 / 3.0,
      ),
      (
          "one_strikes_two_work",
          {
              "Alice": "go to work",
              "Bob": "go to work",
              "Charlie": "join the strike",
          },
          {"Alice": 2.0, "Bob": 2.0, "Charlie": 0.0},
          1.0 / 3.0,
      ),
  )
  def test_mixed_choices(
      self, joint_action, expected_scores, expected_pressure
  ):
    """Workers make different choices."""
    payoff = self._create_payoff(initial_wage=2.0)
    scores = payoff.action_to_scores(joint_action)
    for player, expected in expected_scores.items():
      self.assertAlmostEqual(scores[player], expected)
    self.assertAlmostEqual(payoff.get_strike_pressure(), expected_pressure)

  def test_daily_expenses_applied(self):
    """Daily expenses are deducted from earnings."""
    payoff = self._create_payoff(initial_wage=2.0, daily_expenses=-0.5)
    scores = payoff.action_to_scores({
        "Alice": "go to work",
        "Bob": "join the strike",
        "Charlie": "go to work",
    })
    # Workers who work: 2.0 - 0.5 = 1.5
    # Workers who strike: 0.0 - 0.5 = -0.5
    self.assertAlmostEqual(scores["Alice"], 1.5)
    self.assertAlmostEqual(scores["Bob"], -0.5)
    self.assertAlmostEqual(scores["Charlie"], 1.5)

  def test_cumulative_scores(self):
    """Scores accumulate over multiple rounds."""
    payoff = self._create_payoff(initial_wage=2.0)

    # Round 1
    payoff.action_to_scores({
        "Alice": "go to work",
        "Bob": "go to work",
        "Charlie": "go to work",
    })
    # Round 2
    payoff.action_to_scores({
        "Alice": "join the strike",
        "Bob": "go to work",
        "Charlie": "join the strike",
    })

    cumulative = payoff.get_cumulative_scores()
    self.assertAlmostEqual(cumulative["Alice"], 2.0)  # 2 + 0
    self.assertAlmostEqual(cumulative["Bob"], 4.0)  # 2 + 2
    self.assertAlmostEqual(cumulative["Charlie"], 2.0)  # 2 + 0

  def test_wage_update(self):
    """Wage can be updated (e.g., when boss caves)."""
    payoff = self._create_payoff(initial_wage=1.0)

    # Round 1 at original wage
    scores1 = payoff.action_to_scores(
        {"Alice": "go to work", "Bob": "go to work", "Charlie": "go to work"}
    )
    self.assertAlmostEqual(scores1["Alice"], 1.0)

    # Boss caves, wage increases
    payoff.set_wage(2.5)

    # Round 2 at new wage
    scores2 = payoff.action_to_scores(
        {"Alice": "go to work", "Bob": "go to work", "Charlie": "go to work"}
    )
    self.assertAlmostEqual(scores2["Alice"], 2.5)

  def test_robust_action_matching(self):
    """Action matching handles variations in text."""
    payoff = self._create_payoff(initial_wage=2.0)
    scores = payoff.action_to_scores({
        "Alice": "Join The Strike",  # Different case
        "Bob": "GO TO WORK",  # All caps
        "Charlie": "join the strike today",  # Extra words
    })
    self.assertAlmostEqual(scores["Alice"], 0.0)  # Strike
    self.assertAlmostEqual(scores["Bob"], 2.0)  # Work
    self.assertAlmostEqual(scores["Charlie"], 0.0)  # Strike


class InventoryPayoffHandlerTest(parameterized.TestCase):
  """Tests for the InventoryPayoffHandler class with v2 inventory integration."""

  def _create_inventory_payoff(
      self,
      player_names=("Alice", "Bob", "Charlie"),
      boss_name="Boss",
      initial_wage=2.0,
      daily_expenses=0.0,
      pressure_threshold=0.45,
      raise_wages_option="raise wages",
      wage_increase_factor=2.0,
  ):
    """Create an inventory-backed payoff calculator for testing."""
    model = no_language_model.NoLanguageModel()
    all_player_names = list(player_names) + [boss_name]
    coin_config = inventory_lib.ItemTypeConfig(name="coin")
    initial_endowments = {name: {"coin": 0.0} for name in all_player_names}

    inventory = inventory_lib.Inventory(
        model=model,
        item_type_configs=[coin_config],
        player_initial_endowments=initial_endowments,
        clock_now=datetime.datetime.now,
    )

    return labor_collective_action.InventoryPayoffHandler(
        inventory=inventory,
        player_names=player_names,
        boss_name=boss_name,
        strike_option="join the strike",
        work_option="go to work",
        initial_wage=initial_wage,
        daily_expenses=daily_expenses,
        pressure_threshold=pressure_threshold,
        raise_wages_option=raise_wages_option,
        wage_increase_factor=wage_increase_factor,
    )

  def test_inventory_coin_tracking(self):
    """Coins accumulate in inventory after each round."""
    payoff = self._create_inventory_payoff(initial_wage=2.0)

    # Round 1: all work
    payoff.action_to_scores({
        "Alice": "go to work",
        "Bob": "go to work",
        "Charlie": "go to work",
    })

    # Check inventory directly
    cumulative = payoff.get_cumulative_scores()
    self.assertAlmostEqual(cumulative["Alice"], 2.0)
    self.assertAlmostEqual(cumulative["Bob"], 2.0)
    self.assertAlmostEqual(cumulative["Charlie"], 2.0)

  def test_inventory_multi_round_accumulation(self):
    """Inventory tracks cumulative coins across rounds."""
    payoff = self._create_inventory_payoff(initial_wage=2.0)

    # Round 1
    payoff.action_to_scores({
        "Alice": "go to work",
        "Bob": "join the strike",
        "Charlie": "go to work",
    })
    # Round 2
    payoff.action_to_scores({
        "Alice": "join the strike",
        "Bob": "go to work",
        "Charlie": "go to work",
    })

    cumulative = payoff.get_cumulative_scores()
    self.assertAlmostEqual(cumulative["Alice"], 2.0)  # 2 + 0
    self.assertAlmostEqual(cumulative["Bob"], 2.0)  # 0 + 2
    self.assertAlmostEqual(cumulative["Charlie"], 4.0)  # 2 + 2

  def test_boss_raise_wages_action(self):
    """Boss 'raise wages' action triggers wage update for FUTURE rounds."""
    payoff = self._create_inventory_payoff(
        initial_wage=1.0, wage_increase_factor=2.5
    )

    # Initial wage should be 1.0
    self.assertAlmostEqual(payoff.current_wage, 1.0)

    # Round 1: workers work, boss raises wages
    # Wage update happens AFTER scores, so workers earn at OLD wage this round
    scores = payoff.action_to_scores({
        "Alice": "go to work",
        "Bob": "go to work",
        "Charlie": "go to work",
        "Boss": "raise wages",
    })

    # Wage should now be 2.5 (for future rounds)
    self.assertAlmostEqual(payoff.current_wage, 2.5)

    # Workers earned at OLD wage for this round (1.0)
    self.assertAlmostEqual(scores["Alice"], 1.0)

    # Round 2: workers now earn at the new wage
    scores2 = payoff.action_to_scores({
        "Alice": "go to work",
        "Bob": "go to work",
        "Charlie": "go to work",
    })
    self.assertAlmostEqual(scores2["Alice"], 2.5)

  def test_boss_hold_firm_no_wage_change(self):
    """Boss 'hold firm' action does not change wages."""
    payoff = self._create_inventory_payoff(initial_wage=1.2)

    # Boss holds firm
    payoff.action_to_scores({
        "Alice": "go to work",
        "Bob": "go to work",
        "Charlie": "go to work",
        "Boss": "hold firm",
    })

    # Wage should remain unchanged
    self.assertAlmostEqual(payoff.current_wage, 1.2)

  @parameterized.named_parameters(
      ("exact_match", "raise wages"),
      ("different_case", "Raise Wages"),
      ("with_prefix", "I will raise wages"),
      ("all_caps", "RAISE WAGES"),
  )
  def test_boss_action_matching_variations(self, boss_action):
    """Boss action matching handles text variations."""
    payoff = self._create_inventory_payoff(
        initial_wage=1.0, wage_increase_factor=2.0
    )

    payoff.action_to_scores({
        "Alice": "go to work",
        "Boss": boss_action,
    })

    # Wage should have increased
    self.assertAlmostEqual(payoff.current_wage, 2.0)

  def test_complete_gameplay_loop(self):
    """Full gameplay loop: strike pressure → boss caves → higher wages NEXT round."""
    payoff = self._create_inventory_payoff(
        initial_wage=1.2, daily_expenses=-0.5, wage_increase_factor=2.0
    )

    # Day 1: Workers go to work at low wage
    scores1 = payoff.action_to_scores({
        "Alice": "go to work",
        "Bob": "go to work",
        "Charlie": "go to work",
        "Boss": "hold firm",
    })
    # Each worker: 1.2 - 0.5 = 0.7
    self.assertAlmostEqual(scores1["Alice"], 0.7)
    self.assertAlmostEqual(payoff.get_strike_pressure(), 0.0)

    # Day 2: All workers strike
    scores2 = payoff.action_to_scores({
        "Alice": "join the strike",
        "Bob": "join the strike",
        "Charlie": "join the strike",
        "Boss": "hold firm",
    })
    # Each striker: 0 - 0.5 = -0.5
    self.assertAlmostEqual(scores2["Alice"], -0.5)
    self.assertAlmostEqual(payoff.get_strike_pressure(), 1.0)

    # Day 3: Boss caves but workers still strike - they earn at OLD wage (0)
    # Wage increase is for FUTURE rounds, preserving solidarity incentive
    scores3 = payoff.action_to_scores({
        "Alice": "join the strike",
        "Bob": "join the strike",
        "Charlie": "join the strike",
        "Boss": "raise wages",
    })
    # Strikers still get nothing this round
    self.assertAlmostEqual(scores3["Alice"], -0.5)
    # Wage should now be 2.4 for future rounds
    self.assertAlmostEqual(payoff.current_wage, 2.4)

    # Day 4: Workers return to work at the new higher wage
    scores4 = payoff.action_to_scores({
        "Alice": "go to work",
        "Bob": "go to work",
        "Charlie": "go to work",
    })
    # Each worker: 2.4 - 0.5 = 1.9
    self.assertAlmostEqual(scores4["Alice"], 1.9)

    # Check cumulative from inventory
    cumulative = payoff.get_cumulative_scores()
    # Alice: 0.7 - 0.5 - 0.5 + 1.9 = 1.6
    self.assertAlmostEqual(cumulative["Alice"], 1.6)


if __name__ == "__main__":
  absltest.main()
