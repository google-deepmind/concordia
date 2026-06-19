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

"""Unit tests for StateFormationPayoff class.

Tests the sigmoid-based scoring logic where:
- Players provide activity proportions (free_time, farming, warrior)
- Proportions are transformed via sigmoid functions
- Defense and agriculture act as binary gates
- Score = defense_gate * agriculture_gate * free_time_value
"""

from absl.testing import absltest
from absl.testing import parameterized
from examples.games.state_formation import simulation as state_formation


class StateFormationPayoffTest(parameterized.TestCase):
  """Tests for the StateFormationPayoff scoring logic."""

  def _create_payoff(self, player_names=None):
    """Create a payoff instance for testing."""

    if player_names is None:
      player_names = ["Elder A", "Elder B"]

    return state_formation.StateFormationPayoff(
        player_names=player_names,
        activity_options=("free time", "farming", "training as a warrior"),
        village_a_name="Village A",
        village_b_name="Village B",
        village_assignments={"Elder A": "Village A", "Elder B": "Village B"},
        defense_threshold=0.25,
        starvation_threshold=0.1,
        free_time_reward=1.0,
        farming_reward=0.5,
        warrior_reward=0.5,
        starvation_penalty=-5.0,
        raid_success_penalty=-3.0,
    )

  def test_proportions_parsed_from_action(self):
    """Test that activity proportions are extracted from action text."""
    payoff = self._create_payoff()
    joint_action = {
        "Elder A": "farming: 0.5, training as a warrior: 0.3, free time: 0.2",
        "Elder B": "farming: 0.4, training as a warrior: 0.4, free time: 0.2",
    }
    _ = payoff.action_to_scores(joint_action)

    proportions = payoff.latest_joint_action
    self.assertIn("Elder A", proportions)
    self.assertIn("Elder B", proportions)
    # Verify proportions are dict of floats
    self.assertIsInstance(proportions["Elder A"], dict)
    self.assertIn("farming", proportions["Elder A"])

  def test_scores_returned_for_all_players(self):
    """Test that scores are returned for all players."""
    payoff = self._create_payoff()
    joint_action = {
        "Elder A": "farming: 0.4, training as a warrior: 0.3, free time: 0.3",
        "Elder B": "farming: 0.4, training as a warrior: 0.3, free time: 0.3",
    }
    scores = payoff.action_to_scores(joint_action)

    self.assertIn("Elder A", scores)
    self.assertIn("Elder B", scores)
    self.assertIsInstance(scores["Elder A"], float)
    self.assertIsInstance(scores["Elder B"], float)

  def test_defense_gate_blocks_score_when_low(self):
    """Test that low defense causes zero score (gate = 0)."""
    payoff = self._create_payoff()
    # All farming, no warriors - defense should fail threshold
    joint_action = {
        "Elder A": "farming: 1.0, training as a warrior: 0.0, free time: 0.0",
        "Elder B": "farming: 1.0, training as a warrior: 0.0, free time: 0.0",
    }
    scores = payoff.action_to_scores(joint_action)

    # With 0 warriors, defense gate should be 0, so all scores are 0
    self.assertEqual(scores["Elder A"], 0.0)
    self.assertEqual(scores["Elder B"], 0.0)

  def test_agriculture_gate_blocks_score_when_low(self):
    """Test that low farming causes zero score (gate = 0)."""
    payoff = self._create_payoff()
    # All warriors, no farming - agriculture should fail threshold
    joint_action = {
        "Elder A": "farming: 0.0, training as a warrior: 1.0, free time: 0.0",
        "Elder B": "farming: 0.0, training as a warrior: 1.0, free time: 0.0",
    }
    scores = payoff.action_to_scores(joint_action)

    # With 0 farming, agriculture gate should be 0, so all scores are 0
    self.assertEqual(scores["Elder A"], 0.0)
    self.assertEqual(scores["Elder B"], 0.0)

  def test_balanced_allocation_gives_positive_scores(self):
    """Test that balanced allocation gives positive scores."""
    payoff = self._create_payoff()
    # Balanced allocation should pass both gates
    joint_action = {
        "Elder A": (
            "farming: 0.33, training as a warrior: 0.33, free time: 0.34"
        ),
        "Elder B": (
            "farming: 0.33, training as a warrior: 0.33, free time: 0.34"
        ),
    }
    scores = payoff.action_to_scores(joint_action)

    # Both gates should pass, and free time > 0, so scores > 0
    self.assertGreater(scores["Elder A"], 0.0)
    self.assertGreater(scores["Elder B"], 0.0)

  def test_cumulative_scores_accumulate(self):
    """Test that cumulative scores accumulate across rounds."""
    payoff = self._create_payoff()

    # Round 1
    joint_action1 = {
        "Elder A": (
            "farming: 0.33, training as a warrior: 0.33, free time: 0.34"
        ),
        "Elder B": (
            "farming: 0.33, training as a warrior: 0.33, free time: 0.34"
        ),
    }
    scores1 = payoff.action_to_scores(joint_action1)

    # Round 2
    joint_action2 = {
        "Elder A": (
            "farming: 0.33, training as a warrior: 0.33, free time: 0.34"
        ),
        "Elder B": (
            "farming: 0.33, training as a warrior: 0.33, free time: 0.34"
        ),
    }
    scores2 = payoff.action_to_scores(joint_action2)

    cumulative = payoff.get_cumulative_scores()

    # Cumulative should be sum of both rounds
    expected_a = scores1["Elder A"] + scores2["Elder A"]
    self.assertAlmostEqual(cumulative["Elder A"], expected_a, places=5)

  def test_scores_to_observation_returns_descriptions(self):
    """Test that observation strings are generated."""
    payoff = self._create_payoff()
    joint_action = {
        "Elder A": "farming: 0.5, training as a warrior: 0.3, free time: 0.2",
        "Elder B": "farming: 0.3, training as a warrior: 0.4, free time: 0.3",
    }
    scores = payoff.action_to_scores(joint_action)
    observations = payoff.scores_to_observation(scores)

    self.assertIn("Elder A", observations)
    self.assertIn("Elder B", observations)
    # Observations should contain season outcome info
    self.assertIn("season outcome", observations["Elder A"])
    self.assertIn("allocated", observations["Elder A"])

  def test_default_proportions_when_no_numbers(self):
    """Test that equal default proportions are used for unparseable text."""
    payoff = self._create_payoff()
    # Action without numbers should default to equal split
    joint_action = {"Elder A": "I decide to farm", "Elder B": "I will train"}
    _ = payoff.action_to_scores(joint_action)

    proportions = payoff.latest_joint_action["Elder A"]
    # Default should be 1/3 each
    self.assertAlmostEqual(proportions["farming"], 1.0 / 3.0, places=5)
    self.assertAlmostEqual(
        proportions["training as a warrior"], 1.0 / 3.0, places=5
    )
    self.assertAlmostEqual(proportions["free time"], 1.0 / 3.0, places=5)

  def test_treaty_enables_resource_pooling(self):
    """Test that treaty allows villages to pool agricultural resources."""
    payoff = self._create_payoff()
    payoff.set_treaty(True)

    # Village A has all farming, Village B has all warriors
    # Without treaty, B would starve. With treaty, B uses A's farming.
    joint_action = {
        "Elder A": "farming: 0.8, training as a warrior: 0.1, free time: 0.1",
        "Elder B": "farming: 0.1, training as a warrior: 0.8, free time: 0.1",
    }
    scores = payoff.action_to_scores(joint_action)

    # With treaty, Elder B should have non-zero score (uses Village A farming)
    # Defense gate should pass (overall warrior avg > threshold)
    # Agriculture gate should pass (uses max across villages due to treaty)
    self.assertGreater(scores["Elder B"], 0.0)


class ProportionParsingTest(parameterized.TestCase):
  """Tests for proportion parsing edge cases."""

  def _create_payoff(self):
    """Create a payoff instance for testing."""
    return state_formation.StateFormationPayoff(
        player_names=["Elder A"],
        activity_options=("free time", "farming", "training as a warrior"),
        village_a_name="Village A",
        village_b_name="Village B",
        village_assignments={"Elder A": "Village A"},
    )

  @parameterized.named_parameters(
      (
          "colon_format",
          "farming: 0.5, training as a warrior: 0.3, free time: 0.2",
      ),
      (
          "percent_format",
          "farming 50%, training as a warrior 30%, free time 20%",
      ),
      ("mixed_format", "50% farming, 30 training as a warrior, free time: 0.2"),
  )
  def test_various_formats_parsed(self, action_text):
    """Test that various action text formats are parsed."""
    payoff = self._create_payoff()
    joint_action = {"Elder A": action_text}
    _ = payoff.action_to_scores(joint_action)

    proportions = payoff.latest_joint_action["Elder A"]
    # Should have parsed to non-default values
    total = sum(proportions.values())
    self.assertAlmostEqual(total, 1.0, places=5)


if __name__ == "__main__":
  absltest.main()
