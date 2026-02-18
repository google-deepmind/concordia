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

"""Tests for the logic in pub_coordination simulation."""

from absl.testing import absltest
from absl.testing import parameterized
from examples.games.pub_coordination import simulation as pub_coordination


class PubCoordinationLogicTest(parameterized.TestCase):

  def test_sample_parameters(self):
    venue_prefs = {"Pub A": ["R1"], "Pub B": ["R2"], "Pub C": ["R3"]}
    male_names = ["M1", "M2"]
    female_names = ["F1", "F2"]
    venues, people, _ = pub_coordination.sample_parameters(
        venue_preferences=venue_prefs,
        male_names=male_names,
        female_names=female_names,
        num_venues=2,
        num_people=3,
        seed=42,
    )
    self.assertLen(venues, 2)
    self.assertLen(people, 3)
    for v in venues:
      self.assertIn(v, venue_prefs)
    for p in people:
      self.assertIn(p, male_names + female_names)

  @parameterized.named_parameters(
      (
          "perfect_consensus",
          {"Alice": "Pub A", "Bob": "Pub A", "Charlie": "Pub A"},
          {"Alice": "Pub A", "Bob": "Pub A", "Charlie": "Pub A"},
          [],
          {"Alice": 1.5, "Bob": 1.5, "Charlie": 1.5},
      ),
      (
          "split_decision",
          {"Alice": "Pub A", "Bob": "Pub B", "Charlie": "Pub A"},
          {"Alice": "Pub A", "Bob": "Pub A", "Charlie": "Pub A"},
          [],
          {"Alice": 1.0, "Bob": 0.0, "Charlie": 1.0},
      ),
      (
          "closed_pub",
          {"Alice": "Pub A", "Bob": "Pub A", "Charlie": "Pub A"},
          {"Alice": "Pub A", "Bob": "Pub A", "Charlie": "Pub A"},
          ["Pub A"],
          {"Alice": 0.0, "Bob": 0.0, "Charlie": 0.0},
      ),
      (
          "partial_friendship",
          {"Alice": "Pub A", "Bob": "Pub A", "Charlie": "Pub A"},
          {"Alice": "Pub A", "Bob": "Pub A", "Charlie": "Pub A"},
          [],
          {"Alice": 1.5, "Bob": 1.5, "Charlie": 1.5},
      ),
      (
          "unknown_and_empty_choice",
          {"Alice": "Pub A", "Bob": "Unknown Pub", "Charlie": ""},
          {"Alice": "Pub A", "Bob": "Pub A", "Charlie": "Pub A"},
          [],
          {"Alice": 0.5, "Bob": 0.0, "Charlie": 0.0},
      ),
      (
          "asymmetric_split",
          {"Alice": "Pub A", "Bob": "Pub A", "Charlie": "Pub B"},
          {"Alice": "Pub A", "Bob": "Pub A", "Charlie": "Pub A"},
          [],
          {"Alice": 1.5, "Bob": 1.0, "Charlie": 0.0},
      ),
  )
  def test_action_to_scores(
      self, joint_action, preferences, closed, expected_scores
  ):
    player_names = list(joint_action.keys())

    if self._testMethodName.endswith((
        "partial_friendship",
        "asymmetric_split",
    )):
      relational_matrix = {
          "Alice": {"Alice": 1.0, "Bob": 1.0, "Charlie": 0.0},
          "Bob": {"Alice": 1.0, "Bob": 1.0, "Charlie": 1.0},
          "Charlie": {"Alice": 0.0, "Bob": 1.0, "Charlie": 1.0},
      }
    elif self._testMethodName.endswith("unknown_and_empty_choice"):
      relational_matrix = {
          a: {b: 1.0 for b in player_names} for a in player_names
      }
    else:
      relational_matrix = {
          a: {b: 1.0 for b in player_names} for a in player_names
      }

    option_multipliers = {
        v: 1.0 for v in set(joint_action.values()) | set(preferences.values())
    }
    for v in closed:
      option_multipliers[v] = 0.0

    payoff = pub_coordination.PubCoordinationPayoff(
        player_names=player_names,
        person_preferences=preferences,
        player_multipliers={},
        option_multipliers=option_multipliers,
        relational_matrix=relational_matrix,
    )
    actual_scores = payoff.action_to_scores(joint_action)
    for player, expected in expected_scores.items():
      self.assertAlmostEqual(actual_scores[player], expected)


if __name__ == "__main__":
  absltest.main()
