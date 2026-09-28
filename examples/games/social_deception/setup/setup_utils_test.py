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

"""Tests for social deception game setup utilities."""

from absl.testing import absltest
from examples.games.social_deception.setup import scripts
from examples.games.social_deception.setup import setup_utils


class SetupUtilsTest(absltest.TestCase):

  def test_generate_player_intro(self):
    script = scripts.BasicEpistemicTown()
    player_count = 7

    intro = setup_utils.generate_player_intro(
        "Matchmaker", script, player_count
    )

    self.assertIn("Social Deception Player (Good Team / Townsfolk)", intro)
    self.assertIn("- 5 Townsfolk", intro)
    self.assertIn("- 0 Outsiders", intro)
    self.assertIn("- 1 Minions", intro)
    self.assertIn("- 1 Demon", intro)
    self.assertIn("POTENTIAL SETUP MODIFIERS", intro)
    self.assertIn("- POTENTIAL SCENARIO (If Corruptor is in play):", intro)
    self.assertIn("  - 3 Townsfolk", intro)
    self.assertIn("  - 2 Outsiders", intro)
    self.assertIn("📖 SCRIPT MASTER REFERENCE & LOGICAL DEDUCTIONS", intro)

  def test_base_game_distribution_and_intros(self):
    script = scripts.BaseGame()
    expected_distributions = {
        6: (5, 1),
        7: (5, 2),
        8: (6, 2),
        9: (7, 2),
        10: (8, 2),
        11: (8, 3),
        12: (9, 3),
        13: (10, 3),
        14: (11, 3),
        15: (11, 4),
    }
    for count, (expected_t, expected_d) in expected_distributions.items():
      counts = setup_utils.get_player_count_distribution(count, script=script)
      self.assertEqual(counts["T"], expected_t)
      self.assertEqual(counts["D"], expected_d)
      self.assertEqual(counts["O"], 0)
      self.assertEqual(counts["M"], 0)

      player_names = [f"Player_{i}" for i in range(count)]
      states, modifiers = script.generate_roles(player_names)
      self.assertEmpty(modifiers)
      demons = [p for p in states if p["role"] == "Demon"]
      town = [p for p in states if p["role"] == "BasicTownsfolk"]
      self.assertLen(demons, expected_d)
      self.assertLen(town, expected_t)
      for d in demons:
        self.assertEqual(d["alignment"], "evil")
      for t in town:
        self.assertEqual(t["alignment"], "good")

    # Verify out of range counts raise ValueError
    with self.assertRaises(ValueError):
      script.generate_roles(["P1", "P2", "P3", "P4", "P5"])
    with self.assertRaises(ValueError):
      script.generate_roles([f"P{i}" for i in range(16)])

    # Verify intro generation for BasicTownsfolk in BaseGame
    intro = setup_utils.generate_player_intro("BasicTownsfolk", script, 8)
    self.assertIn("Social Deception Player (Good Team / Townsfolk)", intro)
    self.assertIn("- 6 Basic Townsfolk (Good)", intro)
    self.assertIn("- 2 Demons (Evil)", intro)


if __name__ == "__main__":
  absltest.main()
