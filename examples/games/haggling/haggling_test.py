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

"""Holistic tests for haggling using puppet agents.

These tests run the full simulation with puppet agents (no LLM) to verify
that the simulation completes and scoring logic works correctly end-to-end.
"""

from absl.testing import absltest
from absl.testing import parameterized
from examples.games.haggling import simulation
from examples.games.haggling.configs import puppet as puppet_config
from concordia.language_model import no_language_model
import numpy as np


def _mock_embedder(text: str) -> np.ndarray:
  del text
  return np.ones(3)


class HagglingTest(parameterized.TestCase):
  """Tests using puppet agents for fast, deterministic verification."""

  def test_simulation_runs_to_completion(self):
    """Verifies that the simulation runs correctly with puppet agents."""
    model = no_language_model.NoLanguageModel()
    result = simulation.run_simulation(
        config=puppet_config,
        model=model,
        embedder=_mock_embedder,
    )

    self.assertIsNotNone(result)
    self.assertIn("joint_action", result)
    self.assertIn("scores", result)
    self.assertIn("players", result)
    self.assertIn("pairs", result)
    self.assertIsInstance(result["joint_action"], dict)
    self.assertIsInstance(result["scores"], dict)

  def test_puppets_choose_valid_options(self):
    """Tests that puppets choose valid price and accept/reject options."""
    model = no_language_model.NoLanguageModel()
    result = simulation.run_simulation(
        config=puppet_config,
        model=model,
        embedder=_mock_embedder,
    )

    self.assertIsNotNone(result)
    if result["joint_action"]:
      valid_prices = {"1 coin", "2 coins", "3 coins", "4 coins", "5 coins"}
      valid_accept = {"accept", "reject"}
      for action in result["joint_action"].values():
        self.assertTrue(
            action in valid_prices or action in valid_accept,
            f"Invalid action: {action}",
        )

  def test_scores_are_valid(self):
    """Tests that computed scores are valid numbers."""
    model = no_language_model.NoLanguageModel()
    result = simulation.run_simulation(
        config=puppet_config,
        model=model,
        embedder=_mock_embedder,
    )

    self.assertIsNotNone(result)
    for score in result["scores"].values():
      self.assertIsInstance(score, float)
      self.assertGreaterEqual(score, -10)
      self.assertLessEqual(score, 10)


if __name__ == "__main__":
  absltest.main()
