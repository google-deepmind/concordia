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

"""Holistic tests for pub_coordination using puppet agents.

These tests run the full simulation with puppet agents (no LLM) to verify
that the simulation completes and scoring logic works correctly end-to-end.
"""

from absl.testing import absltest
from absl.testing import parameterized
from examples.games.pub_coordination import simulation
from examples.games.pub_coordination.configs import puppet as puppet_config
from concordia.language_model import no_language_model
import numpy as np


def _mock_embedder(text: str) -> np.ndarray:
  del text
  return np.ones(3)


class PubCoordinationTest(parameterized.TestCase):
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
    self.assertIn("focal_scores", result)
    self.assertIn("joint_action", result)
    self.assertIn("focal_players", result)
    self.assertIsInstance(result["focal_scores"], dict)
    self.assertIsInstance(result["joint_action"], dict)

    for player in result["focal_players"]:
      self.assertIn(player, result["focal_scores"])

  def test_puppets_choose_valid_pubs(self):
    """Tests that all puppets choose valid pubs."""
    model = no_language_model.NoLanguageModel()
    result = simulation.run_simulation(
        config=puppet_config,
        model=model,
        embedder=_mock_embedder,
    )

    self.assertIsNotNone(result)
    if result["joint_action"]:
      valid_pubs = set(puppet_config.VENUE_PREFERENCES.keys())
      for _, action in result["joint_action"].items():
        self.assertIn(action, valid_pubs)


if __name__ == "__main__":
  absltest.main()
