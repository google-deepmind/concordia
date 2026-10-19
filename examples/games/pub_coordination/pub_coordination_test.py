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

from unittest import mock

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

  @parameterized.named_parameters(
      ("single_closed_game", 1, 1.0, [["Pub B"]]),
      ("multiple_open_games", 3, 0.0, [[], [], []]),
      (
          "pub_reopens_between_games",
          3,
          1.0,
          [["Pub A"], ["Pub B"], ["Pub A"]],
      ),
      (
          "final_closure_differs_from_first",
          4,
          1.0,
          [["Pub B"], ["Pub A"], ["Pub B"], ["Pub A"]],
      ),
      (
          "next_game_incomplete",
          3,
          1.0,
          [["Pub A"], ["Pub B"], ["Pub A"]],
          1,
          17,
      ),
  )
  def test_scores_and_feedback_follow_each_games_closures(
      self,
      num_games,
      closure_probability,
      expected_closures,
      completed_games=None,
      max_steps=100,
  ):
    """Scores, feedback and final results use the current game's closure."""
    closures = []
    scores_by_game = []
    feedback_by_game = []
    configure_scenes = simulation.configure_scenes
    action_to_scores = simulation.PubCoordinationPayoff.action_to_scores
    scores_to_observation = (
        simulation.PubCoordinationPayoff.scores_to_observation
    )

    def record_scenes(*args, **kwargs):
      scenes, game_closures = configure_scenes(*args, **kwargs)
      closures.extend(game_closures)
      return scenes, game_closures

    def record_scores(payoff, joint_action):
      scores = action_to_scores(payoff, joint_action)
      scores_by_game.append(dict(scores))
      return scores

    def record_feedback(payoff, scores):
      observations = scores_to_observation(payoff, scores)
      feedback_by_game.append(dict(observations))
      return observations

    with (
        mock.patch.object(simulation.random, "getrandbits", return_value=0),
        mock.patch.multiple(
            puppet_config,
            NUM_GAMES=num_games,
            PUB_CLOSED_PROBABILITY=closure_probability,
            MAX_STEPS=max_steps,
            create=True,
        ),
        mock.patch.object(
            simulation, "configure_scenes", side_effect=record_scenes
        ),
        mock.patch.object(
            simulation.PubCoordinationPayoff,
            "action_to_scores",
            new=record_scores,
        ),
        mock.patch.object(
            simulation.PubCoordinationPayoff,
            "scores_to_observation",
            new=record_feedback,
        ),
    ):
      result = simulation.run_simulation(
          config=puppet_config,
          model=no_language_model.NoLanguageModel(),
          embedder=_mock_embedder,
      )

    assert result is not None
    self.assertEqual(closures, expected_closures)
    completed_games = num_games if completed_games is None else completed_games
    self.assertLen(scores_by_game, completed_games + 1)
    self.assertLen(feedback_by_game, completed_games)
    expected_scores_by_game = []
    for game, closed_venues in enumerate(closures[:completed_games]):
      expected_scores = {"Alice": 0.5, "Bob": 1.0, "Charlie": 1.0}
      for player, favorite in puppet_config.PERSON_PREFERENCES.items():
        if favorite in closed_venues:
          expected_scores[player] = 0.0
      expected_scores_by_game.append(expected_scores)
      self.assertEqual(scores_by_game[game], expected_scores)
      for player, favorite in puppet_config.PERSON_PREFERENCES.items():
        if favorite in closed_venues:
          expected_feedback = f"{player} went to a closed pub."
        else:
          expected_feedback = (
              f"{player} watched the game at their favorite pub."
          )
        self.assertIn(expected_feedback, feedback_by_game[game][player])
    self.assertEqual(scores_by_game[-1], expected_scores_by_game[-1])
    self.assertEqual(result["focal_scores"], expected_scores_by_game[-1])


if __name__ == "__main__":
  absltest.main()
