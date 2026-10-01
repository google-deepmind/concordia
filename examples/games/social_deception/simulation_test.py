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

"""Tests for Social Deception simulation."""

from typing import Any, Collection, Mapping, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from examples.games.social_deception import simulation
from examples.games.social_deception.configs import puppet as puppet_config
from concordia.language_model import language_model
import numpy as np


class MockLanguageModel(language_model.LanguageModel):

  def __init__(self):
    self._kill_target_index = 0

  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    is_named = "Alice" in prompt or "Bob" in prompt
    target_p0 = "Alice" if is_named else "Player_0"
    target_p1 = "Bob" if is_named else "Player_1"
    target_p2 = "Charlie" if is_named else "Player_2"
    target_p3 = "David" if is_named else "Player_3"
    target_p4 = "Eve" if is_named else "Player_4"

    if "Respond with 'Ack'" in prompt:
      return "Ack"
    elif "Choose action: nominate [player] or pass" in prompt:
      return "pass because I have no info"
    elif "Choose an action and respond in JSON format" in prompt:
      return "pass because I have nothing to say"
    elif "Vote on nomination of" in prompt:
      return "no"
    elif "Please select one player to kill." in prompt:
      targets = [target_p0, target_p1, target_p2, target_p3, target_p4]
      target = targets[self._kill_target_index]
      self._kill_target_index = (self._kill_target_index + 1) % len(targets)
      return f"I want to kill {target}"
    elif "Please choose one alive player (not yourself)" in prompt:
      return target_p2
    elif "Please select one player to learn their character." in prompt:
      return target_p1
    elif "Please select one alive player to poison." in prompt:
      return target_p1
    elif (
        "Please select two players." in prompt
        or "Please select exactly two players." in prompt
    ):
      return f"I choose {target_p1} and {target_p2}"
    elif "Choose action:" in prompt and "pass" in prompt:
      return "pass because I have no info"
    elif (
        "You killed" in prompt
        or "You protected" in prompt
        or "That player is the" in prompt
        or "You sense" in prompt
        or "Your choice is noted" in prompt
    ):
      return "Ack"
    else:
      return (
          "I am a player in Social Deception. I am trying to figure out"
          " the roles."
      )

  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    return 0, responses[0], {}


def dummy_embedder(_: str) -> np.ndarray:
  return np.zeros(384)


class TestSetupWithMockLlm(parameterized.TestCase):

  def test_setup(self):
    model = MockLanguageModel()
    player_names = ["Player_0", "Player_1", "Player_2", "Player_3", "Player_4"]

    res = simulation.run_simulation(
        model=model,
        embedder=dummy_embedder,
        player_names=player_names,
        day_to_play_through=1,
    )
    self.assertIsNotNone(res)
    self.assertIn("structured_log", res)

  def test_puppet_simulation(self):
    model = MockLanguageModel()
    res = simulation.run_simulation(
        model=model,
        embedder=dummy_embedder,
        config=puppet_config,
        day_to_play_through=1,
    )
    self.assertIsNotNone(res)
    self.assertIn("structured_log", res)

  @parameterized.parameters(5, 8, 12)
  def test_puppet_simulation_with_player_counts(self, num_players: int):
    model = MockLanguageModel()
    player_names = [f"Player_{i}" for i in range(num_players)]
    res = simulation.run_simulation(
        model=model,
        embedder=dummy_embedder,
        config=puppet_config,
        player_names=player_names,
        day_to_play_through=1,
    )
    self.assertIsNotNone(res)
    self.assertIn("structured_log", res)


if __name__ == "__main__":
  absltest.main()
