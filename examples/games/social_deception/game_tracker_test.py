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

"""Tests for the social deception game tracker component."""

from absl.testing import absltest
from examples.games.social_deception import game_tracker
from examples.games.social_deception.setup import scripts


class GameTrackerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.players = [
        game_tracker.PlayerState(
            name="Alice",
            role="Witness",
            alignment=game_tracker.Alignment.GOOD,
        ),
        game_tracker.PlayerState(
            name="Bob",
            role="Servant",
            alignment=game_tracker.Alignment.GOOD,
        ),
        game_tracker.PlayerState(
            name="Charlie",
            role="Outcast",
            alignment=game_tracker.Alignment.GOOD,
        ),
        game_tracker.PlayerState(
            name="Dave",
            role="Poisoner",
            alignment=game_tracker.Alignment.EVIL,
        ),
        game_tracker.PlayerState(
            name="Eve",
            role="Demon",
            alignment=game_tracker.Alignment.EVIL,
        ),
    ]
    self.tracker = game_tracker.GameTracker(
        self.players, script=scripts.GameScript.BASIC_EPISTEMIC_TOWN
    )

  def test_get_player(self):
    p = self.tracker.get_player("Alice")
    self.assertEqual(p.role, "Witness")
    self.assertEqual(p.alignment, game_tracker.Alignment.GOOD)

  def test_get_neighbors(self):
    neighbors = self.tracker.get_neighbors("Alice")
    self.assertLen(neighbors, 2)
    # Clockwise: Bob, Counter-Clockwise: Eve
    names = {n.name for n in neighbors}
    self.assertEqual(names, {"Bob", "Eve"})

  def test_neighbors_with_dead_player(self):
    # Kill Bob
    self.tracker.kill_player("Bob")
    neighbors = self.tracker.get_neighbors("Alice")
    self.assertLen(neighbors, 2)
    # Clockwise should skip dead Bob and reach Charlie
    names = {n.name for n in neighbors}
    self.assertEqual(names, {"Charlie", "Eve"})

  def test_is_impaired(self):
    self.assertFalse(self.tracker.is_impaired("Alice"))
    self.tracker.set_drunk("Alice", True)
    self.assertTrue(self.tracker.is_impaired("Alice"))
    self.tracker.set_drunk("Alice", False)
    self.tracker.set_poisoned("Alice", True)
    self.assertTrue(self.tracker.is_impaired("Alice"))

  def test_kill_and_resurrect(self):
    self.tracker.kill_player("Alice")
    p = self.tracker.get_player("Alice")
    self.assertEqual(p.status, game_tracker.PlayerStatus.DEAD)
    self.assertIn("Alice", self.tracker.deaths_today)

    self.tracker.resurrect_player("Alice")
    self.assertEqual(p.status, game_tracker.PlayerStatus.ALIVE)
    self.assertFalse(p.has_spent_dead_vote)

  def test_executioner_shot(self):
    # Create Executioner
    self.tracker.players["Frank"] = game_tracker.PlayerState(
        name="Frank",
        role="Executioner",
        alignment=game_tracker.Alignment.GOOD,
    )
    self.tracker.player_names.append("Frank")

    # Shot against Demon Eve kills Eve
    res = self.tracker.resolve_executioner_shot("Frank", "Eve")
    self.assertIn("Bang!", res)
    self.assertEqual(
        self.tracker.get_player("Eve").status, game_tracker.PlayerStatus.DEAD
    )

    # Shot again fails because already used
    res2 = self.tracker.resolve_executioner_shot("Frank", "Alice")
    self.assertIn("already used", res2)

  def test_innocent_nomination_trap(self):
    self.tracker.players["InnocentP"] = game_tracker.PlayerState(
        name="InnocentP",
        role="Innocent",
        alignment=game_tracker.Alignment.GOOD,
    )
    self.tracker.player_names.append("InnocentP")

    # Alice (Witness - Townsfolk) nominates Innocent -> Alice executed instantly
    res = self.tracker.nominate_player("Alice", "InnocentP")
    self.assertIn("instantly executed", res)
    self.assertEqual(
        self.tracker.get_player("Alice").status, game_tracker.PlayerStatus.DEAD
    )

  def test_saint_execution_ends_game(self):
    self.tracker.players["SaintP"] = game_tracker.PlayerState(
        name="SaintP",
        role="Saint",
        alignment=game_tracker.Alignment.GOOD,
    )
    self.tracker.player_names.append("SaintP")

    res = self.tracker.execute_player("SaintP")
    self.assertIn("Game Over", res)
    self.assertIsNotNone(self.tracker.game_over_reason)

  def test_state_serialization(self):
    state = self.tracker.get_state()
    new_tracker = game_tracker.GameTracker([])
    new_tracker.set_state(state)
    self.assertLen(new_tracker.players, len(self.tracker.players))
    self.assertEqual(
        new_tracker.get_player("Alice").role,
        self.tracker.get_player("Alice").role,
    )


if __name__ == "__main__":
  absltest.main()
