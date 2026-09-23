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

"""Tests for role night progression."""

from absl.testing import absltest
from examples.games.social_deception import game_master
from examples.games.social_deception import game_tracker
from examples.games.social_deception.setup import scripts


class RoleNightProgressionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.player_names = [f"Player_{i}" for i in range(5)]
    self.players = [
        game_tracker.PlayerState(
            name=name, role="Soldier", alignment=game_tracker.Alignment.GOOD
        )
        for name in self.player_names
    ]

  def _create_gm(self, roles_map):
    for name, role in roles_map.items():
      for p in self.players:
        if p.name == name:
          p.role = role
          p.perceived_role = role

    g = game_tracker.GameTracker(
        self.players, script=scripts.GameScript.BASIC_EPISTEMIC_TOWN
    )
    gm = game_master.Storyteller(g)
    gm._phase = game_master.Phase.NIGHT
    # Force night 1 but ensure characters who wake up are there
    gm._night_actors = gm._get_night_actors_for_day(1)
    gm._night_actor_index = 0
    return gm

  def test_servant_parsing_issue(self):
    roles = {
        "Player_0": "Servant",
        "Player_1": "Witness",
        "Player_2": "Demon",
    }
    gm = self._create_gm(roles)

    # 1. Demon (Player_2) turn to get bluffs (priority 301)
    acting = gm._handle_next_acting()
    self.assertEqual(acting, "Player_2")
    gm.pre_observe("[putative_event] Player_2: Ack")
    gm._handle_resolve(None)

    # 2. Witness (Player_1) turn (priority 601)
    acting = gm._handle_next_acting()
    self.assertEqual(acting, "Player_1")

    # Witness turn resolution
    gm.pre_observe("[putative_event] Player_1: Ack")
    gm._handle_resolve(None)

    # 3. Servant (Player_0) turn (priority 704)
    acting = gm._handle_next_acting()
    self.assertEqual(acting, "Player_0")

    # Simulate Servant action
    gm.pre_observe("[putative_event] Player_0: Player_1")
    gm._handle_resolve(None)

    # Check if Servant resolved correctly in GameTracker
    self.assertEqual(
        gm._grimoire.get_player("Player_0").servant_master, "Player_1"
    )

  def test_witness_double_prefix(self):
    gm = self._create_gm({"Player_1": "Witness", "Player_2": "Demon"})

    # 1. Demon (Player_2) turn to get bluffs (priority 301)
    acting = gm._handle_next_acting()
    self.assertEqual(acting, "Player_2")
    gm.pre_observe("[putative_event] Player_2: Ack")
    gm._handle_resolve(None)

    # 2. Witness (Player_1) turn (priority 601)
    acting = gm._handle_next_acting()
    self.assertEqual(acting, "Player_1")

    # Prompt for info
    gm._handle_next_action_spec()

    # Witness says "Player_1: Ack"
    gm.pre_observe("[putative_event] Player_1: Ack")
    gm._handle_resolve(None)

    # 3. Remaining players wake on Night 1 for role setup info
    for _ in range(3):
      acting = gm._handle_next_acting()
      gm.pre_observe(f"[putative_event] {acting}: Ack")
      gm._handle_resolve(None)

    # Should transition to day (Discussion)
    gm._handle_next_acting()
    self.assertEqual(gm._phase, game_master.Phase.DISCUSSION)


if __name__ == "__main__":
  absltest.main()
