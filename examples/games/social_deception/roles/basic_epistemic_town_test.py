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

"""Tests for Basic Epistemic Town roles."""

from absl.testing import absltest
from absl.testing import parameterized
from examples.games.social_deception import game_tracker
from examples.games.social_deception.roles import basic_epistemic_town
from examples.games.social_deception.roles import roles


class BasicEpistemicTownRolesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.all_roles = []
    canonical_role_classes = (
        basic_epistemic_town.Witness,
        basic_epistemic_town.Researcher,
        basic_epistemic_town.Investigator,
        basic_epistemic_town.Matchmaker,
        basic_epistemic_town.Empath,
        basic_epistemic_town.Seer,
        basic_epistemic_town.Gravedigger,
        basic_epistemic_town.Guardian,
        basic_epistemic_town.Specter,
        basic_epistemic_town.Innocent,
        basic_epistemic_town.Executioner,
        basic_epistemic_town.Soldier,
        basic_epistemic_town.Mayor,
        basic_epistemic_town.Servant,
        basic_epistemic_town.Drunk,
        basic_epistemic_town.Outcast,
        basic_epistemic_town.Saint,
        basic_epistemic_town.Poisoner,
        basic_epistemic_town.Spy,
        basic_epistemic_town.Apprentice,
        basic_epistemic_town.Corruptor,
        basic_epistemic_town.Demon,
    )
    for cls in canonical_role_classes:
      self.all_roles.append(cls())
    self.tracker = game_tracker.GameTracker([])

  def test_all_roles_present(self):
    self.assertLen(self.all_roles, 22)

    # Verify alignment counts.
    self.assertLen(
        [
            r
            for r in self.all_roles
            if r.alignment == game_tracker.Alignment.GOOD
        ],
        17,
    )
    self.assertLen(
        [
            r
            for r in self.all_roles
            if r.alignment == game_tracker.Alignment.EVIL
        ],
        5,
    )

    # Verify role type counts.
    self.assertLen(
        [r for r in self.all_roles if isinstance(r, roles.Townsfolk)],
        13,
    )
    self.assertLen(
        [r for r in self.all_roles if isinstance(r, roles.Outsider)],
        4,
    )
    self.assertLen(
        [r for r in self.all_roles if isinstance(r, roles.Minion)], 4
    )
    self.assertLen([r for r in self.all_roles if isinstance(r, roles.Demon)], 1)

  def test_overhead_introductions(self):
    for role in self.all_roles:
      intro = role.overhead_introduction()
      self.assertIsInstance(intro, str)
      self.assertNotEmpty(
          intro, f"Role {role.role_name} has empty overhead intro."
      )

  def test_first_night_roles(self):
    first_night_order = [
        "Poisoner",
        "Demon",
        "Witness",
        "Researcher",
        "Investigator",
        "Matchmaker",
        "Empath",
        "Seer",
        "Servant",
        "Spy",
    ]
    first_night_roles = []
    for r in self.all_roles:
      r.player_name = "Dummy"
      dummy_p = game_tracker.PlayerState(
          name="Dummy", role=r.role_name, alignment=r.alignment
      )
      other_p = game_tracker.PlayerState(
          name="Other", role="Soldier", alignment=game_tracker.Alignment.GOOD
      )
      tracker = game_tracker.GameTracker([dummy_p, other_p])
      res = r.resolve_night_action(
          "", tracker, day_num=1, players=["Dummy", "Other"]
      )

      if res:
        first_night_roles.append((r.role_name, r.night_priority))

    first_night_roles.sort(key=lambda x: x[1])
    actual_order = [r[0] for r in first_night_roles]
    self.assertEqual(actual_order, first_night_order)

  def test_impaired_handling(self):
    # Test that Witness impaired info returns a valid string without failing
    witness = basic_epistemic_town.Witness()
    witness.player_name = "Alice"
    p1 = game_tracker.PlayerState(
        name="Alice",
        role="Witness",
        alignment=game_tracker.Alignment.GOOD,
        is_poisoned=True,
    )
    p2 = game_tracker.PlayerState(
        name="Bob",
        role="Investigator",
        alignment=game_tracker.Alignment.GOOD,
    )
    p3 = game_tracker.PlayerState(
        name="Charlie",
        role="Demon",
        alignment=game_tracker.Alignment.EVIL,
    )
    tracker = game_tracker.GameTracker([p1, p2, p3])
    res = witness.resolve_night_action(
        "", tracker, day_num=1, players=["Alice", "Bob", "Charlie"]
    )
    self.assertIn("Game Master reveals", res)

  def test_basic_townsfolk(self):
    bt = basic_epistemic_town.BasicTownsfolk()
    self.assertEqual(bt.role_name, "BasicTownsfolk")
    self.assertEqual(bt.alignment, game_tracker.Alignment.GOOD)
    self.assertEqual(bt.night_priority, 0)
    self.assertNotEmpty(bt.overhead_introduction())
    self.assertNotEmpty(bt.player_introduction())
    self.assertEqual(bt.resolve_if_drunk(self.tracker), "")
    self.assertEqual(
        bt.resolve_night_action("", self.tracker, day_num=1, players=[]), ""
    )
    self.assertEqual(
        bt.resolve_night_action("", self.tracker, day_num=2, players=[]), ""
    )


if __name__ == "__main__":
  absltest.main()
