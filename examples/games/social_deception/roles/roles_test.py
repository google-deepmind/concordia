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

"""Tests for Social Deception roles and night order."""

from absl.testing import absltest
from examples.games.social_deception.roles import basic_epistemic_town

# Canonical night order for Basic Epistemic Town roles
BASIC_EPISTEMIC_TOWN_NIGHT_ORDER = [
    "Apprentice",
    "Poisoner",
    "Guardian",
    "Demon",
    "Specter",
    "Witness",
    "Researcher",
    "Investigator",
    "Matchmaker",
    "Empath",
    "Seer",
    "Gravedigger",
    "Servant",
    "Spy",
]


class RolesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.all_roles = [
        basic_epistemic_town.Witness(),
        basic_epistemic_town.Researcher(),
        basic_epistemic_town.Investigator(),
        basic_epistemic_town.Matchmaker(),
        basic_epistemic_town.Empath(),
        basic_epistemic_town.Seer(),
        basic_epistemic_town.Gravedigger(),
        basic_epistemic_town.Guardian(),
        basic_epistemic_town.Specter(),
        basic_epistemic_town.Innocent(),
        basic_epistemic_town.Executioner(),
        basic_epistemic_town.Soldier(),
        basic_epistemic_town.Mayor(),
        basic_epistemic_town.Servant(),
        basic_epistemic_town.Drunk(),
        basic_epistemic_town.Outcast(),
        basic_epistemic_town.Saint(),
        basic_epistemic_town.Poisoner(),
        basic_epistemic_town.Spy(),
        basic_epistemic_town.Apprentice(),
        basic_epistemic_town.Corruptor(),
        basic_epistemic_town.Demon(),
    ]
    self.all_role_names = set([r.role_name for r in self.all_roles])

  def test_no_duplicate_priorities(self):
    prios = [r.night_priority for r in self.all_roles if r.night_priority != 0]
    self.assertEqual(len(prios), len(set(prios)))

  def test_night_order_ranking(self):
    active_roles = [r for r in self.all_roles if r.night_priority > 0]
    active_roles.sort(key=lambda r: r.night_priority)
    actual_names = [r.role_name for r in active_roles]
    self.assertEqual(actual_names, BASIC_EPISTEMIC_TOWN_NIGHT_ORDER)


if __name__ == "__main__":
  absltest.main()
