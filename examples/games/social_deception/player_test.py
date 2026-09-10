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

"""Tests for Social Deception player."""

from unittest import mock

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from examples.games.social_deception import player


class PlayerTest(absltest.TestCase):

  def test_build(self):
    model = mock.MagicMock()
    memory_bank = mock.MagicMock()

    prefab = player.SocialDeceptionPlayer()
    prefab.params = {
        "name": "Alice",
        "role": "BasicTownsfolk",
        "alignment": "good",
        "setup_intro": "You are the BasicTownsfolk.",
        "strategy": "Find the demon.",
    }

    agent = prefab.build(model, memory_bank)
    self.assertIsInstance(
        agent, entity_agent_with_logging.EntityAgentWithLogging
    )
    self.assertEqual(agent.name, "Alice")
    memory_bank.add.assert_called_once_with(
        "[observation] [SECRET GAME MASTER MESSAGE]\nYou are the"
        " BasicTownsfolk."
    )

  def test_build_without_setup_intro(self):
    model = mock.MagicMock()
    memory_bank = mock.MagicMock()

    prefab = player.SocialDeceptionPlayer()
    prefab.params = {
        "name": "Bob",
        "role": "Investigator",
        "alignment": "good",
    }

    agent = prefab.build(model, memory_bank)
    self.assertIsInstance(
        agent, entity_agent_with_logging.EntityAgentWithLogging
    )
    self.assertEqual(agent.name, "Bob")
    memory_bank.add.assert_not_called()


if __name__ == "__main__":
  absltest.main()
