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

from absl.testing import absltest
from examples.deep_time import mock_language_model
from examples.deep_time import play_audit


class PlayAuditTest(absltest.TestCase):

  def test_three_bots_play_the_main_role(self):
    rows = [
        play_audit.play_one('daisyworld', bot, seed=0, ticks=6)
        for bot in play_audit.BOTS
    ]
    by_bot = {r['bot']: r for r in rows}
    self.assertEqual(by_bot['default']['role'], 'Gaia')
    self.assertEqual(by_bot['passive']['acted'], 0)  # Never acts.
    self.assertGreater(by_bot['random']['acted'], 0)  # Acts at random.
    for row in rows:
      self.assertLen(row['pushes'], row['turns'])
      self.assertLen(row['plannable'], row['turns'])
      # The Steward's menu: what it can pay for now, and all it may plan.
      self.assertGreater(min(row['plannable']), 3)
      self.assertTrue(
          all(n <= m for n, m in zip(row['pushes'], row['plannable']))
      )
    summary = play_audit.summarize(rows)
    self.assertEqual(set(summary['wins']), set(play_audit.BOTS))
    self.assertLen(summary['stars'], 4)  # Games with 0, 1, 2 and 3 stars.
    self.assertEqual(sum(summary['stars']), 1)  # One skilled game.
    self.assertEqual(
        summary['gap_passive'],
        round(summary['wins']['default'] - summary['wins']['passive'], 3),
    )

  def test_mission_control_is_light_delays_main_role(self):
    row = play_audit.play_one('light_delay', 'random', seed=0, ticks=4)
    self.assertEqual(row['role'], 'Mission Control')
    self.assertGreater(row['turns'], 0)
    self.assertGreaterEqual(min(row['pushes']), 3)  # Measured directives.

  def test_the_bots_change_only_the_main_role(self):
    m = mock_language_model
    random_steward = m.VARIANTS[play_audit.variant('random', 'Gaia')]
    self.assertIn(m.random_steward_policy, random_steward)
    self.assertIn(m.polity_policy, random_steward)  # Skilled peoples.
    self.assertIn(m.mission_policy, random_steward)
    passive_mission = m.VARIANTS[
        play_audit.variant('passive', 'Mission Control')
    ]
    self.assertIn(m.passive_mission_policy, passive_mission)
    self.assertIn(m.colony_policy, passive_mission)  # Skilled colonists.
    self.assertIn(m.steward_policy, passive_mission)
    self.assertEqual(play_audit.variant('default', 'Gaia'), 'default')


if __name__ == '__main__':
  absltest.main()
