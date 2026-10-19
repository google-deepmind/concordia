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

import contextlib
import io

from absl.testing import absltest
from examples.deep_time import mock_language_model
from examples.deep_time import model_setup
from examples.deep_time import play_audit
from examples.deep_time import simulation


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

  def test_the_world_gm_logs_every_menu_of_a_step(self):
    sim, _ = simulation.build_simulation(
        'great_acceleration', model_setup.make_models('mock'), max_ticks=3
    )
    with contextlib.redirect_stdout(io.StringIO()):
      log = sim.play(premise='', max_steps=3)
    # Not only the last entity asked (the engine keeps one log a step).
    for role in ('Gaia', 'Amber folk', 'Cedar league'):
      menus = play_audit.menus(log, role)
      self.assertLen(menus, 3)
      self.assertNotEmpty(menus[0]['offered'])
    gaia = play_audit.menus(log, 'Gaia')[0]
    self.assertLessEqual(
        set(gaia['pushes']), {*gaia['offered'], *gaia['later']}
    )

  def test_a_bot_plays_its_role_inside_its_entity(self):
    def gaia_moves(bots):
      sim, context = simulation.build_simulation(
          'daisyworld',
          model_setup.make_models(
              'mock', mock_variant=play_audit.variant('passive', 'Gaia')
          ),
          bots=bots,
      )
      with contextlib.redirect_stdout(io.StringIO()):
        sim.play(premise='', max_steps=8)
      return [action for _, action, _ in context.actors['Gaia'].log]

    # The minds playing the game would leave the Steward passive...
    self.assertTrue(all(m == 'wait and save omega' for m in gaia_moves({})))
    # ...but the bot (the skilled mock) plays it, as its entity's act.
    skilled = model_setup.make_models('mock').fast
    moves = gaia_moves({'Gaia': skilled})
    self.assertTrue(any(m != 'wait and save omega' for m in moves))

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
