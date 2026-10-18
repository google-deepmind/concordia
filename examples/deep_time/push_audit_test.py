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
from examples.deep_time import game as game_lib
from examples.deep_time import push_audit
from examples.deep_time.world import data


class PushAuditTest(absltest.TestCase):

  def test_describes_how_far_and_how_long(self):
    def say(temps, co2=0.0, note=''):
      return push_audit.describe(
          [push_audit.Change(temp_c=temps, co2=co2, bio=0.0)], note
      )

    self.assertEqual(
        say([-6.4, -3.9, -2.3, -1.4]),
        'cools about 6 °C next turn, fading over a few turns',
    )
    self.assertEqual(
        say([2.0, 2.1, 1.9, 1.9], 0.01, 'black petals absorb'),
        'black petals absorb; warms about 2 °C next turn, lasting',
    )
    self.assertEqual(
        say([0.1, 0.3, 0.6, 0.9], -0.09),
        'warms about 1 °C over 4 turns; lowers CO₂ by 0.09',
    )
    self.assertEqual(say([0.2, -0.1, 0.0, 0.1]), 'hardly changes the warmth')

  def test_gives_a_range_where_the_moments_measured_differ(self):
    early = push_audit.Change(
        temp_c=[-6.0, -3.5, -2.0, -1.2], co2=0.01, bio=0.0
    )
    late = push_audit.Change(
        temp_c=[-15.7, -13.6, -12.2, -11.3], co2=0.02, bio=0.0
    )
    self.assertEqual(
        push_audit.describe([early, late]),
        'cools about 6 to 16 °C next turn, lasting; raises CO₂ by up to 0.02',
    )
    self.assertEqual(push_audit.starts('great_acceleration'), (6, 20))

  def test_every_climate_move_has_a_push_measured_on_its_world(self):
    measured = data.load('pushes.yaml')
    moves = {
        i['key']: i for i in data.load('interventions.yaml')['interventions']
    }
    for name, scenario in data.load('scenarios.yaml').items():
      for key in scenario.get('interventions', []):
        if moves[key]['effect']['kind'] in push_audit.MEASURED_KINDS:
          self.assertIn(key, measured.get(name, {}), f'{name}: {key}')

  def test_a_scenarios_own_pushes_and_effects_name_its_moves(self):
    for name, scenario in data.load('scenarios.yaml').items():
      moves = set(scenario.get('interventions', []))
      for key in {**scenario.get('pushes', {}), **scenario.get('effects', {})}:
        self.assertIn(key, moves, f'{name}: {key}')

  def test_a_game_shows_the_push_measured_on_its_world(self):
    measured = data.load('pushes.yaml')
    for name in ('daisyworld', 'great_acceleration'):
      moves = {i.key: i for i in game_lib.new_game(name).planet.interventions()}
      self.assertEqual(
          moves['bright_clouds'].push, measured[name]['bright_clouds']
      )
    snowball = game_lib.new_game('snowball').planet.interventions()
    (haze,) = [i for i in snowball if i.key == 'dark_haze']
    self.assertTrue(haze.push.startswith('may tip the ice'))  # Its own words.

  def test_a_scenario_scales_a_move_to_its_world(self):
    planet = game_lib.new_game('great_acceleration').planet
    (basalt,) = [i for i in planet.interventions() if i.key == 'basalt_dust']
    self.assertEqual(basalt.effect['amount'], 60.0)
    self.assertEqual(basalt.effect['kind'], 'carbon_drawdown')

  def test_a_scripted_steward_moves_in_the_game(self):
    ash = 'bury the sky in ash (a volcanic winter, for years)'
    script = push_audit.Script(push_audit.WAIT, ash)
    _, context = push_audit._play(  # pylint: disable=protected-access
        'daisyworld',
        0,
        8,
        {push_audit.STEWARD: script},
        overrides=push_audit.RICH,
    )
    log = context.actors[push_audit.STEWARD].log
    # The move is made once, as an action the World GM resolved; then waits.
    self.assertEqual([action for _, action, _ in log].count(ash), 1)
    self.assertEqual(log[-1][1], push_audit.WAIT)
    # The winter kills the daisies; with `play_on` the world goes on.
    self.assertEqual(context.planet.outcome, 'defeat')
    self.assertEqual(context.planet.tick, 8)

  def test_the_measured_pushes_are_up_to_date(self):
    # Rerun `push_audit --write` when the physics changes.
    found = push_audit.runs('snowball')
    stored = data.load('pushes.yaml')['snowball']
    self.assertEqual(
        push_audit.pushes('snowball', found=found),
        {k: v for k, v in stored.items() if isinstance(v, str)},
    )
    self.assertEqual(push_audit.profiles(found), stored['profiles'])


if __name__ == '__main__':
  absltest.main()
