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

"""Tests of the moves' motion on the map."""

from absl.testing import absltest
from examples.deep_time import game as game_lib
from examples.deep_time import motion
from examples.deep_time.web import session as session_lib
from examples.deep_time.world import data

MOTIONS = (
    'veil',
    'sprout',
    'rain',
    'dust',
    'embers',
    'seas',
    'coasts',
    'gather',
    'ring',
    'sweep',
    'herd',
    'works',
)


class MotionTest(absltest.TestCase):

  def test_every_steward_move_shows_but_waiting(self):
    for name in (
        'sandbox',
        'great_oxidation',
        'snowball',
        'great_migration',
        'savanna_fire',
        'archipelago',
        'herds',
    ):
      planet = game_lib.new_game(name).planet
      for spec in planet.interventions():
        found = motion.look(
            planet, 'steward', spec.option, session_lib.NEUTRAL_OPTIONS
        )
        if spec.key == 'wait':
          self.assertIsNone(found)
          continue
        self.assertIsNotNone(found, f'{name}: {spec.key}')
        assert found is not None
        self.assertIn(found['motion'], MOTIONS)
        self.assertLen(found['colour'], 3)
        self.assertEqual(found['key'], spec.key)

  def test_a_key_refines_its_kind_and_every_role_moves(self):
    planet = game_lib.new_game('daisyworld').planet
    clouds = motion.look(
        planet, 'steward', 'seed bright clouds (cools the planet) [4 omega]'
    )
    haze = motion.look(
        planet, 'steward', 'release a dark haze (warms the planet) [4 omega]'
    )
    assert clouds is not None and haze is not None
    self.assertEqual(clouds['motion'], haze['motion'])
    self.assertNotEqual(clouds['colour'], haze['colour'])
    for role in data.load('motion.yaml')['roles']:
      found = motion.look(planet, role, 'spread', session_lib.NEUTRAL_OPTIONS)
      assert found is not None
      self.assertIn(found['motion'], MOTIONS)
    self.assertIsNone(
        motion.look(
            planet,
            'people',
            'keep the current balance of labor',
            session_lib.NEUTRAL_OPTIONS,
        )
    )


if __name__ == '__main__':
  absltest.main()
