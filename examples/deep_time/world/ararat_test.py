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

"""Tests for New Ararat on tiles."""

import json

from absl.testing import absltest
from examples.deep_time import game as game_lib
import numpy as np


def _planet():
  return game_lib.new_game('light_delay').planet


class ColonyWorldTest(absltest.TestCase):

  def test_frozen_at_landfall_and_thawing_as_the_sky_warms(self):
    planet = _planet()
    world = planet.colony_world
    self.assertIsNotNone(world)
    self.assertTrue(world.tiles.ice.all())
    self.assertIsNone(_planet_without_colony().colony_world)
    world.colony.temp_c = 0.0
    news = world.update()
    self.assertLess(world.tiles.ice.mean(), 0.5)
    equator = world.tiles.rows // 2
    self.assertFalse(world.tiles.ice[equator].any())  # The warm belt thaws.
    self.assertTrue(world.tiles.ice[0].any())  # The poles stay icy.
    self.assertTrue(any('New Ararat' in n for n in news))

  def test_the_mine_works_the_ice_around_its_site(self):
    planet = _planet()
    world, colony = planet.colony_world, planet.colony
    self.assertEqual(world.mine_factor(), 1.0)  # No site chosen.
    colony.temp_c = 0.0
    world.update()
    icy = tuple(int(v) for v in np.argwhere(world.tiles.ice)[0])
    colony.site = icy
    self.assertGreater(
        world.mine_factor(), 1.0
    )  # Richer than the thawed landing.
    colony.site = world.landing
    self.assertEqual(world.mine_factor(), 1.0)

  def test_colonists_aim_now_and_mission_control_years_later(self):
    planet = _planet()
    colony = planet.colony
    planet.set_target('Colonists', 3, 7)
    self.assertIn('New Ararat', planet.aim_mine('Colonists'))
    self.assertEqual(colony.site, (3, 7))
    planet.set_target('Mission Control', 25, 30)
    planet.aim_mine('Mission Control', by_wire=True)
    self.assertEqual(colony.site, (3, 7))  # Not yet: light takes years.
    news = []
    for _ in range(colony.light_delay):
      news += colony.step()
    self.assertEqual(colony.site, (25, 30))
    self.assertTrue(any('survey reaches' in n for n in news))
    self.assertEqual(planet.aim_mine('Nobody'), '')  # Nothing aimed.

  def test_home_sees_the_colony_years_late_the_colonists_now(self):
    from examples.deep_time.web import session as session_lib  # pylint: disable=g-import-not-at-top

    game = session_lib.GameSession(
        'light_delay', 'play', role='Mission Control'
    )
    world = game.context.planet.colony_world
    assert world is not None
    world.colony.temp_c = 5.0
    world.colony.generation += 1
    world.update()  # A thaw the light has not yet carried home.
    game.context.publish_sheets()  # As the World GM does each step.
    home = game.map('Mission Control', world='ararat')
    self.assertTrue(all(set(row) == {'1'} for row in home['ice']))
    self.assertIn('years ago', home['title'])
    colonists = session_lib.GameSession(
        'light_delay', 'play', role='Colonists of New Ararat'
    )
    there = colonists.context.planet.colony_world
    assert there is not None
    there.colony.temp_c = 5.0
    there.update()
    colonists.context.publish_sheets()
    seen = colonists.map('Colonists of New Ararat', world='ararat')
    self.assertFalse(all(set(row) == {'1'} for row in seen['ice']))

  def test_a_survey_sets_the_mine_in_the_year_it_arrives(self):
    planet = _planet()
    colony, world = planet.colony, planet.colony_world
    colony.temp_c = 0.0
    world.update()
    icy = tuple(int(v) for v in np.argwhere(world.tiles.ice)[0])
    planet.set_target('Mission Control', *icy)
    planet.aim_mine('Mission Control', by_wire=True)
    for _ in range(colony.light_delay):
      colony.step()
    self.assertEqual(colony.site, icy)
    self.assertEqual(colony.mine_factor, world.mine_factor(icy))

  def test_the_page_maps_both_worlds_and_restores_exactly(self):
    from examples.deep_time.web import session as session_lib  # pylint: disable=g-import-not-at-top

    game = session_lib.GameSession('light_delay', 'watch')
    first = game.map(None, full=True)
    self.assertEqual(first['world'], 'ararat')
    self.assertEqual([w['key'] for w in first['worlds']], ['ararat', 'home'])
    self.assertLen(first['terrain']['terrain'], 30)
    self.assertNotEmpty(first['peoples']['settlements'])
    self.assertEqual(game.map(None, world='home')['world'], 'home')
    live = game_lib.new_game('light_delay', seed=1)
    for _ in range(5):
      live.planet.advance()
    saved = json.loads(json.dumps(live.get_state()))
    restored = game_lib.new_game('light_delay', seed=1)
    restored.set_state(saved)
    for _ in range(5):
      live.planet.advance()
      restored.planet.advance()
    self.assertEqual(
        json.dumps(live.planet.get_state(), sort_keys=True),
        json.dumps(restored.planet.get_state(), sort_keys=True),
    )


def _planet_without_colony():
  return game_lib.new_game('daisyworld').planet


if __name__ == '__main__':
  absltest.main()
