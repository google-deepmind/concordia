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

"""Tests for peoples on the map."""

from absl.testing import absltest
from examples.deep_time import game as game_lib
import numpy as np


def _planet(scenario: str = 'fire_and_iron'):
  return game_lib.new_game(scenario).planet


class SettlementsTest(absltest.TestCase):

  def test_capitals_stand_in_their_homelands_and_towns_follow_numbers(self):
    planet = _planet()
    towns = planet.settlements
    for key, polity in planet.civ.polities.items():
      places = towns.places[key]
      capital = places[0]
      self.assertTrue(capital.capital)
      self.assertEqual(
          int(planet.tiles.region[capital.row, capital.col]), polity.home
      )
      for s in places:
        self.assertTrue(planet.tiles.land[s.row, s.col])
        self.assertIn(int(planet.tiles.region[s.row, s.col]), polity.regions)
      want = min(6, 1 + int(polity.population // 3.0))
      self.assertLen(places, want)
    self.assertIsNone(_planet('daisyworld').settlements)

  def test_towns_are_founded_and_abandoned_as_peoples_grow_and_starve(self):
    towns = _planet().settlements
    key = next(iter(towns.places))
    news = towns.update({key: 30.0})
    self.assertLen(towns.places[key], 6)
    self.assertTrue(any('found a town' in n for n in news))
    news = towns.update({key: 1.0})
    self.assertLen(towns.places[key], 1)
    self.assertTrue(towns.places[key][0].capital)
    self.assertTrue(any('abandon their town' in n for n in news))
    spacing = int(towns.config['spacing'])
    towns.update({key: 30.0})
    places = []
    for v in towns.places.values():
      places.extend(v)
    for i, one in enumerate(places):
      for two in places[i + 1 :]:
        self.assertGreaterEqual(
            towns._steps(one.row, one.col, two.row, two.col), spacing  # pylint: disable=protected-access
        )

  def test_a_fire_in_a_towns_woods_cuts_its_timber(self):
    planet = _planet()
    towns = planet.settlements
    covers = {k: v.copy() for k, v in planet.tile_life.covers.items()}
    key = next(iter(towns.places))
    self.assertEqual(towns.local(covers)[key], {'farming': 1.0, 'forest': 1.0})
    area = towns.catchment(key)
    covers['forests'] = np.where(
        area, covers['forests'] * 0.3, covers['forests']
    )
    local = towns.local(covers)[key]
    self.assertLess(local['forest'], 0.8)
    self.assertEqual(local['farming'], 1.0)

  def test_roads_join_towns_and_trade_routes_join_capitals(self):
    planet = _planet()
    towns = planet.settlements
    towns.update({k: 18.0 for k in towns.places})
    ends = set()
    for v in towns.places.values():
      ends.update((s.row, s.col) for s in v)
    roads = [p for kind, p in towns.routes if kind == 'road']
    trade = [p for kind, p in towns.routes if kind == 'trade']
    self.assertNotEmpty(roads)
    for path in roads:
      self.assertIn(path[0], ends)
      self.assertIn(path[-1], ends)
      self.assertTrue(all(planet.tiles.land[r, c] for r, c in path))
    self.assertLen(trade, 3)  # Three peoples, three pairs of capitals.

  def test_checkpoint_round_trip_and_public_map(self):
    planet = _planet()
    for _ in range(3):
      planet.advance()
    state = planet.settlements.get_state()
    other = _planet()
    other.settlements.set_state(state)
    self.assertEqual(other.settlements.get_state(), state)
    self.assertEqual(other.settlements.routes, planet.settlements.routes)
    payload = planet.settlements.payload({})
    self.assertLen(payload['polities'], 3)
    self.assertNotEmpty(payload['routes'])


if __name__ == '__main__':
  absltest.main()
