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
from examples.deep_time.world import planet as planet_lib
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

  def test_a_town_changes_hands_with_its_share_of_the_people(self):
    planet = _planet('great_acceleration')
    towns, civ = planet.settlements, planet.civ
    self.assertTrue(towns.conquest)
    amber = civ.polities['amber']
    basalt = civ.polities['basalt']
    shares = towns.shares('basalt')
    self.assertAlmostEqual(sum(shares.values()), 1.0)
    town = towns.places['basalt'][-1]
    before = (amber.population, basalt.population, len(towns.places['amber']))
    moved = basalt.population * shares[town.id]
    event = planet.change_hands(town.id, 'amber', stay=0.5, loyalty=0.3)
    # The town is the Amber folk's now, founded by the Basalt clans.
    self.assertIs(towns.find(town.id)[0], town)
    self.assertEqual(towns.find(town.id)[1], 'amber')
    self.assertEqual(town.polity, 'basalt')
    self.assertEqual(town.loyalty, 0.3)
    self.assertLen(towns.places['amber'], before[2] + 1)
    # Its share of the people moved; half of it stayed.
    self.assertAlmostEqual(basalt.population, before[1] - moved)
    self.assertAlmostEqual(amber.population, before[0] + 0.5 * moved)
    # News for every people, where it happened; on the map in two colours.
    self.assertIn('Amber folk take a town of the Basalt clans', event.text)
    self.assertEqual(event.where, (town.row, town.col))
    payload = towns.payload({})
    row = next(r for r in payload['settlements'] if r[1] == town.id)
    keys = [p['key'] for p in payload['polities']]
    self.assertEqual((keys[row[0]], keys[row[6]]), ('amber', 'basalt'))
    # A won town is never abandoned for want of people.
    towns.update({'amber': 0.5})
    self.assertEqual(towns.find(town.id)[1], 'amber')
    # It survives a checkpoint, founder, loyalty and all.
    state = towns.get_state()
    other = _planet('great_acceleration').settlements
    other.set_state(state)
    self.assertEqual(other.get_state(), state)
    self.assertEqual(other.find(town.id)[0].polity, 'basalt')

  def test_a_captured_capital_hands_the_rule_to_the_oldest_town(self):
    planet = _planet('great_acceleration')
    towns = planet.settlements
    capital = towns.places['cedar'][0]
    planet.change_hands(capital.id, 'basalt', stay=1.0, loyalty=0.2)
    self.assertFalse(capital.capital)
    self.assertTrue(towns.places['cedar'][0].capital)

  def test_without_conquest_towns_keep_their_old_state(self):
    towns = _planet('fire_and_iron').settlements
    self.assertFalse(towns.conquest)
    self.assertNotIn('towns', towns.get_state())

  def test_a_people_founds_a_town_by_its_move_where_it_aims(self):
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top

    planet = _planet('great_acceleration')
    towns, civ = planet.settlements, planet.civ
    amber = civ.polities['amber']
    people = actors_lib.PolityActor('Amber folk', 'amber', planet, 0.15)
    # Too few settlers yet: not offered, and the outcome says why.
    self.assertIn('settlers for one more need', planet.can_found('amber'))
    self.assertFalse(
        any(o.startswith(actors_lib.FOUND_TOWN) for o in people.options())
    )
    # But shown as a move it is saving for, with why (and plannable).
    (later,) = people.out_of_reach()
    self.assertTrue(later.startswith(actors_lib.FOUND_TOWN))
    self.assertIn('settlers for one more need', people.why_not(later))
    amber.population = 30.0
    self.assertEqual(people.out_of_reach(), [])  # Now offered instead.
    move = next(
        o for o in people.options() if o.startswith(actors_lib.FOUND_TOWN)
    )
    self.assertIn('4 grain', move)  # Its cost, told.
    self.assertTrue(people.aimable(move))
    # Aimed at the sea: no town, and why.
    sea = next(zip(*np.nonzero(~planet.tiles.land)))
    planet.set_target('Amber folk', int(sea[0]), int(sea[1]))
    self.assertIn('no dry land', planet.found_town('amber', 'Amber folk'))
    # Aimed at a free tile of its lands: the town stands there.
    lands = np.isin(planet.tiles.region, amber.regions) & planet.tiles.land
    taken = [(t.row, t.col) for v in towns.places.values() for t in v]
    spot = next(
        (int(r), int(c))
        for r, c in zip(*np.nonzero(lands & ~planet.tiles.ice))
        if all(towns._steps(r, c, tr, tc) >= 3 for tr, tc in taken)  # pylint: disable=protected-access
    )
    before = (len(towns.places['amber']), civ.stock('amber', 'grain'))
    planet.set_target('Amber folk', *spot)
    news = people.resolve(move)
    self.assertIn('Amber folk found a town', news)
    self.assertIn(spot, [(t.row, t.col) for t in towns.places['amber']])
    self.assertLen(towns.places['amber'], before[0] + 1)
    self.assertAlmostEqual(civ.stock('amber', 'grain'), before[1] - 4.0)
    # Every people hears of it.
    self.assertIn((planet_lib.CIV_EVENTS, news), people.drain_outbox())

  def test_where_conquest_is_played_growth_founds_no_town(self):
    towns = _planet('great_acceleration').settlements
    held = len(towns.places['amber'])
    towns.update({'amber': 30.0}, found=False)
    self.assertLen(towns.places['amber'], held)


if __name__ == '__main__':
  absltest.main()
