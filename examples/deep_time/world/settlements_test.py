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
from examples.deep_time.world import news as news_lib
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
    towns = _planet('sandbox').settlements
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
    (later,) = [
        o for o in people.out_of_reach() if o.startswith(actors_lib.FOUND_TOWN)
    ]
    self.assertIn('settlers for one more need', people.why_not(later))
    amber.population = 30.0
    self.assertFalse(  # Now offered instead.
        any(o.startswith(actors_lib.FOUND_TOWN) for o in people.out_of_reach())
    )
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

  def test_arms_are_raised_as_labour_and_join_the_capital(self):
    planet = _planet('great_acceleration')
    towns, civ = planet.settlements, planet.civ
    # Three kinds, each from know-how with a civilian face.
    self.assertEqual(set(towns.arms), {'levies', 'riders', 'guns'})
    known = [
        r.key for r in civ.recipes_for('amber') if r.key.startswith('arm_')
    ]
    self.assertIn('arm_levies', known)  # Tools make spears from the start.
    self.assertNotIn('arm_guns', known)  # Rifles wait for steel.
    self.assertIn('arm_guns', [r.requires and r.key for r in civ.recipes])
    # Made arms join the capital's garrison and eat grain.
    civ.ledger.deposit('amber', 'levies', 3.0, 'test')
    grain = civ.stock('amber', 'grain')
    self.assertEqual(planet._muster(), [])  # pylint: disable=protected-access
    capital = next(t for t in towns.places['amber'] if t.capital)
    self.assertEqual(capital.garrison, {'levies': 3.0})
    self.assertEqual(civ.stock('amber', 'levies'), 0.0)
    self.assertAlmostEqual(civ.stock('amber', 'grain'), grain - 0.9)
    self.assertIn(
        'Our arms: 3.0 levies (strength 3.0', towns.describe_arms('amber')
    )
    self.assertEqual(towns.describe_arms('basalt'), '')  # None raised.
    # Drawn on the map: the town's strength and its arms by label.
    self.assertEqual(
        towns.payload({})['garrisons'][str(capital.id)], [3.0, {'levies': 3.0}]
    )
    # Unfed arms desert, and every people hears it.
    civ.ledger.withdraw('amber', 'grain', civ.stock('amber', 'grain'), 'test')
    (news,) = planet._muster()  # pylint: disable=protected-access
    self.assertIn('Amber folk cannot feed their arms: 1.5 levies desert', news)
    self.assertEqual(capital.garrison, {'levies': 1.5})

  def test_raising_arms_is_offered_apart_from_the_trades(self):
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top

    planet = _planet('great_acceleration')
    people = actors_lib.PolityActor('Basalt clans', 'basalt', planet, 0.15)
    options = people.options()
    raise_levies = 'put more labor into: arm levies with spears'
    self.assertIn(raise_levies, options)
    # Never among the trades it may crowd off a short menu.
    self.assertNotIn(raise_levies, people._focus())  # pylint: disable=protected-access
    self.assertIn('smelt iron', ' '.join(people._focus()))  # pylint: disable=protected-access
    # Chosen, it puts hands to arms, told like any trade.
    self.assertIn('arm levies with spears', people.resolve(raise_levies))
    self.assertGreater(
        planet.civ.polities['basalt'].allocation.get('arm_levies', 0.0), 0.0
    )

  def test_garrisons_are_kept_by_kind_and_older_saves_still_load(self):
    towns = _planet('great_acceleration').settlements
    towns.muster('amber', 'riders', 2.0)
    state = towns.get_state()
    again = _planet('great_acceleration').settlements
    again.set_state(state)
    self.assertEqual(again.arms_of('amber'), {'riders': 2.0})
    for v in state['towns'].values():  # A save from before arms had kinds.
      v[3] = 1.5
    again.set_state(state)
    self.assertEqual(
        again.arms_of('amber')['levies'], 1.5 * len(again.places['amber'])
    )

  def test_arms_belong_to_conquest_worlds_only(self):
    planet = _planet('sandbox')
    self.assertEqual(planet.settlements.arms, {})
    self.assertFalse(any(r.key.startswith('arm_') for r in planet.civ.recipes))

  def _march(self, intent):
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top

    planet = _planet('great_acceleration')
    towns = planet.settlements
    basalt = actors_lib.PolityActor('Basalt clans', 'basalt', planet, 0.15)
    # No arms yet: the march waits among the moves saved for, saying why.
    self.assertIn(actors_lib.MARCH_TAKE, basalt.out_of_reach())
    self.assertEqual(
        basalt.why_not(actors_lib.MARCH_TAKE),
        'there are no arms in their capital',
    )
    towns.muster('basalt', 'levies', 40.0)
    move = actors_lib.MARCH_RAZE if intent == 'raze' else actors_lib.MARCH_TAKE
    self.assertIn(move, basalt.options())
    self.assertTrue(basalt.aimable(move))
    town, holder = towns.nearest_town('basalt', None)
    news = basalt.resolve(move)
    self.assertIn(f'to {intent} it; they arrive in turn', news)
    self.assertIn('strong with its people and walls (odds about', news)
    outbox = basalt.drain_outbox()
    self.assertIn((planet_lib.CIV_EVENTS, news), outbox)  # Every people.
    self.assertIn((planet.war_topic(holder), news), outbox)  # Its target.
    (army,) = towns.armies
    self.assertEqual(towns.capital('basalt').garrison, {})  # It marched.
    self.assertLen(towns.payload({})['armies'], 1)  # Drawn on the map.
    told = []
    while towns.armies:
      told.extend(e.text for e in planet.advance().events)
    self.assertGreaterEqual(planet.tick, army.arrives)
    return planet, towns, town, holder, told

  def test_an_army_marches_takes_a_town_that_settles_or_rises(self):
    planet, towns, town, holder, told = self._march('take')
    self.assertEqual(towns.find(town.id)[1], 'basalt')
    self.assertTrue(
        any('Basalt clans take a town of the' in t for t in told), told
    )
    self.assertGreater(towns.strength(town), 2.0)  # It holds what it took.
    # Each side hears it at once, in its own words.
    self.assertTrue(any(t.startswith('Our army took a town') for t in told))
    self.assertTrue(
        any(t.startswith('The Basalt clans took our town') for t in told)
    )
    self.assertAlmostEqual(town.loyalty, 0.35, 1)
    # Left unheld, it grows restless, warns, and rises to its founder.
    town.garrison = {}
    told = []
    for _ in range(8):
      told.extend(e.text for e in planet.advance().events)
      if towns.find(town.id)[1] == holder:
        break
    self.assertTrue(any('is restless' in t for t in told), told)
    self.assertTrue(any('rises against the Basalt clans' in t for t in told))
    self.assertTrue(any('rose against us' in t for t in told))
    self.assertTrue(
        any(t.startswith('Our town') and 'rises next turn' in t for t in told)
    )
    self.assertEqual(towns.find(town.id)[1], holder)

  def test_an_army_razes_a_town_and_its_people_are_lost(self):
    planet = _planet('great_acceleration')
    planet, towns, town, holder, told = self._march('raze')
    self.assertIsNone(towns.find(town.id))
    self.assertTrue(any('raze a town of the' in t for t in told), told)
    # The survivors came home.
    self.assertGreater(towns.strength(towns.capital('basalt')), 0.0)

  def test_armies_on_the_march_are_saved_and_restored(self):
    planet = _planet('great_acceleration')
    towns = planet.settlements
    towns.muster('basalt', 'riders', 5.0)
    town, _ = towns.nearest_town('basalt', None)
    army, why = towns.march('basalt', town.id, 'take', planet.tick)
    self.assertEqual(why, '')
    self.assertEqual(army.per_turn, 16)  # Riders cover two legs a turn.
    again = _planet('great_acceleration').settlements
    again.set_state(towns.get_state())
    self.assertEqual(again.armies, towns.armies)
    self.assertEqual(again.upkeep('basalt'), 3.0)  # They eat on the march.

  def _courted(self):
    """A Great Acceleration with a town of the Amber folk to court."""
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top

    planet = _planet('great_acceleration')
    towns, civ = planet.settlements, planet.civ
    town, why = towns.found_at('amber', None)
    self.assertEqual(why, '')
    civ.ledger.deposit('basalt', 'grain', 60.0, 'test')
    basalt = actors_lib.PolityActor('Basalt clans', 'basalt', planet, 0.15)
    return planet, towns, civ, town, basalt

  def test_a_town_is_won_over_with_grain_and_envoys(self):
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top

    planet, towns, civ, town, basalt = self._courted()
    (move,) = [
        o for o in basalt.options() if o.startswith(actors_lib.SEND_ENVOYS)
    ]
    self.assertIn('5 grain', move)  # Its cost, told.
    self.assertTrue(basalt.aimable(move))
    # A capital holds to its own people (and the cart is not spent).
    capital = towns.capital('amber')
    grain = civ.stock('basalt', 'grain')
    planet.set_target('Basalt clans', capital.row, capital.col)
    self.assertIn('holds to its own people', basalt.resolve(move))
    self.assertEqual(civ.stock('basalt', 'grain'), grain)
    # A cart to the town: how far it is, said to both sides.
    planet.set_target('Basalt clans', town.row, town.col)
    news = basalt.resolve(move)
    self.assertIn('its regard for them is 0.15 of the 1.00 it needs', news)
    self.assertIn('85% to go', news)
    self.assertAlmostEqual(civ.stock('basalt', 'grain'), grain - 5.0)
    (heard,) = basalt.drain_outbox()
    self.assertEqual(heard[0], planet.war_topic('amber'))
    self.assertIn('to leave us (85% to go)', heard[1])
    self.assertIn('Our envoys court a town', towns.describe_war('basalt'))
    self.assertIn(
        'The Basalt clans court our town', towns.describe_war('amber')
    )
    pi = list(towns.places).index('basalt')
    self.assertEqual(
        towns.payload({})['leaning'], {str(town.id): [pi, 0.15, 1.0]}
    )
    # A hungry town warms twice as fast; a model's cart goes to the town
    # nearest to joining; and the town joins at its need, garrison and all
    # people kept (the garrison walks home).
    civ.polities['amber'].food_ratio = 0.5
    town.garrison = {'levies': 2.0}
    people = (
        civ.polities['amber'].population + civ.polities['basalt'].population
    )
    # A town hears a people's envoys once a turn: a second cart is refused,
    # and costs nothing.
    grain = civ.stock('basalt', 'grain')
    self.assertIn('heard theirs this turn already', basalt.resolve(move))
    self.assertEqual(civ.stock('basalt', 'grain'), grain)
    basalt.drain_outbox()
    told = []
    for _ in range(4):
      if towns.find(town.id)[1] == 'basalt':
        break
      town.heard.clear()  # A new turn.
      told.append(basalt.resolve(move))
    self.assertIn('twice as warm', told[0])
    self.assertIn('0.45 of the 1.00', told[0])
    self.assertIn('and it joins them', told[-1])
    self.assertEqual(towns.find(town.id)[1], 'basalt')
    outbox = basalt.drain_outbox()
    self.assertIn(planet_lib.CIV_EVENTS, [t for t, _ in outbox])
    self.assertTrue(any('win over a town of the Amber' in w for _, w in outbox))
    self.assertTrue(any('walked home' in w for _, w in outbox))
    self.assertEqual(towns.capital('amber').garrison, {'levies': 2.0})
    self.assertAlmostEqual(
        civ.polities['amber'].population + civ.polities['basalt'].population,
        people,
    )
    self.assertAlmostEqual(town.loyalty, 0.6)
    self.assertFalse(town.taken)
    # A town won over is content: it settles without arms.
    told = [e.text for e in planet.advance().events]
    self.assertFalse(any('restless' in t for t in told), told)
    self.assertAlmostEqual(town.loyalty, 0.65)

  def test_envoys_after_skipped_turns_say_the_town_cooled(self):
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top

    planet, _, _, town, basalt = self._courted()
    (move,) = [
        o for o in basalt.options() if o.startswith(actors_lib.SEND_ENVOYS)
    ]
    planet.set_target('Basalt clans', town.row, town.col)
    self.assertNotIn('cooled', basalt.resolve(move))  # The first cart.
    town.heard['basalt'] = planet.tick - 5  # Their last, five turns ago.
    planet.set_target('Basalt clans', town.row, town.col)
    told = basalt.resolve(move)
    self.assertIn('It had cooled since their envoys 5 turns ago', told)
    self.assertIn('regard fades by 0.04 each turn', told)

  def test_a_feast_answers_envoys_and_settles_a_taken_town(self):
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top

    planet, towns, civ, town, basalt = self._courted()
    amber = actors_lib.PolityActor('Amber folk', 'amber', planet, 0.15)
    civ.ledger.deposit('amber', 'grain', 60.0, 'test')

    def feast_move(people):
      return [
          o for o in people.options() if o.startswith(actors_lib.HOLD_FEAST)
      ]

    self.assertEqual(feast_move(amber), [])  # Nothing to answer yet.
    towns.court(town.id, 'basalt', 0.4)
    (move,) = feast_move(amber)
    news = amber.resolve(move)
    self.assertIn('Amber folk feast their town', news)
    self.assertIn("others' regard there falls", news)
    self.assertAlmostEqual(town.regard['basalt'], 0.25)
    (heard,) = amber.drain_outbox()
    self.assertEqual(heard[0], planet.war_topic('basalt'))
    # A town taken by arms is restless; a feast settles it some.
    planet.change_hands(town.id, 'basalt', stay=0.8, loyalty=0.35, how='take')
    self.assertTrue(town.taken)
    self.assertEqual(town.regard, {})
    self.assertEqual(towns.payload({})['restless'], {str(town.id): 0.35})
    (move,) = feast_move(basalt)
    self.assertIn('holds to them at 0.50', basalt.resolve(move))
    self.assertIn('holds to us at 0.50', towns.describe_war('basalt'))
    # Without grain: shown, saying why.
    civ.ledger.withdraw('basalt', 'grain', civ.stock('basalt', 'grain'), 't')
    self.assertEqual(feast_move(basalt), [])
    (later,) = [
        o for o in basalt.out_of_reach() if o.startswith(actors_lib.HOLD_FEAST)
    ]
    self.assertIn('which they lack', basalt.why_not(later))

  def test_a_crowded_menu_keeps_its_letters_and_its_town_moves(self):
    from unittest import mock  # pylint: disable=g-import-not-at-top
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top

    planet, towns, _, town, basalt = self._courted()
    towns.muster('basalt', 'levies', 5.0)
    towns.court(towns.capital('basalt').id, 'amber', 0.1)
    full = basalt.options()
    with mock.patch.object(actors_lib, 'MAX_OPTIONS', len(full) - 2):
      menu = basalt.options()
    self.assertLessEqual(len(menu), len(full) - 2)
    for move in (
        actors_lib.SEND_ENVOYS,
        actors_lib.HOLD_FEAST,
        actors_lib.MARCH_TAKE,
    ):
      self.assertTrue(any(o.startswith(move) for o in menu), move)
    del planet, town

  def test_several_orders_a_turn_one_from_each_town(self):
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top

    planet, towns, civ, amber_town, basalt = self._courted()
    north = planet.tiles.rows - 1
    # A second Basalt town, armed, and the capital armed too.
    second, why = towns.found_at('basalt', None)
    self.assertEqual(why, '')
    second.garrison = {'levies': 6.0}
    capital = towns.capital('basalt')
    # Unarmed, the capital's card says why it cannot march yet.
    unarmed = basalt.town_menu()[str(capital.id)]
    self.assertIn('holds no arms yet', unarmed['waits'][actors_lib.MARCH_TAKE])
    towns.muster('basalt', 'levies', 4.0)

    def at(town):
      return f'{north - town.row},{town.col}'

    # The page's menu: each town's orders (a march only where arms stand).
    menu = basalt.town_menu()
    self.assertIn(actors_lib.MARCH_TAKE, menu[str(second.id)]['orders'])
    self.assertEqual(menu[str(second.id)]['waits'], {})  # Armed: it may.
    self.assertEqual(
        menu[str(second.id)]['at'], [north - second.row, second.col]
    )
    self.assertTrue(
        any(
            o.startswith(actors_lib.SEND_ENVOYS)
            for o in menu[str(capital.id)]['orders']
        )
    )
    # The question says towns may give orders.
    from examples.deep_time.components import action_specs  # pylint: disable=g-import-not-at-top

    spec = action_specs.aimable_spec(basalt, basalt.action_spec())
    self.assertIn(actors_lib.ORDERS_HINT, spec.call_to_action)
    # One answer: the people's order, then one line per town.
    target = f'{north - amber_town.row},{amber_town.col}'
    answer = '\n'.join([
        actors_lib.KEEP_LABOR,
        f'town {at(second)}: {actors_lib.MARCH_RAZE} @ {target}',
        f'town {at(capital)}: {actors_lib.SEND_ENVOYS} [3 grain] @ {target}',
        f'town {at(capital)}: {actors_lib.HOLD_FEAST}',  # A second: refused.
        f'town 0,0: {actors_lib.FOUND_TOWN}',  # Not a town of theirs.
    ])
    self.assertEqual(basalt.understood(answer), answer)
    outcome = basalt.resolve(answer).split('\n')
    self.assertLen(outcome, 5)  # An outcome for every order.
    self.assertIn('keep their labor', outcome[0])
    self.assertIn('to raze it; they arrive in turn', outcome[1])
    self.assertIn('its regard for them is 0.15', outcome[2])
    self.assertIn('gave its order this turn already', outcome[3])
    self.assertIn('hold no town at 0,0', outcome[4])
    # The town marched with its own arms; the capital kept its own.
    (army,) = towns.armies
    self.assertEqual(army.units, {'levies': 6.0})
    self.assertEqual(second.garrison, {})
    self.assertEqual(capital.garrison, {'levies': 4.0})
    # An order written loosely is read by its words (a live mind's, c184).
    planet.advance()
    self.assertIn(
        'feast their town',
        basalt.resolve(f'town {at(capital)}: hold a feast in it'),
    )
    # A town gives only a town's orders.
    self.assertIn(
        'cannot "put more labor into: farm the land"',
        basalt.resolve(
            f'town {at(capital)}: put more labor into: farm the land'
        ),
    )

  def test_a_turns_town_orders_are_kept_for_replays(self):
    from examples.deep_time.web import session as session_lib  # pylint: disable=g-import-not-at-top
    from concordia.typing import entity as entity_lib  # pylint: disable=g-import-not-at-top

    rule = session_lib.ReplayRule(
        [[
            3,
            'march to take a town',
            [4, 5],
            ['town 1,2: hold a feast in a town'],
        ]],
        clock=lambda: 3,
    )
    spec = entity_lib.free_action_spec(call_to_action='?')
    rule.moves = lambda: ['march to take a town']
    self.assertEqual(
        rule(spec, {}),
        'march to take a town @ 4,5\ntown 1,2: hold a feast in a town',
    )

  def test_peace_is_made_and_broken_by_a_march_and_remembered(self):
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top
    from examples.deep_time.prefabs import polity as polity_prefab  # pylint: disable=g-import-not-at-top

    planet = _planet('great_acceleration')
    context = game_lib.GameContext.__new__(game_lib.GameContext)
    towns, treaties = planet.settlements, planet.treaties
    basalt = actors_lib.PolityActor('Basalt clans', 'basalt', planet, 0.15)
    # Any two peoples may offer peace (no diplomacy needed), once either
    # holds arms.
    offer = 'propose a peace pact to the Amber folk'
    self.assertNotIn(offer, basalt.options())
    towns.muster('amber', 'levies', 2.0)
    self.assertIn(offer, basalt.options())
    self.assertIn('neither marches on the', basalt.resolve(offer))
    settled = planet.answer_treaty(
        'amber', 'Reply to Basalt clans (peace pact): accept', True
    )
    self.assertIsNotNone(settled)
    self.assertTrue(treaties.at_peace('basalt', 'amber'))
    del context
    # A march on the partner's town breaks it, for all to see.
    towns.muster('basalt', 'levies', 40.0)
    town, _ = towns.nearest_town('basalt', None)
    if town.polity != 'amber':
      planet.set_target('Basalt clans', *self._amber_town(towns))
    news = basalt.resolve(actors_lib.MARCH_TAKE)
    self.assertIn('break their peace pact with the Amber folk', news)
    self.assertFalse(treaties.at_peace('basalt', 'amber'))
    self.assertTrue(treaties.embargoed('basalt', 'amber'))
    names = {k: p.name for k, p in planet.civ.polities.items()}
    self.assertEqual(
        treaties.describe_grievances('amber', names),
        'We hold grievances against the Basalt clans (broke their peace pact).',
    )
    # Taking the town adds its own grievance; saves keep them.
    while towns.armies:
      planet.advance()
    self.assertIn('took a town', treaties.describe_grievances('amber', names))
    again = _planet('great_acceleration').treaties
    again.set_state(treaties.get_state())
    self.assertEqual(again.grievances, treaties.grievances)
    del polity_prefab

  def _amber_town(self, towns):
    capital = towns.capital('amber')
    return capital.row, capital.col

  def test_a_people_holding_half_the_towns_is_named(self):
    planet = _planet('great_acceleration')
    towns = planet.settlements
    self.assertIsNone(towns.lead())
    # Founding alone never makes one (6 of 16 at most); conquest does.
    for town in [t for t in towns.places['amber'] if not t.capital][:3]:
      planet.change_hands(town.id, 'basalt', stay=0.8, loyalty=0.35)
    key, held, total = towns.lead()
    self.assertEqual((key, held, total), ('basalt', 8, 15))
    told = [e.text for e in planet.advance().events]
    self.assertTrue(
        any('Basalt clans now hold 8 of the 15 towns' in t for t in told), told
    )
    # Winning towns never makes a people abandon its own.
    self.assertLen(towns.places['basalt'], 8)
    self.assertFalse(any('abandon' in t for t in told), told)
    self.assertIn('the other peoples watch us', towns.describe_war('basalt'))
    self.assertIn(
        'The Basalt clans hold 8 of the 15', towns.describe_war('amber')
    )
    told = [e.text for e in planet.advance().events]  # Named once.
    self.assertFalse(any('now hold' in t for t in told))

  def test_words_reach_another_people_as_it_hears_offers(self):
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top

    planet = _planet('great_acceleration')
    basalt = actors_lib.PolityActor('Basalt clans', 'basalt', planet, 0.15)
    self.assertEqual(basalt.speaks_to(), ['Amber folk', 'Cedar league'])
    long = ' '.join(['grain'] * 60)
    answer = '\n'.join([
        actors_lib.KEEP_LABOR,
        'words to the Amber folk: Let us trade in peace.',
        'words to the amber folk: And again.',  # Once a people, a turn.
        f'words to the Cedar league: {long}',
        'words to the Moon folk: Hello?',
    ])
    self.assertEqual(actors_lib.people_order(answer), actors_lib.KEEP_LABOR)
    self.assertEqual(basalt.understood(answer).count('words to the'), 4)
    outcome = basalt.resolve(answer).split('\n')
    self.assertEqual(
        outcome[1:],
        [
            'Basalt clans send words to the Amber folk.',
            'Basalt clans spoke to the Amber folk already.',
            'Basalt clans send words to the Cedar league.',
            'Basalt clans found no people called "Moon folk" to speak to.',
        ],
    )
    outbox = dict(basalt.drain_outbox())
    self.assertEqual(
        outbox['diplomacy.amber.basalt'],
        'The Basalt clans say to the Amber folk: "Let us trade in peace."',
    )
    self.assertLen(  # At most forty words are carried.
        outbox['diplomacy.cedar.basalt'].split('"')[1].split(), 40
    )
    # Cut at the last whole sentence when one ends past halfway.
    said_ = ' '.join(['grain'] * 30) + ' is ours. ' + ' '.join(['more'] * 20)
    basalt.resolve(f'words to the Amber folk: {said_}')
    self.assertTrue(
        dict(basalt.drain_outbox())['diplomacy.amber.basalt'].endswith(
            'is ours."'
        )
    )
    # A people's sheet says where every town stands, to order and aim.
    north = planet.tiles.rows - 1
    capital = planet.settlements.capital('basalt')
    self.assertIn(
        f'Our towns stand at {north - capital.row},{capital.col} (capital)',
        planet.settlements.describe_towns('basalt'),
    )

  def test_three_towns_in_four_conquer_the_map(self):
    planet = _planet('great_acceleration')
    towns = planet.settlements
    others = [
        t for k in ('amber', 'cedar') for t in towns.places[k] if not t.capital
    ]
    for town in others[:6]:  # 11 of 15: not yet.
      planet.change_hands(town.id, 'basalt', stay=0.8, loyalty=0.35)
    planet.advance()
    self.assertIsNone(planet.outcome)
    planet.change_hands(others[6].id, 'basalt', stay=0.8, loyalty=0.35)
    planet.advance()  # 12 of 15: the map is the Basalt clans'.
    self.assertEqual(planet.outcome, planet_lib.DEFEAT)  # The age cut short.
    self.assertEqual(planet.winner, 'basalt')
    self.assertEqual(
        planet.outcome_reason,
        'the Basalt clans conquered the map (12 of the 15 towns)',
    )
    again = _planet('great_acceleration')
    again.set_state(planet.get_state())
    self.assertEqual(again.winner, 'basalt')

  def test_a_people_wiped_off_the_map_does_not_end_the_age(self):
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top

    # The owner's game of 19 Oct: the Amber folk took the Basalt clans'
    # towns, and the age ended as "famine broke the Basalt clans".
    planet = _planet('great_acceleration')
    towns = planet.settlements
    told = [
        planet.change_hands(t.id, 'amber', stay=0.8, loyalty=0.35).text
        for t in list(towns.places['basalt'])
    ]
    self.assertIn(
        'The Basalt clans hold no town any more: they are gone from the map.',
        told[-1],
    )
    self.assertIsNotNone(news_lib.place_of(told[-1]))  # Told where.
    for _ in range(3):
      planet.advance()
    self.assertIsNone(planet.outcome)  # The age goes on.
    basalt = actors_lib.PolityActor('Basalt clans', 'basalt', planet, 0.15)
    self.assertFalse(basalt.is_active())  # And they no longer act.

  def test_regard_fades_and_is_saved(self):
    planet, towns, _, town, _ = self._courted()
    towns.court(town.id, 'basalt', 0.3)
    towns.fade_regard()
    self.assertAlmostEqual(town.regard['basalt'], 0.26)
    state = towns.get_state()
    again = _planet('great_acceleration').settlements
    again.set_state(state)
    self.assertEqual(again.find(town.id)[0].regard, town.regard)
    for v in state['towns'].values():  # A save from before C5.
      del v[5:]
    again.set_state(state)
    self.assertEqual(again.find(town.id)[0].regard, {})
    self.assertFalse(again.find(town.id)[0].taken)
    del planet


if __name__ == '__main__':
  absltest.main()
