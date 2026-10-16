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

"""Tests for polities, labor, food and recipes."""

import json

from absl.testing import absltest
from concordia.contrib.pubsub import topics
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time.world import civ as civ_lib
from examples.deep_time.world import data
from examples.deep_time.world import grid as grid_lib
from examples.deep_time.world import publishing
import numpy as np

GRID = grid_lib.RegionGrid(bands=6, sectors=8)
POLITIES = [
    {
        'key': 'amber',
        'name': 'Amber folk',
        'home': 'auto',
        'population': 6.0,
        'allocation': {
            'farm': 0.6,
            'woodcut': 0.2,
            'charcoal': 0.1,
            'mine': 0.05,
            'bloomery': 0.05,
        },
    },
    {
        'key': 'basalt',
        'name': 'Basalt clans',
        'home': 'auto',
        'population': 5.0,
        'allocation': {'farm': 1.0},
    },
]


def _civ() -> civ_lib.Civilization:
  return civ_lib.Civilization(
      data.load('civ.yaml'),
      POLITIES,
      GRID,
      np.ones(GRID.size),
      seed=3,
      techs=data.load('techs.yaml')['techs'],
  )


class _World:
  """Records what the civilization does to the planet."""

  def __init__(self, temp_c: float = 15.0, forest: float = 0.3):
    self.temps = np.full(GRID.size, temp_c)
    self.forest = np.full(GRID.size, forest)
    self.cleared = np.zeros(GRID.size)
    self.emitted = 0.0

  def step(self, civ: civ_lib.Civilization) -> list[str]:
    def clear(cut):
      self.cleared += cut
      self.forest = np.maximum(0.0, self.forest - cut)

    def emit(amount, reason):
      del reason
      self.emitted += amount

    return civ.step(self.temps, 1.0, self.forest, clear_forest=clear, emit=emit)


class CivilizationTest(absltest.TestCase):

  def test_auto_homes_are_apart_and_settle_their_neighbours(self):
    civ = _civ()
    amber, basalt = civ.polities['amber'], civ.polities['basalt']
    self.assertNotEqual(amber.home, basalt.home)
    self.assertNotIn(basalt.home, GRID.neighbours(amber.home))
    self.assertIn(amber.home, amber.regions)
    self.assertLen(amber.regions, 1 + len(GRID.neighbours(amber.home)))

  def test_iron_chain_burns_forest_and_emits_carbon(self):
    civ = _civ()
    world = _World()
    events = []
    for _ in range(6):
      events.extend(world.step(civ))
    self.assertGreater(civ.made('iron'), 0.0)
    self.assertGreater(civ.made('charcoal'), 0.0)
    self.assertGreater(world.cleared.sum(), 0.0)
    self.assertGreater(world.emitted, 0.0)
    self.assertIn('Amber folk make their first iron.', events)
    self.assertEqual(civ.made('iron'), civ.totals['iron'])
    self.assertEqual(civ.stock('basalt', 'iron'), 0.0)  # Basalt only farm.
    self.assertTrue(civ.ledger.is_conserved())

  def test_climate_sets_harvests_and_famine_shrinks_people(self):
    kind, harsh = _civ(), _civ()
    kind_world, harsh_world = _World(15.0), _World(31.0)
    events = []
    for _ in range(5):
      kind_world.step(kind)
      events.extend(harsh_world.step(harsh))
    start = harsh.start_population['basalt']
    self.assertGreater(kind.polities['basalt'].population, start)
    self.assertLess(harsh.polities['basalt'].population, start)
    self.assertIn('Famine among the Basalt clans.', events)
    self.assertEqual(harsh.collapsed(0.99), ['amber', 'basalt'])

  def test_shift_labor_keeps_shares_normalised(self):
    civ = _civ()
    civ.shift_labor('basalt', 'mine', 0.25)
    allocation = civ.polities['basalt'].allocation
    self.assertAlmostEqual(sum(allocation.values()), 1.0)
    self.assertAlmostEqual(allocation['mine'], 0.25)
    self.assertAlmostEqual(allocation['farm'], 0.75)
    civ.set_allocation('basalt', {'dance': 1.0})
    self.assertEqual(civ.polities['basalt'].allocation, {'farm': 1.0})

  def test_eruptions_expose_ore(self):
    civ = _civ()
    before = civ.ore.copy()
    civ.add_ore(2)
    self.assertTrue(np.all(civ.ore > before))

  def test_state_round_trip(self):
    civ, world = _civ(), _World()
    for _ in range(3):
      world.step(civ)
    resumed = _civ()
    resumed.set_state(civ.get_state())
    self.assertEqual(resumed.payload(), civ.payload())
    world_b = _World()
    world_b.forest = world.forest.copy()
    world.step(civ)
    world_b.step(resumed)
    self.assertEqual(resumed.payload(), civ.payload())


class TechTest(absltest.TestCase):

  def test_learning_needs_research_and_prerequisites(self):
    civ = _civ()
    available = {t.key for t in civ.available_techs('amber')}
    self.assertIn('irrigation', available)
    self.assertNotIn('writing', available)  # Needs granaries.
    self.assertIn('needs', civ.learn('amber', 'irrigation'))
    civ.ledger.deposit('amber', 'research', 20.0, 'test')
    self.assertIn('cannot learn', civ.learn('amber', 'writing'))
    self.assertEqual(
        civ.learn('amber', 'granaries'),
        'Amber folk learn granaries and tally sticks.',
    )
    self.assertIn('writing', {t.key for t in civ.available_techs('amber')})
    self.assertAlmostEqual(civ.stock('amber', 'research'), 14.0)
    self.assertTrue(civ.unlocked('amber', 'reflex:threshold'))
    self.assertFalse(civ.unlocked('basalt', 'reflex:threshold'))

  def test_effects_change_yields_clearing_and_inputs(self):
    plain, skilled = _civ(), _civ()
    skilled.polities['amber'].techs = ['irrigation', 'coppicing', 'bellows']
    self.assertAlmostEqual(
        skilled.effect(skilled.polities['amber'], 'outputs', 'farm'), 1.25
    )
    self.assertAlmostEqual(
        skilled.effect(
            skilled.polities['amber'], 'inputs', 'bloomery', 'charcoal'
        ),
        0.75,
    )
    plain_world, skilled_world = _World(), _World()
    plain_world.step(plain)
    skilled_world.step(skilled)
    self.assertGreater(skilled.made('grain'), plain.made('grain'))
    self.assertLess(skilled_world.cleared.sum(), plain_world.cleared.sum())


class IndustryTest(absltest.TestCase):

  def test_recipes_unlock_with_techs(self):
    civ = _civ()
    keys = [r.key for r in civ.recipes_for('amber')]
    self.assertNotIn('coal_mine', keys)
    civ.polities['amber'].techs = ['shaft_mining', 'coal_mining']
    self.assertIn('coal_mine', [r.key for r in civ.recipes_for('amber')])
    self.assertIn('coal', civ.known_goods('amber'))
    self.assertNotIn('steel', civ.known_goods('amber'))
    self.assertIn('coal', civ.deposits)

  def test_energy_powers_labor_and_coal_makes_soot(self):
    plain, powered = _civ(), _civ()
    for civ in (plain, powered):
      civ.set_allocation('basalt', {'farm': 1.0})
    powered.ledger.deposit('basalt', 'energy', 100.0, 'test')
    for civ in (plain, powered):
      _World().step(civ)
    self.assertGreater(powered.made('grain'), plain.made('grain'))
    self.assertLess(powered.stock('basalt', 'energy'), 100.0)
    smoky = _civ()
    smoky.polities['amber'].techs = ['coal_mining', 'steam_power']
    smoky.set_allocation('amber', {'coal_mine': 0.5, 'coal_plant': 0.5})
    smoky.ledger.deposit('amber', 'coal', 10.0, 'test')
    soot = []
    smoky.step(
        np.full(GRID.size, 15.0),
        1.0,
        np.full(GRID.size, 0.3),
        clear_forest=lambda cut: None,
        emit=lambda amount, reason: None,
        soot=soot.append,
    )
    self.assertNotEmpty(soot)
    self.assertGreater(smoky.made('energy'), 0.0)


class TierFourTest(absltest.TestCase):

  def test_scrubbers_cut_soot_and_panels_make_energy_and_wear(self):
    civ = _civ()
    people = civ.polities['amber']
    people.techs = ['coal_mining', 'steam_power']
    self.assertAlmostEqual(civ.effect(people, 'emissions', 'soot'), 1.0)
    people.techs.append('scrubbers')
    self.assertAlmostEqual(civ.effect(people, 'emissions', 'soot'), 0.3)
    people.techs = ['photovoltaics']
    civ.set_allocation('amber', {'solar_power': 1.0})
    civ.ledger.deposit('amber', 'panels', 100.0, 'test')
    _World().step(civ)
    self.assertGreater(civ.made('energy'), 0.0)
    self.assertLess(civ.stock('amber', 'panels'), 100.0)  # Panels wear.

  def test_sheet_shows_techs_within_reach(self):
    civ = _civ()
    self.assertIn('Within reach: irrigation (6)', civ.sheet('amber'))


class PushTest(absltest.TestCase):

  def test_labor_shift_makes_the_recipes_goods_and_uses_its_inputs(self):
    civ = _civ()
    _World().step(civ)
    change = civ.labor_shift('amber', 'bloomery', 0.15)
    self.assertGreater(change['iron'], 0.0)
    self.assertLess(change['ore'], 0.0)
    self.assertLess(change['grain'], 0.0)  # Hands taken from the fields.
    before = dict(civ.polities['amber'].allocation)
    self.assertEqual(civ.polities['amber'].allocation, before)  # Untouched.

  def test_describes_goods_apart_from_pollutants(self):
    text = civ_lib.describe_change(
        {'iron': 4.8, 'slag': 4.7, 'co2': 3.6, 'ore': -7.7, 'grain': -2.3},
        {'co2': 'smoke', 'slag': 'slag'},
    )
    self.assertEqual(
        text,
        'iron +4.8 a generation; ore −7.7, grain −2.3; smoke +3.6, slag +4.7',
    )

  def test_a_techs_push_is_its_effects_in_words(self):
    civ = civ_lib.Civilization(
        {**data.load('civ.yaml'), **data.load('techs.yaml')},
        POLITIES,
        GRID,
        np.ones(GRID.size),
        seed=3,
        techs=data.load('techs.yaml')['techs'],
    )
    self.assertEqual(civ.describe_tech('irrigation'), 'farm the land +25%')
    self.assertIn('treaties with other peoples', civ.describe_tech('diplomacy'))
    self.assertIn('unlocks', civ.describe_tech('glassmaking'))

  def test_a_people_weighs_its_labor_and_learning(self):
    context = game_lib.new_game('great_acceleration')
    context.planet.advance()
    civ = context.planet.civ
    assert civ is not None
    civ.ledger.deposit('basalt', 'research', 500.0, 'test')
    actor = actors_lib.PolityActor(
        'Basalt clans', 'basalt', context.planet, 0.15, context.automation
    )
    options = actor.options()
    pushes = actor.pushes(options)
    labor = [o for o in options if o.startswith('put more labor into')]
    learn = [o for o in options if o.startswith('learn ')]
    self.assertTrue(labor and learn)
    self.assertContainsSubset(labor + learn, pushes)
    self.assertIn('a generation', pushes[labor[0]])


class AimTest(absltest.TestCase):

  def test_an_aim_shows_the_road_to_a_goal(self):
    civ = game_lib.new_game('great_acceleration').planet.civ
    assert civ is not None
    text = civ.aim('basalt', 'energy')
    self.assertTrue(
        text.startswith(
            'Aim: energy, made by burn coal in steam engines (learn steam'
            ' power first: 30 research'
        ),
        text,
    )
    self.assertIn('which uses 1 coal', text)
    sandbox = game_lib.new_game('sandbox').planet.civ
    assert sandbox is not None
    self.assertIn(
        'Aim: seed ships, made by', sandbox.aim('amber', 'seed_ships')
    )
    self.assertIn('10 seeds', sandbox.aim('amber', 'seed_ships'))

  def test_what_held_back_a_good_is_followed_to_its_weakest_link(self):
    civ = game_lib.new_game('fire_and_iron').planet.civ
    assert civ is not None
    people = civ.polities['basalt']
    people.shortages = {}
    self.assertEqual(civ.held_back('basalt', 'iron'), ([], None))
    people.shortages = {'bloomery': 'charcoal', 'charcoal': 'wood'}
    self.assertEqual(
        civ.held_back('basalt', 'iron'), (['charcoal', 'wood'], 'woodcut')
    )
    people.shortages = {'bloomery': 'ore'}
    self.assertEqual(civ.held_back('basalt', 'iron'), (['ore'], 'mine'))
    people.shortages = {'bloomery': 'charcoal', 'charcoal': 'charcoal'}
    self.assertEqual(  # A loop stops at the good seen again.
        civ.held_back('basalt', 'iron'), (['charcoal'], 'charcoal')
    )
    people.shortages = {'bloomery': 'ore'}
    people.allocation['mine'] = 0.01  # Hardly any miners: the link.
    self.assertEqual(civ.held_back('basalt', 'iron'), (['ore'], 'mine'))
    # Hardly any smelters, short of nothing: with ore and charcoal in store
    # the smelters' hands are the link; without charcoal, the chain goes on.
    people.shortages = {}
    people.allocation['bloomery'] = 0.005
    for good in ('ore', 'charcoal'):
      civ.ledger.deposit('basalt', good, 10.0, 'test')
    self.assertEqual(civ.held_back('basalt', 'iron'), ([], 'bloomery'))
    civ.ledger.withdraw('basalt', 'charcoal', civ.stock('basalt', 'charcoal'))
    short, link = civ.held_back('basalt', 'iron')
    self.assertEqual(short[0], 'charcoal')
    self.assertNotEqual(link, 'bloomery')

  def test_the_tech_worth_learning_helps_the_goals_chain(self):
    civ = game_lib.new_game('fire_and_iron').planet.civ
    assert civ is not None
    self.assertEqual(
        civ.goal_chain('basalt', 'iron'),
        ['bloomery', 'mine', 'charcoal', 'woodcut'],
    )
    self.assertIsNone(civ.best_tech('basalt', 'iron'))  # No research yet.
    nxt = civ.best_tech('basalt', 'iron', affordable=False)
    assert nxt is not None
    self.assertEqual(nxt.key, 'shaft_mining')  # Ore +40%: the most.
    civ.ledger.deposit('basalt', 'research', 8.0, 'test')
    best = civ.best_tech('basalt', 'iron')
    assert best is not None
    self.assertEqual(best.key, 'shaft_mining')
    civ.polities['basalt'].techs.append('shaft_mining')
    after = civ.best_tech('basalt', 'iron')  # Then the next best.
    assert after is not None
    self.assertNotEqual(after.key, 'irrigation')  # Farms are not the chain.

  def test_a_people_is_called_by_its_age(self):
    context = game_lib.new_game('great_acceleration')
    actor = actors_lib.PolityActor(
        'Basalt clans',
        'basalt',
        context.planet,
        0.15,
        age='the age of coal and steam',
    )
    action = actor.action_spec().call_to_action
    self.assertIn('{name}, a people of the age of coal and steam.', action)


class PolityActorTest(absltest.TestCase):

  def test_menus_never_exceed_twenty_six_options(self):
    context = game_lib.new_game('great_acceleration')
    civ = context.planet.civ
    assert civ is not None
    people = civ.polities['basalt']
    people.techs = [t.key for t in civ.techs]  # Everything unlocked.
    civ.ledger.deposit('basalt', 'research', 500.0, 'test')
    actor = actors_lib.PolityActor(
        people.name, 'basalt', context.planet, 0.15, context.automation
    )
    options = actor.options()
    self.assertLessEqual(len(options), 26)
    self.assertIn(actors_lib.KEEP_LABOR, options)
    self.assertIn('put more labor into: farm the land', options)

  def test_treaty_offers_do_not_crowd_plans_off_the_menu(self):
    context = game_lib.new_game('sandbox')
    civ = context.planet.civ
    assert civ is not None
    for people in civ.polities.values():
      people.techs = [t.key for t in civ.techs]  # Every kind, every partner.
    civ.ledger.deposit('basalt', 'research', 500.0, 'test')
    actor = actors_lib.PolityActor(
        civ.polities['basalt'].name,
        'basalt',
        context.planet,
        0.15,
        context.automation,
    )
    options = actor.options()
    self.assertTrue(any(o.startswith('plan a balanced chain') for o in options))
    for what in ('coal treaty', 'fishing quota', 'runoff pact'):
      self.assertTrue(any(f'propose a {what}' in o for o in options), what)

  def test_the_menu_keeps_new_ways_to_make_what_ran_short(self):
    context = game_lib.new_game('great_acceleration')
    civ = context.planet.civ
    assert civ is not None
    people = civ.polities['basalt']
    # Chemistry first: fertiliser is the oldest of many unworked unlocks.
    people.techs = ['chemistry'] + [
        t.key for t in civ.techs if t.key != 'chemistry'
    ]
    actor = actors_lib.PolityActor(people.name, 'basalt', context.planet, 0.15)
    option = 'put more labor into: farm with fertiliser'
    self.assertNotIn(option, actor.options())
    people.shortages = {'bakery': 'grain'}  # Last tick ran short of grain.
    self.assertIn(option, actor.options())

  def test_focus_shifts_labor_and_keep_does_not(self):
    context = game_lib.new_game('fire_and_iron')
    civ = context.planet.civ
    assert civ is not None
    people = civ.polities['amber']
    actor = actors_lib.PolityActor(people.name, 'amber', context.planet, 0.2)
    self.assertIn(actors_lib.KEEP_LABOR, actor.options())
    before = dict(people.allocation)
    self.assertIn('keep their labor', actor.resolve(actors_lib.KEEP_LABOR))
    self.assertEqual(people.allocation, before)
    outcome = actor.resolve('put more labor into: mine ore')
    self.assertIn('mine ore', outcome)
    self.assertAlmostEqual(
        people.allocation['mine'], 0.8 * before.get('mine', 0.0) + 0.2
    )
    self.assertTrue(actor.is_active())
    self.assertFalse(any(o.startswith('learn') for o in actor.options()))
    civ.ledger.deposit('amber', 'research', 6.0, 'test')
    option = 'learn irrigation (6 research)'
    self.assertIn(option, actor.options())
    self.assertEqual(actor.resolve(option), 'Amber folk learn irrigation.')
    self.assertIn('irrigation', people.techs)
    self.assertIn('we know: irrigation', civ.sheet('amber'))
    self.assertIn('people, fed', civ.sheet('amber'))
    self.assertIn('Labor: farm the land', civ.sheet('amber'))


class GreatAccelerationTest(absltest.TestCase):

  def test_peoples_start_industrious_and_soot_darkens_the_planet(self):
    planet = game_lib.new_game('great_acceleration').planet
    civ = planet.civ
    assert civ is not None
    self.assertIn('mathematics', civ.polities['amber'].techs)
    self.assertEqual(civ.stock('amber', 'research'), 30.0)
    self.assertGreater(planet.soot_albedo, 0.0)
    planet.advance()
    self.assertIn('energy_made', planet.history[-1])


class FireAndIronTest(absltest.TestCase):

  def test_planet_runs_the_iron_age_and_publishes_a_census(self):
    context = game_lib.new_game('fire_and_iron')
    planet = context.planet
    civ = planet.civ
    assert civ is not None
    for _ in range(8):
      planet.advance()
    self.assertGreater(planet.history[-1]['population'], 0.0)
    self.assertGreater(planet.history[-1]['iron_made'], 0.0)
    text = publishing.render_civ(
        topics.Message('civ.census', json.dumps(civ.payload()), seq=0)
    )
    self.assertIn('Amber folk', text)
    self.assertIn('Iron forged so far', text)
    resumed = game_lib.new_game('fire_and_iron')
    resumed.planet.set_state(planet.get_state())
    planet.advance()
    resumed.planet.advance()
    self.assertEqual(resumed.planet.history[-1], planet.history[-1])


if __name__ == '__main__':
  absltest.main()
