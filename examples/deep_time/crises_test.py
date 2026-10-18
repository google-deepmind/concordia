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

"""Tests of crises with countdowns."""

import json

from absl.testing import absltest
from concordia.contrib.pubsub import topics
from examples.deep_time import actors as actors_lib
from examples.deep_time import crises
from examples.deep_time import forecast
from examples.deep_time import game as game_lib
from examples.deep_time.prefabs import polity as polity_prefab

CLOUDS = 'seed bright clouds (cools the planet) [4 omega]'
HAZE = 'release a dark haze (warms the planet) [4 omega]'
ASH = 'bury the sky in ash (a volcanic winter, for years) [6 omega]'
PUSHES = {
    CLOUDS: 'cools about 7 °C next turn',
    HAZE: 'warms about 6 °C',
    ASH: 'cools about 53 °C next turn, lasting',
}


def _reading(payload):
  return topics.Message('t', json.dumps(payload), seq=0, time=0)


def _forecast(now, leaves_in=None, edge=''):
  return forecast.Forecast(
      tick=5,
      now=now,
      now_spread=2.0,
      ahead=now,
      ahead_spread=3.0,
      acts_in=1,
      unseen=0.0,
      readings=4,
      leaves_in=leaves_in,
      edge=edge,
  )


class CrisesTest(absltest.TestCase):

  def test_out_of_the_band_counts_down_the_strain_left(self):
    planet = game_lib.new_game('daisyworld').planet
    received = {'Thermometer': _reading({'out_of_band_ticks': 3.0})}
    (found,) = crises.steward_crises(
        planet, received, _forecast(41.0), [CLOUDS, HAZE, ASH], PUSHES
    )
    most = int(planet.scenario['max_ticks_out_of_band'])
    self.assertEqual(found['kind'], 'band')
    self.assertStartsWith(found['title'], 'Too hot: about 41')
    self.assertEqual(found['turns'], most - 3)
    self.assertEqual(found['answers'], [CLOUDS])  # What cools, not wrecks.

  def test_the_forecast_leaving_the_band_is_trouble_coming(self):
    planet = game_lib.new_game('daisyworld').planet
    (found,) = crises.steward_crises(
        planet, {}, _forecast(25.0, 2, 'bottom'), [CLOUDS, HAZE], PUSHES
    )
    self.assertEqual((found['title'], found['turns']), ('Cold coming', 2))
    self.assertEqual(found['answers'], [HAZE])
    calm = crises.steward_crises(planet, {}, _forecast(25.0), [CLOUDS], PUSHES)
    self.assertEmpty(calm)

  def test_hunger_unrest_and_a_goal_floor_are_read_from_the_instruments(self):
    planet = game_lib.new_game('sandbox').planet
    received = {
        'Omega meter': _reading(
            {'biodiversity': 0.12, 'restless': {'Cedar league': 0.8}}
        ),
        'Peoples': _reading({
            'polities': {
                'cedar': {'name': 'Cedar league', 'food_ratio': 0.7},
                'amber': {'name': 'Amber folk', 'food_ratio': 1.0},
            }
        }),
    }
    found = crises.steward_crises(
        planet, received, _forecast(15.0), [CLOUDS], PUSHES
    )
    self.assertEqual(
        [c['kind'] for c in found], ['famine', 'unrest', 'life']
    )  # The hungriest first, then unrest at 80%, then the floor.
    self.assertEqual(found[0]['title'], 'Cedar league go hungry')
    self.assertLen(found, crises.MOST)

  def test_a_goal_to_reach_counts_down_its_last_turns(self):
    planet = game_lib.new_game('great_acceleration').planet
    while planet.max_ticks - planet.tick > 5:
      planet.tick += 1  # Only the clock matters here.
    found = crises.steward_crises(planet, {}, None, [], {})
    self.assertEqual(
        [(c['kind'], c['turns']) for c in found], [('deadline', 5)]
    )

  def test_the_steward_is_offered_only_what_it_can_pay_for(self):
    planet = game_lib.new_game('daisyworld').planet
    steward = actors_lib.StewardActor('Gaia', planet)
    planet.omega = 3.0
    offered, later = steward.options(), steward.out_of_reach()
    self.assertTrue(any('white daisies' in o for o in offered))  # 2 omega.
    self.assertTrue(any('bright clouds' in o for o in later))  # 4 omega.
    self.assertFalse(set(offered) & set(later))


class PolityCrisesTest(absltest.TestCase):

  def test_an_emptying_granary_is_answered_by_hands_in_the_fields(self):
    planet = game_lib.new_game('fire_and_iron').planet
    civ = planet.civ
    assert civ is not None
    key, people = next(iter(civ.polities.items()))
    eats = float(people.population) * float(civ.food_per_pop)
    stock = civ.ledger.stock(key, 'grain')
    if stock > 0.5 * eats:  # Half a season of bread left.
      civ.ledger.withdraw(key, 'grain', stock - 0.5 * eats, 'test')
    farm = 'put more labor into: farm the land'
    found = crises.polity_crises(planet, key, [farm, 'mine ore'], {})
    self.assertEqual(found[0]['kind'], 'famine')
    self.assertEqual(found[0]['answers'], [farm])
    self.assertEqual(found[0]['turns'], 0)
    people.food_ratio = 0.6
    (hungry,) = crises.polity_crises(planet, key, [farm], {})
    self.assertEqual(hungry['title'], 'We go hungry: fed 60%')
    text = crises.words([hungry])
    self.assertStartsWith(text, ' Trouble now: We go hungry: fed 60%:')
    self.assertIn(f'Answer with: {farm}.', text)
    self.assertEqual(crises.words([]), '')


class SharedGoalTest(absltest.TestCase):

  def test_a_people_is_told_what_the_age_asks_and_its_floor(self):
    planet = game_lib.new_game('fire_and_iron').planet
    civ = planet.civ
    assert civ is not None
    key, people = next(iter(civ.polities.items()))
    planet.tick = planet.max_ticks - 20
    made = civ.made('iron')
    planet.history = [{'iron_made': made - 2.0}, {'iron_made': made}]
    people.population = 0.5 * civ.start_population[key]
    found = crises.shared_goal(planet, key)
    assert found is not None
    self.assertEqual((found['left'], found['last']), (20, 2.0))
    self.assertAlmostEqual(found['numbers'], 0.5)
    text = crises.shared_words(found)
    self.assertIn(
        f'the peoples together must forge 110 iron by turn {planet.max_ticks}',
        text,
    )
    self.assertIn(f'about {round((110 - made) / 20, 1):g} a turn', text)
    self.assertIn('a turn among us all; last turn all peoples made 2.', text)
    self.assertIn('falls under 40% of its numbers breaks', text)
    self.assertIn('we stand at 50% of ours', text)
    people.shortages = {'bloomery': 'charcoal', 'charcoal': 'wood'}
    held = crises.shared_goal(planet, key)
    assert held is not None
    self.assertIn(
        'Last generation our iron was held back by charcoal, and the charcoal'
        ' by wood: more hands to cut wood would free it.',
        crises.shared_words(held),
    )
    self.assertEqual(
        crises.held_words({'good': 'iron', 'short': ['ore'], 'recipe': None}),
        'our iron was held back by ore, which we do not make: buy it, or'
        ' learn to make it',
    )
    self.assertStartsWith(
        crises.held_words(
            {'good': 'seed ships', 'short': ['rockets'], 'recipe': 'x'}
        ),
        'our seed ships were held back by rockets',
    )
    self.assertEqual(
        crises.held_words(
            {'good': 'iron', 'short': [], 'recipe': 'smelt iron in a bloomery'}
        ),
        'hardly anyone works to smelt iron in a bloomery, so we make little'
        ' iron: more hands there would make it',
    )
    civ.ledger.deposit(key, 'grain', 100.0 * people.population, 'test')
    self.assertIn(
        'seasons of bread: plenty; hands could go elsewhere.',
        crises.shared_words(crises.shared_goal(planet, key)),
    )

  def test_each_age_says_its_own_terms(self):
    planet = game_lib.new_game('great_acceleration').planet
    civ = planet.civ
    assert civ is not None
    text = crises.shared_words(
        crises.shared_goal(planet, next(iter(civ.polities)))
    )
    self.assertIn('make 600 energy', text)
    self.assertIn('keep the air under 1.3 times its old carbon', text)
    planet = game_lib.new_game('sandbox').planet
    civ = planet.civ
    assert civ is not None
    text = crises.shared_words(
        crises.shared_goal(planet, next(iter(civ.polities)))
    )
    self.assertIn('a seed ship must leave for the stars', text)
    daisies = game_lib.new_game('daisyworld').planet
    self.assertIsNone(crises.shared_goal(daisies, 'anyone'))
    self.assertEqual(crises.shared_words(None), '')


class PressingTest(absltest.TestCase):

  def test_the_crisis_answer_and_the_weakest_link_end_the_question(self):
    context = game_lib.new_game('fire_and_iron')
    planet = context.planet
    civ = planet.civ
    assert civ is not None
    actor = actors_lib.PolityActor(
        'Basalt clans', 'basalt', planet, 0.15, age='the early iron age'
    )
    key = actor.polity
    people = civ.polities[key]
    people.food_ratio = 0.6  # Hungry.
    people.shortages = {'bloomery': 'charcoal', 'charcoal': 'wood'}
    options = list(actor.options())
    found = crises.pressing(planet, key, options, {})
    self.assertEqual(
        [(p['kind'], p['move']) for p in found],
        [
            ('famine', 'put more labor into: farm the land'),
            ('link', 'put more labor into: cut wood'),
        ],
    )
    question = actor.action_spec().call_to_action
    self.assertIn(
        'Most pressing now: We go hungry: fed 60%; put more labor into: farm'
        ' the land answers it.',
        question,
    )
    self.assertIn(
        'Our weakest link: our iron was held back by charcoal, and the'
        ' charcoal by wood: more hands to cut wood would free it.',
        question,
    )
    self.assertIn(  # Short of research: the next tech, and how far off.
        'The next tech for it: shaft mining (mine ore +40%) at 8 research;'
        ' we have 0',
        crises.shared_words(crises.shared_goal(planet, key)),
    )
    civ.ledger.deposit(key, 'research', 8.0, 'test')
    learn = [p for p in crises.pressing(planet, key, list(actor.options()), {})]
    self.assertIn(
        ('learn', 'learn shaft mining (8 research)'),
        [(p['kind'], p['move']) for p in learn],
    )
    self.assertIn(
        'Worth learning now: shaft mining (mine ore +40%).',
        actor.action_spec().call_to_action,
    )
    # A live mind plays the people as a character: no playbook lines.
    character = actors_lib.PolityActor(
        'Basalt clans', 'basalt', planet, 0.15, coached=False
    )
    self.assertNotIn('Most pressing', character.action_spec().call_to_action)
    self.assertNotIn('Worth learning', character.action_spec().call_to_action)
    self.assertIn('What the age asks', polity_prefab.storehouse(context, key))
    context.coached = False
    self.assertNotIn(
        'What the age asks', polity_prefab.storehouse(context, key)
    )
    civ.ledger.withdraw(key, 'research', 8.0, 'test')
    people.food_ratio = 1.0
    people.shortages = {}
    civ.ledger.deposit(key, 'grain', 100.0, 'test')
    self.assertEqual(crises.pressing(planet, key, options, {}), [])
    self.assertEqual(crises.pressing_words([]), '')


class ColonyCrisesTest(absltest.TestCase):

  RECIPES = [
      ('grow food under glass', {'food': 1.0}),
      ('split water for air', {'air': 1.5}),
      ('run the terraforming plants', {'warmth': 1.0}),
  ]

  def test_stores_running_out_and_a_lagging_sky_have_answers(self):
    report = {
        'colonists': 20.0,
        'stocks': {'food': 4.0, 'air': 30.0},
        'made': {'food': 1.0, 'air': 3.0},
        'shortfall': {},
        'temp_c': -40.0,
        'target_temp_c': 0.0,
        'warmed': 1.0,
        'left': 10,
    }
    options = [
        'put more labor into: grow food under glass',
        'put more labor into: run the terraforming plants',
    ]
    found = crises.colony_crises(
        report, {'food': 0.1, 'air': 0.1}, self.RECIPES, options, {}
    )
    food, sky = found
    self.assertEqual((food['title'], food['turns']), ('Food runs out', 4))
    self.assertEqual(food['answers'], [options[0]])
    self.assertEqual(sky['turns'], 10)  # It must warm 4 C a generation.
    self.assertEqual(sky['answers'], [options[1]])
    # Home reads the same report four turns late: the countdowns are less.
    late = crises.colony_crises(
        report, {'food': 0.1, 'air': 0.1}, self.RECIPES, options, {}, age=4
    )
    self.assertEqual([c['turns'] for c in late], [0, 6])
    self.assertIn('reported 4 turns ago', late[0]['detail'])

  def test_a_shortfall_is_now(self):
    report = {
        'colonists': 5.0,
        'stocks': {},
        'made': {},
        'shortfall': {'air': 0.4},
    }
    (short,) = crises.colony_crises(report, {'air': 0.1}, self.RECIPES, [], {})
    self.assertEqual((short['title'], short['turns']), ('Short of air', 0))


class SpeciesCrisesTest(absltest.TestCase):

  OPTIONS = [
      'spread aggressively',
      'hold steady',
      'go dormant',
      'propose a pact to the Black daisies',
  ]

  def test_a_dwindling_cold_outgrown_species_is_told_its_options(self):
    planet = game_lib.new_game('herds').planet
    planet.history[-1].update(optimum_white=24.0, tolerance_white=2.0)
    census = {
        'cover': {'white': 0.03, 'black': 0.4},
        'growth': {'white': 0.8, 'black': 1.1},
        'local_temp_c': {'white': 15.0, 'black': 20.0},
    }
    found = crises.species_crises(
        planet, 'white', {'Census': _reading(census)}, self.OPTIONS, {}
    )
    self.assertEqual([c['kind'] for c in found], ['species', 'band', 'species'])
    dwindling, cold, outgrown = found
    self.assertEqual(
        dwindling['answers'], ['spread aggressively', 'go dormant']
    )
    self.assertEqual(cold['title'], 'Too cold for us: 15 °C')
    self.assertEqual(cold['answers'], ['go dormant'])
    self.assertEqual(outgrown['title'], 'The Black daisies outgrow us')
    self.assertIn('propose a pact to the Black daisies', outgrown['answers'])

  def test_shrinking_herds_and_thin_meadows(self):
    planet = game_lib.new_game('herds').planet
    for _ in range(4):
      planet.advance()
    planet.history[-4]['grazers'] = 1.0
    planet.history[-1].update(grazers=0.7, grazer_diet_white=0.8)
    census = {'cover': {'white': 0.05, 'black': 0.5}}
    options = ['graze heavily', 'browse lightly', 'wander to new pastures']
    found = crises.grazer_crises(
        planet, {'Census': _reading(census)}, options, {}
    )
    self.assertEqual(found[0]['title'], 'Our herds shrink: -30% in 3 turns')
    self.assertEqual(found[1]['title'], 'The white meadows are thin: 5%')
    self.assertEqual(
        found[1]['answers'], ['browse lightly', 'wander to new pastures']
    )


class VaultCrisisTest(absltest.TestCase):

  def test_an_unfilled_vault_is_answered_by_gathering(self):
    planet = game_lib.new_game('sandbox').planet
    gather = 'gather the seeds of the wild for the arks [2 omega]'
    meter = {'biodiversity': 0.3, 'vault': {'held': 0.05, 'need': 0.17}}
    (found,) = crises.steward_crises(
        planet, {'Omega meter': _reading(meter)}, None, [gather], {}
    )
    self.assertEqual(found['title'], 'The seed vault holds 0.05')
    self.assertEqual(found['answers'], [gather])
    meter['vault']['held'] = 0.2
    self.assertEmpty(
        crises.steward_crises(
            planet, {'Omega meter': _reading(meter)}, None, [gather], {}
        )
    )


class SideObjectiveCrisisTest(absltest.TestCase):

  def test_a_side_objective_near_its_deadline_is_a_crisis(self):
    planet = game_lib.new_game('great_oxidation').planet
    flood = (
        'wash iron into the seas (soaks up oxygen, delays the crisis) [4 omega]'
    )
    task = {
        'key': 'methanogens',
        'text': 'Keep the Methanogen mats on 2% of the planet until turn 20',
        'kind': 'hold',
        'from': 1,
        'until': 20,
        'reward': 2.0,
        'state': 'open',
    }
    meter = {'biodiversity': 0.3, 'side': [task]}
    planet.tick = 10  # Ten turns to go: not yet pressing.
    self.assertEmpty(
        crises.steward_crises(
            planet, {'Omega meter': _reading(meter)}, None, [flood], {}
        )
    )
    planet.tick = 17
    (found,) = crises.steward_crises(
        planet, {'Omega meter': _reading(meter)}, None, [flood], {}
    )
    self.assertEqual(found['kind'], 'side')
    self.assertEqual(found['title'], 'Side objective: +2 Ω')
    self.assertEqual(found['turns'], 3)
    self.assertIn('Methanogen mats on 2%', found['detail'])
    self.assertEqual(found['answers'], [flood])
    self.assertIsNone(found['where'])  # No census read: nowhere.
    census = {
        'cover': {'methanogens': 0.03},
        'bands': [
            {'band': '45°N', 'cover': {'methanogens': 0.01}},
            {'band': '15°S', 'cover': {'methanogens': 0.06}},
        ],
    }
    (found,) = crises.steward_crises(
        planet,
        {'Omega meter': _reading(meter), 'Census': _reading(census)},
        None,
        [flood],
        {},
    )
    self.assertEqual(found['where'], {'lat': -15.0})
    task['state'] = 'missed'  # As last read: nothing left to save.
    self.assertEmpty(
        crises.steward_crises(
            planet, {'Omega meter': _reading(meter)}, None, [flood], {}
        )
    )


class CrisisPlacesTest(absltest.TestCase):

  def test_each_crisis_is_placed_as_its_role_can_know_it(self):
    planet = game_lib.new_game('daisyworld').planet
    (hot,) = crises.steward_crises(
        planet, {}, _forecast(41.0), [CLOUDS], PUSHES
    )
    self.assertEqual(hot['where'], {'world': 'hot'})
    (cold,) = crises.steward_crises(
        planet, {}, _forecast(25.0, 2, 'bottom'), [HAZE], PUSHES
    )
    self.assertEqual(cold['where'], {'world': 'cold'})
    census = {
        'cover': {'white': 0.01, 'black': 0.4},
        'growth': {'white': 0.9},
        'bands': [
            {'band': '75°N', 'cover': {'white': 0.001}},
            {'band': '15°N', 'cover': {'white': 0.02}},
            {'band': '45°S', 'cover': {'white': 0.005}},
        ],
    }
    (dwindling,) = crises.steward_crises(
        planet, {'Census': _reading(census)}, _forecast(25.0), [], PUSHES
    )
    self.assertEqual(dwindling['where'], {'lat': 15.0})
    self.assertEqual(dwindling['title'], 'White daisies dwindling')
    sandbox = game_lib.new_game('sandbox').planet
    received = {
        'Omega meter': _reading({'restless': {'Cedar league': 0.8}}),
        'Peoples': _reading(
            {'polities': {'cedar': {'name': 'Cedar league', 'food_ratio': 0.7}}}
        ),
    }
    found = crises.steward_crises(sandbox, received, _forecast(15.0), [], {})
    self.assertEqual(
        [c['where'] for c in found],
        [{'people': 'Cedar league'}, {'people': 'Cedar league'}],
    )
    civ = sandbox.civ
    assert civ is not None
    key, people = next(iter(civ.polities.items()))
    people.food_ratio = 0.6
    (hungry,) = crises.polity_crises(sandbox, key, [], {})
    self.assertEqual(hungry['where'], {'people': people.name})


if __name__ == '__main__':
  absltest.main()
