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

"""Tests for the phone game session (a person plays through HumanActComponent)."""

import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Callable

from absl.testing import absltest
from concordia.components.agent import human_act_component
from concordia.contrib.pubsub import topics
from examples.deep_time import actors as actors_lib
from examples.deep_time import scenarios
from examples.deep_time import views
from examples.deep_time.web import session as session_lib
from examples.deep_time.world import treaties as treaties_lib
from concordia.typing import entity as entity_lib
from concordia.typing import human_input
import numpy as np

# A biome tally's words: 'The desert spreads from 40 to 60 of the 1200 tiles.'
TALLY = re.compile(r' of the \d+ tiles')


def _wait_for(predicate: Callable[[], Any], timeout: float = 60.0) -> Any:
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    value = predicate()
    if value:
      return value
    time.sleep(0.05)
  raise AssertionError('timed out')


def _question(game: session_lib.GameSession) -> dict[str, Any]:
  return _wait_for(lambda: game.state()['question'])


class GameSessionTest(absltest.TestCase):

  def test_the_steward_acts_through_the_upstream_human_act_component(self):
    for mode, human in (('play', True), ('watch', False)):
      gaia = session_lib.GameSession('daisyworld', mode).role_entity()
      assert gaia is not None
      self.assertEqual(gaia.name, 'Gaia')
      self.assertEqual(
          isinstance(
              gaia.get_act_component(), human_act_component.HumanActComponent
          ),
          human,
          mode,
      )

  def test_the_person_answers_gaias_requests(self):
    game = session_lib.GameSession('daisyworld', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    question = _question(game)
    self.assertIn('wait and save omega', question['options'])
    self.assertIn('Gaia', question['call_to_action'])
    # The request carries the prompt a model playing Gaia would read.
    self.assertIn('Thermometer (climate.global)', question['context'])
    self.assertIn('Advisor briefing', question['context'])
    white = next(o for o in question['options'] if 'white daisies' in o)
    self.assertIsNotNone(game.choose('stale-id', white))
    self.assertIsNotNone(game.choose(question['request_id'], 'dance'))
    tick = game.state()['tick']
    self.assertIsNone(game.choose(question['request_id'], white))
    self.assertEqual(game.choices, [white])
    # Answered: the same request can't be answered twice.
    self.assertIsNotNone(game.choose(question['request_id'], white))
    _wait_for(lambda: game.state()['tick'] > tick)
    self.assertEqual(game.state()['error'], '')

  def test_a_cancelled_game_stops_early(self):
    game = session_lib.GameSession('daisyworld', 'play').start()
    _question(game)
    game.cancel()
    game.join(30)
    self.assertTrue(game.finished)
    state = game.state()
    self.assertEqual(state['error'], '')
    self.assertEqual(state['outcome'], '')  # Stopped well before the end.
    self.assertLess(state['tick'], 5)
    self.assertIsNone(state['question'])

  def test_watch_mode_plays_to_the_end(self):
    game = session_lib.GameSession(
        'daisyworld', 'watch', pace_seconds=0.0
    ).start()
    game.join(120)
    state = game.state()
    self.assertTrue(state['finished'])
    self.assertEqual(state['error'], '')
    self.assertIn(state['outcome'], ('victory', 'defeat'))
    self.assertIsNone(state['question'])
    self.assertEqual(state['choices'], [])
    self.assertTrue(state['news'])
    self.assertTrue(state['revealed'])
    self.assertEqual(state['stats'][1]['label'], '°C')
    self.assertIn('Thermometer', [r['label'] for r in state['readings']])

  def test_a_player_sees_only_their_readings_until_the_end(self):
    game = session_lib.GameSession('daisyworld', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    for _ in range(4):  # Long enough for the late thermometer to report.
      question = _question(game)
      game.choose(question['request_id'], question['options'][0])
    _question(game)
    state = game.state()
    self.assertFalse(state['revealed'])
    self.assertEqual(state['climate'], '')
    self.assertEqual(state['news'], [])
    self.assertEqual(state['chronicle'], '')
    self.assertEqual(state['readings'], [])  # The request's context instead.
    # The strip shows what the thermometer last delivered, and its age.
    temp = state['stats'][1]
    believed = game.context.believed('Gaia', 'climate.global', 'temp_c')
    assert believed is not None
    self.assertEqual(temp['value'], f'{believed:.1f}')
    self.assertStartsWith(temp['label'], '°C read')
    game.cancel()
    game.join(30)
    revealed = game.state()
    self.assertTrue(revealed['revealed'])
    self.assertNotEqual(revealed['climate'], '')
    self.assertEqual(revealed['stats'][1]['label'], '°C')

  def test_light_delay_is_played_as_mission_control(self):
    game = session_lib.GameSession('light_delay', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    self.assertEqual(game.state()['role'], 'Mission Control')
    question = _question(game)
    self.assertIn('listen and wait', question['options'])
    self.assertIn('Colony relay', question['context'])
    labels = [s['label'] for s in game.state()['stats']]
    self.assertTrue(any(label.startswith('colonists') for label in labels))
    self.assertNotIn('biodiversity', labels)

  def test_a_person_can_play_a_species(self):
    game = session_lib.GameSession(
        'daisyworld', 'play', role='White daisies'
    ).start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    self.assertEqual(game.role, 'White daisies, a species')
    labels = [s['label'] for s in game.state()['stats']]
    self.assertEqual(labels, ['turn', 'cover', 'seed bank'])
    species = game.role_entity()
    assert species is not None
    self.assertIsInstance(
        species.get_act_component(), human_act_component.HumanActComponent
    )
    question = _question(game)
    self.assertIn('hold steady', question['options'])
    self.assertIn('White daisies', question['call_to_action'])
    tick = game.state()['tick']
    self.assertIsNone(game.choose(question['request_id'], 'hold steady'))
    _wait_for(lambda: game.state()['tick'] > tick)
    self.assertEqual(game.state()['error'], '')

  def test_a_person_can_play_a_people_whose_plans_still_land(self):
    game = session_lib.GameSession(
        'fire_and_iron', 'play', role='Amber folk'
    ).start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    self.assertEqual(game.role, 'Amber folk, a people')
    labels = [s['label'] for s in game.state()['stats']]
    # The goal all the peoples share, in place of the head count.
    self.assertEqual(labels, ['turn', 'fed', 'grain', 'iron made'])
    for _ in range(5):
      question = _question(game)
      self.assertIn('Our storehouse and labor', question['context'])
      game.choose(question['request_id'], question['options'][0])
    question = _question(game)
    # The people's background plan (a contrib Publisher) still deliberates.
    self.assertNotIn('No plan yet', question['context'])
    state = game.state()
    self.assertEqual(state['error'], '')
    # Gaia's Advisor is private to Gaia until the game ends.
    self.assertIsNotNone(game.context.bus.latest('gaia.briefings'))
    self.assertEqual(state['briefing'], '')

  def test_a_seat_enacts_a_draft_with_its_words(self):
    request = human_input.HumanInputRequest(
        request_id='1',
        entity_name='Cedar league',
        action_spec=entity_lib.free_action_spec(call_to_action='Act'),
        contexts={
            "Our printers' sheet": 'clean air: Smoke Kills. [tick 4, fresh]',
            "Our scribes' draft": (
                'Standing orders ...: sell charcoal. As'
                ' written: [{"name": "x"}] [tick 3, 1 tick old]'
            ),
        },
    )
    move = actors_lib.PUBLISH_BROADSHEET
    self.assertEqual(
        session_lib.with_draft(request, move),
        f'{move}: clean air: Smoke Kills.',
    )
    self.assertEqual(
        session_lib.with_draft(request, actors_lib.ADOPT_ORDERS),
        f'{actors_lib.ADOPT_ORDERS}: [{{"name": "x"}}]',
    )
    self.assertEqual(session_lib.with_draft(request, 'wait'), 'wait')

  def test_a_seat_takes_a_move_with_its_words_from_the_gms_menu(self):
    game = session_lib.GameSession(
        'great_acceleration', 'play', role='Cedar league'
    ).start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    _question(game)
    menu = views.menu(game.context, 'Cedar league')
    move = menu['offered'][0]
    check = game._check  # pylint: disable=protected-access
    self.assertEqual(check('Cedar league', move), '')
    self.assertEqual(check('Cedar league', f'{move}: our words'), '')
    self.assertIn('is not one of the moves', check('Cedar league', 'dance'))
    # Whether a move may be aimed, and what may be planned: the GM's menu.
    self.assertEqual(
        [o for o in menu['offered'] if game._targetable('Cedar league', o)],  # pylint: disable=protected-access
        [o for o in menu['offered'] if o in menu['aimable']],
    )

  def test_a_watched_game_may_have_a_bot_play_a_role(self):
    game = session_lib.GameSession(
        'daisyworld', 'watch', players=['Gaia'], bots=['Gaia']
    )
    self.assertIn('Gaia', game.context.bots)
    with self.assertRaises(ValueError):  # A person's seat is the person's.
      session_lib.GameSession('daisyworld', 'play', role='Gaia', bots=['Gaia'])
    with self.assertRaises(KeyError):
      session_lib.GameSession('daisyworld', 'watch', bots=['Nobody'])

  def test_each_role_sees_its_own_goal(self):
    steward = session_lib.GameSession('sandbox', 'play').state()['goal']
    people = session_lib.GameSession(
        'sandbox', 'play', role='Amber folk'
    ).state()['goal']
    self.assertIn('seed ship', steward)
    self.assertNotEqual(people, steward)
    self.assertTrue(people)
    colony = session_lib.GameSession('light_delay', 'play').state()
    self.assertNotIn('Ω', [s['label'] for s in colony['stats']])

  def test_a_script_that_ends_mid_question_does_not_hang(self):
    if sys.executable is None:
      with session_lib.GameSession('daisyworld', 'play').start() as game:
        _wait_for(lambda: game.state()['question'])
      self.assertTrue(game.finished)
      return
    script = (
        'import time\n'
        'from examples.deep_time.web import session\n'
        "with session.GameSession('daisyworld', 'play').start() as game:\n"
        "  while not game.state()['question']:\n"
        '    time.sleep(0.05)\n'
        "  print('asked', flush=True)\n"
    )
    done = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    self.assertIn('asked', done.stdout)

  def test_two_people_play_one_game_from_two_phones(self):
    game = session_lib.GameSession(
        'great_acceleration', 'play', players=['Amber folk', 'Basalt clans']
    ).start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    amber = _wait_for(lambda: game.state('Amber folk')['question'])
    basalt = _wait_for(lambda: game.state('Basalt clans')['question'])
    self.assertNotEqual(amber['request_id'], basalt['request_id'])
    self.assertIn('You are Amber folk', amber['call_to_action'])
    self.assertIn('You are Basalt clans', basalt['call_to_action'])
    view = game.state('Basalt clans')
    self.assertEqual(view['seat'], 'Basalt clans')
    self.assertEqual(view['role'], 'Basalt clans, a people')
    self.assertEqual(
        [p['name'] for p in view['seats']], ['Amber folk', 'Basalt clans']
    )
    self.assertEqual(view['waiting_for'], ['Amber folk, a people'])
    # An answer reaches its own seat only.
    self.assertIsNone(game.choose(amber['request_id'], amber['options'][0]))
    self.assertEqual(game.state('Amber folk')['choices'], [amber['options'][0]])
    self.assertEqual(game.state('Basalt clans')['choices'], [])
    self.assertEqual(
        game.state('Basalt clans')['question']['request_id'],
        basalt['request_id'],
    )
    # Anyone else (a stale link, a plain URL) only watches: no one's
    # question, prompt, goal or choices.
    for stranger in ('Cedar league', None):
      view = game.state(stranger)
      self.assertTrue(view['spectator'])
      self.assertIsNone(view['question'])
      self.assertEqual((view['goal'], view['choices']), ('', []))
    # Alone, a player may omit the seat.
    solo = session_lib.GameSession('daisyworld', 'play')
    self.assertFalse(solo.state(None)['spectator'])
    with self.assertRaises(ValueError):
      session_lib.GameSession(
          'great_acceleration', players=['Amber folk', 'Amber folk']
      )

  def test_a_turn_timer_counts_down(self):
    seat = session_lib.Seat('Gaia', 'Gaia, the Steward', turn_seconds=60.0)
    self.assertIsNone(seat.seconds_left())  # Nothing asked yet.
    untimed = session_lib.Seat('Gaia', 'Gaia, the Steward')
    self.assertIsNone(untimed.seconds_left())

  def test_a_decided_game_asks_no_one_and_cannot_be_held(self):
    seat = session_lib.Seat('Gaia', 'Gaia, the Steward')
    seat.over = lambda: True
    request = human_input.HumanInputRequest(
        request_id='last',
        entity_name='Gaia',
        action_spec=entity_lib.choice_action_spec(
            call_to_action='Your move?',
            options=('raise the albedo', 'wait and save omega'),
        ),
        contexts={},
    )
    self.assertEqual(seat(request), 'wait and save omega')  # At once.
    self.assertEqual(seat.choices, [])  # Not one of the player's moves.
    self.assertIsNone(seat.pending())

  def test_a_hold_with_no_neutral_move_stops_and_says_why(self):
    seat = session_lib.Seat('Gaia', 'Gaia, the Steward')
    seat.hold = 3
    request = human_input.HumanInputRequest(
        request_id='odd',
        entity_name='Gaia',
        action_spec=entity_lib.choice_action_spec(
            call_to_action='Your move?', options=('raise', 'lower')
        ),
        contexts={},
    )
    self.assertIsNone(seat._held(request))  # pylint: disable=protected-access
    self.assertEqual((seat.hold, seat.stopped[:13]), (0, 'this question'))

  def test_out_of_time_the_role_makes_its_most_neutral_move(self):
    game = session_lib.GameSession(
        'daisyworld',
        'play',
        players=['Gaia', 'White daisies'],
        turn_seconds=0.3,
    ).start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    question = _wait_for(lambda: game.state('Gaia')['question'])
    self.assertIsNotNone(question)
    _wait_for(lambda: game.state()['tick'] >= 3)
    self.assertEqual(game.choices[0], '(out of time) wait and save omega')
    self.assertEqual(
        game.state('White daisies')['choices'][0], '(out of time) hold steady'
    )
    self.assertEqual(session_lib.neutral(['a', 'hold steady']), 'hold steady')
    self.assertEqual(session_lib.neutral(['a', 'b']), 'a')

  def test_the_map_shows_terrain_and_hides_the_truth_from_players(self):
    game = session_lib.GameSession('great_acceleration', 'play')
    played = game.map()
    self.assertEqual((played['rows'], played['cols']), (6, 8))
    self.assertLen(played['regions'], 48)
    self.assertEqual(played['layers'], ['biomes', 'terrain', 'pollution'])
    self.assertLen(played['biomes'], 30)  # Biomes are public.
    self.assertIsNone(played['tile_temp'])
    self.assertIsNone(played['rain'])
    self.assertTrue(all('temp_c' not in r for r in played['regions']))
    self.assertTrue(
        any(r['land'] == 0 for r in played['regions'])
        and any(r['land'] == 1 for r in played['regions'])
    )
    # North first: the top row is the northernmost band.
    north = next(r for r in played['regions'] if r['row'] == 0)
    self.assertIn('N', north['name'])
    self.assertIsNone(played['ice'])  # Ice is truth too.
    self.assertNotIn('terrain', played)  # Static terrain only on request.
    self.assertEqual((played['tile_rows'], played['tile_cols']), (30, 40))
    full = game.map(full=True)
    self.assertLen(full['terrain']['terrain'], 30)
    self.assertGreaterEqual(len(full['biome_legend']), 10)
    self.assertEqual(full['game_id'], played['game_id'])
    watched = session_lib.GameSession('great_acceleration', 'watch').map()
    self.assertEqual(
        watched['layers'],
        [
            'biomes',
            'terrain',
            'temperature',
            'rainfall',
            'seasons',
            'life',
            'fire',
            'pollution',
        ],
    )
    self.assertLen(watched['tile_temp'], 30)
    life = watched['life_tiles']
    self.assertEqual(
        [sp['key'] for sp in life['species']],
        list(
            session_lib.GameSession(
                'great_acceleration', 'watch'
            ).context.planet.daisies.species()
        ),
    )
    self.assertLen(life['dominant'], 30)
    self.assertIn('.', ''.join(life['dominant']))  # Seas hold no plants.
    self.assertLen(life['covers'], len(life['species']))
    self.assertTrue(all(len(rows) == 30 for rows in life['covers']))
    self.assertTrue(
        all(set(''.join(rows)) <= set(session_lib._BASE36) for rows in life['covers'])  # pylint: disable=protected-access
    )
    self.assertIsNone(played['life_tiles'])
    self.assertLen(played['burns'], 30)  # Scars are seen by all; risk is not.
    self.assertIsNone(played['fire_risk'])
    self.assertLen(watched['fire_risk'], 30)
    self.assertLen(played['pollution']['soot'], 30)  # Smog is public.
    herds = played['herds']  # Animals are seen by all.
    self.assertEqual(
        [s['key'] for s in herds['species']],
        ['bison', 'wolves', 'fish', 'geese'],
    )
    self.assertTrue(
        all(0 <= h[2] < 30 and 0 <= h[3] < 40 for h in herds['herds'])
    )
    self.assertLen(watched['rain'][0], 40)
    self.assertTrue(all('temp_c' in r for r in watched['regions']))
    self.assertLen(watched['ice'], 30)
    self.assertNotEqual(watched['game_id'], played['game_id'])

  def test_watchers_get_progress_and_time_controls_players_do_not(self):
    watched = session_lib.GameSession('great_acceleration', 'watch')
    state = watched.state()
    labels = [p['label'] for p in state['progress']]
    self.assertEqual(labels[:3], ['turn', 'energy', 'CO₂'])
    self.assertTrue(all(0 <= p['fraction'] <= 1 for p in state['progress']))
    self.assertEqual(state['speed'], 'play')
    self.assertEqual(watched.set_speed('pause'), 'pause')
    self.assertEqual(watched.set_speed('fast'), 'fast')
    self.assertEqual(watched.set_speed('step'), 'fast')
    # Play after a step plays on: the step gives way and will not re-pause.
    self.assertEqual(watched.set_speed('play'), 'play')
    self.assertFalse(watched._step_once)  # pylint: disable=protected-access
    with self.assertRaises(ValueError):
      watched.set_speed('rewind')
    played = session_lib.GameSession('great_acceleration', 'play').state()
    self.assertIsNone(played['progress'])  # The truth waits for the reveal.
    self.assertIsNone(played['speed'])

  def test_clouds_are_public_per_region_and_absent_from_a_clear_sky(self):
    played = session_lib.GameSession('great_acceleration', 'play')
    planet = played.context.planet
    self.assertIsNone(played.map()['clouds'])  # A clear sky.
    weights = np.zeros(planet.grid.size)
    weights[planet.grid.size - 1] = 1.0  # The far north-east region.
    planet.daisies.add_albedo_pulse(0.06, weights)
    planet.daisies.add_albedo_pulse(-0.02)
    clouds = played.map()['clouds']  # Seen by a player too: the sky.
    self.assertLen(clouds, planet.grid.bands)
    self.assertGreater(clouds[0][-1], 0.0)  # North first: the top row.
    self.assertAlmostEqual(clouds[-1][0], -0.02)  # Haze over the rest.

  def test_seasons_are_a_truth_layer_with_static_ranges_sent_once(self):
    watched = session_lib.GameSession('great_acceleration', 'watch')
    full = watched.map(full=True)
    self.assertIn('seasons', full['layers'])
    biomes = watched.context.planet.biomes
    self.assertLen(full['seasonality'], biomes.seasonality.shape[0])
    # North first, like every tile layer; interiors far from the equator
    # swing most.
    self.assertEqual(
        full['seasonality'][0][0], round(biomes.seasonality[-1][0])
    )
    self.assertNotIn('seasonality', watched.map())  # Static: sent once.
    played = session_lib.GameSession('great_acceleration', 'play').map()
    self.assertNotIn('seasons', played['layers'])  # Truth waits.

  def test_the_trace_shows_readings_and_the_truth_only_once_revealed(self):
    watched = session_lib.GameSession('daisyworld', 'watch')
    for _ in range(4):
      watched.context.planet.advance()
    trace = watched.state()['trace']
    self.assertEqual(trace['band'], list(watched.context.planet.band()))
    self.assertLen(trace['true'], len(trace['ticks']))
    self.assertLen(trace['seen'], len(trace['ticks']))
    self.assertEqual(trace['reader'], 'Gaia')  # Whose readings they are.
    played = session_lib.GameSession('daisyworld', 'play').state()
    self.assertIsNotNone(played['trace'])  # The Steward sees its readings,
    self.assertIsNone(played['trace']['true'])  # but not the truth.
    ld = session_lib.GameSession('light_delay', 'watch').state()
    self.assertIsNone(ld['trace'])  # No Steward reads that planet.
    thaw = session_lib.GameSession('snowball', 'watch').state()['trace']
    self.assertIsNone(thaw['band'])  # A band that never binds is not drawn;
    self.assertEqual(thaw['target'], 0.0)  # the thaw's goal is.
    self.assertIsNone(trace['target'])
    people = session_lib.GameSession(
        'great_acceleration', 'play', role='Amber folk'
    )
    self.assertIsNone(people.state()['trace'])  # Not the Steward's counsel.

  def test_fire_risk_is_a_layer_only_where_something_can_burn(self):
    daisies = session_lib.GameSession('daisyworld', 'watch').map()
    grass = session_lib.GameSession('great_acceleration', 'watch').map()
    self.assertNotIn('fire', daisies['layers'])
    self.assertIn('fire', grass['layers'])

  def test_daughter_species_get_their_own_colours(self):
    colours = {
        'methanogens': {'light': [140, 84, 140], 'dark': [104, 60, 104]},
        'cyanobacteria': {'light': [52, 150, 140], 'dark': [36, 112, 104]},
    }
    taken = [c['light'] for c in colours.values()]
    daughter = session_lib._species_colours('cyanobacteria4', 0.4, colours, taken)  # pylint: disable=protected-access
    for c in taken:  # Far from every colour already on the map.
      self.assertGreater(
          sum(abs(a - b) for a, b in zip(daughter['light'], c)), 100
      )
    self.assertEqual(
        session_lib._species_colours('methanogens', 0.4, colours),  # pylint: disable=protected-access
        colours['methanogens'],
    )
    # A species keeps its colour for the whole game.
    game = session_lib.GameSession('great_oxidation', 'watch')
    first = game.map(None)['life_tiles']['species']
    again = game.map(None)['life_tiles']['species']
    self.assertEqual(first, again)

  def test_rejects_unknown_scenarios_modes_and_roles(self):
    with self.assertRaises(KeyError):
      session_lib.GameSession('atlantis')
    with self.assertRaises(ValueError):
      session_lib.GameSession('daisyworld', 'spectate')
    with self.assertRaises(KeyError):
      session_lib.GameSession('daisyworld', role='Amber folk')
    with self.assertRaises(KeyError):
      session_lib.GameSession('daisyworld', role='Chronicler')

  def test_scenario_list_names_each_role(self):
    found = session_lib.scenario_list()
    self.assertEqual([s['key'] for s in found], sorted(scenarios.SCENARIOS))
    self.assertTrue(all(s['title'] and s['premise'] for s in found))
    roles = {s['key']: [r['name'] for r in s['roles']] for s in found}
    self.assertEqual(
        roles['daisyworld'], ['Gaia', 'White daisies', 'Black daisies']
    )
    self.assertEqual(
        roles['light_delay'], ['Mission Control', 'Colonists of New Ararat']
    )
    self.assertEqual(roles['fire_and_iron'][0], 'Gaia')
    self.assertIn('Amber folk', roles['fire_and_iron'])
    self.assertIn('Grazers', roles['herds'])

  def test_hunger_is_seen_on_the_map(self):
    game = session_lib.GameSession('sandbox', 'play')
    peoples = game.map()['peoples']
    self.assertTrue(all(p['fed'] == 1.0 for p in peoples['polities']))
    self.assertTrue(all(not p['famine'] for p in peoples['polities']))
    civ = game.context.planet.civ
    assert civ is not None
    polity = civ.polities['cedar']
    polity.food_ratio = 0.6
    cedar = next(
        p for p in game.map()['peoples']['polities'] if p['key'] == 'cedar'
    )
    self.assertEqual(cedar['fed'], 0.6)  # Public, like the towns.

  def test_ripples_say_what_happened_to_those_who_may_read_it(self):
    def ripples(game):
      planet = game.context.planet
      for _ in range(12):
        if planet.ripples:
          break
        planet.advance()
      self.assertNotEmpty(planet.ripples)
      return game.map()['ripples']

    watched = ripples(session_lib.GameSession('great_acceleration', 'watch'))
    self.assertTrue(all(isinstance(r[4], str) and r[4] for r in watched))
    for r in watched:  # A tally of shifting biomes ripples as a 'shift'.
      self.assertEqual(r[3] == 'shift', bool(TALLY.search(r[4])))
    gaia = ripples(session_lib.GameSession('great_acceleration', 'play'))
    self.assertTrue(all(r[4] for r in gaia))  # The Steward hears world news.
    folk = ripples(
        session_lib.GameSession('great_acceleration', 'play', role='Amber folk')
    )
    for r in folk:  # The peoples' own news at once; the rest by couriers.
      self.assertEqual(r[4] is not None, r[3] == 'civ')
    pair = session_lib.GameSession(
        'great_acceleration', 'play', players=['Gaia', 'Amber folk']
    )
    planet = pair.context.planet
    while not planet.ripples:
      planet.advance()
    self.assertTrue(all(r[4] is None for r in pair.map('stranger')['ripples']))

  def test_a_peoples_player_sees_its_own_people(self):
    game = session_lib.GameSession('fire_and_iron', 'play', role='Amber folk')
    for _ in range(2):  # The first harvests fill the storehouse.
      game.context.planet.advance()
    people = game.state()['people']
    self.assertEqual(people['name'], 'Amber folk')
    self.assertAlmostEqual(sum(l['share'] for l in people['labor']), 1.0, 2)
    self.assertIn('farm the land', [l['label'] for l in people['labor']])
    self.assertIn('grain', people['stocks'])
    self.assertAlmostEqual(
        people['seasons'], people['stocks']['grain'] / people['population'], 0
    )
    self.assertEqual(people['shared']['need'], 110.0)  # The iron goal.
    self.assertIsNone(
        session_lib.GameSession('fire_and_iron', 'play').state()['people']
    )
    self.assertEqual(people['trades']['farm the land'], 'farm')  # Every trade.
    pair = session_lib.GameSession(
        'fire_and_iron', 'play', players=['Amber folk', 'Basalt clans']
    )
    self.assertIsNone(pair.state('stranger')['people'])  # A spectator: none.

  def test_holding_course_plays_the_neutral_move_until_it_is_needed(self):
    game = session_lib.GameSession('daisyworld', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    _wait_for(lambda: game.state()['question'])
    game.seats[game.entity].hold_pace = 0.05  # Quick turns for the test.
    self.assertEqual(game.state()['hold'], {'turns': 0, 'stopped': ''})
    self.assertIsNone(game.hold_course(None, 3))  # Answers the pending turn.
    _wait_for(lambda: game.state()['hold']['turns'] == 0)
    held = [c for c in game.choices if c.startswith('(holding course)')]
    self.assertLen(held, 3)
    self.assertTrue(all(c.endswith('wait and save omega') for c in held))
    self.assertEqual(game.state()['hold']['turns'], 0)
    # When something needs the player, holding stops by itself and says why.
    seat = game.seats[game.entity]
    seat.guard = lambda: 'the climate has left its band'
    self.assertIsNone(game.hold_course(None, 5))
    _wait_for(lambda: game.state()['hold']['stopped'])
    self.assertEqual(
        game.state()['hold'],
        {'turns': 0, 'stopped': 'the climate has left its band'},
    )
    watched = session_lib.GameSession('daisyworld', 'watch')
    self.assertIsNotNone(watched.hold_course(None, 3))  # Players only.

  def test_a_planned_move_is_made_on_its_turn_and_counted_ahead(self):
    game = session_lib.GameSession('daisyworld', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    question = _question(game)
    seat = game.seats[game.entity]
    seat.hold_pace = 0.05
    clouds = next(o for o in question['options'] if 'bright clouds' in o)
    now = game.state()['tick']
    self.assertIsNotNone(game.plan(None, 'wait and save omega', 2))
    self.assertIsNotNone(game.plan(None, clouds, 9))  # Too far ahead.
    self.assertIsNone(game.plan(None, clouds, 2))
    self.assertIsNotNone(game.plan(None, clouds, 2))  # One move a turn.
    state = game.state()
    self.assertEqual(
        state['plans'], [{'tick': now + 2, 'move': clouds, 'in': 2}]
    )
    forecast = (state['trace'] or {}).get('forecast')
    if forecast is not None:  # Readings yet: the course counts the plan.
      self.assertEqual(forecast['planned'], [[now + 2, 'bright_clouds']])
    self.assertIsNone(game.hold_course(None, 4))  # Wait until the plan.
    _wait_for(
        lambda: not game.state()['plans'] and game.state()['tick'] > now + 2
    )
    planned = [t for t in seat.played if t['how'] == 'planned']
    self.assertLen(planned, 1)
    self.assertEqual(
        (planned[0]['tick'], planned[0]['move']), (now + 2, clouds)
    )
    self.assertIn((now + 2, 'bright_clouds'), game.context.planet.made)
    watched = session_lib.GameSession('daisyworld', 'watch')
    self.assertIsNotNone(watched.plan(None, clouds, 2))  # Players only.

  def test_a_plan_that_cannot_be_made_asks_the_player_and_says_why(self):
    game = session_lib.GameSession('daisyworld', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    question = _question(game)
    clouds = next(o for o in question['options'] if 'bright clouds' in o)
    now = game.state()['tick']
    self.assertIsNone(game.plan(None, clouds, 1))
    game.context.planet.omega = 0.0  # Spent before the plan's turn.
    game.choose(question['request_id'], 'wait and save omega')
    _wait_for(
        lambda: game.state()['question']
        and game.state()['tick'] == now + 1
        and game.state()['hold']['stopped']
    )
    self.assertStartsWith(
        game.state()['hold']['stopped'],
        'your plan to seed bright clouds failed: it needs',
    )
    self.assertEqual(game.state()['plans'], [])
    self.assertEqual(
        game.cancel_plan(None, 99), 'Nothing is planned for turn 99.'
    )

  def test_a_held_steward_stops_on_its_own_thermometer_not_the_truth(self):
    game = session_lib.GameSession('daisyworld', 'play')
    low, high = game.context.planet.band()
    reading = {'temp_c': None}
    game.context.believed = lambda owner, topic, path: reading['temp_c']
    # What the World GM publishes for the seat (`views`): its own readings.
    needs = lambda entity: views.needs_you(game.context, entity)
    self.assertEqual(needs('Gaia'), '')  # No reading yet: no reason.
    reading['temp_c'] = (low + high) / 2
    self.assertEqual(needs('Gaia'), '')
    reading['temp_c'] = high + 5
    self.assertEqual(needs('Gaia'), 'your thermometer reads outside the band')

  def test_a_held_people_stops_for_a_treaty_offer(self):
    game = session_lib.GameSession(
        'great_acceleration', 'play', role='Amber folk'
    )
    treaties = game.context.planet.treaties
    self.assertIsNotNone(treaties)
    needs = lambda entity: views.needs_you(game.context, entity)
    self.assertEqual(needs('Amber folk'), '')
    actor = game.context.actors['Amber folk']
    assert isinstance(actor, actors_lib.PolityActor)
    polity = actor.polity
    treaties.pending[polity] = treaties_lib.Proposal(
        'basalt', polity, 'cap coal at 5 a season'
    )
    self.assertEqual(needs('Amber folk'), 'an offer awaits your answer')

  def test_a_decided_game_is_recapped(self):
    game = session_lib.GameSession('fire_and_iron', 'play', role='Amber folk')
    self.assertIsNone(game.state()['recap'])
    planet = game.context.planet
    for _ in range(3):
      planet.advance()
    seat = game.seats['Amber folk']
    seat.choices += [
        'put more labor into: farm the land',
        'put more labor into: farm the land',
        '(holding course) keep the current balance of labor',
        'plan where our next town will stand',
    ]
    planet.outcome, planet.outcome_reason = 'defeat', 'only 90 iron was forged'
    recap = game.state()['recap']
    self.assertEqual(recap['outcome'], 'defeat')
    self.assertLen(recap['temps'], len(planet.history))
    self.assertEqual(
        recap['moves'][0], ['put more labor into: farm the land', 2]
    )
    self.assertEqual(recap['held'], 1)
    self.assertEqual(recap['band'], list(planet.band()))
    self.assertIsNone(recap['play'])  # No turn was played through the page.
    self.assertIsNone(game.state()['hold'])  # Nothing left to hold.
    self.assertEqual(game.hold_course(None, 5), 'The game is decided.')

  def test_the_recap_marks_where_each_move_acted(self):
    game = session_lib.GameSession('daisyworld', 'play')
    planet = game.context.planet
    for _ in range(4):
      planet.advance()
    clouds = next(
        i.option for i in planet.interventions() if i.key == 'bright_clouds'
    )
    turn = {'offered': [], 'pushes': 0, 'seconds': None}
    game.seats['Gaia'].played += [
        {**turn, 'tick': 1, 'move': clouds, 'how': 'chose'},
        {**turn, 'tick': 2, 'move': 'wait and save omega', 'how': 'chose'},
        {**turn, 'tick': 3, 'move': 'wait and save omega', 'how': 'held'},
    ]
    planet.outcome, planet.outcome_reason = 'victory', 'the planet stayed alive'
    recap = game.state()['recap']
    # Acted the turn after it was chosen; cooled, as measured on this world.
    self.assertEqual(recap['marks'], [[2, 'seed bright clouds', 'cool']])
    self.assertIsInstance(recap['seen'], list)  # The Steward's readings.
    # A cold snap after the clouds acted: a turning point, told with them.
    for row, temp in zip(planet.history[1:4], (20.0, 9.0, 2.0)):
      row['temp_c'] = temp
    turning = game.state()['recap']['turning']
    (point,) = [p for p in turning if p['to_c'] < p['from_c']]
    watched = session_lib.GameSession('daisyworld', 'watch')
    watched.context.planet.outcome = 'victory'
    self.assertEqual(watched.state()['recap']['turning'], [])  # No player.
    self.assertEqual((point['move'], point['move_tick']), (clouds, 1))
    self.assertLess(point['to_c'] - point['from_c'], -8)

  def test_each_role_sees_how_old_what_it_knows_is(self):
    game = session_lib.GameSession('daisyworld', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    game.seats[game.entity].hold_pace = 0.05
    _wait_for(lambda: game.state()['question'])
    game.hold_course(None, 4)
    _wait_for(lambda: game.state()['tick'] >= 4)
    state = game.state()
    timing = {t['label']: t for t in state['timing']}
    self.assertIn('Thermometer', timing)
    tick = state['tick']
    for t in timing.values():
      self.assertGreaterEqual(t['age'], 0)
      self.assertEqual(t['tick'], tick - t['age'])
    pair = session_lib.GameSession(
        'daisyworld', 'play', players=['Gaia', 'White daisies']
    )
    self.assertIsNone(pair.state('stranger')['timing'])  # A spectator: none.

  def test_a_people_sees_what_is_on_its_way_and_when_it_lands(self):
    game = session_lib.GameSession(
        'fire_and_iron', 'play', role='Amber folk'
    ).start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    game.seats[game.entity].hold_pace = 0.05
    _question(game)
    labels = set()
    # The World GM lists what it knows is on its way: a council that began
    # deliberating when the people was asked, from its next publication.
    for _ in range(4):
      tick = game.state()['tick']
      game.hold_course(None, 1)
      _wait_for(lambda: game.state()['tick'] > tick and _question(game))
      state = game.state()
      flight = state['in_flight']
      labels |= {f['label'] for f in flight}
      for f in flight:  # Everything listed is still to come.
        self.assertGreater(f['lands'], state['tick'])
      self.assertEqual(
          [f['lands'] for f in flight], sorted(f['lands'] for f in flight)
      )
      self.assertEqual(state['acts_in'], 1)
    self.assertIn('Council plan', labels)  # Its council deliberating.
    self.assertIn('Broadsheet', labels)  # Its press writing.

  def test_each_reading_comes_as_data_as_it_reached_the_role(self):
    game = session_lib.GameSession('daisyworld', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    game.seats[game.entity].hold_pace = 0.05
    _question(game)
    game.hold_course(None, 3)
    _wait_for(lambda: game.state()['tick'] >= 3 and game.state()['question'])
    found = {i['label']: i for i in game.state()['instruments']}
    thermometer = found['Thermometer']
    self.assertEqual(thermometer['topic'], 'climate.global')
    # As it arrived: late and noisy, the very reading the Steward believes.
    self.assertAlmostEqual(
        thermometer['data']['temp_c'],
        game.context.believed('Gaia', 'climate.global', 'temp_c'),
    )
    self.assertGreater(thermometer['age'], 0)
    census = found['Census']['data']
    self.assertEqual(census['look']['white']['name'], 'White daisies')
    # An instrument that has received nothing shows nothing (no leak).
    self.assertIsNone(found['Air gauge']['data'])
    news = found['World news']
    self.assertEqual(news['topic'], 'news')
    self.assertTrue(all('text' in item for item in news['data']['items']))
    # A people's subscriptions are found by their labels, its plan included.
    people = session_lib.GameSession(
        'great_acceleration', 'play', role='Amber folk'
    )
    topics = {i['label']: i['topic'] for i in people.instruments('Amber folk')}
    self.assertEqual(topics['Census of the peoples'], 'civ.census')
    self.assertEqual(topics['Current plan'], 'civ.amber.plan')
    # A species knows its own genome.
    grazers = session_lib.GameSession('herds', 'play', role='Grazers')
    (genome,) = [
        i for i in grazers.instruments('Grazers') if i['topic'] == 'genome'
    ]
    self.assertIn('white_diet', genome['data'])
    self.assertIsNone(
        session_lib.GameSession(
            'daisyworld', 'play', players=['Gaia', 'White daisies']
        ).state('stranger')['instruments']
    )

  def test_a_reading_swapped_out_of_attention_freezes_and_says_so(self):
    game = session_lib.GameSession('daisyworld', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    seat = game.seats[game.entity]
    question = _question(game)
    swap = 'watch the air gauge instead of the thermometer'
    self.assertIn(swap, question['options'])
    seat.choose(question['request_id'], swap)
    _wait_for(lambda: game.state()['tick'] >= 2 and game.state()['question'])
    frozen = {i['label']: i for i in game.state()['instruments']}
    self.assertTrue(frozen['Thermometer']['unwatched'])
    self.assertFalse(frozen['Air gauge']['unwatched'])
    seat.hold_pace = 0.05
    game.hold_course(None, 2)
    _wait_for(lambda: game.state()['tick'] >= 4 and game.state()['question'])
    later = {i['label']: i for i in game.state()['instruments']}
    # No new reading reaches an instrument no one is watching.
    self.assertEqual(
        later['Thermometer']['data'], frozen['Thermometer']['data']
    )
    air = later['Air gauge']['data']  # Published without a tick: given one.
    self.assertIsInstance(air['tick'], int)

  def test_a_played_game_writes_how_it_went(self):
    folder = self.create_tempdir().full_path
    game = session_lib.GameSession(
        'daisyworld', 'play', telemetry_dir=folder
    ).start()
    seat = game.seats[game.entity]
    question = _question(game)
    seed = next(o for o in question['options'] if o.startswith('seed white'))
    # What the player looked at before choosing: kept once each, in order.
    for what in ('sheet', 'card:storycard', 'sheet', '  card:goalcard '):
      self.assertIsNone(game.look(None, what))
    seat.choose(question['request_id'], seed)
    _wait_for(lambda: game.state()['tick'] >= 2 and game.state()['question'])
    seat.hold_pace = 0.05
    game.hold_course(None, 1)
    _wait_for(lambda: game.state()['tick'] >= 3 and game.state()['question'])
    game.cancel()
    game.join(30)
    (name,) = [n for n in os.listdir(folder) if n.endswith('.json')]
    with open(os.path.join(folder, name)) as f:
      report = json.load(f)
    self.assertEqual(report['scenario'], 'daisyworld')
    self.assertEqual(report['outcome'], 'abandoned')
    turns = report['seats']['Gaia']['turns']
    self.assertEqual([t['how'] for t in turns], ['chose', 'held'])
    first, held = turns
    self.assertEqual(first['move'], seed)
    self.assertFalse(first['neutral'])
    self.assertGreaterEqual(first['seconds'], 0)
    self.assertGreater(first['pushes'], 3)  # The Steward's measured moves.
    self.assertTrue(held['neutral'])
    self.assertIsNone(held['seconds'])
    self.assertEqual(
        first['looked'], ['sheet', 'card:storycard', 'card:goalcard']
    )
    self.assertEqual(held['looked'], [])  # A new turn starts with nothing.
    # What the Steward read and what was true, to review a turn by.
    self.assertGreaterEqual(first['omega'], 0)
    self.assertIsInstance(first['true_c'], float)
    self.assertIn('read_c', first)
    self.assertIn('forecast_c', first)

  def test_the_steward_reads_the_peoples_goal_on_its_strip(self):
    game = session_lib.GameSession('great_acceleration', 'play')
    context = game.context

    def reading(topic, payload, tick, kept):
      kept[('Gaia', topic)] = topics.Message(
          topic, json.dumps(payload), seq=0, time=tick
      )

    stats = game._stats(False, 'Gaia')  # pylint: disable=protected-access
    self.assertEqual(
        [s['label'] for s in stats][-2:], ['CO₂ / cap', 'energy made']
    )
    self.assertEqual([s['value'] for s in stats][-2:], ['?/1.3', '0/600'])
    context.planet.tick = 6
    # What the instruments last read, with its age; past the cap is marked.
    reading('climate.global', {'temp_c': 14.0, 'co2': 1.34}, 5, context.beliefs)
    # The peoples' census reaches the Steward at once (as 'Peoples').
    reading('Peoples', {'made': {'energy': 340.4}}, 6, context.received)
    context.publish_sheets()  # As the World GM does each step.
    co2, energy = game._stats(False, 'Gaia')[-2:]  # pylint: disable=protected-access
    self.assertEqual(
        (co2['label'], co2['value'], co2['warn']),
        ('CO₂ / cap, 1 tick old', '1.34/1.3', True),
    )
    self.assertEqual(
        (energy['label'], energy['value']), ('energy made', '340/600')
    )
    # Holding course stops for it.
    self.assertIn('past ×1.3 carbon', views.needs_you(game.context, 'Gaia'))
    # Worlds whose goal has no such numbers keep biodiversity.
    daisies = session_lib.GameSession('daisyworld', 'play')
    self.assertEqual(
        daisies._stats(False, 'Gaia')[-1]['label'], 'biodiversity'  # pylint: disable=protected-access
    )

  def test_the_steward_speaks_and_the_printers_retell_it(self):
    game = session_lib.GameSession('great_acceleration', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    question = _question(game)
    omen = question['omen']
    self.assertEqual(
        omen['signs'], ['a red dawn', 'a comet', 'ash on the snow']
    )
    self.assertTrue(omen['ready'])
    self.assertIn(omen['option'], question['options'])  # Moves stay offered.
    self.assertIn('wait and save omega', question['options'])
    words = 'a comet: spend the black stone slowly, or the fields go dry'
    self.assertIsNone(game.choose(question['request_id'], words))
    _wait_for(lambda: game.state()['tick'] >= 2 and game.state()['question'])
    self.assertEqual(game.context.planet.omens[0][1], 'a comet')
    self.assertFalse(game.state()['question']['omen']['ready'])  # Recharging.
    # Holding course works on a free question: the neutral move is offered.
    game.seats[game.entity].hold_pace = 0.05
    game.hold_course(None, 2)
    _wait_for(lambda: game.state()['tick'] >= 4 and game.state()['question'])
    told = [
        i
        for i in game.state()['story']
        if i['who'] == 'The Gazette on the omen'
    ]
    self.assertLen(told, 1)  # The printers' front page, in full, once.
    self.assertIn('spend the black stone slowly', told[0]['text'])
    gazette = game.context.planet.gazette
    assert gazette is not None
    self.assertEqual(gazette.printed[-1].kind, 'omen')
    held = [
        t['move'] for t in game.seats[game.entity].played if t['how'] == 'held'
    ]
    self.assertEqual(held, ['wait and save omega'] * 2)

  def test_a_turn_keeps_each_peoples_hands_for_review(self):
    game = session_lib.GameSession('fire_and_iron', 'play', role='Amber folk')
    world = game._turn_world('Amber folk')  # pylint: disable=protected-access
    hands = world['peoples']['Amber folk']['hands']
    self.assertLessEqual(len(hands), 3)  # The busiest trades.
    self.assertIn('farm', hands)
    self.assertEqual(world['peoples']['Basalt clans']['fed'], 1.0)
    steward = session_lib.GameSession('fire_and_iron', 'play')
    self.assertIn(
        'peoples', steward._turn_world('Gaia')  # pylint: disable=protected-access
    )

  def test_each_climate_move_says_what_the_forecast_says_it_brings(self):
    game = session_lib.GameSession('daisyworld', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    for _ in range(4):  # Readings to reckon from.
      question = _question(game)
      game.choose(question['request_id'], 'wait and save omega')
    question = _question(game)
    what_if = question['what_if']
    ash = next(o for o in question['options'] if o.startswith('bury the sky'))
    clouds = next(o for o in question['options'] if 'bright clouds' in o)
    self.assertGreater(what_if[clouds]['then'], what_if[ash]['then'] + 30)
    self.assertEqual(
        (what_if[ash]['leaves_in'], what_if[ash]['edge']), (1, 'bottom')
    )
    self.assertIsNone(what_if[clouds]['leaves_in'])
    self.assertNotIn('wait and save omega', what_if)  # No profile: no what-if.

  def test_a_player_sees_life_and_their_sowing_as_their_census_saw_them(self):
    game = session_lib.GameSession('daisyworld', 'play', seed=7).start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    first = game.map(None)
    self.assertNotIn('life', first['layers'])  # No census read yet.
    for _ in range(3):
      question = _question(game)
      game.choose(question['request_id'], 'wait and save omega')
    question = _question(game)
    seen = game.map(None)
    self.assertEqual(seen['layers'][0], 'life')  # Daisies are this world.
    self.assertEqual(
        seen['life_seen']['age'], seen['tick'] - seen['life_seen']['tick']
    )
    self.assertGreater(seen['life_seen']['age'], 0)  # As late as the census.
    sow = next(o for o in question['options'] if o.startswith('seed black'))
    sown_on = game.state()['tick']
    game.choose(question['request_id'], sow, target=(13, 24))
    self.assertEqual(game.map(None)['plantings'], [])  # Not yet seen.
    plantings = []
    for _ in range(6):  # Until a census taken after it is read.
      question = _question(game)
      plantings = game.map(None)['plantings']
      if plantings:
        break
      game.choose(question['request_id'], 'wait and save omega')
    (planted,) = plantings
    self.assertEqual(
        (planted['name'], planted['sown']), ('Black daisies', sown_on)
    )
    self.assertGreater(sum(row.count('#') for row in planted['tiles']), 0)

  def test_fire_danger_and_band_temperatures_as_the_census_saw_them(self):
    game = session_lib.GameSession('savanna_fire', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    for _ in range(3):
      question = _question(game)
      game.choose(question['request_id'], 'wait and save omega')
    _question(game)
    seen = game.map(None)
    self.assertEqual(seen['layers'][-2:], ['temperature', 'fire'])
    lats = [lat for lat, _ in seen['band_temps']]
    self.assertEqual(lats, sorted(lats))  # South first, by band.
    self.assertLen(seen['fire_risk'], seen['tile_rows'])
    self.assertIsNone(seen['tile_temp'])  # Tile by tile: the truth, later.

  def test_a_held_course_says_what_came_of_it(self):
    game = session_lib.GameSession('daisyworld', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    game.seats['Gaia'].hold_pace = 0.01
    question = _question(game)
    game.choose(question['request_id'], 'wait and save omega')
    _question(game)
    self.assertIsNone(game.state()['held_run'])  # Chosen, not held.
    game.hold_course(None, 3)
    run = _wait_for(
        lambda: game.state()['question'] and game.state()['held_run']
    )
    self.assertEqual((run['turns'], run['to'] - run['from']), (3, 2))
    self.assertLen(run['omega'], 2)
    self.assertGreater(run['omega'][1], run['omega'][0])  # Saved while held.
    question = _question(game)
    game.choose(question['request_id'], 'wait and save omega')
    _question(game)
    self.assertIsNone(game.state()['held_run'])  # Gone once they move.

  def test_the_advisors_pick_says_how_old_its_counsel_is(self):
    game = session_lib.GameSession('snowball', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    self.assertIsNone(_question(game)['advice'])  # No briefing yet.
    advice = None
    for _ in range(8):
      question = _question(game)
      advice = question['advice']
      if advice:
        break
      game.choose(question['request_id'], 'wait and save omega')
    assert advice is not None
    self.assertIn(advice['move'], [*question['options'], *question['saving']])
    self.assertGreaterEqual(advice['age'], 2)  # The Advisor's own delay.
    # The briefing's age on the page is the same: from the readings it used.
    (briefing,) = [t for t in game.timing() if t['label'] == 'Advisor briefing']
    self.assertGreaterEqual(briefing['age'], 2)

  def test_a_peoples_question_names_what_matters_most(self):
    game = session_lib.GameSession(
        'fire_and_iron', 'play', role='Basalt clans'
    ).start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    question = _question(game)  # Basalt starts short of bread.
    (famine,) = [p for p in question['pressing'] if p['kind'] == 'famine']
    self.assertEqual(famine['move'], 'put more labor into: farm the land')
    self.assertIn(famine['move'], question['options'])
    self.assertIn('Most pressing now:', question['call_to_action'])
    steward = session_lib.GameSession('fire_and_iron', 'play').start()
    self.addCleanup(steward.join, 30)
    self.addCleanup(steward.cancel)
    self.assertEqual(_question(steward)['pressing'], [])  # Peoples only.

  def test_each_move_comes_back_with_what_it_did_and_where(self):
    game = session_lib.GameSession('daisyworld', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    question = _question(game)
    sent_on = game.state()['tick']
    self.assertEqual(game.state()['moves'], [])
    seed = next(o for o in question['options'] if o.startswith('seed white'))
    self.assertIsNone(game.choose(question['request_id'], seed, target=(9, 20)))
    (move,) = game.state()['moves']  # Sent at once, its outcome to come.
    self.assertEqual((move['tick'], move['move']), (sent_on, seed))
    self.assertEqual((move['how'], move['target']), ('chose', [9, 20]))
    _wait_for(lambda: game.state()['tick'] > sent_on)
    move = _wait_for(
        lambda: game.state()['moves'][0]['outcome'] and game.state()['moves'][0]
    )
    self.assertIn('white daisies', move['outcome'].lower())
    self.assertEqual(game.map(None)['pins'], [[9, 20, sent_on, seed]])
    # The turn it acts, the map shows it acting where it was aimed.
    self.assertEqual(game.state()['tick'], sent_on + 1)
    (acting,) = game.map(None)['acting']
    self.assertEqual(
        (acting['motion'], acting['key'], acting['target'], acting['move']),
        ('sprout', 'seed_white', [9, 20], seed),
    )
    # Watchers have no moves and no pins.
    watched = session_lib.GameSession('daisyworld', 'watch')
    self.assertIsNone(watched.state()['moves'])
    self.assertEqual(watched.map(None)['pins'], [])

  def test_news_on_the_road_rides_to_the_capital_on_the_map(self):
    game = session_lib.GameSession('fire_and_iron', 'play', role='Amber folk')
    planet = game.context.planet
    assert planet.civ is not None and planet.settlements is not None
    home = planet.civ.polities['amber'].home
    rows, cols = np.nonzero(planet.tiles.region >= 0)
    row, col = next(
        (int(r), int(c))
        for r, c in zip(rows, cols)
        if planet.region_steps(home, int(planet.tiles.region[r, c])) == 2
    )
    planet.ripples.append({
        'tick': planet.tick,
        'row': row,
        'col': col,
        'topic': 'civ.events',
        'text': 'A town is founded.',
        'local': True,
    })
    game.context.publish_sheets()  # As the World GM does each step.
    (news,) = [f for f in game.in_flight() if f['kind'] == 'news']
    per = float(planet.news_config['ticks_per_region'])
    self.assertEqual(news['lands'], planet.tick + round(2 * per))
    north = planet.tiles.rows - 1
    self.assertEqual(news['from'], [north - row, col])
    capital = planet.settlements.places['amber'][0]
    self.assertEqual(news['to'], [north - capital.row, capital.col])
    # Cut from the map (a tap only zoomed it); a data flag brings it back.
    self.assertEqual(game.map(None)['couriers'], [])
    planet.settlements.config = {**planet.settlements.config, 'couriers': True}
    self.assertEqual(
        game.map(None)['couriers'],
        [[*news['from'], *news['to'], planet.tick, news['lands'], 'civ']],
    )
    # Only a people's player has couriers: the ripples show everyone else.
    watched = session_lib.GameSession('fire_and_iron', 'watch')
    self.assertEqual(watched.map(None)['couriers'], [])

  def test_mission_control_sees_its_transmissions_cross_the_void(self):
    game = session_lib.GameSession('light_delay', 'play').start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    seat = game.seats[game.entity]
    question = _wait_for(lambda: game.state()['question'])
    design = next(o for o in question['options'] if o.startswith('transmit'))
    seat.choose(question['request_id'], design)
    _wait_for(lambda: game.state()['tick'] >= 2 and game.state()['question'])
    state = game.state()
    colony = game.context.planet.colony
    assert colony is not None
    delay = colony.light_delay
    self.assertEqual(state['acts_in'], delay)
    flight = state['in_flight']
    self.assertEqual([f['kind'] for f in flight], ['wire'])
    self.assertEqual(flight[0]['lands'], 1 + delay)  # Sent on turn 1.
    self.assertIn('Design for New Ararat', flight[0]['about'])


def _wait_to(game: session_lib.GameSession, tick: int) -> None:
  """Answers the first seat's turns with its neutral move until `tick`."""
  deadline = time.monotonic() + 120
  while time.monotonic() < deadline:
    state = game.state()
    question = state['question']
    if question and state['tick'] >= tick:
      return
    if question:
      game.choose(
          question['request_id'], session_lib.neutral(question['options'])
      )
    time.sleep(0.02)
  raise AssertionError(f'did not reach tick {tick}')


class CoopTest(absltest.TestCase):

  def test_two_phone_worlds_name_their_seats_and_share_crises(self):
    coops = {
        s['key']: s['coop'] for s in session_lib.scenario_list() if s['coop']
    }
    self.assertEqual(
        coops['light_delay']['players'],
        ['Mission Control', 'Colonists of New Ararat'],
    )
    self.assertEqual(coops['herds']['players'], ['Gaia', 'Grazers'])
    for key, coop in coops.items():
      seats = dict(session_lib.roles(key))
      self.assertContainsSubset(coop['players'], seats)
    game = session_lib.GameSession(
        'light_delay', 'play', players=coops['light_delay']['players']
    ).start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    colonists = 'Colonists of New Ararat'

    def troubled():  # Both seats hold course until the colonists see famine.
      for seat in game.seats:
        question = game.state(seat)['question']
        if question:
          game.choose(
              question['request_id'], session_lib.neutral(question['options'])
          )
      found = game.state(colonists)['crises']
      return found if 'famine' in {c['kind'] for c in found} else None

    found = _wait_for(troubled)
    self.assertIn('famine', {c['kind'] for c in found})  # Stores run out.


class ReplayTest(absltest.TestCase):

  def test_a_decided_game_replays_exactly(self):
    folder = self.create_tempdir().full_path
    game = session_lib.GameSession(
        'snowball', 'play', seed=2, telemetry_dir=folder
    ).start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    deadline = time.monotonic() + 120
    while not game.finished and time.monotonic() < deadline:
      question = game.state()['question']
      if question:  # Rouse the volcanoes when offered, else wait.
        pick = next(
            (o for o in question['options'] if 'volcanoes' in o),
            session_lib.neutral(question['options']),
        )
        game.choose(question['request_id'], pick)
      time.sleep(0.02)
    self.assertTrue(game.context.planet.outcome)
    (listed,) = session_lib.list_replays(folder)
    self.assertEqual(listed['game_id'], game.game_id)
    self.assertNotIn('moves', listed)
    record = session_lib.load_replay(
        os.path.join(folder, session_lib.REPLAYS, f'{game.game_id}.json')
    )
    assert record is not None
    self.assertTrue(
        any('volcanoes' in m[1] for m in record['moves'][game.entity])
    )
    again = session_lib.GameSession(
        'snowball',
        'watch',
        seed=2,
        pace_seconds=0.0,
        players=record['players'],
        replay=record,
    ).start()
    self.addCleanup(again.join, 30)
    self.addCleanup(again.cancel)
    _wait_for(lambda: again.finished, timeout=120)
    self.assertEqual(again.replay_of, game.game_id)

    def history(g):
      return json.dumps(g.context.planet.history, sort_keys=True, default=str)

    self.assertEqual(history(again), history(game))
    self.assertEqual(again.context.planet.outcome, game.context.planet.outcome)


class SaveTest(absltest.TestCase):

  def _game(self, folder: str, resume=None) -> session_lib.GameSession:
    game = session_lib.GameSession(
        'daisyworld', 'play', seed=3, telemetry_dir=folder, resume=resume
    )
    game.start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    return game

  def test_a_saved_game_plays_on_exactly_after_a_restart(self):
    folder = self.create_tempdir().full_path
    first = self._game(folder)
    _wait_to(first, 5)
    path = os.path.join(folder, session_lib.SAVES, f'{first.game_id}.json')
    save = session_lib.load_save(path)
    assert save is not None
    saved_at = save['tick']
    # Saved at the end of the last step, before the world turned again.
    self.assertEqual(saved_at, first.context.planet.tick - 1)
    latest = session_lib.latest_save(folder)
    assert latest is not None
    self.assertEqual(latest['game_id'], first.game_id)
    # A new server: a fresh session from the save plays the same turns on.
    second = self._game(self.create_tempdir().full_path, resume=save)
    self.assertEqual(second.context.planet.tick, saved_at)
    self.assertEqual(second.game_id, first.game_id)
    self.assertEqual(second.choices, first.choices)
    for game in (first, second):
      _wait_to(game, saved_at + 6)

    def after(game):
      return json.dumps(
          [r for r in game.context.planet.history if r['tick'] > saved_at],
          sort_keys=True,
          default=str,
      )

    self.assertEqual(after(second), after(first))
    self.assertEqual(second.choices, first.choices)

  def test_a_given_up_game_leaves_no_save(self):
    folder = self.create_tempdir().full_path
    game = self._game(folder)
    _wait_to(game, 2)
    self.assertIsNotNone(session_lib.latest_save(folder))
    game.cancel()
    game.join(30)
    self.assertIsNone(session_lib.latest_save(folder))
    self.assertIsNone(session_lib.latest_save(''))


class StoryTest(absltest.TestCase):

  def _play_to(self, players, tick):
    game = session_lib.GameSession('fire_and_iron', 'play', players=players)
    game.start()
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    for name in players:
      game.seats[name].hold_pace = 0.0

    def step():
      for name in players:
        seat = game.seats[name]
        request = seat.pending()
        if request is not None:
          seat.choose(
              request.request_id,
              session_lib.neutral(seat.offered(request)),
          )
      return game.context.planet.tick >= tick

    _wait_for(step)
    return game

  def test_each_role_reads_its_own_story(self):
    game = self._play_to(['Gaia', 'Amber folk'], 8)
    steward = game.state('Gaia')['story']
    people = game.state('Amber folk')['story']
    self.assertIn('advisor', {i['kind'] for i in steward})
    self.assertIn('news', {i['kind'] for i in steward})
    self.assertNotIn('advisor', {i['kind'] for i in people})  # Not its Advisor.
    for story in (steward, people):
      self.assertNotIn('chronicle', {i['kind'] for i in story})  # Not yet.
      ticks = [i['tick'] for i in story]
      self.assertEqual(ticks, sorted(ticks, reverse=True))  # Newest first.
      self.assertLessEqual(len(story), session_lib.STORY_ITEMS)


if __name__ == '__main__':
  absltest.main()
