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

"""Tests of the Steward's forecast."""

import contextlib
import io

from absl.testing import absltest
from examples.deep_time import forecast
from examples.deep_time import game as game_lib
from examples.deep_time import model_setup
from examples.deep_time import simulation

CLOUDS = {'bright_clouds': [-6.0, -4.0, -2.0, -1.0, -0.5, -0.2]}


def _rows(values, latency=1, sigma=1.0, start=0):
  """History rows: each tick's reading shows the planet `latency` before."""
  return [
      {
          'tick': start + i,
          'temp_c': v,
          'believed_temp_c': v,
          'read_tick': start + i - latency,
          'read_sigma': sigma,
      }
      for i, v in enumerate(values)
  ]


class ForecastTest(absltest.TestCase):

  def test_moves_made_since_the_reading_are_reckoned_in(self):
    rows = _rows([30.0, 30.0, 30.0, 30.0, 30.0])
    # Clouds seeded at tick 3 act at tick 4; the reading at tick 4 shows 3.
    found = forecast.forecast(rows, [(3, 'bright_clouds')], CLOUDS, tick=4)
    assert found is not None
    self.assertEqual(found.unseen, -6.0)
    self.assertAlmostEqual(found.now, 24.0, delta=0.6)
    # Next turn the clouds fade a little: back up about 2 degrees.
    self.assertAlmostEqual(found.ahead - found.now, 2.0, delta=0.2)

  def test_without_moves_the_forecast_is_the_readings(self):
    rows = _rows([20.0, 20.0, 20.0])
    for row in rows[1:]:  # Nothing new arrived: the reading of tick 0 held.
      row['read_tick'] = rows[0]['read_tick']
    found = forecast.forecast(rows, [], CLOUDS, tick=2)
    assert found is not None
    self.assertEqual((found.now, found.ahead, found.unseen), (20.0, 20.0, 0.0))
    self.assertEqual(found.readings, 1)  # A held reading counts once.
    self.assertIsNone(forecast.forecast([], [], CLOUDS, tick=2))

  def test_the_range_widens_ahead_and_with_a_restless_planet(self):
    calm = forecast.forecast(_rows([20.0, 20.5, 19.8, 20.2]), [], {}, tick=3)
    wild = forecast.forecast(_rows([20.0, 27.0, 14.0, 22.0]), [], {}, tick=3)
    assert calm is not None and wild is not None
    self.assertGreater(calm.ahead_spread, calm.now_spread)
    self.assertGreater(wild.now_spread, 2 * calm.now_spread)

  def test_late_readings_widen_the_range(self):
    prompt = forecast.forecast(_rows([20.0] * 4, latency=0), [], {}, tick=3)
    late = forecast.forecast(_rows([20.0] * 4, latency=2), [], {}, tick=3)
    assert prompt is not None and late is not None
    self.assertGreater(late.now_spread, prompt.now_spread)

  def test_the_course_says_when_fading_moves_let_the_planet_leave(self):
    rows = _rows([29.0] * 5)
    found = forecast.forecast(
        rows, [(3, 'bright_clouds')], CLOUDS, tick=4, band=(5.0, 27.0)
    )
    assert found is not None
    self.assertLen(found.course, 6)
    self.assertEqual((found.leaves_in, found.edge), (3, 'top'))
    text = forecast.words(found)
    self.assertIn('Your moves since that reading have cooled the planet', text)
    self.assertIn('Your forecast: about 23 ± ', text)
    self.assertIn('leaves the band by the top in about 3 turns', text)

  def test_moves_and_profiles_reach_the_planet_and_survive_a_checkpoint(self):
    planet = game_lib.new_game('daisyworld').planet
    self.assertIn('bright_clouds', planet.move_profiles)
    (clouds,) = [i for i in planet.interventions() if i.key == 'bright_clouds']
    planet.omega = 50
    planet.apply_intervention('Gaia', clouds)
    self.assertEqual(planet.made, [(0, 'bright_clouds')])
    state = planet.get_state()
    planet.made = []
    planet.set_state(state)
    self.assertEqual(planet.made, [(0, 'bright_clouds')])

  def test_in_a_real_game_the_forecast_beats_the_raw_readings(self):
    # Calibrated over every scenario (data/forecast.yaml); one game here.
    sim, context = simulation.build_simulation(
        'great_oxidation', model_setup.make_models('mock'), seed=0
    )
    with contextlib.redirect_stdout(io.StringIO()):
      sim.play(premise='', max_steps=context.planet.max_ticks)
    planet = context.planet
    self.assertTrue(planet.made)
    self.assertTrue(any('read_tick' in r for r in planet.history))
    record = forecast.track_record(
        planet.history, planet.made, planet.move_profiles
    )
    assert record is not None
    self.assertLess(record['now_error_c'], record['raw_error_c'])
    self.assertBetween(record['inside_share'], 0.65, 0.97)


class WhatIfTest(absltest.TestCase):

  def _found(self, now, course):
    return forecast.Forecast(
        tick=10,
        now=now,
        now_spread=2.0,
        ahead=course[0],
        ahead_spread=3.0,
        acts_in=1,
        unseen=0.0,
        readings=4,
        course=tuple(course),
    )

  def test_a_move_adds_its_profile_to_the_course_from_the_turn_it_acts(self):
    profiles = {'clouds': [-6.0, -4.0, -2.0], 'ash': [-50.0, -50.0, -50.0]}
    found = self._found(20.0, [20.0, 20.0, 20.0, 20.0])
    clouds = forecast.what_if(found, profiles, 'clouds', (10.0, 35.0))
    assert clouds is not None
    self.assertEqual(clouds['then'], 14.0)
    self.assertEqual(
        clouds['course'], [14.0, 16.0, 18.0, 18.0]
    )  # Holds its last.
    self.assertEqual(clouds['spread'], 3.0)
    self.assertIsNone(clouds['leaves_in'])
    ash = forecast.what_if(found, profiles, 'ash', (10.0, 35.0))
    assert ash is not None
    self.assertEqual((ash['leaves_in'], ash['edge']), (1, 'bottom'))
    self.assertIsNone(forecast.what_if(found, profiles, 'wait'))
    self.assertIsNone(forecast.what_if(None, profiles, 'clouds'))

  def test_a_move_can_bring_the_world_back_into_the_band(self):
    profiles = {'clouds': [-6.0, -8.0, -10.0]}
    found = self._found(41.0, [42.0, 43.0, 44.0])
    back = forecast.what_if(found, profiles, 'clouds', (10.0, 35.0))
    assert back is not None
    self.assertEqual(back['course'], [36.0, 35.0, 34.0])
    self.assertEqual(back['returns_in'], 2)
    self.assertIsNone(back['leaves_in'])


class TurningPointsTest(absltest.TestCase):

  def test_the_steepest_swings_are_told_with_the_move_before_them(self):
    # The owner's Herds game (cycle 149): two ash winters and a rescue.
    true = {7: 21.4, 8: 19.9, 9: 3.6, 10: 15.6, 11: 22.3, 12: 27.1, 13: 28.7}
    true.update({27: 45.2, 28: 35.4, 29: 24.2, 30: -42.2, 31: -40.3})
    true.update({32: -35.8, 33: 25.3, 34: 18.0})
    history = [{'tick': t, 'temp_c': v} for t, v in sorted(true.items())]
    ash = 'bury the sky in ash (a volcanic winter, for years) [6 omega]'
    black = 'seed black daisies [2 omega]'
    played = [
        {'tick': 7, 'move': black, 'how': 'chose', 'true_c': 21.4},
        {
            'tick': 8,
            'move': ash,
            'how': 'chose',
            'read_c': 19.4,
            'forecast_c': 21.9,
            'true_c': 19.9,
        },
        {
            'tick': 29,
            'move': ash,
            'how': 'chose',
            'read_c': 46.0,
            'forecast_c': 35.4,
            'true_c': 24.2,
        },
        {
            'tick': 32,
            'move': black,
            'how': 'chose',
            'read_c': -44.1,
            'forecast_c': -24.8,
            'true_c': -35.8,
        },
    ]
    found = forecast.turning_points(
        history, played, (10.0, 35.0), 1, {ash: 'cool', black: 'warm'}
    )
    self.assertEqual(
        [(p['from'], p['way'], p['move_tick']) for p in found],
        [(7, 'left', 8), (28, 'across', 29), (31, 'back', 32)],
    )
    self.assertEqual(
        (found[1]['read_c'], found[1]['forecast_c'], found[1]['true_c']),
        (46.0, 35.4, 24.2),
    )
    calm = forecast.turning_points(
        [{'tick': t, 'temp_c': 20.0 + 0.5 * t} for t in range(20)],
        [],
        (10.0, 35.0),
        1,
        {},
    )
    self.assertEqual(calm, [])  # No swing of 8 °C in two turns.


if __name__ == '__main__':
  absltest.main()
