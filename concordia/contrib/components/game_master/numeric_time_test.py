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

"""Tests for numeric time."""

from absl.testing import absltest
from absl.testing import parameterized
from concordia.components.game_master import interrupt_scheduling
from concordia.components.game_master import interrupt_time_model
from concordia.contrib.components.game_master import numeric_time


class ParseDurationTest(parameterized.TestCase):

  @parameterized.parameters(
      ('10Myr', 1e7),
      ('0.5 Myr', 5e5),
      ('100yr', 100.0),
      ('3', 3.0),
      ('2kyr 500yr', 2500.0),
      ('1Gyr', 1e9),
      ('-2.5Gyr', -2.5e9),
      ('1e3yr', 1e3),
      ('6mo', 0.5),
  )
  def test_parses_year_units(self, text, expected):
    self.assertAlmostEqual(numeric_time.parse_duration(text), expected)

  def test_hours_and_minutes_match_interrupt_time_model(self):
    years = numeric_time.parse_duration('1h30m')
    seconds = interrupt_time_model.parse_duration_seconds('1h30m')
    self.assertAlmostEqual(years * 365.25 * 24 * 3600, seconds)

  @parameterized.parameters('', '5 parsecs', 'abc', '10Myr?')
  def test_rejects_invalid(self, text):
    with self.assertRaises(ValueError):
      numeric_time.parse_duration(text)

  def test_custom_units(self):
    turns = {'turn': 1.0, 'round': 4.0}
    self.assertEqual(
        numeric_time.parse_duration('2round 1turn', turns, 'turn'), 9.0
    )


class FormatYearsTest(parameterized.TestCase):

  @parameterized.parameters(
      (2.5e9, '2.5 Gyr'),
      (-3.8e9, '-3.8 Gyr'),
      (1.2e6, '1.2 Myr'),
      (1500.0, '1.5 kyr'),
      (12.0, '12 yr'),
      (0.0, '0 yr'),
      (0.5, '183 days'),
  )
  def test_formats(self, years, expected):
    self.assertEqual(numeric_time.format_years(years), expected)


class NumericTimeModelTest(absltest.TestCase):

  def test_is_time_model_and_round_trips(self):
    model = numeric_time.NumericTimeModel(start=-4e9)
    self.assertIsInstance(model, interrupt_time_model.TimeModel)
    t = model.add_duration(model.initial_time(), '10Myr')
    self.assertEqual(t, -3.99e9)
    self.assertEqual(model.deserialize_time(model.serialize_time(t)), t)
    self.assertEqual(model.format_time(t), '-3.99 Gyr')

  def test_parse_absolute_time_must_be_in_future(self):
    model = numeric_time.NumericTimeModel(start=0.0)
    self.assertEqual(model.parse_absolute_time('2kyr', 1000.0), 2000.0)
    with self.assertRaises(ValueError):
      model.parse_absolute_time('500yr', 1000.0)

  def test_timestamps_order_interrupt_events(self):
    model = numeric_time.NumericTimeModel(start=0.0)
    later = interrupt_scheduling.Event(
        timestamp=model.add_duration(0.0, '2Myr'),
        tag='geo.events',
        source='lithosphere',
        description='eruption',
    )
    sooner = interrupt_scheduling.Event(
        timestamp=model.add_duration(0.0, '1Myr'),
        tag='geo.events',
        source='lithosphere',
        description='quake',
    )
    self.assertEqual(sorted([later, sooner]), [sooner, later])

  def test_rejects_nan(self):
    with self.assertRaises(ValueError):
      numeric_time.NumericTimeModel().deserialize_time('nan')

  def test_custom_formatter(self):
    model = numeric_time.NumericTimeModel(formatter=lambda t: f'turn {t:g}')
    self.assertEqual(model.format_time(3.0), 'turn 3')


class NumericClockTest(absltest.TestCase):

  def test_advances_with_variable_steps(self):
    clock = numeric_time.NumericClock(
        numeric_time.NumericTimeModel(start=-4e9), step='10Myr'
    )
    clock.advance()
    clock.advance()
    self.assertEqual(clock.now(), -3.98e9)
    clock.set_step('0.5Myr')
    self.assertEqual(clock.step(), '0.5Myr')
    self.assertEqual(clock.step_size(), 5e5)
    clock.advance()
    clock.advance('100yr')
    self.assertAlmostEqual(clock.now(), -3.98e9 + 5e5 + 100)
    self.assertEqual(clock.ticks(), 4)
    self.assertEqual(clock.formatted_now(), '-3.98 Gyr')

  def test_state_round_trip(self):
    clock = numeric_time.NumericClock(step='1kyr')
    clock.advance()
    clock.set_step('5yr')
    state = clock.get_state()
    restored = numeric_time.NumericClock()
    restored.set_state(state)
    self.assertEqual(restored.now(), 1000.0)
    self.assertEqual(restored.ticks(), 1)
    self.assertEqual(restored.step(), '5yr')
    restored.advance()
    self.assertEqual(restored.now(), 1005.0)


if __name__ == '__main__':
  absltest.main()
