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

"""Tests for interrupt_time_model module."""

import datetime
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.components.game_master import interrupt_time_model


class ParseDurationSecondsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('0m', 0),
      ('0', 0),
      ('5m', 300),
      ('2h', 7200),
      ('1h30m', 5400),
      ('', 0),
      ('3h', 10800),
      ('15m', 900),
  )
  def test_parse_duration_seconds(self, input_str, expected):
    self.assertEqual(
        interrupt_time_model.parse_duration_seconds(input_str),
        expected,
    )

  def test_unparseable_defaults_to_one_hour(self):
    self.assertEqual(
        interrupt_time_model.parse_duration_seconds('abc'),
        3600,
    )


class ParseTimeOfDaySecondsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('0:00', 0),
      ('8:00', 28800),
      ('14:30', 52200),
      ('23:59', 86340),
  )
  def test_valid_times(self, input_str, expected):
    self.assertEqual(
        interrupt_time_model.parse_time_of_day_seconds(input_str),
        expected,
    )

  @parameterized.parameters(
      '',
      'abc',
      '25:00',
      '12:60',
      '8:00 AM',
      '8',
  )
  def test_invalid_times_return_none(self, input_str):
    self.assertIsNone(
        interrupt_time_model.parse_time_of_day_seconds(input_str)
    )


class DatetimeTimeModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.start_time = datetime.datetime(2026, 1, 1, 9, 0, 0)
    self.model = interrupt_time_model.DatetimeTimeModel(self.start_time)

  def test_initial_time(self):
    self.assertEqual(self.model.initial_time(), self.start_time)

  def test_add_duration_30m(self):
    result = self.model.add_duration(self.start_time, '30m')
    expected = datetime.datetime(2026, 1, 1, 9, 30, 0)
    self.assertEqual(result, expected)

  def test_add_duration_2h(self):
    result = self.model.add_duration(self.start_time, '2h')
    expected = datetime.datetime(2026, 1, 1, 11, 0, 0)
    self.assertEqual(result, expected)

  def test_add_duration_1h30m(self):
    result = self.model.add_duration(self.start_time, '1h30m')
    expected = datetime.datetime(2026, 1, 1, 10, 30, 0)
    self.assertEqual(result, expected)

  def test_add_duration_zero(self):
    result = self.model.add_duration(self.start_time, '0')
    self.assertEqual(result, self.start_time)

  def test_format_time(self):
    formatted = self.model.format_time(self.start_time)
    self.assertEqual(formatted, '2026-01-01 09:00:00')

  def test_serialize_deserialize_roundtrip(self):
    serialised = self.model.serialize_time(self.start_time)
    deserialised = self.model.deserialize_time(serialised)
    self.assertEqual(deserialised, self.start_time)

  def test_parse_absolute_same_day(self):
    """At 09:00, '14:00' should resolve to same-day 14:00."""
    current = datetime.datetime(2026, 1, 1, 9, 0)
    result = self.model.parse_absolute_time('14:00', current)
    self.assertEqual(result, datetime.datetime(2026, 1, 1, 14, 0))

  def test_parse_absolute_next_day(self):
    """At 21:00, '8:00' should resolve to next-day 08:00."""
    current = datetime.datetime(2026, 1, 1, 21, 0)
    result = self.model.parse_absolute_time('8:00', current)
    self.assertEqual(result, datetime.datetime(2026, 1, 2, 8, 0))

  def test_parse_absolute_at_exact_time_advances_day(self):
    """At exactly 14:00, '14:00' means tomorrow 14:00."""
    current = datetime.datetime(2026, 1, 1, 14, 0)
    result = self.model.parse_absolute_time('14:00', current)
    self.assertEqual(result, datetime.datetime(2026, 1, 2, 14, 0))

  def test_parse_absolute_invalid_raises(self):
    current = datetime.datetime(2026, 1, 1, 9, 0)
    with self.assertRaises(ValueError):
      self.model.parse_absolute_time('garbage', current)

  def test_timestamps_are_comparable(self):
    t1 = self.model.initial_time()
    t2 = self.model.add_duration(t1, '30m')
    self.assertLess(t1, t2)
    self.assertGreater(t2, t1)
    self.assertEqual(t1, t1)


class GenerativeTimeModelTest(absltest.TestCase):

  def _make_model(self, format_response='narrated time'):
    """Creates a GenerativeTimeModel with a mock LLM."""
    mock_model = mock.MagicMock()
    mock_model.sample_text.return_value = format_response
    model = interrupt_time_model.GenerativeTimeModel(
        model=mock_model,
        clock_prompt='This simulation takes place in ancient Rome.',
        start_time_label='Dawn on the Ides of March, 44 BCE',
    )
    return model

  def test_initial_time_is_zero(self):
    model = self._make_model()
    self.assertEqual(model.initial_time(), 0)

  def test_add_duration_30m(self):
    model = self._make_model()
    result = model.add_duration(0, '30m')
    self.assertEqual(result, 1800)

  def test_add_duration_2h(self):
    model = self._make_model()
    result = model.add_duration(0, '2h')
    self.assertEqual(result, 7200)

  def test_add_duration_is_cumulative(self):
    model = self._make_model()
    t1 = model.add_duration(0, '1h')
    t2 = model.add_duration(t1, '30m')
    self.assertEqual(t2, 5400)  # 1h30m = 5400 seconds

  def test_format_time_at_zero_returns_cached_label(self):
    model = self._make_model()
    # T=0 should return the start_time_label without an LLM call.
    formatted = model.format_time(0)
    self.assertEqual(formatted, 'Dawn on the Ides of March, 44 BCE')

  def test_format_time_caches_results(self):
    model = self._make_model(format_response='mid-morning')
    # First call at T=1800 should invoke LLM.
    result1 = model.format_time(1800)
    self.assertEqual(result1, 'mid-morning')
    # Second call at same timestamp should return cached value.
    result2 = model.format_time(1800)
    self.assertEqual(result2, 'mid-morning')

  def test_serialize_deserialize_roundtrip(self):
    model = self._make_model()
    t = 5400
    serialised = model.serialize_time(t)
    self.assertEqual(serialised, '5400')
    deserialised = model.deserialize_time(serialised)
    self.assertEqual(deserialised, t)

  def test_parse_absolute_time_calls_llm(self):
    """LLM returns offset '3600' → current + 3600."""
    model = self._make_model(format_response='3600')
    result = model.parse_absolute_time('8:00', 1000)
    self.assertEqual(result, 4600)

  def test_parse_absolute_time_next_day_clamp(self):
    """LLM returns '0' (non-positive offset) → adds 86400."""
    model = self._make_model(format_response='0')
    result = model.parse_absolute_time('8:00', 1000)
    self.assertEqual(result, 87400)  # 1000 + 0 + 86400

  def test_parse_absolute_time_negative_offset(self):
    """LLM returns '-500' (negative offset) → adds 86400."""
    model = self._make_model(format_response='-500')
    result = model.parse_absolute_time('8:00', 1000)
    self.assertEqual(result, 86900)  # 1000 + (-500) + 86400

  def test_parse_absolute_time_bad_llm_response_raises(self):
    model = self._make_model(format_response='not a number')
    with self.assertRaises(ValueError):
      model.parse_absolute_time('8:00', 1000)

  def test_timestamps_are_comparable(self):
    model = self._make_model()
    t1 = model.initial_time()
    t2 = model.add_duration(t1, '1h')
    self.assertLess(t1, t2)
    self.assertGreater(t2, t1)
    self.assertEqual(t1, t1)

  def test_timestamps_have_unlimited_range(self):
    model = self._make_model()
    # 1 million years in seconds — well beyond datetime's range.
    far_future = 1_000_000 * 365 * 24 * 3600
    result = model.add_duration(0, '0')
    # Verify we can represent enormous timestamps without error.
    t = far_future
    serialised = model.serialize_time(t)
    deserialised = model.deserialize_time(serialised)
    self.assertEqual(deserialised, t)
    # Comparison still works.
    self.assertGreater(t, result)


if __name__ == '__main__':
  absltest.main()
