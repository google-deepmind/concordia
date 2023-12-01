# Copyright 2023 DeepMind Technologies Limited.
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

"""Tests for the measurements library."""

from unittest import mock
from absl.testing import absltest
from concordia.utils import measurements


class MeasurementsTest(absltest.TestCase):

  def test_empty_channels(self):
    msrmnts = measurements.Measurements()
    self.assertEmpty(msrmnts.available_channels())

  def test_get_channel_makes_available(self):
    msrmnts = measurements.Measurements()
    _ = msrmnts.get_channel("test_channel")
    self.assertIn("test_channel", msrmnts.available_channels())

  def test_get_channel_twice_is_same_object(self):
    msrmnts = measurements.Measurements()
    channel1 = msrmnts.get_channel("test_channel")
    channel2 = msrmnts.get_channel("test_channel")
    self.assertEqual(channel1, channel2)

  def test_close_calls_on_complete(self):
    msrmnts = measurements.Measurements()
    channel = msrmnts.get_channel("test_channel")
    sentinel = mock.MagicMock()
    channel.subscribe(on_completed=sentinel)
    msrmnts.close_channel("test_channel")
    sentinel.assert_called_once()

  def test_publish_calls_on_next_early_subscribe(self):
    msrmnts = measurements.Measurements()
    channel = msrmnts.get_channel("test_channel")

    sentinel = mock.MagicMock()
    channel.subscribe(on_next=sentinel)

    datum = "datum"
    msrmnts.publish_datum("test_channel", datum)

    msrmnts.close()
    sentinel.on_next.assert_called_once_with(datum)

  def test_publish_calls_on_next_late_subscribe(self):
    msrmnts = measurements.Measurements()
    channel = msrmnts.get_channel("test_channel")

    datum = "datum"
    msrmnts.publish_datum("test_channel", datum)

    sentinel = mock.MagicMock()
    channel.subscribe(on_next=sentinel)

    msrmnts.close()
    sentinel.on_next.assert_called_once_with(datum)

  def test_publish_calls_on_next_post_close_subscribe(self):
    msrmnts = measurements.Measurements()
    channel = msrmnts.get_channel("test_channel")

    datum = "datum"
    msrmnts.publish_datum("test_channel", datum)

    msrmnts.close()

    sentinel = mock.MagicMock()
    channel.subscribe(on_next=sentinel)
    sentinel.on_next.assert_called_once_with(datum)


if __name__ == "__main__":
  absltest.main()
