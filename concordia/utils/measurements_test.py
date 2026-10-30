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

"""Tests for keeping only recent data in measurement channels."""

from absl.testing import absltest
from concordia.utils import async_measurements
from concordia.utils import measurements as measurements_lib


class ChannelRetentionTest(absltest.TestCase):

  def test_without_retention_every_datum_is_kept(self):
    measurements = measurements_lib.Measurements()
    for i in range(100):
      measurements.publish_datum('log', i)
    self.assertLen(measurements.get_channel('log'), 100)

  def test_only_the_most_recent_data_are_kept(self):
    measurements = measurements_lib.Measurements(channel_retention=3)
    for i in range(10):
      measurements.publish_datum('log', i)
      measurements.publish_datum('other', -i)

    self.assertEqual(measurements.get_channel('log'), [7, 8, 9])
    self.assertEqual(measurements.get_channel('other'), [-7, -8, -9])
    self.assertEqual(measurements.get_last_datum('log'), 9)

  def test_retention_must_be_positive(self):
    with self.assertRaises(ValueError):
      measurements_lib.Measurements(channel_retention=0)

  def test_reactive_measurements_keep_recent_data_and_still_capture(self):
    measurements = async_measurements.ReactiveMeasurements(channel_retention=2)
    with measurements.capture('Ada') as captured:
      for i in range(5):
        measurements.publish_datum('log', i, capture_key='Ada')

    self.assertEqual(measurements.get_channel('log'), [3, 4])
    self.assertEqual(captured, {'log': 4})


if __name__ == '__main__':
  absltest.main()
