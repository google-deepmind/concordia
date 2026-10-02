# Copyright 2024 DeepMind Technologies Limited.
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

"""Tests for the Measurements registry."""

import threading

from absl.testing import absltest
from concordia.utils import measurements as measurements_lib


class MeasurementsTest(absltest.TestCase):

  def test_publish_datum_creates_channel(self):
    m = measurements_lib.Measurements()
    m.publish_datum('channel_a', 1)
    self.assertEqual(m.available_channels(), {'channel_a'})

  def test_publish_datum_appends_in_order(self):
    m = measurements_lib.Measurements()
    m.publish_datum('channel_a', 1)
    m.publish_datum('channel_a', 2)
    m.publish_datum('channel_a', 3)
    self.assertEqual(m.get_channel('channel_a'), [1, 2, 3])

  def test_get_channel_creates_empty_channel_if_missing(self):
    m = measurements_lib.Measurements()
    self.assertEqual(m.get_channel('missing'), [])
    self.assertEqual(m.available_channels(), {'missing'})

  def test_get_last_datum_returns_most_recent(self):
    m = measurements_lib.Measurements()
    m.publish_datum('channel_a', 'first')
    m.publish_datum('channel_a', 'second')
    self.assertEqual(m.get_last_datum('channel_a'), 'second')

  def test_get_last_datum_returns_none_for_empty_channel(self):
    m = measurements_lib.Measurements()
    self.assertIsNone(m.get_last_datum('missing'))

  def test_get_all_channels_returns_all_data(self):
    m = measurements_lib.Measurements()
    m.publish_datum('a', 1)
    m.publish_datum('b', 2)
    self.assertEqual(m.get_all_channels(), {'a': [1], 'b': [2]})

  def test_get_all_channels_returns_a_copy(self):
    m = measurements_lib.Measurements()
    m.publish_datum('a', 1)
    snapshot = m.get_all_channels()
    m.publish_datum('a', 2)
    self.assertEqual(snapshot, {'a': [1]})
    self.assertEqual(m.get_channel('a'), [1, 2])

  def test_close_channel_removes_channel(self):
    m = measurements_lib.Measurements()
    m.publish_datum('a', 1)
    m.close_channel('a')
    self.assertEqual(m.available_channels(), set())

  def test_close_channel_on_missing_channel_is_a_no_op(self):
    m = measurements_lib.Measurements()
    # Should not raise, per the method's documented behavior.
    m.close_channel('never_published')
    self.assertEqual(m.available_channels(), set())

  def test_close_removes_all_channels(self):
    m = measurements_lib.Measurements()
    m.publish_datum('a', 1)
    m.publish_datum('b', 2)
    m.publish_datum('c', 3)
    m.close()
    self.assertEqual(m.available_channels(), set())

  def test_close_does_not_deadlock_with_multiple_channels(self):
    # Regression test: close() used to hold _channels_lock while calling
    # close_channel(), which itself acquired the same non-reentrant lock,
    # deadlocking whenever there was at least one channel.
    m = measurements_lib.Measurements()
    m.publish_datum('a', 1)
    m.publish_datum('b', 2)

    completed = threading.Event()

    def run_close():
      m.close()
      completed.set()

    thread = threading.Thread(target=run_close)
    thread.start()
    thread.join(timeout=5)
    self.assertTrue(completed.is_set(), 'Measurements.close() deadlocked.')

  def test_close_on_empty_measurements_does_not_raise(self):
    m = measurements_lib.Measurements()
    m.close()
    self.assertEqual(m.available_channels(), set())

  def test_get_channel_or_create_requires_lock(self):
    m = measurements_lib.Measurements()
    with self.assertRaises(RuntimeError):
      m._get_channel_or_create('a')  # pylint: disable=protected-access


if __name__ == '__main__':
  absltest.main()
