# Copyright 2025 DeepMind Technologies Limited.
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

"""Tests for ReactiveMeasurements."""

from absl.testing import absltest
from concordia.utils import async_measurements


class ReactiveMeasurementsTest(absltest.TestCase):

  def test_publish_datum_still_stores_in_channel(self):
    m = async_measurements.ReactiveMeasurements()
    m.publish_datum('a', 1)
    self.assertEqual(m.get_channel('a'), [1])

  def test_subscribe_receives_published_data(self):
    m = async_measurements.ReactiveMeasurements()
    received = []
    subscription = m.subscribe(lambda pair: received.append(pair))
    m.publish_datum('a', 1)
    m.publish_datum('b', 2)
    subscription.dispose()
    self.assertEqual(received, [('a', 1), ('b', 2)])

  def test_capture_collects_only_data_with_matching_key(self):
    m = async_measurements.ReactiveMeasurements()
    with m.capture('entity_1') as captured:
      m.publish_datum('a', 'mine', capture_key='entity_1')
      m.publish_datum('b', 'not_mine', capture_key='entity_2')
      m.publish_datum('c', 'uncaptured')
    self.assertEqual(captured, {'a': 'mine'})

  def test_capture_only_keeps_last_datum_per_channel(self):
    m = async_measurements.ReactiveMeasurements()
    with m.capture('entity_1') as captured:
      m.publish_datum('a', 'first', capture_key='entity_1')
      m.publish_datum('a', 'second', capture_key='entity_1')
    self.assertEqual(captured, {'a': 'second'})

  def test_capture_stops_collecting_after_context_exits(self):
    m = async_measurements.ReactiveMeasurements()
    with m.capture('entity_1') as captured:
      m.publish_datum('a', 'inside', capture_key='entity_1')
    m.publish_datum('a', 'outside', capture_key='entity_1')
    self.assertEqual(captured, {'a': 'inside'})

  def test_nested_captures_with_different_keys_do_not_cross_contaminate(self):
    m = async_measurements.ReactiveMeasurements()
    with m.capture('entity_1') as captured_1:
      with m.capture('entity_2') as captured_2:
        m.publish_datum('a', 'for_1', capture_key='entity_1')
        m.publish_datum('a', 'for_2', capture_key='entity_2')
    self.assertEqual(captured_1, {'a': 'for_1'})
    self.assertEqual(captured_2, {'a': 'for_2'})

  def test_close_still_works_on_reactive_subclass(self):
    # Regression coverage: close()/close_channel() must not deadlock on the
    # ReactiveMeasurements subclass either.
    m = async_measurements.ReactiveMeasurements()
    m.publish_datum('a', 1)
    m.publish_datum('b', 2)
    m.close()
    self.assertEqual(m.available_channels(), set())

  def test_dispose_completes_subject(self):
    m = async_measurements.ReactiveMeasurements()
    received = []
    completed = []
    m.subscribe(lambda pair: received.append(pair))
    m._subject.subscribe(on_completed=lambda: completed.append(True))  # pylint: disable=protected-access
    m.publish_datum('a', 1)
    m.dispose()
    self.assertEqual(received, [('a', 1)])
    self.assertTrue(completed)


if __name__ == '__main__':
  absltest.main()
