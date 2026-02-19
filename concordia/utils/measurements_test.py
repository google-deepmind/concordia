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

import unittest

from absl.testing import absltest
from concordia.utils import measurements


class TestMeasurements(unittest.TestCase):

  def test_publish_and_retrieve(self):
    m = measurements.Measurements()
    m.publish_datum('score', 42)
    self.assertEqual(m.get_last_datum('score'), 42)

  def test_close_does_not_deadlock(self):
    """Regression: close() used to call close_channel() while holding the lock,
    causing a deadlock with non-reentrant threading.Lock."""
    m = measurements.Measurements()
    m.publish_datum('a', 1)
    m.publish_datum('b', 2)
    m.close()
    self.assertEqual(m.available_channels(), set())

  def test_close_channel(self):
    m = measurements.Measurements()
    m.publish_datum('x', 'hello')
    m.close_channel('x')
    self.assertNotIn('x', m.available_channels())


if __name__ == '__main__':
  absltest.main()
