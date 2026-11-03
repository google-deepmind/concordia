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

import threading

from absl.testing import absltest
from concordia.utils import measurements as measurements_lib


_CLOSE_TIMEOUT_SECONDS = 5.0


class MeasurementsTest(absltest.TestCase):

  def test_close_returns_instead_of_deadlocking(self):
    # Regression test: `close()` used to acquire `_channels_lock` and then
    # call `close_channel()`, which acquires the same non-reentrant lock
    # again. The call hung forever. Run close in a thread so a regression
    # fails the assertion instead of hanging the whole test run.
    registry = measurements_lib.Measurements()
    registry.publish_datum('channel', 1)
    close_thread = threading.Thread(target=registry.close, daemon=True)
    close_thread.start()
    close_thread.join(timeout=_CLOSE_TIMEOUT_SECONDS)
    self.assertFalse(
        close_thread.is_alive(),
        'close() deadlocked: channels lock never released',
    )
    self.assertFalse(registry._channels_lock.locked())  # pylint: disable=protected-access

  def test_close_removes_all_channels(self):
    registry = measurements_lib.Measurements()
    registry.publish_datum('a', 1)
    registry.publish_datum('b', 2)
    registry.close()
    self.assertEmpty(registry.available_channels())
    self.assertEmpty(registry.get_all_channels())

  def test_close_on_empty_registry_is_a_noop(self):
    registry = measurements_lib.Measurements()
    registry.close()
    self.assertEmpty(registry.available_channels())

  def test_close_does_not_block_concurrent_publish(self):
    # Another thread must be able to publish before/after close, and close
    # must leave the lock released (not stuck held by a deadlocked thread).
    registry = measurements_lib.Measurements()
    registry.publish_datum('a', 1)
    close_thread = threading.Thread(target=registry.close, daemon=True)
    close_thread.start()
    close_thread.join(timeout=_CLOSE_TIMEOUT_SECONDS)
    self.assertFalse(close_thread.is_alive())
    # Lock must be acquirable by another thread now.
    self.assertTrue(registry._channels_lock.acquire(timeout=1.0))  # pylint: disable=protected-access
    registry._channels_lock.release()  # pylint: disable=protected-access
    # Registry is usable again after close.
    registry.publish_datum('fresh', 42)
    self.assertEqual(registry.get_last_datum('fresh'), 42)

  def test_close_channel_still_removes_single_channel(self):
    # Unchanged sibling API: close_channel removes exactly one channel.
    registry = measurements_lib.Measurements()
    registry.publish_datum('keep', 1)
    registry.publish_datum('drop', 2)
    registry.close_channel('drop')
    self.assertCountEqual(registry.available_channels(), ['keep'])
    self.assertEqual(registry.get_last_datum('keep'), 1)


if __name__ == '__main__':
  absltest.main()
