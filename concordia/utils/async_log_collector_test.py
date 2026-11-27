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

"""Tests for the asynchronous engine's log collector."""

import threading

from absl.testing import absltest
from concordia.utils import async_log_collector as collector_lib


def _event(step: int) -> collector_lib.RawLogEvent:
  return collector_lib.RawLogEvent(
      step=step,
      entity_name='Ada',
      game_master_name='rules',
      entity_log={},
      game_master_log={},
      action=f'Ada: step {step}',
  )


class AsyncLogCollectorTest(absltest.TestCase):

  def test_materialize_empties_the_buffer_into_the_log(self):
    collector = collector_lib.AsyncLogCollector()
    log = []
    for step in range(3):
      collector.emit(_event(step))

    collector.materialize(log)

    self.assertEqual([entry['Step'] for entry in log], [0, 1, 2])
    self.assertEqual(collector.pending_count(), 0)

  def test_batches_materialized_concurrently_keep_their_order(self):
    collector = collector_lib.AsyncLogCollector()
    log = []

    def emit_and_materialize(start: int) -> None:
      for step in range(start, start + 50):
        collector.emit(_event(step))
        if step % 7 == 0:
          collector.materialize(log)

    threads = [
        threading.Thread(target=emit_and_materialize, args=(start,))
        for start in (0, 1000, 2000, 3000)
    ]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join(timeout=10)
    collector.materialize(log)

    steps = [entry['Step'] for entry in log]
    self.assertLen(steps, 200)
    for start in (0, 1000, 2000, 3000):  # Each thread's events in order.
      mine = [step for step in steps if start <= step < start + 50]
      self.assertEqual(mine, list(range(start, start + 50)))


if __name__ == '__main__':
  absltest.main()
