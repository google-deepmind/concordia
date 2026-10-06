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

"""Lightweight performance profiler for Concordia components.

Provides per-component timing instrumentation with minimal overhead.
Designed to diagnose where simulation time is spent:
- LLM call latency vs local compute
- Per-component breakdown within entity action cycles
- Conversation overhead

Usage:
  perf = PerfLogger()

  with perf.track('SelfPerception'):
      result = self_perception.pre_act(action_spec)

  perf.summary()  # Returns dict of per-component stats
"""

import collections
import contextlib
import threading
import time
from typing import Any

from absl import logging


@contextlib.contextmanager
def timed(label: str, log: bool = True):
  """Simple context manager that logs elapsed time.

  Args:
    label: Description of the timed operation.
    log: Whether to log the timing (True for INFO, False for silent).

  Yields:
    A dict with 'elapsed' key populated after the block completes.
  """
  result = {'elapsed': 0.0}
  start = time.monotonic()
  try:
    yield result
  finally:
    result['elapsed'] = time.monotonic() - start
    if log:
      logging.info('⏱️  %s: %.3fs', label, result['elapsed'])


class PerfLogger:
  """Thread-safe per-component performance accumulator.

  Tracks cumulative timing stats (count, total time, max time) for
  named components. All methods are thread-safe.

  Typical usage:
    perf = PerfLogger()

    # In a component wrapper:
    with perf.track('SelfPerception'):
        value = self_perception._make_pre_act_value()

    # At end of simulation:
    for name, stats in perf.summary().items():
        logging.info('%s: %d calls, %.1fs total, %.3fs avg, %.3fs max',
                     name, stats['count'], stats['total'],
                     stats['avg'], stats['max'])
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._stats: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {'count': 0, 'total': 0.0, 'max': 0.0}
    )

  @contextlib.contextmanager
  def track(self, component_name: str):
    """Context manager that accumulates timing for a named component.

    Args:
      component_name: Name of the component being timed.

    Yields:
      None. Timing is recorded automatically.
    """
    start = time.monotonic()
    try:
      yield
    finally:
      elapsed = time.monotonic() - start
      with self._lock:
        stats = self._stats[component_name]
        stats['count'] += 1
        stats['total'] += elapsed
        stats['max'] = max(stats['max'], elapsed)

  def summary(self) -> dict[str, dict[str, Any]]:
    """Returns per-component timing statistics.

    Returns:
      Dict mapping component name to stats dict with keys:
        count: Number of calls
        total: Cumulative wall time (seconds)
        avg: Average wall time per call (seconds)
        max: Maximum wall time for a single call (seconds)
    """
    with self._lock:
      result = {}
      for name, stats in sorted(
          self._stats.items(), key=lambda x: x[1]['total'], reverse=True
      ):
        count = int(stats['count'])
        result[name] = {
            'count': count,
            'total': stats['total'],
            'avg': stats['total'] / count if count > 0 else 0.0,
            'max': stats['max'],
        }
      return result

  def log_summary(self, prefix: str = 'PERF') -> None:
    """Logs a formatted summary of all tracked components."""
    logging.info('=' * 60)
    logging.info('%s  Performance Summary', prefix)
    logging.info('=' * 60)
    for name, stats in self.summary().items():
      logging.info(
          '  %-30s  %4d calls  %7.1fs total  %5.2fs avg  %5.2fs max',
          name,
          stats['count'],
          stats['total'],
          stats['avg'],
          stats['max'],
      )
    logging.info('=' * 60)

  def reset(self) -> None:
    """Clears all accumulated stats."""
    with self._lock:
      self._stats.clear()
