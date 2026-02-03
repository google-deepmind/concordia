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

"""Performance profiling utilities for Concordia simulations.

This module provides lightweight profiling tools to help identify performance
bottlenecks in multi-agent simulations. It tracks timing for different
categories of operations (LLM calls, memory retrieval, agent processing, etc.)
with minimal overhead.

Usage:
  from concordia.utils import profiler

  # Enable profiling
  profiler.enable()

  # Track a block of code
  with profiler.track("my_operation"):
      # ... code to profile
      pass

  # Or use as a decorator
  @profiler.track_time("agent_action")
  def my_function():
      pass

  # Print report
  profiler.print_report()

  # Reset for next simulation
  profiler.reset()
"""

import contextlib
import functools
import threading
import time
from typing import Any, Callable, TypeVar

from absl import logging


_T = TypeVar('_T')


class ProfilerContext:
  """Thread-safe profiling context for tracking timing and metrics.

  This class maintains timing statistics for different categories of operations
  and provides thread-safe access for concurrent simulations.
  """

  def __init__(self):
    """Initialize the profiler context."""
    self._enabled = False
    self._lock = threading.Lock()

    # Timing data: category -> list of durations
    self._timings: dict[str, list[float]] = {}

    # Counters: name -> count
    self._counters: dict[str, int] = {}

    # Values: name -> list of values
    self._values: dict[str, list[float]] = {}

  def enable(self) -> None:
    """Enable profiling."""
    self._enabled = True
    logging.info('Profiler enabled')

  def disable(self) -> None:
    """Disable profiling."""
    self._enabled = False
    logging.info('Profiler disabled')

  def is_enabled(self) -> bool:
    """Check if profiling is enabled."""
    return self._enabled

  def reset(self) -> None:
    """Reset all profiling data."""
    with self._lock:
      self._timings.clear()
      self._counters.clear()
      self._values.clear()
    logging.info('Profiler reset')

  def record_time(self, category: str, duration: float) -> None:
    """Record a timing measurement.

    Args:
      category: The category of operation being timed.
      duration: The duration in seconds.
    """
    if not self._enabled:
      return

    try:
      with self._lock:
        if category not in self._timings:
          self._timings[category] = []
        self._timings[category].append(duration)
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Never let profiler errors break the simulation
      logging.warning('Profiler error recording time: %s', e)

  def increment_counter(self, name: str, amount: int = 1) -> None:
    """Increment a counter.

    Args:
      name: The name of the counter.
      amount: The amount to increment by (default 1).
    """
    if not self._enabled:
      return

    try:
      with self._lock:
        if name not in self._counters:
          self._counters[name] = 0
        self._counters[name] += amount
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('Profiler error incrementing counter: %s', e)

  def record_value(self, name: str, value: float) -> None:
    """Record a numeric value.

    Args:
      name: The name of the value being recorded.
      value: The numeric value.
    """
    if not self._enabled:
      return

    try:
      with self._lock:
        if name not in self._values:
          self._values[name] = []
        self._values[name].append(value)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('Profiler error recording value: %s', e)

  def get_stats(self) -> dict[str, Any]:
    """Get all profiling statistics.

    Returns:
      A dictionary containing timing, counter, and value statistics.
    """
    with self._lock:
      return {
          'timings': dict(self._timings),
          'counters': dict(self._counters),
          'values': dict(self._values),
      }

  @contextlib.contextmanager
  def track(self, category: str):
    """Context manager for tracking execution time.

    Args:
      category: The category of operation being tracked.

    Yields:
      None
    """
    if not self._enabled:
      yield
      return

    start_time = time.perf_counter()
    try:
      yield
    finally:
      duration = time.perf_counter() - start_time
      self.record_time(category, duration)


# Global profiler instance
_global_profiler = ProfilerContext()


def enable() -> None:
  """Enable the global profiler."""
  _global_profiler.enable()


def disable() -> None:
  """Disable the global profiler."""
  _global_profiler.disable()


def is_enabled() -> bool:
  """Check if the global profiler is enabled."""
  return _global_profiler.is_enabled()


def reset() -> None:
  """Reset the global profiler."""
  _global_profiler.reset()


def record_time(category: str, duration: float) -> None:
  """Record a timing measurement to the global profiler.

  Args:
    category: The category of operation being timed.
    duration: The duration in seconds.
  """
  _global_profiler.record_time(category, duration)


def increment_counter(name: str, amount: int = 1) -> None:
  """Increment a counter in the global profiler.

  Args:
    name: The name of the counter.
    amount: The amount to increment by (default 1).
  """
  _global_profiler.increment_counter(name, amount)


def record_value(name: str, value: float) -> None:
  """Record a numeric value to the global profiler.

  Args:
    name: The name of the value being recorded.
    value: The numeric value.
  """
  _global_profiler.record_value(name, value)


def get_stats() -> dict[str, Any]:
  """Get all profiling statistics from the global profiler.

  Returns:
    A dictionary containing timing, counter, and value statistics.
  """
  return _global_profiler.get_stats()


@contextlib.contextmanager
def track(category: str):
  """Context manager for tracking execution time.

  Usage:
    with track("my_operation"):
        # ... code to profile
        pass

  Args:
    category: The category of operation being tracked.

  Yields:
    None
  """
  with _global_profiler.track(category):
    yield


def track_time(
    category: str,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
  """Decorator for tracking function execution time.

  Usage:
    @track_time("agent_action")
    def my_function():
        pass

  Args:
    category: The category of operation being tracked.

  Returns:
    A decorator that wraps the function with timing.
  """

  def decorator(func: Callable[..., _T]) -> Callable[..., _T]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> _T:
      if not _global_profiler.is_enabled():
        return func(*args, **kwargs)

      start_time = time.perf_counter()
      try:
        return func(*args, **kwargs)
      finally:
        duration = time.perf_counter() - start_time
        _global_profiler.record_time(category, duration)

    return wrapper

  return decorator


@contextlib.contextmanager
def profile_simulation():
  """Context manager for profiling an entire simulation.

  Automatically enables profiling, runs the simulation, and prints a report.

  Usage:
    with profile_simulation():
        # ... run simulation
        pass
  """
  enable()
  reset()
  try:
    yield
  finally:
    print_report()


def print_report() -> None:
  """Print a formatted profiling report to the console."""
  stats = get_stats()

  logging.info('\n%s', '=' * 70)
  logging.info('Concordia Simulation Performance Report')
  logging.info('%s', '=' * 70)

  timings = stats['timings']
  counters = stats['counters']
  values = stats['values']

  # Calculate total time
  total_time = sum(sum(durations) for durations in timings.values())

  if total_time > 0:
    logging.info('\nTotal simulation time: %.2fs', total_time)

  # Timing breakdown
  if timings:
    logging.info('\nTime breakdown:')
    logging.info('-' * 70)

    # Print all timings sorted by total time
    all_items = sorted(timings.items(), key=lambda x: sum(x[1]), reverse=True)

    for category, durations in all_items:
      count = len(durations)
      total = sum(durations)
      avg = total / count if count > 0 else 0
      percentage = (total / total_time * 100) if total_time > 0 else 0

      logging.info(
          '  %s %s (%s) - %s calls, avg %s',
          f'{category:30s}',
          f'{total:8.2f}s',
          f'{percentage:5.1f}%',
          f'{count:4d}',
          f'{avg:.3f}s/call',
      )

  # LLM-specific statistics
  has_llm_data = any(k.startswith('llm_') for k in counters.keys())
  has_llm_data |= any(k.startswith('llm_') for k in values.keys())

  if has_llm_data:
    logging.info('\nLLM Statistics:')
    logging.info('-' * 70)

    # Call counts
    total_calls = counters.get('llm_calls_total', 0)
    sample_text_calls = counters.get('llm_calls_sample_text', 0)
    sample_choice_calls = counters.get('llm_calls_sample_choice', 0)
    success_calls = counters.get('llm_calls_success', 0)
    failed_calls = counters.get('llm_calls_failed', 0)

    if total_calls > 0:
      logging.info('  Total LLM calls: %s', total_calls)
      if sample_text_calls > 0:
        logging.info('    sample_text: %s', sample_text_calls)
      if sample_choice_calls > 0:
        logging.info('    sample_choice: %s', sample_choice_calls)
      if success_calls > 0 or failed_calls > 0:
        success_rate = (
            success_calls / total_calls * 100 if total_calls > 0 else 0
        )
        logging.info(
            '  Success rate: %s (%s succeeded, %s failed)',
            f'{success_rate:.1f}%',
            success_calls,
            failed_calls,
        )

    # Token statistics
    prompt_tokens = values.get('llm_prompt_tokens', [])
    completion_tokens = values.get('llm_completion_tokens', [])
    total_tokens = values.get('llm_total_tokens', [])

    if total_tokens:
      total_sum = sum(total_tokens)
      logging.info('  Estimated tokens: ~%s', f'{total_sum:,}')
      if prompt_tokens:
        logging.info('    Prompt tokens: ~%s', f'{sum(prompt_tokens):,}')
      if completion_tokens:
        logging.info(
            '    Completion tokens: ~%s', f'{sum(completion_tokens):,}'
        )

    # Latency statistics
    latencies = values.get('llm_latency_seconds', [])
    if latencies:
      avg_latency = sum(latencies) / len(latencies)
      min_latency = min(latencies)
      max_latency = max(latencies)
      logging.info(
          '  Latency: avg %s, min %s, max %s',
          f'{avg_latency:.3f}s',
          f'{min_latency:.3f}s',
          f'{max_latency:.3f}s',
      )

  # Memory statistics (if present)
  memory_queries = counters.get('memory_queries', 0)
  if memory_queries > 0:
    logging.info('\nMemory Statistics:')
    logging.info('-' * 70)
    logging.info('  Total queries: %s', memory_queries)

    memory_result_sizes = values.get('memory_result_size', [])
    if memory_result_sizes:
      avg_results = sum(memory_result_sizes) / len(memory_result_sizes)
      logging.info('  Avg results per query: %s', f'{avg_results:.1f}')

  # Other counters
  other_counters = {
      k: v
      for k, v in counters.items()
      if not k.startswith('llm_') and k != 'memory_queries'
  }
  if other_counters:
    logging.info('\nOther Counters:')
    logging.info('-' * 70)
    for name, count in sorted(other_counters.items()):
      logging.info('  %s %s', f'{name:40s}', f'{count:10d}')

  # Other values
  other_values = {
      k: v
      for k, v in values.items()
      if not k.startswith('llm_') and k != 'memory_result_size'
  }
  if other_values:
    logging.info('\nOther Metrics:')
    logging.info('-' * 70)
    for name, vals in sorted(other_values.items()):
      count = len(vals)
      total = sum(vals)
      avg = total / count if count > 0 else 0
      min_val = min(vals)
      max_val = max(vals)
      logging.info(
          '  %s count=%s, avg=%s, min=%s, max=%s',
          f'{name:30s}',
          count,
          f'{avg:.2f}',
          f'{min_val:.2f}',
          f'{max_val:.2f}',
      )

  # Recommendations
  if total_time > 0 and timings:
    logging.info('\nRecommendations:')
    logging.info('-' * 70)

    # Check if LLM calls dominate
    llm_time = sum(
        sum(durations)
        for cat, durations in timings.items()
        if cat.startswith('llm_')
    )
    if llm_time / total_time > 0.6:
      logging.info(
          '  [!] LLM calls dominate runtime (%.0f%%)',
          llm_time / total_time * 100,
      )
      logging.info('      - Consider caching similar queries')
      logging.info('      - Review if all LLM calls are necessary')
      logging.info('      - Try using faster/smaller models for simple tasks')

    # Check for high failure rate
    llm_failed = counters.get('llm_calls_failed', 0)
    llm_total = counters.get('llm_calls_total', 0)
    if llm_failed > 0 and llm_total > 0:
      failure_rate = llm_failed / llm_total
      if failure_rate > 0.1:
        logging.info(
            '  [!] High LLM failure rate (%.0f%%)', failure_rate * 100
        )
        logging.info('      - Check API limits and rate limiting')
        logging.info('      - Review timeout settings')

    # Check memory efficiency
    memory_time = sum(
        sum(durations)
        for cat, durations in timings.items()
        if 'memory' in cat.lower()
    )
    if memory_time / total_time < 0.05 and memory_time > 0:
      logging.info('  [OK] Memory retrieval is efficient')

  logging.info('%s\n', '=' * 70)
