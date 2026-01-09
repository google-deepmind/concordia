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

"""Tests for profiler module."""

import time
from concordia.utils import profiler
import pytest


class TestProfiler:
  """Tests for the profiler module."""

  def setup_method(self):
    """Reset profiler before each test."""
    profiler.reset()
    profiler.disable()

  def test_enable_disable(self):
    """Test enabling and disabling the profiler."""
    assert not profiler.is_enabled()

    profiler.enable()
    assert profiler.is_enabled()

    profiler.disable()
    assert not profiler.is_enabled()

  def test_track_context_manager(self):
    """Test the track context manager."""
    profiler.enable()

    with profiler.track('test_operation'):
      time.sleep(0.01)  # Sleep for 10ms

    stats = profiler.get_stats()
    assert 'test_operation' in stats['timings']
    assert len(stats['timings']['test_operation']) == 1
    # Check that duration is roughly 10ms (allow some margin)
    assert stats['timings']['test_operation'][0] >= 0.01

  def test_track_when_disabled(self):
    """Test that tracking does nothing when disabled."""
    profiler.disable()

    with profiler.track('test_operation'):
      time.sleep(0.01)

    stats = profiler.get_stats()
    assert 'test_operation' not in stats['timings']

  def test_track_time_decorator(self):
    """Test the track_time decorator."""
    profiler.enable()

    @profiler.track_time('decorated_function')
    def test_func():
      time.sleep(0.01)
      return 42

    result = test_func()
    assert result == 42

    stats = profiler.get_stats()
    assert 'decorated_function' in stats['timings']
    assert len(stats['timings']['decorated_function']) == 1

  def test_track_time_decorator_when_disabled(self):
    """Test that decorator works but doesn't record when disabled."""
    profiler.disable()

    @profiler.track_time('decorated_function')
    def test_func():
      return 42

    result = test_func()
    assert result == 42

    stats = profiler.get_stats()
    assert 'decorated_function' not in stats['timings']

  def test_multiple_calls_same_category(self):
    """Test multiple calls to the same category."""
    profiler.enable()

    for i in range(5):
      with profiler.track('repeated_operation'):
        time.sleep(0.001)

    stats = profiler.get_stats()
    assert len(stats['timings']['repeated_operation']) == 5

  def test_multiple_categories(self):
    """Test tracking multiple different categories."""
    profiler.enable()

    with profiler.track('operation_a'):
      time.sleep(0.001)

    with profiler.track('operation_b'):
      time.sleep(0.001)

    stats = profiler.get_stats()
    assert 'operation_a' in stats['timings']
    assert 'operation_b' in stats['timings']

  def test_reset(self):
    """Test resetting the profiler."""
    profiler.enable()

    with profiler.track('test_operation'):
      time.sleep(0.001)

    profiler.increment_counter('test_counter')
    profiler.record_value('test_value', 42.0)

    stats = profiler.get_stats()
    assert stats['timings']
    assert stats['counters']
    assert stats['values']

    profiler.reset()

    stats = profiler.get_stats()
    assert not stats['timings']
    assert not stats['counters']
    assert not stats['values']

  def test_increment_counter(self):
    """Test counter incrementing."""
    profiler.enable()

    profiler.increment_counter('test_counter')
    profiler.increment_counter('test_counter')
    profiler.increment_counter('test_counter', 3)

    stats = profiler.get_stats()
    assert stats['counters']['test_counter'] == 5

  def test_counter_when_disabled(self):
    """Test that counters don't increment when disabled."""
    profiler.disable()

    profiler.increment_counter('test_counter')

    stats = profiler.get_stats()
    assert 'test_counter' not in stats['counters']

  def test_record_value(self):
    """Test recording values."""
    profiler.enable()

    profiler.record_value('test_metric', 10.5)
    profiler.record_value('test_metric', 20.5)
    profiler.record_value('test_metric', 30.5)

    stats = profiler.get_stats()
    assert len(stats['values']['test_metric']) == 3
    assert sum(stats['values']['test_metric']) == 61.5

  def test_record_value_when_disabled(self):
    """Test that values aren't recorded when disabled."""
    profiler.disable()

    profiler.record_value('test_metric', 10.5)

    stats = profiler.get_stats()
    assert 'test_metric' not in stats['values']

  def test_nested_tracking(self):
    """Test nested tracking contexts."""
    profiler.enable()

    with profiler.track('outer'):
      time.sleep(0.01)
      with profiler.track('inner'):
        time.sleep(0.01)

    stats = profiler.get_stats()
    assert 'outer' in stats['timings']
    assert 'inner' in stats['timings']
    # Outer should be longer than inner
    assert stats['timings']['outer'][0] >= stats['timings']['inner'][0]

  def test_profile_simulation_context(self):
    """Test the profile_simulation context manager."""
    # Profiler should be disabled initially
    profiler.disable()

    with profiler.profile_simulation():
      # Should be enabled inside context
      assert profiler.is_enabled()

      with profiler.track('simulation_step'):
        time.sleep(0.001)

    # Should still be enabled after (only disabled manually)
    assert profiler.is_enabled()

    stats = profiler.get_stats()
    assert 'simulation_step' in stats['timings']

  def test_thread_safety(self):
    """Test that profiler is thread-safe."""
    import threading

    profiler.enable()

    def worker(worker_id: int):
      for i in range(10):
        with profiler.track(f'worker_{worker_id}'):
          time.sleep(0.001)
        profiler.increment_counter(f'counter_{worker_id}')

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    stats = profiler.get_stats()

    # Each worker should have recorded 10 timings
    for i in range(3):
      assert len(stats['timings'][f'worker_{i}']) == 10
      assert stats['counters'][f'counter_{i}'] == 10

  def test_decorator_preserves_function_metadata(self):
    """Test that decorator preserves function name and docstring."""
    @profiler.track_time('test_func')
    def my_function():
      """This is a test function."""
      return 42

    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'This is a test function.'

  def test_exception_in_tracked_code(self):
    """Test that profiler still records time when exception occurs."""
    profiler.enable()

    with pytest.raises(ValueError):
      with profiler.track('failing_operation'):
        time.sleep(0.01)
        raise ValueError('Test error')

    # Should still have recorded the timing
    stats = profiler.get_stats()
    assert 'failing_operation' in stats['timings']
    assert len(stats['timings']['failing_operation']) == 1

  def test_print_report_doesnt_crash(self):
    """Test that print_report doesn't crash."""
    profiler.enable()

    with profiler.track('test_op'):
      time.sleep(0.001)

    profiler.increment_counter('test_counter', 5)
    profiler.record_value('test_value', 42.0)

    # Should not raise an exception
    profiler.print_report()
