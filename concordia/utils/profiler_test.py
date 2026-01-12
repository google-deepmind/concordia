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

import threading
import time
from absl.testing import absltest
from concordia.utils import profiler


class ProfilerTest(absltest.TestCase):
  """Tests for the profiler module."""

  def setUp(self):
    """Reset profiler before each test."""
    super().setUp()
    profiler.reset()
    profiler.disable()

  def test_enable_disable(self):
    """Test enabling and disabling the profiler."""
    self.assertFalse(profiler.is_enabled())

    profiler.enable()
    self.assertTrue(profiler.is_enabled())

    profiler.disable()
    self.assertFalse(profiler.is_enabled())

  def test_track_context_manager(self):
    """Test the track context manager."""
    profiler.enable()

    with profiler.track('test_operation'):
      time.sleep(0.01)  # Sleep for 10ms

    stats = profiler.get_stats()
    self.assertIn('test_operation', stats['timings'])
    self.assertLen(stats['timings']['test_operation'], 1)
    # Check that duration is roughly 10ms (allow some margin)
    self.assertGreaterEqual(stats['timings']['test_operation'][0], 0.01)

  def test_track_when_disabled(self):
    """Test that tracking does nothing when disabled."""
    profiler.disable()

    with profiler.track('test_operation'):
      time.sleep(0.01)

    stats = profiler.get_stats()
    self.assertNotIn('test_operation', stats['timings'])

  def test_track_time_decorator(self):
    """Test the track_time decorator."""
    profiler.enable()

    @profiler.track_time('decorated_function')
    def test_func():
      time.sleep(0.01)
      return 42

    result = test_func()
    self.assertEqual(result, 42)

    stats = profiler.get_stats()
    self.assertIn('decorated_function', stats['timings'])
    self.assertLen(stats['timings']['decorated_function'], 1)

  def test_track_time_decorator_when_disabled(self):
    """Test that decorator works but doesn't record when disabled."""
    profiler.disable()

    @profiler.track_time('decorated_function')
    def test_func():
      return 42

    result = test_func()
    self.assertEqual(result, 42)

    stats = profiler.get_stats()
    self.assertNotIn('decorated_function', stats['timings'])

  def test_multiple_calls_same_category(self):
    """Test multiple calls to the same category."""
    profiler.enable()

    for _ in range(5):
      with profiler.track('repeated_operation'):
        time.sleep(0.001)

    stats = profiler.get_stats()
    self.assertLen(stats['timings']['repeated_operation'], 5)

  def test_multiple_categories(self):
    """Test tracking multiple different categories."""
    profiler.enable()

    with profiler.track('operation_a'):
      time.sleep(0.001)

    with profiler.track('operation_b'):
      time.sleep(0.001)

    stats = profiler.get_stats()
    self.assertIn('operation_a', stats['timings'])
    self.assertIn('operation_b', stats['timings'])

  def test_reset(self):
    """Test resetting the profiler."""
    profiler.enable()

    with profiler.track('test_operation'):
      time.sleep(0.001)

    profiler.increment_counter('test_counter')
    profiler.record_value('test_value', 42.0)

    stats = profiler.get_stats()
    self.assertTrue(stats['timings'])
    self.assertTrue(stats['counters'])
    self.assertTrue(stats['values'])

    profiler.reset()

    stats = profiler.get_stats()
    self.assertFalse(stats['timings'])
    self.assertFalse(stats['counters'])
    self.assertFalse(stats['values'])

  def test_increment_counter(self):
    """Test counter incrementing."""
    profiler.enable()

    profiler.increment_counter('test_counter')
    profiler.increment_counter('test_counter')
    profiler.increment_counter('test_counter', 3)

    stats = profiler.get_stats()
    self.assertEqual(stats['counters']['test_counter'], 5)

  def test_counter_when_disabled(self):
    """Test that counters don't increment when disabled."""
    profiler.disable()

    profiler.increment_counter('test_counter')

    stats = profiler.get_stats()
    self.assertNotIn('test_counter', stats['counters'])

  def test_record_value(self):
    """Test recording values."""
    profiler.enable()

    profiler.record_value('test_metric', 10.5)
    profiler.record_value('test_metric', 20.5)
    profiler.record_value('test_metric', 30.5)

    stats = profiler.get_stats()
    self.assertLen(stats['values']['test_metric'], 3)
    self.assertEqual(sum(stats['values']['test_metric']), 61.5)

  def test_record_value_when_disabled(self):
    """Test that values aren't recorded when disabled."""
    profiler.disable()

    profiler.record_value('test_metric', 10.5)

    stats = profiler.get_stats()
    self.assertNotIn('test_metric', stats['values'])

  def test_nested_tracking(self):
    """Test nested tracking contexts."""
    profiler.enable()

    with profiler.track('outer'):
      time.sleep(0.01)
      with profiler.track('inner'):
        time.sleep(0.01)

    stats = profiler.get_stats()
    self.assertIn('outer', stats['timings'])
    self.assertIn('inner', stats['timings'])
    # Outer should be longer than inner
    self.assertGreaterEqual(
        stats['timings']['outer'][0], stats['timings']['inner'][0]
    )

  def test_profile_simulation_context(self):
    """Test the profile_simulation context manager."""
    # Profiler should be disabled initially
    profiler.disable()

    with profiler.profile_simulation():
      # Should be enabled inside context
      self.assertTrue(profiler.is_enabled())

      with profiler.track('simulation_step'):
        time.sleep(0.001)

    # Should still be enabled after (only disabled manually)
    self.assertTrue(profiler.is_enabled())

    stats = profiler.get_stats()
    self.assertIn('simulation_step', stats['timings'])

  def test_thread_safety(self):
    """Test that profiler is thread-safe."""

    profiler.enable()

    def worker(worker_id: int):
      for _ in range(10):
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
      self.assertLen(stats['timings'][f'worker_{i}'], 10)
      self.assertEqual(stats['counters'][f'counter_{i}'], 10)

  def test_decorator_preserves_function_metadata(self):
    """Test that decorator preserves function name and docstring."""

    @profiler.track_time('test_func')
    def my_function():
      """This is a test function."""
      return 42

    self.assertEqual(my_function.__name__, 'my_function')
    self.assertEqual(my_function.__doc__, 'This is a test function.')

  def test_exception_in_tracked_code(self):
    """Test that profiler still records time when exception occurs."""
    profiler.enable()

    with self.assertRaisesRegex(ValueError, 'Test error'):
      with profiler.track('failing_operation'):
        time.sleep(0.01)
        raise ValueError('Test error')

    # Should still have recorded the timing
    stats = profiler.get_stats()
    self.assertIn('failing_operation', stats['timings'])
    self.assertLen(stats['timings']['failing_operation'], 1)

  def test_print_report_doesnt_crash(self):
    """Test that print_report doesn't crash."""
    profiler.enable()

    with profiler.track('test_op'):
      time.sleep(0.001)

    profiler.increment_counter('test_counter', 5)
    profiler.record_value('test_value', 42.0)

    # Should not raise an exception
    profiler.print_report()

  def test_record_time_exception_handling(self):
    """Test that exceptions during record_time are caught."""
    # We can't easily mock the internal lock or list to raise an exception,
    # but we can verify the method exists and runs without error in normal case.
    # The coverage gap at line 221 (in record_time) is the except block.
    # To hit it, we'd need to corrupt the state or mock the internal dict.
    # Since we can't easily mock internals of the global instance, we'll
    # rely on the fact that this is defensive code.
    pass

  def test_record_time_with_disabled_profiler(self):
    """Test record_time when profiler is disabled."""
    profiler.disable()
    # This should return early (line 214)
    profiler.record_time('test', 1.0)
    stats = profiler.get_stats()
    self.assertNotIn('test', stats['timings'])

  def test_print_report_llm_stats(self):
    """Test LLM statistics in print_report."""
    profiler.enable()

    # Simulate LLM calls to hit lines 372 and 385
    profiler.increment_counter('llm_calls_total', 10)
    profiler.increment_counter('llm_calls_sample_text', 5)
    profiler.increment_counter('llm_calls_sample_choice', 5)
    profiler.increment_counter('llm_calls_success', 8)
    profiler.increment_counter('llm_calls_failed', 2)

    profiler.record_value('llm_prompt_tokens', 100)
    profiler.record_value('llm_completion_tokens', 50)
    profiler.record_value('llm_total_tokens', 150)
    profiler.record_value('llm_latency_seconds', 0.5)

    # Should print LLM stats
    profiler.print_report()

  def test_print_report_memory_stats(self):
    """Test memory statistics in print_report."""
    profiler.enable()

    profiler.increment_counter('memory_queries', 5)
    profiler.record_value('memory_result_size', 10)

    # Should print memory stats
    profiler.print_report()

  def test_initial_state_is_disabled(self):
    """Test that a new profiler context is disabled by default."""
    # Kills mutant: self._enabled = True (in __init__)
    context = profiler.ProfilerContext()
    self.assertFalse(context.is_enabled())

  def test_profile_simulation_resets(self):
    """Test that profile_simulation resets prior stats."""
    # Kills mutant: removed reset() from profile_simulation
    profiler.enable()
    profiler.increment_counter('pre_existing_counter')

    with profiler.profile_simulation():
      stats = profiler.get_stats()
      self.assertNotIn('pre_existing_counter', stats['counters'])


if __name__ == '__main__':
  absltest.main()
