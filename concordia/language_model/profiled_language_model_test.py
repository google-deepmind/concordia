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

"""Tests for profiled_language_model module."""

import time
from absl.testing import absltest
from concordia.language_model import language_model
from concordia.language_model import profiled_language_model
from concordia.utils import profiler


class MockLanguageModel(language_model.LanguageModel):
  """Mock language model for testing."""

  def __init__(self, delay: float = 0.01):
    """Initialize mock model.

    Args:
      delay: Simulated delay for each call (in seconds).
    """
    self._delay = delay
    self.sample_text_calls = 0
    self.sample_choice_calls = 0

  def sample_text(self, prompt: str, **kwargs) -> str:
    """Sample text with simulated delay."""
    time.sleep(self._delay)
    self.sample_text_calls += 1
    return 'This is a mock response to: ' + prompt[:20]

  def sample_choice(self, prompt: str, responses, **kwargs):
    """Sample choice with simulated delay."""
    time.sleep(self._delay)
    self.sample_choice_calls += 1
    return 0, responses[0], {'score': 0.9}


class FailingMockLanguageModel(language_model.LanguageModel):
  """Mock language model that fails."""

  def sample_text(self, prompt: str, **kwargs) -> str:
    """Always raises an exception."""
    raise RuntimeError('Mock LLM failure')

  def sample_choice(self, prompt: str, responses, **kwargs):
    """Always raises an exception."""
    raise RuntimeError('Mock LLM failure')


class ProfiledLanguageModelTest(absltest.TestCase):
  """Tests for the ProfiledLanguageModel wrapper."""

  def setUp(self):
    """Reset profiler before each test."""
    super().setUp()
    profiler.reset()
    profiler.disable()

  def test_sample_text_without_profiling(self):
    """Test that wrapper works when profiling is disabled."""
    base_model = MockLanguageModel()
    profiled_model = profiled_language_model.ProfiledLanguageModel(
        base_model, model_name='test-model'
    )

    profiler.disable()

    response = profiled_model.sample_text('Hello')
    assert response.startswith('This is a mock response')
    assert base_model.sample_text_calls == 1

    # No profiling data should be recorded
    stats = profiler.get_stats()
    assert not stats['timings']
    assert not stats['counters']

  def test_sample_text_with_profiling(self):
    """Test that profiling tracks sample_text calls."""
    base_model = MockLanguageModel()
    profiled_model = profiled_language_model.ProfiledLanguageModel(
        base_model, model_name='test-model'
    )

    profiler.enable()

    response = profiled_model.sample_text('Hello, how are you?')
    self.assertStartsWith(response, 'This is a mock response')

    stats = profiler.get_stats()

    # Check timing data
    self.assertIn('llm_sample_text', stats['timings'])
    self.assertLen(stats['timings']['llm_sample_text'], 1)

    # Check counters
    self.assertEqual(stats['counters']['llm_calls_total'], 1)
    self.assertEqual(stats['counters']['llm_calls_sample_text'], 1)
    self.assertEqual(stats['counters']['llm_calls_success'], 1)

    # Check token estimates
    self.assertIn('llm_prompt_tokens', stats['values'])
    self.assertIn('llm_completion_tokens', stats['values'])
    self.assertIn('llm_total_tokens', stats['values'])

  def test_sample_choice_with_profiling(self):
    """Test that profiling tracks sample_choice calls."""
    base_model = MockLanguageModel()
    profiled_model = profiled_language_model.ProfiledLanguageModel(
        base_model, model_name='test-model'
    )

    profiler.enable()

    responses = ['Option A', 'Option B', 'Option C']
    index, response, _ = profiled_model.sample_choice('Choose one', responses)

    self.assertEqual(index, 0)
    self.assertEqual(response, 'Option A')

    stats = profiler.get_stats()

    # Check timing data
    self.assertIn('llm_sample_choice', stats['timings'])
    self.assertLen(stats['timings']['llm_sample_choice'], 1)

    # Check counters
    self.assertEqual(stats['counters']['llm_calls_total'], 1)
    self.assertEqual(stats['counters']['llm_calls_sample_choice'], 1)
    self.assertEqual(stats['counters']['llm_calls_success'], 1)

  def test_multiple_calls(self):
    """Test profiling with multiple calls."""
    base_model = MockLanguageModel()
    profiled_model = profiled_language_model.ProfiledLanguageModel(
        base_model, model_name='test-model'
    )

    profiler.enable()

    # Make multiple calls
    for i in range(5):
      profiled_model.sample_text(f'Prompt {i}')

    stats = profiler.get_stats()

    self.assertLen(stats['timings']['llm_sample_text'], 5)
    self.assertEqual(stats['counters']['llm_calls_total'], 5)
    self.assertEqual(stats['counters']['llm_calls_sample_text'], 5)
    self.assertEqual(stats['counters']['llm_calls_success'], 5)

  def test_failed_call_tracking(self):
    """Test that failed calls are tracked correctly."""
    failing_model = FailingMockLanguageModel()
    profiled_model = profiled_language_model.ProfiledLanguageModel(
        failing_model, model_name='test-model'
    )

    profiler.enable()

    # Try to make a call that will fail
    with self.assertRaisesRegex(RuntimeError, 'Mock LLM failure'):
      profiled_model.sample_text('This will fail')

    stats = profiler.get_stats()

    # Check that failure was tracked
    self.assertEqual(stats['counters']['llm_calls_total'], 1)
    self.assertEqual(stats['counters']['llm_calls_sample_text'], 1)
    self.assertEqual(stats['counters']['llm_calls_failed'], 1)
    self.assertNotIn('llm_calls_success', stats['counters'])

    # Timing should still be recorded
    self.assertIn('llm_sample_text', stats['timings'])

  def test_token_estimation(self):
    """Test token estimation functionality."""
    # Test the estimate_tokens function directly
    short_text = 'Hi'
    medium_text = 'This is a medium length sentence.'
    long_text = 'This is a much longer piece of text ' * 10

    short_tokens = profiled_language_model.estimate_tokens(short_text)
    medium_tokens = profiled_language_model.estimate_tokens(medium_text)
    long_tokens = profiled_language_model.estimate_tokens(long_text)

    # Check relative sizes
    self.assertLess(short_tokens, medium_tokens)
    self.assertLess(medium_tokens, long_tokens)

    # Check minimum of 1 token
    self.assertEqual(profiled_language_model.estimate_tokens(''), 1)

    # Check exact token count for a known length to kill off-by-one mutation.
    # '1234' has length 4. 4 // 4 = 1.
    # If mutation is (len // 4 + 1), result would be 2.
    self.assertEqual(profiled_language_model.estimate_tokens('1234'), 1)

  def test_latency_tracking(self):
    """Test that latency is tracked properly."""
    base_model = MockLanguageModel(delay=0.02)  # 20ms delay
    profiled_model = profiled_language_model.ProfiledLanguageModel(
        base_model, model_name='test-model'
    )

    profiler.enable()

    profiled_model.sample_text('Test latency')

    stats = profiler.get_stats()

    # Check latency was recorded
    latencies = stats['values']['llm_latency_seconds']
    self.assertLen(latencies, 1)
    # Should be roughly 20ms (allow some margin)
    self.assertGreaterEqual(latencies[0], 0.02)

  def test_mixed_calls(self):
    """Test profiling with both sample_text and sample_choice."""
    base_model = MockLanguageModel()
    profiled_model = profiled_language_model.ProfiledLanguageModel(
        base_model, model_name='test-model'
    )

    profiler.enable()

    # Make different types of calls
    profiled_model.sample_text('Hello')
    profiled_model.sample_choice('Choose', ['A', 'B'])
    profiled_model.sample_text('World')

    stats = profiler.get_stats()

    self.assertEqual(stats['counters']['llm_calls_total'], 3)
    self.assertEqual(stats['counters']['llm_calls_sample_text'], 2)
    self.assertEqual(stats['counters']['llm_calls_sample_choice'], 1)
    self.assertEqual(stats['counters']['llm_calls_success'], 3)

  def test_sample_choice_without_profiling(self):
    """Test sample_choice when profiling is disabled (hits fast path)."""
    base_model = MockLanguageModel()
    profiled_model = profiled_language_model.ProfiledLanguageModel(
        base_model, model_name='test-model'
    )

    profiler.disable()

    responses = ['Option A', 'Option B']
    index, response, _ = profiled_model.sample_choice('Choose', responses)

    self.assertEqual(index, 0)
    self.assertEqual(response, 'Option A')
    self.assertEqual(base_model.sample_choice_calls, 1)

    # No profiling data should be recorded
    stats = profiler.get_stats()
    self.assertFalse(stats['timings'])
    self.assertFalse(stats['counters'])


if __name__ == '__main__':
  absltest.main()
