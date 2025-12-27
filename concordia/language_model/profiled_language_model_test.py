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
from concordia.language_model import language_model
from concordia.language_model import profiled_language_model
from concordia.utils import profiler
import pytest


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


class TestProfiledLanguageModel:
  """Tests for the ProfiledLanguageModel wrapper."""

  def setup_method(self):
    """Reset profiler before each test."""
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
    assert response.startswith('This is a mock response')

    stats = profiler.get_stats()

    # Check timing data
    assert 'llm_sample_text' in stats['timings']
    assert len(stats['timings']['llm_sample_text']) == 1

    # Check counters
    assert stats['counters']['llm_calls_total'] == 1
    assert stats['counters']['llm_calls_sample_text'] == 1
    assert stats['counters']['llm_calls_success'] == 1

    # Check token estimates
    assert 'llm_prompt_tokens' in stats['values']
    assert 'llm_completion_tokens' in stats['values']
    assert 'llm_total_tokens' in stats['values']

  def test_sample_choice_with_profiling(self):
    """Test that profiling tracks sample_choice calls."""
    base_model = MockLanguageModel()
    profiled_model = profiled_language_model.ProfiledLanguageModel(
        base_model, model_name='test-model'
    )

    profiler.enable()

    responses = ['Option A', 'Option B', 'Option C']
    index, response, info = profiled_model.sample_choice('Choose one', responses)

    assert index == 0
    assert response == 'Option A'

    stats = profiler.get_stats()

    # Check timing data
    assert 'llm_sample_choice' in stats['timings']
    assert len(stats['timings']['llm_sample_choice']) == 1

    # Check counters
    assert stats['counters']['llm_calls_total'] == 1
    assert stats['counters']['llm_calls_sample_choice'] == 1
    assert stats['counters']['llm_calls_success'] == 1

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

    assert len(stats['timings']['llm_sample_text']) == 5
    assert stats['counters']['llm_calls_total'] == 5
    assert stats['counters']['llm_calls_sample_text'] == 5
    assert stats['counters']['llm_calls_success'] == 5

  def test_failed_call_tracking(self):
    """Test that failed calls are tracked correctly."""
    failing_model = FailingMockLanguageModel()
    profiled_model = profiled_language_model.ProfiledLanguageModel(
        failing_model, model_name='test-model'
    )

    profiler.enable()

    # Try to make a call that will fail
    with pytest.raises(RuntimeError, match='Mock LLM failure'):
      profiled_model.sample_text('This will fail')

    stats = profiler.get_stats()

    # Check that failure was tracked
    assert stats['counters']['llm_calls_total'] == 1
    assert stats['counters']['llm_calls_sample_text'] == 1
    assert stats['counters']['llm_calls_failed'] == 1
    assert 'llm_calls_success' not in stats['counters']

    # Timing should still be recorded
    assert 'llm_sample_text' in stats['timings']

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
    assert short_tokens < medium_tokens < long_tokens

    # Check minimum of 1 token
    assert profiled_language_model.estimate_tokens('') == 1

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
    assert len(latencies) == 1
    # Should be roughly 20ms (allow some margin)
    assert latencies[0] >= 0.02

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

    assert stats['counters']['llm_calls_total'] == 3
    assert stats['counters']['llm_calls_sample_text'] == 2
    assert stats['counters']['llm_calls_sample_choice'] == 1
    assert stats['counters']['llm_calls_success'] == 3

  def test_profiling_overhead_minimal(self):
    """Test that profiling adds minimal absolute overhead."""
    base_model = MockLanguageModel(delay=0.05)  # Use 50ms delay

    # Measure without profiling
    profiler.disable()
    model_without = profiled_language_model.ProfiledLanguageModel(base_model)

    start = time.perf_counter()
    for _ in range(20):
      model_without.sample_text('Test')
    time_without = time.perf_counter() - start

    # Measure with profiling
    profiler.enable()
    profiler.reset()
    base_model_2 = MockLanguageModel(delay=0.05)
    model_with = profiled_language_model.ProfiledLanguageModel(base_model_2)

    start = time.perf_counter()
    for _ in range(20):
      model_with.sample_text('Test')
    time_with = time.perf_counter() - start

    # Absolute overhead should be small (less than 50ms for 20 calls)
    absolute_overhead = time_with - time_without
    assert absolute_overhead < 0.05, \
        f'Profiling overhead too high: {absolute_overhead*1000:.1f}ms'
