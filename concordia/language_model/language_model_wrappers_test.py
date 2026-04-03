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

"""Tests for call_limit_wrapper and retry_wrapper modules."""

from collections.abc import Mapping, Sequence
from typing import Any

from absl.testing import absltest
from concordia.language_model import call_limit_wrapper
from concordia.language_model import language_model
from concordia.language_model import retry_wrapper
import tenacity


class MockLanguageModel(language_model.LanguageModel):
  """Mock language model that records calls and supports forced failures."""

  def __init__(
      self,
      fail_times: int = 0,
      response: str = 'mock response',
  ) -> None:
    """Initialize mock model.

    Args:
      fail_times: Number of times to raise an exception before succeeding.
      response: The text to return from sample_text on success.
    """
    self._fail_times = fail_times
    self._response = response
    self.sample_text_calls = 0
    self.sample_choice_calls = 0

  def sample_text(self, prompt: str, **kwargs) -> str:
    self.sample_text_calls += 1
    if self._fail_times > 0:
      self._fail_times -= 1
      raise ConnectionError('Simulated API error')
    return self._response

  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    self.sample_choice_calls += 1
    if self._fail_times > 0:
      self._fail_times -= 1
      raise ConnectionError('Simulated API error')
    return 1, responses[1], {'score': 0.8}


# ---------------------------------------------------------------------------
# CallLimitLanguageModel tests
# ---------------------------------------------------------------------------


class CallLimitLanguageModelTest(absltest.TestCase):
  """Tests for CallLimitLanguageModel."""

  def test_sample_text_within_limit(self):
    mock = MockLanguageModel(response='hello')
    model = call_limit_wrapper.CallLimitLanguageModel(mock, max_calls=5)
    result = model.sample_text('prompt')
    self.assertEqual(result, 'hello')
    self.assertEqual(mock.sample_text_calls, 1)

  def test_sample_text_at_limit_returns_empty_string(self):
    mock = MockLanguageModel(response='hello')
    model = call_limit_wrapper.CallLimitLanguageModel(mock, max_calls=2)
    model.sample_text('prompt 1')
    model.sample_text('prompt 2')
    # Third call exceeds limit
    result = model.sample_text('prompt 3')
    self.assertEqual(result, '')
    # Underlying model was called exactly max_calls times
    self.assertEqual(mock.sample_text_calls, 2)

  def test_sample_text_exactly_at_limit_is_still_served(self):
    mock = MockLanguageModel(response='last')
    model = call_limit_wrapper.CallLimitLanguageModel(mock, max_calls=1)
    result = model.sample_text('prompt')
    self.assertEqual(result, 'last')
    self.assertEqual(mock.sample_text_calls, 1)

  def test_sample_choice_within_limit(self):
    mock = MockLanguageModel()
    model = call_limit_wrapper.CallLimitLanguageModel(mock, max_calls=5)
    idx, text, _ = model.sample_choice('prompt', ['a', 'b', 'c'])
    self.assertEqual(idx, 1)
    self.assertEqual(text, 'b')
    self.assertEqual(mock.sample_choice_calls, 1)

  def test_sample_choice_at_limit_returns_first_response(self):
    mock = MockLanguageModel()
    model = call_limit_wrapper.CallLimitLanguageModel(mock, max_calls=1)
    model.sample_choice('prompt', ['a', 'b', 'c'])
    # Second call exceeds limit — should return first response
    idx, text, _ = model.sample_choice('prompt', ['x', 'y'])
    self.assertEqual(idx, 0)
    self.assertEqual(text, 'x')
    self.assertEqual(mock.sample_choice_calls, 1)

  def test_call_count_shared_between_text_and_choice(self):
    mock = MockLanguageModel()
    model = call_limit_wrapper.CallLimitLanguageModel(mock, max_calls=2)
    model.sample_text('t1')
    model.sample_choice('c1', ['a', 'b'])
    # Both calls count; next call of either type should be blocked
    result = model.sample_text('t2')
    self.assertEqual(result, '')
    _, text, _ = model.sample_choice('c2', ['p', 'q'])
    self.assertEqual(text, 'p')

  def test_multiple_calls_beyond_limit_all_return_fallback(self):
    mock = MockLanguageModel()
    model = call_limit_wrapper.CallLimitLanguageModel(mock, max_calls=1)
    model.sample_text('warmup')
    for _ in range(5):
      result = model.sample_text('over-limit')
      self.assertEqual(result, '')
    self.assertEqual(mock.sample_text_calls, 1)


# ---------------------------------------------------------------------------
# RetryLanguageModel tests
# ---------------------------------------------------------------------------


class RetryLanguageModelTest(absltest.TestCase):
  """Tests for RetryLanguageModel."""

  def test_sample_text_succeeds_on_first_try(self):
    mock = MockLanguageModel(fail_times=0, response='success')
    model = retry_wrapper.RetryLanguageModel(
        mock,
        retry_on_exceptions=(ConnectionError,),
        retry_tries=3,
        retry_delay=0.0,
        jitter=(0.0, 0.0),
        exponential_backoff=False,
    )
    result = model.sample_text('prompt')
    self.assertEqual(result, 'success')
    self.assertEqual(mock.sample_text_calls, 1)

  def test_sample_text_retries_and_succeeds(self):
    mock = MockLanguageModel(fail_times=2, response='eventual success')
    model = retry_wrapper.RetryLanguageModel(
        mock,
        retry_on_exceptions=(ConnectionError,),
        retry_tries=5,
        retry_delay=0.0,
        jitter=(0.0, 0.0),
        exponential_backoff=False,
    )
    result = model.sample_text('prompt')
    self.assertEqual(result, 'eventual success')
    self.assertEqual(mock.sample_text_calls, 3)

  def test_sample_text_raises_after_all_retries_exhausted(self):
    mock = MockLanguageModel(fail_times=10)
    model = retry_wrapper.RetryLanguageModel(
        mock,
        retry_on_exceptions=(ConnectionError,),
        retry_tries=3,
        retry_delay=0.0,
        jitter=(0.0, 0.0),
        exponential_backoff=False,
    )
    # Tenacity wraps exhausted retries in RetryError; the original exception is
    # accessible via RetryError.__cause__.
    with self.assertRaises(tenacity.RetryError) as cm:
      model.sample_text('prompt')
    self.assertIsInstance(cm.exception.__cause__, ConnectionError)
    self.assertEqual(mock.sample_text_calls, 3)

  def test_sample_text_does_not_retry_unregistered_exception(self):
    class CustomError(Exception):
      pass

    class RaisingModel(language_model.LanguageModel):
      def sample_text(self, prompt, **kwargs):
        raise CustomError('not retried')
      def sample_choice(self, prompt, responses, *, seed=None):
        raise CustomError('not retried')

    model = retry_wrapper.RetryLanguageModel(
        RaisingModel(),
        retry_on_exceptions=(ConnectionError,),
        retry_tries=5,
        retry_delay=0.0,
        jitter=(0.0, 0.0),
        exponential_backoff=False,
    )
    with self.assertRaises(CustomError):
      model.sample_text('prompt')

  def test_sample_choice_succeeds_on_first_try(self):
    mock = MockLanguageModel(fail_times=0)
    model = retry_wrapper.RetryLanguageModel(
        mock,
        retry_on_exceptions=(ConnectionError,),
        retry_tries=3,
        retry_delay=0.0,
        jitter=(0.0, 0.0),
        exponential_backoff=False,
    )
    idx, text, _ = model.sample_choice('prompt', ['a', 'b', 'c'])
    self.assertEqual(idx, 1)
    self.assertEqual(text, 'b')
    self.assertEqual(mock.sample_choice_calls, 1)

  def test_sample_choice_retries_and_succeeds(self):
    mock = MockLanguageModel(fail_times=1)
    model = retry_wrapper.RetryLanguageModel(
        mock,
        retry_on_exceptions=(ConnectionError,),
        retry_tries=3,
        retry_delay=0.0,
        jitter=(0.0, 0.0),
        exponential_backoff=False,
    )
    idx, text, _ = model.sample_choice('prompt', ['a', 'b', 'c'])
    self.assertEqual(idx, 1)
    self.assertEqual(text, 'b')
    self.assertEqual(mock.sample_choice_calls, 2)

  def test_sample_choice_raises_after_all_retries_exhausted(self):
    mock = MockLanguageModel(fail_times=10)
    model = retry_wrapper.RetryLanguageModel(
        mock,
        retry_on_exceptions=(ConnectionError,),
        retry_tries=3,
        retry_delay=0.0,
        jitter=(0.0, 0.0),
        exponential_backoff=False,
    )
    # Tenacity wraps exhausted retries in RetryError; the original exception is
    # accessible via RetryError.__cause__.
    with self.assertRaises(tenacity.RetryError) as cm:
      model.sample_choice('prompt', ['a', 'b'])
    self.assertIsInstance(cm.exception.__cause__, ConnectionError)
    self.assertEqual(mock.sample_choice_calls, 3)

  def test_fixed_wait_no_exponential_backoff(self):
    """Verify that exponential_backoff=False does not raise on construction."""
    mock = MockLanguageModel()
    model = retry_wrapper.RetryLanguageModel(
        mock,
        retry_on_exceptions=(Exception,),
        retry_tries=2,
        retry_delay=0.0,
        jitter=(0.0, 0.0),
        exponential_backoff=False,
    )
    result = model.sample_text('prompt')
    self.assertEqual(result, 'mock response')

  def test_default_retries_on_any_exception(self):
    """Default retry_on_exceptions=(Exception,) catches all exceptions."""
    call_count = 0

    class FlakeyModel(language_model.LanguageModel):
      def sample_text(self, prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
          raise ValueError('flakey')
        return 'stable'

      def sample_choice(self, prompt, responses, *, seed=None):
        return 0, responses[0], {}

    model = retry_wrapper.RetryLanguageModel(
        FlakeyModel(),
        retry_tries=5,
        retry_delay=0.0,
        jitter=(0.0, 0.0),
        exponential_backoff=False,
    )
    result = model.sample_text('prompt')
    self.assertEqual(result, 'stable')
    self.assertEqual(call_count, 3)


if __name__ == '__main__':
  absltest.main()
