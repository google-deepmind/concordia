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

"""Tests for CallLimitLanguageModel."""

import threading

from absl.testing import absltest
from concordia.language_model import call_limit_wrapper
from concordia.language_model import language_model


class _CountingModel(language_model.LanguageModel):
  """A model that counts how many times it was actually invoked."""

  def __init__(self):
    self.sample_text_calls = 0
    self.sample_choice_calls = 0
    self._lock = threading.Lock()

  def sample_text(self, prompt, **kwargs):
    del prompt, kwargs
    with self._lock:
      self.sample_text_calls += 1
    return 'response'

  def sample_choice(self, prompt, responses, **kwargs):
    del prompt, kwargs
    with self._lock:
      self.sample_choice_calls += 1
    return 0, responses[0], {}


class CallLimitLanguageModelTest(absltest.TestCase):

  def test_calls_pass_through_under_the_limit(self):
    model = _CountingModel()
    wrapped = call_limit_wrapper.CallLimitLanguageModel(model, max_calls=3)
    for _ in range(3):
      self.assertEqual(wrapped.sample_text('prompt'), 'response')
    self.assertEqual(model.sample_text_calls, 3)

  def test_sample_text_returns_empty_string_once_limit_reached(self):
    model = _CountingModel()
    wrapped = call_limit_wrapper.CallLimitLanguageModel(model, max_calls=1)
    self.assertEqual(wrapped.sample_text('prompt'), 'response')
    self.assertEqual(wrapped.sample_text('prompt'), '')
    self.assertEqual(wrapped.sample_text('prompt'), '')
    # The underlying model must not be called once the limit is reached.
    self.assertEqual(model.sample_text_calls, 1)

  def test_sample_choice_returns_first_response_once_limit_reached(self):
    model = _CountingModel()
    wrapped = call_limit_wrapper.CallLimitLanguageModel(model, max_calls=1)
    wrapped.sample_choice('prompt', ['a', 'b', 'c'])
    result = wrapped.sample_choice('prompt', ['x', 'y', 'z'])
    self.assertEqual(result, (0, 'x', {}))
    self.assertEqual(model.sample_choice_calls, 1)

  def test_sample_text_and_sample_choice_share_the_same_budget(self):
    model = _CountingModel()
    wrapped = call_limit_wrapper.CallLimitLanguageModel(model, max_calls=2)
    wrapped.sample_text('prompt')
    wrapped.sample_choice('prompt', ['a', 'b'])
    # Budget is now exhausted; further calls of either kind should not reach
    # the underlying model.
    wrapped.sample_text('prompt')
    wrapped.sample_choice('prompt', ['a', 'b'])
    self.assertEqual(model.sample_text_calls, 1)
    self.assertEqual(model.sample_choice_calls, 1)

  def test_call_limit_is_enforced_exactly_under_concurrency(self):
    # Regression test: the call counter used to be an unsynchronized
    # `self._calls += 1` shared across every thread that calls the wrapped
    # model. EntityAgent runs component calls concurrently via a
    # ThreadPoolExecutor and typically shares one language model instance
    # across an entire simulation, so this counter is genuinely
    # multi-threaded in normal use: an unsynchronized read-modify-write on a
    # shared counter is a data race regardless of whether any particular
    # CPython build's GIL happens to hide it, and it's an outright bug on
    # free-threaded (no-GIL) Python builds. With the counter under a lock,
    # exactly `max_calls` of many more concurrent attempts must get through.
    num_threads = 200
    max_calls = 37
    model = _CountingModel()
    wrapped = call_limit_wrapper.CallLimitLanguageModel(
        model, max_calls=max_calls
    )

    def worker():
      wrapped.sample_text('prompt')

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertEqual(model.sample_text_calls, max_calls)
    self.assertEqual(wrapped._calls, max_calls)  # pylint: disable=protected-access


if __name__ == '__main__':
  absltest.main()
