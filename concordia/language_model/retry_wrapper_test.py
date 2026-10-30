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

import contextlib
from typing import Sequence
from unittest import mock

from absl.testing import absltest
import tenacity
from concordia.language_model import language_model
from concordia.language_model import retry_wrapper


class _ScriptedModel(language_model.LanguageModel):
  """A model that raises the next exception from a script before succeeding.

  When the failure script is exhausted, either succeeds or keeps raising the
  last exception, depending on `raise_when_exhausted`.
  """

  def __init__(self, failures, response='response', raise_when_exhausted=False):
    self._failures = list(failures)
    self._response = response
    self._keep_raising = raise_when_exhausted
    self.call_count = 0

  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Sequence[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    del prompt, max_tokens, terminators, temperature, top_p, top_k
    del timeout, seed
    self.call_count += 1
    if self._failures:
      raise self._failures.pop(0)
    if self._keep_raising:
      raise ValueError('boom')
    return self._response

  def sample_choice(self, prompt, responses, *, seed=None):
    del prompt, responses, seed
    raise NotImplementedError('not used in these tests')


class _RecordingSleeper:
  """Replaces tenacity's sleep clock and records every requested wait."""

  def __init__(self):
    self.waits = []

  def __call__(self, seconds):
    self.waits.append(seconds)


@contextlib.contextmanager
def _record_waits():
  """Routes tenacity sleeps for new Retrying instances into a recorder.

  tenacity binds its default sleep function at class-definition time, so
  patching `tenacity.nap.sleep` has no effect on instances created later.
  Patching the `Retrying` class used by `tenacity.retry()` is the reliable
  hook: each decoration constructs a fresh instance, and the instance-level
  `sleep` attribute is swapped for the recorder right after construction.
  """
  sleeper = _RecordingSleeper()

  class _RecordingRetrying(tenacity.Retrying):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.sleep = sleeper

  with mock.patch.object(tenacity, 'Retrying', _RecordingRetrying):
    yield sleeper


class RetryLanguageModelTest(absltest.TestCase):

  def test_final_exception_is_reraised_unchanged(self):
    model = _ScriptedModel(failures=[], raise_when_exhausted=True)
    wrapped = retry_wrapper.RetryLanguageModel(
        model, retry_tries=3, retry_delay=0.01, jitter=(0, 0)
    )
    # The original ValueError must propagate, not tenacity.RetryError, and
    # the model must have been attempted exactly `retry_tries` times.
    with self.assertRaises(ValueError) as context_manager:
      wrapped.sample_text('prompt')
    self.assertEqual(str(context_manager.exception), 'boom')
    self.assertEqual(model.call_count, 3)

  def test_transient_error_succeeds_on_second_try(self):
    model = _ScriptedModel(
        failures=[ValueError('transient')], response='recovered'
    )
    wrapped = retry_wrapper.RetryLanguageModel(
        model, retry_tries=3, retry_delay=0.01, jitter=(0, 0)
    )
    self.assertEqual(wrapped.sample_text('prompt'), 'recovered')
    self.assertEqual(model.call_count, 2)

  def test_fixed_wait_used_when_exponential_backoff_disabled(self):
    model = _ScriptedModel(failures=[ValueError('x'), ValueError('y')])
    wrapped = retry_wrapper.RetryLanguageModel(
        model,
        retry_tries=3,
        retry_delay=1.5,
        jitter=(0, 0),
        exponential_backoff=False,
    )
    with _record_waits() as sleeper:
      self.assertEqual(wrapped.sample_text('prompt'), 'response')
    # With no exponential backoff and no jitter, every retry waits exactly
    # the fixed delay.
    self.assertEqual(model.call_count, 3)
    self.assertEqual(sleeper.waits, [1.5, 1.5])

  def test_exponential_backoff_grows_between_attempts(self):
    model = _ScriptedModel(failures=[ValueError('x'), ValueError('y')])
    wrapped = retry_wrapper.RetryLanguageModel(
        model,
        retry_tries=3,
        retry_delay=1.0,
        jitter=(0, 0),
        exponential_backoff=True,
        backoff_factor=2.0,
    )
    with _record_waits() as sleeper:
      self.assertEqual(wrapped.sample_text('prompt'), 'response')
    self.assertEqual(model.call_count, 3)
    self.assertEqual(len(sleeper.waits), 2)
    self.assertGreater(sleeper.waits[1], sleeper.waits[0])


if __name__ == '__main__':
  absltest.main()
