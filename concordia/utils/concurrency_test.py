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

import asyncio
from concurrent import futures
import contextlib
import functools
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.utils import concurrency


class ExpectedError(Exception):
  pass


def wait_for(seconds):
  time.sleep(seconds)


def error_after(seconds):
  time.sleep(seconds)
  raise ExpectedError()


def return_after(seconds, value):
  time.sleep(seconds)
  return value


class ConcurrencyTest(absltest.TestCase):

  def test_executor_fails_fast(self):
    start_time = time.time()
    try:
      with concurrency._executor() as executor:
        executor.submit(wait_for, 5)
        raise ExpectedError()
    except ExpectedError:
      pass
    end_time = time.time()
    self.assertLess(end_time - start_time, 2)

  def test_run_tasks_fails_fast(self):
    tasks = {
        'wait': functools.partial(wait_for, 5),
        'error': functools.partial(error_after, 1),
    }
    start_time = time.time()
    try:
      concurrency.run_tasks(tasks)
    except ExpectedError:
      pass
    end_time = time.time()
    self.assertLess(end_time - start_time, 2)

  def test_run_tasks_error(self):
    tasks = {
        'wait': functools.partial(wait_for, 5),
        'error': functools.partial(error_after, 1),
    }
    with self.assertRaises(ExpectedError):
      concurrency.run_tasks(tasks)

  def test_run_tasks_timeout(self):
    tasks = {
        'wait': functools.partial(wait_for, 5),
    }
    with self.assertRaises(TimeoutError):
      concurrency.run_tasks(tasks, timeout=1)

  def test_run_tasks_success(self):
    tasks = {
        'a': functools.partial(return_after, 1, 'a'),
        'b': functools.partial(return_after, 0.1, 'b'),
        'c': functools.partial(return_after, 0.1, 'c'),
    }
    results = concurrency.run_tasks(tasks)
    self.assertEqual(results, {'a': 'a', 'b': 'b', 'c': 'c'})

  def test_run_tasks_max_workers_none(self):
    tasks = {
        'a': functools.partial(return_after, 0.1, 'a'),
        'b': functools.partial(return_after, 0.1, 'b'),
    }
    results = concurrency.run_tasks(tasks, max_workers=None)
    self.assertEqual(results, {'a': 'a', 'b': 'b'})

  def test_run_tasks_in_background(self):
    tasks = {
        'a': functools.partial(return_after, 1, 'a'),
        'b': functools.partial(return_after, 0.1, 'b'),
        'c': functools.partial(return_after, 0.1, 'c'),
        'error': functools.partial(error_after, 1),
        'wait': functools.partial(wait_for, 5),
    }
    results, errors = concurrency.run_tasks_in_background(tasks, timeout=2)
    with self.subTest('results'):
      self.assertEqual(results, {'a': 'a', 'b': 'b', 'c': 'c'})
    with self.subTest('errors'):
      self.assertEqual(
          {key: type(error) for key, error in errors.items()},
          {'error': ExpectedError, 'wait': TimeoutError},
      )

  def test_run_tasks_empty(self):
    results = concurrency.run_tasks({})
    self.assertEmpty(results)

  def test_map_parallel(self):
    results = concurrency.map_parallel(
        return_after, [1, 0.5, 0.1], ['a', 'b', 'c']
    )
    self.assertEqual(results, ['a', 'b', 'c'])


class TaskErrorLoggingTest(parameterized.TestCase):

  @parameterized.product(
      exception_type=(
          KeyboardInterrupt,
          SystemExit,
          GeneratorExit,
          asyncio.CancelledError,
          ValueError,
          RuntimeError,
          futures.CancelledError,
      ),
      background=(False, True),
      reuse_executor=(False, True),
  )
  def test_only_ordinary_exceptions_are_logged(
      self, exception_type, background, reuse_executor
  ):
    error = exception_type('expected task failure')

    def fail():
      raise error

    executor_context = (
        futures.ThreadPoolExecutor(max_workers=1)
        if reuse_executor
        else contextlib.nullcontext(None)
    )
    with executor_context as executor:
      with mock.patch.object(concurrency.logging, 'exception') as log_error:
        if background:
          results, errors = concurrency.run_tasks_in_background(
              {'task': fail}, executor=executor
          )
          self.assertEmpty(results)
          self.assertIs(errors['task'], error)
        else:
          with self.assertRaises(exception_type) as caught:
            concurrency.run_tasks({'task': fail}, executor=executor)
          self.assertIs(caught.exception, error)

        if isinstance(error, Exception):
          log_error.assert_called_once_with('Error in task %s', 'task')
        else:
          log_error.assert_not_called()

      # Externally owned executors must remain usable after a failed task.
      if executor is not None:
        self.assertEqual(executor.submit(lambda: 42).result(timeout=5), 42)


if __name__ == '__main__':
  absltest.main()
