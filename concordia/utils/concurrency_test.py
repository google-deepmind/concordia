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

import functools
import time

from absl.testing import absltest
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


if __name__ == '__main__':
  absltest.main()
