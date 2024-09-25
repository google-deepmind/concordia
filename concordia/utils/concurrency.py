# Copyright 2023 DeepMind Technologies Limited.
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

"""Concurrency helpers."""

from collections.abc import Collection, Iterator, Mapping, Sequence
from concurrent import futures
import contextlib
import functools
from typing import Any, Callable, TypeVar

from absl import logging

_T = TypeVar('_T')


@contextlib.contextmanager
def _executor(**kwargs) -> Iterator[futures.ThreadPoolExecutor]:
  """Context manager for a concurrent.futures.ThreadPoolExecutor.

  On normal __exit__ this context manager will behave like
  `ThreadPoolExecutor.__exit__`: it will block until all running and pending
  threads complete.

  However, on an __exit__ due to an error, the executor will be shutdown
  immediately without waiting for the running futures to complete, and all
  pending futures will be cancelled. This allows errors to quickly propagate to
  the caller.

  Args:
    **kwargs: Forwarded to ThreadPoolExecutor.

  Yields:
    A thread pool executor.
  """
  thread_executor = futures.ThreadPoolExecutor(**kwargs)
  try:
    yield thread_executor
  except BaseException:
    thread_executor.shutdown(wait=False, cancel_futures=True)
    raise
  else:
    thread_executor.shutdown()


def _run_task(key: str, fn: Callable[[], _T]) -> Callable[[], _T]:
  """Returns fn() and logs any error."""
  try:
    return fn()
  except:
    logging.exception('Error in task %s', key)
    raise


def _as_completed(
    tasks: Mapping[str, Callable[[], _T]],
    *,
    timeout: float | None = None,
    max_workers: int | None = None,
) -> Iterator[tuple[str, futures.Future[_T]]]:
  """Maps a function to a sequence of values in parallel.

  IMPORTANT: Passed callables must be threadsafe.

  Args:
    tasks: callables to execute (MUST BE THREADSAFE)
    timeout: the maximum number of seconds to wait for all tasks to complete.
    max_workers: them maximum number of parallel jobs. If None will use as many
      workers as there are tasks.

  Yields:
    (key, future) as tasks complete.

  Raises:
    TimeoutError: If all the results are not generated before the timeout.
  """
  if not tasks:
    return
  if max_workers is None:
    max_workers = len(tasks)
  with _executor(max_workers=max_workers) as executor_:
    key_by_future = {
        executor_.submit(_run_task, key, task): key
        for key, task in tasks.items()
    }
    for future in futures.as_completed(key_by_future, timeout=timeout):
      yield key_by_future[future], future


def run_tasks(
    tasks: Mapping[str, Callable[[], _T]],
    *,
    timeout: float | None = None,
    max_workers: int | None = None,
) -> Mapping[str, _T]:
  """Runs the callables in parallel, blocks until first failure.

  IMPORTANT: Passed callables must be threadsafe.

  Args:
    tasks: callables to execute (MUST BE THREADSAFE)
    timeout: the maximum number of seconds to wait.
    max_workers: them maximum number of parallel jobs. If None will use as many
      workers as there are tasks.

  Returns:
    The results fn(*arg) for arg in args]
    However, the calls will be executed concurrently.

  Raises:
    TimeoutError: If all the results are not generated before the timeout.
    Exception: If any task raises an exception.
  """
  return {
      key: future.result()
      for key, future in _as_completed(
          tasks, timeout=timeout, max_workers=max_workers
      )
  }


def run_tasks_in_background(
    tasks: Mapping[str, Callable[[], _T]],
    *,
    timeout: float | None = None,
    max_workers: int | None = None,
) -> tuple[Mapping[str, _T], Mapping[str, BaseException]]:
  """Runs the callables in parallel, blocks until all complete.

  IMPORTANT: Passed callables must be threadsafe.

  Args:
    tasks: callables to execute (MUST BE THREADSAFE)
    timeout: the maximum number of seconds to wait.
    max_workers: them maximum number of parallel jobs. If None will use as many
      workers as there are tasks.

  Returns:
    (results, errors): a mappings from key to the result of the callable or the
    exception it raised. Thus if no task raised an error, errors will be empty.
  """
  results = {}
  errors = {}
  try:
    for key, future in _as_completed(
        tasks, timeout=timeout, max_workers=max_workers
    ):
      error = future.exception()
      if error is not None:
        errors[key] = error
      else:
        results[key] = future.result()
  except TimeoutError as error:
    unfinished = tasks.keys() - results.keys() - errors.keys()
    for key in unfinished:
      errors[key] = error
  return results, errors


def map_parallel(
    fn: Callable[..., _T],
    *args: Collection[Any],
    timeout: float | None = None,
    max_workers: int | None = None,
) -> Sequence[_T]:
  """Runs `map(*args)` in parallel.

  IMPORTANT: Passed callables must be threadsafe.

  Args:
    fn: function to execute (MUST BE THREADSAFE)
    *args: arguments to pass to function.
    timeout: the maximum number of seconds to wait.
    max_workers: them maximum number of parallel jobs. If None, will use as many
      workers as there are arguments.

  Returns:
    [fn(arg0, arg1, ...) for arg0, arg1, ...  in zip(*args)]
    However, the calls will be executed concurrently.

  Raises:
    TimeoutError: If all the results are not generated before the timeout.
    Exception: If fn(*args) raises for any values.
  """
  tasks = {
      str(n): functools.partial(fn, *arg)
      for n, arg in enumerate(zip(*args, strict=True))
  }
  results = run_tasks(tasks, timeout=timeout, max_workers=max_workers)
  return [results[key] for key in tasks]
