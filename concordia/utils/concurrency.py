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

"""Better error handling for ThreadPoolExecutors."""

from collections.abc import Collection, Iterator
from concurrent import futures
import contextlib
from typing import Any, Callable, TypeVar

_T = TypeVar('_T')


@contextlib.contextmanager
def executor(**kwargs) -> Iterator[futures.ThreadPoolExecutor]:
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


def map_parallel(
    fn: Callable[..., _T],
    *iterables: Collection[Any],
    timeout: float | None = None,
) -> Iterator[_T]:
  """Maps a function to a sequence of values in parallel.

  Convenience wrapper for ThreadPoolExecutor.map.

  Args:
    fn: function to execute
    *iterables: arguments to pass to function.
    timeout: the maximum number of seconds to wait.

  Returns:
    An iterator equivalent to: map(func, *iterables) but the calls may be
    evaluated out-of-order.

  Raises:
    TimeoutError: If the entire result iterator could not be generated before
        the given timeout.
    Exception: If fn(*args) raises for any values.
  """
  with executor(max_workers=len(iterables)) as executor_:
    return executor_.map(fn, *iterables, timeout=timeout)
