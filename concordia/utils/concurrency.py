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

"""Better error handling for threads.
"""

from collections.abc import Iterator
from concurrent import futures
import contextlib


@contextlib.contextmanager
def executor(
    **kwargs,
) -> Iterator[futures.ThreadPoolExecutor]:
  """Context manager for a thread executor.

  On __exit__ the thread executor will be shutdown and all futures cancelled,
  this allows errors to quickly propogate to the caller. When using this ensure
  you wait on all futures before __exit__.

  Args:
    **kwargs: Args to pass to the thread executor.

  Yields:
    A thread pool executor.
  """

  with contextlib.ExitStack() as stack:
    thread_executor = futures.ThreadPoolExecutor(**kwargs)
    stack.callback(thread_executor.shutdown, wait=False, cancel_futures=True)
    yield thread_executor
