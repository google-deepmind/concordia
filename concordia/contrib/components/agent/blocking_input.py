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

"""A `HumanInput` reader that another thread answers: the core of a transport.

`HumanActComponent` calls its reader on the simulation's thread and blocks
until the person answers. A web page, a chat bot or a GUI runs on another
thread. `BlockingInput` joins the two:

- the simulation thread calls the reader, which publishes the request and
  waits;
- the transport thread reads `pending()`, shows it to the person, and calls
  `submit(request_id, answer)`;
- `close()` releases a waiting simulation (raising `InputClosedError` there),
  for example when the person leaves.

Answers to a request that is no longer pending are refused, so a slow page
cannot answer the wrong question. When `HumanActComponent` rejects an answer
it asks again with the same `request_id` and an `error`; the new request
replaces the pending one.

Example (a minimal server loop):

```python
reader = blocking_input.BlockingInput()
# Simulation thread: an entity built with HumanActComponent(reader).
# Transport thread:
request = reader.wait_for_request(timeout=30)
if request is not None:
  reader.submit(request.request_id, 'my answer')
```
"""

import threading

from concordia.typing import human_input


class InputClosedError(RuntimeError):
  """Raised in the simulation thread when the reader is closed."""


class BlockingInput:
  """A thread-safe `HumanInput` reader answered from another thread."""

  def __init__(self):
    self._condition = threading.Condition()
    self._pending: human_input.HumanInputRequest | None = None
    self._answer: str | None = None
    self._closed = False
    self.answered = 0

  def __call__(self, request: human_input.HumanInputRequest) -> str:
    with self._condition:
      if self._closed:
        raise InputClosedError('The input was closed.')
      self._pending = request
      self._answer = None
      self._condition.notify_all()
      while self._answer is None and not self._closed:
        self._condition.wait()
      if self._answer is None:
        self._pending = None
        raise InputClosedError('The input was closed before an answer.')
      answer, self._answer = self._answer, None
      self._pending = None
      self.answered += 1
      self._condition.notify_all()
      return answer

  def _waiting(self) -> human_input.HumanInputRequest | None:
    if self._pending is not None and self._answer is None:
      return self._pending
    return None

  def pending(self) -> human_input.HumanInputRequest | None:
    """The request waiting for an answer (not yet answered), if any."""
    with self._condition:
      return self._waiting()

  def wait_for_request(
      self, timeout: float | None = None
  ) -> human_input.HumanInputRequest | None:
    """Blocks until a request is pending (or the timeout, or close)."""
    with self._condition:
      self._condition.wait_for(
          lambda: self._waiting() is not None or self._closed, timeout=timeout
      )
      return self._waiting()

  def submit(self, request_id: str, answer: str) -> bool:
    """Answers the pending request; False if `request_id` is not pending."""
    with self._condition:
      waiting = self._waiting()
      if self._closed or waiting is None or waiting.request_id != request_id:
        return False
      self._answer = answer
      self._condition.notify_all()
      return True

  def close(self) -> None:
    """Releases a waiting simulation thread; later calls raise."""
    with self._condition:
      self._closed = True
      self._condition.notify_all()

  @property
  def closed(self) -> bool:
    with self._condition:
      return self._closed
