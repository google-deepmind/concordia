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

"""`HumanInput` readers that answer without a person: a policy or a script.

`HumanActComponent` hands its reader exactly what a person controlling the
entity would see: the ordered context and the action spec. A reader that
answers from code therefore exercises the same path a person uses, which is
useful for:

- **bots in a human seat**: skilled, random or passive policies that measure
  whether a game rewards skill, playing through the person's interface;
- **tests** of anything built on `HumanActComponent` (servers, prefabs,
  game masters) without a terminal or a browser;
- **replays** of a person's recorded answers.

Example:

```python
from concordia.components.agent import human_act_component
from concordia.contrib.components.agent import scripted_input

reader = scripted_input.ScriptedInput(['cooperate', 'defect'])
act = human_act_component.HumanActComponent(reader)
```
"""

from collections.abc import Callable, Iterable
import threading

from concordia.typing import human_input


class PolicyInput:
  """A reader that answers each request with `policy(request)`.

  Every request and answer is kept (`requests`, `answers`), so a test or an
  audit can inspect what the seat was shown and what it chose.
  """

  def __init__(self, policy: Callable[[human_input.HumanInputRequest], str]):
    """Initializes the reader.

    Args:
      policy: Maps a request to an answer. It sees the same context and action
        spec a person would.
    """
    self._policy = policy
    self._lock = threading.Lock()
    self.requests: list[human_input.HumanInputRequest] = []
    self.answers: list[str] = []

  def __call__(self, request: human_input.HumanInputRequest) -> str:
    answer = self._policy(request)
    with self._lock:
      self.requests.append(request)
      self.answers.append(answer)
    return answer


class ScriptedInput(PolicyInput):
  """A reader that answers with a fixed sequence of answers, in order.

  A request that `HumanActComponent` retries after a validation error
  (same `request_id`, with `error` set) consumes the next answer too, so a
  script can include a correction. When the script runs out, the reader
  raises `IndexError`: `HumanInput` readers raise rather than invent an
  action.
  """

  def __init__(self, answers: Iterable[str]):
    """Initializes the reader.

    Args:
      answers: The answers to give, in order.
    """
    self._script = list(answers)
    self._next = 0
    super().__init__(self._take)

  def _take(self, request: human_input.HumanInputRequest) -> str:
    del request
    if self._next >= len(self._script):
      raise IndexError(
          f'Scripted input exhausted after {len(self._script)} answers.'
      )
    answer = self._script[self._next]
    self._next += 1
    return answer

  @property
  def remaining(self) -> int:
    """How many scripted answers are left."""
    return len(self._script) - self._next
