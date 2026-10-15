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

"""Who sits in the player's seat: a person, or a bot in a person's place.

Both act through Concordia's `HumanActComponent`, which hands its reader the
same context and action spec a person would see. A bot is just a reader that
answers from a policy, so bots and people play through one path.
"""

import random
import sys
from typing import Any

from concordia.typing import human_input

from examples.crucible.toys import Policy


class PolicyReader:
  """A `HumanInput` reader that answers with a bot policy."""

  def __init__(self, policy: Policy, toy: Any, seed: int = 0):
    self._policy = policy
    self._toy = toy
    self._rng = random.Random(seed)
    self.answers: list[str] = []

  def __call__(self, request: human_input.HumanInputRequest) -> str:
    answer = self._policy(self._toy, request.entity_name, self._rng)
    self.answers.append(answer)
    return answer


class TerminalReader:
  """A `HumanInput` reader for a person at a terminal."""

  def __init__(self, toy: Any):
    self._toy = toy

  def __call__(self, request: human_input.HumanInputRequest) -> str:
    out = sys.__stdout__
    print('\n' + '=' * 72, file=out)
    print(request.context, file=out)
    print('-' * 72, file=out)
    print(request.action_spec.call_to_action, file=out)
    if request.error:
      print(f'(Not accepted: {request.error})', file=out)
    while True:
      answer = input('> ').strip()
      if answer.startswith('?'):
        draft = answer[1:].strip()
        preview = self._toy.preview(request.entity_name, draft)
        print('Preview:', preview.summary, file=out)
        continue
      return answer
