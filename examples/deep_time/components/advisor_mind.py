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

"""The Advisor's mind: the upstream question-of-memories, plus an almanac.

The Advisor thinks slowly (a `pubsub.Publisher`, a few ticks late), so its
briefings are stale; to be worth more than they cost, it knows what the
Steward cannot see: the sun ahead and the eruptions the geologists expect.
The almanac is added to its question only, never to the
Steward's own context.
"""

from collections.abc import Callable
from typing import Any, override

from concordia.components.agent import question_of_recent_memories


class AdvisorMind(question_of_recent_memories.QuestionOfRecentMemories):
  """`QuestionOfRecentMemories` whose question carries a private almanac."""

  def __init__(self, almanac: Callable[[], str], **kwargs: Any):
    """Initializes the mind.

    Args:
      almanac: Returns what the Advisor alone knows, as text.
      **kwargs: Arguments of `QuestionOfRecentMemories`.
    """
    super().__init__(**kwargs)
    self._almanac = almanac
    self._base_question = self._question

  @override
  def _make_pre_act_value(self) -> str:
    known = self._almanac().replace('{', '{{').replace('}', '}}')
    self._question = f'{known} {self._base_question}'
    try:
      return super()._make_pre_act_value()
    finally:
      self._question = self._base_question
