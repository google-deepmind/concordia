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

"""A person's situation now, as the World last told it.

The World's yearly status line (the year, the person's age and land, its
stores, who is near, the season) arrives among the year's other news, and
the stock observation component shows it in the history of all a person
saw. `Now` keeps only the latest status and the person's prefab shows it
right after who the person is, so a small mind reads its situation first:
small live minds were seen to starve on spent land when the better land
they were told of lay buried in the history.
"""

import re
import threading
from typing import override

from concordia.components.agent import action_spec_ignored
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

# The World's status line for a person (`World.status`).
STATUS = re.compile(r'^Year \d+\. .+? is \d+ years old .*$', re.MULTILINE)
NOW = 'Now'


class Now(action_spec_ignored.ActionSpecIgnored):
  """The latest status the World told the person, shown as its situation."""

  def __init__(self, pre_act_label: str = NOW, shown: bool = True):
    """Initializes the component.

    Args:
      pre_act_label: Its label in the person's prompt.
      shown: Whether it adds anything to the prompt. A person not shown its
        situation first still keeps the component, so that any act
        component order that names it (a save's) finds it.
    """
    super().__init__(pre_act_label)
    self._shown = shown
    self._latest = ''
    self._now_lock = threading.Lock()

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    return super().pre_act(action_spec) if self._shown else ''

  @override
  def pre_observe(self, observation: str) -> str:
    found = STATUS.findall(observation)
    if found:
      with self._now_lock:
        self._latest = found[-1]
    return ''

  @override
  def _make_pre_act_value(self) -> str:
    with self._now_lock:
      return self._latest

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._now_lock:
      return {'latest': self._latest}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._now_lock:
      self._latest = str(state.get('latest', ''))
