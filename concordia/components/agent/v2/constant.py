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

"""A simple acting component that aggregates contexts from components."""

from concordia.components.agent.v2 import action_spec_ignored
from concordia.typing import entity as entity_lib

DEFAULT_PRE_ACT_LABEL = 'Constant'


class Constant(action_spec_ignored.ActionSpecIgnored):
  """A simple component that returns a constant.
  """

  def __init__(
      self,
      state: str,
      pre_act_label: str = DEFAULT_PRE_ACT_LABEL,
  ):
    """Initializes the agent.

    Args:
      state: the state of the component.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    self._state = state
    self._pre_act_label = pre_act_label

  def _make_pre_act_context(self) -> str:
    return self._state

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    context = super().pre_act(action_spec)
    return  f'{self._pre_act_label}: {context}'

  def get_last_log(self):
    return {'State': self._state}
