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

from concordia.components.agent import action_spec_ignored
from concordia.typing import entity_component

DEFAULT_PRE_ACT_LABEL = 'Constant'


class Constant(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A simple component that returns a constant."""

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
    super().__init__(pre_act_label)
    self._state = state

  def _make_pre_act_value(self) -> str:
    self._logging_channel(
        {'Key': self.get_pre_act_label(), 'Value': self._state})
    return self._state

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    pass
