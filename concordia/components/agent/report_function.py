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


"""This components report what the function returns at the moment.

For example, can be used for reporting current time
current_time_component = ReportFunction(
    'Current time',
    function=clock.current_time_interval_str)
"""

from typing import Callable
from concordia.components.agent import action_spec_ignored
from concordia.typing import entity_component

DEFAULT_PRE_ACT_LABEL = 'Report'


class ReportFunction(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A component that reports what the function returns at the moment."""

  def __init__(
      self,
      function: Callable[[], str],
      *,
      pre_act_label: str = DEFAULT_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      function: the function that returns a string to report as state of the
        component.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
    """
    super().__init__(pre_act_label)
    self._function = function

  def _make_pre_act_value(self) -> str:
    """Returns state of this component obtained by calling a function."""
    value = self._function()
    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': value,
    })
    return value

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to JSON data."""
    with self._lock:
      return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the component state from JSON data."""
    pass
