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
from concordia.typing.deprecated import component


class ReportFunction(component.Component):
  """A component that reports what the function returns at the moment."""

  def __init__(self, function: Callable[[], str], name: str = 'State'):
    """Initializes the component.

    Args:
      function: the function that returns a string to report as state of the
        component.
      name: The name of the component.
    """
    self._function = function
    self._name = name

  def name(self) -> str:
    """Returns the name of the component."""
    return self._name

  def state(self) -> str:
    """Returns the state of the component."""
    return self._function()

  def update(self) -> None:
    """This component always returns the same string, update does nothing."""
    pass
