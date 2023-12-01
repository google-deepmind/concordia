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


"""This components report what the get_state returns at the moment.

For example, can be used for reporting current time 
current_time_component = ReportState(
    'Current time', 
    get_state=clock.current_time_interval_str)
    
"""

from typing import Callable
from concordia.typing import component


class ReportState(component.Component):
  """A component that shows the current time interval."""

  def __init__(self, get_state: Callable[[], str], name: str = 'State'):
    """Initializes the component.

    Args:
      get_state: the game clock.
      name: The name of the component.
    """
    self._get_state = get_state
    self._name = name

  def name(self) -> str:
    """Returns the name of the component."""
    return self._name

  def state(self) -> str:
    """Returns the state of the component."""
    return self._get_state()

  def update(self) -> None:
    """This component always returns the same string, update does nothing."""
    pass
