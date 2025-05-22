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


"""This component always returns the same string."""

from concordia.typing.deprecated import component


class ConstantComponent(component.Component):
  """A constant memory component."""

  def __init__(self, state: str, name: str = 'constant'):
    """Initializes the constant component.

    Args:
      state: The state of the memory component.
      name: The name of the memory component.
    """
    self._state = state
    self._name = name

  def name(self) -> str:
    """Returns the name of the memory component."""
    return self._name

  def state(self) -> str:
    """Returns the state of the memory component."""
    return self._state

  def update(self) -> None:
    """This component always returns the same string, update does nothing."""
    pass

  def set_state(self, state: str) -> None:
    """Set the constant state."""
    self._state = state
