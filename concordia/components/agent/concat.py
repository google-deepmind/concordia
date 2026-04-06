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

"""Concatenate pre-act values from multiple components."""

from collections.abc import Sequence

from concordia.components.agent import action_spec_ignored
from concordia.typing import entity_component


class Concatenate(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """Concatenate pre-act values from multiple components."""

  def __init__(
      self,
      components: Sequence[str] = (),
      pre_act_label: str = 'Context',
  ):
    """Initialize a component to concatenate pre-act values of specified components.

    Args:
      components: Keys of the components to concatenate.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
    """
    super().__init__(pre_act_label)
    self._components = tuple(components)

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'{self.get_component_pre_act_label(key)}:\n'
        f'{self.get_named_component_pre_act_value(key)}')

  def _make_pre_act_value(self) -> str:
    result = '\n'.join(
        [self._component_pre_act_display(key) for key in self._components]
    )
    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': result,
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    with self._lock:
      return {
          'components': tuple(self._components),
          'pre_act_label': self.get_pre_act_label(),
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    with self._lock:
      if 'components' in state:
        self._components = tuple(state['components'])  # type: ignore
