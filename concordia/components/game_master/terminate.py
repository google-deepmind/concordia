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

"""Component that helps a game master terminate the simulation."""

from concordia.components.game_master import scene_tracker as scene_tracker_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_TERMINATE_COMPONENT_KEY = '__terminate__'
DEFAULT_TERMINATE_PRE_ACT_LABEL = '\nTerminate'


class Terminate(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component with a function to decides whether to terminate the simulation.
  """

  def __init__(
      self,
      pre_act_label: str = DEFAULT_TERMINATE_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
    """
    super().__init__()
    self._pre_act_label = pre_act_label
    self._terminate_now = False

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      return 'Yes' if self._terminate_now else 'No'

    return result

  def terminate(self):
    self._terminate_now = True

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'terminate_now': self._terminate_now,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._terminate_now = state['terminate_now']


class NeverTerminate(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that never terminates the simulation.
  """

  def __init__(
      self,
      pre_act_label: str = DEFAULT_TERMINATE_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
    """
    super().__init__()
    self._pre_act_label = pre_act_label

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      return 'No'

    return result

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    pass


class SceneBasedTerminator(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Terminates the simulation when the SceneTracker is done."""

  def __init__(self, scene_tracker_component_key: str):
    """Initializes the terminator.

    Args:
      scene_tracker_component_key: The component key for the scene tracker.
    """
    super().__init__()
    self._scene_tracker_component_key = scene_tracker_component_key

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    """Returns 'Yes' if the termination scene is done, else 'No'."""
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:

      scene_tracker = self.get_entity().get_component(
          self._scene_tracker_component_key,
          type_=scene_tracker_lib.SceneTracker,
      )
      if scene_tracker.is_done():
        return 'Yes'
    return 'No'

  def terminate(self) -> None:
    """Terminates the component."""
    pass

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    pass
