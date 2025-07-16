# Copyright 2025 DeepMind Technologies Limited.
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

"""Component helping a game master ask a questionnaire."""

from collections.abc import Sequence
from typing import Dict

from concordia.components.game_master import event_resolution
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

PUTATIVE_EVENT_TAG = event_resolution.PUTATIVE_EVENT_TAG

_TERMINATE_SIGNAL = 'Yes'


class Script(entity_component.ContextComponent):
  """A component runs a script."""

  def __init__(
      self,
      script: Sequence[Dict[str, str]],
      pre_act_label: str = 'Current Question',
  ):
    """Initializes the component.

    Args:
      script: A list of dictionaries. Each dictionary defines a step in the
        script. Each dictionary should be structured as follows: { "name": str,
        # Name of the entity to which the line is associated "line": str,  # The
        line of the script }
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
    """
    super().__init__()
    self._pre_act_label = pre_act_label
    self._script = script
    self._script_idx = 0

  def is_done(self) -> bool:
    return self._script_idx >= len(self._script)

  def get_current_line(self) -> dict[str, str] | None:
    if self._script_idx >= len(self._script):
      return None

    line = self._script[self._script_idx]
    return line

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Prepares the action for the actor based on the current questionnaire state.

    This method checks the current state of the questionnaire and generates
    the appropriate action spec or a formatted string for the actor.

    Args:
        action_spec: The action specification indicating the desired output type

    Returns:
        str:
            - If action_spec.output_type is TERMINATE:
              - Returns _TERMINATE_SIGNAL if the script is done.
              - Returns an empty string if the script is not done.
            - If action_spec.output_type is MAKE_OBSERVATION:
              - Returns a formatted string containing the previous line.
            - If action_spec.output_type is RESOLVE:
              - Returns a formatted string containing the current line.
            - If action_spec.output_type is NEXT_ACTION_SPEC:
              - Returns a free action spec.
            - Otherwise, returns an empty string.
    """
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      if self.is_done():
        return _TERMINATE_SIGNAL
      else:
        return ''

    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      if self._script_idx == 0 or self._script_idx >= len(self._script):
        return ''
      else:
        previous_line = self._script[self._script_idx - 1]
        return f'{previous_line["name"]} : {previous_line["line"]}'

    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      current_line = self.get_current_line()
      if current_line is None:
        return ''
      self._script_idx += 1
      return f'{current_line["name"]} : {current_line["line"]}'

    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      entity_name = self.get_current_line()['name']
      call_to_speech = entity_lib.DEFAULT_CALL_TO_SPEECH.format(
          name=entity_name
      )
      return f'prompt: "{call_to_speech}";;type: free'

    return ''

  def pre_observe(self, observation: str) -> str:
    """Stores the observation for later use in post_observe."""
    self._last_observation = observation
    return ''

  def post_observe(self) -> str:
    return ''

  def reset(self) -> None:
    """Resets the component to its initial state."""
    self._script_idx = 0

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'script': self._script,
        'script_idx': self._script_idx,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._script = state['script']
    self._script_idx = state['questionnaire_idx']
