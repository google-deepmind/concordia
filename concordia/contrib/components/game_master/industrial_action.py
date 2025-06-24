# Copyright 2022 DeepMind Technologies Limited.
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

"""A component for computing pressure created through labor collective action.
"""

from collections.abc import Callable, Mapping, Sequence
import datetime

from concordia.agents import entity_agent
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import event_resolution as event_resolution_component
from concordia.components.game_master import switch_act
from concordia.environment.scenes import runner as scene_runner
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging
import numpy as np
import termcolor


# A function that maps number of cooperators to a scalar.
CollectiveActionProductionFunction = Callable[[int], float]
PlayersT = Sequence[entity_agent.EntityAgent]


def get_pressure_str(pressure: float, pressure_threshold: float) -> str:
  """Convert a numerical amount of pressure to a string description."""
  low_level_of_pressure = pressure_threshold / 10.0
  if pressure <= low_level_of_pressure:
    return 'The workers seem mostly content. The project is progressing well.'
  elif pressure > low_level_of_pressure and pressure <= pressure_threshold:
    return ('A significant fraction of workers are on strike. The shareholders '
            'are starting to get worried.')
  elif pressure > pressure_threshold and pressure <= 1.0:
    return ('Most workers joined the strike. The shareholders are furious the '
            'project won\'t be completed on time, and are demanding '
            'immediate action from management to get things back on track.')
  else:
    raise ValueError('Pressure must be between 0 and 1.')


class LaborStrike(entity_component.ContextComponent):
  """A component for computing pressure created through labor collective action.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      cooperative_option: str,
      resolution_scene: str,
      production_function: CollectiveActionProductionFunction,
      all_player_names: Sequence[str],
      acting_player_names: Sequence[str],
      players_to_inform: Sequence[str],
      clock_now: Callable[[], datetime.datetime],
      pressure_threshold: float,
      event_resolution_component_key: str = (
          switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY),
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      pre_act_label: str = '',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
      verbose: bool = False,
  ):
    """Initialize a component for computing pressure created by a labor strike.

    Args:
      model: a language model
      cooperative_option: which option choice constitutes cooperation
      resolution_scene: on which scene type should this component be updated
        after the event, i.e. when to check the joint action and compute results
      production_function: pressure produced a function of he number of
        individuals joining the strike
      all_player_names: sequence of names (a superset of the active players)
      acting_player_names: sequence of names of players who act each stage. In
        a labor strike scenario these players would typically be the workers.
        They have to decide whether to join the strike or not.
      players_to_inform: names of players who observe the amount of pressure. In
        a labor strike scenario these players would typically repressent the
        management of the company.
      clock_now: Function to call to get current time.
      pressure_threshold: the threshold above which the boss will feel compelled
        to take action.
      event_resolution_component_key: The name of the event resolution
        component.
      memory_component_key: The name of the memory component.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to log debug information to.
      verbose: whether to print the full update chain of thought or not
    """
    self._pre_act_label = pre_act_label
    self._logging_channel = logging_channel

    self._model = model
    self._memory_component_key = memory_component_key
    self._event_resolution_component_key = event_resolution_component_key
    self._cooperative_option = cooperative_option
    self._production_function = production_function
    self._all_player_names = all_player_names
    self._acting_player_names = acting_player_names
    self._players_to_inform = players_to_inform
    self._clock_now = clock_now
    self._pressure_threshold = pressure_threshold
    self._resolution_scene = resolution_scene

    self._verbose = verbose

    self._history = []
    self._state = ''
    self._player_scores = {name: 0 for name in self._all_player_names}
    self._latest_action_spec = None

    self.reset()

  def reset(self) -> None:
    self._stage_idx = 0
    # Map each player's name to their component of the joint action.
    self._partial_joint_action = {
        name: None for name in self._acting_player_names}

  def _get_current_scene_type(self) -> str:
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    return scene_runner.get_current_scene_type(memory=memory)

  def _joint_action_is_complete(
      self, joint_action: Mapping[str, str | None]
  ) -> bool:
    for acting_player_name in self._acting_player_names:
      if joint_action[acting_player_name] is None:
        return False
    return True

  def _binarize_joint_action(
      self,
      joint_action: Mapping[str, str | None]) -> Mapping[str, bool]:
    binary_joint_action = {name: act == self._cooperative_option
                           for name, act in joint_action.items()}
    return binary_joint_action

  def _get_pressure_from_joint_action(
      self, binary_joint_action: Mapping[str, bool]) -> float:
    num_cooperators = np.sum(list(binary_joint_action.values()))
    return self._production_function(num_cooperators)

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    self._latest_action_spec = action_spec
    current_scene_type = self._get_current_scene_type()
    if (
        current_scene_type == self._resolution_scene
        and action_spec.output_type == entity_lib.OutputType.RESOLVE
    ):
      event_resolution = self.get_entity().get_component(
          self._event_resolution_component_key,
          type_=event_resolution_component.EventResolution,
      )
      player_name = event_resolution.get_active_entity_name()
      choice = event_resolution.get_putative_action()
      if player_name in self._acting_player_names:
        self._partial_joint_action[player_name] = choice

    self._logging_channel({
        'Key': self._pre_act_label,
        'Value': action_spec,
    })
    return ''

  def post_act(
      self,
      event_statement: str,
  ) -> str:
    current_scene_type = self._get_current_scene_type()
    # joint_action_for_log = ''
    finished = False
    if (
        current_scene_type == self._resolution_scene
        and self._latest_action_spec == entity_lib.OutputType.RESOLVE
    ):
      # Check if all players have acted so far in the current stage game.
      joint_action = self._partial_joint_action.copy()
      if self._joint_action_is_complete(joint_action):
        # Map the joint action to an amount of pressure produced.
        binary_joint_action = self._binarize_joint_action(joint_action)
        pressure = self._get_pressure_from_joint_action(binary_joint_action)
        pressure_str = get_pressure_str(pressure, self._pressure_threshold)

        memory = self.get_entity().get_component(
            self._memory_component_key, type_=memory_component.Memory
        )
        memory.add(pressure_str)
        lowercase_pressure_str = pressure_str[0].lower() + pressure_str[1:]
        for player_name in self._players_to_inform:
          memory.add(f'{player_name} learned that {lowercase_pressure_str}')

        # joint_action_for_log = str(self._partial_joint_action)
        finished = True

        if self._verbose:
          print(termcolor.colored(self._state, 'yellow'))

    if finished:
      # Advance to the next stage.
      self._stage_idx += 1
      self._partial_joint_action = {
          name: None for name in self._acting_player_names}

    return ''

  def get_scores(self) -> Mapping[str, float]:
    """Return the cumulative score for each player."""
    return self._player_scores

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to JSON data."""
    return {'state': self._state}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the component state from JSON data."""
    self._state = state['state']
