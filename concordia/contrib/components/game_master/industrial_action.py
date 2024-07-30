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

from concordia.agents import basic_agent
from concordia.agents import entity_agent
from concordia.associative_memory import associative_memory
from concordia.components.game_master import current_scene
from concordia.language_model import language_model
from concordia.typing import component
import numpy as np

import termcolor


# A function that maps number of cooperators to a scalar.
CollectiveActionProductionFunction = Callable[[int], float]
PlayersT = Sequence[basic_agent.BasicAgent | entity_agent.EntityAgent]


def _get_pressure_str(pressure: float, pressure_threshold: float) -> str:
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


class LaborStrike(component.Component):
  """A component for computing pressure created through labor collective action.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      cooperative_option: str,
      resolution_scene: str,
      production_function: CollectiveActionProductionFunction,
      players: Sequence[basic_agent.BasicAgent | entity_agent.EntityAgent],
      acting_player_names: Sequence[str],
      players_to_inform: Sequence[str],
      clock_now: Callable[[], datetime.datetime],
      pressure_threshold: float,
      name: str = 'pressure from industrial action',
      verbose: bool = False,
  ):
    """Initialize a component for computing pressure created by a labor strike.

    Args:
      model: a language model
      memory: an associative memory
      cooperative_option: which option choice constitutes cooperation
      resolution_scene: on which scene type should this component be updated
        after the event, i.e. when to check the joint action and compute results
      production_function: pressure produced a function of he number of
        individuals joining the strike
      players: sequence of agents (a superset of the active players)
      acting_player_names: sequence of names of players who act each stage. In
        a labor strike scenario these players would typically be the workers.
        They have to decide whether to join the strike or not.
      players_to_inform: names of players who observe the amount of pressure. In
        a labor strike scenario these players would typically repressent the
        management of the company.
      clock_now: Function to call to get current time.
      pressure_threshold: the threshold above which the boss will feel compelled
        to take action.
      name: name of this component e.g. Possessions, Account, Property, etc
      verbose: whether to print the full update chain of thought or not
    """
    self._model = model
    self._memory = memory
    self._cooperative_option = cooperative_option
    self._production_function = production_function
    self._players = players
    self._acting_player_names = acting_player_names
    self._players_to_inform = players_to_inform
    self._clock_now = clock_now
    self._name = name
    self._pressure_threshold = pressure_threshold
    self._verbose = verbose

    self._history = []
    self._state = ''
    self._partial_states = {player.name: '' for player in self._players}
    self._player_scores = {player.name: 0 for player in self._players}

    self._resolution_scene = resolution_scene
    self._current_scene = current_scene.CurrentScene(
        name='current scene type',
        memory=self._memory,
        clock_now=self._clock_now,
        verbose=self._verbose,
    )

    self._map_names_to_players = {
        player.name: player for player in self._players}

    self.reset()
    # Set the initial state's string representation.
    self.update()

  def reset(self) -> None:
    self._stage_idx = 0
    # Map each player's name to their component of the joint action.
    self._partial_joint_action = {
        name: None for name in self._acting_player_names}

  def name(self) -> str:
    """Returns the name of this component."""
    return self._name

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def get_history(self):
    return self._history.copy()

  def state(self) -> str:
    return self._state

  def partial_state(
      self,
      player_name: str,
  ) -> str:
    """Return a player-specific view of the component's state."""
    return self._partial_states[player_name]

  def update(self) -> None:
    self._current_scene.update()

  def _joint_action_is_complete(self, joint_action: Mapping[str, str]) -> bool:
    for acting_player_name in self._acting_player_names:
      if joint_action[acting_player_name] is None:
        return False
    return True

  def _binarize_joint_action(
      self,
      joint_action: Mapping[str, str]) -> Mapping[str, bool]:
    binary_joint_action = {name: act == self._cooperative_option
                           for name, act in joint_action.items()}
    return binary_joint_action

  def _get_pressure_from_joint_action(
      self, binary_joint_action: Mapping[str, bool]) -> float:
    num_cooperators = np.sum(list(binary_joint_action.values()))
    return self._production_function(num_cooperators)

  def update_before_event(self, player_action_attempt: str) -> None:
    # `player_action_attempt` is formatted as "name: attempt".
    current_scene_type = self._current_scene.state()
    if current_scene_type == self._resolution_scene:
      player_name, choice_str = player_action_attempt.split(': ')
      if player_name in self._acting_player_names:
        self._partial_joint_action[player_name] = choice_str

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    current_scene_type = self._current_scene.state()
    payoffs_for_log = ''
    joint_action_for_log = ''
    finished = False
    if current_scene_type == self._resolution_scene:
      # Check if all players have acted so far in the current stage game.
      joint_action = self._partial_joint_action.copy()
      if self._joint_action_is_complete(joint_action):
        # Map the joint action to an amount of pressure produced.
        binary_joint_action = self._binarize_joint_action(joint_action)
        pressure = self._get_pressure_from_joint_action(binary_joint_action)
        pressure_str = _get_pressure_str(pressure, self._pressure_threshold)
        for player_name in self._players_to_inform:
          self._partial_states[player_name] = pressure_str
          self._map_names_to_players[player_name].observe(pressure_str)

        joint_action_for_log = str(self._partial_joint_action)
        finished = True

        if self._verbose:
          print(termcolor.colored(self.state(), 'yellow'))

    num_players_already_acted = np.sum(
        [value is not None for value in self._partial_joint_action.values()])
    total_num_players_to_act = len(self._acting_player_names)
    update_log = {
        'date': self._clock_now(),
        'Summary': self.name(),
        'Stage index': self._stage_idx,
        'How many players acted so far this stage': (
            f'{num_players_already_acted}/{total_num_players_to_act}'),
        'Schelling diagram payoffs': payoffs_for_log,
        'Joint action': joint_action_for_log,
    }
    self._history.append(update_log)

    if finished:
      # Advance to the next stage.
      self._stage_idx += 1
      self._partial_joint_action = {
          name: None for name in self._acting_player_names}

  def get_scores(self) -> Mapping[str, float]:
    """Return the cumulative score for each player."""
    return self._player_scores
