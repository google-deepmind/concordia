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

"""A component for computing and delivering payoffs in a coordination game."""

from collections.abc import Callable, Mapping, Sequence
import copy
import datetime
from typing import Protocol

from concordia.agents.deprecated import deprecated_agent
from concordia.agents.deprecated import entity_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.components.game_master.deprecated import current_scene
from concordia.language_model import language_model
from concordia.typing.deprecated import component
import numpy as np
import termcolor


class OutcomeSummarizationFn(Protocol):
  """Protocol for outcome summarization function."""

  def __call__(
      self,
      joint_action: Mapping[str, str],
      rewards: Mapping[str, float],
      relational_matrix: Mapping[str, Mapping[str, float]],
      player_multipliers: Mapping[str, Mapping[str, float]],
      option_multipliers: Mapping[str, float],
  ) -> Mapping[str, str]:
    """Function of joint actions, rewards, relational matrix and player multipliers which returns an outcome description message for each player.

    Args:
      joint_action: A mapping from player name to their chosen action.
      rewards: A mapping from player name to their reward.
      relational_matrix: A matrix of relationships between players. The entry
        [i][j] specifies the value for player i of making the same choice as
        player j. Matrix is not assumed to be symmetric or having a particular
        value on the diagonal. If `None`, all players are assumed to have value
        of 1, including self relationships (diagonal).
      player_multipliers: A mapping from player name to a mapping from action to
        their multiplier.
      option_multipliers: A mapping from option to their multiplier.

    Returns:
      A mapping from player name to their outcome description message.
    """
    ...


class CoordinationPayoffs(component.Component):
  """Define payoffs for coordination games.

  The players reward is proportional to the number of players who choose the
  same option as them, multiplied by the option's multiplier and player
  multiplier, divided by the number of players.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      option_multipliers: Mapping[str, float],
      player_multipliers: Mapping[str, Mapping[str, float]],
      resolution_scene: str,
      players: Sequence[deprecated_agent.BasicAgent | entity_agent.EntityAgent],
      acting_player_names: Sequence[str],
      outcome_summarization_fn: OutcomeSummarizationFn,
      clock_now: Callable[[], datetime.datetime],
      relational_matrix: Mapping[str, Mapping[str, float]] | None = None,
      name: str = 'scoring function',
      verbose: bool = False,
  ):
    """Initialize a scoring function component.

    Args:
      model: a language model
      memory: an associative memory
      option_multipliers: per option multipliers of rewards
      player_multipliers: per player multipliers of rewards
      resolution_scene: on which scene type should this component be updated
        after the event, i.e. when to check the joint action and compute results
      players: sequence of agents (a superset of the active players)
      acting_player_names: sequence of names of players who act each stage
      outcome_summarization_fn: Function of joint actions, rewards, relational
        matrix and player multipliers which returns an outcome description
        message for each player
      clock_now: Function to call to get current time.
      relational_matrix: a matrix of relationships between players. The entry
        [i][j] specifies the value for player i of making the same choice as
        player j. Matrix is not assumed to be symmetric or having a particular
        value on the diagonal. If `None`, all players are assumed to have value
        of 1, including self relationships (diagonal).
      name: name of this component e.g. Possessions, Account, Property, etc
      verbose: whether to print the full update chain of thought or not
    """
    self._model = model
    self._memory = memory

    self._option_multipliers = option_multipliers
    self._player_multipliers = player_multipliers

    self._players = players
    self._acting_player_names = acting_player_names
    self._outcome_summarization_fn = outcome_summarization_fn
    self._clock_now = clock_now
    self._name = name
    self._verbose = verbose

    self._history = []
    self._state = ''
    self._partial_states = {player.name: '' for player in self._players}
    self._player_scores = {player.name: 0 for player in self._players}

    if relational_matrix is None:
      self._relational_matrix = {
          name: {name_b: 1.0 for name_b in self._acting_player_names}
          for name in self._acting_player_names
      }
      for name in self._acting_player_names:
        self._relational_matrix[name][name] = 0.0
    else:
      if len(relational_matrix) != len(self._acting_player_names):
        raise ValueError(
            'Relational matrix must have the same length as the number of'
            ' acting players.'
        )
      for _, row in relational_matrix.items():
        if len(row) != len(self._acting_player_names):
          raise ValueError(
              'Relational matrix rows must have the same length as the number'
              ' of acting players.'
          )
      self._relational_matrix = copy.deepcopy(relational_matrix)

    self._resolution_scene = resolution_scene
    self._current_scene = current_scene.CurrentScene(
        name='current scene type',
        memory=self._memory,
        clock_now=self._clock_now,
        verbose=self._verbose,
    )

    self.reset()
    # Set the initial state's string representation.
    self.update()

  def reset(self) -> None:
    self._stage_idx = 0
    # Map each player's name to their component of the joint action.
    self._partial_joint_action = {
        name: None for name in self._acting_player_names
    }

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

  def _count_string_occurrences(self, target_string, dictionary):
    count = 0
    for value in dictionary.values():
      if value == target_string:
        count += 1
    return count

  def _get_rewards_from_joint_action(
      self, joint_action: Mapping[str, str]
  ) -> Mapping[str, float]:

    rewards = {}
    for player in self._players:
      player_action = joint_action[player.name]
      same_choice_by_relation = 0
      for other_player in self._players:
        if player_action == joint_action[other_player.name]:
          same_choice_by_relation += self._relational_matrix[player.name][
              other_player.name
          ]
      player_preference = self._player_multipliers[player.name][player_action]
      option_multiplier = self._option_multipliers[player_action]
      max_reward_possible = sum(
          max(0, r) for r in self._relational_matrix[player.name].values()
      )
      rewards[player.name] = (
          same_choice_by_relation
          * player_preference
          * option_multiplier
          / max_reward_possible
      )
    return rewards

  def _set_outcome_messages(
      self,
      joint_action: Mapping[str, str],
      rewards: Mapping[str, float],
  ) -> None:
    # Only the game master sees the actual reward values.
    game_master_private_state = '\n'.join([
        f'{player.name}: {self._player_scores[player.name]}'
        for player in self._players
    ])
    # Players see a text-based summarization of the events, which may or may not
    # include the actual reward values.
    partial_states = self._outcome_summarization_fn(
        joint_action=joint_action,
        rewards=rewards,
        relational_matrix=self._relational_matrix,
        player_multipliers=self._player_multipliers,
        option_multipliers=self._option_multipliers,
    )
    common_view_of_player_obs = '\n'.join([
        f'{name} observed: {observation}'
        for name, observation in partial_states.items()
    ])

    # State is only observed by the game master since players get
    # their observations from `partial_states`.
    self._state = f'{common_view_of_player_obs}\n{game_master_private_state}'

    # The game master gets a memory of the state.
    self._memory.add(self._state)
    # Active players observe their own partial state description and inactive
    # players get the common description.
    for player in self._players:
      if player.name in self._acting_player_names:
        player.observe(partial_states[player.name])
      else:
        player.observe(common_view_of_player_obs)

  def update_before_event(self, player_action_attempt: str) -> None:
    # `player_action_attempt` is formatted as "name: attempt".
    player_name, choice_str = player_action_attempt.split(': ')
    self._partial_joint_action[player_name] = choice_str
    self._state = ''

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
      if self._joint_action_is_complete(joint_action):  # pytype: disable=wrong-arg-types
        # Map the joint action to rewards per player.
        rewards = self._get_rewards_from_joint_action(joint_action)  # pytype: disable=wrong-arg-types

        # Accumulate the rewards per player.
        for name in self._acting_player_names:
          self._player_scores[name] += rewards[name]

        # Use the outcome summarization function to get the state.
        self._set_outcome_messages(joint_action, rewards)  # pytype: disable=wrong-arg-types
        self._memory.extend([
            self.state(),
        ])

        joint_action_for_log = str(self._partial_joint_action)
        payoffs_for_log = self.state()
        finished = True

        if self._verbose:
          print(termcolor.colored(self.state(), 'yellow'))

    num_players_already_acted = np.sum(
        [value is not None for value in self._partial_joint_action.values()]
    )
    total_num_players_to_act = len(self._partial_joint_action)
    update_log = {
        'date': self._clock_now(),
        'Summary': self.name(),
        'Stage index': self._stage_idx,
        'How many players acted so far this stage': (
            f'{num_players_already_acted}/{total_num_players_to_act}'
        ),
        'Payoffs': payoffs_for_log,
        'Joint action': joint_action_for_log,
    }
    self._history.append(update_log)

    if finished:
      # Advance to the next stage.
      self._stage_idx += 1
      self._partial_joint_action = {
          name: None for name in self._acting_player_names
      }

  def get_scores(self) -> Mapping[str, float]:
    """Return the cumulative score for each player."""
    return self._player_scores
