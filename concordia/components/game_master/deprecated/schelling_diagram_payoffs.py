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

"""A component for computing and delivering payoffs using a Schelling diagram.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import datetime

from concordia.agents.deprecated import deprecated_agent
from concordia.agents.deprecated import entity_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.components.game_master.deprecated import current_scene
from concordia.language_model import language_model
from concordia.typing.deprecated import component
import numpy as np
import termcolor


# A Schelling Function maps number of cooperators to reward.
SchellingFunction = Callable[[int], float]


# A Schelling diagram consists of two Schelling functions: one describing the
# reward for cooperation and the other describing the reward for defection.
# See: Schelling, T. C. (1973). Hockey helmets, concealed weapons, and daylight
# saving: A study of binary choices with externalities. Journal of Conflict
# resolution, 17(3), 381-428.
@dataclasses.dataclass(frozen=True)
class SchellingDiagram:
  """A Schelling diagram."""
  cooperation: SchellingFunction
  defection: SchellingFunction


class SchellingPayoffs(component.Component):
  """Define payoffs for minigames using a Schelling diagram.

  Schelling diagrams are a game representation described in:

  Schelling, T.C., 1973. Hockey helmets, concealed weapons, and daylight saving:
  A study of binary choices with externalities. Journal of Conflict resolution,
  17(3), pp.381-428.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      cooperative_option: str,
      resolution_scene: str,
      cooperator_reward_fn: SchellingFunction,
      defector_reward_fn: SchellingFunction,
      players: Sequence[deprecated_agent.BasicAgent | entity_agent.EntityAgent],
      acting_player_names: Sequence[str],
      outcome_summarization_fn: Callable[
          [Mapping[str, int],
           Mapping[str, str],
           Mapping[str, float],
           Mapping[str, float]],
          Mapping[str, str],
      ],
      clock_now: Callable[[], datetime.datetime],
      active_players_observe_joint_action_and_outcome: bool = False,
      name: str = 'scoring function',
      verbose: bool = False,
  ):
    """Initialize a scoring function component.

    Args:
      model: a language model
      memory: an associative memory
      cooperative_option: which option choice constitutes cooperation
      resolution_scene: on which scene type should this component be updated
        after the event, i.e. when to check the joint action and compute results
      cooperator_reward_fn: reward obtained by cooperators as a function of
        the number of other cooperators
      defector_reward_fn: reward obtained by defectors as a function of the
        number of other defectors
      players: sequence of agents (a superset of the active players)
      acting_player_names: sequence of names of players who act each stage
      outcome_summarization_fn: function of binarized joint actions and
        rewards which returns an outcome description message for each player
      clock_now: Function to call to get current time.
      active_players_observe_joint_action_and_outcome: False by default, if set
        to True, then active players observe the full joint action and outcome,
        otherwise they observe only their own actions and rewards description.
        Inactive players always observe the full joint action and outcome.
      name: name of this component e.g. Possessions, Account, Property, etc
      verbose: whether to print the full update chain of thought or not
    """
    self._model = model
    self._memory = memory
    self._cooperative_option = cooperative_option
    self._cooperator_reward_fn = cooperator_reward_fn
    self._defector_reward_fn = defector_reward_fn
    self._players = players
    self._acting_player_names = acting_player_names
    self._outcome_summarization_fn = outcome_summarization_fn
    self._clock_now = clock_now
    self._name = name
    self._active_players_observe_joint_action_and_outcome = (
        active_players_observe_joint_action_and_outcome
    )
    self._verbose = verbose

    self._history = []
    self._state = ''
    self._partial_states = {player.name: '' for player in self._players}
    self._player_scores = {player.name: 0 for player in self._players}

    self._map_names_to_players = {
        player.name: player for player in self._players}

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

  def _get_rewards_from_joint_action(
      self, binary_joint_action: Mapping[str, bool]) -> Mapping[str, float]:
    # This scoring function only supports "Schelling style" (binary choice with
    # externalities) type of game representations. This means the critical
    # factor is the number of players picking the cooperate option.
    num_cooperators = np.sum(list(binary_joint_action.values()))

    rewards = {}
    for player_name, is_cooperator in binary_joint_action.items():
      if is_cooperator:
        rewards[player_name] = self._cooperator_reward_fn(num_cooperators)
      else:
        rewards[player_name] = self._defector_reward_fn(num_cooperators)

    return rewards

  def _set_outcome_messages(
      self,
      rewards: Mapping[str, float],
      binary_joint_action: Mapping[str, bool],
      joint_action: Mapping[str, str],
  ) -> None:
    # Only the game master sees the actual reward values.
    game_master_private_state = '\n'.join(
        [f'{player.name}: {self._player_scores[player.name]}'
         for player in self._players])
    # Players see a text-based summarization of the events, which may or may not
    # include the actual reward values.
    partial_states = self._outcome_summarization_fn(
        binary_joint_action, joint_action, rewards, self._player_scores
    )
    common_view_of_player_obs = '\n'.join(
        [f'{observation}' for observation in partial_states.values()]
    )

    # State is only observed by the game master since players get
    # their observations from `partial_states`.
    self._state = f'{common_view_of_player_obs}\n{game_master_private_state}'

    # The game master gets a memory of the state.
    self._memory.add(self._state)
    # By default, active players observe only their own partial state
    # description, but if `active_players_observe_joint_action_and_outcome` is
    # True then they observe the full joint action and outcome. Inactive players
    # always observe the full joint action/outcome.
    for player in self._players:
      if player.name in self._acting_player_names:
        if self._active_players_observe_joint_action_and_outcome:
          self._partial_states[player.name] = common_view_of_player_obs
        else:
          self._partial_states[player.name] = partial_states[player.name]
      else:
        self._partial_states[player.name] = common_view_of_player_obs

  def update_before_event(self, player_action_attempt: str) -> None:
    # `player_action_attempt` is formatted as "name: attempt".
    # we assume that the player action attempt is in the format
    # 'player_name: player_choice'. All other occurrences of ':' will be treated
    # as a part of the player choice.
    player_name, choice_str = player_action_attempt.split(': ', 1)
    if player_name not in self._acting_player_names:
      return
    self._partial_joint_action[player_name] = choice_str
    self._state = ''

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    current_scene_type = self._current_scene.state()
    payoffs_for_log = ''
    joint_action_for_log = ''
    self._partial_states = {player.name: '' for player in self._players}
    finished = False
    if current_scene_type == self._resolution_scene:
      # Check if all players have acted so far in the current stage game.
      joint_action = self._partial_joint_action.copy()
      if self._joint_action_is_complete(joint_action):  # pytype: disable=wrong-arg-types
        # Map the joint action to rewards per player.
        binary_joint_action = self._binarize_joint_action(joint_action)  # pytype: disable=wrong-arg-types
        rewards = self._get_rewards_from_joint_action(binary_joint_action)

        # Accumulate the rewards per player.
        for name in self._acting_player_names:
          self._player_scores[name] += rewards[name]

        # Use the outcome summarization function to get the state.
        self._set_outcome_messages(rewards, binary_joint_action, joint_action)  # pytype: disable=wrong-arg-types
        self._memory.extend([self.state(),])
        for player_name, partial_state in self._partial_states.items():
          if partial_state:
            if isinstance(self._map_names_to_players[player_name],
                          entity_agent.EntityAgent):
              self._map_names_to_players[player_name].observe(partial_state)  # pytype: disable=attribute-error

        joint_action_for_log = str(self._partial_joint_action)
        payoffs_for_log = self.state()
        finished = True

        if self._verbose:
          print(termcolor.colored(self.state(), 'yellow'))

    num_players_already_acted = np.sum(
        [value is not None for value in self._partial_joint_action.values()])
    total_num_players_to_act = len(self._partial_joint_action)
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
