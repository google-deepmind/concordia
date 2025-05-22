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

"""A component for computing and delivering payoffs in a bargaining game."""

from collections.abc import Callable, Mapping, Sequence
import datetime
import re

from concordia.agents.deprecated import entity_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.components.game_master.deprecated import current_scene
from concordia.language_model import language_model
from concordia.typing.deprecated import component
import numpy as np
import termcolor


class MultiItemBargainPayoffs(component.Component):
  """Define payoffs for bargaining game with multiple items.

  The game has only two players. One player suggest the price and an item, the
  other one can accept or reject. The sellers reward equals to the proposed
  value, if the offer is accepted. Otherwise, seller gets zero reward. Buyers
  reward is a constant value minus the price, if the offer is accepted.
  Otherwise, buyer gets zero reward.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      buyer_base_reward_per_item: Mapping[str, float],
      seller_base_reward_per_item: Mapping[str, float],
      action_to_reward: Mapping[str, float],
      multi_action_formatting_string: str,
      buyer: entity_agent.EntityAgent,
      seller: entity_agent.EntityAgent,
      resolution_scene: str,
      acting_player_names: Sequence[str],
      outcome_summarization_fn: Callable[
          [Mapping[str, str], Mapping[str, float]], Mapping[str, str]
      ],
      clock_now: Callable[[], datetime.datetime],
      name: str = 'scoring function',
      verbose: bool = False,
  ):
    """Initialize a scoring function component.

    Args:
      model: a language model
      memory: an associative memory
      buyer_base_reward_per_item: the base reward for the buyer per item
      seller_base_reward_per_item: the base reward for the seller per item
      action_to_reward: a mapping from action to reward
      multi_action_formatting_string: a format string for a multi-item action,
        where {item} and {price} are replaced by the item and price strings.
      buyer: the buyer agent
      seller: the seller agent
      resolution_scene: on which scene type should this component be updated
        after the event, i.e. when to check the joint action and compute results
      acting_player_names: sequence of names of players who act each stage
      outcome_summarization_fn: function of joint actions and rewards which
        returns an outcome description message for each player
      clock_now: Function to call to get current time.
      name: name of this component e.g. Possessions, Account, Property, etc
      verbose: whether to print the full update chain of thought or not
    """

    self._model = model
    self._memory = memory

    self._players = [buyer, seller]
    self._buyer_base_reward_per_item = buyer_base_reward_per_item
    self._seller_base_reward_per_item = seller_base_reward_per_item

    # check that they have the same keys
    if set(buyer_base_reward_per_item.keys()) != set(
        seller_base_reward_per_item.keys()
    ):
      raise ValueError(
          'buyer_base_reward_per_item and seller_base_reward_per_item must have'
          ' the same keys.'
      )

    self._acting_player_names = acting_player_names
    self._action_to_reward = action_to_reward
    self._outcome_summarization_fn = outcome_summarization_fn
    self._clock_now = clock_now
    self._name = name
    self._verbose = verbose
    self._buyer = buyer
    self._seller = seller
    self._multi_action_formatting_string = multi_action_formatting_string

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

    self.reset()
    # Set the initial state's string representation.
    self.update()

  def _extract_variables(self, realization) -> Mapping[str, str]:
    """Extracts the values of variables from a string realization based on a format string.

    Args:
      realization: The concrete string with actual values in place of variables.

    Returns:
      A dictionary mapping variable names to their extracted values.
    """

    # Escape special characters in the format string
    escaped_format = re.escape(self._multi_action_formatting_string)

    pattern = escaped_format.replace(r'\{', r'(?P<').replace(r'\}', r'>.*)')

    # Match the realization against the pattern
    match = re.match(pattern, realization)

    if match:
      # Extract the values from the named capture groups
      return match.groupdict()
    else:
      return {}

  def _extract_offer_details(self, offer_dict):
    """Extracts the acceptance status (True/False) and the proposal string from a dictionary.

    Args:
        offer_dict: A dictionary where keys are player names and values are
          either 'accept'/'reject' or item and price string.

    Returns:
        A tuple containing:
            - accepted: True if the offer was accepted, False otherwise.
            - proposal: The proposal string (e.g., the price).
    """

    values = list(offer_dict.values())

    # Check if 'accept' is present, indicating acceptance
    accepted = 'accept' in values

    # Find the proposal (the value that's not 'accept' or 'reject')
    item_and_proposal = next(
        value for value in values if value not in ('accept', 'reject')
    )

    item_and_proposal_dict = self._extract_variables(item_and_proposal)

    item = item_and_proposal_dict['item']
    proposal = item_and_proposal_dict['price']

    return accepted, proposal, item

  def _get_rewards_from_joint_action(
      self, joint_action: Mapping[str, str]
  ) -> Mapping[str, float]:

    rewards = {}

    accepted, proposal, item = self._extract_offer_details(joint_action)

    if accepted:
      rewards[self._buyer.name] = (
          self._buyer_base_reward_per_item[item]
          - self._action_to_reward[proposal]
      )
      rewards[self._seller.name] = (
          self._action_to_reward[proposal]
          - self._seller_base_reward_per_item[item]
      )
    else:
      rewards[self._buyer.name] = 0
      rewards[self._seller.name] = 0

    return rewards

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
    partial_states = self._outcome_summarization_fn(joint_action, rewards)
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

    for player in self._players:
      player.observe(event_statement)

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
