# Copyright 2024 DeepMind Technologies Limited.
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

"""A component for sequential bargaining games with intermediate observations.

Unlike PayoffMatrix which is designed for simultaneous games, this component
sends observations after each player acts, so the second player can see what
the first player proposed before deciding how to respond.
"""

from collections.abc import Callable, Mapping, Sequence
import copy

from concordia.components.agent import memory as memory_component
from concordia.components.game_master import event_resolution as event_resolution_component
from concordia.components.game_master import make_observation as make_observation_component
from concordia.components.game_master import scene_tracker
from concordia.components.game_master import switch_act
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import termcolor


class SequentialBargainPayoff(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component for computing payoffs in sequential bargaining games.

  Key difference from PayoffMatrix: this component sends intermediate
  observations after each player acts (not just after all players act).
  This allows the second player to see what the first player proposed.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      buyer_name: str,
      seller_name: str,
      action_to_scores: Callable[[Mapping[str, str]], Mapping[str, float]],
      scores_to_observation: Callable[[Mapping[str, float]], Mapping[str, str]],
      seller_costs_registry: (
          Mapping[str, float | Mapping[str, float]] | None
      ) = None,
      event_resolution_component_key: str = (
          switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
      ),
      observation_component_key: str | None = (
          make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      memory_component_key: str | None = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      scene_tracker_component_key: str | None = (
          scene_tracker.DEFAULT_SCENE_TRACKER_COMPONENT_KEY
      ),
      pre_act_label: str = '',
      verbose: bool = False,
  ):
    """Initialize a component for sequential bargaining payoffs.

    Args:
      model: a language model
      buyer_name: name of the buyer (first player to act)
      seller_name: name of the seller (second player to act)
      action_to_scores: function that maps a dictionary of actions by players to
        a dictionary of scores for each player
      scores_to_observation: function that maps a dictionary of scores for each
        player to a dictionary of observations for each player.
      seller_costs_registry: mapping from seller names to their costs. Can be
        either a simple float (single item cost) or a Mapping[str, float]
        (per-item costs for multi-item scenarios). Used to generate proposal
        observations with correct per-seller cost info.
      event_resolution_component_key: The key of the event resolution component.
      observation_component_key: The key of the observation component to send
        observations to players. If None, no observations will be sent.
      memory_component_key: The key of the memory component to the observations
        by players. If None, no observations will be added to the memory.
      scene_tracker_component_key: The key of the scene tracker component.
      pre_act_label: Prefix to add to the output of the component.
      verbose: whether to print the full update chain of thought or not
    """
    self._pre_act_label = pre_act_label
    self._model = model
    self._buyer_name = buyer_name
    self._seller_name = seller_name
    self._acting_player_names = [buyer_name, seller_name]
    self._observation_component_key = observation_component_key
    self._memory_component_key = memory_component_key
    self._event_resolution_component_key = event_resolution_component_key
    self._action_to_scores = action_to_scores
    self._scores_to_observation = scores_to_observation
    self._seller_costs_registry = seller_costs_registry or {}
    self._scene_tracker_component_key = scene_tracker_component_key
    self._verbose = verbose

    self._history = []
    self._player_scores = {name: 0.0 for name in self._acting_player_names}
    self._latest_action_spec_output_type = None

    self.reset()

  def reset(self) -> None:
    self._stage_idx = 0
    self._partial_joint_action = {
        name: None for name in self._acting_player_names
    }
    self._buyer_has_proposed = False

  def _get_current_scene_participants(self) -> Sequence[str]:
    if self._scene_tracker_component_key:
      scene_tracker_component = self.get_entity().get_component(
          self._scene_tracker_component_key,
          type_=scene_tracker.SceneTracker,
      )
      return scene_tracker_component.get_participants()
    return self._acting_player_names

  def _get_current_buyer_and_seller(self) -> tuple[str, str]:
    """Get the current buyer and seller from the scene participants.

    By convention, participants are ordered [buyer, seller] in scene config.
    Falls back to initialized names if scene doesn't have exactly 2
    participants.

    Returns:
      A tuple of (buyer_name, seller_name).
    """
    participants = list(self._get_current_scene_participants())
    if len(participants) == 2:
      return participants[0], participants[1]
    return self._buyer_name, self._seller_name

  def _joint_action_is_complete(self, joint_action: Mapping[str, str]) -> bool:
    """Check if all current scene participants have submitted non-None actions."""

    for acting_player_name in self._get_current_scene_participants():
      # Return False if participant is missing OR has None action
      if acting_player_name not in joint_action:
        return False
      if joint_action[acting_player_name] is None:
        return False
    return True

  def _default_proposal_observation(
      self, buyer_name: str, buyer_action: str
  ) -> str:
    """Generate a default observation for the seller about the buyer's proposal."""
    return f'{buyer_name} proposed: {buyer_action}'

  def _format_proposal_observation(
      self,
      buyer_name: str,
      buyer_action: str,
      seller_costs: float | Mapping[str, float] | None,
  ) -> str:
    """Format the proposal observation based on cost type.

    Args:
      buyer_name: Name of the buyer making the proposal.
      buyer_action: The action/proposal the buyer made.
      seller_costs: The seller's costs - can be: - None: No cost info, use
        default format - float: Single item cost (simple haggling) -
        Mapping[str, float]: Per-item costs (multi-item haggling)

    Returns:
      Formatted observation string for the seller.
    """
    if seller_costs is None or seller_costs == 0:
      return self._default_proposal_observation(buyer_name, buyer_action)

    if isinstance(seller_costs, Mapping):
      cost_str = '; '.join(
          [f'{item}: {int(cost)} coins' for item, cost in seller_costs.items()]
      )
      return (
          f'{buyer_name} proposed: {buyer_action}. '
          f'Your costs are: {cost_str}. '
          'Accept if the offered price exceeds your cost for that item.'
      )
    else:
      return (
          f'{buyer_name} has offered {buyer_action} for the fruit. Your cost '
          f'to acquire this fruit was {int(seller_costs)} coin(s). You will '
          f'profit if you accept any offer above {int(seller_costs)} coin(s).'
      )

  def _send_observation(self, player_name: str, observation: str) -> None:
    """Send an observation to a specific player."""
    if self._observation_component_key:
      make_observation = self.get_entity().get_component(
          self._observation_component_key,
          type_=make_observation_component.MakeObservation,
      )
      make_observation.add_to_queue(player_name, observation)

    if self._memory_component_key:
      memory = self.get_entity().get_component(
          self._memory_component_key,
          type_=memory_component.Memory,
      )
      memory.add(f'{player_name} observed: {observation}')

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    self._latest_action_spec_output_type = action_spec.output_type
    return ''

  def post_act(
      self,
      event_statement: str,
  ) -> str:
    finished = False
    is_action_complete = False

    if self._latest_action_spec_output_type == entity_lib.OutputType.RESOLVE:
      event_resolution = self.get_entity().get_component(
          self._event_resolution_component_key,
          type_=event_resolution_component.EventResolution,
      )

      player_name = event_resolution.get_active_entity_name()
      choice = event_resolution.get_putative_action()

      # Get current buyer/seller from scene participants (not hardcoded names)
      current_buyer, current_seller = self._get_current_buyer_and_seller()

      # Get current scene participants - this is the key fix!
      # We must check against current scene participants, not the initial pair
      current_participants = self._get_current_scene_participants()
      if player_name in current_participants and choice:
        self._partial_joint_action[player_name] = choice

        # Check if the player who just acted is the buyer for this scene
        if player_name == current_buyer and not self._buyer_has_proposed:
          self._buyer_has_proposed = True

          seller_costs = self._seller_costs_registry.get(current_seller)
          proposal_obs = self._format_proposal_observation(
              player_name, choice, seller_costs
          )

          self._send_observation(current_seller, proposal_obs)

          if self._verbose:
            print(
                termcolor.colored(
                    f'Sent proposal observation to {current_seller}: '
                    f'{proposal_obs}',
                    'cyan',
                )
            )

      joint_action = dict(self._partial_joint_action)
      is_action_complete = self._joint_action_is_complete(joint_action)

      if is_action_complete:
        if self._verbose:
          print(
              termcolor.colored(
                  f'Joint action is complete: {joint_action}', 'yellow'
              )
          )

        this_step_scores = self._action_to_scores(joint_action)
        for name in this_step_scores:
          if name in self._player_scores:
            self._player_scores[name] += this_step_scores[name]
          else:
            self._player_scores[name] = this_step_scores[name]
        finished = True

        observations_for_players = self._scores_to_observation(this_step_scores)
        for name, observation in observations_for_players.items():
          self._send_observation(name, observation)

        if self._verbose:
          print(termcolor.colored(self._player_scores, 'yellow'))

    self._logging_channel(
        copy.deepcopy({
            'Joint Action': self._partial_joint_action,
            'Player Scores': self._player_scores,
            'Action Complete': is_action_complete,
            'Buyer Has Proposed': self._buyer_has_proposed,
            'Key': self._pre_act_label,
            'Value': self._latest_action_spec_output_type,
        })
    )

    if finished:
      self._stage_idx += 1
      # Reset partial_joint_action for the NEXT scene's participants
      # This will be populated dynamically when the next scene starts
      self._partial_joint_action = {}
      self._buyer_has_proposed = False
      if self._verbose:
        print(
            termcolor.colored(f'Stage {self._stage_idx} is complete.', 'yellow')
        )
    return ''

  def get_scores(self) -> Mapping[str, float]:
    """Return the cumulative score for each player."""
    return self._player_scores

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'stage_idx': self._stage_idx,
        'partial_joint_action': self._partial_joint_action,
        'player_scores': self._player_scores,
        'buyer_has_proposed': self._buyer_has_proposed,
        'history': self._history,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._stage_idx = state['stage_idx']
    self._partial_joint_action = state['partial_joint_action']
    self._player_scores = state['player_scores']
    self._buyer_has_proposed = state['buyer_has_proposed']
    self._history = state['history']
