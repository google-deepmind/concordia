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

"""A component for computing pressure created through labor collective action."""

from collections.abc import Callable, Mapping, Sequence
import copy

from concordia.agents import entity_agent
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import event_resolution as event_resolution_component
from concordia.components.game_master import make_observation as make_observation_component
from concordia.components.game_master import scene_tracker
from concordia.components.game_master import switch_act
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import termcolor


# A function that maps number of cooperators to a scalar.
CollectiveActionProductionFunction = Callable[[int], float]
PlayersT = Sequence[entity_agent.EntityAgent]


class PayoffMatrix(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component for computing payoffs for a game."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      acting_player_names: Sequence[str],
      action_to_scores: Callable[[Mapping[str, str]], Mapping[str, float]],
      scores_to_observation: Callable[[Mapping[str, float]], Mapping[str, str]],
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
    """Initialize a component for computing payoffs.

    Args:
      model: a language model
      acting_player_names: sequence of names of players whos actions influence
        the payoff.
      action_to_scores: function that maps a dictionary of actions by players to
        a dictionary of scores for each player
      scores_to_observation: function that maps a dictionary of scores for each
        player to a dictionary of observations for each player.
      event_resolution_component_key: The key of the event resolution component.
      observation_component_key: The key of the observation component to send
        observations to players. If None, no observations will be sent.
      memory_component_key: The key of the memory component to the observations
        by players. If None, no observations will be added to the memory.
      scene_tracker_component_key: The key of the scene tracker component, so
        that only participants in the current scene need to act.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
      verbose: whether to print the full update chain of thought or not
    """
    self._pre_act_label = pre_act_label

    self._model = model
    self._observation_component_key = observation_component_key
    self._memory_component_key = memory_component_key
    self._event_resolution_component_key = event_resolution_component_key
    self._acting_player_names = acting_player_names
    self._action_to_scores = action_to_scores
    self._scores_to_observation = scores_to_observation
    self._scene_tracker_component_key = scene_tracker_component_key

    self._verbose = verbose

    self._history = []
    self._player_scores = {name: 0.0 for name in self._acting_player_names}
    self._latest_action_spec_output_type = None

    self.reset()

  def reset(self) -> None:
    self._stage_idx = 0
    # Map each player's name to their component of the joint action.
    self._partial_joint_action = {
        name: None for name in self._acting_player_names
    }

  def _get_current_scene_participants(self) -> Sequence[str]:
    if self._scene_tracker_component_key:
      scene_tracker_component = self.get_entity().get_component(
          self._scene_tracker_component_key,
          type_=scene_tracker.SceneTracker,
      )
      return scene_tracker_component.get_participants()
    return self._acting_player_names

  def _joint_action_is_complete(self, joint_action: Mapping[str, str]) -> bool:
    for acting_player_name in self._get_current_scene_participants():
      if joint_action[acting_player_name] is None:
        return False
    return True

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
    # joint_action_for_log = ''
    finished = False
    is_action_complete = False
    if self._latest_action_spec_output_type == entity_lib.OutputType.RESOLVE:

      event_resolution = self.get_entity().get_component(
          self._event_resolution_component_key,
          type_=event_resolution_component.EventResolution,
      )

      player_name = event_resolution.get_active_entity_name()
      choice = event_resolution.get_putative_action()
      if player_name in self._acting_player_names and choice:
        self._partial_joint_action[player_name] = choice

      # Check if all players have acted so far in the current stage game.
      joint_action = self._partial_joint_action.copy()
      is_action_complete = self._joint_action_is_complete(joint_action)
      if is_action_complete:
        if self._verbose:
          print(
              termcolor.colored(
                  f'Joint action is complete: {joint_action}', 'yellow'
              )
          )
        # Get the scores for each player for the current step.
        this_step_scores = self._action_to_scores(joint_action)
        for player_name in this_step_scores:
          if player_name in self._player_scores:
            self._player_scores[player_name] += this_step_scores[player_name]
          else:
            self._player_scores[player_name] = this_step_scores[player_name]
        finished = True

        # Inform players of the outcome of the current step.
        observations_for_players = self._scores_to_observation(
            self._player_scores
        )
        for player_name, observation in observations_for_players.items():
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
            memory.add(f'{player_name} learned that {observation}')

        if self._verbose:
          print(termcolor.colored(self._player_scores, 'yellow'))

    self._logging_channel(copy.deepcopy({
        'Joint Action': self._partial_joint_action,
        'Player Scores': self._player_scores,
        'Action Complete': is_action_complete,
        'Key': self._pre_act_label,
        'Value': self._latest_action_spec_output_type,
    }))

    if finished:
      # Advance to the next stage.
      self._stage_idx += 1
      self._partial_joint_action = {
          name: None for name in self._acting_player_names
      }
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
        'history': self._history,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._stage_idx = state['stage_idx']
    self._partial_joint_action = state['partial_joint_action']
    self._player_scores = state['player_scores']
    self._history = state['history']
