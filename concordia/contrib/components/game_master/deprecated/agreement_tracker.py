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
import datetime

from concordia.agents.deprecated import deprecated_agent
from concordia.agents.deprecated import entity_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.components.game_master.deprecated import current_scene
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import component
import numpy as np
import termcolor

_DEFAULT_CHAIN_OF_THOUGHT_PREFIX = (
    'This is a social science experiment. It is structured as a '
    'tabletop roleplaying game. You are the game master and storyteller. '
    'Your job is to make sure the game runs smoothly and accurately tracks '
    'the state of the world, subject to the laws of logic and physics. Next, '
    'you will be asked a series of questions to help you reason through '
    'whether a group of negotiating players have agreed with one another or '
    'not and what they agreed on if they did. '
    'It is important to the experiment we never mention that it is a '
    'game and we always use third-person limited perspective, even when '
    'speaking directly to the participants.'
)


class AgreementTracker(component.Component):
  """Track whether negotiating agents have agreed and what they agreed on.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      negotiating_players: Sequence[
          deprecated_agent.BasicAgent | entity_agent.EntityAgent],
      informed_players: Sequence[
          deprecated_agent.BasicAgent | entity_agent.EntityAgent],
      clock_now: Callable[[], datetime.datetime],
      resolution_scenes: Sequence[str],
      chain_of_thought_prefix: str = _DEFAULT_CHAIN_OF_THOUGHT_PREFIX,
      basic_setting: str = 'unspecified',
      name: str = 'agreement tracker',
      seed: int | None = None,
      verbose: bool = False,
  ):
    """Initialize an agreement tracker component.

    Args:
      model: a language model
      memory: an associative memory
      negotiating_players: the players who are negotiating
      informed_players: the players who are informed of the negotiation
      outcome
      clock_now: Function to call to get current time.
      resolution_scenes: Scene types in which to check for agreement.
      chain_of_thought_prefix: include this string in context before all
        reasoning steps for handling the agreements.
      basic_setting: a string to include in the context before all reasoning
      name: name of this component e.g. Possessions, Account, Property, etc
      seed: random seed
      verbose: whether to print the full update chain of thought or not
    """
    self._seed = seed
    self._model = model
    self._memory = memory
    self._negotiating_players = negotiating_players
    self._informed_players = informed_players
    self._clock_now = clock_now
    self._chain_of_thought_prefix = chain_of_thought_prefix
    self._basic_setting = basic_setting
    self._name = name
    self._verbose = verbose

    self._history = []
    self._involved_players = list(negotiating_players) + list(informed_players)

    self._resolution_scenes = resolution_scenes
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
        player.name: None for player in self._negotiating_players}

  def name(self) -> str:
    """Returns the name of this component."""
    return self._name

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def get_history(self):
    return self._history.copy()

  def update(self) -> None:
    self._current_scene.update()

  def _joint_action_is_complete(self, joint_action: Mapping[str, str]) -> bool:
    for player in self._negotiating_players:
      if joint_action[player.name] is None:
        return False
    return True

  def update_before_event(self, player_action_attempt: str) -> None:
    # `player_action_attempt` is formatted as "name: attempt".
    # we assume that the player action attempt is in the format
    # 'player_name: player_choice'. All other occurrences of ':' will be treated
    # as a part of the player choice.
    player_name, choice_str = player_action_attempt.split(': ', 1)
    if player_name not in [player.name for player in self._involved_players]:
      return
    self._partial_joint_action[player_name] = choice_str

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    chain_of_thought_str = ''
    current_scene_type = self._current_scene.state()
    finished = False
    negotiator_names = [player.name for player in self._negotiating_players]
    if current_scene_type in self._resolution_scenes:
      # Check if all players have acted so far in the current stage game.
      joint_action = self._partial_joint_action.copy()
      if self._joint_action_is_complete(joint_action):  # pytype: disable=wrong-arg-types
        # Check if negotiators agree.
        chain_of_thought = interactive_document.InteractiveDocument(
            self._model, rng=np.random.default_rng(self._seed))
        chain_of_thought.statement(
            f'{self._chain_of_thought_prefix}\nSetting: {self._basic_setting}')
        chain_of_thought.statement(f'List of negotiators: {negotiator_names}')
        chain_of_thought.statement('Statements of negotiators:')
        for negotiator in self._negotiating_players:
          chain_of_thought.statement(
              f'{negotiator.name}\'s statement: '
              f'{joint_action[negotiator.name]}')
        _ = chain_of_thought.open_question(
            question='Have the negotiators agreed? Explain your reasoning.',
            max_tokens=800,
        )
        agreement = chain_of_thought.open_question(
            question=(
                'What did they agree on? If they did not agree then respond'
                ' with "The negotiators were unable to come to an agreement."'
            ),
            max_tokens=500,
        )
        agreement = f'Agreement: {agreement}'
        for player in self._involved_players:
          player.observe(agreement)
          self._memory.add(agreement)

        finished = True
        chain_of_thought_str = chain_of_thought.view().text()

        if self._verbose:
          print(termcolor.colored(chain_of_thought_str, 'yellow'))

    num_players_already_acted = np.sum(
        [value is not None for value in self._partial_joint_action.values()])
    total_num_players_to_act = len(self._partial_joint_action)
    update_log = {
        'date': self._clock_now(),
        'Summary': self.name(),
        'Stage index': self._stage_idx,
        'How many players acted so far this stage': (
            f'{num_players_already_acted}/{total_num_players_to_act}'),
        'Joint action': str(self._partial_joint_action),
        'Chain of thought': chain_of_thought_str,
    }
    self._history.append(update_log)

    if finished:
      # Advance to the next stage.
      self._stage_idx += 1
      self._partial_joint_action = {
          player.name: None for player in self._negotiating_players}
