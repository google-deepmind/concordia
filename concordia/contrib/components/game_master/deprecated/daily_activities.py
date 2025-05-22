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

"""Represents each agent's daily activities over a period of time."""

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import datetime
import random
import threading
from typing import Any

from concordia.agents.deprecated import entity_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.components.game_master.deprecated import current_scene
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import component
import termcolor


_DEFAULT_CHAIN_OF_THOUGHT_PREFIX = (
    'This is a social science experiment. It is structured as a '
    'tabletop roleplaying game. You are the game master and storyteller. '
    'Your job is to make sure the game runs smoothly and accurately tracks '
    'the state of the world, subject to the laws of logic and physics. Next, '
    'you will be asked a series of questions to help you reason through '
    'whether a specific event should be deemed as having caused a change in '
    'the proportion of time an individual spends in various daily activities. '
    'For instance, if someone verbally commits to spending more time on a '
    'particular activity then you should account for that and update the '
    'record of their daily activities accordingly. '
    'Never mention that it is a game. Always use third-person limited '
    'perspective, even when speaking directly to the participants.'
)

_DEFAULT_FRACTION = 0.0

ActivityFractionsType = (
    Mapping[str, float | int] | Mapping[str, Mapping[str, float | int]]
)


def _is_floatable(element: Any) -> bool:
  # If you expect None to be passed:
  if element is None:
    return False
  try:
    float(element)
    return True
  except ValueError:
    return False


@dataclasses.dataclass(frozen=True)
class ActivityConfig:
  """Class for configuring a type of activity to track in a DailyActivities."""

  name: str
  minimum: float = 0.0
  maximum: float = 1.0

  def check_valid(self, amount: float) -> None:
    """Checks if activity daily amount is valid for this activity."""
    if amount < self.minimum or amount > self.maximum:
      raise ValueError('Amount out of bounds')


class DailyActivities(component.Component):
  """A grounded variable tracking proportions of time per day in python."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      activity_configs: Sequence[ActivityConfig],
      resolution_scene: str,
      players: Sequence[entity_agent.EntityAgent],
      player_initial_activity_distribution: dict[str, dict[str, float]],
      clock_now: Callable[[], datetime.datetime],
      num_to_retrieve: int = 10,
      chain_of_thought_prefix: str = _DEFAULT_CHAIN_OF_THOUGHT_PREFIX,
      basic_setting: str = '',
      name: str = 'DailyActivities',
      verbose: bool = False,
  ):
    """Initialize a grounded activities component tracking objects in python.

    Args:
      model: a language model
      memory: an associative memory
      activity_configs: sequence of activity configurations
      resolution_scene: on which scene type should this component be updated
        after the event, i.e. when to check the joint action and compute results
      players: sequence of players who have an activities and will observe it.
      player_initial_activity_distribution: dict mapping player name to a
        dictionary with activities as keys and initial daily time as values.
      clock_now: Function to call to get current time.
      num_to_retrieve: number of recent memories to retrieve for context.
      chain_of_thought_prefix: include this string in context before all
        reasoning steps for handling the activities.
      basic_setting: a string to include in the context before all reasoning
      name: the name of this component e.g. Possessions, Account, Property, etc
      verbose: whether to print the full update chain of thought or not
    """
    self._model = model
    self._memory = memory
    self._players = players
    self._player_initial_activity_distribution = (
        player_initial_activity_distribution)
    self._chain_of_thought_prefix = chain_of_thought_prefix
    self._clock_now = clock_now
    self._num_to_retrieve = num_to_retrieve
    self._basic_setting = basic_setting
    self._name = name
    self._verbose = verbose

    self._activity_names = [config.name for config in activity_configs]
    self._activity_configs_dict = {
        config.name: config for config in activity_configs
    }
    self._player_names = list(player_initial_activity_distribution.keys())
    self._names_to_players = {player.name: player for player in self._players}

    self._activities = {}
    for player_name, fractions in player_initial_activity_distribution.items():
      self._activities[player_name] = {
          activity_name: fractions.get(activity_name, _DEFAULT_FRACTION)
          for activity_name in self._activity_names
      }

    self._resolution_scene = resolution_scene
    self._current_scene = current_scene.CurrentScene(
        name='current scene type',
        memory=self._memory,
        clock_now=self._clock_now,
        verbose=verbose,
    )

    self._latest_update_log = None
    self._state = ''
    self._partial_states = {name: '' for name in self._player_names}

    self._lock = threading.Lock()

    # Set the initial state's string representation.
    self.update()

  def name(self) -> str:
    """Returns the name of this component."""
    return self._name

  def get_last_log(self):
    if self._latest_update_log is not None:
      return self._latest_update_log

  def _get_player_daily_activities_str(self, player_name: str) -> str:
    return f"{player_name}'s {self._name}: " + str(
        self._activities[player_name]
    )

  def state(self) -> str:
    return self._state

  def update(self) -> None:
    with self._lock:
      self._previous_scene_type = self._current_scene.state()
      self._current_scene.update()

      self._state = '\n'.join(
          [self._get_player_daily_activities_str(name)
           for name in self._player_names]
      )
      self._memory.add(self._state)
      self._partial_states = {
          name: self._get_player_daily_activities_str(name)
          for name in self._player_names
      }

  def update_before_event(self, player_action_attempt: str) -> None:
    # we assume that the player action attempt is in the format
    # 'player_name: player_choice'. All other occurrences of ':' will be treated
    # as a part of the player's action attempt.
    player_name, _ = player_action_attempt.split(': ', 1)
    if player_name not in [player.name for player in self._players]:
      return
    self._current_player = player_name

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    with self._lock:
      current_scene_type = self._current_scene.state()
      chain_of_thought = interactive_document.InteractiveDocument(self._model)
      activity_fraction_effects = []
      if (current_scene_type == self._resolution_scene or
          self._previous_scene_type == self._resolution_scene):
        chain_of_thought.statement(
            f'{self._chain_of_thought_prefix}\nSetting: {self._basic_setting}')
        chain_of_thought.statement(f'List of individuals: {self._player_names}')
        chain_of_thought.statement(
            f'List of activities: {self._activity_names}')
        chain_of_thought.statement(f'Statement: {event_statement}')
        proceed = chain_of_thought.yes_no_question(
            question=(
                f'In the statement, did {self._current_player} '
                'say they will devote specific fractions of time '
                'to any of the listed activities?'
            )
        )
        if proceed:
          _ = chain_of_thought.open_question(
              question=(
                  'Which individuals made a commitment '
                  'to change their behavior? What '
                  'commitment did each of them make? What proportion of '
                  'their time will they probably spend on each activity '
                  'in the future? When answering, include fractions of '
                  'time spent in each activity in the format:\n'
                  '`name` will spend `fraction`% of their time on '
                  '`activity`. Only consider the activities listed above. '
                  'Fractions should be between 0 and 1 and must sum to 1.'
              ),
              terminators=('\n\n',)
          )

          cloned_thought_chain = chain_of_thought.copy()
          fractions_string = cloned_thought_chain.open_question(
              question=(
                  f'What fraction of time will {self._current_player} '
                  'devote to each of the following activities: '
                  f'{self._activity_names}? '
                  'Delimit fractions per activity with commas. Fractions '
                  'must sum to 1. For example, if the activities are:\n'
                  '[sleeping, hunting, dancing]\nthen a valid response '
                  'would be:\n[0.15,0.25,0.60]\nwhich would indicate an '
                  'intention to spend 15% of the time sleeping, 25% of the '
                  'time hunting, and 60% of the time dancing.')
          )
          if self._verbose:
            print(termcolor.colored(cloned_thought_chain.view().text(), 'red'))
          can_parse = True
          if fractions_string:
            fractions_list = fractions_string.strip().replace(
                '[', '').replace(']', '').split(',',
                                                len(self._activity_names) - 1)
            if len(fractions_list) == len(self._activity_names):
              for fraction in fractions_list:
                if not _is_floatable(fraction):
                  can_parse = False
              if can_parse:
                fractions_list = [
                    float(fraction) for fraction in fractions_list]
                self._activities[self._current_player] = {
                    activity_name: fraction for activity_name, fraction
                    in zip(self._activity_names, fractions_list)}
          else:
            self._activities[self._current_player] = {
                activity_name: 1.0 / len(self._activity_names)
                for activity_name in self._activity_names}

    # Update the string representation of all daily activity fractions.
    self.update()

    if current_scene_type == self._resolution_scene:
      if self._verbose:
        print(termcolor.colored(chain_of_thought.view().text(), 'yellow'))
        print(termcolor.colored(self.state(), 'yellow'))

      self._latest_update_log = {
          'date': self._clock_now(),
          'Summary': str(self._activities),
          'Daily activities': self.state(),
          'Chain of thought': {
              'Summary': f'{self._name} chain of thought',
              'Chain': chain_of_thought.view().text().splitlines(),
          },
      }
      self._memory.extend(activity_fraction_effects)

  def get_daily_activity_proportions(
      self,
      player_name: str | None = None,
  ) -> ActivityFractionsType:
    """Return the daily activity fractions of player `player_name`."""
    with self._lock:
      if player_name is None:
        return copy.deepcopy(self._activities)
      else:
        return copy.deepcopy(self._activities[player_name])


class Payoffs(component.Component):
  """This component assigns score based on daily activities."""

  def __init__(
      self,
      memory: associative_memory.AssociativeMemory,
      daily_activities: DailyActivities,
      players: Sequence[entity_agent.EntityAgent],
      clock_now: Callable[[], datetime.datetime],
      player_score_fn: Callable[
          [
              str,
              ActivityFractionsType,
              str,
              associative_memory.AssociativeMemory,
              str,
          ],
          tuple[float, Sequence[str]],
      ],
      get_timepoint_fn: Callable[[], str],
      name: str = '   \n',
  ):
    """Initialize a component to compute scores from daily activities of all.

    Args:
      memory: an associative memory
      daily_activities: the component to use to get the activities of players.
      players: sequence of players who have daily activities.
      clock_now: Function to call to get current time.
      player_score_fn: function to compute an individual's score
        the value returned by this function is added to the score of the
        specified player.
      get_timepoint_fn: function with no arguments to get the current timepoint,
        scores will be able to depend on the value returned by this function
        e.g. a function returning the current year could be used to restrict
        a score calculation to events that occurred in the current year.
      name: the name of this component e.g. Payoffs
    """
    self._memory = memory
    self._daily_activities = daily_activities
    self._players = players
    self._name = name
    self._clock_now = clock_now
    self._player_score_fn = player_score_fn
    self._get_timepoint_fn = get_timepoint_fn

    self._player_scores = {player.name: 0.0 for player in self._players}
    self._names_to_players = {player.name: player for player in self._players}

    self._current_scene = current_scene.CurrentScene(
        name='current scene type',
        memory=self._memory,
        clock_now=self._clock_now,
        verbose=False,
    )

    self._partial_states = {player.name: '' for player in self._players}
    self._latest_score_increments = {
        player.name: None for player in self._players}

  def name(self) -> str:
    """Returns the name of this component."""
    return self._name

  def state(self) -> str:
    return '\n'.join({player_name: partial_state for player_name, partial_state
                      in self._partial_states.items()})

  def partial_state(
      self,
      player_name: str,
  ) -> str:
    """Return a player-specific view of the component's state."""
    if player_name in self._partial_states:
      partial_state = self._partial_states[player_name]
      self._memory.add(partial_state)
      return partial_state
    else:
      return ''

  def update(self) -> None:
    self._current_scene.update()
    current_scene_type = self._current_scene.state()
    activity_fractions = self._daily_activities.get_daily_activity_proportions()
    for player in self._players:
      score_increment, events = self._player_score_fn(
          current_scene_type,
          activity_fractions,
          player.name,
          self._memory,
          self._get_timepoint_fn())
      self._latest_score_increments[player.name] = score_increment
      events = list(events)
      random.shuffle(events)
      events_str = ' '.join(events)
      if events_str:
        self._partial_states[player.name] = f'Events this year: {events_str}'

  def update_before_event(self, player_action_attempt: str) -> None:
    # we assume that the player action attempt is in the format
    # 'player_name: player_choice'. All other occurrences of ':' will be treated
    # as a part of the player's action attempt.
    player_name, _ = player_action_attempt.split(': ', 1)
    if player_name not in [player.name for player in self._players]:
      return
    if self._latest_score_increments[player_name] is not None:
      self._player_scores[player_name] += self._latest_score_increments[
          player_name]
    self._latest_score_increments = {
        player.name: None for player in self._players}

  def update_after_event(
      self,
      unused_event_statement: str,
  ) -> None:
    self.update()

  def get_scores(self) -> Mapping[str, float]:
    """Return the cumulative score for each player."""
    return self._player_scores
