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


"""Metric of player's opinion of other players."""

from collections.abc import Sequence
import concurrent.futures
from typing import Callable

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import component
from concordia.utils import measurements as measurements_lib

DEFAULT_SCALE = (
    'very negative',
    'somewhat negative',
    'neutral',
    'somewhat positive',
    'very positive',
)
DEFAULT_CHANNEL_NAME = 'opinion_of_others'


class OpinionOfOthersMetric(component.Component):
  """Metric of opinion of other players by a player.

  This component triggers a series of questions on `update`, one for each player
  in `player_names`. The context for all questions is given by the callable
  `context_fn`, which is called only once. The responses to the question are
  evaluated with the given scale, and logged as a datum in the specified channel
  of the measurements.
  """

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      player_name: str,
      player_names: Sequence[str],
      context_fn: Callable[[], str],
      clock: game_clock.GameClock,
      name: str = 'Opinion',
      scale: Sequence[str] = DEFAULT_SCALE,
      verbose: bool = False,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = DEFAULT_CHANNEL_NAME,
      question: str = 'What is {opining_player}\'s opinion of {of_player}?',
  ):
    """Initializes the metric.

    Args:
      model: Language model to use for the question.
      player_name: The name of the player opining on others.
      player_names: List of player names, might include the opining player.
      context_fn: The function to get the context text for the question.
        (typically this is the player state). This function will be called on
        `update`.
      clock: Clock for logging.
      name: Name of the metric.
      scale: Scale of the metric, uses default if None.
      verbose: Whether to print logs during execution.
      measurements: The measurements object to publish data to.
      channel: Channel to use for logging the metric.
      question: The question to ask the player about opinions on other players.
        Must have two formatting fields: "{opining_player}" and "{of_player}".

    Raises:
      ValueError: If player_names or scale are empty.
    """
    self._model = model
    self._name = name
    self._clock = clock
    self._verbose = verbose
    self._player_name = player_name
    if player_names:
      self._player_names = list(player_names)
    else:
      raise ValueError('player_names must be specified.')
    self._context_fn = context_fn
    if scale:
      self._scale = list(scale)
    else:
      raise ValueError('scale must be specified.')
    self._measurements = measurements
    self._channel = channel
    # Get the channel so it is initialized. This is not strictly necessary, but
    # enables us to know which channels exist after initialization of agents and
    # GM.
    if self._measurements:
      self._measurements.get_channel(self._channel)
    self._question = question

    self._timestep = 0

  def name(
      self,
  ) -> str:
    """Returns the name of the measurement."""
    return self._name

  def _get_opinion(self, of_player: str) -> None:
    if of_player == self._player_name:
      return  # No self opinions.

    prompt = interactive_document.InteractiveDocument(self._model)
    parent_state = self._context_fn()
    prompt.statement(parent_state)

    question = self._question.format(
        opining_player=self._player_name,
        of_player=of_player,
    )

    answer = prompt.multiple_choice_question(
        question=question, answers=self._scale,
    )
    answer_str = self._scale[answer]

    answer_float = float(answer) / float(len(self._scale) - 1)
    datum = {
        'time_str': self._clock.now().strftime('%H:%M:%S'),
        'clock_step': self._clock.get_step(),
        'timestep': self._timestep,
        'value_float': answer_float,
        'value_str': answer_str,
        'opining_player': self._player_name,
        'of_player': of_player,
    }
    if self._measurements:
      self._measurements.publish_datum(self._channel, datum)

    datum['time'] = self._clock.now()
    if self._verbose:
      print(
          f'{self._name} of {of_player} as viewed by '
          f'{self._player_name}: {answer_str}'
      )

    return

  def update(self) -> None:
    """See base class."""

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(self._player_names)
    ) as executor:
      executor.map(self._get_opinion, self._player_names)
    self._timestep += 1

  def state(
      self,
  ) -> str | None:
    """Returns the current state of the component."""
    return ''
