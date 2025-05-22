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

"""Metric to track goal achievement for a player."""

from collections.abc import Sequence

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import clock as game_clock
from concordia.typing.deprecated import component
from concordia.utils.deprecated import measurements as measurements_lib

DEFAULT_SCALE = (
    'activity unrelated to the goal',
    'somewhat working towards the goal',
    'working towards the goal',
    'goal achieved',
)
DEFAULT_CHANNEL_NAME = 'goal_achievement'


class GoalAchievementMetric(component.Component):
  """Metric of goal achievement for a player and its goal."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_name: str,
      player_goal: str,
      clock: game_clock.GameClock,
      name: str = 'Goal Achievement',
      scale: Sequence[str] = DEFAULT_SCALE,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = DEFAULT_CHANNEL_NAME,
      verbose: bool = False,
  ):
    """Initializes the metric.

    Args:
      model: Language model to use for the question.
      player_name: player name.
      player_goal: player goal.
      clock: Clock for logging.
      name: Name of the metric.
      scale: Scale of the metric, uses default if None.
      measurements: The measurements object to publish data to.
      channel: Channel to use for logging the metric.
      verbose: Whether to print logs during execution.
    """
    self._model = model
    self._player_name = player_name
    self._player_goal = player_goal
    self._clock = clock
    self._name = name
    self._scale = scale
    self._measurements = measurements
    self._channel = channel
    # Get the channel so it is initialized. This is not strictly necessary, but
    # enables us to know which channels exist after initialization of agents and
    # GM.
    if self._measurements:
      self._measurements.get_channel(self._channel)
    self._verbose = verbose

    self._timestep = 0

  def name(
      self,
  ) -> str:
    """See base class."""
    return self._name

  def update_after_event(self, action: str) -> None:
    """See base class."""
    doc = interactive_document.InteractiveDocument(self._model)
    answer = doc.multiple_choice_question(
        (
            'Evaluate if the following activity brings'
            f' {self._player_name} closer to their goal'
            f' "{self._player_goal} .\n Activity: {action}\n'
        ),
        self._scale,
    )
    answer_str = self._scale[answer]

    answer = float(answer) / float(len(self._scale) - 1)

    datum = {
        'time_str': self._clock.now().strftime('%H:%M:%S'),
        'clock_step': self._clock.get_step(),
        'timestep': self._timestep,
        'value_float': answer,
        'value_str': answer_str,
        'player': self._player_name,
        'goal': self._player_goal,
    }
    datum['time'] = self._clock.now()

    if self._measurements:
      self._measurements.publish_datum(self._channel, datum)
    if self._verbose:
      print(f'{self._name} of {self._player_name}: {answer_str}')
    self._timestep += 1

  def state(
      self,
  ) -> str | None:
    """Returns the current state of the component."""
    return ''
