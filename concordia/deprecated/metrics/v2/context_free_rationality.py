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

"""Track extent to which an agent is taking rational actions toward a goal."""

from collections.abc import Sequence

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import clock as game_clock
from concordia.typing.deprecated import entity_component
from concordia.typing.deprecated import logging
from concordia.utils.deprecated import measurements as measurements_lib

DEFAULT_SCALE = (
    'very irrational',
    'somewhat irrational',
    'somewhat rational',
    'very rational',
)
DEFAULT_CHANNEL_NAME = 'context_free_rationality'


class RationalityMetric(entity_component.ContextComponent):
  """Metric of goal achievement for a player and its goal."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_goal: str,
      clock: game_clock.GameClock,
      scale: Sequence[str] = DEFAULT_SCALE,
      # Logging channel is for the html logs.
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
      # Measurements and metric_channel are for metric logging.
      measurements: measurements_lib.Measurements | None = None,
      metric_channel: str = DEFAULT_CHANNEL_NAME,
  ):
    """Initializes the metric.

    Args:
      model: Language model to use for the question.
      player_goal: player goal.
      clock: Clock for logging.
      scale: Scale of the metric, uses default if None.
      logging_channel: The name of the channel to push data for HTML logging.
      measurements: The measurements object to use.
      metric_channel: The name of the channel to push metric data to.
    """
    self._model = model
    self._player_goal = player_goal
    self._clock = clock
    self._scale = scale
    self._logging_channel = logging_channel

    self._timestep = 0

    self._measurements = measurements
    self._metric_channel = metric_channel
    # Get the channel so it is initialized. This is not strictly necessary, but
    # enables us to know which channels exist after initialization of agents and
    # GM.
    if self._measurements:
      self._measurements.get_channel(self._metric_channel)

  def post_act(
      self,
      action_attempt: str,
  ) -> str:
    """See base class."""
    # This should take in more context, right now it's hard to imagine how
    # it could be very accurate since it doesn't know any context beyond the
    # action attempt and goal.
    agent_name = self.get_entity().name
    doc = interactive_document.InteractiveDocument(self._model)
    answer = doc.multiple_choice_question(
        (
            f'Given that {agent_name} has goal \'{self._player_goal}\', '
            f'evaluate whether how rational it would be for them to take the '
            f'the following action:\n{action_attempt}\n'
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
        'player': agent_name,
        'goal': self._player_goal,
    }
    if self._measurements:
      self._measurements.publish_datum(self._metric_channel, datum)
    self._logging_channel(datum)

    self._timestep += 1

    return ''
