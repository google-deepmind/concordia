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


"""Metric for tracking the answer to a configurable question."""

from collections.abc import Sequence
from typing import Callable

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import clock as game_clock
from concordia.typing.deprecated import component
from concordia.utils.deprecated import measurements as measurements_lib
import termcolor


DEFAULT_SCALE = (
    'Definitively not',
    'Maybe not',
    'Maybe yes',
    'Definitively yes',
)

DEFAULT_QUESTION = 'Would {player_name} talk to a stranger?'
DEFAULT_CHANNEL_NAME = 'question'


class Question(component.Component):
  """Metrics for tracking the answer to a configurable question.

  This component triggers a question on `update`. The context for the question
  is given by the callable `context_fn`, which is called only once. The response
  to the question is evaluated with the given scale, and logged as a datum in
  the specified channel of the measurements.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_name: str,
      context_fn: Callable[[], str],
      clock: game_clock.GameClock,
      name: str = 'Question',
      question: str = DEFAULT_QUESTION,
      scale: Sequence[str] = DEFAULT_SCALE,
      verbose: bool = False,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = DEFAULT_CHANNEL_NAME,
  ):
    """Initializes the component.

    Args:
      model: The model (LLM) to use.
      player_name: The name of the player.
      context_fn: The function to get the parent state (typically a player)
      clock: The clock of the simulation.
      name: The name of the component.
      question: The question to ask. Might have the formatting "{player_name}"
        which will be replaced by the player's name.
      scale: The possible answer options for the question.
      verbose: whether to `print` the outcome.
      measurements: the measurements object to publish data.
      channel: Name of the channel to publish measurements to.
    """
    self._model = model
    self._player_name = player_name
    self._context_fn = context_fn
    self._clock = clock
    self._name = name
    self._question = question
    if scale:
      self._scale = list(scale)
    else:
      raise ValueError('scale must be specified.')
    self._verbose = verbose
    self._measurements = measurements
    self._channel = channel
    # Get the channel so it is initialized. This is not strictly necessary, but
    # enables us to know which channels exist after initialization of agents and
    # GM.
    if self._measurements:
      self._measurements.get_channel(self._channel)

    self._timestep = 0

  def name(
      self,
  ) -> str:
    """Returns the name of the measurement."""
    return self._name

  def update(self) -> None:
    """See base class."""
    prompt = interactive_document.InteractiveDocument(self._model)
    parent_state = self._context_fn()
    prompt.statement(parent_state)

    question = self._question.format(player_name=self._player_name)

    answer = prompt.multiple_choice_question(
        question=question, answers=self._scale,
    )
    answer_str = self._scale[answer]

    answer_float = answer / (len(self._scale) - 1)
    datum = {
        'time_str': self._clock.now().strftime('%H:%M:%S'),
        'clock_step': self._clock.get_step(),
        'timestep': self._timestep,
        'value_float': answer_float,
        'value_str': answer_str,
        'player': self._player_name,
    }
    if self._measurements is not None:
      self._measurements.publish_datum(self._channel, datum)

    datum['time'] = self._clock.now()
    if self._verbose:
      print(termcolor.colored(prompt.view().text(), 'green'), end='')
      print(f'{question}\n{self._player_name}: {answer_str}')
    self._timestep += 1

  def state(
      self,
  ) -> str | None:
    """Returns the current state of the component."""
    return ''
