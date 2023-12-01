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


"""Metrics of player`s reputation among other players."""

from collections.abc import Sequence
import concurrent.futures
from typing import Any

from concordia.agents import basic_agent
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import agent as simulacrum_agent
from concordia.typing import clock as game_clock
from concordia.typing import metric
import numpy as np

DEFAULT_SCALE = [
    'very negative',
    'somewhat negative',
    'neutral',
    'somewhat positive',
    'very positive',
]


class ReputationMetric(metric.Metric):
  """Metric of players reputation among the each other."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      players: Sequence[basic_agent.BasicAgent],
      clock: game_clock.GameClock,
      name: str = 'Reputation',
      scale: Sequence[str] | None = None,
      verbose: bool = False,
      writer=None,
      question: str = 'What is {opining_player}\'s opinion of {of_player}?',
  ):
    """Initializes the metric.

    Args:
      model: Language model to use for the question.
      players: List of players.
      clock: Clock for logging.
      name: Name of the metric.
      scale: Scale of the metric, uses default if None.
      verbose: Whether to print logs during execution.
      writer: Writer to use for logging.
      question: The question to ask players about opinions on other players.
        Must have two formatting fields: "{opining_player}" and "{of_player}".
    """
    self._model = model
    self._name = name
    self._state = []
    self._clock = clock
    self._verbose = verbose
    self._players = players
    self._scale = scale or DEFAULT_SCALE
    self._writer = writer
    self._question = question

    self._player_by_name = {player.name: player for player in players}

  def name(
      self,
  ) -> str:
    """Returns the name of the measurement."""
    return self._name

  def update(
      self,
      observation: str,
      acting_player_name: str,
      doc: interactive_document.InteractiveDocument,
  ) -> None:
    del doc, observation  # this metric doesn't use either

    def get_reputation(current_player: basic_agent.BasicAgent) -> None:
      if current_player.name == acting_player_name:
        return

      question = (
          self._question.format(opining_player=current_player.name,
                                of_player=acting_player_name)
      )
      action_spec = simulacrum_agent.ActionSpec(
          call_to_action=question,
          output_type='CHOICE',
          options=self._scale,
      )

      with current_player.interrogate():
        answer_str = current_player.act(action_spec, memorize=False)
      answer = np.where(np.array(self._scale) == answer_str)[0][0]

      answer = float(answer) / float(len(self._scale) - 1)
      datum = {
          'time_str': self._clock.now().strftime('%H:%M:%S'),
          'clock_step': self._clock.get_step(),
          'step_metric': len(self._state),
          'value_float': answer,
          'value_str': answer_str,
          'player': acting_player_name,
          'rating_player': current_player.name,
      }
      if self._writer is not None:
        self._writer.write(datum)
        self._writer.flush()

      datum['time'] = self._clock.now()
      self._state.append(datum)
      if self._verbose:
        print(
            f'{self._name} of {acting_player_name} as viewed by '
            f'{current_player.name}:'
            f' {answer_str}'
        )

      return

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(self._players)
    ) as executor:
      executor.map(get_reputation, self._players)

  def state(self) -> list[dict[str, Any]]:
    """Return the current state of all the tracked variables."""
    return self._state.copy()

  def get_scale(self) -> Sequence[str]:
    return self._scale
