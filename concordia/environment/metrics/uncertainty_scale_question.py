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


"""Metrics for tracking the answer to a configurable question."""

from collections.abc import Sequence
from typing import Any

from concordia.agents import basic_agent
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import agent as simulacrum_agent
from concordia.typing import clock as game_clock
from concordia.typing import metric
import numpy as np


DEFAULT_SCALE = [
    'Definitively not',
    'Maybe not',
    'Maybe yes',
    'Definitively yes',
]

DEFAULT_QUESTION = 'Would {agent_name} talk to a stranger?'


class Question(metric.Metric):
  """Metrics for tracking the answer to a configurable question."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      players: Sequence[basic_agent.BasicAgent],
      clock: game_clock.GameClock,
      name: str = 'Question',
      question: str | None = None,
      scale: Sequence[str] | None = None,
      verbose: bool = False,
      writer=None,
  ):
    self._model = model
    self._name = name
    self._state = []
    self._clock = clock
    self._verbose = verbose
    self._players = players
    self._scale = scale or DEFAULT_SCALE
    self._writer = writer
    self._question = question or DEFAULT_QUESTION

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
    question = self._question.format(agent_name=acting_player_name)
    action_spec = simulacrum_agent.ActionSpec(
        call_to_action=question,
        output_type='CHOICE',
        options=self._scale,
    )
    current_player = self._player_by_name[acting_player_name]

    with current_player.interrogate():
      answer_str = current_player.act(action_spec)
    answer = np.where(np.array(self._scale) == answer_str)[0][0]

    answer = float(answer) / float(len(self._scale) - 1)
    datum = {
        'time_str': self._clock.now().strftime('%H:%M:%S'),
        'clock_step': self._clock.get_step(),
        'step_metric': len(self._state),
        'value_float': answer,
        'value_str': answer_str,
        'player': acting_player_name,
    }
    if self._writer is not None:
      self._writer.write(datum)
      self._writer.flush()

    datum['time'] = self._clock.now()
    self._state.append(datum)
    if self._verbose:
      print(f'{question}\n{acting_player_name}: {answer_str}')

  def state(self) -> list[dict[str, Any]]:
    """Return the current state of all the tracked variables."""
    return self._state.copy()

  def get_scale(self) -> Sequence[str]:
    return self._scale
