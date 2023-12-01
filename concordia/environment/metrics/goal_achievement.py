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


"""Metrics of goal achievement per player."""

from collections.abc import Sequence
from typing import Any

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import metric

DEFAULT_SCALE = [
    'activity unrelated to the goal',
    'somewhat working towards the goal',
    'working towards the goal',
    'goal achieved',
]


class GoalAchievementMetric(metric.Metric):
  """Metric of goal achievement per player / goal pair."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_goals: dict[str, str],
      clock: game_clock.GameClock,
      name: str = 'Goal Achievement',
      scale: Sequence[str] | None = None,
      verbose: bool = False,
      writer=None,
  ):
    """Initializes the metric.

    Args:
      model: Language model to use for the question.
      player_goals: Dictionary of player name to player goal.
      clock: Clock for logging.
      name: Name of the metric.
      scale: Scale of the metric, uses default if None.
      verbose: Whether to print logs during execution.
      writer: Writer to use for logging.
    """
    self._model = model
    self._name = name
    self._state = []
    self._clock = clock
    self._verbose = verbose
    self._player_goals = player_goals
    self._scale = scale or DEFAULT_SCALE
    self._writer = writer

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
    acting_player_goal = self._player_goals[acting_player_name]
    doc = doc.new()
    answer = doc.multiple_choice_question(
        (
            'Evaluate if the following activity brings'
            f' {acting_player_name} closer to their goal'
            f' "{acting_player_goal} .\n Activity: {observation}\n'
        ),
        self._scale,
    )
    answer_str = self._scale[answer]

    answer = float(answer) / float(len(self._scale) - 1)

    datum = {
        'time_str': self._clock.now().strftime('%H:%M:%S'),
        'clock_step': self._clock.get_step(),
        'step_metric': len(self._state),
        'value_float': answer,
        'value_str': answer_str,
        'player': acting_player_name,
        'goal': acting_player_goal,
    }
    if self._writer is not None:
      self._writer.write(datum)
      self._writer.flush()
    datum['time'] = self._clock.now()

    self._state.append(datum)
    if self._verbose:
      print(f'{self._name} of {acting_player_name}: {answer_str}')

  def state(self) -> list[dict[str, Any]]:
    """Return the current state of all the tracked variables."""
    return self._state.copy()

  def get_scale(self) -> Sequence[str]:
    return self._scale
