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


"""Metric of election outcome."""

from collections.abc import Sequence
from typing import Any

from concordia.document import interactive_document
from examples.village.components import elections
from concordia.typing import clock as game_clock
from concordia.typing import metric


class Elections(metric.Metric):
  """A metric to track votes in an election."""

  def __init__(
      self,
      clock: game_clock.GameClock,
      election_externality: elections.Elections,
      name: str = 'Vote count',
      verbose: bool = False,
      writer=None,
  ):
    self._name = name
    self._state = []
    self._clock = clock
    self._verbose = verbose
    self._writer = writer
    self._election_tracker = election_externality
    self._vote_count = election_externality.get_vote_count()

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
    self._vote_count = self._election_tracker.get_vote_count()
    if acting_player_name not in self._vote_count.keys():
      return
    answer = self._vote_count[acting_player_name]
    answer_str = str(answer)
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
      print(f'{self._name} of {acting_player_name}: {answer_str}')

  def state(self) -> list[dict[str, Any]]:
    """Return the current state of all the tracked variables."""
    return self._state.copy()

  def get_scale(self) -> Sequence[str]:
    return list(set([str(i) for i in self._vote_count.values()]))
