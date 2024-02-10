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


"""This construct track the status and location of the ball."""

from collections.abc import Callable, Sequence
import datetime

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component


class BallStatus(component.Component):
  """Tracks the status of the ball."""

  def __init__(
      self,
      clock_now: Callable[[], datetime.datetime],
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_names: Sequence[str],
      initial_ball_holder: str = '',
      num_memories_to_retrieve: int = 10,
      verbose: bool = False,
  ):
    self._memory = memory
    self._model = model
    self._state = f'{initial_ball_holder} has the ball.'
    self._player_names = player_names
    self._partial_states = {name: self._state for name in self._player_names}
    self._verbose = verbose
    self._history = []
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve

  def name(self) -> str:
    return 'Player who has the ball now'

  def state(self) -> str:
    return self._state

  def get_history(self):
    return self._history.copy()

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def partial_state(
      self,
      player_name: str,
  ) -> str:
    """Return a player-specific view of who has the ball."""
    return self._partial_states[player_name]

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    self._state = '\n'
    self._partial_states = {name: '' for name in self._player_names}

    prompt = interactive_document.InteractiveDocument(self._model)

    time_now = self._clock_now().strftime('[%d %b %Y %H:%M:%S]')

    memories = self._memory.retrieve_by_regex('ball')
    mems = memories[-self._num_memories_to_retrieve:]
    prompt.statement(f'Some recent events:\n{mems}')
    prompt.statement(f'The latest event: {time_now} {event_statement}')

    prompt.statement(f'The current time is: {time_now}\n')
    ball_location_idx = prompt.multiple_choice_question(
        question=('Given all the above events and their timestamps, who has ' +
                  'the ball now?'),
        answers=self._player_names,
    )
    ball_location = self._player_names[ball_location_idx]
    if self._verbose:
      print(prompt.view().text())

    state_string = f'{ball_location} has the ball.'
    self._state = state_string
    for player_name in self._player_names:
      self._partial_states[player_name] = state_string

    update_log = {
        'date': self._clock_now(),
        'state': self._state,
        'partial states': self._partial_states,
        'context': prompt.view().text().splitlines(),
    }
    self._history.append(update_log)
