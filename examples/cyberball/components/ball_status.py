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
      verbose: bool = False,
  ):
    self._memory = memory
    self._model = model
    self._state = ''
    self._player_names = player_names
    self._partial_states = {name: '' for name in self._player_names}
    self._verbose = verbose
    self._history = []
    self._clock_now = clock_now

  def name(self) -> str:
    return 'Status of the ball'

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
    """Return a player-specific view of the ball's state."""
    return self._partial_states[player_name]

  def update(self) -> None:
    self._state = '\n'
    self._partial_states = {name: '' for name in self._player_names}
    per_player_prompt = {}
    for player_name in self._player_names:
      query = f'{player_name}'
      mems = (
          '\n'.join(
              self._memory.retrieve_associative(query, k=10, add_time=True)
          )
          + '\n'
      )
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(f'Events:\n{mems}')
      time_now = self._clock_now().strftime('[%d %b %Y %H:%M:%S]')
      prompt.statement(f'The current time is: {time_now}\n')
      ball_location = prompt.open_question(
          'Given the above events and their time, who has the ball now?'
      )
      per_player_prompt[player_name] = prompt.view().text().splitlines()
      if self._verbose:
        print(prompt.view().text())

      # Indent player status outputs.
      state_string = f'  {ball_location} has the ball.'
      self._partial_states[player_name] = state_string
      self._state = self._state + state_string

    update_log = {
        'date': self._clock_now(),
        'state': self._state,
        'partial states': self._partial_states,
        'per player prompts': per_player_prompt,
    }
    self._history.append(update_log)
