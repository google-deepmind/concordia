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


"""This construct track the status and location of players."""

from collections.abc import Callable, Sequence
import datetime

from concordia.associative_memory.deprecated import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import component


class Relationships(component.Component):
  """Tracks the status of players."""

  def __init__(
      self,
      clock_now: Callable[[], datetime.datetime],
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      other_agent_names: Sequence[str],
      num_memories_to_retrieve: int = 10,
      verbose: bool = False,
  ):
    """Constructs a Relationships component.

    Args:
      clock_now: A function that returns the current time.
      model: A language model.
      memory: An associative memory.
      agent_name: name of the focal agent.
      other_agent_names: A list of agent names to track.
      num_memories_to_retrieve: The number of memories to retrieve (max).
      verbose: Whether to print the prompt to the console.
    """
    self._memory = memory
    self._model = model
    self._state = ''
    self._agent_name = agent_name
    self._other_agent_names = other_agent_names
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._partial_states = {name: '' for name in self._other_agent_names}
    self._verbose = verbose
    self._history = []
    self._clock_now = clock_now
    self._last_update = self._clock_now() - datetime.timedelta(days=365)

  def name(self) -> str:
    return 'relationships'

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
    """Return a player-specific view of the construct's state."""
    return self._partial_states[player_name]

  def update(self) -> None:
    if self._clock_now() == self._last_update:
      return
    self._last_update = self._clock_now()
    new_state = ''
    self._partial_states = {name: '' for name in self._other_agent_names}
    per_player_prompt = {}
    for player_name in self._other_agent_names:
      memories = self._memory.retrieve_by_regex(player_name)
      memories = memories[-self._num_memories_to_retrieve :]
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement('Events:\n' + '\n'.join(memories) + '\n')
      time_now = self._clock_now().strftime('[%d %b %Y %H:%M:%S]')
      prompt.statement(f'The current time is: {time_now}\n')
      relationship = (
          prompt.open_question(
              'Given the above events and their time, write a short summary of'
              f' the relationships between {self._agent_name} and'
              f' {player_name}. ',
              answer_prefix=f'{self._agent_name} and {player_name} are ',
          )
          + '\n'
      )
      per_player_prompt[player_name] = prompt.view().text().splitlines()
      if self._verbose:
        print(prompt.view().text())

      # Indent player status outputs.
      player_state_string = (
          f'{self._agent_name} and {player_name} are ' + relationship
      )
      self._partial_states[player_name] = player_state_string
      new_state = new_state + player_state_string

    self._state = new_state

    update_log = {
        'date': self._clock_now(),
        'state': self._state,
        'partial states': self._partial_states,
        'per player prompts': per_player_prompt,
    }
    self._history.append(update_log)
