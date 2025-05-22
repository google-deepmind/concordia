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


"""Component that implements elections within a game master."""

from collections.abc import Callable, Sequence
import datetime

from concordia.associative_memory.deprecated import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import agent
from concordia.typing.deprecated import component
from concordia.utils.deprecated import measurements as measurements_lib
import termcolor

DEFAULT_CHANNEL_NAME = 'election'


class Elections(component.Component):
  """Tracks elections."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      voters: Sequence[agent.GenerativeAgent],
      candidates: Sequence[str],
      clock_now: Callable[[], datetime.datetime],
      verbose: bool = False,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = DEFAULT_CHANNEL_NAME,
  ):
    """Initializes the election tracker.

    Args:
      model: The language model to use.
      memory: The memory to use.
      voters: The agent voters.
      candidates: The candidates in the election.
      clock_now: Function to call to get current time. Used for logging.
      verbose: Whether to print verbose messages.
      measurements: Optional object to publish data from the elections.
      channel: Channel in measurements to publish to.
    """
    self._model = model
    self._memory = memory
    self._voters = voters
    self._candidates = candidates
    self._clock_now = clock_now
    self._verbose = verbose
    self._measurements = measurements
    self._channel = channel

    self._voter_names = [voter.name for voter in self._voters]
    self._vote_count = {candidate: 0 for candidate in self._candidates}
    self._citizens_who_already_voted = set()

    self._voter_by_name = {voter.name: voter for voter in self._voters}
    self._state = 'Polls are not open yet.'
    self._partial_states = None
    self._history = []

    self._polls_open = False
    self._winner_declared = False
    self._timestep = 0

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def get_history(self):
    return self._history.copy()

  def open_polls(self) -> None:
    self._polls_open = True
    self._state = 'Polls are open, voting in progress.'

  def declare_winner(self) -> None:
    if not self._winner_declared:
      self._winner_declared = True
      self._polls_open = False
      winner = max(self._vote_count, key=self._vote_count.get)
      self._state = f'Polls are closed. {winner} won the election.'
      if self._verbose:
        print(termcolor.colored('\n' + self._state, 'red'), end='')

      self._memory.add(self._state, tags=['election tracker'])

  def name(self) -> str:
    return 'State of election'

  def state(self) -> str:
    return self._state

  def update(self) -> None:
    pass

  def get_vote_count(self) -> dict[str, int]:
    return self._vote_count

  def partial_state(
      self,
      player_name: str,
  ) -> str:
    """Return a player-specific view of the construct's state."""
    return self._state

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    if not self._polls_open:
      update_log = {
          'date': self._clock_now(),
          'Summary': 'Polls are not open.',
          'Vote count': str(self._vote_count),
      }
      self._history.append(update_log)
      return

    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement(event_statement)
    chain_of_thought.statement(f'List of citizens: {self._voter_names}')
    active_voter_id = chain_of_thought.multiple_choice_question(
        question='In the above transcript, which citizen took an action?',
        answers=self._voter_names,
    )
    vote = None
    active_voter = self._voter_names[active_voter_id]
    if active_voter not in self._citizens_who_already_voted:
      did_vote = chain_of_thought.yes_no_question(
          question=f'Did {active_voter} vote in the above transcript?'
      )
      if did_vote:
        question = (
            f'Current activity: {event_statement}.\nGiven the above, who whould'
            f' {active_voter} vote for?'
        )
        action_spec = agent.choice_action_spec(
            call_to_action=question,
            options=self._candidates,
            tag='vote',
        )
        vote = self._voter_by_name[active_voter].act(action_spec)
        action_spec.validate(vote)

        self._vote_count[vote] += 1
        self._citizens_who_already_voted.add(active_voter)
        self._memory.add(
            f'{active_voter} voted for {vote}', tags=['election tracker']
        )
        if self._verbose:
          print(
              termcolor.colored(
                  f'\n {active_voter} voted for {vote}\n', 'magenta'
              )
          )
      else:
        if self._verbose:
          print(
              termcolor.colored(
                  f'\n {active_voter} did not vote in the transcript.\n',
                  'magenta',
              )
          )
    else:
      chain_of_thought.statement(f'{active_voter} already voted.')

    update_log = {
        'date': self._clock_now(),
        'Summary': str(self._vote_count),
        'Vote count': str(self._vote_count),
        'Chain of thought': {
            'Summary': 'Election tracker chain of thought',
            'Chain': chain_of_thought.view().text().splitlines(),
        },
    }
    self._history.append(update_log)

    if self._verbose:
      print(
          termcolor.colored(
              f'{self._vote_count}\n' + chain_of_thought.view().text(),
              'magenta',
          )
      )

    if self._measurements is not None and vote is not None:
      answer = self._vote_count[vote]
      answer_str = str(answer)
      datum = {
          'time_str': self._clock_now().strftime('%H:%M:%S'),
          'timestep': self._timestep,
          'value_float': answer,
          'value_str': answer_str,
          'player': vote,
      }
      self._measurements.publish_datum(channel=self._channel, datum=datum)
      self._timestep += 1
