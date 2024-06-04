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


"""Externality for the Game Master, which tracks direct effect on players."""

from collections.abc import Callable, Sequence
import concurrent.futures
import datetime

from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
from concordia.utils import helper_functions
import termcolor


class DirectEffect(component.Component):
  """Tracks direct effect on players.

  A direct effect is an event that directly affects a player in the list of
  players.
  """

  def __init__(
      self,
      players: Sequence[basic_agent.BasicAgent],
      clock_now: Callable[[], datetime.datetime],
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      components: Sequence[component.Component] | None = None,
      verbose: bool = False,
      print_colour: str = 'magenta',
  ):
    self._players = players
    self._verbose = verbose
    self._print_colour = print_colour
    self._components = components or []
    self._clock_now = clock_now
    self._history = []
    self._model = model
    self._memory = memory

  def name(self) -> str:
    return 'Direct effects of the event on others'

  def _print(self, entry: str):
    print(termcolor.colored(entry, self._print_colour), end='')

  def get_player_names(self):
    return [player.name for player in self._players]

  def get_history(self):
    return self._history.copy()

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def get_components(self) -> Sequence[component.Component]:
    return self._components

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    document = interactive_document.InteractiveDocument(self._model)

    for construct in self._components:
      document.statement(construct.name() + ': ' + construct.state() + '\n')

    player_names = self.get_player_names()
    direct_effect_on_someone = document.yes_no_question(
        'Does the following event directly affect anyone from this '
        + f'list?\n List: {player_names}.\n Event: {event_statement}'
    )
    effect_unknown = []
    effect_known = []

    def _update_player(player):
      player_name = player.name
      player_doc = helper_functions.filter_copy_as_statement(document)
      affected = player_doc.yes_no_question(
          f'Does the event affect {player_name}?'
      )
      if affected:
        if self._verbose:
          self._print(f'\n{player_name} affected, might not know it.')
        known = player_doc.yes_no_question(
            f'Does {player_name} know about the event?'
        )
        if known:
          if self._verbose:
            self._print(f'\n{player_name} known.')
          _ = player_doc.open_question(
              f'What does {player_name} know about the event?',
              max_tokens=2500,
          )
          how_player_saw_event_first_person = player_doc.open_question(
              f"Summarize the event from {player_name}'s "
              + 'perspective using third-person limited point of view. '
              + 'If the event contains a direct quotation of anything said '
              + 'or written by anyone then it is important to include the '
              + 'quote verbatim in the summary.',
              max_tokens=2500,
          )
          player.observe(how_player_saw_event_first_person)
          if self._verbose:
            self._print(
                f'\nEffect on {player_name}:'
                f' {how_player_saw_event_first_person}'
            )
          effect_known.append(how_player_saw_event_first_person)
        else:  # not known
          if self._verbose:
            self._print(f'\n{player_name} not known.')
          effect_despite_ignorance = player_doc.open_question(
              f"How does the event affect {player_name}'s status, despite them"
              ' not knowing about it?',
              max_tokens=2500,
          )
          if self._verbose:
            self._print(
                f'\nUnknown effect on {player_name}: {effect_despite_ignorance}'
            )
          effect = f'[effect on {player_name}] {effect_despite_ignorance}'
          self._memory.add(effect)
          effect_unknown.append(effect)

    # Determined whether externality has happened
    # if yes, then propagate the event
    if direct_effect_on_someone:
      if self._verbose:
        self._print(
            '\nThe event had a direct effect on one of the players, resolving.'
        )

      with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(_update_player, self._players)

    update_log = {
        'date': self._clock_now(),
        'Event statement': event_statement,
        'Summary': f'The effect of "{event_statement}"',
        'Known effect': effect_known,
        'Unknown effect': effect_unknown,
        'Chain of thought': {
            'Summary': 'Direct effect chain of thought',
            'Chain': document.view().text().splitlines(),
        },
    }
    self._history.append(update_log)
