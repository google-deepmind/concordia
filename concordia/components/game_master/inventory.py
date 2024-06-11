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

"""A component to represent each agent's inventory or possessions."""

from collections.abc import Callable, Sequence
import concurrent
import dataclasses
import datetime

from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
from concordia.utils import helper_functions
import numpy as np
import termcolor


_DEFAULT_QUANTITY = 0


@dataclasses.dataclass(frozen=True)
class ItemTypeConfig:
  """Class for configuring a type of item to track in an Inventory."""

  name: str
  minimum: float = -np.inf
  maximum: float = np.inf
  force_integer: bool = False


def _many_or_much_fn(is_count_noun: bool) -> str:
  """Return 'many' if input is True and 'much' if input is False."""
  if is_count_noun:
    return 'many'
  else:
    return 'much'


class Inventory(component.Component):
  """A grounded inventory tracking amounts of items in python."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      item_type_configs: Sequence[ItemTypeConfig],
      players: Sequence[basic_agent.BasicAgent],
      player_initial_endowments: dict[str, dict[str, float]],
      clock_now: Callable[[], datetime.datetime],
      financial: bool = False,
      name: str = 'Inventory',
      verbose: bool = False,
  ):
    """Initialize a grounded inventory component tracking objects in python.

    Args:
      model: a language model
      memory: an associative memory
      item_type_configs: sequence of item type configurations
      players: sequence of players who have an inventory and will observe it.
      player_initial_endowments: dict mapping player name to a dictionary with
        item types as keys and initial endownments as values.
      clock_now: Function to call to get current time.
      financial: If set to True then include special questions to handle the
        fact that agents typically say "Alice bought (or sold) X" which is
        a different way of speaking than "Alice exchanged X for Y".
      name: the name of this component e.g. Possessions, Account, Property, etc
      verbose: whether to print the full update chain of thought or not
    """
    self._model = model
    self._memory = memory
    self._players = players
    self._player_initial_endowments = player_initial_endowments
    self._financial = financial
    self._clock_now = clock_now
    self._name = name
    self._verbose = verbose

    self._item_types = [config.name for config in item_type_configs]
    self._item_types_dict = {
        config.name: config for config in item_type_configs
    }
    self._player_names = list(player_initial_endowments.keys())

    self._inventories = {}
    for player_name, endowment in player_initial_endowments.items():
      self._inventories[player_name] = {
          item_type: endowment.get(item_type, _DEFAULT_QUANTITY)
          for item_type in self._item_types
      }

    self._history = []
    self._state = ''
    self._partial_states = {name: '' for name in self._player_names}

    # Determine if each item type is a count noun or a mass noun.
    self._is_count_noun = {}

    def check_if_count_noun(item_type):
      self._is_count_noun[item_type] = helper_functions.is_count_noun(
          item_type, self._model
      )
      return

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(self._item_types)
    ) as executor:
      executor.map(check_if_count_noun, self._item_types)

    # Set the initial state's string representation.
    self.update()

  def name(self) -> str:
    """Returns the name of this component."""
    return self._name

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def get_history(self):
    return self._history.copy()

  def _get_player_inventory_str(self, player_name: str) -> str:
    return f"{player_name}'s {self._name}: " + str(
        self._inventories[player_name]
    )

  def state(self) -> str:
    return self._state

  def update(self) -> None:
    self._state = '\n'.join(
        [self._get_player_inventory_str(name) for name in self._player_names]
    )
    self._partial_states = {
        name: self._get_player_inventory_str(name)
        for name in self._player_names
    }
    # Explicitly pass partial states to agents here in `update` instead of
    # relying on the game master to call partial state on all players. This is
    # because we frequently have supporting characters who participate in
    # conversations but do not take active turns with the top-level game master
    # themselves. This method of passing the partial state information ensures
    # that theses players still get to observe their inventory.
    for player in self._players:
      player.observe(self._partial_states[player.name])

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement(f'List of individuals: {self._player_names}')
    chain_of_thought.statement(f'List of item types: {self._item_types}')
    chain_of_thought.statement(f'Event: {event_statement}')

    inventory_effects = []

    proceed = chain_of_thought.yes_no_question(
        question=(
            'In the above transcript, did any of the listed individuals '
            + 'gain or lose any items on the list of item types?  Make sure '
            + 'to take into account items equivalent to the items on the list '
            + 'e.g. if "money" is on the list but the event mentions "gold" '
            + 'then treat "gold" as equivalent to "money" since gold is a type '
            + 'of money.'
        )
    )
    if proceed:
      if self._financial:
        _ = chain_of_thought.open_question(
            question=(
                'If the event mentions any financial transaction (buying or '
                'selling), what price(s) were involved? If no price(s) were '
                'mentioned then pick logical values for them. If there was no '
                'transaction then respond with "NA".'
            )
        )
      for item_type in self._item_types:
        this_item_changed = chain_of_thought.yes_no_question(
            question=f'Did any listed individual gain or lose {item_type}?',
        )
        if this_item_changed:
          players_who_changed_str = chain_of_thought.open_question(
              question=(
                  f'Which individuals gained or lost {item_type}?\n'
                  + 'Respond with a comma-separated list, for example: \n'
                  + 'Jacob,Alfred,Patricia. Note that transactions between '
                  + 'named individuals must be balanced. If someone gained '
                  + 'something then someone else must have lost it.'
              )
          )
          players_whose_inventory_changed = players_who_changed_str.split(',')
          for player in players_whose_inventory_changed:
            formatted_player = player.lstrip(' ').rstrip(' ')
            if formatted_player in self._player_names:
              prefix = f"[effect on {formatted_player}'s {self._name}]"
              many_or_much = _many_or_much_fn(self._is_count_noun[item_type])
              amount = chain_of_thought.open_question(
                  question=(
                      f'How {many_or_much} '
                      + f'{item_type} did {player} gain '
                      + f'as a result of the event? If they lost {item_type} '
                      + 'then respond with a negative number. Be precise. If '
                      + 'the original event was imprecise then pick a specific '
                      + 'value that is consistent with all the text above. '
                      + 'Respond in the format: "number|explanation".'
                  )
              )
              try:
                if '|' in amount:
                  amount = amount.split('|')[0]
                amount = float(amount)
              except ValueError:
                # Assume worst case, if player gained item, they gain 1 unit. If
                # player lost item, they lose all units they have.
                increased = chain_of_thought.yes_no_question(
                    question=(f'Did the amount of {item_type} possessed '
                              f'by {player} increase?'))
                if increased:
                  amount = 1.0
                else:
                  amount = -self._inventories[player][item_type]

              if self._item_types_dict[item_type].force_integer:
                if not float(amount).is_integer():
                  inventory_effects.append(
                      f'{prefix} no effect since amount of {item_type} must '
                      + f'be a whole number but {amount} is not.'
                  )
                  continue

              old_total = self._inventories[formatted_player][item_type]
              self._inventories[formatted_player][item_type] += amount
              maximum = self._item_types_dict[item_type].maximum
              minimum = self._item_types_dict[item_type].minimum
              self._inventories[formatted_player][item_type] = np.min(
                  [self._inventories[formatted_player][item_type], maximum]
              )
              self._inventories[formatted_player][item_type] = np.max(
                  [self._inventories[formatted_player][item_type], minimum]
              )
              # Get amount actually gained/lost once bounds accounted for.
              amount = (
                  self._inventories[formatted_player][item_type] - old_total)
              effect = ''
              if amount > 0:
                effect = f'{prefix} gained {amount} {item_type}'
              if amount < 0:
                absolute_amount = np.abs(amount)
                effect = f'{prefix} lost {absolute_amount} {item_type}'
              if effect:
                if self._is_count_noun[item_type] and np.abs(amount) > 1:
                  # Add 's' to the end of the noun if it is a count noun.
                  effect = effect + 's'
                inventory_effects.append(effect)
                if self._verbose:
                  print(termcolor.colored(effect, 'yellow'))

    # Update the string representation of all inventories.
    self.update()

    if self._verbose:
      print(termcolor.colored(chain_of_thought.view().text(), 'yellow'))
      print(termcolor.colored(self.state(), 'yellow'))

    update_log = {
        'date': self._clock_now(),
        'Summary': str(self._inventories),
        'Inventories': self.state(),
        'Chain of thought': {
            'Summary': f'{self._name} chain of thought',
            'Chain': chain_of_thought.view().text().splitlines(),
        },
    }
    self._memory.extend(inventory_effects)
    self._history.append(update_log)

  def get_player_inventory(self, player_name: str) -> dict[str, float | int]:
    """Return the inventory of player `player_name`."""
    return self._inventories[player_name]
