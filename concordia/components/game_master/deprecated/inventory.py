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

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import datetime
import functools
import threading

from concordia.agents.deprecated import deprecated_agent
from concordia.agents.deprecated import entity_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import component
from concordia.utils import concurrency
from concordia.utils import helper_functions
import numpy as np
import termcolor


_DEFAULT_CHAIN_OF_THOUGHT_PREFIX = (
    'This is a social science experiment. It is structured as a '
    'tabletop roleplaying game. You are the game master and storyteller. '
    'Your job is to make sure the game runs smoothly and accurately tracks '
    'the state of the world, subject to the laws of logic and physics. Next, '
    'you will be asked a series of questions to help you reason through '
    'whether a specific event should be deemed as having caused a change in '
    'the number or amount of items possessed or owned by specific individuals. '
    'Never mention that it is a game. Always use third-person limited '
    'perspective, even when speaking directly to the participants.'
)

_DEFAULT_QUANTITY = 0

InventoryType = Mapping[str, dict[str, float]]


@dataclasses.dataclass(frozen=True)
class ItemTypeConfig:
  """Class for configuring a type of item to track in an Inventory."""

  name: str
  minimum: float = -np.inf
  maximum: float = np.inf
  force_integer: bool = False

  def check_valid(self, amount: float) -> None:
    """Checks if amount is valid for this item type."""
    if amount < self.minimum or amount > self.maximum:
      raise ValueError('Amount out of bounds')
    if self.force_integer and amount != int(amount):
      raise ValueError('Amount not right type')


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
      players: Sequence[deprecated_agent.BasicAgent | entity_agent.EntityAgent],
      player_initial_endowments: dict[str, dict[str, float]],
      clock_now: Callable[[], datetime.datetime],
      chain_of_thought_prefix: str = _DEFAULT_CHAIN_OF_THOUGHT_PREFIX,
      financial: bool = False,
      never_increase: bool = False,
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
      chain_of_thought_prefix: include this string in context before all
        reasoning steps for handling the inventory.
      financial: If set to True then include special questions to handle the
        fact that agents typically say "Alice bought (or sold) X" which is a
        different way of speaking than "Alice exchanged X for Y".
      never_increase: If set to True then this component never increases the
        amount of any item. Events where an item would have been gained lead to
        no change in the inventory and the game master instead invents a reason
        for why the item was not gained.
      name: the name of this component e.g. Possessions, Account, Property, etc
      verbose: whether to print the full update chain of thought or not
    """
    self._model = model
    self._memory = memory
    self._players = players
    self._player_initial_endowments = player_initial_endowments
    self._chain_of_thought_prefix = chain_of_thought_prefix
    self._financial = financial
    self._clock_now = clock_now
    self._never_increase = never_increase
    self._name = name
    self._verbose = verbose

    self._item_types = [config.name for config in item_type_configs]
    self._item_types_dict = {
        config.name: config for config in item_type_configs
    }
    self._player_names = list(player_initial_endowments.keys())
    self._names_to_players = {player.name: player for player in self._players}

    self._inventories = {}
    for player_name, endowment in player_initial_endowments.items():
      self._inventories[player_name] = {
          item_type: endowment.get(item_type, _DEFAULT_QUANTITY)
          for item_type in self._item_types
      }

    self._latest_update_log = None
    self._state = ''
    self._partial_states = {name: '' for name in self._player_names}

    # Determine if each item type is a count noun or a mass noun.
    self._is_count_noun = {}
    self._lock = threading.Lock()

    self._is_count_noun = concurrency.run_tasks({
        item_type: functools.partial(
            helper_functions.is_count_noun, item_type, self._model
        )
        for item_type in self._item_types
    })

    # Set the initial state's string representation.
    self.update()

  def name(self) -> str:
    """Returns the name of this component."""
    return self._name

  def get_last_log(self):
    if self._latest_update_log is not None:
      return self._latest_update_log

  def _get_player_inventory_str(self, player_name: str) -> str:
    return f"{player_name}'s {self._name}: " + str(
        self._inventories[player_name]
    )

  def _send_message_to_player_and_game_master(
      self, player_name: str, message: str
  ) -> None:
    """Send `message` to player `player_name`."""
    player = self._names_to_players[player_name]
    player.observe(message)  # pytype: disable=attribute-error
    self._memory.add(message)

  def state(self) -> str:
    return self._state

  def update(self) -> None:
    with self._lock:
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
      # conversations but do not take active turns with the top-level game
      # master themselves. This method of passing the partial state information
      # ensures that theses players still get to observe their inventory.
      for player in self._players:
        player.observe(self._partial_states[player.name])

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    with self._lock:
      chain_of_thought = interactive_document.InteractiveDocument(self._model)
      chain_of_thought.statement(self._chain_of_thought_prefix)
      chain_of_thought.statement(f'List of individuals: {self._player_names}')
      chain_of_thought.statement(f'List of item types: {self._item_types}')
      chain_of_thought.statement(f'Event: {event_statement}')

      inventory_effects = []

      proceed = chain_of_thought.yes_no_question(
          question=(
              'In the above transcript, did any of the listed individuals '
              'gain or lose any items on the list of item types?  Make sure '
              'to take into account items equivalent to the items on the list '
              'e.g. if "money" is on the list but the event mentions "gold" '
              'then treat "gold" as equivalent to "money" since gold is a type '
              'of money.'
          )
      )
      if proceed:
        new_inventories = dict(self._inventories)
        if self._financial:
          _ = chain_of_thought.open_question(
              question=(
                  'If the event mentions any financial transaction (buying or'
                  ' selling), what price(s) were involved? If no price(s) were'
                  ' mentioned then pick logical values for them. If there was'
                  ' no transaction then respond with "NA".'
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
                new_inventories[formatted_player] = dict(
                    new_inventories[formatted_player]
                )
                prefix = f"[effect on {formatted_player}'s {self._name}]"
                many_or_much = _many_or_much_fn(self._is_count_noun[item_type])
                amount = chain_of_thought.open_question(
                    question=(
                        f'How {many_or_much} '
                        + f'{item_type} did {player} gain '
                        + f'as a result of the event? If they lost {item_type} '
                        + 'then respond with a negative number. Be precise. If '
                        + 'the original event was imprecise then pick a'
                        ' specific '
                        + 'value that is consistent with all the text above. '
                        + 'Respond in the format: "number|explanation".'
                    )
                )
                try:
                  if '|' in amount:
                    amount = amount.split('|')[0]
                  amount = float(amount)
                except ValueError:
                  # Assume worst case, if player gained item, they gain 1 unit.
                  # If player lost item, they lose all units they have.
                  increased = chain_of_thought.yes_no_question(
                      question=(
                          f'Did the amount of {item_type} possessed '
                          f'by {player} increase?'
                      )
                  )
                  if increased:
                    amount = 1.0
                  else:
                    amount = -new_inventories[player][item_type]

                if self._item_types_dict[item_type].force_integer:
                  if not float(amount).is_integer():
                    inventory_effects.append(
                        f'{prefix} no effect since amount of {item_type} must '
                        + f'be a whole number but {amount} is not.'
                    )
                    continue

                if amount < 0 or not self._never_increase:
                  maximum = self._item_types_dict[item_type].maximum
                  minimum = self._item_types_dict[item_type].minimum

                  old_total = new_inventories[formatted_player][item_type]
                  new_inventories[formatted_player][item_type] += amount
                  new_inventories[formatted_player][item_type] = np.min(
                      [new_inventories[formatted_player][item_type], maximum]
                  )
                  new_inventories[formatted_player][item_type] = np.max(
                      [new_inventories[formatted_player][item_type], minimum]
                  )
                  # Get amount actually gained/lost once bounds accounted for.
                  amount = (
                      new_inventories[formatted_player][item_type] - old_total
                  )
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
                    self._send_message_to_player_and_game_master(
                        player_name=formatted_player, message=effect
                    )
                    if self._verbose:
                      print(termcolor.colored(effect, 'yellow'))
                else:
                  chain_of_thought.statement(
                      f'So {formatted_player} would have gained '
                      f'{amount} {item_type}.'
                  )
                  chain_of_thought.statement(
                      'However, the rules indicate that the amount of'
                      f' {item_type} cannot change. Therefore, it will not'
                      ' change. The job of the game master is to invent a'
                      ' reason why the events that appeared to increase'
                      f' {item_type} did not actually happen or did not cause'
                      ' the amount to change after all.'
                  )
                  reason_for_no_change_clause = chain_of_thought.open_question(
                      question=(
                          f'What is the reason that the amount of {item_type} '
                          'did not change despite the event suggesting that it '
                          'would have? Be specific and consistent with the '
                          'text above.'
                      ),
                      answer_prefix=(
                          f'The reason {formatted_player} did not gain any '
                          f'{item_type} is '
                      ),
                  )
                  reason_for_no_change = (
                      f'However, {formatted_player} did not gain any '
                      f'{item_type} because {reason_for_no_change_clause}'
                  )
                  self._send_message_to_player_and_game_master(
                      player_name=formatted_player, message=reason_for_no_change
                  )
                  inventory_effects.append(reason_for_no_change)
                  if self._verbose:
                    print(termcolor.colored(reason_for_no_change, 'yellow'))
        self._inventories = new_inventories

    # Update the string representation of all inventories.
    self.update()

    if self._verbose:
      print(termcolor.colored(chain_of_thought.view().text(), 'yellow'))
      print(termcolor.colored(self.state(), 'yellow'))

    self._latest_update_log = {
        'date': self._clock_now(),
        'Summary': str(self._inventories),
        'Inventories': self.state(),
        'Chain of thought': {
            'Summary': f'{self._name} chain of thought',
            'Chain': chain_of_thought.view().text().splitlines(),
        },
    }
    self._memory.extend(inventory_effects)

  def get_player_inventory(self, player_name: str) -> Mapping[str, float | int]:
    """Return the inventory of player `player_name`."""
    with self._lock:
      return copy.deepcopy(self._inventories[player_name])

  def apply(self, fn: Callable[[InventoryType], InventoryType]) -> None:
    """Apply `function` to `args` and update the inventory accordingly."""
    with self._lock:
      old_inventories = copy.deepcopy(self._inventories)
      new_inventories = fn(old_inventories)
      if set(new_inventories.keys()) != set(old_inventories.keys()):
        raise RuntimeError(
            'New inventory keys do not match old inventory keys.'
        )
      for item_type, config in self._item_types_dict.items():
        for key in new_inventories:
          missing = set(self._item_types_dict) - set(new_inventories[key])
          if missing:
            raise RuntimeError(f'{key} inventory missing {missing} types.')
          invalid = set(new_inventories[key]) - set(self._item_types_dict)
          if invalid:
            raise RuntimeError(f'{key} inventory has invalid {invalid} types.')
          config.check_valid(new_inventories[key][item_type])
      self._inventories = copy.deepcopy(new_inventories)
