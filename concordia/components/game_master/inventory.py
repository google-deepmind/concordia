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

"""A game master component to represent each player's inventory."""

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import datetime
import functools
import threading

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.agent import observation as observation_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
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


class Inventory(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A grounded inventory tracking amounts of items in python."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      item_type_configs: Sequence[ItemTypeConfig],
      player_initial_endowments: dict[str, dict[str, float]],
      clock_now: Callable[[], datetime.datetime],
      observations_component_name: str = (
          observation_component.DEFAULT_OBSERVATION_COMPONENT_KEY),
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      chain_of_thought_prefix: str = _DEFAULT_CHAIN_OF_THOUGHT_PREFIX,
      financial: bool = False,
      never_increase: bool = False,
      pre_act_label: str = 'Inventory',
      verbose: bool = False,
  ):
    """Initialize a grounded inventory component tracking objects in python.

    Args:
      model: a language model
      item_type_configs: sequence of item type configurations
      player_initial_endowments: dict mapping player name to a dictionary with
        item types as keys and initial endownments as values.
      clock_now: Function to call to get current time.
      observations_component_name: The name of the component that contains the
        observations.
      memory_component_name: The name of the memory component.
      chain_of_thought_prefix: include this string in context before all
        reasoning steps for handling the inventory.
      financial: If set to True then include special questions to handle the
        fact that agents typically say "Alice bought (or sold) X" which is a
        different way of speaking than "Alice exchanged X for Y".
      never_increase: If set to True then this component never increases the
        amount of any item. Events where an item would have been gained lead to
        no change in the inventory and the game master instead invents a reason
        for why the item was not gained.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
      verbose: whether to print the full update chain of thought or not
    """
    self._pre_act_label = pre_act_label
    self._model = model
    self._observations_component_name = observations_component_name
    self._memory_component_name = memory_component_name
    self._player_initial_endowments = player_initial_endowments
    self._chain_of_thought_prefix = chain_of_thought_prefix
    self._financial = financial
    self._clock_now = clock_now
    self._never_increase = never_increase
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

    # Determine if each item type is a count noun or a mass noun.
    self._is_count_noun = {}
    self._lock = threading.Lock()

    self._is_count_noun = concurrency.run_tasks({
        item_type: functools.partial(
            helper_functions.is_count_noun, item_type, self._model
        )
        for item_type in self._item_types
    })

  def _get_player_inventory_str(self, player_name: str) -> str:
    return f"{player_name}'s {self._pre_act_label}: " + str(
        self._inventories[player_name]
    )

  def _add_to_game_master_memory(
      self,
      message: str,
  ) -> None:
    """Add `message` to memory."""
    memory = self.get_entity().get_component(
        self._memory_component_name, type_=memory_component.Memory
    )
    memory.add(message)

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    display_chain_of_thought = ''
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      with self._lock:
        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.Memory)

        observations = (
            self.get_entity().get_component(
                self._observations_component_name,
                type_=action_spec_ignored.ActionSpecIgnored,
            ).get_pre_act_value()
        )

        chain_of_thought = interactive_document.InteractiveDocument(self._model)
        chain_of_thought.statement(self._chain_of_thought_prefix)
        chain_of_thought.statement(f'List of individuals: {self._player_names}')
        chain_of_thought.statement(f'List of item types: {self._item_types}')
        chain_of_thought.statement(f'Event: {observations}')

        inventory_effects = []

        proceed = chain_of_thought.yes_no_question(
            question=(
                'In the above transcript, did any of the listed individuals'
                ' gain or lose any items on the list of item types?  Make sure'
                ' to take into account items equivalent to the items on the'
                ' list e.g. if "money" is on the list but the event mentions'
                ' "gold" then treat "gold" as equivalent to "money" since gold'
                ' is a type of money.'
            )
        )
        if proceed:
          new_inventories = dict(self._inventories)
          if self._financial:
            _ = chain_of_thought.open_question(
                question=(
                    'If the event mentions any financial transaction (buying or'
                    ' selling), what price(s) were involved? If no price(s)'
                    ' were mentioned then pick logical values for them. If'
                    ' there was no transaction then respond with "NA".'
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
              players_whose_inventory_changed = players_who_changed_str.split(
                  ','
              )
              for player in players_whose_inventory_changed:
                formatted_player = player.lstrip(' ').rstrip(' ')
                if formatted_player in self._player_names:
                  new_inventories[formatted_player] = dict(
                      new_inventories[formatted_player]
                  )
                  prefix = (
                      f"[effect on {formatted_player}'s {self._pre_act_label}]"
                  )
                  many_or_much = _many_or_much_fn(
                      self._is_count_noun[item_type]
                  )
                  amount = chain_of_thought.open_question(
                      question=(
                          f'How {many_or_much} {item_type} did {player} gain'
                          f' as a result of the event? If they lost {item_type}'
                          ' then respond with a negative number. Be precise.'
                          ' If the original event was imprecise then pick a'
                          ' specific value that is consistent with all the text'
                          ' above. Respond in the format: "number|explanation".'
                      )
                  )
                  try:
                    if '|' in amount:
                      amount = amount.split('|')[0]
                    amount = float(amount)
                  except ValueError:
                    # Assume worst case, if player gained item, they gain 1
                    # unit. If player lost item, they lose all units they have.
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
                          f'{prefix} no effect since amount of {item_type} '
                          + f'must be a whole number but {amount} is not.'
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
                      self._add_to_game_master_memory(
                          message=effect
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
                            'What is the reason that the amount of'
                            f' {item_type} did not change despite the event'
                            ' suggesting that it would have? Be specific and'
                            ' consistent with the text above.'
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
                    self._add_to_game_master_memory(
                        message=reason_for_no_change
                    )
                    inventory_effects.append(reason_for_no_change)
                    if self._verbose:
                      print(termcolor.colored(reason_for_no_change, 'yellow'))
          self._inventories = new_inventories

        memory.extend(inventory_effects)

      display_chain_of_thought = chain_of_thought.view().text().splitlines()
      if self._verbose:
        print(termcolor.colored(chain_of_thought.view().text(), 'yellow'))

    self._logging_channel({
        'Key': self._pre_act_label,
        'Value': str(self._inventories),
        'Chain of thought': display_chain_of_thought,
    })
    return ''

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

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    with self._lock:
      self._inventories = state['inventories']

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {'inventories': self._inventories}


class Score(entity_component.ContextComponent,
            entity_component.ComponentWithLogging):
  """This component assigns score based on possession of items in inventory."""

  def __init__(
      self,
      inventory: Inventory,
      player_names: Sequence[str],
      targets: Mapping[str, Sequence[str]],
      pre_act_label: str = '   \n',
      verbose: bool = False,
  ):
    """Initialize a grounded inventory component tracking objects in python.

    Args:
      inventory: the inventory component to use to get the inventory of players.
      player_names: sequence of players who have an inventory.
      targets: Mapping of player name to their target items. They will be scored
        by the number of items of the specified types in their inventory.
      pre_act_label: the name of this component to use in pre_act.
      verbose: whether to print the full update chain of thought or not
    """
    self._pre_act_label = pre_act_label
    self._inventory = inventory
    self._player_names = player_names
    self._targets = targets
    self._verbose = verbose

  def get_scores(self) -> Mapping[str, float]:
    """Return the cumulative score for each player."""
    player_scores = {name: 0.0 for name in self._player_names}
    for name in self._player_names:
      inventory = self._inventory.get_player_inventory(name)
      targets = self._targets[name]
      for target in targets:
        if self._verbose:
          print(termcolor.colored(
              f'{name} -- target = {target}, inventory = {inventory}',
              'yellow'))
        if target in list(inventory.keys()) and inventory[target] > 0:
          if self._verbose:
            print(termcolor.colored('    target found in inventory.', 'yellow'))
          num_on_target = inventory[target]
          player_scores[name] += num_on_target

    return player_scores

  def pre_act(
      self,
      unused_action_spec: entity_lib.ActionSpec,
  ) -> str:
    del unused_action_spec
    self._logging_channel(
        {'Key': 'score based on inventory',
         'Value': self.get_scores()}
    )
    return ''

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._inventory.set_state(state['inventory'])

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {'inventory': self._inventory.get_state()}
