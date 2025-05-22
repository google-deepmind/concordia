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

"""A component to assign a score based on possession of certain items."""

from collections.abc import Mapping, Sequence

from concordia.agents.deprecated import deprecated_agent
from concordia.agents.deprecated import entity_agent
from concordia.components.game_master.deprecated import inventory as inventory_gm_component
from concordia.typing.deprecated import component
import termcolor


class Score(component.Component):
  """This component assigns score based on possession of items in inventory."""

  def __init__(
      self,
      inventory: inventory_gm_component.Inventory,
      players: Sequence[deprecated_agent.BasicAgent | entity_agent.EntityAgent],
      targets: Mapping[str, Sequence[str]],
      name: str = '   \n',
      verbose: bool = False,
  ):
    """Initialize a grounded inventory component tracking objects in python.

    Args:
      inventory: the inventory component to use to get the inventory of players.
      players: sequence of players who have an inventory and will observe it.
      targets: Mapping of player name to their target items. They will be scored
        by the number of items of the specified types in their inventory. 
      name: the name of this component e.g. Possessions, Account, Property, etc
      verbose: whether to print the full update chain of thought or not
    """
    self._inventory = inventory
    self._players = players
    self._targets = targets
    self._name = name
    self._verbose = verbose

  def name(self) -> str:
    """Returns the name of this component."""
    return self._name

  def state(self) -> str:
    return ''

  def update(self) -> None:
    pass

  def get_scores(self) -> Mapping[str, float]:
    """Return the cumulative score for each player."""
    player_scores = {player.name: 0.0 for player in self._players}
    for player in self._players:
      inventory = self._inventory.get_player_inventory(player.name)
      targets = self._targets[player.name]
      for target in targets:
        if self._verbose:
          print(termcolor.colored(
              f'{player.name} -- target = {target}, inventory = {inventory}',
              'yellow'))
        if target in list(inventory.keys()) and inventory[target] > 0:
          if self._verbose:
            print(termcolor.colored('    target found in inventory.', 'yellow'))
          num_on_target = inventory[target]
          player_scores[player.name] += num_on_target

    return player_scores
