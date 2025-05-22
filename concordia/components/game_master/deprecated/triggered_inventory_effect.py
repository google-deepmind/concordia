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

"""A component to modify inventories based on events."""

from collections.abc import Callable, Sequence
import dataclasses
import datetime
import functools

from concordia.agents.deprecated import deprecated_agent
from concordia.agents.deprecated import entity_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.components.game_master.deprecated import current_scene
from concordia.components.game_master.deprecated import inventory as inventory_gm_component
from concordia.typing.deprecated import component

MemoryT = associative_memory.AssociativeMemory
PlayerT = deprecated_agent.BasicAgent | entity_agent.EntityAgent
PlayersT = Sequence[PlayerT]


@dataclasses.dataclass
class PreEventFnArgsT:
  """A specification of the arguments to a pre-event function.

  Attributes:
    player_name: The name of the player.
    player_choice: The choice of the player on the current timestep.
    current_scene_type: The type of the current scene.
    memory: The game master's associative memory.
    player: Player object for the acting player.
  """

  player_name: str
  player_choice: str
  current_scene_type: str
  memory: MemoryT
  player: PlayerT


def _get_player_by_name(player_name: str, players: PlayersT) -> PlayerT | None:
  """Get a player object by name. Assumes no duplicate names."""
  for player in players:
    if player.name == player_name:
      return player
  return None


class TriggeredInventoryEffect(component.Component):
  """A component to modify inventories based on events."""

  def __init__(
      self,
      function: Callable[
          [PreEventFnArgsT, inventory_gm_component.InventoryType],
          inventory_gm_component.InventoryType,
      ],
      inventory: inventory_gm_component.Inventory,
      memory: associative_memory.AssociativeMemory,
      players: PlayersT,
      clock_now: Callable[[], datetime.datetime],
      name: str = '   \n',
      verbose: bool = False,
  ):
    """Initialize a component to track how events change inventories.

    Args:
      function: user-provided function that can modify the inventory based on an
        action attempt.
      inventory: the inventory component to use to get the inventory of players.
      memory: an associative memory
      players: sequence of players who can trigger an inventory event.
      clock_now: Function to call to get current time.
      name: the name of this component e.g. Possessions, Account, Property, etc
      verbose: whether to print the full update chain of thought or not
    """
    self._verbose = verbose

    self._memory = memory
    self._name = name
    self._clock_now = clock_now

    self._function = function
    self._inventory = inventory
    self._players = players

    self._current_scene = current_scene.CurrentScene(
        name='current scene type',
        memory=self._memory,
        clock_now=self._clock_now,
        verbose=self._verbose,
    )

    self._latest_update_log = None

  def name(self) -> str:
    """Returns the name of this component."""
    return self._name

  def state(self) -> str:
    return ''

  def update(self) -> None:
    self._current_scene.update()

  def update_before_event(self, player_action_attempt: str) -> None:

    # we assume that the player action attempt is in the format
    # 'player_name: player_choice'. All other occurrences of ':' will be treated
    # as a part of the player choice.
    player_name, choice = player_action_attempt.split(': ', 1)
    if player_name not in [player.name for player in self._players]:
      return
    current_scene_type = self._current_scene.state()
    player = _get_player_by_name(player_name, self._players)
    update = functools.partial(
        self._function,
        PreEventFnArgsT(
            player_name=player_name,
            player_choice=choice,
            current_scene_type=current_scene_type,
            memory=self._memory,
            player=player,
        ),
    )
    self._inventory.apply(update)
    self._latest_update_log = {
        'date': self._clock_now(),
        'Summary': self.name(),
        'Current scene type': current_scene_type,
        'current active player': player_name,
    }

  def get_last_log(self):
    if self._latest_update_log is not None:
      return self._latest_update_log
