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

from concordia.agents import basic_agent
from concordia.agents import entity_agent
from concordia.associative_memory import associative_memory
from concordia.components.game_master import current_scene
from concordia.typing import component

MemoryT = associative_memory.AssociativeMemory
PlayersT = Sequence[basic_agent.BasicAgent | entity_agent.EntityAgent]


@dataclasses.dataclass
class PreEventFnArgsT:
  """A specification of the arguments to a pre-event function.

  Attributes:
    player_name: The name of the player.
    player_choice: The choice of the player on the current timestep.
    current_scene_type: The type of the current scene.
    players: Sequence of player objects.
    memory: The game master's associative memory.
  """

  player_name: str
  player_choice: str
  current_scene_type: str
  players: PlayersT
  memory: MemoryT


@dataclasses.dataclass
class PostEventFnArgsT:
  """A specification of the arguments to a post-event function.

  Attributes:
    event_statement: The event that resulted from the player's choice.
    current_scene_type: The type of the current scene.
    players: Sequence of player objects.
    memory: The game master's associative memory.
  """

  event_statement: str
  current_scene_type: str
  players: PlayersT
  memory: MemoryT


class TriggeredFunction(component.Component):
  """A component to modify inventories based on events."""

  def __init__(
      self,
      memory: MemoryT,
      players: PlayersT,
      clock_now: Callable[[], datetime.datetime],
      pre_event_fn: Callable[[PreEventFnArgsT], None] | None = None,
      post_event_fn: Callable[[PostEventFnArgsT], None] | None = None,
      name: str = '    \n',
      verbose: bool = False,
  ):
    """Initialize a component to track how events change inventories.

    Args:
      memory: an associative memory
      players: sequence of players who have an inventory and will observe it.
      clock_now: Function to call to get current time.
      pre_event_fn: function to call with the action attempt before
        computing the event.
      post_event_fn: function to call with the event statement.
      name: the name of this component e.g. Possessions, Account, Property, etc
      verbose: whether to print the full update chain of thought or not
    """
    self._verbose = verbose

    self._memory = memory
    self._name = name
    self._clock_now = clock_now

    self._pre_event_fn = pre_event_fn
    self._post_event_fn = post_event_fn
    self._players = players

    self._current_scene = current_scene.CurrentScene(
        name='current scene type',
        memory=self._memory,
        clock_now=self._clock_now,
        verbose=self._verbose,
    )

  def name(self) -> str:
    """Returns the name of this component."""
    return self._name

  def state(self) -> str:
    return ''

  def update(self) -> None:
    self._current_scene.update()

  def update_before_event(self, player_action_attempt: str) -> None:
    if self._pre_event_fn is None:
      return
    player_name, choice = player_action_attempt.split(': ')
    if player_name not in [player.name for player in self._players]:
      return
    current_scene_type = self._current_scene.state()
    self._pre_event_fn(
        PreEventFnArgsT(player_name=player_name,
                        player_choice=choice,
                        current_scene_type=current_scene_type,
                        players=self._players,
                        memory=self._memory)
    )

  def update_after_event(self, event_statement: str) -> None:
    if self._post_event_fn is None:
      return
    current_scene_type = self._current_scene.state()
    self._post_event_fn(
        PostEventFnArgsT(event_statement=event_statement,
                         current_scene_type=current_scene_type,
                         players=self._players,
                         memory=self._memory)
    )
