# Copyright 2026 DeepMind Technologies Limited.
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

"""A game master component that keeps the players: who joins, who leaves.

The roster holds the list of player entities that the simulation plays: the
`entities` a game master prefab is given, which the engine runs. Players join
the game, built from a prefab, and leave it only through the roster, and the
game master's other components find a player by name here. Nothing else is
needed of the simulation or the engine: the list itself changes, so an engine
that runs it sees the change (the asynchronous engine starts a loop for a
player who joins and ends the loop of one who leaves).

  roster = player_roster.PlayerRoster(
      players=entities,  # the game master prefab's `self.entities`
      player_prefabs={'person': person_prefab},
      model=model,
      embedder=embedder,
  )
  roster.join('person', {'name': 'Ada'})
  ada = roster.get_player_by_name('Ada')
  roster.leave('Ada')

A player who joins is built from its prefab with the prefab's params updated
by the player's own: what all players of a prefab share (e.g. measurements)
stays on the prefab, and the player's own params (its name among them) must be
JSON-serializable, since the roster's state keeps them.

A checkpoint of a `generic.Simulation` keeps the players it began with. The
roster's state keeps the rest: who joined and is still playing (the prefab,
params and state of each) and which of the first players left. Restored, the
roster takes away the first players who left and rebuilds those who joined.
"""

from collections.abc import Callable, Mapping, MutableSequence
import copy
import dataclasses
import json
import threading
from typing import Any, cast

from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
import numpy as np

DEFAULT_PLAYER_ROSTER_COMPONENT_KEY = '__player_roster__'


class PlayerRoster(entity_component.ContextComponent):
  """The players of a game, and the only way they join and leave it."""

  def __init__(
      self,
      players: MutableSequence[entity_lib.Entity],
      player_prefabs: Mapping[str, prefab_lib.Prefab] | None = None,
      model: language_model.LanguageModel | None = None,
      embedder: Callable[[str], np.ndarray] | None = None,
  ):
    """Initializes the roster.

    Args:
      players: The list of player entities the simulation plays (a game master
        prefab's `entities`), changed in place as players join and leave. The
        players in it now are the game's first players.
      player_prefabs: The prefabs players may join as, by name. Their params
        are those all players of the prefab share.
      model: The language model players who join are built with.
      embedder: The sentence embedder of the memory of players who join.
    """
    super().__init__()
    self._players = players
    self._player_prefabs = dict(player_prefabs or {})
    self._model = model
    self._embedder = embedder
    self._lock = threading.RLock()
    # The first players' names are never reused: a simulation's checkpoint
    # keeps the first players under their names.
    self._first = frozenset(player.name for player in players)
    # Each player who joined and plays still: the prefab and params it was
    # built from, in the order they joined.
    self._joined: dict[str, dict[str, Any]] = {}
    # The first players who have left, in the order they left.
    self._left: list[str] = []

  def get_player_by_name(self, name: str) -> entity_lib.Entity:
    """Returns the player of a name.

    Args:
      name: The player's name.

    Raises:
      KeyError: If no player of that name plays.
    """
    with self._lock:
      player = self._find(name)
    if player is None:
      raise KeyError(f'No player named {name} plays.')
    return player

  def get_player_names(self) -> list[str]:
    """Returns the names of the players, in the order they play."""
    with self._lock:
      return [player.name for player in self._players]

  def join(self, prefab: str, params: Mapping[str, Any]) -> entity_lib.Entity:
    """Builds a player of a prefab and adds it to the game.

    Args:
      prefab: The name of the player's prefab (a key of `player_prefabs`).
      params: The player's own params, which update the prefab's. They must
        be JSON-serializable.

    Returns:
      The player who joined.

    Raises:
      KeyError: If there is no such prefab.
      TypeError: If the params are not JSON-serializable.
      ValueError: If the roster cannot build players, or if the player's name
        is taken (by a player who plays, or by one of the first players).
    """
    own = json.loads(json.dumps(dict(params)))
    with self._lock:
      player = self._build(prefab, own)
      if self._find(player.name) is not None or player.name in self._first:
        raise ValueError(f'The name {player.name} is taken.')
      self._players.append(player)
      self._joined[player.name] = {'prefab': prefab, 'params': own}
      return player

  def leave(self, name: str) -> entity_lib.Entity:
    """Takes a player out of the game.

    Args:
      name: The player's name.

    Returns:
      The player who left (e.g. to keep a record of it).

    Raises:
      KeyError: If no player of that name plays.
    """
    with self._lock:
      player = self.get_player_by_name(name)
      self._players.remove(player)
      if self._joined.pop(name, None) is None:
        self._left.append(name)
      return player

  def get_state(self) -> entity_component.ComponentState:
    """Returns who joined (with each one's state) and who left."""
    with self._lock:
      joined = {}
      for name, built_from in self._joined.items():
        player = self._find(name)
        state = (
            player.get_state()
            if isinstance(player, entity_component.EntityWithComponents)
            else {}
        )
        joined[name] = {**copy.deepcopy(built_from), 'state': state}
      return {'joined': joined, 'left': list(self._left)}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Makes the players those of the state.

    The first players who left are taken away, players who joined since the
    state was taken leave, and players who joined before it are rebuilt (if
    they no longer play) and given their states. The first players' own
    states are not the roster's: a simulation's checkpoint restores them.

    Args:
      state: A state from `get_state`.
    """
    saved = cast(Mapping[str, Any], state)
    left = [str(name) for name in saved.get('left', ())]
    joined: Mapping[str, Any] = saved.get('joined', {})
    with self._lock:
      for name in left + [n for n in self._joined if n not in joined]:
        player = self._find(name)
        if player is not None:
          self._players.remove(player)
      self._left = left
      self._joined = {}
      for name, built_from in joined.items():
        player = self._find(name)
        if player is None:
          player = self._build(built_from['prefab'], built_from['params'])
          self._players.append(player)
        if isinstance(player, entity_component.EntityWithComponents):
          player.set_state(built_from.get('state', {}))
        self._joined[name] = {
            'prefab': built_from['prefab'],
            'params': copy.deepcopy(dict(built_from['params'])),
        }

  def _find(self, name: str) -> entity_lib.Entity | None:
    return next((p for p in self._players if p.name == name), None)

  def _build(self, prefab: str, params: Mapping[str, Any]) -> entity_lib.Entity:
    """Builds a player of a prefab, its params updating the prefab's."""
    if prefab not in self._player_prefabs:
      raise KeyError(f'No player prefab named {prefab}.')
    if self._model is None or self._embedder is None:
      raise ValueError('The roster has no model or embedder to build with.')
    shared = self._player_prefabs[prefab]
    # Not a deep copy: the prefab's params may hold objects all its players
    # share (e.g. measurements).
    player_prefab = dataclasses.replace(
        shared, params={**shared.params, **params}
    )
    memory_bank = associative_memory.AssociativeMemoryBank(
        sentence_embedder=self._embedder,
    )
    return player_prefab.build(model=self._model, memory_bank=memory_bank)
