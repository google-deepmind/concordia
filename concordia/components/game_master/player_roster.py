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
`entities` a game master prefab is given, the very list the engine runs.
Players join the game, built from a prefab, and leave it only through the
roster, and the game master's other components find a player by name here.
They may also condition on who plays: the roster is `ActionSpecIgnored`, and
its pre-act value is the players' names. Nothing else is needed of the
simulation or the engine: the list itself changes, so an engine that runs it
sees the change (the asynchronous engine starts a loop for a player who joins
and ends the loop of one who leaves).

  roster = player_roster.PlayerRoster(
      players=entities,  # the list of entities the engine runs
      player_prefabs={'person': person_prefab},
      model=model,
      embedder=embedder,
  )
  roster.join('person', {'name': 'Ada'})
  ada = roster.get_player_by_name('Ada')
  roster.leave('Ada')

A player who joins is built from its prefab with the prefab's params updated
by the player's own. What all players of a prefab share (e.g. measurements)
stays on the prefab, which is not deep-copied. The player's own params (its
name among them) must be JSON-serializable, since the roster's state keeps
them. (A simulation's `add_entity`, by contrast, replaces a deep copy's
params.)

A checkpoint of a `generic.Simulation` keeps the players it began with. The
roster's state keeps the rest: who joined and is still playing (the prefab,
params and state of each) and which of the first players left. Restored, the
roster takes away the first players who left and rebuilds those who joined.
Two things follow:
- the simulation's checkpoint logs a warning for each player who joined (it
  has no prefab config of theirs), though the roster keeps them;
- a checkpoint is consistent when taken where no player joins or leaves (e.g.
  in the game master's own turn), since the simulation reads the list without
  the roster's lock.

Other things to know:
- with the asynchronous engine, each player who joins gets `max_steps` of its
  own, and the action of a player who leaves while acting is still resolved;
- stock components given fixed `player_names` (e.g. `MakeObservation`'s
  broadcast to all) do not follow the roster; condition on the roster
  instead.
"""

from collections.abc import Callable, Mapping, MutableSequence, Sequence
import copy
import dataclasses
import json
import threading
from typing import Any, cast, override

from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.components.agent import action_spec_ignored
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
import numpy as np

DEFAULT_PLAYER_ROSTER_COMPONENT_KEY = '__player_roster__'
DEFAULT_PLAYER_ROSTER_PRE_ACT_LABEL = '\nPlayers'


def _close_entity(player: entity_lib.Entity) -> None:
  """Closes `player` and any of its components that define `close()`."""
  targets: list[Any] = [player]
  if hasattr(player, 'get_act_component'):
    targets.append(getattr(player, 'get_act_component')())
  if hasattr(player, 'get_all_context_components'):
    targets.extend(getattr(player, 'get_all_context_components')().values())
  seen_ids: set[int] = set()
  for target in targets:
    if id(target) in seen_ids:
      continue
    seen_ids.add(id(target))
    close_fn = getattr(target, 'close', None)
    if callable(close_fn):
      close_fn()


class PlayerRoster(action_spec_ignored.ActionSpecIgnored):
  """The players of a game, and the only way they join and leave it.

  Its pre-act value is the names of the players, in the order they play, so
  other components may condition on who plays
  (`get_named_component_pre_act_value`).
  """

  def __init__(
      self,
      players: Sequence[entity_lib.Entity],
      player_prefabs: Mapping[str, prefab_lib.Prefab] | None = None,
      model: language_model.LanguageModel | None = None,
      embedder: Callable[[str], np.ndarray] | None = None,
      pre_act_label: str = DEFAULT_PLAYER_ROSTER_PRE_ACT_LABEL,
      *,
      close_on_leave: bool = False,
  ):
    """Initializes the roster.

    Args:
      players: The list of player entities the simulation plays (the list the
        engine runs), changed in place as players join and leave. The players
        in it now are the game's first players.
      player_prefabs: The prefabs players may join as, by name. Their params
        are those all players of the prefab share.
      model: The language model players who join act with.
      embedder: The sentence embedder of the memory of players who join.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
      close_on_leave: Whether to call `close()` on a player who leaves (and on
        those of its components that define it, e.g. pubsub subscribers) to
        release background subscriptions. Off by default, since `leave`
        returns the player to its caller; do not enable it if players share
        components with `close()`.

    Raises:
      TypeError: If `players` cannot be changed in place (e.g. a tuple).
    """
    if not isinstance(players, MutableSequence):
      raise TypeError(
          'The roster changes the list of players in place: give it the list'
          f' the simulation plays, not a {type(players).__name__}.'
      )
    super().__init__(pre_act_label)
    self._players: MutableSequence[entity_lib.Entity] = players
    self._player_prefabs = dict(player_prefabs or {})
    self._model = model
    self._embedder = embedder
    self._close_on_leave = close_on_leave
    # The players' lock, apart from the pre-act value's (`_lock`). The value is
    # made holding `_lock`, then this one; `_lock` is never taken holding this.
    self._players_lock = threading.RLock()
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

    Returns:
      The player.

    Raises:
      KeyError: If no player of that name plays.
    """
    with self._players_lock:
      player = self._find(name)
    if player is None:
      raise KeyError(f'No player named {name} plays.')
    return player

  def get_player_names(self) -> list[str]:
    """Returns the names of the players, in the order they play."""
    with self._players_lock:
      return [player.name for player in self._players]

  def join(
      self,
      prefab: str,
      params: Mapping[str, Any],
      *,
      initial_memories: Sequence[str] = (),
  ) -> entity_lib.Entity:
    """Builds a player of a prefab and adds it to the game.

    Args:
      prefab: The name of the player's prefab (a key of `player_prefabs`).
      params: The player's own params, which update the prefab's. They must
        be JSON-serializable.
      initial_memories: Observations the player is given (via `observe`)
        before it joins the list of players, so that an engine never runs a
        player who has not yet received them.

    Returns:
      The player who joined.

    Raises:
      KeyError: If there is no such prefab.
      TypeError: If the params are not JSON-serializable.
      ValueError: If the roster cannot build players, or if the player's name
        is taken (by a player who plays, or by one of the first players).
    """
    own = json.loads(json.dumps(dict(params)))
    with self._players_lock:
      player = self._build(prefab, own)
      if self._find(player.name) is not None or player.name in self._first:
        raise ValueError(f'The name {player.name} is taken.')
      for memory in initial_memories:
        player.observe(memory)
      self._players.append(player)
      self._joined[player.name] = {'prefab': prefab, 'params': own}
    self._players_changed()
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
    with self._players_lock:
      player = self.get_player_by_name(name)
      self._players.remove(player)
      if self._joined.pop(name, None) is None:
        self._left.append(name)
    self._players_changed()
    if self._close_on_leave:
      _close_entity(player)
    return player

  @override
  def get_state(self) -> entity_component.ComponentState:
    """Returns who joined (with each one's state) and who left."""
    with self._players_lock:
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

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    """Makes the players those of the state.

    The first players who left are taken away, players who joined since the
    state was taken leave, and players who joined before it are rebuilt (if
    they no longer play) and given their states. The first players' own
    states are not the roster's: a simulation's checkpoint restores them.
    Every player is built before the list changes, so a player that cannot
    be built leaves the players as they were.

    Args:
      state: A state from `get_state`.
    """
    saved = cast(Mapping[str, Any], state)
    left = [str(name) for name in saved.get('left', ())]
    joined: Mapping[str, Any] = saved.get('joined', {})
    removed_players: list[entity_lib.Entity] = []
    with self._players_lock:
      built = {
          name: self._build(built_from['prefab'], built_from['params'])
          for name, built_from in joined.items()
          if self._find(name) is None
      }
      for name in left + [n for n in self._joined if n not in joined]:
        player = self._find(name)
        if player is not None:
          self._players.remove(player)
          removed_players.append(player)
      self._left = left
      self._joined = {}
      for name, built_from in joined.items():
        player = self._find(name)
        if player is None:
          player = built[name]
          self._players.append(player)
        if isinstance(player, entity_component.EntityWithComponents):
          player.set_state(built_from.get('state', {}))
        self._joined[name] = {
            'prefab': built_from['prefab'],
            'params': copy.deepcopy(dict(built_from['params'])),
        }
    self._players_changed()
    if self._close_on_leave:
      for player in removed_players:
        _close_entity(player)

  def _make_pre_act_value(self) -> str:
    """Returns the players' names, in the order they play."""
    return ', '.join(self.get_player_names())

  def _players_changed(self) -> None:
    """Clears the pre-act value, so that it names the players who play now."""
    with self._lock:
      self._pre_act_value = None

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
