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

"""The actor registry: the World GM's rules for each acting entity.

Each acting entity has an `Actor` (`actors.py`): the rules that give it its
action spec and resolve what it does. The registry is one of the World GM's
components (under `DEFAULT_ACTOR_REGISTRY_KEY`), so the stock checkpoint
keeps every actor's state (what it resolved, its log, its pending outbox,
its decision counts) with the rest of the game master; `ActorActionSpecs`
and `ActorResolution` look their actors up here.
"""

from collections.abc import Mapping
from typing import Any, override

from examples.deep_time import actors as actors_lib
from concordia.typing import entity_component

DEFAULT_ACTOR_REGISTRY_KEY = '__actors__'


class ActorRegistry(entity_component.ContextComponent):
  """The actors of the acting entities, by entity name."""

  def __init__(
      self,
      actors: dict[str, actors_lib.Actor],
      access: dict[str, tuple[str, ...]],
  ):
    """Initializes the registry.

    Args:
      actors: The actors by entity name (the game context shares this dict).
      access: The topic prefixes each entity hears as observations.
    """
    super().__init__()
    self.actors = actors
    self.access = access

  def add(self, actor: actors_lib.Actor, hears: tuple[str, ...] = ()) -> None:
    """Registers an actor and the topics it hears as observations."""
    self.actors[actor.name] = actor
    self.access[actor.name] = tuple(hears)

  @override
  def pre_act(self, action_spec: Any) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {
        name: actor.get_state() for name, actor in sorted(self.actors.items())
    }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    for name, kept in state.items():
      if str(name) in self.actors and isinstance(kept, Mapping):
        self.actors[str(name)].set_state(dict(kept))
