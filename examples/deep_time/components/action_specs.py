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

"""Per-entity action specs: the World GM's `__next_action_spec__` component.

`next_acting.FixedActionSpec` returns one spec for everybody. Deep Time's
entities need different specs (a Steward picks an intervention, a species
picks a behaviour), so this component asks each entity's `Actor` instead.
Entities without an active actor get `SKIP_THIS_STEP`.
"""

import dataclasses
import json
from typing import Any, override

from concordia.components.game_master import next_acting
from concordia.environment import engine as engine_lib
from examples.deep_time import actors as actors_lib
from examples.deep_time import crises as crises_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.components import actor_registry
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

SKIP_SPEC = entity_lib.ActionSpec(
    call_to_action='', output_type=entity_lib.OutputType.SKIP_THIS_STEP
)


def aimable_spec(
    actor: actors_lib.Actor, spec: entity_lib.ActionSpec
) -> entity_lib.ActionSpec:
  """A person's spec, free when a move may be aimed (the aim is in the answer).

  A choice must be answered with an option exactly, so a move a person aims
  on the map comes as a free answer, `"<move> @ row,col"`, listing the moves
  as a choice lists them; the actor's `resolve` reads the aim. Models keep
  the choice (they do not aim).

  Args:
    actor: The person's actor.
    spec: Its spec this tick.

  Returns:
    The spec to ask with.
  """
  if spec.output_type == entity_lib.OutputType.CHOICE:
    if not any(actor.aimable(o) for o in spec.options):
      return spec
    letters = 'abcdefghijklmnopqrstuvwxyz'
    moves = '\n'.join(
        f'({letters[i % 26]}) {o}' for i, o in enumerate(spec.options)
    )
    return entity_lib.free_action_spec(
        call_to_action=f'{spec.call_to_action}\n{moves}\n{actors_lib.AIM_HINT}',
        tag=spec.tag,
    )
  if spec.output_type == entity_lib.OutputType.FREE and (
      actors_lib.AIM_HINT not in spec.call_to_action
  ):
    moves = getattr(actor, 'options', lambda: [])()
    if any(actor.aimable(o) for o in moves):
      return dataclasses.replace(
          spec, call_to_action=f'{spec.call_to_action}\n{actors_lib.AIM_HINT}'
      )
  return spec


class ActorActionSpecs(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Returns each entity's action spec from the game's actor registry."""

  def __init__(
      self,
      context: game_lib.GameContext,
      call_to_next_action_spec: str = (
          next_acting.DEFAULT_CALL_TO_NEXT_ACTION_SPEC
      ),
  ):
    """Initializes the component.

    Args:
      context: Shared game state holding the actor registry.
      call_to_next_action_spec: The engine's NEXT_ACTION_SPEC call to action,
        used to recover the entity name.
    """
    super().__init__()
    self._context = context
    self._prefix, self._suffix = call_to_next_action_spec.split('{name}')
    # This step's menus, by entity (the engine logs only the last call).
    self._menus: dict[str, dict[str, Any]] = {}
    self._menus_tick: int | None = None

  def _entity_name(self, call_to_action: str) -> str:
    if not (
        call_to_action.startswith(self._prefix)
        and call_to_action.endswith(self._suffix)
    ):
      raise ValueError(f'Unexpected call to action: {call_to_action!r}')
    end = len(call_to_action) - len(self._suffix)
    return call_to_action[len(self._prefix) : end]

  def _actors(self) -> dict[str, actors_lib.Actor]:
    """The acting entities' actors: the World GM's registry component."""
    try:
      return (
          self.get_entity()
          .get_component(
              actor_registry.DEFAULT_ACTOR_REGISTRY_KEY,
              type_=actor_registry.ActorRegistry,
          )
          .actors
      )
    except (AttributeError, KeyError, ValueError, RuntimeError):
      return self._context.actors  # Not yet on a game master (tests).

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTION_SPEC:
      return ''
    name = self._entity_name(action_spec.call_to_action)
    actor = self._actors().get(name)
    spec = actor.action_spec() if actor and actor.is_active() else SKIP_SPEC
    if actor is not None and name in self._context.humans:
      spec = aimable_spec(actor, spec)
    if self._menus_tick != self._context.planet.tick:
      self._menus, self._menus_tick = {}, self._context.planet.tick
    if actor is not None and spec is not SKIP_SPEC:
      self._menus[name] = self._menu(actor, name, spec)
      # The role's menu, for the person who plays it (their seat and page).
      self._context.bus.publish(
          perception.sheet_topic(name, 'menu'),
          json.dumps(self._menus[name]),
          source='world',
      )
    self._logging_channel({
        'Key': 'Action spec',
        'Summary': f'{name}: {spec.output_type.name}',
        'Value': spec.to_dict(),
        # Every menu offered so far this step, with what each move does.
        'Menus': dict(self._menus),
    })
    return engine_lib.action_spec_to_string(spec)

  def _menu(
      self, actor: actors_lib.Actor, name: str, spec: entity_lib.ActionSpec
  ) -> dict[str, Any]:
    """What an entity was offered: its moves, those saved for, their pushes.

    Args:
      actor: The entity's actor.
      name: The entity's name.
      spec: The spec it is asked with.

    Returns:
      `offered` (a Steward that may speak still has its moves), `later`
      (moves it is saving for, plannable for a later turn) and `pushes`
      (move -> what it does to the world, measured, for those that say).
    """
    offered = list(spec.options) or list(getattr(actor, 'options', list)())
    later = list(getattr(actor, 'out_of_reach', list)())
    return {
        'offered': offered,
        'later': later,
        'pushes': self._context.pushes([*offered, *later], name),
        # Whether it may answer in its own words, and why the moves it is
        # saving for cannot be made yet.
        'words': actor.speaks(),
        'why_not': {m: why for m in later if (why := actor.why_not(m))},
        # For a people a person plays: the moves answering its most urgent
        # crisis and strengthening its weakest link.
        **(
            {
                'pressing': crises_lib.pressing(
                    self._context.planet,
                    actor.polity,
                    offered,
                    self._context.pushes(offered, name),
                )
            }
            if isinstance(actor, actors_lib.PolityActor)
            and name in self._context.humans
            else {}
        ),
    }

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state
