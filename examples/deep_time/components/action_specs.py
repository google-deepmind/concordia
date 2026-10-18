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
from typing import override

from concordia.components.game_master import next_acting
from concordia.environment import engine as engine_lib
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
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

  def _entity_name(self, call_to_action: str) -> str:
    if not (
        call_to_action.startswith(self._prefix)
        and call_to_action.endswith(self._suffix)
    ):
      raise ValueError(f'Unexpected call to action: {call_to_action!r}')
    end = len(call_to_action) - len(self._suffix)
    return call_to_action[len(self._prefix) : end]

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTION_SPEC:
      return ''
    name = self._entity_name(action_spec.call_to_action)
    actor = self._context.actors.get(name)
    spec = actor.action_spec() if actor and actor.is_active() else SKIP_SPEC
    if actor is not None and name in self._context.humans:
      spec = aimable_spec(actor, spec)
    self._logging_channel({
        'Key': 'Action spec',
        'Summary': f'{name}: {spec.output_type.name}',
        'Value': spec.to_dict(),
    })
    return engine_lib.action_spec_to_string(spec)

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state
