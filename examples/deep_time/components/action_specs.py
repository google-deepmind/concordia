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

from typing import override

from concordia.components.game_master import next_acting
from concordia.environment import engine as engine_lib
from examples.deep_time import game as game_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

SKIP_SPEC = entity_lib.ActionSpec(
    call_to_action='', output_type=entity_lib.OutputType.SKIP_THIS_STEP
)


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
