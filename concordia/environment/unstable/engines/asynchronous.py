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

"""Asynchronous engine.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from concordia.environment.unstable import engine as engine_lib
from concordia.typing.unstable import entity as entity_lib


DEFAULT_CALL_TO_MAKE_OBSERVATION = 'What does {name} observe?'
DEFAULT_CALL_TO_NEXT_ACTING = 'Which entities act next?'
DEFAULT_CALL_TO_NEXT_ACTION_SPEC = (
    'How should the game master ask {name} for their action?')
DEFAULT_CALL_TO_RESOLVE = 'What happens next?'
DEFAULT_CALL_TO_CHECK_TERMINATION = 'Is the game over?'


class Asynchronous(engine_lib.Engine):
  """Synchronous engine."""

  def __init__(
      self,
      call_to_make_observation: str = DEFAULT_CALL_TO_MAKE_OBSERVATION,
      call_to_next_acting: str = DEFAULT_CALL_TO_NEXT_ACTING,
      call_to_next_action_spec: str = DEFAULT_CALL_TO_NEXT_ACTION_SPEC,
      call_to_resolve: str = DEFAULT_CALL_TO_RESOLVE,
      call_to_check_termination: str = DEFAULT_CALL_TO_CHECK_TERMINATION,
  ):
    """Synchronous engine constructor."""
    self._call_to_make_observation = call_to_make_observation
    self._call_to_next_acting = call_to_next_acting
    self._call_to_next_action_spec = call_to_next_action_spec
    self._call_to_resolve = call_to_resolve
    self._call_to_check_termination = call_to_check_termination

  def make_observation(self,
                       game_master: entity_lib.Entity,
                       entity: entity_lib.Entity) -> str:
    """Make an observation for a game object."""
    observation = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_make_observation.format(
                name=entity.name),
            output_type=entity_lib.OutputType.MAKE_OBSERVATION,
        )
    )
    return observation

  def next_acting(
      self,
      game_master: entity_lib.Entity,
      entities: Sequence[entity_lib.Entity],
  ) -> tuple[Sequence[entity_lib.Entity], entity_lib.ActionSpec]:
    """Return the next entity or entities to act."""
    entities_by_name = {
        entity.name: entity for entity in entities
    }
    next_object_names_string = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_next_acting,
            output_type=entity_lib.OutputType.NEXT_ACTING,
            options=tuple(entities_by_name.keys()),
        )
    )
    next_entity_names = next_object_names_string.split(',')
    next_action_spec_string = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_next_action_spec.format(
                name=self._call_to_next_action_spec),
            output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
            options=[action_type.name for action_type
                     in entity_lib.PLAYER_ACTION_TYPES],
        )
    )
    next_action_spec = engine_lib.action_spec_parser(next_action_spec_string)
    return [
        entities_by_name[entity_name] for entity_name in next_entity_names
    ], next_action_spec

  def resolve(self,
              game_master: entity_lib.Entity,
              event: str) -> None:
    """Resolve an event."""
    game_master.observe(observation=event)
    result = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_resolve,
            output_type=entity_lib.OutputType.RESOLVE,
        )
    )
    game_master.observe(observation=result)

  def terminate(self,
                game_master: entity_lib.Entity) -> bool:
    """Decide if the episode should terminate."""
    should_terminate_string = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_check_termination,
            output_type=entity_lib.OutputType.TERMINATE,
            options=tuple(entity_lib.BINARY_OPTIONS.values()),
        )
    )
    return should_terminate_string == entity_lib.BINARY_OPTIONS['affirmative']

  def run_loop(
      self,
      game_master: entity_lib.Entity,
      entities: Sequence[entity_lib.Entity],
      premise: str = '',
      max_steps: int = 100,
      verbose: bool = False,
      log: list[Mapping[str, Any]] | None = None,
  ):
    """Run a game loop."""
    if premise:
      self.resolve(game_master, premise)
    steps = 0
    while not self.terminate(game_master) and steps < max_steps:
      next_entities, next_action_spec = self.next_acting(game_master, entities)
      # In the future we will make the following loop concurrent
      for entity in next_entities:
        observation = self.make_observation(game_master, entity)
        entity.observe(observation)
        action = entity.act(next_action_spec)
        self.resolve(game_master, action)
      steps += 1
