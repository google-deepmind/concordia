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

"""Engine base class.
"""

import abc
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from concordia.typing import entity as entity_lib

_TYPE_SKIP_THIS_STEP = 'type: __SKIP_THIS_STEP__'


class Engine(metaclass=abc.ABCMeta):
  """Engine interface."""

  @abc.abstractmethod
  def make_observation(
      self,
      game_master: entity_lib.Entity,
      entity: entity_lib.Entity,
  ) -> str:
    """Make an observation for an entity."""

  @abc.abstractmethod
  def next_acting(
      self,
      game_master: entity_lib.Entity,
      entities: Sequence[entity_lib.Entity],
  ) -> tuple[entity_lib.Entity, entity_lib.ActionSpec]:
    """Return the next entity or entities to act."""

  @abc.abstractmethod
  def resolve(
      self,
      game_master: entity_lib.Entity,
      event: str,
  ) -> None:
    """Resolve the event."""

  @abc.abstractmethod
  def terminate(
      self,
      game_master: entity_lib.Entity,
  ) -> bool:
    """Decide if the episode should terminate or continue."""

  @abc.abstractmethod
  def next_game_master(
      self,
      game_master: entity_lib.Entity,
      game_masters: Sequence[entity_lib.Entity],
  ) -> entity_lib.Entity:
    """Return the game master that will be responsible for the next step."""

  @abc.abstractmethod
  def run_loop(
      self,
      game_masters: Sequence[entity_lib.Entity],
      entities: Sequence[entity_lib.Entity],
      premise: str,
      max_steps: int,
      verbose: bool,
      log: list[Mapping[str, Any]] | None,
      checkpoint_callback: Callable[[int], None] | None = None,
  ):
    """Run a game loop."""


def action_spec_parser(next_action_spec_string: str) -> entity_lib.ActionSpec:
  """Parse the next action spec string into an action spec."""
  if 'type: free' in next_action_spec_string:
    splits = next_action_spec_string.split(';;')
    if len(splits) != 2:
      call_to_action = entity_lib.DEFAULT_CALL_TO_ACTION
    else:
      call_to_action = splits[0].split('prompt: ')[1]
    return entity_lib.ActionSpec(
        call_to_action=call_to_action,
        output_type=entity_lib.OutputType.FREE,
    )
  elif (
      'type: choice' in next_action_spec_string
      and 'options: ' in next_action_spec_string
  ):
    splits = next_action_spec_string.split(';;')
    if 'prompt: ' not in next_action_spec_string:
      call_to_action = entity_lib.DEFAULT_CALL_TO_ACTION
    else:
      call_to_action = splits[0].split('prompt: ')[1]
    return entity_lib.ActionSpec(
        call_to_action=call_to_action,
        output_type=entity_lib.OutputType.CHOICE,
        options=tuple(next_action_spec_string.split('options: ')[1].split(',')),
    )
  elif _TYPE_SKIP_THIS_STEP in next_action_spec_string:
    return entity_lib.skip_this_step_action_spec()
  else:
    raise RuntimeError(
        'Invalid next action spec string: \"{}\"'.format(
            next_action_spec_string))


def action_spec_to_string(action_spec: entity_lib.ActionSpec) -> str:
  """Convert an action spec to a string."""
  if action_spec.output_type == entity_lib.OutputType.FREE:
    return f'prompt: {action_spec.call_to_action};;type: free'
  elif action_spec.output_type == entity_lib.OutputType.CHOICE:
    return (
        f'prompt: {action_spec.call_to_action};;type: choice options: '
        + ', '.join(action_spec.options)
    )
  elif action_spec.output_type == entity_lib.OutputType.SKIP_THIS_STEP:
    return f'prompt: {action_spec.call_to_action};;{_TYPE_SKIP_THIS_STEP}'
  else:
    raise RuntimeError(
        'Invalid action spec output type: \"{}\"'.format(
            action_spec.output_type))
