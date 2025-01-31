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
from collections.abc import Sequence

from concordia.typing import entity as entity_lib


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
  ) -> Sequence[entity_lib.Entity]:
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
  def run_loop(
      self,
      game_master: entity_lib.Entity,
      entities: Sequence[entity_lib.Entity],
      premise: str,
      max_steps: int,
      verbose: bool,
  ):
    """Run a game loop."""


def action_spec_parser(next_action_spec_string: str) -> entity_lib.ActionSpec:
  """Parse the next action spec string into an action spec."""
  if 'type: free' in next_action_spec_string:
    return entity_lib.ActionSpec(
        call_to_action=next_action_spec_string,
        output_type=entity_lib.OutputType.FREE,
    )
  elif 'type: choice' in next_action_spec_string:
    return entity_lib.ActionSpec(
        call_to_action=next_action_spec_string,
        output_type=entity_lib.OutputType.CHOICE,
        options=tuple(next_action_spec_string.split('options: ')[1].split(',')),
    )
  else:
    raise RuntimeError(
        'Invalid next action spec string: \"{}\"'.format(
            next_action_spec_string))
