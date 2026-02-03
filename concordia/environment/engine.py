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

"""Engine base class."""

import abc
from collections.abc import Callable, Mapping, Sequence
import json
import re
from typing import Any

from absl import logging
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


def _legacy_action_spec_parser(
    next_action_spec_string: str,
) -> entity_lib.ActionSpec:
  """Parse the next action spec string using the legacy format.

  This exists for backward compatibility with the old string format:
  "prompt: <call_to_action>;;type: <free|choice> [options: opt1, opt2, ...]"

  Args:
    next_action_spec_string: The string representation of the next action spec.

  Returns:
    The parsed action spec.

  Raises:
    RuntimeError: If the next action spec string is invalid.
  """
  if 'type: free' in next_action_spec_string:
    splits = next_action_spec_string.split(';;')

    if splits and 'prompt: ' in splits[0]:
      call_to_action = splits[0].split('prompt: ', 1)[1]
    else:
      call_to_action = entity_lib.DEFAULT_CALL_TO_ACTION

    return entity_lib.ActionSpec(
        call_to_action=call_to_action,
        output_type=entity_lib.OutputType.FREE,
    )

  elif 'type: choice' in next_action_spec_string:
    splits = next_action_spec_string.split(';;')

    if 'prompt: ' in splits[0]:
      call_to_action = splits[0].split('prompt: ', 1)[1]
    else:
      call_to_action = entity_lib.DEFAULT_CALL_TO_ACTION

    if 'options: ' not in next_action_spec_string:
      return entity_lib.ActionSpec(
          call_to_action=call_to_action,
          output_type=entity_lib.OutputType.FREE,
      )

    options_str = next_action_spec_string.split('options: ', 1)[1]
    parts = re.split(r'(?<!\\),', options_str)
    options = tuple(
        dict.fromkeys(
            part.replace(r'\,', ',').strip() for part in parts if part.strip()
        )
    )
    return entity_lib.ActionSpec(
        call_to_action=call_to_action,
        output_type=entity_lib.OutputType.CHOICE,
        options=options,
    )
  elif _TYPE_SKIP_THIS_STEP in next_action_spec_string:
    return entity_lib.skip_this_step_action_spec()
  else:
    raise RuntimeError(
        'Invalid next action spec string: "{}"'.format(next_action_spec_string)
    )


def action_spec_parser(next_action_spec_string: str) -> entity_lib.ActionSpec:
  """Parse the next action spec string into an action spec.

  Supports both JSON format (preferred) and legacy string format (for backward
  compatibility).

  Args:
    next_action_spec_string: The string representation of the next action spec.

  Returns:
    The parsed action spec.
  """
  start_idx = next_action_spec_string.find('{')
    
  if start_idx != -1:
      balance = 0
      end_idx = -1
      
      # Walk through the string to find the matching closing brace
      for i, char in enumerate(next_action_spec_string[start_idx:], start=start_idx):
          if char == '{':
              balance += 1
          elif char == '}':
              balance -= 1
              
          # When balance hits zero, we found the end of the first object
          if balance == 0:
              end_idx = i
              break
      
      # Cut the string to keep ONLY the first valid JSON object
      if end_idx != -1:
          next_action_spec_string = next_action_spec_string[start_idx : end_idx + 1]
  try:
    spec_dict = json.loads(next_action_spec_string)
    return entity_lib.action_spec_from_dict(spec_dict)
  except json.JSONDecodeError:
    logging.warning(
        'Using legacy action spec parser. Please migrate to JSON format. '
        'Input was: %s...',
        next_action_spec_string[:100],
    )
    return _legacy_action_spec_parser(next_action_spec_string)


def action_spec_to_string(action_spec: entity_lib.ActionSpec) -> str:
  """Convert an action spec to a JSON string."""
  return json.dumps(action_spec.to_dict())
