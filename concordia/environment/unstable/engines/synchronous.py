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

"""Synchronous engine.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from concordia.components.game_master.unstable import event_resolution as event_resolution_components
from concordia.components.game_master.unstable import make_observation as make_observation_component
from concordia.components.game_master.unstable import next_acting as next_acting_components
from concordia.components.game_master.unstable import switch_act as switch_act_component
from concordia.environment.unstable import engine as engine_lib
from concordia.typing.unstable import agent as agent_lib
from concordia.typing.unstable import entity as entity_lib
import termcolor


DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    make_observation_component.DEFAULT_CALL_TO_MAKE_OBSERVATION)
DEFAULT_CALL_TO_NEXT_ACTING = next_acting_components.DEFAULT_CALL_TO_NEXT_ACTING
DEFAULT_CALL_TO_NEXT_ACTION_SPEC = (
    next_acting_components.DEFAULT_CALL_TO_NEXT_ACTION_SPEC)
DEFAULT_CALL_TO_RESOLVE = event_resolution_components.DEFAULT_CALL_TO_RESOLVE
DEFAULT_CALL_TO_CHECK_TERMINATION = 'Is the game/simulation finished?'

DEFAULT_ACT_COMPONENT_NAME = switch_act_component.DEFAULT_ACT_COMPONENT_NAME

_PRINT_COLOR = 'cyan'


def _get_empty_log_entry():
  """Returns a dictionary to store a single log entry."""
  return {
      'terminate': {},
      'make_observation': {},
      'resolve': {},
  }


class Synchronous(engine_lib.Engine):
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
  ) -> tuple[entity_lib.Entity, entity_lib.ActionSpec]:
    """Return the next entity or entities to act."""
    entities_by_name = {
        entity.name: entity for entity in entities
    }
    next_object_name = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_next_acting,
            output_type=entity_lib.OutputType.NEXT_ACTING,
            options=tuple(entities_by_name.keys()),
        )
    )
    next_action_spec_string = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_next_action_spec.format(
                name=next_object_name),
            output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
        )
    )
    next_action_spec = engine_lib.action_spec_parser(next_action_spec_string)
    return (entities_by_name[next_object_name], next_action_spec)

  def resolve(self,
              game_master: entity_lib.Entity,
              event: str,
              verbose: bool = False) -> None:
    """Resolve an event."""
    if verbose:
      print(termcolor.colored(
          f'The suggested action or event to resolve was: {event}',
          _PRINT_COLOR))
    game_master.observe(observation=event)
    result = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_resolve,
            output_type=entity_lib.OutputType.RESOLVE,
        )
    )
    game_master.observe(observation=result)
    if verbose:
      print(termcolor.colored(
          f'The resolved event was: {result}', _PRINT_COLOR))

  def terminate(self,
                game_master: entity_lib.Entity,
                verbose: bool = False) -> bool:
    """Decide if the episode should terminate."""
    should_terminate_string = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_check_termination,
            output_type=entity_lib.OutputType.TERMINATE,
            options=tuple(entity_lib.BINARY_OPTIONS.values()),
        )
    )
    if verbose:
      print(termcolor.colored(
          f'Terminate? {should_terminate_string}', _PRINT_COLOR))
    return should_terminate_string == entity_lib.BINARY_OPTIONS['affirmative']

  def run_loop(
      self,
      game_master: entity_lib.Entity | agent_lib.GenerativeAgent,
      entities: Sequence[entity_lib.Entity | agent_lib.GenerativeAgent],
      premise: str = '',
      max_steps: int = 100,
      verbose: bool = False,
      log: list[Mapping[str, Any]] | None = None,
  ):
    """Run a game loop."""
    log_entry = _get_empty_log_entry()
    steps = 0
    if premise:
      self.resolve(game_master, premise, verbose=verbose)
      if log is not None and hasattr(game_master, 'get_last_log'):
        assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
        log_entry['resolve'] = game_master.get_last_log()
        log.append({
            'Game Master': log_entry,
            'Entity': {},
            'Step': steps,
        })
        log_entry = _get_empty_log_entry()
    while not self.terminate(game_master, verbose) and steps < max_steps:
      if log is not None:
        if hasattr(game_master, 'get_last_log'):
          assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
          log_entry['terminate'] = game_master.get_last_log()
      for entity in entities:
        observation = self.make_observation(game_master, entity)
        if log is not None and hasattr(game_master, 'get_last_log'):
          assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
          log_entry['make_observation'][entity.name] = (
              game_master.get_last_log())
        if verbose:
          print(termcolor.colored(
              f'Entity {entity.name} observed: {observation}', _PRINT_COLOR))
        entity.observe(observation)
      next_entity, entity_spec_to_use = self.next_acting(game_master, entities)
      if verbose:
        print(termcolor.colored(
            f'Entity {next_entity.name} is next to act. They must respond '
            f' in the format: "{entity_spec_to_use}".', _PRINT_COLOR))
      raw_action = next_entity.act(entity_spec_to_use)
      if next_entity.name in raw_action:
        action = raw_action
      else:
        action = f'{next_entity.name}: {raw_action}'
      if verbose:
        print(termcolor.colored(
            f'Entity {next_entity.name} chose action: {action}', _PRINT_COLOR))
      self.resolve(game_master=game_master, event=action, verbose=verbose)

      steps += 1
      if log is not None and hasattr(game_master, 'get_last_log'):
        assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
        log_entry['resolve'] = game_master.get_last_log()
        next_entity_log = ''
        game_master_key = 'Game Master'
        entity_key = 'Entity'
        if hasattr(next_entity, 'get_last_log'):
          assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
          next_entity_log = next_entity.get_last_log()
          entity_key = f'{entity_key} [{next_entity.name}]'
        if DEFAULT_ACT_COMPONENT_NAME in log_entry['resolve']:
          event_to_log = log_entry['resolve'][
              DEFAULT_ACT_COMPONENT_NAME
          ]['Value']
          game_master_key = f'{game_master_key} --- {event_to_log}'
        log.append({
            game_master_key: log_entry,
            entity_key: next_entity_log,
            'Step': steps,
        })
        log_entry = _get_empty_log_entry()
