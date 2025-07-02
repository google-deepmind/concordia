# Copyright 2025 DeepMind Technologies Limited.
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

"""Simultaneous engine.
"""

from collections.abc import Mapping, Sequence
import functools
from typing import Any, Callable

from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import make_observation as make_observation_component
from concordia.components.game_master import next_acting as next_acting_components
from concordia.components.game_master import switch_act as switch_act_component
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from concordia.utils import concurrency
import termcolor
from typing_extensions import override


DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    make_observation_component.DEFAULT_CALL_TO_MAKE_OBSERVATION)
DEFAULT_CALL_TO_NEXT_ACTING = 'Which entities act next?'
DEFAULT_CALL_TO_NEXT_ACTION_SPEC = (
    next_acting_components.DEFAULT_CALL_TO_NEXT_ACTION_SPEC)
DEFAULT_CALL_TO_RESOLVE = 'Because of all that came before, what happens next?'
DEFAULT_CALL_TO_CHECK_TERMINATION = 'Is the game/simulation finished?'
DEFAULT_CALL_TO_NEXT_GAME_MASTER = (
    'Which rule set should we use for the next step?')

DEFAULT_ACT_COMPONENT_KEY = switch_act_component.DEFAULT_ACT_COMPONENT_KEY

PUTATIVE_EVENT_TAG = event_resolution_components.PUTATIVE_EVENT_TAG
EVENT_TAG = event_resolution_components.EVENT_TAG

_PRINT_COLOR = 'cyan'


def _get_empty_log_entry():
  """Returns a dictionary to store a single log entry."""
  return {
      'terminate': {},
      'next_game_master': {},
      'make_observation': {},
      'next_acting': {},
      'next_action_spec': {},
      'resolve': {},
  }


class Simultaneous(engine_lib.Engine):
  """Engine for simultaneous move games."""

  def __init__(
      self,
      call_to_make_observation: str = DEFAULT_CALL_TO_MAKE_OBSERVATION,
      call_to_next_acting: str = DEFAULT_CALL_TO_NEXT_ACTING,
      call_to_next_action_spec: str = DEFAULT_CALL_TO_NEXT_ACTION_SPEC,
      call_to_resolve: str = DEFAULT_CALL_TO_RESOLVE,
      call_to_check_termination: str = DEFAULT_CALL_TO_CHECK_TERMINATION,
      call_to_next_game_master: str = DEFAULT_CALL_TO_NEXT_GAME_MASTER,
  ):
    """Simultaneous engine constructor."""
    self._call_to_make_observation = call_to_make_observation
    self._call_to_next_acting = call_to_next_acting
    self._call_to_next_action_spec = call_to_next_action_spec
    self._call_to_resolve = call_to_resolve
    self._call_to_check_termination = call_to_check_termination
    self._call_to_next_game_master = call_to_next_game_master

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

  @override
  def next_acting(
      self,
      game_master: entity_lib.Entity,
      entities: Sequence[entity_lib.Entity],
      log_entry: Mapping[str, Any] | None = None,
      log: list[Mapping[str, Any]] | None = None,
  ) -> tuple[Sequence[entity_lib.Entity],
             Sequence[entity_lib.ActionSpec]]:  # pytype: disable=signature-mismatch
    """Return the next action spec for an entity."""
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
    if log is not None and hasattr(game_master, 'get_last_log'):
      assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
      log_entry['next_acting'] = game_master.get_last_log()

    action_spec_by_name = {}
    for next_entity_name in next_entity_names:
      next_action_spec_string = game_master.act(
          action_spec=entity_lib.ActionSpec(
              call_to_action=self._call_to_next_action_spec.format(
                  name=next_entity_name),
              output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
          )
      )
      action_spec_by_name[next_entity_name] = engine_lib.action_spec_parser(
          next_action_spec_string)

      if log is not None and hasattr(game_master, 'get_last_log'):
        assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
        log_entry['next_action_spec'] = game_master.get_last_log()

    return (
        [entities_by_name[entity_name] for entity_name in next_entity_names],
        [action_spec_by_name[entity_name] for entity_name in next_entity_names]
    )

  def resolve(self,
              game_master: entity_lib.Entity,
              putative_event: str,
              verbose: bool = False) -> None:
    """Resolve an event."""
    if verbose:
      print(termcolor.colored(
          f'The suggested action or event to resolve was: {putative_event}',
          _PRINT_COLOR))
    game_master.observe(observation=f'{PUTATIVE_EVENT_TAG} {putative_event}')
    result = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_resolve,
            output_type=entity_lib.OutputType.RESOLVE,
        )
    )
    game_master.observe(observation=f'{EVENT_TAG} {result}')
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

  def next_game_master(self,
                       game_master: entity_lib.Entity,
                       game_masters: Sequence[entity_lib.Entity],
                       verbose: bool = False) -> entity_lib.Entity:
    """Select which game master to use for the next step."""
    if len(game_masters) == 1:
      return game_masters[0]
    game_masters_by_name = {
        game_master_.name: game_master_ for game_master_ in game_masters
    }
    next_game_master_name = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_next_game_master,
            output_type=entity_lib.OutputType.NEXT_GAME_MASTER,
            options=tuple(game_masters_by_name.keys()),
        )
    )
    if verbose:
      print(termcolor.colored(
          f'Game master: {next_game_master_name}', _PRINT_COLOR))
    if next_game_master_name not in game_masters_by_name:
      raise ValueError(
          f'Selected game master {next_game_master_name} not found in:'
          f' {game_masters_by_name.keys()}'
      )
    return game_masters_by_name[next_game_master_name]

  def run_loop(
      self,
      game_masters: Sequence[entity_lib.Entity],
      entities: Sequence[entity_lib.Entity],
      premise: str = '',
      max_steps: int = 100,
      verbose: bool = False,
      log: list[Mapping[str, Any]] | None = None,
      checkpoint_callback: Callable[[int], None] | None = None,
  ):
    """Run a game loop."""
    if not game_masters:
      raise ValueError('No game masters provided.')

    log_entry = _get_empty_log_entry()
    game_master = game_masters[0]
    steps = 0
    if premise:
      premise = f'{EVENT_TAG} {premise}'
      game_master.observe(premise)
    while not self.terminate(game_master, verbose) and steps < max_steps:
      if log is not None and hasattr(game_master, 'get_last_log'):
        assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
        log_entry['terminate'] = game_master.get_last_log()

      game_master = self.next_game_master(game_master, game_masters, verbose)
      if log is not None and hasattr(game_master, 'get_last_log'):
        assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
        log_entry['next_game_master'] = game_master.get_last_log()

      next_entities, next_action_specs = self.next_acting(
          game_master, entities, log_entry=log_entry, log=log
      )

      if (
          next_action_specs[0].output_type
          == entity_lib.OutputType.SKIP_THIS_STEP
      ):
        # For initialization, it is often useful to have a special game master
        # that initializes other players and game masters but does not itself
        # allow players to take actions. In this case, we skip the current
        # step and continue to the next step.
        if verbose:
          print(
              termcolor.colored(
                  '\nSkipping the action phase for the current time step.\n'
              )
          )
        skip_actions = True
        if checkpoint_callback is not None:
          print(f'Calling checkpoint callback at step {steps}')
          checkpoint_callback(steps)
      else:
        skip_actions = False

      def _entity_act(
          entity: entity_lib.Entity, action_spec: entity_lib.ActionSpec,
          skip_actions: bool = False,
      ) -> str:
        """Make observation, get action and resolution for one entity."""
        observation = self.make_observation(game_master, entity)
        entity.observe(observation)
        if log is not None and hasattr(game_master, 'get_last_log'):
          assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
          log_entry['make_observation'][
              entity.name
          ] = game_master.get_last_log()
        if verbose:
          print(
              termcolor.colored(
                  f'Entity {entity.name} observed: {observation}', _PRINT_COLOR
              )
          )

        if skip_actions:
          return ''

        if verbose:
          print(
              termcolor.colored(
                  f'Entity {entity.name} is next to act. They must respond '
                  f' in the format: "{action_spec}".',
                  _PRINT_COLOR,
              )
          )
        raw_action = entity.act(action_spec)
        if entity.name in raw_action:
          action = raw_action
        else:
          action = f'{entity.name}: {raw_action}'
        if verbose:
          print(
              termcolor.colored(
                  f'Entity {entity.name} chose action: {action}', _PRINT_COLOR
              )
          )

        return action

      tasks = {}
      entities_to_process = entities if skip_actions else next_entities
      for i, entity in enumerate(entities_to_process):
        if skip_actions:
          action_spec = entity_lib.ActionSpec(
              call_to_action='',
              output_type=entity_lib.OutputType.SKIP_THIS_STEP,
          )
        else:
          action_spec = next_action_specs[i]
        tasks[entity.name] = functools.partial(
            _entity_act, entity, action_spec, skip_actions
        )

      # Run entity actions concurrently
      actions = concurrency.run_tasks(tasks)

      if skip_actions:
        continue

      resolve_input = '\n'.join(actions.values())
      self.resolve(game_master, resolve_input, verbose=verbose)
      if log is not None and hasattr(game_master, 'get_last_log'):
        assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
        log_entry['resolve'] = game_master.get_last_log()

      entity_logs = {}
      for entity_name in actions:
        entity = next(e for e in entities if e.name == entity_name)
        if hasattr(entity, 'get_last_log'):
          assert hasattr(entity, 'get_last_log')  # Assertion for pytype
          entity_logs[entity.name] = entity.get_last_log()

      steps += 1
      if log is not None:
        game_master_key = game_master.name
        self._log(
            log=log,
            steps=steps,
            entity_logs=entity_logs,
            game_master_key=game_master_key,
            game_master_log=log_entry,
        )
        log_entry = _get_empty_log_entry()
      if checkpoint_callback is not None:
        checkpoint_callback(steps)

  def _log(
      self,
      log: list[Mapping[str, Any]],
      steps: int,
      entity_logs: Mapping[str, Any],
      game_master_key: str,
      game_master_log: Mapping[str, Any],
  ):
    """Modify log in place to append a new entry."""
    game_master_finalized_log = {}
    for segment_key, segment_log in game_master_log.items():
      game_master_finalized_log[segment_key] = {}
      for component_key, component_value in segment_log.items():
        if component_value:
          tmp_log_dict = {
              key: value for key, value in component_value.items() if value
          }
          if len(tmp_log_dict) > 1:
            # Only log if component logged more than just a key.
            game_master_finalized_log[segment_key][component_key] = tmp_log_dict

    log_entry = {
        'Step': steps,
        game_master_key: game_master_finalized_log,
        'Summary': f'Step {steps} {game_master_key}',
    }
    for entity_name, entity_log in entity_logs.items():
      log_entry[f'Entity [{entity_name}]'] = entity_log
    log.append(log_entry)
