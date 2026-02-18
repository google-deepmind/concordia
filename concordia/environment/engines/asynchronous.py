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

"""Fully asynchronous engine.

Each player entity runs its own independent observe-act loop concurrently.
Unlike the simultaneous engine (which synchronizes all players per round),
players interact with the game master independently and at their own pace.
The game master and its components must be thread-safe.
"""

from collections.abc import Mapping, Sequence
import functools
import threading
import time
from typing import Any, Callable, override

from absl import logging
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import make_observation as make_observation_component
from concordia.components.game_master import next_acting as next_acting_components
from concordia.components.game_master import switch_act as switch_act_component
from concordia.environment import engine as engine_lib
from concordia.environment import step_controller as step_controller_lib
from concordia.typing import entity as entity_lib
from concordia.utils import concurrency
import termcolor


DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    make_observation_component.DEFAULT_CALL_TO_MAKE_OBSERVATION
)
DEFAULT_CALL_TO_NEXT_ACTING = 'Which entities act next?'
DEFAULT_CALL_TO_NEXT_ACTION_SPEC = (
    next_acting_components.DEFAULT_CALL_TO_NEXT_ACTION_SPEC
)
DEFAULT_CALL_TO_RESOLVE = 'Because of all that came before, what happens next?'
DEFAULT_CALL_TO_CHECK_TERMINATION = 'Is the game/simulation finished?'
DEFAULT_CALL_TO_NEXT_GAME_MASTER = (
    'Which rule set should we use for the next step?'
)

DEFAULT_ACT_COMPONENT_KEY = switch_act_component.DEFAULT_ACT_COMPONENT_KEY

PUTATIVE_EVENT_TAG = event_resolution_components.PUTATIVE_EVENT_TAG
EVENT_TAG = event_resolution_components.EVENT_TAG

_PRINT_COLOR = 'cyan'
_DEFAULT_SLEEP_TIME = 0.1


def _get_empty_log_entry():
  return {
      'terminate': {},
      'make_observation': {},
      'next_acting': {},
      'next_action_spec': {},
      'resolve': {},
  }


class Asynchronous(engine_lib.Engine):
  """Fully asynchronous engine.

  Each player entity runs its own independent observe-act loop concurrently in
  a separate thread. Players interact with the game master independently and
  at their own pace. The game master is assumed to be thread-safe.

  Termination uses both a shared threading.Event (for global signaling) and a
  per-player max iterations cap.

  A global pause event is supported for UI play/pause functionality.
  """

  def __init__(
      self,
      call_to_make_observation: str = DEFAULT_CALL_TO_MAKE_OBSERVATION,
      call_to_next_acting: str = DEFAULT_CALL_TO_NEXT_ACTING,
      call_to_next_action_spec: str = DEFAULT_CALL_TO_NEXT_ACTION_SPEC,
      call_to_resolve: str = DEFAULT_CALL_TO_RESOLVE,
      call_to_check_termination: str = DEFAULT_CALL_TO_CHECK_TERMINATION,
      call_to_next_game_master: str = DEFAULT_CALL_TO_NEXT_GAME_MASTER,
      sleep_time: float = _DEFAULT_SLEEP_TIME,
  ):
    self._call_to_make_observation = call_to_make_observation
    self._call_to_next_acting = call_to_next_acting
    self._call_to_next_action_spec = call_to_next_action_spec
    self._call_to_resolve = call_to_resolve
    self._call_to_check_termination = call_to_check_termination
    self._call_to_next_game_master = call_to_next_game_master
    self._sleep_time = sleep_time
    self._pause_event = threading.Event()
    self._pause_event.set()

  def pause(self) -> None:
    """Pause all player threads. They will block until play() is called."""
    self._pause_event.clear()

  def play(self) -> None:
    """Resume all player threads after a pause."""
    self._pause_event.set()

  def make_observation(
      self, game_master: entity_lib.Entity, entity: entity_lib.Entity
  ) -> str:
    observation = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_make_observation.format(
                name=entity.name
            ),
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
  ) -> tuple[
      Sequence[entity_lib.Entity], Sequence[entity_lib.ActionSpec]
  ]:  # pytype: disable=signature-mismatch
    entities_by_name = {entity.name: entity for entity in entities}
    next_object_names_string = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_next_acting,
            output_type=entity_lib.OutputType.NEXT_ACTING,
            options=tuple(entities_by_name.keys()),
        )
    )
    next_entity_names = [
        name.strip() for name in next_object_names_string.split(',')
    ]
    if (
        log is not None
        and log_entry is not None
        and hasattr(game_master, 'get_last_log')
    ):
      assert hasattr(game_master, 'get_last_log')
      log_entry['next_acting'] = game_master.get_last_log()

    action_spec_by_name = {}
    for next_entity_name in next_entity_names:
      next_action_spec_string = game_master.act(
          action_spec=entity_lib.ActionSpec(
              call_to_action=self._call_to_next_action_spec.format(
                  name=next_entity_name
              ),
              output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
          )
      )
      action_spec_by_name[next_entity_name] = engine_lib.action_spec_parser(
          next_action_spec_string
      )

      if (
          log is not None
          and log_entry is not None
          and hasattr(game_master, 'get_last_log')
      ):
        assert hasattr(game_master, 'get_last_log')
        log_entry['next_action_spec'] = game_master.get_last_log()

    invalid_names = [
        name for name in next_entity_names if name not in entities_by_name
    ]
    if invalid_names:
      raise ValueError(
          f'Game master returned invalid entity names: {invalid_names}. '
          f'Valid options: {list(entities_by_name.keys())}'
      )

    return (
        [entities_by_name[entity_name] for entity_name in next_entity_names],
        [action_spec_by_name[entity_name] for entity_name in next_entity_names],
    )

  def resolve(
      self,
      game_master: entity_lib.Entity,
      putative_event: str,
      verbose: bool = False,
  ) -> None:
    if verbose:
      print(
          termcolor.colored(
              f'The suggested action or event to resolve was: {putative_event}',
              _PRINT_COLOR,
          )
      )
    game_master.observe(observation=f'{PUTATIVE_EVENT_TAG} {putative_event}')
    result = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_resolve,
            output_type=entity_lib.OutputType.RESOLVE,
        )
    )
    game_master.observe(observation=f'{EVENT_TAG} {result}')
    if verbose:
      print(
          termcolor.colored(f'The resolved event was: {result}', _PRINT_COLOR)
      )

  def _log(
      self,
      log: list[Mapping[str, Any]],
      steps: int,
      entity_key: str,
      entity_log: Mapping[str, Any],
      game_master_key: str,
      game_master_log: Mapping[str, Any],
      thread: str = '',
  ):
    game_master_finalized_log = {}
    for segment_key, segment_log in game_master_log.items():
      game_master_finalized_log[segment_key] = {}
      for component_key, component_value in segment_log.items():
        if component_value:
          tmp_log_dict = {
              key: value for key, value in component_value.items() if value
          }
          if len(tmp_log_dict) > 1:
            game_master_finalized_log[segment_key][component_key] = tmp_log_dict

    entry = {
        'Step': steps,
        entity_key: entity_log,
        game_master_key: game_master_finalized_log,
        'Summary': f'Step {steps} {game_master_key}',
    }
    if thread:
      entry['thread'] = thread
    log.append(entry)

  def terminate(
      self, game_master: entity_lib.Entity, verbose: bool = False
  ) -> bool:
    should_terminate_string = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_check_termination,
            output_type=entity_lib.OutputType.TERMINATE,
            options=tuple(entity_lib.BINARY_OPTIONS.values()),
        )
    )
    if verbose:
      print(
          termcolor.colored(
              f'Terminate? {should_terminate_string}', _PRINT_COLOR
          )
      )
    return should_terminate_string == entity_lib.BINARY_OPTIONS['affirmative']

  def next_game_master(
      self,
      game_master: entity_lib.Entity,
      game_masters: Sequence[entity_lib.Entity],
      verbose: bool = False,
  ) -> entity_lib.Entity:
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
      print(
          termcolor.colored(
              f'Game master: {next_game_master_name}', _PRINT_COLOR
          )
      )
    if next_game_master_name not in game_masters_by_name:
      raise ValueError(
          f'Selected game master {next_game_master_name} not found in:'
          f' {game_masters_by_name.keys()}'
      )
    return game_masters_by_name[next_game_master_name]

  def _entity_loop(
      self,
      entity: entity_lib.Entity,
      entities: Sequence[entity_lib.Entity],
      game_master: entity_lib.Entity,
      max_steps: int,
      verbose: bool,
      terminate_event: threading.Event,
      log: list[Mapping[str, Any]] | None = None,
      checkpoint_callback: Callable[[int], None] | None = None,
      step_callback: (
          Callable[[step_controller_lib.StepData], None] | None
      ) = None,
      step_controller: step_controller_lib.StepController | None = None,
  ) -> None:
    """Run the observe-act loop for a single entity.

    Args:
      entity: The entity to run.
      entities: All entities in the simulation.
      game_master: The game master (must be thread-safe).
      max_steps: Maximum number of iterations for this entity.
      verbose: Whether to print debug information.
      terminate_event: Shared event signaling global termination.
      log: Optional log list for structured logging.
      checkpoint_callback: Optional callback invoked after each iteration.
      step_callback: Optional callback invoked after each iteration with step
        data.
      step_controller: Optional controller to manage stepping through the
        simulation.
    """
    iteration = 0
    while not terminate_event.is_set() and iteration < max_steps:
      if step_controller is not None:
        if not step_controller.wait_for_step_permission():
          terminate_event.set()
          break

      self._pause_event.wait()

      if terminate_event.is_set():
        break

      if self.terminate(game_master, verbose):
        terminate_event.set()
        break

      log_entry = _get_empty_log_entry()

      if log is not None and hasattr(game_master, 'get_last_log'):
        assert hasattr(game_master, 'get_last_log')
        log_entry['terminate'] = game_master.get_last_log()

      acting_entities, action_specs = self.next_acting(
          game_master, entities, log_entry=log_entry, log=log
      )
      acting_names = [e.name for e in acting_entities]
      iteration += 1
      if entity.name not in acting_names:
        time.sleep(self._sleep_time)
        continue

      entity_index = acting_names.index(entity.name)
      action_spec = action_specs[entity_index]

      if action_spec.output_type == entity_lib.OutputType.SKIP_THIS_STEP:
        time.sleep(self._sleep_time)
        continue

      observation = self.make_observation(game_master, entity)
      if log is not None and hasattr(game_master, 'get_last_log'):
        assert hasattr(game_master, 'get_last_log')
        log_entry['make_observation'][entity.name] = game_master.get_last_log()
      if observation and observation.strip():
        if verbose:
          print(
              termcolor.colored(
                  f'Entity {entity.name} observed: {observation}',
                  _PRINT_COLOR,
              )
          )
        entity.observe(observation)

      if verbose:
        print(
            termcolor.colored(
                f'Entity {entity.name} is next to act. They must respond'
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

      self.resolve(game_master, action, verbose=verbose)

      if log is not None and hasattr(game_master, 'get_last_log'):
        assert hasattr(game_master, 'get_last_log')
        log_entry['resolve'] = game_master.get_last_log()
        next_entity_log = {}
        game_master_key = game_master.name
        entity_key = f'Entity [{entity.name}]'
        if hasattr(entity, 'get_last_log'):
          next_entity_log = entity.get_last_log()
        if DEFAULT_ACT_COMPONENT_KEY in log_entry['resolve']:
          event_to_log = log_entry['resolve'][DEFAULT_ACT_COMPONENT_KEY][
              'Value'
          ]
          game_master_key = f'{game_master_key} --- {event_to_log}'
        self._log(
            log=log,
            steps=iteration,
            entity_key=entity_key,
            entity_log=next_entity_log,
            game_master_key=game_master_key,
            game_master_log=log_entry,
            thread=entity.name,
        )

      if checkpoint_callback is not None:
        logging.debug(
            'Calling checkpoint callback for %s at iteration %s',
            entity.name,
            iteration,
        )
        checkpoint_callback(iteration)

      if step_callback is not None:
        entity_logs = {}
        if hasattr(entity, 'get_last_log'):
          assert hasattr(entity, 'get_last_log')
          entity_logs[entity.name] = entity.get_last_log()
        step_data = step_controller_lib.StepData(
            step=iteration,
            acting_entity=entity.name,
            action=action,
            entity_actions={entity.name: action},
            entity_logs=entity_logs,
            game_master=game_master.name,
        )
        step_callback(step_data)

  def run_loop(
      self,
      game_masters: Sequence[entity_lib.Entity],
      entities: Sequence[entity_lib.Entity],
      premise: str = '',
      max_steps: int = 10,
      verbose: bool = False,
      log: list[Mapping[str, Any]] | None = None,
      checkpoint_callback: Callable[[int], None] | None = None,
      step_controller=None,
      step_callback=None,
  ):
    if not game_masters:
      raise ValueError('No game masters provided.')

    game_master = game_masters[0]
    if premise:
      premise = f'{EVENT_TAG} {premise}'
      game_master.observe(premise)

    init_steps = 0
    max_init_steps = max_steps
    while init_steps < max_init_steps:
      if self.terminate(game_master, verbose):
        break

      game_master = self.next_game_master(game_master, game_masters, verbose)

      acting_entities, action_specs = self.next_acting(
          game_master, entities, log_entry=None, log=log
      )

      if not acting_entities:
        break

      action_spec = action_specs[0]
      if action_spec.output_type == entity_lib.OutputType.SKIP_THIS_STEP:
        if verbose:
          print(
              termcolor.colored(
                  '\nSkipping the action phase for the current time step.\n',
                  _PRINT_COLOR,
              )
          )

        # Flush observations to all entities. This is crucial for ensuring
        # observations are properly delivered whenever there is a transition
        # between game masters. Without this, observations queued by the
        # outgoing game master would be lost when control transfers to the
        # next game master.
        for entity in entities:
          observation = self.make_observation(game_master, entity)
          if observation and observation.strip():
            if verbose:
              print(
                  termcolor.colored(
                      f'Entity {entity.name} observed: {observation}',
                      _PRINT_COLOR,
                  )
              )
            entity.observe(observation)

        init_steps += 1
        if step_callback is not None:
          step_data = step_controller_lib.StepData(
              step=init_steps,
              acting_entity='(setup)',
              action='Skipping action phase',
              entity_actions={},
              entity_logs={},
              game_master=game_master.name,
          )
          step_callback(step_data)
        continue

      break

    terminate_event = threading.Event()

    tasks = {}
    for entity in entities:
      tasks[entity.name] = functools.partial(
          self._entity_loop,
          entity=entity,
          entities=entities,
          game_master=game_master,
          max_steps=max_steps,
          verbose=verbose,
          terminate_event=terminate_event,
          log=log,
          checkpoint_callback=checkpoint_callback,
          step_callback=step_callback,
          step_controller=step_controller,
      )

    concurrency.run_tasks(tasks)
