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
import re
import threading
from typing import Any, Callable, override

from absl import logging
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import make_observation as make_observation_component
from concordia.components.game_master import next_acting as next_acting_components
from concordia.components.game_master import switch_act as switch_act_component
from concordia.environment import engine as engine_lib
from concordia.environment import step_controller as step_controller_lib
from concordia.typing import entity as entity_lib
from concordia.utils import async_log_collector as collector_lib
from concordia.utils import async_measurements as async_measurements_lib
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
_BASE64_TRUNCATE_PATTERN = re.compile(r'(base64,)[A-Za-z0-9+/=]{100,}')

PUTATIVE_EVENT_TAG = event_resolution_components.PUTATIVE_EVENT_TAG
EVENT_TAG = event_resolution_components.EVENT_TAG

_PRINT_COLOR = 'cyan'
_DEFAULT_SLEEP_TIME = 0.1
# How often run_loop looks for entities that joined the list.
_MEMBERSHIP_POLL_SECONDS = 0.05
_DEFAULT_LOG_BATCH_SIZE = 64


def _get_reactive_measurements(
    entity: entity_lib.Entity,
) -> async_measurements_lib.ReactiveMeasurements:
  """Returns the ReactiveMeasurements instance from an entity."""
  if not hasattr(entity, 'measurements'):
    raise ValueError(f'Entity {entity.name} has no measurements property. ')
  measurements = entity.measurements
  if not isinstance(measurements, async_measurements_lib.ReactiveMeasurements):
    raise ValueError(
        f'Entity {entity.name} uses {type(measurements).__name__} but the '
        'asynchronous engine requires ReactiveMeasurements. Pass '
        'measurements=ReactiveMeasurements() to EntityAgentWithLogging.'
    )
  return measurements


def _get_empty_log_entry():
  return {
      'terminate': {},
      'make_observation': {},
      'next_acting': {},
      'next_action_spec': {},
      'resolve': {},
  }


def _is_member(
    entity: entity_lib.Entity, entities: Sequence[entity_lib.Entity]
) -> bool:
  """Whether this very entity (not just one of its name) is in `entities`."""
  return any(member is entity for member in list(entities))


def _first_entity_checkpoints(
    callback: Callable[[int], None],
    entities: Sequence[entity_lib.Entity],
) -> Callable[[entity_lib.Entity, int], None]:
  """Wraps a checkpoint callback so that the first entity's loop calls it.

  The loop of the first entity in `entities` checkpoints after each of its
  iterations, as entity 0's loop always has. If that entity leaves the list,
  the next first entity's loop takes over, and step numbers go on rising.

  Args:
    callback: The checkpoint callback, called with the step number.
    entities: The entities the engine runs.

  Returns:
    A thread-safe callback, called by a loop with its entity and iteration.
  """
  lock = threading.Lock()
  last_step = 0

  def checkpoint(entity: entity_lib.Entity, iteration: int) -> None:
    nonlocal last_step
    members = list(entities)
    if not members or members[0] is not entity:
      return
    with lock:
      last_step = max(iteration, last_step + 1)
      callback(last_step)

  return checkpoint


class Asynchronous(engine_lib.Engine):
  """Fully asynchronous engine.

  Each player entity runs its own independent observe-act loop concurrently in
  a separate thread. Players interact with the game master independently and
  at their own pace. The game master is assumed to be thread-safe.

  Termination uses both a shared threading.Event (for global signaling) and a
  per-player max iterations cap.

  The entities may change while the loop runs: loops follow the `entities`
  list passed to `run_loop` (via `concurrency.run_dynamic_tasks`). An entity
  appended to it gets a loop of its own; one removed from it ends its loop at
  its next iteration. Checkpoints are taken by the loop of the list's first
  entity, after each of its iterations.

  Log events are finalized into the log in batches as the loop runs
  (`log_batch_size`), so a long run keeps no unbounded buffer of them.

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
      log_batch_size: int | None = _DEFAULT_LOG_BATCH_SIZE,
  ):
    """Asynchronous engine constructor.

    Args:
      call_to_make_observation: The call to action asking for an observation.
      call_to_next_acting: The call to action asking who acts next.
      call_to_next_action_spec: The call to action asking for an action spec.
      call_to_resolve: The call to action asking to resolve an event.
      call_to_check_termination: The call to action asking whether to stop.
      call_to_next_game_master: The call to action asking for the next GM.
      sleep_time: Seconds an entity waits after a step in which it may not act.
      log_batch_size: How many log events may wait before they are finalized
        into the log. None waits for a pause or the end of the run.

    Raises:
      ValueError: If log_batch_size is not positive.
    """
    if log_batch_size is not None and log_batch_size < 1:
      raise ValueError(f'log_batch_size must be positive: {log_batch_size}')
    self._log_batch_size = log_batch_size
    self._call_to_make_observation = call_to_make_observation
    self._call_to_next_acting = call_to_next_acting
    self._call_to_next_action_spec = call_to_next_action_spec
    self._call_to_resolve = call_to_resolve
    self._call_to_check_termination = call_to_check_termination
    self._call_to_next_game_master = call_to_next_game_master
    self._sleep_time = sleep_time
    self._pause_event = threading.Event()
    self._pause_event.set()
    self._collector = collector_lib.AsyncLogCollector()
    self._log_list: list[Mapping[str, Any]] | None = None

  def pause(self) -> None:
    """Pause all player threads. They will block until play() is called."""
    self._pause_event.clear()
    if self._log_list is not None:
      self._collector.materialize(self._log_list)

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
  def next_acting(  # pyrefly: ignore[bad-override]
      self,
      game_master: entity_lib.Entity,
      entities: Sequence[entity_lib.Entity],
      log_entry: dict[str, Any] | None = None,
      log: list[Mapping[str, Any]] | None = None,
      gm_measurements: (
          async_measurements_lib.ReactiveMeasurements | None
      ) = None,
      capture_key: str | None = None,
  ) -> tuple[Sequence[entity_lib.Entity], Sequence[entity_lib.ActionSpec]]:
    entities_by_name = {entity.name: entity for entity in entities}
    if not entities_by_name:
      return ([], [])

    if gm_measurements is not None and log_entry is not None:
      key = capture_key or game_master.name
      with gm_measurements.capture(key) as captured:
        next_object_names_string = game_master.act(
            action_spec=entity_lib.ActionSpec(
                call_to_action=self._call_to_next_acting,
                output_type=entity_lib.OutputType.NEXT_ACTING,
                options=tuple(entities_by_name.keys()),
            )
        )
      log_entry['next_acting'] = dict(captured)
    else:
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
    next_entity_names = [
        name for name in next_entity_names if name in entities_by_name
    ]
    if not next_entity_names:
      return ([], [])

    action_spec_by_name = {}
    for next_entity_name in next_entity_names:
      if gm_measurements is not None and log_entry is not None:
        key = capture_key or game_master.name
        with gm_measurements.capture(key) as captured:
          next_action_spec_string = game_master.act(
              action_spec=entity_lib.ActionSpec(
                  call_to_action=self._call_to_next_action_spec.format(
                      name=next_entity_name
                  ),
                  output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
              )
          )
        log_entry['next_action_spec'] = dict(captured)
      else:
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

    return (
        [entities_by_name[entity_name] for entity_name in next_entity_names],
        [action_spec_by_name[entity_name] for entity_name in next_entity_names],
    )

  def resolve(  # pyrefly: ignore[bad-override]
      self,
      game_master: entity_lib.Entity,
      putative_event: str,
      verbose: bool = False,
  ) -> None:
    if verbose:
      display_event = _BASE64_TRUNCATE_PATTERN.sub(
          r'\1[IMAGE DATA]', putative_event
      )
      print(
          termcolor.colored(
              f'The suggested action or event to resolve was: {display_event}',
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
      game_master: entity_lib.Entity,
      max_steps: int,
      verbose: bool,
      terminate_event: threading.Event,
      game_masters: Sequence[entity_lib.Entity] | None = None,
      log: list[Mapping[str, Any]] | None = None,
      checkpoint_callback: Callable[[int], None] | None = None,
      step_callback: (
          Callable[[step_controller_lib.StepData], None] | None
      ) = None,
      step_controller: step_controller_lib.StepController | None = None,
      entities: Sequence[entity_lib.Entity] | None = None,
  ) -> None:
    """Run the observe-act loop for a single entity.

    Args:
      entity: The entity to run.
      game_master: The game master (must be thread-safe).
      max_steps: Maximum number of iterations for this entity.
      verbose: Whether to print debug information.
      terminate_event: Shared event signaling global termination.
      game_masters: Optional sequence of all game masters for multi-GM
        switching.
      log: Optional log list for structured logging.
      checkpoint_callback: Optional callback invoked after each iteration.
      step_callback: Optional callback invoked after each iteration with step
        data.
      step_controller: Optional controller to manage stepping through the
        simulation.
      entities: Optional live population. If given, the loop ends at the start
        of the first iteration in which `entity` is no longer in it.
    """
    if game_masters is None:
      game_masters = [game_master]
    gm_measurements = _get_reactive_measurements(game_master)
    entity_measurements = _get_reactive_measurements(entity)

    # Register this thread's entity name on the game master so that
    # game_master.act()/observe() publish log data with entity.name as
    # the capture_key instead of game_master.name. This prevents
    # cross-thread log contamination when multiple entity threads share
    # the same game master.
    thread_id = threading.current_thread().ident
    if hasattr(game_master, 'set_capture_key_for_thread'):
      game_master.set_capture_key_for_thread(thread_id, entity.name)

    iteration = 0
    try:
      while not terminate_event.is_set() and iteration < max_steps:
        if step_controller is not None:
          if not step_controller.wait_for_step_permission():
            terminate_event.set()
            break

        while not self._pause_event.wait(timeout=_MEMBERSHIP_POLL_SECONDS):
          if terminate_event.is_set() or (
              entities is not None and not _is_member(entity, entities)
          ):
            break

        if terminate_event.is_set():
          break

        if entities is not None and not _is_member(entity, entities):
          break

        # --- Multi-GM switching ---
        # When multiple game masters are registered, ask the current GM
        # which GM should run next.  This enables time-based transitions
        if len(game_masters) > 1:
          new_gm = self.next_game_master(game_master, game_masters, verbose)
          if new_gm is not game_master:
            # Flush pending observations from the outgoing GM so they
            # are not lost during the transition.
            observation = self.make_observation(game_master, entity)
            if observation and observation.strip():
              entity.observe(observation)
            # Transfer thread capture key to the new GM.
            if hasattr(game_master, 'clear_capture_key_for_thread'):
              game_master.clear_capture_key_for_thread(thread_id)
            game_master = new_gm
            if hasattr(game_master, 'set_capture_key_for_thread'):
              game_master.set_capture_key_for_thread(thread_id, entity.name)
            gm_measurements = _get_reactive_measurements(game_master)

        with gm_measurements.capture(entity.name) as terminate_log:
          should_terminate = self.terminate(game_master, verbose)

        if should_terminate:
          terminate_event.set()
          break

        log_entry = _get_empty_log_entry()
        if log is not None:
          log_entry['terminate'] = terminate_log

        acting_entities, action_specs = self.next_acting(
            game_master,
            [entity],
            log_entry=log_entry,
            log=log,
            gm_measurements=gm_measurements,
            capture_key=entity.name,
        )
        iteration += 1
        if not acting_entities:
          terminate_event.wait(self._sleep_time)
          continue

        action_spec = action_specs[0]

        if action_spec.output_type == entity_lib.OutputType.SKIP_THIS_STEP:
          terminate_event.wait(self._sleep_time)
          continue

        with gm_measurements.capture(entity.name) as obs_log:
          observation = self.make_observation(game_master, entity)
        if log is not None:
          log_entry['make_observation'][entity.name] = obs_log

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

        with entity_measurements.capture(entity.name) as entity_act_log:
          raw_action = entity.act(action_spec)

        if raw_action.startswith(f'{entity.name}:'):
          action = raw_action
        else:
          action = f'{entity.name}: {raw_action}'
        if verbose:
          display_action = _BASE64_TRUNCATE_PATTERN.sub(
              r'\1[IMAGE DATA]', action
          )
          print(
              termcolor.colored(
                  f'Entity {entity.name} chose action: {display_action}',
                  _PRINT_COLOR,
              )
          )

        with gm_measurements.capture(entity.name) as resolve_log:
          self.resolve(game_master, action, verbose=verbose)
        if log is not None:
          log_entry['resolve'] = resolve_log

        if log is not None:
          self._collector.emit(
              collector_lib.RawLogEvent(
                  step=iteration,
                  entity_name=entity.name,
                  game_master_name=game_master.name,
                  entity_log=dict(entity_act_log),
                  game_master_log=log_entry,
                  action=action,
              )
          )
          if (
              self._log_batch_size is not None
              and self._collector.pending_count() >= self._log_batch_size
          ):
            self._collector.materialize(log)

        if checkpoint_callback is not None:
          logging.debug(
              'Calling checkpoint callback for %s at iteration %s',
              entity.name,
              iteration,
          )
          checkpoint_callback(iteration)

        if step_callback is not None:
          step_data = step_controller_lib.StepData(
              step=iteration,
              acting_entity=entity.name,
              action=action,
              entity_actions={entity.name: action},
              entity_logs={entity.name: dict(entity_act_log)},
              game_master=game_master.name,
          )
          step_callback(step_data)
    except BaseException:
      terminate_event.set()
      self._pause_event.set()
      if step_controller is not None:
        step_controller.stop()
      raise
    finally:
      # Clean up thread mapping when entity loop exits.
      if hasattr(game_master, 'clear_capture_key_for_thread'):
        game_master.clear_capture_key_for_thread(thread_id)

  def run_loop(
      self,
      game_masters: Sequence[entity_lib.Entity],
      entities: Sequence[entity_lib.Entity],
      premise: str = '',
      max_steps: int = 10,
      verbose: bool = False,
      log: list[Mapping[str, Any]] | None = None,
      checkpoint_callback: Callable[[int], None] | None = None,
      step_controller: step_controller_lib.StepController | None = None,
      step_callback: (
          Callable[[step_controller_lib.StepData], None] | None
      ) = None,
  ):
    if not game_masters:
      raise ValueError('No game masters provided.')

    self._log_list = log

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
    # Only the first entity's loop saves checkpoints (make_checkpoint_data()
    # snapshots every entity and game master), as entity 0's always has; if
    # it leaves the list, the next first entity's loop takes over.
    checkpoint = (
        _first_entity_checkpoints(checkpoint_callback, entities)
        if checkpoint_callback is not None
        else None
    )

    def run_entity(entity: entity_lib.Entity) -> None:
      self._entity_loop(
          entity=entity,
          game_master=game_master,
          game_masters=game_masters,
          max_steps=max_steps,
          verbose=verbose,
          terminate_event=terminate_event,
          log=log,
          checkpoint_callback=(
              functools.partial(checkpoint, entity) if checkpoint else None
          ),
          step_callback=step_callback,
          step_controller=step_controller,
          entities=entities,
      )

    def current_tasks() -> Mapping[str, Callable[[], None]]:
      # One loop per entity now in the list (which may change while loops
      # run), keyed by identity so that a newcomer reusing the name of one who
      # left gets a loop of its own. Once terminating, no new loops start.
      if terminate_event.is_set():
        return {}
      return {
          f'{entity.name} ({id(entity)})': functools.partial(run_entity, entity)
          for entity in list(entities)
      }

    try:
      concurrency.run_dynamic_tasks(
          current_tasks, poll_interval=_MEMBERSHIP_POLL_SECONDS
      )
    except BaseException:
      terminate_event.set()  # Stop the loops still running.
      raise

    if log is not None:
      self._collector.materialize(log)
    self._log_list = None
