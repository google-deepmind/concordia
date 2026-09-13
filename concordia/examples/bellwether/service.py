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

"""One authoritative Bellwether fixture service used by all three transports."""

from collections.abc import Callable
import copy
import logging
import pathlib
import threading

from concordia.environment.engines import sequential
from concordia.examples.astral_canticle import human_io
from concordia.examples.bellwether import scenario
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.typing import entity_component
from concordia.utils import operation_service as ops
from concordia.utils import simulation_server
import numpy as np


class Bellwether:
  """One-turn fixture; residents are built but do not yet make live decisions.

  The only editable target is a designated Constant on Mara. No general memory,
  cancellation, branching, checkpoint restore or multi-field transactions are
  advertised. Pause requests do not authorize edits while any worker is alive.
  """

  backend = 'fixture'
  player_audiences = ('player',)
  response_handlers = None
  view_handlers = None
  initial_status = 'Ready for the Bellwether fixture'
  run_description = (
      'Run one fixture turn through standard Sequential. Cannot restart or'
      ' replace.'
  )
  response_description = 'Submit the pending coordinator action (fixture only).'

  def __init__(
      self,
      output: pathlib.Path,
      *,
      session_id=None,
      port=0,
      config_factory=None,
      model=None,
      embedder: Callable[[str], np.ndarray] | None = None,
      max_steps=1
  ):
    if embedder is not None and not callable(embedder):
      raise ValueError('embedder must be a runtime callable')
    self.embedding = (
        {'kind': 'constant_placeholder', 'dimensions': 8}
        if embedder is None
        else {'kind': 'provided_callable'}
    )
    self.operations = ops.OperationService(
        project_id='bellwether-fixture-v1', session_id=session_id
    )
    self.inbox = human_io.HumanSession(
        on_request=self._requested,
        initial_status=self.initial_status,
    )
    self.config = (config_factory or scenario.configuration)(self.inbox)
    self.max_steps = max_steps
    self.simulation = generic.Simulation(
        self.config,
        model or no_language_model.NoLanguageModel(),
        embedder if embedder is not None else lambda _: np.ones(8),
        engine=sequential.Sequential(),
    )
    self.output = output
    self.phase = 'paused'
    self._worker: threading.Thread | None = None
    self._monitor = None
    self._failure = None
    self._log = None
    self.server = simulation_server.SimulationServer(
        port=port, operation_service=self.operations
    )
    self.server.set_simulation(self.simulation)
    self._seed()
    self._checkpoint = self.simulation.make_checkpoint_data()
    self.operations.set_view('developer', self.developer_view)
    self.operations.set_view('player', self.player_view)
    self._register()

  def _seed(self):
    for actor in self.simulation.get_entities():
      if actor.name == scenario.PLAYER:
        actor.observe(scenario.PUBLIC['opening'])
      else:
        actor.observe(scenario.RESIDENTS[actor.name][1])

  def _register(self):
    register = self.operations.register
    register(
        ops.Operation(
            'session.inspect',
            'Inspect last safe runtime snapshot, initial mechanics and trace.',
            {},
            lambda _: self.developer_view(),
        )
    )
    register(
        ops.Operation(
            'player.view',
            'Only the human coordinator’s delivered information.',
            {},
            lambda _: self.player_view(),
            audiences=self.player_audiences,
            audience_handlers=self.view_handlers,
        )
    )
    register(
        ops.Operation(
            'component.edit',
            'Edit Mara’s designated previous-storm account at a quiescent'
            ' boundary.',
            {
                'value': ops.Parameter(
                    'string',
                    'Exact replacement account; does not inform anyone else.',
                )
            },
            self._edit,
            mutation=True,
        )
    )
    register(
        ops.Operation(
            'run.start',
            self.run_description,
            {},
            self._start,
            mutation=True,
        )
    )
    register(
        ops.Operation(
            'run.pause',
            'Request pause; in-flight edits remain rejected until execution'
            ' fully returns.',
            {},
            self._pause,
            mutation=True,
        )
    )
    register(
        ops.Operation(
            'human.respond',
            self.response_description,
            {
                'request_id': ops.Parameter(
                    'string', 'Current pending request ID'
                ),
                'response': ops.Parameter('string', 'Your action attempt'),
            },
            self._respond,
            audiences=self.player_audiences,
            audience_handlers=self.response_handlers,
            mutation=True,
        )
    )

  def _requested(self):
    self.operations.publish({'kind': 'human.requested'})

  def player_view(self):
    return {
        'scenario': copy.deepcopy(scenario.PUBLIC),
        'human': self.inbox.snapshot(),
        'phase': self.phase,
        'fixture': True,
    }

  def developer_view(self):
    return {
        'phase': self.phase,
        'quiescent': self._worker is None or not self._worker.is_alive(),
        'initial': copy.deepcopy(scenario.INITIAL),
        'embedding': copy.deepcopy(self.embedding),
        'target': self._account(),
        'target_path': ['Mara', scenario.ACCOUNT, 'state'],
        'components_at_last_boundary': copy.deepcopy(self._checkpoint),
        'trace': (
            self.simulation.get_raw_log() if self.phase == 'completed' else []
        ),
        'events': self.operations.events(),
        'failure': self._failure,
        'capability_gaps': [
            'live residents',
            'watch accounting',
            'checkpoints/restore',
            'branches/experiments',
            'in-flight edits/cancellation',
            'initial-project authoring',
            'multi-field transactions',
            'undo/redo',
        ],
    }

  def _account(self):
    mara = next(e for e in self.simulation.get_entities() if e.name == 'Mara')
    assert isinstance(mara, entity_component.EntityWithComponents)
    return mara.get_component(scenario.ACCOUNT).get_state()['state']

  def _edit(self, args):
    if self._worker is not None and self._worker.is_alive():
      raise ops.OperationError(
          'not_quiescent',
          'Execution is still active (including pending human input); finish'
          ' this fixture first.',
      )
    if not self.server.step_controller.is_paused:
      raise ops.OperationError('not_paused', 'Pause before editing.')
    before = self._account()
    self.simulation.set_component_dynamic_state(
        'Mara', scenario.ACCOUNT, 'state', args['value']
    )
    self._checkpoint = self.simulation.make_checkpoint_data()
    self.operations.publish({
        'kind': 'component.edited',
        'target': ['Mara', scenario.ACCOUNT, 'state'],
        'before': before,
        'after': args['value'],
        'effective': 'next entity act',
        'informs': [],
    })
    return {'value': self._account()}

  def _start(self, args):
    del args
    if self.phase != 'paused' or self._worker is not None:
      raise ops.OperationError(
          'run_exists',
          'This run cannot be replaced/replayed; create a new service for a'
          ' fresh fixture.',
      )
    self.phase = 'running'
    self.server.step_controller.play()
    self._worker = threading.Thread(target=self._execute, daemon=True)
    self._worker.start()
    self._monitor = threading.Thread(target=self._join, daemon=True)
    self._monitor.start()
    self.operations.publish({'kind': 'run.started', 'backend': self.backend})
    return {'phase': self.phase}

  def _execute(self):
    try:
      self._log = self.simulation.play(
          max_steps=self.max_steps,
          step_controller=self.server.step_controller,
          step_callback=self._step,
      )
      self.output.mkdir(parents=True, exist_ok=True)
      (self.output / 'simulation.json').write_text(
          self._log.to_json(), encoding='utf-8'
      )
      (self.output / 'log.html').write_text(
          self._log.to_html(), encoding='utf-8'
      )
    except Exception as error:  # pylint: disable=broad-exception-caught
      logging.exception('Bellwether execution failed')
      self._failure = type(error).__name__

  def _join(self):
    assert self._worker is not None
    self._worker.join()
    with self.operations.lock:
      self.phase = 'failed' if self._failure else 'completed'
      self.server.broadcast_completion()
      self._checkpoint = self.simulation.make_checkpoint_data()
      self._finish()
      self.operations.publish(
          {'kind': 'run.' + self.phase, 'backend': self.backend}
      )

  def _step(self, data):
    self.server.broadcast_step(data)

  def _finish(self):
    if not self._failure:
      self.inbox.add_observation(scenario.RESOLUTION)
    self.inbox.finish(
        'Fixture complete.' if not self._failure else 'Fixture stopped.'
    )

  def _pause(self, args):
    del args
    self.server.step_controller.pause()
    self.operations.publish({'kind': 'run.pause_requested'})
    return {'quiescent': self._worker is None or not self._worker.is_alive()}

  def _respond(self, args):
    try:
      changed = self.inbox.submit(args['request_id'], args['response'])
    except (ValueError, TypeError) as error:
      raise ops.OperationError('invalid_action', str(error)) from error
    if changed:
      self.operations.publish(
          {'kind': 'human.accepted', 'action': args['response']}
      )
    return {'accepted': changed}

  def close(self):
    self.inbox.finish('Service closed.')
    self.server.step_controller.stop()
    if self._monitor is not None:
      self._monitor.join(timeout=10)
    self.server.stop()
