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

"""Capability adapter for the existing project server and graphical editor.

No execution loop or serialization codec lives here. Server project transactions,
OperationService's retry ledger, standard visual_interface rendering and the
StepController's acknowledged boundary own those responsibilities.
"""

from collections.abc import Callable
import copy
from typing import Any, TYPE_CHECKING
import uuid

from concordia.components.agent import constant
from concordia.typing import prefab as prefab_lib
from concordia.utils import operation_service as ops
from concordia.utils import project_config
from concordia.utils import visual_interface

if TYPE_CHECKING:
  from concordia.utils import simulation_server


class ProjectEditor:
  """One developer view attached to a SimulationServer before it starts."""

  def __init__(
      self,
      server: 'simulation_server.SimulationServer',
      registry: project_config.Registry,
      preview: Callable[[prefab_lib.Config], dict[str, Any]] | None,
  ):
    self.server = server
    self.registry = registry
    self.preview = preview
    self.service = ops.OperationService(project_id='registered-project')
    self.definition_view: dict[str, Any] | None = None
    self.runtime_view: dict[str, Any] | None = None
    self.runtime_config: prefab_lib.Config | None = None
    self.steps: list[dict[str, Any]] = []
    self.reset_requested = False
    self.stepping = False
    self.service.set_view('developer', self.snapshot)
    string = ops.Parameter('string', 'Literal text')
    self._register(
        'project.save',
        {
            'text': ops.Parameter('string', 'Registered project JSON', 900000),
            'revision': ops.Parameter('integer', 'Saved project revision'),
        },
        lambda args: server.replace_project(args['text'], args['revision']),
    )
    self._register(
        'project.run',
        {
            'revision': ops.Parameter('integer', 'Saved project revision'),
        },
        lambda args: server.run_project(args['revision']),
    )
    self._register('project.reset', {}, lambda _: self.reset())
    for command in ('play', 'pause', 'step'):
      self._register(
          'runtime.' + command,
          {},
          lambda _, command=command: self.command(command),
      )
    self._register(
        'runtime.edit',
        {
            'instance_id': string,
            'component': string,
            'value': string,
        },
        self.edit,
    )

  def _register(self, name, parameters, handler):
    def invoke(arguments):
      try:
        result = handler(arguments)
      except (ValueError, KeyError, TypeError, RuntimeError) as error:
        raise ops.OperationError('invalid_edit', str(error)) from error
      self.service.publish({'kind': name})
      return result

    self.service.register(
        ops.Operation(
            name=name,
            description=name,
            parameters=parameters,
            handler=invoke,
            mutation=True,
        )
    )

  def prepare(self, document: dict[str, Any]) -> dict[str, Any]:
    """Build an optional trusted inspection preview, never a running simulation."""
    config = self.registry.to_config(document)
    checkpoint = self.preview(config) if self.preview else None
    svg, entities = visual_interface.visualize_config(config, checkpoint)
    return {
        'svg': svg,
        'entities': entities,
        'inspector': self.registry.inspector(document),
        'catalog': self.registry.catalog(document),
        'component_catalog': self.registry.component_catalog(document),
    }

  def begin(self, config: prefab_lib.Config) -> None:
    # Called under the same lock as dispatch, before the run thread starts.
    self.service.references['run_id'] = str(uuid.uuid4())
    self.runtime_config = config
    self.runtime_view = None
    self.steps = []
    self.reset_requested = False
    self.stepping = False

  def state(self) -> str:
    run = self.server.get_project()['run']
    if run['status'] == 'not_started':
      return 'ready'
    if run['status'] != 'active':
      return run['status']
    if self.reset_requested or self.server.step_controller.should_stop():
      return 'stopping'
    if self.server.simulation is None:
      return 'starting'
    if self.server.step_controller.at_pause_boundary:
      return 'paused'
    if self.server.step_controller.is_running:
      return 'running'
    return 'stepping' if self.stepping else 'pausing'

  def snapshot(self) -> dict[str, Any]:
    return {
        **self.server.get_project(),
        'state': self.state(),
        'definition': self.definition_view,
        'runtime': self.runtime_view,
        'steps': self.steps,
        'current_step': self.server.get_status()['current_step'],
    }

  def receive_checkpoint(self, checkpoint: dict[str, Any]) -> None:
    with self.service.lock:
      if self.runtime_config is not None:
        svg, entities = visual_interface.visualize_config(
            self.runtime_config, checkpoint
        )
        self.runtime_view = {'svg': svg, 'entities': entities}
        self.service.publish({'kind': 'runtime.snapshot'})

  def record_step(self, step: dict[str, Any]) -> None:
    with self.service.lock:
      self.steps.append(copy.deepcopy(step))
      self.service.publish({'kind': 'runtime.step'})

  def finished(self) -> None:
    # The runner has returned; it will no longer use this controller/simulation.
    if self.reset_requested:
      self.server._project_run = {'status': 'not_started'}
    elif self.server.step_controller.should_stop():
      self.server._project_run['status'] = 'stopped'
    self.service.publish({'kind': 'runtime.finished'})

  def reset(self) -> None:
    if self.server.get_project()['run']['status'] == 'active':
      self.reset_requested = True
      self.server.step_controller.stop()
    else:
      self.server._project_run = {'status': 'not_started'}
    # Keep the previous runtime and log inspectable until the next explicit Run.

  def command(self, command: str) -> dict[str, Any]:
    required = {'play': 'paused', 'step': 'paused', 'pause': 'running'}
    if self.state() != required[command]:
      raise ValueError(f'Cannot {command} while {self.state()}.')
    result = self.server.execute_command(command)
    if result.get('status') == 'error':
      raise ValueError(result['message'])
    self.stepping = command == 'step'
    return result

  def edit(self, arguments: dict[str, Any]) -> None:
    if self.state() != 'paused':
      raise ValueError('Runtime edits require an acknowledged pause.')
    component_name = arguments['component']
    if component_name not in ('Instructions', 'Goal'):
      raise ValueError('Only registered Instructions/Goal text is editable.')
    arguments['value'].encode('utf-8')
    document = self.server.get_project()['document']
    instance = next(
        (
            item
            for item in document['instances']
            if item['id'] == arguments['instance_id']
        ),
        None,
    )
    if instance is None:
      raise ValueError('Unknown registered instance ID.')
    simulation = self.server.simulation
    with self.server.step_controller.paused_boundary():
      entity = next(
          (
              entity
              for entity in simulation.get_entities()
              if entity.name == instance['params']['name']
          ),
          None,
      )
      if entity is None:
        raise ValueError('Only actor text components are editable.')
      component = entity.get_component(component_name)
      if not isinstance(component, constant.Constant):
        raise ValueError('This component is not an editable Constant.')
      previous = component.get_state()
      try:
        simulation.set_component_dynamic_state(
            entity.name, component_name, 'state', arguments['value']
        )
        checkpoint = simulation.make_checkpoint_data()
      except Exception:
        component.set_state(previous)
        raise
    # Never acquire server locks while holding the controller boundary lock.
    self.server.broadcast_entity_info(checkpoint)
