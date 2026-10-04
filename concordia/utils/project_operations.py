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

"""Connect project authoring and simulation controls to editor API operations.

SimulationServer.configure_project(integrated=True) creates a ProjectEditor.
It registers save, run, reset, pause, step and component-edit handlers with
OperationService, and supplies snapshots containing the saved definition,
rendered prefab/component views, runtime status and recorded actions. The
project_view browser UI consumes these snapshots and dispatches operations
through SimulationServer HTTP endpoints. CLI clients can use the same service.

Use this adapter to let clients edit a registered Config, run it through a
caller-supplied runner and inspect the result. Runtime component edits require
the standard StepController to acknowledge a paused boundary.
"""

from collections.abc import Callable
import copy
import json
from typing import Any, TYPE_CHECKING
import urllib.parse
import uuid

from concordia.command_line_interface import concordia_log
from concordia.typing import prefab as prefab_lib
from concordia.utils import operation_service as ops
from concordia.utils import project_config
from concordia.utils import session_commands
from concordia.utils import structured_logging
from concordia.utils import visual_interface

if TYPE_CHECKING:
  from concordia.utils import simulation_server


class ProjectEditor:
  """One developer view attached to a SimulationServer before it starts."""

  def __init__(
      self,
      server: 'simulation_server.SimulationServer',
      registry: project_config.Registry,
      preview: Callable[[prefab_lib.Config], Any] | None,
      *,
      viewers: dict[str, Callable[[], str] | str] | None = None,
  ):
    self.server = server
    self.registry = registry
    self.preview = preview
    self.viewers = dict(viewers or {})
    if any(not key or key == 'default' for key in self.viewers):
      raise ValueError(
          'Viewer names must be nonempty and distinct from default.'
      )
    self.service = ops.OperationService(project_id='registered-project')
    self.definition_view: dict[str, Any] | None = None
    self.runtime_view: dict[str, Any] | None = None
    self.runtime_config: prefab_lib.Config | None = None
    self.runtime_document: dict[str, Any] | None = None
    self.steps: list[dict[str, Any]] = []
    self.reset_requested = False
    self.stepping = False
    self.structured_log = None
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
    run_parameters = {
        'revision': ops.Parameter('integer', 'Saved project revision'),
    }
    if server.get_project()['run_limits'] is not None:
      run_parameters['requested_steps'] = ops.Parameter(
          'integer',
          'Steps for this run, within the saved configuration maximum',
      )
    self._register(
        'project.run',
        run_parameters,
        lambda args: server.run_project(
            args['revision'], args.get('requested_steps')
        ),
    )
    self._register(
        'session.plan', {'line': string}, self.command_plan, mutation=False
    )
    draft_text = {
        'text': ops.Parameter('string', 'Registered project JSON', 900000)
    }
    self._register(
        'project.validate',
        draft_text,
        lambda args: self.preview_draft(args)['document'],
        mutation=False,
    )
    self._register(
        'project.preview', draft_text, self.preview_draft, mutation=False
    )
    self._register(
        'log.query',
        {
            'command': string,
            'source': string,
            'arguments': string,
            'imported': ops.Parameter(
                'string', 'Client-owned structured log JSON', 900000
            ),
        },
        self.query_log,
        mutation=False,
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
    self._register(
        'runtime.edit_state',
        {
            'instance_id': string,
            'component': string,
            'field': string,
            'value': ops.Parameter('string', 'JSON field value', 900000),
        },
        lambda args: self.edit({**args, 'value': json.loads(args['value'])}),
    )

    self._register(
        'viewer.read', {'name': string}, self.read_viewer, mutation=False
    )

  def read_viewer(self, arguments):
    name = arguments['name']
    if name not in self.viewers:
      raise ValueError('Unknown registered viewer.')
    source = self.viewers[name]
    if callable(source):
      html = source()
      if not isinstance(html, str) or len(html.encode('utf-8')) > 5000000:
        raise ValueError('Viewer provider must return at most 5 MB of HTML.')
      return {'name': name, 'html': html}
    parsed = urllib.parse.urlsplit(source)
    if (
        parsed.scheme not in ('http', 'https')
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
      raise ValueError(
          'Viewer URL must be HTTP(S) without embedded credentials.'
      )
    return {'name': name, 'url': source}

  def preview_draft(self, arguments):
    document = self.registry.loads(arguments['text'])
    return {'document': document, 'definition': self.prepare(document)}

  def command_plan(self, arguments):
    plan = session_commands.parse(arguments['line'])
    operation = session_commands.operation(plan, self.snapshot())
    if operation is not None:
      plan['operation'], plan['arguments'] = operation
    return plan

  def query_log(self, arguments):
    import json

    source = arguments['source']
    if source == 'imported':
      log = structured_logging.SimulationLog.from_json(arguments['imported'])
    elif source == 'current':
      if self.structured_log is not None:
        log = self.structured_log
      else:
        simulation = self.server.simulation
        if simulation is None:
          raise ValueError(
              'No current structured log. Run a project or use log import.'
          )
        if self.server.get_project()['run']['status'] == 'active':
          raise ValueError(
              'Current structured log is available after the run finishes; use'
              ' an imported log meanwhile.'
          )
        log = structured_logging.SimulationLog.from_raw_log(
            simulation.get_raw_log()
        )
    else:
      raise ValueError('Log source must be current or imported.')
    command = arguments['command']
    allowed = (
        'overview',
        'entities',
        'actions',
        'context',
        'step',
        'timeline',
        'search',
        'memories',
        'components',
        'export',
        'dump',
        'bundle',
    )
    if command not in allowed:
      raise ValueError('Unknown log analysis command.')
    values = json.loads(arguments['arguments'])
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
      raise ValueError('Log arguments must be a JSON string list.')
    if command == 'export':
      if values:
        raise ValueError('log export accepts no analysis arguments.')
      result = {
          'text': 'Structured SimulationLog JSON ready for import.',
          'download': {
              'name': 'simulation-log.json',
              'type': 'application/json',
              'content': log.to_json(),
          },
      }
    else:
      result = concordia_log.analyze(log, command, values)
    result['source'] = source
    return result

  def _register(self, name, parameters, handler, *, mutation=True):
    def invoke(arguments):
      try:
        result = handler(arguments)
      except (ValueError, KeyError, TypeError, RuntimeError) as error:
        raise ops.OperationError('invalid_edit', str(error)) from error
      if mutation:
        self.service.publish({'kind': name})
      return result

    self.service.register(
        ops.Operation(
            name=name,
            description=name,
            parameters=parameters,
            handler=invoke,
            mutation=mutation,
        )
    )

  def prepare(self, document: dict[str, Any]) -> dict[str, Any]:
    """Build an optional trusted inspection preview, never a running simulation."""
    config = self.registry.to_config(document)
    built = self.preview(config) if self.preview else None
    if hasattr(built, 'make_checkpoint_data'):
      self.registry.apply_dynamic_states(document, built)
      checkpoint = built.make_checkpoint_data()
    else:
      if document.get('dynamic_states'):
        raise ValueError(
            'Initial dynamic state requires a Simulation preview factory.'
        )
      checkpoint = built
    svg, entities = visual_interface.visualize_config(config, checkpoint)
    return {
        'initial_state_editable': hasattr(built, 'set_component_dynamic_state'),
        'engine': checkpoint.get('engine') if checkpoint else None,
        'document': copy.deepcopy(document),
        'svg': svg,
        'entities': entities,
        'inspector': self.registry.inspector(document),
        'templates': self.registry.template_keys(),
        'catalog': self.registry.catalog(document),
        'component_catalog': self.registry.component_catalog(document),
        'prefab_diagnostics': self.registry.prefab_diagnostics(),
    }

  def begin(self, config: prefab_lib.Config) -> None:
    # Called under the same lock as dispatch, before the run thread starts.
    self.service.references['run_id'] = str(uuid.uuid4())
    self.runtime_config = config
    self.runtime_document = self.server.get_project()['document']
    self.structured_log = None
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
        'viewers': list(self.viewers),
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
        self.runtime_view = {
            'engine': checkpoint.get('engine'),
            'svg': svg,
            'entities': entities,
            'document': self.runtime_document,
        }
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
    json.dumps(arguments['value'], allow_nan=False)
    component_name = arguments['component']
    document = self.runtime_document or self.server.get_project()['document']
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
    if simulation is None:
      raise ValueError('No bound simulation to edit.')
    with self.server.step_controller.paused_boundary():
      entity = next(
          (
              entity
              for entity in [
                  *simulation.get_entities(),
                  *simulation.get_game_masters(),
              ]
              if entity.name == instance['params']['name']
          ),
          None,
      )
      if entity is None:
        raise ValueError('Entity is not present in this runtime.')
      component = entity.get_component(component_name)
      previous = component.get_state()
      try:
        simulation.set_component_dynamic_state(
            entity.name,
            component_name,
            arguments.get('field', 'state'),
            arguments['value'],
        )
        checkpoint = simulation.make_checkpoint_data()
      except Exception:
        component.set_state(previous)
        raise
    # Never acquire server locks while holding the controller boundary lock.
    self.server.broadcast_entity_info(checkpoint)
