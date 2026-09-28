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

"""Explicit bounded Sequential/mock acceptance run; not an auto-collected test.

Launching this module executes real Concordia simulations. Display the exact
command and obtain the coordinator's acknowledgement before invoking it.
"""

import argparse
import copy
import json
from pathlib import Path
import time
import uuid

from concordia.environment.engines import sequential
from concordia.language_model import no_language_model

from examples.project_editor import run
from examples.project_editor import template


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  registry = template.registry()
  document = registry.default_document(template.TEMPLATE_KEY)
  document['max_steps'] = 6
  server = run.create_editor(document, output=args.output, step_delay=0.25)
  service = server.operation_service
  assert service is not None

  def snapshot():
    return service.snapshot('developer')

  def dispatch(name, arguments=None):
    current = snapshot()
    return service.dispatch(
        'developer',
        {
            'operation': name,
            'arguments': arguments or {},
            'revision': current['revision'],
            'references': current['references'],
            'retry_key': str(uuid.uuid4()),
        },
    )

  def wait(predicate):
    deadline = time.monotonic() + 30
    while True:
      state = snapshot()['result']
      if predicate(state):
        return state
      if state['state'] == 'failed':
        raise AssertionError(state['run'])
      if time.monotonic() >= deadline:
        raise AssertionError(f'Timed out: {state["state"]}')
      time.sleep(0.01)

  try:
    document['instances'][0]['params'][
        'custom_instructions'
    ] = 'Listen carefully before suggesting a song.'
    dispatch('project.save', {'text': json.dumps(document), 'revision': 0})
    saved = copy.deepcopy(server.get_project()['document'])
    dispatch('project.run', {'revision': server.get_project()['revision']})
    first = wait(lambda state: state['current_step'] >= 1)
    assert isinstance(server.simulation._engine, sequential.Sequential)
    assert isinstance(
        server.simulation._model, no_language_model.NoLanguageModel
    )
    dispatch('runtime.pause')
    paused = wait(lambda state: state['state'] == 'paused')
    baseline = paused['current_step']
    assert baseline == 1, first['current_step']
    dispatch(
        'runtime.edit',
        {
            'instance_id': 'bob',
            'component': 'Goal',
            'value': 'Runtime-only goal',
        },
    )
    assert server.get_project()['document'] == saved
    runtime = snapshot()['result']['runtime']['entities']['entity_1']
    assert (
        runtime['component_info']['context_components']['Goal']['state'][
            'state'
        ]
        == 'Runtime-only goal'
    )
    dispatch('runtime.step')
    stepped = wait(
        lambda state: state['state'] == 'paused'
        and state['current_step'] == baseline + 1
    )
    time.sleep(0.1)
    assert snapshot()['result']['current_step'] == stepped['current_step']
    dispatch('runtime.play')
    completed = wait(lambda state: state['state'] == 'completed')
    assert completed['current_step'] == 6
    assert len(completed['steps']) == 6
    old_simulation = server.simulation
    old_run = snapshot()['references']['run_id']
    thread = server._project_thread
    assert thread is not None
    thread.join(3)
    dispatch('project.reset')
    assert snapshot()['result']['state'] == 'ready'
    dispatch('project.run', {'revision': server.get_project()['revision']})
    wait(lambda state: state['current_step'] >= 1)
    dispatch('runtime.pause')
    wait(lambda state: state['state'] == 'paused')
    assert server.simulation is not old_simulation
    assert snapshot()['references']['run_id'] != old_run
    runtime = snapshot()['result']['runtime']['entities']['entity_1']
    assert (
        runtime['component_info']['context_components']['Goal']['state'][
            'state'
        ]
        == saved['instances'][1]['params']['goal']
    )
    dispatch('project.reset')
    wait(lambda state: state['state'] == 'ready')
    exports = list(args.output.glob('*/initial-project.json'))
    assert len(exports) == 2
    for exported in exports:
      assert json.loads(exported.read_text()) == saved
      assert json.loads(exported.with_name('log.json').read_text())
    print(
        'PASS: real Sequential + NoLanguageModel; pause acknowledgement,'
        ' runtime-only edit, one Step, resume/completion, reset and fresh run;'
        ' two preserved logs.',
        flush=True,
    )
  finally:
    server.step_controller.stop()
    thread = server._project_thread
    if thread is not None:
      thread.join(3)
      assert not thread.is_alive(), 'Runner did not exit'


if __name__ == '__main__':
  main()
