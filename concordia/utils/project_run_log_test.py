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

"""Runner-to-bottom-console tests with guarded runners and a Node DOM stub."""

import json
from pathlib import Path
from unittest import mock

from concordia.environment import step_controller
from concordia.utils import project_test_support
from concordia.utils import project_view
from concordia.utils import project_view_test
from concordia.utils import simulation_server
from concordia.utils import visual_interface
import pytest


def function(source, name, following):
  return source[
      source.index('  function ' + name) : source.index(
          following, source.index('  function ' + name)
      )
  ]


def console_script():
  # Execute the production DOM sink, including its text-node escaping.
  sink = function(
      Path(visual_interface.__file__).read_text(),
      'logConsole(',
      '    function updateControlState(',
  )
  sink = sink.replace('{{', '{').replace('}}', '}')
  return (
      r"""
const assert=require('node:assert/strict');
class Element {
  constructor(){this.children=[];this.textContent='';}
  append(...nodes){this.children.push(...nodes);}
  appendChild(node){this.append(node);}
  replaceChildren(){this.children=[];}
}
const output=new Element();
const document={getElementById:id=>{assert.equal(id,'console-output');return output;},
  createElement:()=>new Element(),createTextNode:text=>({textContent:text})};
const $=id=>document.getElementById(id);
const messages=()=>output.children.map(line=>line.children.at(-1).textContent);
let loggedRun,loggedSteps=0,loggedFailure;
"""
      + sink
      + function(
          project_view.EDITOR_SCRIPT,
          'renderSimulationLog(',
          '  function receive(',
      )
  )


@pytest.mark.parametrize('after_step', [False, True])
def test_failed_runner_snapshot_reaches_bottom_log_once(after_step):
  registry = project_test_support.registry()
  server = simulation_server.SimulationServer(port=0)
  message = 'Repair scene <script>alert(1)</script> & retry'

  def runner(_):
    if after_step:
      server.broadcast_step(
          step_controller.StepData(
              step=1,
              acting_entity='Alice',
              action='Hello',
              entity_actions={'Alice': 'Hello'},
              entity_logs={},
          )
      )
    raise ValueError(message)

  server.configure_project(
      registry,
      registry.default_document(project_test_support.TEMPLATE_KEY),
      runner,
      integrated=True,
  )
  # Exercise real Run dispatch and completion with only the fake runner.
  with mock.patch.object(
      server, 'start', side_effect=AssertionError('No listener')
  ):
    server.run_project(0)
    server._project_thread.join(3)
    assert not server._project_thread.is_alive()
  snapshot = server.operation_service.snapshot('developer')
  assert snapshot['result']['run']['status'] == 'failed'
  assert snapshot['result']['run']['message'] == message
  assert 'renderSimulationLog(envelope);' in project_view.EDITOR_SCRIPT
  project_view_test.javascript(
      console_script() + 'const snapshot=' + json.dumps(snapshot) + r""";
renderSimulationLog(snapshot);
const first=messages();
assert.match(first.at(-1),/^Run failed at step [01]: Repair scene <script>/);
assert.equal(output.children.at(-1).className,'console-line error');
assert.equal(first.length,snapshot.result.steps.length+1);
renderSimulationLog(snapshot);renderSimulationLog(snapshot);
assert.deepEqual(messages(),first); // Polling/SSE do not duplicate errors.
// A reconnect/new page can replay the retained failure.
loggedRun=undefined;renderSimulationLog(snapshot);assert.deepEqual(messages(),first);
// A new Run clears previous output, and the same error can be reported again.
snapshot.references.run_id='next-run';snapshot.result.steps=[];
snapshot.result.run={status:'active'};renderSimulationLog(snapshot);
assert.deepEqual(messages(),[]);
snapshot.result.run={status:'failed',message:'Again'};renderSimulationLog(snapshot);
assert.deepEqual(messages(),[`Run failed at step ${snapshot.result.current_step}: Again`]);
"""
  )


@pytest.mark.parametrize(
    'operation',
    [
        'project.run',
        'runtime.play',
        'runtime.pause',
        'runtime.step',
        'runtime.edit',
    ],
)
@pytest.mark.parametrize('network', [False, True])
def test_dispatch_errors_use_bottom_console(operation, network):
  script = console_script()
  script += function(
      project_view.EDITOR_SCRIPT, 'report(', '  function catalog('
  )
  source = project_view.EDITOR_SCRIPT
  script += source[
      source.index('  async function dispatch(') : source.index(
          '  let events, timer, polling=false;'
      )
  ]
  script += (
      'const operation='
      + json.dumps(operation)
      + ';const network='
      + json.dumps(network)
      + ';'
  )
  script += r"""
const error=new Element();
let connected=true,sending=false;
const envelope={revision:0,references:{}};
const draft={};
const ProjectValidation={forSave:()=>null};
const controls=()=>{};
const refresh=async()=>{};
const crypto={randomUUID:()=> 'request-id'};
const fetch=async()=>{
  if(network) throw Error('Network unavailable');
  return {ok:false,json:async()=>({error:{code:'origin',message:'Same-origin requests only. Open the editor at https://editor.example/'}})};
};
(async()=>{
  assert.equal(await dispatch(operation),false);
  assert.equal(sending,false);
  assert.equal(messages().length,1);
  assert.equal(messages()[0],operation+' failed: '+error.textContent);
  assert.match(messages()[0],network?/Network unavailable/:/Same-origin requests only/);
  assert.equal(output.children[0].className,'console-line error');
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
  project_view_test.javascript(script)
