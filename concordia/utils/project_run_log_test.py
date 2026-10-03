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
const selectedTabs=[];const tab=name=>selectedTabs.push(name);
let loggedRun,loggedSteps=0,loggedFailure,loggedCompletion;
"""
      + sink
      + 'const notices=new Set();\n'
      + function(project_view.EDITOR_SCRIPT, 'notice(', '  let connectionKnown')
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
    assert server._project_thread is not None
    server._project_thread.join(3)
    assert not server._project_thread.is_alive()
  assert server.operation_service is not None
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
assert.deepEqual(selectedTabs,['log']);
assert.equal(first.length,snapshot.result.steps.length+1);
renderSimulationLog(snapshot);renderSimulationLog(snapshot);
assert.deepEqual(messages(),first); // Polling/SSE do not duplicate errors.
// A reconnect/new page can replay the retained failure.
loggedRun=undefined;renderSimulationLog(snapshot);assert.deepEqual(messages(),first);
// A new Run retains previous output, and a new run error can be reported.
snapshot.references.run_id='next-run';snapshot.result.steps=[];
snapshot.result.run={status:'active'};renderSimulationLog(snapshot);
assert.deepEqual(messages(),[...first,'New run attached.']);
snapshot.result.run={status:'failed',message:'Again'};renderSimulationLog(snapshot);
assert.deepEqual(messages(),[...first,'New run attached.',`Run failed at step ${snapshot.result.current_step}: Again`]);
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
  assert.equal(error.textContent,''); // No duplicate toolbar destination.
  assert.ok(messages()[0].startsWith(operation+' failed: '));
  assert.match(messages()[0],network?/Network unavailable/:/Same-origin requests only/);
  assert.equal(output.children[0].className,'console-line error');
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
  project_view_test.javascript(script)


def test_completed_snapshot_replays_reason_once_and_resets_for_new_run():
  project_view_test.javascript(console_script() + r"""
const snapshot={references:{session_id:'session',run_id:'one'},result:{
  steps:[{step:1,acting_entity:'Alice',action:'Alice: Alice'}],current_step:1,
  run:{status:'completed',message:'Configured step limit reached (1).'}}};
renderSimulationLog(snapshot);
assert.deepEqual(messages(),['Step 1 · Player entity action · Alice\nAlice: Alice',
  'Completed at step 1: Configured step limit reached (1).']);
renderSimulationLog(snapshot);assert.equal(messages().length,2);
loggedRun=undefined;renderSimulationLog(snapshot);assert.equal(messages().length,2);
snapshot.references.run_id='two';snapshot.result.steps=[];
snapshot.result.run={status:'active'};renderSimulationLog(snapshot);
assert.equal(messages().length,3);assert.equal(messages().at(-1),'New run attached.');
""")


def test_cards_restore_each_actor_after_svg_replacement_and_completion():
  script = function(
      project_view.EDITOR_SCRIPT,
      'renderEntityActions(',
      '  function renderSimulationLog(',
  )
  project_view_test.javascript(
      r"""
const assert=require('node:assert/strict');
let runtimeMode=true;
const entityData={a:{name:'Alice'},b:{name:'Bob'},g:{name:'Conversation'}};
const values={a:{},b:{},g:{}};
const document={querySelectorAll:()=>Object.keys(values).map(id=>({
  dataset:{entityId:id},querySelector:()=>values[id]}))};
"""
      + script
      + r"""
const state={run:{status:'active'},steps:[]};
renderEntityActions(state);assert.equal(values.a.textContent,'No action recorded yet.');
state.steps=[{acting_entity:'Alice',action:'Alice: first'},
 {acting_entity:'Bob',action:'Bob: reply'},
 {acting_entity:'Alice',action:'Alice: latest <literal>'}];
renderEntityActions(state);
assert.equal(values.a.textContent,'Alice: latest <literal>');
assert.equal(values.b.textContent,'Bob: reply');
values.a.textContent='New SVG empty state';state.run.status='completed';
renderEntityActions(state);assert.equal(values.a.textContent,'Alice: latest <literal>');
assert.equal(values.g.textContent,'No action recorded in this run.');
state.steps=[];state.run.status='active';renderEntityActions(state);
assert.equal(values.a.textContent,'No action recorded yet.');
state.steps=[{acting_entity:'Alice',action:''}];renderEntityActions(state);
assert.equal(values.a.textContent,'Empty action reported.');
runtimeMode=false;renderEntityActions(state);
assert.match(values.a.textContent,/Initial definition/);
"""
  )


def test_completion_reason_is_retained_and_cleared_by_fresh_run():
  registry = project_test_support.registry()
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry,
      registry.default_document(project_test_support.TEMPLATE_KEY),
      lambda _: server.broadcast_completion('Scene sequence exhausted.'),
      integrated=True,
  )
  server.run_project(0)
  assert server._project_thread is not None
  server._project_thread.join(3)
  assert server.operation_service is not None
  snapshot = server.operation_service.snapshot('developer')['result']
  assert snapshot['run']['message'] == 'Scene sequence exhausted.'
  assert server.get_status()['completion_reason'] == 'Scene sequence exhausted.'
  server._project_runner = lambda _: None
  server.run_project(0)
  assert server._project_thread is not None
  server._project_thread.join(3)
  assert server.get_project()['run']['message'] == 'Runner completed.'
  assert server.get_status()['completion_reason'] == ''
