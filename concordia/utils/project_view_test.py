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

"""Execute the editor's actual draft history in Node, without a browser/run."""

import json
import shutil
import subprocess
from unittest import mock

from concordia.utils import project_config
from concordia.utils import project_test_support
from concordia.utils import project_view
from concordia.utils import simulation_server
import pytest


def javascript(source):
  node = shutil.which('node')
  if node is None:
    pytest.skip('Node is required for JavaScript contract tests')
  result = subprocess.run(
      [node, '-e', source], capture_output=True, text=True, check=False
  )
  assert result.returncode == 0, result.stderr
  return result.stdout


def test_history_literal_types_selection_coalescing_and_branch():
  javascript(project_view.DRAFT_HISTORY_SCRIPT + r"""
const assert = require('node:assert/strict');
const history = new ProjectDraftHistory(3);
const original = {document:{text:'literal 🎵\n</script>', flag:false, count:2,
  instances:[{id:'stable', reference:'gm'}]}, selectedId:'stable'};
let current=structuredClone(original);
history.record(current,'text'); current.document.text='first';
history.record(current,'text'); current.document.text='second';
assert.equal(history.past.length,1);
let previous=history.undo(current);
assert.deepEqual(previous,original);
previous.document.instances[0].id='mutated return';
assert.equal(history.future[0].document.instances[0].id,'stable');
current=history.redo(original);
assert.equal(current.document.text,'second');
history.endGroup(); history.record(current,'flag'); current.document.flag=true;
history.endGroup(); history.record(current,'count'); current.document.count=null;
current.selectedId='gm';
previous=history.undo(current);
assert.equal(previous.document.count,2);
assert.equal(previous.document.flag,true);
assert.equal(previous.selectedId,'stable');
history.record(previous,'text'); previous.document.text='new branch';
assert.equal(history.future.length,0);
assert.equal(history.redo(previous),null);
for(let i=0;i<10;i++) {history.record(previous); previous.document.count=i;}
assert.equal(history.past.length,3);
history.clear();
assert.equal(history.undo(previous),null);
assert.equal(history.redo(previous),null);
""")


def test_integrated_script_parses_without_execution():
  javascript(
      'new Function('
      + json.dumps(
          project_view.EDITOR_SCRIPT.removeprefix('<script>\n').removesuffix(
              '</script>\n'
          )
      )
      + ');'
  )


def test_structural_draft_journey_and_undo():
  registry = project_test_support.builder_registry()
  document = registry.default_document('builder-v1')
  catalog = registry.catalog(document)
  javascript(
      project_view.DRAFT_HISTORY_SCRIPT
      + 'const initial='
      + json.dumps(document)
      + ';\n'
      + 'const catalog='
      + json.dumps(catalog)
      + ';\n'
      + r"""
const assert=require('node:assert/strict');
const history=new ProjectDraftHistory();
let draft=structuredClone(initial), selectedId='alice';
function change(fn) {
  const before={document:draft,selectedId};const next=structuredClone(draft);
  const selected=fn(next);history.record(before);draft=next;selectedId=selected;
}
change(next=>ProjectDraftOperations.add(next,catalog,'alice','third','alice'));
assert.equal(draft.instances.at(-1).params.name,'Alice 2');
assert.equal(draft.instances.at(-1).prototype,'alice');
change(next=>ProjectDraftOperations.add(next,catalog,'conversation','second-gm','conversation'));
assert.equal(draft.instances.at(-1).params.next_game_master_name,'second-gm');
change(next=>{ProjectDraftOperations.move(next,'second-gm',-1);return 'second-gm';});
assert.equal(draft.instances[2].id,'second-gm');
const saved=JSON.parse(JSON.stringify(draft));
let previous=history.undo({document:draft,selectedId});
assert.equal(previous.document.instances[4].id,'second-gm');
let restored=history.redo(previous);assert.deepEqual(restored.document,saved);
draft=restored.document;
draft.instances[2].params.next_game_master_name='conversation';
const before=structuredClone(draft);
assert.throws(()=>ProjectDraftOperations.remove(draft,catalog,'conversation'),/references/);
assert.deepEqual(draft,before);
change(next=>ProjectDraftOperations.remove(next,catalog,'second-gm'));
assert.equal(draft.instances.length,4);
assert.throws(()=>ProjectDraftOperations.remove(draft,catalog,'conversation'),/at least/);
assert.throws(()=>ProjectDraftOperations.add(draft,catalog,'unregistered','bad'),/registered/);
assert.equal(initial.instances.length,3);
"""
  )


def test_scene_group_component_production_journey():
  registry = project_test_support.scene_registry()
  document = registry.default_document('scenes-v1')
  javascript(
      project_view.DRAFT_HISTORY_SCRIPT
      + 'const d='
      + json.dumps(document)
      + ';\n'
      + 'const prefabs='
      + json.dumps(registry.catalog(document))
      + ';\n'
      + 'const components='
      + json.dumps(registry.component_catalog(document))
      + ';\n'
      + r"""
const assert=require('node:assert/strict');
const history=new ProjectDraftHistory();
history.record({document:d,selectedId:'alice'});
ProjectSceneOperations.add(d,prefabs,'groups','other-group');
ProjectSceneOperations.add(d,prefabs,'scene_types','other-type');
d.scene_types[1].group='other-group';
ProjectSceneOperations.add(d,prefabs,'scenes','encore');
d.scenes[1].scene_type='other-type';
assert.throws(()=>ProjectSceneOperations.remove(d,'groups:other-group'),/used/);
assert.throws(()=>ProjectSceneOperations.remove(d,'scene_types:other-type'),/used/);
assert.throws(()=>ProjectDraftOperations.remove(d,prefabs,'alice'),/referenced/);
ProjectSceneOperations.move(d,'scenes:encore',-1);
assert.equal(d.scenes[0].id,'encore');
ProjectComponentOperations.add(d,components,'alice','constant','identity');
d.components[0].params.state='Literal 🎵\n</script>';
ProjectComponentOperations.add(d,components,'alice','recent-observations','observations');
assert.throws(()=>ProjectComponentOperations.add(d,components,'conversation','recent-observations','bad'),/compatible/);
ProjectDraftOperations.add(d,prefabs,'alice','copy','alice');
assert.equal(d.components.length,4);
assert.notEqual(d.components[0].id,d.components[2].id);
assert.equal(d.components[2].params.state,d.components[0].params.state);
ProjectDraftOperations.remove(d,prefabs,'copy');
assert.equal(d.components.length,2);
const snapshot=JSON.parse(JSON.stringify(d));
const previous=history.undo({document:d,selectedId:'scenes:encore'});
assert.equal(previous.document.scenes.length,1);
assert.equal(previous.document.components.length,0);
assert.deepEqual(history.redo(previous).document,snapshot);
ProjectSceneOperations.remove(d,'components:identity');
ProjectSceneOperations.remove(d,'components:observations');
assert.equal(d.components.length,0);
"""
  )


def test_component_crud_history_and_invalid_operations():
  registry = project_test_support.scene_registry()
  document = registry.default_document('scenes-v1')
  javascript(
      project_view.DRAFT_HISTORY_SCRIPT
      + 'const d='
      + json.dumps(document)
      + ';\n'
      + 'const catalog='
      + json.dumps(registry.component_catalog(document))
      + ';\n'
      + r"""
const assert=require('node:assert/strict');
const history=new ProjectDraftHistory();
let selection='bob';
const snapshots=[];
function edit(action) {
  const before={document:structuredClone(d),selectedId:selection};
  history.record(before); snapshots.push(before);
  selection=action() || selection;
}
edit(()=>ProjectComponentOperations.add(d,catalog,'bob','constant','identity'));
edit(()=>ProjectComponentOperations.configure(d,'identity','state','Literal </script> 🎵\n'));
edit(()=>ProjectComponentOperations.rename(d,'identity','New name'));
edit(()=>ProjectComponentOperations.add(d,catalog,'alice','constant','alice-note'));
edit(()=>ProjectComponentOperations.add(d,catalog,'bob','recent-observations','recent'));
edit(()=>{ProjectSceneOperations.move(d,'components:recent',-1);return 'components:recent';});
assert.deepEqual(d.components.filter(x=>x.instance==='bob').map(x=>x.id),['recent','identity']);
edit(()=>ProjectSceneOperations.duplicate(d,'components:identity','copy'));
assert.equal(d.components.find(x=>x.id==='copy').params.state,'Literal </script> 🎵\n');
edit(()=>ProjectSceneOperations.remove(d,'components:identity'));
const final={document:structuredClone(d),selectedId:selection};
let current=final;
for(let i=snapshots.length-1;i>=0;i--){current=history.undo(current);assert.deepEqual(current,snapshots[i]);}
for(let i=0;i<snapshots.length;i++)current=history.redo(current);
assert.deepEqual(current,final);
assert.deepEqual(JSON.parse(JSON.stringify(d)),final.document);
const before=structuredClone(d);
for(const action of [
 ()=>ProjectComponentOperations.add(d,catalog,'bob','constant','copy'),
 ()=>ProjectComponentOperations.add(d,catalog,'bob','constant','../bad'),
 ()=>ProjectComponentOperations.add(d,catalog,'missing','constant','other'),
 ()=>ProjectComponentOperations.add(d,catalog,'conversation','recent-observations','other'),
 ()=>ProjectComponentOperations.configure(d,'missing','state','x'),
 ()=>ProjectComponentOperations.configure(d,'copy','constructor','x'),
 ()=>ProjectComponentOperations.rename(d,'missing','x'),
 ()=>ProjectSceneOperations.move(d,'components:missing',1),
 ()=>ProjectSceneOperations.move(d,'components:copy',0),
 ()=>ProjectSceneOperations.duplicate(d,'components:copy','recent'),
 ()=>ProjectSceneOperations.remove(d,'components:missing'),
]){assert.throws(action);assert.deepEqual(d,before);}
"""
  )


def test_reference_replacement_roundtrip_and_atomic_history():
  registry = project_test_support.scene_registry()
  initial = registry.default_document('scenes-v1')
  script = (
      project_view.DRAFT_HISTORY_SCRIPT
      + 'const d='
      + json.dumps(initial)
      + ';\n'
      + 'const catalog='
      + json.dumps(registry.catalog(initial))
      + ';\n'
      + 'const components='
      + json.dumps(registry.component_catalog(initial))
      + ';\n'
      + r"""
const assert=require('node:assert/strict');
ProjectComponentOperations.add(d,components,'alice','constant','identity');
d.components[0].params.state='alice'; // Literal text is never a reference.
const before={document:structuredClone(d),selectedId:'alice'};
const history=new ProjectDraftHistory();
history.record(before);
assert.deepEqual(ProjectReferences.uses(d,catalog,'alice').map(x=>x.selection),
  ['groups:ensemble','scenes:opening','components:identity']);
const selection=ProjectReferences.replace(d,catalog,'alice','bob');
assert.equal(selection,'alice');
assert.deepEqual(d.groups[0].participants,['bob']);
assert.deepEqual(d.scenes[0].participants,['bob']);
assert.deepEqual(d.components,before.document.components);
const final={document:structuredClone(d),selectedId:selection};
const previous=history.undo(final);assert.deepEqual(previous,before);
assert.deepEqual(history.redo(previous),final);
// Normal removal still owns component deletion; replacement never deletes it.
const removed=structuredClone(d);ProjectDraftOperations.remove(removed,catalog,'alice');
assert.equal(removed.components.length,0);
console.log(JSON.stringify(d));
"""
  )
  document = json.loads(javascript(script))
  assert registry.loads(registry.dumps(document)) == document
  config = registry.to_config(document)
  scenes = config.instances[-1].params['scenes']
  assert scenes[0].participants == ['Bob']
  assert scenes[0].scene_type.possible_participants == ['Bob']
  assert document['components'][0]['instance'] == 'alice'
  first = simulation_server.SimulationServer(port=0)
  first.configure_project(
      registry,
      initial,
      mock.Mock(side_effect=AssertionError('no run')),
      integrated=True,
  )
  first.replace_project(registry.dumps(document), 0)
  reopened = simulation_server.SimulationServer(port=0)
  reopened.configure_project(
      registry,
      registry.loads(registry.dumps(first.get_project()['document'])),
      mock.Mock(side_effect=AssertionError('no run')),
      integrated=True,
  )
  assert reopened.get_project()['document'] == document
  saved = first.get_project()
  with pytest.raises(ValueError, match='another tab'):
    first.replace_project(registry.dumps(initial), 0)
  assert first.get_project() == saved
  assert first.simulation is None
  assert reopened.simulation is None


def test_reference_replacement_v2_registered_fields_only():
  registry = project_test_support.builder_registry()
  initial = registry.default_document('builder-v1')
  document = json.loads(
      javascript(
          project_view.DRAFT_HISTORY_SCRIPT
          + 'const d='
          + json.dumps(initial)
          + ';\n'
          + 'const catalog='
          + json.dumps(registry.catalog(initial))
          + ';\n'
          + r"""
const assert=require('node:assert/strict');
ProjectDraftOperations.add(d,catalog,'conversation','other','conversation');
d.instances[0].params.goal='conversation';
d.instances.at(-1).params.next_game_master_name='conversation';
assert.equal(ProjectReferences.uses(d,catalog,'conversation').length,2);
ProjectReferences.replace(d,catalog,'conversation','other');
assert.equal(d.instances[0].params.goal,'conversation');
assert.equal(d.instances[2].params.next_game_master_name,'other');
assert.equal(d.instances[3].params.next_game_master_name,'other');
console.log(JSON.stringify(d));
"""
      )
  )
  assert registry.loads(registry.dumps(document)) == document
  config = registry.to_config(document)
  assert config.instances[2].params['next_game_master_name'] == 'Conversation 2'


def test_reference_namespaces_and_invalid_requests_are_atomic():
  registry = project_test_support.scene_registry()
  initial = registry.default_document('scenes-v1')
  javascript(
      project_view.DRAFT_HISTORY_SCRIPT
      + 'const d='
      + json.dumps(initial)
      + ';\n'
      + 'const catalog='
      + json.dumps(registry.catalog(initial))
      + ';\n'
      + r"""
const assert=require('node:assert/strict');
// IDs are unique within a section, not across all sections.
d.groups[0].id='alice';d.scene_types[0].group='alice';
d.scene_types[0].id='alice';d.scenes[0].scene_type='alice';
assert.deepEqual(ProjectReferences.uses(d,catalog,'alice').map(x=>x.selection),
  ['groups:alice','scenes:opening']);
assert.deepEqual(ProjectReferences.uses(d,catalog,'groups:alice').map(x=>x.selection),['scene_types:alice']);
assert.deepEqual(ProjectReferences.uses(d,catalog,'scene_types:alice').map(x=>x.selection),['scenes:opening']);
assert.deepEqual(ProjectReferences.uses(d,catalog,'groups:missing'),[]);
ProjectDraftOperations.add(d,catalog,'conversation','other','conversation');
const legacy=structuredClone(catalog.find(x=>x.instance.prototype==='conversation'));
legacy.instance.prototype='legacy';legacy.accepts_scenes=false;catalog.push(legacy);
ProjectDraftOperations.add(d,catalog,'legacy','legacy-gm');
assert.deepEqual(ProjectReferences.candidates(d,catalog,'conversation').map(x=>x.id),['other']);
const before=structuredClone(d);
for(const [source,target] of [
  ['missing','bob'],['alice','missing'],['alice','alice'],['alice','conversation'],
  ['conversation','legacy-gm'],['other','conversation'],['groups:alice','bob'],
]){
  assert.throws(()=>ProjectReferences.replace(d,catalog,source,target));
  assert.deepEqual(d,before);
}
assert.throws(()=>ProjectReferences.replace(d,[],'alice','bob'));
assert.deepEqual(d,before);
ProjectReferences.replace(d,catalog,'conversation','other');
assert.equal(d.scene_types[0].game_master,'other');
// An unsaved draft that once referenced Alice must not preserve stale controls.
ProjectReferences.replace(d,catalog,'alice','bob');
assert.throws(()=>ProjectReferences.replace(d,catalog,'alice','bob'),/no replaceable/);
"""
  )


@pytest.mark.parametrize(
    ('path', 'value', 'selection', 'field'),
    [
        (('max_steps',), 0, 'simulation', 'editor-max-steps'),
        (('premise',), None, 'simulation', 'editor-premise'),
        (('instances', 1, 'params', 'name'), '', 'bob', 'editor-bob-name'),
        (('components', 0, 'name'), '', 'components:note', 'world-name'),
        (
            ('components', 0, 'params', 'state'),
            7,
            'components:note',
            'component-param-state',
        ),
        (('scenes', 0, 'num_rounds'), 0, 'scenes:opening', 'world-rounds'),
        (('scenes', 0, 'participants'), [], 'scenes:opening', None),
        (('groups', 0, 'participants'), [], 'groups:ensemble', None),
        (
            ('scene_types', 0, 'game_master'),
            'missing',
            'scene_types:discussion',
            'world-game-master',
        ),
    ],
)
def test_save_validation_navigation_uses_actual_registry_errors(
    path, value, selection, field
):
  registry = project_test_support.scene_registry()
  document = registry.default_document('scenes-v1')
  document['components'] = [{
      'id': 'note',
      'instance': 'alice',
      'type': 'constant',
      'name': 'Note',
      'params': {'state': 'Literal </script> 🎵', 'pre_act_label': 'Context'},
  }]
  owner = document
  for part in path[:-1]:
    owner = owner[part]
  owner[path[-1]] = value
  with pytest.raises(project_config.ValidationError) as caught:
    registry.normalize(document)
  result = json.loads(
      javascript(
          project_view.DRAFT_HISTORY_SCRIPT
          + 'const d='
          + json.dumps(document)
          + ';\n'
          + 'const error='
          + json.dumps({'code': 'invalid_edit', 'message': str(caught.value)})
          + ';\n'
          + "console.log(JSON.stringify(ProjectValidation.forSave(d,'project.save',{text:JSON.stringify(d)},error)));"
      )
  )
  assert result == {'selection': selection, 'field': field}


def test_validation_navigation_rejects_stale_imports_and_ambiguous_paths():
  javascript(project_view.DRAFT_HISTORY_SCRIPT + r"""
const assert=require('node:assert/strict');
const d={instances:[{id:'alice',params:{name:'A'}},{id:'0',params:{name:'B'}}]};
const original=structuredClone(d);
assert.equal(ProjectValidation.target(d,'$.instances[0].params.name: error'),null);
assert.equal(ProjectValidation.target(d,'$.instances[missing].params.name: error'),null);
assert.equal(ProjectValidation.target(d,'$.instances[999999].params.name: error'),null);
assert.equal(ProjectValidation.target(d,'<script>alert(1)</script>: error'),null);
assert.equal(ProjectValidation.target(d,'$.template: unknown template'),null);
assert.equal(ProjectValidation.target(d,'Definition changed in another tab.'),null);
const error={code:'invalid_edit',message:'$.instances[alice].params.name: empty'};
assert.deepEqual(ProjectValidation.forSave(d,'project.save',{text:JSON.stringify(d)},error),
  {selection:'alice',field:'editor-alice-name'});
// An imported file is a different document; never jump into the current draft.
assert.equal(ProjectValidation.forSave(d,'project.save',{text:'{}'},error),null);
assert.equal(ProjectValidation.forSave(d,'runtime.edit',{text:JSON.stringify(d)},error),null);
assert.equal(ProjectValidation.forSave(d,'project.save',{text:JSON.stringify(d)},
  {...error,code:'stale_revision'}),null);
assert.deepEqual(ProjectValidation.target({scenes:[{id:'opening'}]},'$.scenes[opening].constructor: error'),
  {selection:'scenes:opening',field:null});
const args={text:JSON.stringify(d)};d.instances[0].params.name='changed while saving';
assert.equal(ProjectValidation.forSave(d,'project.save',args,error),null);
d.instances[0].params.name='A';assert.deepEqual(d,original);
d.instances.push(structuredClone(d.instances[0]));
assert.equal(ProjectValidation.target(d,error.message),null);
""")


def test_component_move_owner_preserves_configuration_history_and_reopen():
  registry = project_test_support.scene_registry()
  document = registry.default_document('scenes-v1')
  result = javascript(
      project_view.DRAFT_HISTORY_SCRIPT
      + 'let d='
      + json.dumps(document)
      + ';const catalog='
      + json.dumps(registry.component_catalog(document))
      + r""";
const assert=require('node:assert/strict');
ProjectComponentOperations.add(d,catalog,'bob','recent-observations','recent');
ProjectComponentOperations.configure(d,'recent','history_length',7);
ProjectComponentOperations.rename(d,'recent','My observations');
ProjectComponentOperations.add(d,catalog,'alice','constant','existing');
const before={document:structuredClone(d),selectedId:'components:recent'};
const history=new ProjectDraftHistory();history.record(before);
const selectedId=ProjectComponentOperations.moveOwner(d,catalog,'recent','alice');
assert.equal(selectedId,'components:recent');
assert.deepEqual(d.components.map(x=>x.id),['existing','recent']);
assert.deepEqual(d.components[1],{...before.document.components[0],instance:'alice'});
const after={document:structuredClone(d),selectedId};
assert.deepEqual(history.undo(after),before);
assert.deepEqual(history.redo(before),after);
for(const [id,target] of [['missing','bob'],['recent','missing'],['recent','conversation']]){
  assert.throws(()=>ProjectComponentOperations.moveOwner(d,catalog,id,target));
  assert.deepEqual(d,after.document);
}
ProjectComponentOperations.moveOwner(d,catalog,'recent','alice');
assert.deepEqual(d,after.document);
assert.throws(()=>ProjectComponentOperations.moveOwner(d,[],'recent','bob'));
assert.deepEqual(d,after.document);
console.log(JSON.stringify(d));
"""
  )
  moved = registry.loads(result)
  assert registry.loads(registry.dumps(moved)) == moved
  built = project_test_support.build(registry.to_config(moved))
  actors = {actor.name: actor for actor in built.get_entities()}
  assert 'authored_recent' not in actors['Bob'].get_all_context_components()
  recent = actors['Alice'].get_component('authored_recent')
  assert recent.get_state()['history_length'] == 7
  assert actors['Alice'].get_act_component().get_state()['component_order'][
      -2:
  ] == ['authored_existing', 'authored_recent']
