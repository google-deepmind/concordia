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

from concordia.utils import project_test_support
from concordia.utils import project_view
import pytest


def javascript(source):
  node = shutil.which('node')
  if node is None:
    pytest.skip('Node is required for JavaScript contract tests')
  subprocess.run(
      [node, '-e', source], check=True, capture_output=True, text=True
  )


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
