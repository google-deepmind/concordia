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

"""Explicit CLI-owned draft journal using the editor's existing JavaScript actions.

Node executes only the packaged action implementation. User values are JSON on
stdin, never executable source or shell strings. No network or simulation runs.
"""

import json
import shutil
import subprocess

from concordia.utils import project_view

_SCRIPT = r"""
const fs=require('node:fs');
const crypto=require('node:crypto');
const {journal,plan,runtime,savedDocument}=JSON.parse(fs.readFileSync(0,'utf8'));
const history=new ProjectDraftHistory();history.past=journal.past || [];history.future=journal.future || [];
const current={document:journal.document,selectedId:journal.selectedId};
const action=plan.action,args=plan.args;
let result;
if(action==='undo' || action==='redo'){
  const next=history[action](current);if(!next)throw Error('No '+action+' available.');
  journal.document=next.document;journal.selectedId=next.selectedId;
 }else if(['select','inspect','catalog','list','locate','panel'].includes(action)){
  const readAction=action==='panel'?(args[0]==='inspector'?'inspect':'list'):action==='select'?'inspect':action;
  const readArgs=action==='panel'?[]:args;
  result=ProjectDraftCommands.read(journal.document,journal.metadata,journal.view==='runtime'?savedDocument:(journal.preview_document || journal.base),runtime,journal.view,journal.selectedId,readAction,readArgs);
  if(action==='select' || action==='inspect') {journal.selectedId=args[0]==='.'?journal.selectedId:args[0] || journal.selectedId;journal.component=args[1] || null;}
  if(action==='locate') {journal.selectedId=result.selection;journal.view='definition';}
  if(action==='panel')journal.panel=args[0];
}else if(action==='references')result=ProjectReferences.uses(journal.document,journal.metadata.catalog,args[0]==='.'?journal.selectedId:args[0]);
else if(action==='view'){journal.view=args[0];result={view:journal.view,message:'Client inspection source changed.'};}
else if(action==='search'){
  journal.search=args[0];
  result=[...journal.document.instances,...(journal.document.components || [])].filter(item=>{
    const index=((journal.view==='runtime'?savedDocument:journal.preview_document || journal.base)?.instances || journal.document.instances).findIndex(x=>x.id===item.id);
    const entities=journal.view==='runtime'?runtime?.entities:journal.metadata.entities;
    const components=Object.keys(entities?.['entity_'+index]?.component_info?.context_components || {});
    return ProjectDraftCommands.matches(item,args[0],components,(journal.document.components || []).filter(c=>c.instance===item.id));
  });
}else{
  if(journal.view==='runtime')throw Error('Use view definition before changing an authored draft.');
  const next=structuredClone(journal.document);
  const selection=ProjectDraftCommands.apply(next,journal.metadata,journal.selectedId,action,args,plan.id);
  if(JSON.stringify(next)!==JSON.stringify(journal.document)){history.record(current);journal.document=next;journal.selectedId=selection || journal.selectedId;journal.search='';}
  else result={message:'No draft change.'};
}
journal.past=history.past;journal.future=history.future;
process.stdout.write(JSON.stringify({journal,result:result || {message:'Client draft updated; not saved.',selected:journal.selectedId}}));
"""


def apply(journal, plan, runtime=None, saved_document=None):
  node = shutil.which('node')
  if node is None:
    raise ValueError(
        'CLI draft commands require Node to reuse the editor action'
        ' implementation.'
    )
  try:
    result = subprocess.run(
        [node, '-e', project_view.DRAFT_HISTORY_SCRIPT + _SCRIPT],
        input=json.dumps({
            'journal': journal,
            'plan': plan,
            'runtime': runtime,
            'savedDocument': saved_document,
        }),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
  except subprocess.TimeoutExpired as error:
    raise ValueError(
        'Draft action timed out; the local journal was not changed.'
    ) from error
  if result.returncode:
    # Never include generated script/input dumps from Node in client output.
    lines = result.stderr.splitlines()
    message = next(
        (
            line
            for line in lines
            if line.startswith('Error:') or line.startswith('SyntaxError:')
        ),
        'Draft action failed.',
    )
    raise ValueError(message)
  return json.loads(result.stdout)
