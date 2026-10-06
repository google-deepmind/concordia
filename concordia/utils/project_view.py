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

"""Initial-project controls embedded in the standard visual interface.

No project text is interpolated into HTML or executable JavaScript. Data comes
from the configured SimulationServer's JSON endpoint; fields use DOM values.
"""

import json

from concordia.utils import session_commands

STYLE = """
<style>
.sim-controls {display:none !important;}
.header {flex-wrap:wrap;}
.header a {color:#90c8ff;}
.project-field {display:block; margin:12px 0; white-space:pre-wrap;}
.project-field textarea, .project-field input:not([type=checkbox]) {
  display:block; width:100%; margin-top:6px; color:inherit; background:#252526;
  border:1px solid #777; padding:8px; font:inherit;
}
.project-field textarea {min-height:110px; resize:vertical;}
.project-button {margin:4px; padding:7px; cursor:pointer; color:inherit; background:#383838; border:1px solid #777; border-radius:3px;}
.project-button:disabled {opacity:0.5; cursor:default;}
#project-status {white-space:pre-wrap; margin:8px;}
#project-hierarchy button {display:block; width:95%; text-align:left;}
</style>
"""

SCRIPT = r"""
<script>
(() => {
  let state = null;
  let draft = null;
  let dirty = false;
  let selected = null;
  const legacyNotices=new Set();
  function legacyNotice(message,type='info'){if(legacyNotices.has(message))return;legacyNotices.add(message);logConsole(message,type);}
  const header = document.querySelector('.header');
  document.querySelector('.sim-controls').hidden = true;
  const tools = document.createElement('div');
  header.append(tools);
  const save = button('Save project', saveProject);
  const open = button('Open project', () => file.click());
  const run = button('Run saved project', runProject);
  const runtime = document.createElement('a');
  runtime.textContent = 'Runtime inspector (separate state)';
  runtime.href = '/runtime'; runtime.target = '_blank'; runtime.hidden = true;
  tools.append(runtime);
  const file = document.createElement('input');
  file.type = 'file'; file.accept = '.json,application/json';
  file.id = 'project-file'; file.hidden = true; tools.append(file);
  const sidebar = document.querySelector('.left-sidebar');
  sidebar.replaceChildren();
  const status = document.createElement('p'); status.id = 'project-status';
  const error = {set textContent(value){if(value) legacyNotice(value,'error');}};
  const fields = document.createElement('div');
  const hierarchy = document.createElement('div'); hierarchy.id = 'project-hierarchy';
  sidebar.append(status, fields, hierarchy);
  const content = document.getElementById('inspector-content');
  document.getElementById('inspector-empty').hidden = true;
  content.style.display = 'block';

  function button(label, action) {
    const b = document.createElement('button'); b.textContent = label;
    b.className = 'project-button'; b.onclick = action; tools.append(b);
    return b;
  }
  async function request(path, body) {
    const response = await fetch(path, body === undefined ? {} : {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Request failed');
    return result;
  }
  function report(e) { legacyNotice(e.message, 'error'); }
  function controls() {
    const active = state && state.run.status === 'active';
    if(state?.run.message)legacyNotice(state.run.message,state.run.status==='failed'?'error':'info');
    save.disabled = !state || active; open.disabled = !state || active;
    run.disabled = !state || dirty || active;
    fields.querySelectorAll('input,textarea').forEach(x => x.disabled = active);
    content.querySelectorAll('input,textarea').forEach(x => x.disabled = active);
    runtime.hidden = !state || state.run.status === 'not_started';
    status.textContent = state ? (
      'Initial project — ' + (dirty ? 'unsaved changes' : 'saved draft') +
      '\nRun: ' + state.run.status +
      (state.run.revision !== undefined && state.run.revision !== state.revision
        ? ' (earlier saved revision)' : '') +
      ''
    ) : 'Opening initial project…';
  }
  function field(container, label, value, change, id) {
    const wrapper = document.createElement('label'); wrapper.className = 'project-field';
    wrapper.textContent = label;
    const input = document.createElement(typeof value === 'string' ? 'textarea' : 'input');
    input.id = id;
    if (typeof value === 'boolean') {input.type = 'checkbox'; input.checked = value;}
    else {if (typeof value === 'number') input.type = 'number'; input.value = value;}
    input.oninput = () => {
      change(typeof value === 'boolean' ? input.checked :
        typeof value === 'number' ? (input.value === '' ? null : Number(input.value)) : input.value);
      dirty = true; error.textContent = ''; controls();
    };
    wrapper.append(input); container.append(wrapper);
  }
  function inspect(id) {
    selected = id;
    const item = draft.instances.find(x => x.id === id);
    document.getElementById('inspector-title').textContent = item.params.name;
    document.getElementById('inspector-subtitle').textContent =
      'Initial configuration · ' + item.role + ' · ' + item.prefab + ' · ID: ' + id;
    content.replaceChildren();
    for (const [key, value] of Object.entries(item.params)) {
      field(content, key, value, updated => item.params[key] = updated, 'project-' + id + '-' + key);
    }
    controls();
  }
  // Use stable document IDs; runtime's positional card selection stays separate.
  document.addEventListener('click', event => {
    const card = event.target.closest('.entity-card');
    if (card && draft) {
      event.stopImmediatePropagation();
      const item = state.document.instances.find(x => x.params.name === card.dataset.entityName);
      if (item) inspect(item.id);
    }
  }, true);
  function render(result) {
    state = result; draft = structuredClone(result.document); dirty = false;
    fields.replaceChildren(); hierarchy.replaceChildren();
    field(fields, 'Initial premise', draft.premise, v => draft.premise = v, 'project-premise');
    field(fields, 'Simulation maximum steps (1–1000)', draft.max_steps, v => draft.max_steps = v, 'project-max-steps');
    for (const item of draft.instances) {
      const b = document.createElement('button');
      b.className = 'project-button'; b.dataset.projectId = item.id;
      b.textContent = item.params.name + ' · ' + item.role;
      b.onclick = () => inspect(item.id); hierarchy.append(b);
    }
    inspect(selected && draft.instances.some(x => x.id === selected) ? selected : draft.instances[0].id);
  }
  async function refreshDiagram() {
    const page = new DOMParser().parseFromString(await (await fetch('/')).text(), 'text/html');
    document.querySelector('.svg-container').replaceChildren(
      ...page.querySelector('.svg-container').childNodes);
  }
  async function saveProject() {
    try {
      const result = await request('/project', {text:JSON.stringify(draft), revision:state.revision});
      render(result); await refreshDiagram();
      const blob = new Blob([JSON.stringify(result.document, null, 2) + '\n'], {type:'application/json'});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a'); link.href = url; link.download = 'concordia-project.json';
      link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch(e) { report(e); }
  }
  file.onchange = async () => {
    try {
      if (!file.files.length) return;
      if (file.files[0].size > 900000) throw new Error('Project file is too large (maximum 900 kB).');
      render(await request('/project', {text:await file.files[0].text(), revision:state.revision}));
      await refreshDiagram(); error.textContent = '';
    } catch(e) { report(e); }
    finally { file.value = ''; }
  };
  async function runProject() {
    try {
      state = await request('/project/run', {revision:state.revision}); error.textContent = ''; controls();
    } catch(e) { report(e); }
  }
  request('/project').then(render).catch(report);
  // Observe run completion without overwriting another tab's unsaved fields.
  const timer = setInterval(async () => {
    try {
      const result = await request('/project');
      if (state && result.revision !== state.revision) {
        error.textContent = 'Draft changed in another tab. Reload before saving; your unsaved fields are still shown.';
      }
      if (state) {state.run = result.run; controls();}
    } catch(e) { report(e); }
  }, 1000);
  window.addEventListener('pagehide', () => clearInterval(timer));
})();
</script>
"""


# Integrated mode adds a single-page editor; project_html is the standalone
# form.
EDITOR_STYLE = """
<style>
body {height:100dvh;}
.layout {height:100dvh;gap:0;grid-template-columns:var(--editor-left,220px) 8px minmax(0,1fr) 8px var(--editor-right,300px);
 grid-template-rows:auto minmax(0,1fr) 8px var(--editor-terminal,180px);}
.header {grid-row:1;max-height:40dvh;overflow:auto;}
.left-sidebar {grid-column:1;grid-row:2;}
.center-panel {grid-column:3;grid-row:2;}
.right-sidebar {grid-column:5;grid-row:2;}
.bottom-panel {grid-row:4;}
.editor-splitter {background:#3c3c3c;touch-action:none;user-select:none;position:relative;z-index:2;}
.editor-splitter:hover,.editor-splitter:focus-visible {background:#90c8ff;outline:2px solid #90c8ff;outline-offset:-2px;}
.editor-splitter[aria-orientation=vertical] {grid-row:2;cursor:col-resize;}
.editor-splitter[aria-orientation=horizontal] {grid-column:1 / -1;grid-row:3;cursor:row-resize;}
.editor-splitter::after {content:'';position:absolute;inset:0 -4px;}
.editor-splitter[aria-orientation=horizontal]::after {inset:-4px 0;}
.layout {--editor-font:.75rem;font-size:var(--editor-font);}
.left-sidebar button,.left-sidebar input,.center-panel,.console-output,
.editor-component,.sidebar-title {font-size:var(--editor-font);}
.left-sidebar button,.left-sidebar input {font-family:inherit;}
/* Preserve the inspector's established desktop hierarchy, using scalable units. */
.right-sidebar {font-size:1rem;}
.inspector-title {font-size:.875rem;}
.inspector-subtitle,.param-row,.component-item-header {font-size:.6875rem;}
.component-header {font-size:.75rem;}
.component-item-class,.component-item-toggle,.component-state {font-size:.625rem;}
.param-value,.state-value {font-size:inherit;}
.svg-container > svg {width:calc(var(--editor-graph-width) * var(--editor-graph-scale,1));max-width:none;height:auto;}
@media(pointer:coarse), (max-width:700px) {
 .layout {--editor-font:1rem;--editor-graph-scale:1.4;}
 .inspector-title,.inspector-subtitle,.param-row,.component-item-header,
 .component-header,.component-item-class,.component-item-toggle,.component-state,
 .editor-field input,.editor-field textarea,.editor-field select,#editor-command {font-size:1rem;}
}
.header {display:block;padding:8px;}
.header > :not(#editor-toolbar) {display:none;}
#editor-toolbar button, #editor-tabs button {min-height:44px;min-width:44px;}
#editor-toolbar {display:flex;flex-wrap:wrap;gap:4px;align-items:center;}
.editor-button {font:inherit;color:inherit;background:#383838;border:1px solid #777;
 border-radius:5px;padding:8px;cursor:pointer;}
.editor-button:disabled {opacity:.5;cursor:default;}
#editor-heading {flex-basis:100%;font-size:14px;}
.header h1 {display:none;}
#editor-status {flex-basis:100%;margin:2px;white-space:pre-wrap;overflow-wrap:anywhere;}
#editor-command-form {display:flex;gap:6px;flex-wrap:wrap;align-items:center;padding:8px 0;margin:0;}
#editor-command-prompt {align-self:center;}
.console-output {min-height:0;overflow-anchor:none;}
#editor-command {color:inherit;background:transparent;border:0;border-bottom:1px solid #777;padding:4px;}
#editor-command:focus-visible {outline:2px solid #90c8ff;outline-offset:-2px;}
#editor-command {min-width:0;flex:1;font:inherit;}
#editor-command-form button,#editor-command {min-height:44px;}
#editor-tabs {display:flex;flex-basis:100%;gap:4px;}
#editor-tabs button[aria-selected=true] {border-color:#90c8ff;background:#17466c;}
.left-sidebar,.center-panel,.right-sidebar,.bottom-panel {min-width:0;min-height:0;}
.left-sidebar button {display:block;min-height:44px;width:100%;text-align:left;margin:4px 0;}
.editor-component {padding-left:24px;font-size:var(--editor-font);}
.editor-field {display:block;margin:12px 0;white-space:pre-wrap;}
.editor-field input:not([type=checkbox]),.editor-field textarea,.editor-field select {
 display:block;width:100%;max-width:100%;min-height:44px;margin-top:5px;padding:8px;
 color:inherit;background:#252526;border:1px solid #777;font:inherit;}
.editor-field textarea {min-height:110px;resize:vertical;}
.editor-field input[type=checkbox] {width:24px;height:24px;vertical-align:middle;margin:10px;}
.component-item-header {min-height:44px;}
.state-value,.param-value,.console-line {overflow-wrap:anywhere;white-space:pre-wrap;}
.dynamic-row-editor {flex-wrap:wrap;}
.dynamic-input {flex-basis:100%;font-size:16px;}
.dynamic-save-btn {min-height:44px;}
#editor-mode {font:inherit;min-height:44px;max-width:100%;}
#editor-step-summary {padding:8px;white-space:pre-wrap;overflow-wrap:anywhere;}
@media(max-width:700px) {
 .layout {display:flex;flex-direction:column;}
 .editor-splitter {display:none;}
 .bottom-panel {overflow:hidden;}
 .header {flex:none;}
 .left-sidebar,.right-sidebar,.center-panel,.bottom-panel {
   display:none;flex:1;overflow:auto; padding:12px;}
 .layout[data-tab=log] .bottom-panel {overflow:hidden;}
 .layout[data-tab=hierarchy] .left-sidebar,
 .layout[data-tab=inspector] .right-sidebar,
 .layout[data-tab=simulation] .center-panel {display:block;}
 .layout[data-tab=log] .bottom-panel {display:flex;flex-direction:column;}
 .console-output {flex:1;min-height:0;}
 .inspector-header {margin:0 0 12px;}
 .svg-container {overflow:auto;min-height:0;}
 .header h1 {display:none;}
 #editor-toolbar {font-size:14px;}
 #editor-tabs {justify-content:space-between;}
 #editor-tabs button {flex:1;padding:6px 2px;}
}
</style>
"""

# Shared with the integrated editor; contains no DOM or project constructors.
DRAFT_HISTORY_SCRIPT = r"""
class ProjectDraftHistory {
  constructor(limit=50) { this.limit=limit; this.clear(); }
  clear() { this.past=[]; this.future=[]; this.group=null; }
  endGroup() { this.group=null; }
  record(before, group=null) {
    if(group===null || group!==this.group) {
      this.past.push(structuredClone(before));
      if(this.past.length>this.limit) this.past.shift();
    }
    this.future=[]; this.group=group;
  }
  move(from, to, current) {
    if(!from.length) return null;
    to.push(structuredClone(current)); this.endGroup();
    return structuredClone(from.pop());
  }
  undo(current) { return this.move(this.past,this.future,current); }
  redo(current) { return this.move(this.future,this.past,current); }
}
class ProjectComponentState {
  static reset(document,id,component=null,field=null){
    const states=document.dynamic_states;
    if(!states || !states[id])return;
    if(!component)delete states[id];
    else if(!field)delete states[id][component];
    else if(states[id][component]){delete states[id][component][field];if(!Object.keys(states[id][component]).length)delete states[id][component];}
    if(states[id] && !Object.keys(states[id]).length)delete states[id];
  }
  static transfer(document,from,to,oldKey,newKey,copy=false){
    const value=document.dynamic_states?.[from]?.[oldKey];
    if(value){(document.dynamic_states[to] ??= {})[newKey]=structuredClone(value);if(!copy)this.reset(document,from,oldKey);}
  }
}
class ProjectDraftOperations {
  static add(document, catalog, prototype, id, sourceId=null) {
    const matches=catalog.filter(x=>sourceId?x.instance.prototype===prototype && (prototype.startsWith('installed:')?x.key===x.instance.prefab:x.kind==='preset'):(x.key || x.instance.prototype)===prototype);
    if(matches.length!==1) throw Error('Choose one unambiguous registered prefab or preset key from catalog prefabs.');
    const entry=matches[0];
    if(document.instances.length>=100) throw Error('A project supports at most 100 instances.');
    if(document.instances.some(x=>x.id===id)) throw Error('Instance ID already exists.');
    const source=sourceId ? document.instances.find(x=>x.id===sourceId) : entry.instance;
    if(!source || source.prototype!==entry.instance.prototype) throw Error('Unknown source instance.');
    const item=structuredClone(source);item.id=id;
    const names=new Set(document.instances.map(x=>x.params.name));
    const base=source.params.name;let name=base, suffix=2;
    while(names.has(name)) name=base+' '+suffix++;
    item.params.name=name;
    for(const [field,role] of Object.entries(entry.references)) {
      if(item.params[field]===source.id) item.params[field]=id;
      else if(!document.instances.some(x=>x.id===item.params[field] && x.role===role)) {
        const target=document.instances.find(x=>x.role===role);
        if(!target) throw Error('Create a '+role+' reference target first.');
        item.params[field]=target.id;
      }
    }
    const owned=(document.components || []).filter(x=>x.instance===sourceId);
    if((document.components?.length || 0)+owned.length>100) throw Error('At most 100 authored components.');
    document.instances.push(item);
    if(sourceId && document.dynamic_states?.[sourceId])document.dynamic_states[id]=Object.fromEntries(Object.entries(document.dynamic_states[sourceId]).filter(([key])=>!key.startsWith('authored_')).map(([key,value])=>[key,structuredClone(value)]));
    if(sourceId && document.components) {
      for(const component of owned){const newId=crypto.randomUUID();document.components.push({...structuredClone(component),id:newId,instance:id});ProjectComponentState.transfer(document,sourceId,id,'authored_'+component.id,'authored_'+newId,true);}
    }
    return id;
  }
  static remove(document,catalog,id) {
    const item=document.instances.find(x=>x.id===id);
    if(!item) throw Error('Select an instance.');
    if(['entity','game_master'].includes(item.role) && document.instances.filter(x=>x.role===item.role).length===1)
      throw Error('Keep at least one player entity and one game master entity.');
    for(const other of document.instances.filter(x=>x.id!==id)) {
      const entry=catalog.find(x=>x.instance.prototype===other.prototype);
      for(const field of Object.keys(entry?.references || {}))
        if(other.params[field]===id) throw Error(other.params.name+' references this instance through '+field+'. Change that reference first.');
    }
    document.instances=document.instances.filter(x=>x.id!==id);
    if(document.components)document.components=document.components.filter(x=>x.instance!==id);
    if(document.dynamic_states)delete document.dynamic_states[id];
    return document.instances[0]?.id || 'simulation';
  }
  static move(document,id,offset) {
    const index=document.instances.findIndex(x=>x.id===id);
    if(index<0) return;
    let target=index+offset;
    while(target>=0 && target<document.instances.length && document.instances[target].role!==document.instances[index].role) target+=offset;
    if(target<0 || target>=document.instances.length) return;
    const [item]=document.instances.splice(index,1);document.instances.splice(target,0,item);
  }
}
class ProjectRecordOperations {
  static locate(document, selection) {
    const [kind,id]=String(selection).split(':');
    if(!['components'].includes(kind)) return null;
    const index=document[kind]?.findIndex(x=>x.id===id) ?? -1;
    return index<0 ? null : {kind,index,item:document[kind][index]};
  }
  static remove(document,selection) {
    const found=this.locate(document,selection);
    if(!found) throw Error('Select a component.');
    const {kind,item}=found;
    ProjectComponentState.reset(document,item.instance,'authored_'+item.id);
    document[kind].splice(found.index,1);return kind==='components'?item.instance:kind+':'+document[kind][0].id;
  }
  static duplicate(document,selection,id) {
    const found=this.locate(document,selection);
    if(!found) throw Error('Selection no longer exists.');
    const {kind,item}=found;
    ProjectComponentOperations.checkId(document[kind],id);
    if(document[kind].length>=100) throw Error('At most 100 records per section.');
    ProjectComponentState.transfer(document,item.instance,item.instance,'authored_'+item.id,'authored_'+id,true);
    const copy=structuredClone(item);copy.id=id;const base=copy.name+' copy';copy.name=base;let n=2;while(document[kind].some(x=>x.name===copy.name))copy.name=base+' '+n++;document[kind].push(copy);return kind+':'+id;
  }
  static move(document,selection,offset) {
    const found=this.locate(document,selection);
    if(!found) throw Error('Selection no longer exists.');
    if(offset!==1 && offset!==-1) throw Error('Move one position at a time.');
    const {kind,index,item}=found;let target=index+offset;
    while(kind==='components' && target>=0 && target<document[kind].length && document[kind][target].instance!==item.instance) target+=offset;
    if(target<0 || target>=document[kind].length) return;
    document[kind].splice(index,1);document[kind].splice(target,0,item);
  }
}

class ProjectValidation {
  static target(document,message) {
    // Registry paths contain either list positions or stable IDs. Never guess
    // when a numeric ID could denote a different record at that position.
    const dynamic=/^\$\.dynamic_states\.([A-Za-z0-9_-]+)\./.exec(message);
    if(dynamic && document.instances.some(x=>x.id===dynamic[1]))return {selection:dynamic[1],field:null};
    const root=/^\$\.(premise|max_steps): /.exec(message);
    if(root) return {selection:'simulation',field:root[1]==='premise'?'editor-premise':'editor-max-steps'};
    const match=/^\$\.(instances|components)\[([A-Za-z0-9_-]{1,128})\](?:\.([A-Za-z0-9_.-]+))?: /.exec(message);
    if(!match) return null;
    const [,kind,key,path='']=match, records=document[kind] || [];
    const candidates=records.filter(x=>x.id===key);
    const index=/^(0|[1-9][0-9]*)$/.test(key)?Number(key):-1;
    if(records[index] && !candidates.includes(records[index])) candidates.push(records[index]);
    if(candidates.length!==1) return null;
    const item=candidates[0];
    if(typeof item.id!=='string' || records.filter(x=>x.id===item.id).length!==1) return null;
    let field=null;
    if(kind==='instances' && path.startsWith('params.') && item.params && Object.hasOwn(item.params,path.slice(7))) field='editor-'+item.id+'-'+path.slice(7);
    if(kind==='components' && path.startsWith('params.') && item.params && Object.hasOwn(item.params,path.slice(7))) field='component-param-'+path.slice(7);
    if(kind!=='instances' && path==='name') field='world-name';
    return {selection:kind==='instances'?item.id:kind+':'+item.id,field};
  }
  static forSave(document,operation,args,error) {
    if(operation!=='project.save' || error?.code!=='invalid_edit' || args.text!==JSON.stringify(document)) return null;
    return this.target(document,error.message);
  }
}

class ProjectReferences {
  static uses(document,catalog,selection) {
    const uses=[];
    const add=(owner,field,label,owned=false)=>uses.push({selection:owner,field,label,owned});
    const instance=document.instances.find(x=>x.id===selection);
    if(instance){
      for(const item of document.instances){
        const spec=catalog.find(x=>x.instance.prototype===item.prototype);
        for(const field of Object.keys(spec?.references || {}))
          if(item.params[field]===instance.id) add(item.id,field,item.params.name+' · '+field);
      }
      for(const item of document.components || [])
        if(item.instance===instance.id) add('components:'+item.id,'instance',item.name+' · owned component',true);
    }
    return uses;
  }
  static candidates(document,catalog,id) {
    const source=document.instances.find(x=>x.id===id);
    if(!source || !catalog.some(x=>x.instance.prototype===source.prototype)) return [];
    return document.instances.filter(x=>x.id!==id && x.role===source.role &&
      catalog.some(c=>c.instance.prototype===x.prototype && c.instance.role===x.role));
  }
  static replace(document,catalog,id,targetId) {
    if(!this.candidates(document,catalog,id).some(x=>x.id===targetId))
      throw Error('Choose an existing compatible reference target.');
    const uses=this.uses(document,catalog,id).filter(x=>!x.owned);
    if(!uses.length) throw Error('This instance has no replaceable references.');
    // Plan against a copy so a stale/invalid request never partially edits a draft.
    const next=structuredClone(document);
    for(const use of uses){
      const instance=next.instances.find(x=>x.id===use.selection);
      const owner=instance?.params || ProjectRecordOperations.locate(next,use.selection)?.item;
      if(!owner || !Object.hasOwn(owner,use.field)) throw Error('Reference no longer exists.');
      owner[use.field]=Array.isArray(owner[use.field])
        ? [...new Set(owner[use.field].map(x=>x===id?targetId:x))] : targetId;
    }
    Object.assign(document,next);
    return id;
  }
}

class ProjectComponentOperations {
  static checkId(records,id) {
    if(typeof id!=='string' || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(id) || records.some(x=>x.id===id))
      throw Error('Choose a unique stable component ID.');
  }
  static add(document,catalog,instanceId,type,id) {
    const instance=document.instances.find(x=>x.id===instanceId), spec=catalog.find(x=>x.key===type);
    if(!instance || !spec?.prototypes.includes(instance.prototype)) throw Error('Choose a compatible registered component.');
    if(!Array.isArray(document.components)) throw Error('This template does not support component authoring.');
    this.checkId(document.components,id);
    if(document.components.length>=100) throw Error('At most 100 authored components.');
    let name=type,n=2;while(document.components.some(x=>x.instance===instanceId && x.name===name))name=type+' '+n++;
    document.components.push({id,instance:instanceId,type,name,params:structuredClone(spec.defaults)});
    return 'components:'+id;
  }
  static moveOwner(document,catalog,id,targetId) {
    const item=document.components?.find(x=>x.id===id);
    if(!item) throw Error('Component no longer exists.');
    const spec=catalog.find(x=>x.key===item.type);
    const target=document.instances.find(x=>x.id===targetId);
    if(!target || !spec?.prototypes.includes(target.prototype)) throw Error('Choose a compatible component owner.');
    if(item.instance===targetId) return 'components:'+id;
    document.components.splice(document.components.indexOf(item),1);
    ProjectComponentState.transfer(document,item.instance,targetId,'authored_'+item.id,'authored_'+item.id);
    item.instance=targetId;document.components.push(item);
    return 'components:'+id;
  }
  static configure(document,id,field,value) {
    const item=document.components?.find(x=>x.id===id);
    if(!item) throw Error('Component no longer exists.');
    if(!Object.hasOwn(item.params,field)) throw Error('Unknown component parameter.');
    item.params[field]=value;
  }
  static rename(document,id,name) {
    const item=document.components?.find(x=>x.id===id);
    if(!item) throw Error('Component no longer exists.');
    item.name=name;
  }
}

class ProjectDraftCommands {
  static matches(item,query,components=[],owned=[]){
    const values=item.params?[item.params.name,item.name,item.type,item.id,item.prefab,...components,...owned.flatMap(c=>[c.name,c.type])]:[item.name,item.id];
    return values.some(value=>String(value).toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()));
  }
  static read(document,metadata,base,runtime,view,selection,action,args){
    const id=args[0]==='.'?selection:(args[0] || selection);
    if(action==='catalog')return args[0]==='templates'?(metadata.templates || [document.template]):args[0]==='components'?metadata.component_catalog:args[0]==='prefabs'?metadata.catalog:{template:document.template,prefabs:metadata.catalog,components:metadata.component_catalog,unavailable:metadata.prefab_diagnostics || []};
    if(action==='list')return args.length?(document[args[0]] || []):Object.fromEntries(['instances','components'].map(k=>[k,document[k] || []]));
    if(action==='locate'){
      const target=ProjectValidation.target(document,args[0]);
      if(!target)throw Error('No unambiguous field in this draft matches that validation message.');
      return target;
    }
    if(action==='get' || action==='params')return ProjectDraftCommands.describe(document,metadata,base,runtime,view,selection,action,args);
    const item=id==='simulation'?document:ProjectRecordOperations.locate(document,id)?.item || document.instances.find(x=>x.id===id);
    if(!item)throw Error('Unknown selection.');
    if(id==='simulation' || !document.instances.includes(item)){if(args[1])throw Error('Component inspection requires an instance ID.');return item;}
    const index=(base?.instances || document.instances).findIndex(x=>x.id===id);
    if(view==='runtime' && !runtime)throw Error('No current runtime to inspect.');
    const entity=(view==='runtime'?runtime:metadata)?.entities?.['entity_'+index];
    const components=entity?.component_info?.context_components || {};
    if(args[1]){if(!Object.hasOwn(components,args[1]))throw Error('Unknown component; inspect the instance to list available components.');return components[args[1]];}
    return {instance:item,fields:metadata.catalog?.find(x=>x.instance.prototype===item.prototype)?.inspector,entity:entity || null,authored_components:(document.components || []).filter(c=>c.instance===id),preview_note:view==='runtime'?'Current runtime state.':entity?'Preview reflects the last save/load.':'Save to build a component preview.'};
  }
  static preview(value){const text=typeof value==='string'?value:JSON.stringify(value);return text!==undefined && text.length>100?text.slice(0,99)+'…':value;}
  static describe(document,metadata,base,runtime,view,selection,action,args){
    const id=args[0]==='.'?selection:(args[0] || selection);
    const inspect=a=>ProjectDraftCommands.read(document,metadata,base,runtime,view,selection,'inspect',a);
    if(action==='get' && args.length===3){
      const component=inspect([id,args[1]]),state=component.state || {},dynamic=component.dynamic_state || {};
      const source=Object.hasOwn(state,args[2])?state:Object.hasOwn(dynamic,args[2])?dynamic:null;
      if(!source)throw Error('Unknown component state field; params '+id+' '+args[1]+' lists its fields.');
      return {id,component:args[1],field:args[2],value:source[args[2]],source:view==='runtime'?'current runtime':'preview of the saved definition'};
    }
    const item=id==='simulation'?document:ProjectRecordOperations.locate(document,id)?.item || document.instances.find(x=>x.id===id);
    if(!item)throw Error('Unknown selection.');
    if(action==='get'){
      const parts=args[1].split('.');
      const target=parts.length===2 && parts[0]==='params'?item.params:parts.length===1?item:null;
      const key=parts.at(-1);
      if(!target || !Object.hasOwn(target,key))throw Error('Unknown field; params '+id+' lists the configurable fields.');
      return {id,field:args[1],value:target[key]};
    }
    if(id==='simulation')return {id,fields:['premise','max_steps'].filter(k=>Object.hasOwn(item,k)).map(k=>({name:k,value:ProjectDraftCommands.preview(item[k]),set_with:'set simulation '+k+' JSON'}))};
    if(!document.instances.includes(item)){
      const type=(metadata.component_catalog || []).find(x=>x.key===item.type);
      return {id:'components:'+item.id,type:item.type,owner:item.instance,description:type?.description,
        parameters:Object.entries(item.params || {}).map(([name,value])=>({name,value:ProjectDraftCommands.preview(value),default:type?.defaults?.[name],set_with:'set components:'+item.id+' params.'+name+' JSON'}))};
    }
    if(args[1]){
      const component=inspect([id,args[1]]);
      return {id,component:args[1],class_name:component.class_name,module:component.module,
        constructor_parameters:component.parameters || [],
        editable_state:Object.keys(component.dynamic_state || {}),
        state_fields:Object.keys(component.state || {}),
        note:'The prefab passes constructor parameters when it builds this component; configure them through the prefab parameters (params '+id+'). Editable state uses state-field (initial) or edit (paused runtime).'};
    }
    const entry=(metadata.catalog || []).find(x=>x.instance.prototype===item.prototype);
    const fixed=entry?.fixed_parameters || [], labels=entry?.inspector || {};
    const index=(base?.instances || document.instances).findIndex(x=>x.id===id);
    const entity=(view==='runtime'?runtime:metadata)?.entities?.['entity_'+index];
    return {id,prefab:item.prefab,role:item.role,
      parameters:Object.entries(item.params || {}).map(([name,value])=>({name,value:ProjectDraftCommands.preview(value),default:ProjectDraftCommands.preview(entry?.instance?.params?.[name]),type:value===null?'null':Array.isArray(value)?'list':typeof value,label:labels[name]?.label,set_with:'set '+id+' params.'+name+' JSON'})),
      fixed_parameters:fixed,
      components:Object.keys(entity?.component_info?.context_components || {}),
      note:'get '+id+' params.NAME prints a full value; params '+id+' COMPONENT lists one component.'};
  }
  static apply(document,metadata,selection,action,args,identifier=null){
    const catalog=metadata.catalog || [], components=metadata.component_catalog || [];
    args=args.map(x=>x==='.'?selection:x);
    const id=args[0] || selection;
    if(action==='state-reset'){if(!document.instances.some(x=>x.id===id))throw Error('Unknown entity.');ProjectComponentState.reset(document,id,args[1],args[2]);return id;}
    if(action==='state-field'){
      if(!document.dynamic_states || !metadata.initial_state_editable)throw Error('Initial component editing requires a document with dynamic state and a fresh Simulation preview factory.');
      const instance=document.instances.find(x=>x.id===id);
      if(!instance)throw Error('Unknown entity.');
      const index=metadata.document?.instances?.findIndex(x=>x.id===id) ?? document.instances.findIndex(x=>x.id===id);
      const dynamic=metadata.entities?.['entity_'+index]?.component_info?.context_components?.[args[1]]?.dynamic_state;
      if(!dynamic || !Object.hasOwn(dynamic,args[2]))throw Error('Field is not advertised as editable; save newly created entities to inspect their components.');
      const value=JSON.parse(args[3]);
      ((document.dynamic_states[id] ??= {})[args[1]] ??= {})[args[2]]=value;
      return id;
    }
    if(action==='move'){
      if(ProjectRecordOperations.locate(document,id))ProjectRecordOperations.move(document,id,args[1]==='up'?-1:1);
      else {if(!document.instances.some(x=>x.id===id))throw Error('Unknown instance.');ProjectDraftOperations.move(document,id,args[1]==='up'?-1:1);}
      return id;
    }
    if(action==='add') {
      identifier=identifier ?? crypto.randomUUID();
      ProjectComponentOperations.checkId([...document.instances,...(document.components || [])],identifier);
      if(args[0]==='instance') return ProjectDraftOperations.add(document,catalog,args[1],identifier);
      if(args[0]==='component') return ProjectComponentOperations.add(document,components,args[2],args[1],identifier);
      throw Error('Choose an instance or component.');
    }
    if(action==='duplicate') {
      if(ProjectRecordOperations.locate(document,id)) return ProjectRecordOperations.duplicate(document,id,crypto.randomUUID());
      const source=document.instances.find(x=>x.id===id);
      if(!source) throw Error('Unknown instance.');
      return ProjectDraftOperations.add(document,catalog,source.prototype,crypto.randomUUID(),id);
    }
    if(action==='remove') return ProjectRecordOperations.locate(document,id)?ProjectRecordOperations.remove(document,id):ProjectDraftOperations.remove(document,catalog,id);
    if(action==='replace'){ProjectReferences.replace(document,catalog,id,args[1]);return selection;}
    if(action==='move-component'){ProjectComponentOperations.moveOwner(document,components,id.replace(/^components:/,''),args[1]);return selection;}
    if(action==='set'){
      const item=id==='simulation'?document:ProjectRecordOperations.locate(document,id)?.item || document.instances.find(x=>x.id===id);
      if(!item) throw Error('Unknown authored record.');
      const parts=args[1].split('.');
      const target=parts.length===2 && parts[0]==='params'?item.params:parts.length===1?item:null;
      const key=parts.at(-1);
      if(!target || !Object.hasOwn(target,key) || ['id','prototype','prefab','role','schema_version','template','instances','components','dynamic_states','params','instance','type'].includes(key)) throw Error('Choose an existing editable field (or params.FIELD).');
      target[key]=JSON.parse(args[2]);return id;
    }
    throw Error('Not an authored draft action.');
  }
}

"""

EDITOR_SCRIPT = (
    '<script>\n'
    + 'const SESSION_HELP='
    + json.dumps(session_commands.HELP)
    + ';\n'
    + DRAFT_HISTORY_SCRIPT
    + r"""
(() => {
  const $ = id => document.getElementById(id);
  const layout = document.querySelector('.layout');
  const toolbar = document.createElement('div'); toolbar.id = 'editor-toolbar';
  document.querySelector('.header').append(toolbar);
  const heading=document.createElement('strong'); heading.id='editor-heading';
  const editorTitle=document.querySelector('.header h1').textContent;
  heading.textContent=editorTitle.split(' · ')[0];toolbar.append(heading);
  let envelope, draft, draftRevision, draftDefinition, draftBaseDocument, selectedId, connected = false, sending = false;
  let dirty = false, runtimeMode = false, renderedView = '', loggedRun, loggedSteps = 0, loggedFailure, loggedCompletion;
  const runtimeDrafts = new Map();
  const history = new ProjectDraftHistory();
  const draftSnapshot = () => ({document:draft, selectedId});
  function button(label, action, parent=toolbar) {
    const b = document.createElement('button'); b.textContent=label;
    b.className='editor-button'; b.onclick=action; parent.append(b); return b;
  }
  const runStepsLabel=document.createElement('label');runStepsLabel.textContent='Steps to run ';
  const runSteps=document.createElement('input');runSteps.id='editor-requested-steps';
  runSteps.type='number';runSteps.min='1';runSteps.step='1';runSteps.style.width='5em';
  runStepsLabel.append(runSteps);toolbar.append(runStepsLabel);
  runSteps.oninput=()=>controls();
  const run = button('Run', () => {
    if(state().state==='paused') return dispatch('runtime.play');
    const args={revision:draftRevision};
    if(state().run_limits) {
      if(!validRequestedSteps()) return report(Error('Enter a whole number of steps within the saved maximum.'));
      args.requested_steps=Number(runSteps.value);
    }
    return dispatch('project.run',args);
  });
  const pause = button('Pause', () => dispatch('runtime.pause'));
  const step = button('Step', () => dispatch('runtime.step'));
  const reset = button('Reset', () => dispatch('project.reset'));
  const save = button('Save draft', async () => {
    const ok = await dispatch('project.save', {text:JSON.stringify(draft), revision:draftRevision});
    if (ok) {error.textContent=''; adopt(true);}
  });
  const undo = button('Undo', () => restoreHistory('undo'));
  const redo = button('Redo', () => restoreHistory('redo'));
  const prototypePicker=document.createElement('select');prototypePicker.id='editor-prototype';
  prototypePicker.setAttribute('aria-label','Prefab or named preset');toolbar.append(prototypePicker);
  const add=button('Add instance',()=>structural(next=>ProjectDraftOperations.add(next,catalog(),prototypePicker.value,crypto.randomUUID())));
  const duplicate=button('Duplicate',()=>structural(next=>{
    if(ProjectRecordOperations.locate(next,selectedId)) return ProjectRecordOperations.duplicate(next,selectedId,crypto.randomUUID());
    const source=next.instances.find(x=>x.id===selectedId);
    if(!source) throw Error('Select an instance to duplicate.');
    return ProjectDraftOperations.add(next,catalog(),source.prototype,crypto.randomUUID(),source.id);
  }));
  const remove=button('Remove',()=>structural(next=>ProjectRecordOperations.locate(next,selectedId)?ProjectRecordOperations.remove(next,selectedId):ProjectDraftOperations.remove(next,catalog(),selectedId)));
  const exportButton = button('Export JSON', () => download(JSON.stringify(draft,null,2)+'\n','concordia-project.json'));
  const file=document.createElement('input'); file.id='editor-file';
  file.type='file'; file.accept='.json,application/json'; file.hidden=true; toolbar.append(file);
  const open=button('Open JSON',()=>file.click());
  file.onchange=async()=>{
    try {
      if(!file.files.length)return;
      if(file.files[0].size>900000)throw Error('Maximum project size is 900 kB.');
      if(dirty)throw Error('Unsaved draft: save/export it or reload --discard before load.');
      const preview=await query('project.preview',{text:await file.files[0].text()});
      draft=preview.document;draftDefinition=preview.definition;draftBaseDocument=preview.document;
      runtimeMode=false;mode.value='definition';
      history.clear();dirty=JSON.stringify(draft)!==JSON.stringify(state().document);
      selectedId=draft.instances[0].id;renderedView='';render();
      notice('Loaded local draft. Save explicitly to update the session.','success');
    } catch(e) {report(e);} finally {file.value='';}
  };
  button('Reload saved',()=>{dirty=false;runtimeDrafts.clear();adopt();notice('Reloaded saved definition; local draft and undo history discarded.');});
  const mode=document.createElement('select'); mode.id='editor-mode'; mode.setAttribute('aria-label','Inspector source');
  for(const [value,label] of [['definition','Initial definition'],['runtime','Current runtime']]) {
    const option=document.createElement('option'); option.value=value; option.textContent=label; mode.append(option);
  }
  toolbar.append(mode);
  mode.onchange=()=>{runtimeMode=mode.value==='runtime';renderedView='';render();};
  const status=document.createElement('p');status.id='editor-status';status.setAttribute('role','status');
  const error={set textContent(value){if(value) report(Error(value));}};
  const tabs=document.createElement('nav');tabs.id='editor-tabs';tabs.setAttribute('aria-label','Editor panels');
  toolbar.append(status,tabs);
  function tab(name) {
    layout.dataset.tab=name;
    tabs.querySelectorAll('button').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.tab===name)));
  }
  for(const name of ['hierarchy','inspector','simulation','log']) {
    const b=button(name[0].toUpperCase()+name.slice(1),()=>tab(name),tabs);b.dataset.tab=name;
  }
  tab('hierarchy');
  const hierarchy=document.querySelector('.left-sidebar');hierarchy.replaceChildren();
  const search=document.createElement('input');search.type='search';search.id='editor-search';
  search.placeholder='Find entities or components';search.setAttribute('aria-label','Search hierarchy');
  toolbar.append(search);search.oninput=()=>{renderHierarchy();controls();};
  const engineLabel=document.createElement('div');engineLabel.id='editor-engine';toolbar.append(engineLabel);
  const viewerSelect=document.createElement('select');viewerSelect.id='editor-viewer';viewerSelect.setAttribute('aria-label','Central viewer');
  const defaultViewer=document.createElement('option');defaultViewer.value='default';defaultViewer.textContent='Default visualization';viewerSelect.append(defaultViewer);toolbar.append(viewerSelect);
  const viewerFrame=document.createElement('iframe');viewerFrame.id='editor-viewer-frame';viewerFrame.title='Custom simulation viewer';viewerFrame.hidden=true;
  viewerFrame.setAttribute('sandbox','allow-scripts allow-forms allow-popups');viewerFrame.referrerPolicy='no-referrer';viewerFrame.style.cssText='width:100%;height:100%;min-height:12rem;border:0;';
  document.querySelector('.center-panel').append(viewerFrame);
  const localViewers=new Map();let currentViewer='default',viewerBlob=null;
  window.addEventListener('pagehide',()=>{if(viewerBlob)URL.revokeObjectURL(viewerBlob);});
  function offerViewer(name){if(![...viewerSelect.options].some(x=>x.value===name)){const option=document.createElement('option');option.value=name;option.textContent=name;viewerSelect.append(option);}}
  async function showViewer(name,force=false){
    if(name===currentViewer && !force)return;
    if(name==='default'){viewerFrame.hidden=true;document.querySelector('.svg-container').hidden=false;viewerSelect.value=name;currentViewer=name;if(force)await refresh();return;}
    const source=localViewers.get(name) || await query('viewer.read',{name});
    viewerFrame.removeAttribute('srcdoc');viewerFrame.removeAttribute('src');
    if(viewerBlob){URL.revokeObjectURL(viewerBlob);viewerBlob=null;}
    if(source.html!==undefined){viewerBlob=URL.createObjectURL(new Blob([source.html],{type:'text/html'}));viewerFrame.src=viewerBlob;}else viewerFrame.src=source.url;
    viewerFrame.hidden=false;document.querySelector('.svg-container').hidden=true;offerViewer(name);viewerSelect.value=name;currentViewer=name;
    if(source.url)notice('HTML viewer opened. If the site blocks embedding, open its URL in a separate tab; the editor cannot inspect cross-origin frame errors.','info');
  }
  viewerSelect.onchange=()=>showViewer(viewerSelect.value).catch(report);
  button('Refresh viewer',()=>showViewer(currentViewer,true).catch(report));
  const viewerFile=document.createElement('input');viewerFile.type='file';viewerFile.accept='.html,text/html';viewerFile.hidden=true;toolbar.append(viewerFile);
  button('Open viewer HTML',()=>viewerFile.click());
  viewerFile.onchange=async()=>{try{const file=viewerFile.files[0];if(!file)return;if(file.size>5000000)throw Error('Maximum viewer HTML size is 5 MB.');const name='Local: '+file.name;localViewers.set(name,{html:await file.text()});await showViewer(name,true);}catch(e){report(e);}finally{viewerFile.value='';}};
  const viewerURL=document.createElement('input');viewerURL.type='url';viewerURL.placeholder='https://example.org/viewer';viewerURL.setAttribute('aria-label','Viewer URL');toolbar.append(viewerURL);
  function openViewerURL(value){const url=new URL(value);if(!['http:','https:'].includes(url.protocol) || url.username || url.password)throw Error('Use an HTTP(S) viewer URL without embedded credentials.');localViewers.set('Custom URL',{url:url.href});return showViewer('Custom URL',true);}
  button('Open viewer URL',()=>{try{openViewerURL(viewerURL.value).catch(report);}catch(e){report(e);}});
  const summary=document.createElement('div');summary.id='editor-step-summary';
  document.querySelector('.center-panel').prepend(summary);
  document.querySelector('.console-header').textContent='Simulation log';
  logConsole(editorTitle,'info');
  logConsole('Entries show engine-reported player actions, not dialogue transcripts. Timestamps are browser-local display times, including replay after reconnect; they are not simulation time.', 'info');
  const limit=document.createElement('p');limit.id='editor-run-limit';
  summary.before(limit);
  const notices=new Set();
  function notice(message,type='info',key){
    if(key && notices.has(key)) return null;
    if(key) notices.add(key);
    return logConsole(message,type);
  }
  let connectionKnown=false;
  function connection(value){
    if(!connectionKnown || value!==connected) notice(value?'Connected to editor.':'Disconnected; drafts kept. Commands are not replayed.','info');
    connectionKnown=true;connected=value;
  }
  function state(){return envelope?.result;}
  function inspectionDocument(){return runtimeMode?(state().runtime?.document || state().document):draftBaseDocument;}
  function report(e, operation){
    const message=`${operation ? operation+' failed: ' : ''}${e.message || String(e)}`;
    const key=operation?undefined:'error:'+message+':'+JSON.stringify(e.validationTarget?draft:null);
    const line=notice(message,'error',key);
    if(line)tab('log');
    if(e.validationTarget && line){
      const target=e.validationTarget, submitted=JSON.stringify(draft);
      const link=button(target.field?'Show invalid field':'Show invalid item',()=>{
        if(runtimeMode) return;
        if(JSON.stringify(draft)!==submitted){report(Error('Draft changed since validation. Save again to locate the current error.'));return;}
        search.value='';choose(target.selection);
        const input=target.field && $(target.field);
        if(input){input.focus();input.scrollIntoView({block:'center'});}
      },line);
      link.dataset.validationAction='true';
    }
  }
  function catalog(){return draftDefinition?.catalog || [];}
  function structural(change,propagate=false){
    if(runtimeMode || !connected || sending || state()?.run.status==='active'){if(propagate)throw Error('Authoring unavailable in this state.');return false;}
    try {
      const next=structuredClone(draft), selection=change(next);
      if(JSON.stringify(next)===JSON.stringify(draft)) return false;
      history.record(draftSnapshot());draft=next;selectedId=selection;
      dirty=JSON.stringify(draft)!==JSON.stringify(state().document);
      error.textContent='';search.value='';renderedView='';render();tab('inspector');notice('Draft updated; not saved.','success');return true;
    } catch(e) {if(propagate)throw e;report(e);return false;}
  }
  function restoreHistory(direction){
    if(runtimeMode || !connected || sending || state()?.run.status==='active') return;
    const restored=history[direction](draftSnapshot());
    if(!restored) return;
    draft=restored.document; selectedId=restored.selectedId;
    dirty=JSON.stringify(draft)!==JSON.stringify(state().document);
    error.textContent=''; renderedView=''; render();notice(direction+' applied to this draft.','success');
    // Keep the original draft revision: undo must not resolve a stale-tab conflict.
    if(draftRevision!==state().revision) report(Error('Definition changed in another tab. Review before Reload saved.'));
  }
  function adopt(preserveHistory=false){
    if(!state()) return;
    if(!preserveHistory) history.clear(); else history.endGroup();
    draftDefinition=state().definition;draftBaseDocument=state().document;
    draft=structuredClone(state().document);draftRevision=state().revision;dirty=false;
    selectedId=selectedId || draft.instances[0].id;renderedView='';render();
  }
  function validRequestedSteps(){
    const maximum=state()?.run_limits?.maximum_steps;
    const value=Number(runSteps.value);
    return !maximum || (runSteps.value.trim()!=='' && Number.isInteger(value) && value>=1 && value<=maximum);
  }
  function updateRunSteps(){
    const s=state(), limits=s?.run_limits;
    runStepsLabel.hidden=!limits;
    if(!limits) return;
    runSteps.max=String(limits.maximum_steps);
    runSteps.title=`Saved simulation maximum: ${limits.maximum_steps} steps`;
    if(runSteps.dataset.initialized!=='true') {
      runSteps.value=String(limits.default_requested_steps);
      runSteps.dataset.initialized='true';
    }
    if(s.run.status==='active' && s.run.requested_steps!==undefined) {
      runSteps.value=String(s.run.requested_steps);
    }
  }
  function controls(){
    updateRunSteps();
    const s=state(), phase=s?.state;
    const active=s?.run.status==='active', unavailable=!connected || sending || !s;
    run.textContent=phase==='paused'?'Resume':'Run';
    runSteps.disabled=unavailable || active;
    run.disabled=unavailable || (phase!=='paused' && (active || dirty || draftRevision!==s.revision || !validRequestedSteps()));
    pause.disabled=unavailable || phase!=='running';step.disabled=unavailable || phase!=='paused';
    reset.disabled=unavailable || phase==='ready' || phase==='stopping';
    save.disabled=unavailable || active;open.disabled=unavailable || active;
    undo.disabled=unavailable || active || runtimeMode || !history.past.length;
    redo.disabled=unavailable || active || runtimeMode || !history.future.length;
    const structureDisabled=unavailable || active || runtimeMode || !catalog().length;
    for(const control of [prototypePicker,add,duplicate,remove]) {
      control.hidden=!catalog().length;control.disabled=structureDisabled;
    }
    const position=draft?.instances.findIndex(x=>x.id===selectedId) ?? -1;
    duplicate.disabled=remove.disabled=structureDisabled || position<0;
    const world=draft && ProjectRecordOperations.locate(draft,selectedId);
    if(world) duplicate.disabled=remove.disabled=structureDisabled;
    document.querySelectorAll('[data-author-action]').forEach(x=>x.disabled=structureDisabled);
    document.querySelectorAll('[data-validation-action]').forEach(x=>x.disabled=runtimeMode || !draft);
    exportButton.disabled=!s;
    document.querySelectorAll('[data-definition-field]').forEach(x=>x.disabled=unavailable || active || runtimeMode);
    document.querySelectorAll('.dynamic-save-btn').forEach(b=>{
      const data=entityData[selectedEntity], component=data?.component_info?.context_components?.[b.dataset.component];
      const supported=Object.hasOwn(component?.dynamic_state || {},b.dataset.stateKey);
      b.hidden=!supported || (!runtimeMode && (!draft.dynamic_states || !draftDefinition.initial_state_editable));
      b.disabled=unavailable || (runtimeMode?phase!=='paused':active);
      const input=$(b.dataset.inputId); if(input) input.readOnly=!supported || unavailable || (runtimeMode?phase!=='paused':active || !draftDefinition.initial_state_editable);
    });
    status.textContent=(!connected?'Disconnected':phase || 'Connecting')+
      (s ? ` · Step ${s.current_step} · ${dirty?'unsaved changes':'saved definition'}` : '');
  }
  function field(container,key,value,change,id,spec={}){
    const label=document.createElement('label');label.className='editor-field';
    label.textContent=spec.label || key;
    const input=document.createElement(spec.choices?'select':typeof value==='string'?'textarea':'input');
    input.id=id;input.dataset.definitionField='true';
    if(spec.choices) for(const choice of spec.choices){
      const option=document.createElement('option');option.value=typeof choice==='string'?choice:choice.value;
      option.textContent=typeof choice==='string'?choice:choice.label;input.append(option);
    }
    if(typeof value==='boolean'){input.type='checkbox';input.checked=value;}
    else {if(typeof value==='number') input.type='number';input.value=value;}
    input.onblur=()=>history.endGroup();
    input.oninput=()=>{
      const before=structuredClone(draftSnapshot());
      change(typeof value==='boolean'?input.checked:typeof value==='number'?(input.value===''?null:Number(input.value)):input.value);
      if(JSON.stringify(before.document)!==JSON.stringify(draft)) history.record(before,id);
      dirty=JSON.stringify(draft)!==JSON.stringify(state().document);
      error.textContent='';renderHierarchy();if(spec.refresh)inspect();controls();
    };
    label.append(input);container.append(label);
  }
  function inspectReferences(content,selection){
    if(runtimeMode || !catalog().length) return;
    const section=document.createElement('section');section.setAttribute('aria-label','Used by');
    const heading=document.createElement('h3');heading.textContent='Used by';section.append(heading);
    const uses=ProjectReferences.uses(draft,catalog(),selection);
    for(const use of uses){
      const link=button(use.label,()=>choose(use.selection),section);link.dataset.referenceSource=use.selection;
    }
    if(!uses.length){const hint=document.createElement('p');hint.textContent='No incoming authored references.';section.append(hint);}
    const replaceable=uses.filter(x=>!x.owned);
    const instance=draft.instances.find(x=>x.id===selection);
    if(instance && catalog().length && replaceable.length){
      const choices=ProjectReferences.candidates(draft,catalog(),selection);
      const hint=document.createElement('p');hint.textContent='Replace '+replaceable.length+' reference(s) in this draft. Owned components stay with this instance. Undo restores all affected references.';section.append(hint);
      if(choices.length){
        const label=document.createElement('label');label.className='editor-field';label.textContent='Replacement instance';
        const picker=document.createElement('select');picker.id='reference-target';picker.dataset.definitionField='true';
        for(const item of choices){const option=document.createElement('option');option.value=item.id;option.textContent=item.params.name;picker.append(option);}
        label.append(picker);section.append(label);
        const replace=button('Replace references',()=>structural(next=>ProjectReferences.replace(next,catalog(),selection,picker.value)),section);replace.dataset.authorAction='true';
      } else {const hint=document.createElement('p');hint.textContent='Create another compatible instance before replacing references.';section.append(hint);}
    }
    content.append(section);
  }
  function inspectWorld(world){
    const {kind,item}=world, content=$('inspector-content');
    content.replaceChildren();content.style.display='block';$('inspector-empty').style.display='none';
    $('inspector-title').textContent=item.name;
    $('inspector-subtitle').textContent='Initial definition · '+kind.replaceAll('_',' ')+' · '+item.id;
    if(kind==='components'){
      const spec=draftDefinition.component_catalog.find(x=>x.key===item.type);
      const description=document.createElement('p');description.textContent=spec.description+' Requires: '+(spec.dependencies.length?'the owner’s memory':'no other components')+'. Name is an editor label; context label controls the text supplied to the entity. Save to rebuild the preview. Runtime edits are separate.';content.append(description);
      field(content,'Component name',item.name,v=>ProjectComponentOperations.rename(draft,item.id,v),'world-name');
      for(const [key,value] of Object.entries(item.params))field(content,key,value,v=>ProjectComponentOperations.configure(draft,item.id,key,v),'component-param-'+key,{label:({state:'Context text',pre_act_label:'Context label',history_length:'Recent observations (1–1000)'})[key] || key});
      const owner=draft.instances.find(x=>x.id===item.instance);
      button('Owner: '+owner.params.name,()=>choose(owner.id),content);
      const targets=draft.instances.filter(x=>x.id!==owner.id && spec.prototypes.includes(x.prototype));
      if(targets.length){
        const label=document.createElement('label');label.className='editor-field';label.textContent='Move component to';
        const picker=document.createElement('select');picker.id='component-owner';picker.dataset.definitionField='true';
        for(const target of targets){const option=document.createElement('option');option.value=target.id;option.textContent=target.params.name;picker.append(option);}
        label.append(picker);content.append(label);
        const hint=document.createElement('p');hint.textContent='Keeps the component ID and settings; places it last among the new owner’s authored components. Undo restores its previous owner and order.';content.append(hint);
        const move=button('Move component',()=>structural(next=>ProjectComponentOperations.moveOwner(next,draftDefinition.component_catalog,item.id,picker.value)),content);move.dataset.authorAction='true';
      }
      controls();return;
    }
  }
  function inspect(){
    if(!draft) return;
    const world=ProjectRecordOperations.locate(draft,selectedId);
    if(world){inspectWorld(world);return;}
    const index=draft.instances.findIndex(x=>x.id===selectedId);
    const content=$('inspector-content');content.style.display='block';$('inspector-empty').style.display='none';
    if(index<0){
      $('inspector-title').textContent='Simulation';$('inspector-subtitle').textContent='Initial definition';content.replaceChildren();
      field(content,'Initial premise',draft.premise,v=>draft.premise=v,'editor-premise');
      field(content,'Simulation maximum steps (1–1000)',draft.max_steps,v=>draft.max_steps=v,'editor-max-steps');
      controls();return;
    }
    const previewIndex=inspectionDocument().instances.findIndex(x=>x.id===selectedId);
    const sameEntity=selectedEntity==='entity_'+previewIndex;
    const expanded=sameEntity ? [...content.querySelectorAll('.component-state.expanded')].map(x=>x.id) : [];
    const focused=sameEntity && content.contains(document.activeElement) ? document.activeElement : null;
    const focusState=focused ? {id:focused.id,start:focused.selectionStart,end:focused.selectionEnd} : null;
    selectedEntity='entity_'+previewIndex;
    if(previewIndex>=0) updateInspector(selectedEntity);
    else {
      content.replaceChildren();
      notice('Save draft to build the component preview. No simulation is executed.','info','preview:'+selectedId+':'+draftRevision);
    }
    if(!runtimeMode) $('inspector-title').textContent=draft.instances[index].params.name;
    const item=draft.instances[index];
    $('inspector-subtitle').textContent=(runtimeMode?'Current runtime':'Initial definition')+' · '+item.prefab+' · '+item.id;
    if(!runtimeMode){
      const fields=document.createElement('section');fields.setAttribute('aria-label','Editable initial fields');
      const entry=catalog().find(x=>x.instance.prototype===item.prototype);
      if(entry?.fixed_parameters?.length)notice('Application-owned prefab parameters: '+entry.fixed_parameters.join(', ')+'. Configure these in the Python preset.','info','fixed:'+item.prototype);
      if(entry?.supports_extra_components===false)notice('This prefab does not implement extra_components yet. Its built-in components remain available.','info','extras:'+item.prototype);
      const specs=structuredClone(entry?.inspector || draftDefinition.inspector[item.id] || {});
      for(const [field,role] of Object.entries(entry?.references || {})) {
        specs[field]={...specs[field],choices:draft.instances.filter(x=>x.role===role).map(x=>({value:x.id,label:x.params.name})),refresh:true};
      }
      for(const [key,value] of Object.entries(item.params)) field(fields,key,value,v=>item.params[key]=v,'editor-'+item.id+'-'+key,specs[key]);
      const overrides=draft.dynamic_states?.[item.id] || {};
      if(Object.keys(overrides).length){
        const section=document.createElement('section');section.setAttribute('aria-label','Initial component overrides');
        const title=document.createElement('h3');title.textContent='Initial component overrides';section.append(title);
        for(const [component,values] of Object.entries(overrides))for(const key of Object.keys(values)){
          const row=document.createElement('div');row.textContent=component+' · '+key+' ';
          const reset=button('Reset to prefab',()=>structural(next=>ProjectDraftCommands.apply(next,draftDefinition,item.id,'state-reset',[item.id,component,key])),row);reset.setAttribute('aria-label','Reset '+component+' '+key+' to prefab');reset.dataset.authorAction='true';section.append(row);
        }
        fields.append(section);
      }
      content.prepend(fields);
      const entries=(draftDefinition.component_catalog || []).filter(x=>x.prototypes.includes(item.prototype));
      if(entries.length){
        const section=document.createElement('section'), heading=document.createElement('h3');heading.textContent='Component catalogue';section.append(heading);
        const picker=document.createElement('select');picker.id='component-type';picker.setAttribute('aria-label','Registered component type');picker.dataset.definitionField='true';
        for(const entry of entries){const option=document.createElement('option');option.value=entry.key;option.textContent=entry.key;picker.append(option);}
        const description=document.createElement('p');
        const describe=()=>{const spec=entries.find(x=>x.key===picker.value);description.textContent=spec.description+' Dependencies: '+(spec.dependencies.join(', ') || 'none');};
        picker.onchange=describe;describe();section.append(picker,description);
        const addComponent=button('Add component',()=>structural(next=>ProjectComponentOperations.add(next,entries,item.id,picker.value,crypto.randomUUID())),section);addComponent.dataset.authorAction='true';
        content.prepend(section);
      }
      inspectReferences(content,selectedId);
      content.querySelectorAll('.dynamic-save-btn').forEach(b=>{
        const values=draft.dynamic_states?.[selectedId]?.[b.dataset.component];
        if(values && Object.hasOwn(values,b.dataset.stateKey)){
          const value=values[b.dataset.stateKey];$(b.dataset.inputId).value=typeof value==='string'?value:JSON.stringify(value);$('json_'+b.dataset.inputId).checked=typeof value!=='string';
        }
      });
    } else {
      content.querySelectorAll('.dynamic-input').forEach(input=>{
        const key=envelope.references.run_id+':'+selectedId+':'+input.id;
        const format=$('json_'+input.id);
        if(runtimeDrafts.has(key)){input.value=runtimeDrafts.get(key).value;format.checked=runtimeDrafts.get(key).json;}
        const capture=()=>{runtimeDrafts.set(key,{value:input.value,json:format.checked,envelope:runtimeDrafts.get(key)?.envelope || structuredClone(envelope)});};
        input.oninput=capture;format.addEventListener('change',capture);
      });
    }
    for(const id of expanded){const el=$(id);if(el){el.classList.add('expanded');if($('toggle_'+id))$('toggle_'+id).textContent='▼';}}
    controls();
    if(focusState){const input=$(focusState.id);if(input && !input.disabled){input.focus({preventScroll:true});if(input.setSelectionRange && focusState.start!==null)input.setSelectionRange(focusState.start,focusState.end);}}
  }
  function choose(id,component){history.endGroup();selectedId=id;renderHierarchy();inspect();tab('inspector');
    if(component){const el=$('comp_'+component.replace(/[^a-zA-Z0-9]/g,'_'));if(el){el.classList.add('expanded');el.scrollIntoView({block:'nearest'});}}
  }
  function renderHierarchy(){
    if(!draft) return;
    hierarchy.replaceChildren();button('Simulation settings',()=>{runtimeMode=false;mode.value='definition';selectedId='simulation';renderedView='';render();tab('inspector');},hierarchy);
    const query=search.value.trim().toLocaleLowerCase();let matches=0;
    for(const role of ['entity','game_master','initializer']){
      const rows=[];
      for(const item of draft.instances.filter(x=>x.role===role)) {
        const idx=inspectionDocument().instances.findIndex(x=>x.id===item.id);
        const components=Object.keys(entityData['entity_'+idx]?.component_info?.context_components || {});
        if(!ProjectDraftCommands.matches(item,query,components,(draft.components || []).filter(c=>c.instance===item.id)))continue;
        rows.push({item,components});
      }
      if(!rows.length)continue;
      const title=document.createElement('h3');title.textContent={entity:'Player entities',game_master:'Game master entities',initializer:'Initializers'}[role];hierarchy.append(title);
      for(const {item,components} of rows){
        matches++;
        const b=button(item.params.name,()=>choose(item.id),hierarchy);b.dataset.instanceId=item.id;
        b.setAttribute('aria-pressed',String(item.id===selectedId));
        for(const component of (draft.components || []).filter(x=>x.instance===item.id)) {
          const c=button(component.name+' · '+component.type,()=>choose('components:'+component.id),hierarchy);c.dataset.componentId=component.id;c.classList.add('editor-component');
        }
        for(const component of components.filter(x=>!x.startsWith('authored_'))) {
          const c=button(component,()=>choose(item.id,component),hierarchy);c.classList.add('editor-component');
        }
      }
    }
    if(!matches){notice('No matching instances or components.','info','search:'+query+':'+draftRevision);}
  }
  function render(){
    if(!draft) return;
    const s=state(), view=runtimeMode?s.runtime:draftDefinition;
    const signature=JSON.stringify([runtimeMode,view]);
    if(signature!==renderedView){
      renderedView=signature;
      for(const key of Object.keys(entityData)) delete entityData[key];
      Object.assign(entityData,view?.entities || draftDefinition.entities);
      // SVG comes only from the standard escaped server renderer, never imported HTML.
      document.querySelector('.svg-container').innerHTML=view?.svg || draftDefinition.svg;
      const graph=document.querySelector('.svg-container > svg');
      if(graph?.viewBox.baseVal.width)graph.style.setProperty('--editor-graph-width',graph.viewBox.baseVal.width/16+'rem');
      const chosen=prototypePicker.value;prototypePicker.replaceChildren();
      for(const entry of catalog()) {
        const option=document.createElement('option');option.value=entry.key || entry.instance.prototype;
        const role=entry.instance.role==='entity'?'player entity':entry.instance.role==='game_master'?'game master entity':entry.instance.role;
        option.textContent=(entry.key || entry.instance.prototype)+' · '+role+' · '+(entry.kind==='prefab'?'prefab':entry.instance.prefab+' preset');
        option.title=entry.description;prototypePicker.append(option);
      }
      if(catalog().some(x=>(x.key || x.instance.prototype)===chosen)) prototypePicker.value=chosen;
      renderHierarchy();
      inspect();
    }
    for(const name of s.viewers || [])offerViewer(name);
    renderSimulationLog(envelope);
    for(const issue of draftDefinition.prefab_diagnostics || [])notice('Prefab '+issue.module+' unavailable: install optional package '+issue.dependency+'.','error','prefab:'+issue.module+':'+issue.dependency);
    renderEntityActions(s);
    const limitMessage=`Saved simulation maximum: ${s.document.max_steps} engine steps. The game master may end earlier.`;
    notice(limitMessage,'info','limits:'+limitMessage);
    limit.textContent=s.run.requested_steps!==undefined?`Target ${s.run.requested_steps} / Maximum ${s.run.maximum_steps}`:`Maximum ${s.document.max_steps}`;
    const engine=(runtimeMode?s.runtime?.engine:draftDefinition?.engine);
    engineLabel.textContent='Engine: '+(engine?engine.module+'.'+engine.name:'not reported by preview');
    const latest=s.steps.at(-1);summary.textContent=latest?`Step ${latest.step} · ${latest.acting_entity}`:`Step ${s.current_step}`;
    controls();
  }
  document.addEventListener('click',event=>{
    const card=event.target.closest('.entity-card');if(!card || !draft)return;
    event.stopImmediatePropagation();const index=Number(card.dataset.entityId.split('_')[1]);
    const id=inspectionDocument().instances[index]?.id;
    if(draft.instances.some(x=>x.id===id))choose(id);
  },true);
  saveComponentState=async (_entity,component,field,inputId)=>{
    let value;
    try{value=$('json_'+inputId)?.checked?JSON.parse($(inputId).value):$(inputId).value;}catch(error){notice('Invalid JSON field value: '+error.message,'error');return;}
    if(!runtimeMode){structural(next=>ProjectDraftCommands.apply(next,{...draftDefinition,document:inspectionDocument()},selectedId,'state-field',[selectedId,component,field,JSON.stringify(value)]));return;}
    const key=envelope.references.run_id+':'+selectedId+':'+inputId, saved=runtimeDrafts.get(key);
    if(await dispatch('runtime.edit_state',{instance_id:selectedId,component,field,value:JSON.stringify(value)},saved?.envelope)){
      runtimeDrafts.delete(key);renderedView='';await refresh();
    }
  };
  function renderEntityActions(s){
    const actions=new Map();
    if(runtimeMode) for(const entry of s.steps) {
      // StepData.action is the acting entity's action, not the GM resolution.
      if(entry.acting_entity) actions.set(entry.acting_entity,entry.action);
    }
    for(const card of document.querySelectorAll('.entity-card')) {
      const data=entityData[card.dataset.entityId];
      const action=card.querySelector('foreignObject div');
      if(!data || !action) continue;
      if(!runtimeMode) action.textContent='Initial definition · actions appear in Current runtime.';
      else if(actions.has(data.name)) action.textContent=actions.get(data.name) || 'Empty action reported.';
      else action.textContent=s.run.status==='active' ? 'No action recorded yet.' : 'No action recorded in this run.';
    }
  }
  function renderSimulationLog(next){
    const s=next.result;
    const runKey=JSON.stringify([next.references.session_id,next.references.run_id]);
    if(loggedRun!==runKey){
      const hadRun=loggedRun!==undefined;loggedRun=runKey;loggedSteps=0;loggedFailure=null;loggedCompletion=null;
      if(hadRun) notice('New run attached.','info',runKey+':start');
    }
    for(const entry of s.steps.slice(loggedSteps)) notice(`Step ${entry.step} · Player entity action · ${entry.acting_entity || 'No player'}\n${entry.action || 'Empty action reported.'}`,'info',runKey+':step:'+entry.step);
    loggedSteps=s.steps.length;
    if(['completed','stopped'].includes(s.run.status) && loggedCompletion!==s.run.status){
      notice(`${s.run.status==='completed'?'Completed':'Stopped'} at step ${s.current_step}: ${s.run.message || 'Runner returned.'}`,'info',runKey+':complete:'+s.run.status);
      loggedCompletion=s.run.status;
    }
    const failure=s.run.message || 'Unknown runner error.';
    if(s.run.status==='failed' && loggedFailure!==failure){
      if(notice(`Run failed at step ${s.current_step}: ${failure}`,'error',runKey+':failed:'+failure))tab('log');
      loggedFailure=failure;
    }
  }
  function receive(next){
    if(envelope && next.references.session_id===envelope.references.session_id && next.revision<envelope.revision)return;
    const newSession=envelope && next.references.session_id!==envelope.references.session_id;
    envelope=next;connection(true);
    if(newSession && dirty){
      draftRevision=-1;renderedView='';
      report(Error('The server session changed. Unsaved fields are kept; review before Reload saved.'));
    } else if(!draft || newSession)adopt();
    else if(draftRevision!==state().revision){
      if(dirty && JSON.stringify(draft)!==JSON.stringify(state().document))report(Error('Definition changed in another tab. Your fields are kept. Review before Reload saved.'));
      else adopt(sending || dirty);
    }
    render();
  }
  async function refresh(){
    try {const response=await fetch('/api/state');if(!response.ok)throw Error('Cannot read editor state.');receive(await response.json());}
    catch(e){connection(false);controls();report(e);}
  }
  async function query(operation,args){
    if(!connected) throw Error('Disconnected; command was not sent.');
    const response=await fetch('/api/dispatch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({operation,arguments:args})});
    const result=await response.json();
    if(!response.ok) throw Error(result.error.message);
    return result.result;
  }
  function download(content,name,type='application/json'){
    const url=URL.createObjectURL(new Blob([content],{type})), link=document.createElement('a');
    link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    notice('Download prepared: '+name,'success');
  }
  let importedLog='';
  const logFile=document.createElement('input');logFile.type='file';logFile.accept='.json,application/json';logFile.hidden=true;toolbar.append(logFile);
  logFile.onchange=async()=>{try{
    if(!logFile.files.length)return;
    if(logFile.files[0].size>900000)throw Error('Maximum imported log size is 900 kB; use concordia-log locally for larger files.');
    const text=await logFile.files[0].text();
    await query('log.query',{command:'overview',source:'imported',arguments:'[]',imported:text});
    importedLog=text;notice('Imported structured log selected. Use --source imported.','success');
  }catch(e){report(e);}finally{logFile.value='';}};
  async function executePlan(plan){
    if(plan.kind==='text'){notice(plan.text);return;}
    if(plan.kind==='call'){
      if(dirty || runtimeDrafts.size)throw Error('Unsaved client fields: save or explicitly reload before call.');
      await dispatch(plan.operation,plan.arguments,undefined,true);return;
    }
    if(plan.action==='discover'){
      const response=await fetch('/api/operations'), result=await response.json();
      if(!response.ok)throw Error(result.error.message);
      notice(JSON.stringify(result.result,null,2));return;
    }
    if(plan.action==='watch'){notice(connected?'Live stream already enabled; updates appear in Simulation log.':'Live stream reconnecting; no commands are replayed.');return;}
    if(plan.kind==='log'){
      if(plan.source==='imported' && !importedLog)throw Error('Use log import to choose a structured log first.');
      const result=await query('log.query',{command:plan.command,source:plan.source,arguments:JSON.stringify(plan.args),imported:plan.source==='imported'?importedLog:''});
      notice('Log source: '+result.source+'\n'+result.text);
      if(result.download)download(result.download.content,result.download.name,result.download.type);
      return;
    }
    const action=plan.action,args=plan.args.map(x=>x==='.'?selectedId:x);
    if(['catalog','list','locate','get','params'].includes(action)){const result=ProjectDraftCommands.read(draft,draftDefinition,inspectionDocument(),state().runtime,mode.value,selectedId,action,args);if(action==='locate'){runtimeMode=false;mode.value='definition';choose(result.selection);if(result.field)$(result.field)?.focus();}notice(JSON.stringify(result,null,2));return;}
    if(action==='viewer'){if(args.length)await showViewer(args[0]);else notice(JSON.stringify({selected:currentViewer,available:[...viewerSelect.options].map(x=>x.value)}));return;}
    if(action==='viewer-load'){viewerFile.click();return;}
    if(action==='viewer-url'){await openViewerURL(args[0]);return;}
    if(action==='viewer-refresh'){await showViewer(args[0] || currentViewer,true);return;}
    if(action==='layout'){notice(JSON.stringify(editorLayout(args),null,2));return;}
    if(plan.operation){
      if(action==='edit' && runtimeDrafts.size)throw Error('Save pending runtime fields before a typed edit. No runtime draft was discarded.');
      if(action==='run' && dirty)throw Error('Unsaved draft: save before run. No changes were discarded.');
      if(action==='run' && args[0]!==null && state().run_limits)runSteps.value=String(args[0]);
      if(action==='run' && args[0]===null && state().run_limits)plan.arguments.requested_steps=Number(runSteps.value);
      await dispatch(plan.operation,plan.arguments);
      return;
    }
    if(action==='state'){notice(JSON.stringify({...state(),client:{dirty,selectedId,view:mode.value}},null,2));return;}
    if(action==='log-import'){logFile.click();return;}
    if(action==='view'){mode.value=args[0];mode.onchange();notice('View: '+args[0]);return;}
    if(action==='panel'){tab(args[0]);notice('Panel: '+args[0]);return;}
    if(action==='search'){search.value=args[0];search.oninput();notice('Hierarchy search: '+args[0]);return;}
    if(action==='select' || action==='inspect'){
      const id=args[0] || selectedId;
      if(id!=='simulation' && !draft.instances.some(x=>x.id===id) && !ProjectRecordOperations.locate(draft,id))throw Error('Unknown selection.');
      if(id==='simulation'){runtimeMode=false;mode.value='definition';}
      if(args[1])ProjectDraftCommands.read(draft,draftDefinition,inspectionDocument(),state().runtime,mode.value,selectedId,'inspect',args);
      choose(id,args[1]);if(action==='select')notice('Selected '+id+(args[1]?' · '+args[1]:''));
      if(action==='inspect' && runtimeMode && !state().runtime)throw Error('No current runtime to inspect.');
      if(action==='inspect')notice(JSON.stringify(ProjectDraftCommands.read(draft,draftDefinition,inspectionDocument(),state().runtime,mode.value,selectedId,action,args),null,2));
      return;
    }
    if(action==='references'){notice(JSON.stringify(ProjectReferences.uses(draft,catalog(),args[0]),null,2));return;}
    if(action==='export'){download(JSON.stringify(draft,null,2)+'\n','concordia-project.json');return;}
    if(action==='validate'){
      try{await query('project.validate',{text:JSON.stringify(draft)});notice('Draft is valid. No save performed.','success');}
      catch(e){e.validationTarget=ProjectValidation.target(draft,e.message);throw e;}return;
    }
    if(!connected || sending || state().run.status==='active')throw Error('Authoring is unavailable while disconnected, busy or running.');
    if(action==='save'){await save.onclick();return;}
    if(action==='load'){if(dirty)throw Error('Unsaved draft: save/export it or reload --discard before load.');file.click();return;}
    if(action==='reload'){adopt();notice('Reloaded saved definition; local draft and undo history discarded.');return;}
    if(runtimeMode)throw Error('Use view definition before changing an authored draft.');
    if(action==='undo' || action==='redo'){
      if(!(action==='undo'?history.past:history.future).length)throw Error('No '+action+' available.');
      restoreHistory(action);return;
    }
    if(!structural(next=>ProjectDraftCommands.apply(next,draftDefinition,selectedId,action,args,plan.id),true))notice('No draft change.');
  }
  const commandForm=document.createElement('form');commandForm.id='editor-command-form';
  const commandInput=document.createElement('input');commandInput.id='editor-command';commandInput.type='text';commandInput.autocomplete='off';
  commandInput.setAttribute('aria-label','Simulation log command');commandInput.placeholder='Command (help lists commands)';commandInput.enterKeyHint='send';
  const commandSubmit=document.createElement('button');commandSubmit.className='editor-button';commandSubmit.type='submit';commandSubmit.textContent='Send';
  commandForm.setAttribute('aria-live','off');
  const commandPrompt=document.createElement('span');commandPrompt.id='editor-command-prompt';commandPrompt.textContent='>';commandPrompt.setAttribute('aria-hidden','true');
  commandForm.append(commandPrompt,commandInput,commandSubmit);document.querySelector('.console-output').append(commandForm);
  button('Command help',()=>submitCommand('help'),commandForm).type='button';
  const commandHistory=[];let historyPosition=0, commandBusy=false, pendingInput='';
  commandInput.onkeydown=event=>{
    if(event.isComposing){if(event.key==='Enter')event.preventDefault();return;}
    if(event.key==='ArrowUp' || event.key==='ArrowDown'){
      event.preventDefault();if(historyPosition===commandHistory.length)pendingInput=commandInput.value;
      historyPosition=Math.max(0,Math.min(commandHistory.length,historyPosition+(event.key==='ArrowUp'?-1:1)));
      commandInput.value=historyPosition===commandHistory.length?pendingInput:commandHistory[historyPosition];
    }
  };
  async function submitCommand(line){
    if(commandBusy){report(Error('A command is already pending; not queued.'));return;}
    if(!line.trim())return;
    notice('> '+line);commandHistory.push(line);historyPosition=commandHistory.length;pendingInput='';
    commandBusy=true;commandSubmit.disabled=true;
    try{const plan=line.trim()==='help'?{kind:'text',text:SESSION_HELP}:await query('session.plan',{line});await executePlan(plan);if(!['select','view','panel','inspect'].includes(plan.action))tab('log');}
    catch(e){report(e,'command');}
    finally{commandBusy=false;commandSubmit.disabled=false;}
  }
  commandForm.onsubmit=event=>{event.preventDefault();if(event.isComposing)return;const line=commandInput.value;commandInput.value='';submitCommand(line);};
  async function dispatch(operation,args={},source,showResult=false){
    if(!connected || sending)throw Error('Disconnected or busy; command was not sent.');
    sending=true;controls();error.textContent='';
    const from=source || envelope;
    try{
      const response=await fetch('/api/dispatch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({operation,arguments:args,revision:from.revision,references:from.references,retry_key:crypto.randomUUID()})});
      const result=await response.json();
      if(!response.ok){
        const failure=Error(result.error.message);
        failure.validationTarget=ProjectValidation.forSave(draft,operation,args,result.error);
        throw failure;
      }
      if(operation==='project.run'){runtimeMode=true;mode.value='runtime';tab('log');const output=$('console-output');output.scrollTop=output.scrollHeight;}
      notice(operation==='project.run'?'Run accepted; waiting for runner output.':showResult?JSON.stringify(result.result,null,2):operation+' completed.','success');await refresh();return true;
    }catch(e){report(e,operation);await refresh();return false;}
    finally{sending=false;controls();}
  }

  let editorLayout;
  function installEditorLayout(){
    const key='concordia.editor.layout.v1', narrow=matchMedia('(max-width:700px)');
    const sizes={left:.22,right:.28,terminal:.25};
    try{
      const saved=JSON.parse(localStorage.getItem(key));
      for(const name of Object.keys(sizes))if(Number.isFinite(saved?.[name]) && saved[name]>0 && saved[name]<1)sizes[name]=saved[name];
    }catch(_){} // Storage may be unavailable; sizing still works in this tab.
    const handles={};let drag=null;
    const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
    function bounds(){
      const width=layout.clientWidth-16;
      const height=Math.max(0,layout.clientHeight-document.querySelector('.header').getBoundingClientRect().height-8);
      return {width,height,leftMin:140,rightMin:180,centerMin:160,terminalMin:Math.min(100,height),terminalMax:Math.max(0,height-64)};
    }
    function apply(){
      if(narrow.matches)return;
      const b=bounds();
      const left=clamp(sizes.left*b.width,b.leftMin,b.width-b.rightMin-b.centerMin);
      const right=clamp(sizes.right*b.width,b.rightMin,b.width-left-b.centerMin);
      const terminal=clamp(sizes.terminal*b.height,Math.min(b.terminalMin,b.terminalMax),b.terminalMax);
      for(const [name,value,min,max] of [
        ['left',left,b.leftMin,b.width-right-b.centerMin],
        ['right',right,b.rightMin,b.width-left-b.centerMin],
        ['terminal',terminal,Math.min(b.terminalMin,b.terminalMax),b.terminalMax]]){
        layout.style.setProperty('--editor-'+name,value+'px');
        const handle=handles[name];
        handle.setAttribute('aria-valuenow',Math.round(value));handle.setAttribute('aria-valuemin',Math.round(min));handle.setAttribute('aria-valuemax',Math.round(max));
        handle.setAttribute('aria-valuetext',Math.round(value)+' pixels');
      }
    }
    function store(){try{localStorage.setItem(key,JSON.stringify(sizes));}catch(_){}}
    function change(name,value){
      const b=bounds(),handle=handles[name];
      const bounded=clamp(value,Number(handle.getAttribute('aria-valuemin')),Number(handle.getAttribute('aria-valuemax')));
      sizes[name]=bounded/(name==='terminal'?b.height:b.width || 1);apply();
    }
    for(const [name,label,orientation,column,selector] of [
      ['left','Hierarchy width','vertical','2','.left-sidebar'],
      ['right','Inspector width','vertical','4','.right-sidebar'],
      ['terminal','Simulation log height','horizontal',null,'.bottom-panel']]){
      const handle=document.createElement('div');handle.className='editor-splitter';handle.tabIndex=0;
      handle.setAttribute('role','separator');handle.setAttribute('aria-label',label);handle.setAttribute('aria-orientation',orientation);
      handle.title=label+' — drag or use arrow keys, Home and End';
      const panel=document.querySelector(selector);if(!panel.id)panel.id='editor-'+name+'-panel';handle.setAttribute('aria-controls',panel.id);
      if(column)handle.style.gridColumn=column;layout.append(handle);handles[name]=handle;
      handle.onpointerdown=e=>{
        if(e.button!==0 || narrow.matches)return;
        e.preventDefault();handle.focus();handle.setPointerCapture(e.pointerId);
        drag={name,id:e.pointerId,start:name==='terminal'?e.clientY:e.clientX,value:Number(handle.getAttribute('aria-valuenow'))};
      };
      handle.onpointermove=e=>{
        if(!drag || drag.name!==name || drag.id!==e.pointerId)return;
        const delta=(name==='terminal'?e.clientY:e.clientX)-drag.start;
        change(name,drag.value+(name==='left'?delta:-delta));
      };
      const finish=()=>{if(drag?.name===name){drag=null;store();}};
      handle.onpointerup=finish;handle.onpointercancel=finish;handle.onlostpointercapture=finish;
      handle.onkeydown=e=>{
        const negative=name==='terminal'?'ArrowDown':name==='right'?'ArrowRight':'ArrowLeft';
        const positive=name==='terminal'?'ArrowUp':name==='right'?'ArrowLeft':'ArrowRight';
        if(![negative,positive,'Home','End'].includes(e.key))return;
        e.preventDefault();
        const value=e.key==='Home'?Number(handle.getAttribute('aria-valuemin')):e.key==='End'?Number(handle.getAttribute('aria-valuemax')):Number(handle.getAttribute('aria-valuenow'))+(e.key===positive?1:-1)*(e.shiftKey?40:10);
        change(name,value);store();
      };
    }
    const observer=new ResizeObserver(apply);observer.observe(layout);observer.observe(document.querySelector('.header'));
    narrow.addEventListener('change',apply);apply();
    editorLayout=args=>{if(args.length){if(narrow.matches)throw Error('Use panel tabs on narrow screens; resizing is unavailable.');change(args[0],Number(args[1]));store();}return {narrow:narrow.matches,panels:Object.fromEntries(Object.entries(handles).map(([name,h])=>[name,{pixels:Number(h.getAttribute('aria-valuenow')),min:Number(h.getAttribute('aria-valuemin')),max:Number(h.getAttribute('aria-valuemax'))}]))};};
  }
  let events, timer, polling=false;
  function connect(){
    events=new EventSource('/api/events');
    events.onmessage=e=>receive(JSON.parse(e.data));
    events.onerror=()=>{connection(false);controls();};
    // Boundary acknowledgement changes without a completed-step event.
    timer=setInterval(async()=>{if(!polling){polling=true;try{await refresh();}finally{polling=false;}}},1000);
    refresh();
  }
  window.addEventListener('pagehide',()=>{clearInterval(timer);events.close();});
  window.addEventListener('pageshow',event=>{if(event.persisted)connect();});
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh();});
  connect();
  installEditorLayout();
  window.addEventListener('beforeunload',e=>{if(dirty || runtimeDrafts.size){e.preventDefault();e.returnValue='';}});
  refresh();controls();
})();
</script>
"""
)
