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
#project-error {color:#ffb4b4; white-space:pre-wrap; overflow-wrap:anywhere;}
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
  const error = document.createElement('p'); error.id = 'project-error';
  error.setAttribute('role', 'alert');
  const fields = document.createElement('div');
  const hierarchy = document.createElement('div'); hierarchy.id = 'project-hierarchy';
  sidebar.append(status, error, fields, hierarchy);
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
  function report(e) { error.textContent = e.message; }
  function controls() {
    const active = state && state.run.status === 'active';
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
      (state.run.message ? '\n' + state.run.message : '') +
      '\nSave downloads initial configuration, not a runtime checkpoint.'
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
    field(fields, 'Maximum steps (1–1000)', draft.max_steps, v => draft.max_steps = v, 'project-max-steps');
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


# Integrated mode preserves the legacy project surface for existing callers.
EDITOR_STYLE = """
<style>
body {height:100dvh;}
.layout {height:100dvh; grid-template-rows:auto minmax(0,1fr) 180px;}
.header {display:block;padding:8px;}
.header > :not(#editor-toolbar) {display:none;}
#editor-toolbar button, #editor-tabs button {min-height:44px;min-width:44px;}
#editor-toolbar {display:flex;flex-wrap:wrap;gap:4px;align-items:center;}
.editor-button {font:inherit;color:inherit;background:#383838;border:1px solid #777;
 border-radius:5px;padding:8px;cursor:pointer;}
.editor-button:disabled {opacity:.5;cursor:default;}
#editor-heading {flex-basis:100%;font-size:14px;}
#editor-status,#editor-error {flex-basis:100%;margin:2px;white-space:pre-wrap;overflow-wrap:anywhere;}
#editor-error {color:#ffb4b4;}
#editor-tabs {display:flex;flex-basis:100%;gap:4px;}
#editor-tabs button[aria-selected=true] {border-color:#90c8ff;background:#17466c;}
.left-sidebar,.center-panel,.right-sidebar,.bottom-panel {min-width:0;min-height:0;}
.left-sidebar button {display:block;min-height:44px;width:100%;text-align:left;margin:4px 0;}
.editor-component {padding-left:24px;font-size:13px;}
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
 .header {flex:none;}
 .left-sidebar,.right-sidebar,.center-panel,.bottom-panel {
   display:none;flex:1;overflow:auto; padding:12px;}
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
class ProjectDraftOperations {
  static add(document, catalog, prototype, id, sourceId=null) {
    const entry=catalog.find(x=>x.instance.prototype===prototype);
    if(!entry) throw Error('Choose a registered prefab prototype.');
    if(document.instances.length>=100) throw Error('A project supports at most 100 instances.');
    if(document.instances.some(x=>x.id===id)) throw Error('Instance ID already exists.');
    const source=sourceId ? document.instances.find(x=>x.id===sourceId) : entry.instance;
    if(!source || source.prototype!==prototype) throw Error('Unknown source instance.');
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
    document.instances.push(item);return id;
  }
  static remove(document,catalog,id) {
    const item=document.instances.find(x=>x.id===id);
    if(!item) throw Error('Select an instance.');
    if(['entity','game_master'].includes(item.role) && document.instances.filter(x=>x.role===item.role).length===1)
      throw Error('Keep at least one actor and one game master.');
    for(const other of document.instances.filter(x=>x.id!==id)) {
      const entry=catalog.find(x=>x.instance.prototype===other.prototype);
      for(const field of Object.keys(entry?.references || {}))
        if(other.params[field]===id) throw Error(other.params.name+' references this instance through '+field+'. Change that reference first.');
    }
    document.instances=document.instances.filter(x=>x.id!==id);
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
"""

EDITOR_SCRIPT = '<script>\n' + DRAFT_HISTORY_SCRIPT + r"""
(() => {
  const $ = id => document.getElementById(id);
  const layout = document.querySelector('.layout');
  const toolbar = document.createElement('div'); toolbar.id = 'editor-toolbar';
  document.querySelector('.header').append(toolbar);
  const heading=document.createElement('strong'); heading.id='editor-heading';
  heading.textContent=document.querySelector('.header h1').textContent; toolbar.append(heading);
  let envelope, draft, draftRevision, selectedId, connected = false, sending = false;
  let dirty = false, runtimeMode = false, renderedView = '', loggedRun, loggedSteps = 0;
  const runtimeDrafts = new Map();
  const history = new ProjectDraftHistory();
  const draftSnapshot = () => ({document:draft, selectedId});
  function button(label, action, parent=toolbar) {
    const b = document.createElement('button'); b.textContent=label;
    b.className='editor-button'; b.onclick=action; parent.append(b); return b;
  }
  const run = button('Run', () => dispatch(state().state === 'paused'
    ? 'runtime.play' : 'project.run', state().state === 'paused' ? {} : {revision:draftRevision}));
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
  prototypePicker.setAttribute('aria-label','Registered prefab prototype');toolbar.append(prototypePicker);
  const add=button('Add instance',()=>structural(next=>ProjectDraftOperations.add(next,catalog(),prototypePicker.value,crypto.randomUUID())));
  const duplicate=button('Duplicate',()=>structural(next=>{
    const source=next.instances.find(x=>x.id===selectedId);
    if(!source) throw Error('Select an instance to duplicate.');
    return ProjectDraftOperations.add(next,catalog(),source.prototype,crypto.randomUUID(),source.id);
  }));
  const remove=button('Remove',()=>structural(next=>ProjectDraftOperations.remove(next,catalog(),selectedId)));
  const up=button('Move earlier',()=>structural(next=>{ProjectDraftOperations.move(next,selectedId,-1);return selectedId;}));
  const down=button('Move later',()=>structural(next=>{ProjectDraftOperations.move(next,selectedId,1);return selectedId;}));
  const exportButton = button('Export JSON', () => {
    const blob = new Blob([JSON.stringify(state().document,null,2)+'\n'], {type:'application/json'});
    const url=URL.createObjectURL(blob), link=document.createElement('a');
    link.href=url; link.download='concordia-project.json'; link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  const file=document.createElement('input'); file.id='editor-file';
  file.type='file'; file.accept='.json,application/json'; file.hidden=true; toolbar.append(file);
  const open=button('Open JSON',()=>file.click());
  file.onchange=async()=>{
    try {
      if (!file.files.length) return;
      if(file.files[0].size>900000) throw Error('Maximum project size is 900 kB.');
      if(await dispatch('project.save',{text:await file.files[0].text(),revision:draftRevision})) {
        dirty=false; error.textContent=''; await refresh(); adopt();
      }
    } catch(e) {report(e);} finally {file.value='';}
  };
  button('Reload saved',()=>{dirty=false;runtimeDrafts.clear();adopt();});
  const mode=document.createElement('select'); mode.id='editor-mode'; mode.setAttribute('aria-label','Inspector source');
  for(const [value,label] of [['definition','Initial definition'],['runtime','Current runtime']]) {
    const option=document.createElement('option'); option.value=value; option.textContent=label; mode.append(option);
  }
  toolbar.append(mode);
  mode.onchange=()=>{runtimeMode=mode.value==='runtime';renderedView='';render();};
  const status=document.createElement('p');status.id='editor-status';status.setAttribute('role','status');
  const error=document.createElement('p');error.id='editor-error';error.setAttribute('role','alert');
  const tabs=document.createElement('nav');tabs.id='editor-tabs';tabs.setAttribute('aria-label','Editor panels');
  toolbar.append(status,error,tabs);
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
  search.placeholder='Find actors, GMs or components';search.setAttribute('aria-label','Search hierarchy');
  toolbar.append(search);search.oninput=()=>{renderHierarchy();controls();};
  const summary=document.createElement('div');summary.id='editor-step-summary';
  document.querySelector('.center-panel').prepend(summary);
  document.querySelector('.console-header').textContent='Simulation log';
  function state(){return envelope?.result;}
  function report(e){error.textContent=e.message || String(e);}
  function catalog(){return state()?.definition.catalog || [];}
  function structural(change){
    if(runtimeMode || !connected || sending || state()?.run.status==='active') return;
    try {
      const next=structuredClone(draft), selection=change(next);
      if(JSON.stringify(next)===JSON.stringify(draft)) return;
      history.record(draftSnapshot());draft=next;selectedId=selection;
      dirty=JSON.stringify(draft)!==JSON.stringify(state().document);
      error.textContent='';search.value='';renderedView='';render();tab('inspector');
    } catch(e) {report(e);}
  }
  function restoreHistory(direction){
    if(runtimeMode || !connected || sending || state()?.run.status==='active') return;
    const restored=history[direction](draftSnapshot());
    if(!restored) return;
    draft=restored.document; selectedId=restored.selectedId;
    dirty=JSON.stringify(draft)!==JSON.stringify(state().document);
    error.textContent=''; renderedView=''; render();
    // Keep the original draft revision: undo must not resolve a stale-tab conflict.
    if(draftRevision!==state().revision) report(Error('Definition changed in another tab. Review before Reload saved.'));
  }
  function adopt(preserveHistory=false){
    if(!state()) return;
    if(!preserveHistory) history.clear(); else history.endGroup();
    draft=structuredClone(state().document);draftRevision=state().revision;dirty=false;
    selectedId=selectedId || draft.instances[0].id;renderedView='';render();
  }
  function controls(){
    const s=state(), phase=s?.state;
    const active=s?.run.status==='active', unavailable=!connected || sending || !s;
    run.textContent=phase==='paused'?'Resume':'Run';
    run.disabled=unavailable || (phase!=='paused' && (active || dirty || draftRevision!==s.revision));
    pause.disabled=unavailable || phase!=='running';step.disabled=unavailable || phase!=='paused';
    reset.disabled=unavailable || phase==='ready' || phase==='stopping';
    save.disabled=unavailable || active;open.disabled=unavailable || active;
    undo.disabled=unavailable || active || runtimeMode || !history.past.length;
    redo.disabled=unavailable || active || runtimeMode || !history.future.length;
    const structureDisabled=unavailable || active || runtimeMode || !catalog().length;
    for(const control of [prototypePicker,add,duplicate,remove,up,down]) {
      control.hidden=!catalog().length;control.disabled=structureDisabled;
    }
    const position=draft?.instances.findIndex(x=>x.id===selectedId) ?? -1;
    duplicate.disabled=remove.disabled=structureDisabled || position<0;
    const role=draft?.instances[position]?.role;
    up.disabled=structureDisabled || position<0 || !draft.instances.slice(0,position).some(x=>x.role===role);
    down.disabled=structureDisabled || position<0 || !draft.instances.slice(position+1).some(x=>x.role===role);
    exportButton.disabled=!s;
    document.querySelectorAll('[data-definition-field]').forEach(x=>x.disabled=unavailable || active);
    document.querySelectorAll('.dynamic-save-btn').forEach(b=>{
      const data=entityData[selectedEntity], component=data?.component_info?.context_components?.[b.dataset.component];
      const supported=['Instructions','Goal'].includes(b.dataset.component) && component?.class_name==='Constant' && b.dataset.stateKey==='state';
      b.hidden=!runtimeMode || !supported;
      b.disabled=unavailable || phase!=='paused';
      const input=$(b.dataset.inputId); if(input) input.readOnly=!runtimeMode || !supported || unavailable || phase!=='paused';
    });
    status.textContent=(!connected?'Disconnected · reconnecting; drafts kept':phase || 'Connecting')+
      (s ? ` · Step ${s.current_step} · ${dirty?'unsaved changes':'saved definition'}` : '')+
      (s?.run.message ? '\n'+s.run.message : '');
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
      error.textContent='';renderHierarchy();controls();
    };
    label.append(input);container.append(label);
  }
  function inspect(){
    if(!draft) return;
    const index=draft.instances.findIndex(x=>x.id===selectedId);
    const content=$('inspector-content');content.style.display='block';$('inspector-empty').style.display='none';
    if(index<0){
      $('inspector-title').textContent='Simulation';$('inspector-subtitle').textContent='Initial definition';content.replaceChildren();
      field(content,'Initial premise',draft.premise,v=>draft.premise=v,'editor-premise');
      field(content,'Maximum steps (1–1000)',draft.max_steps,v=>draft.max_steps=v,'editor-max-steps');
      controls();return;
    }
    const previewIndex=state().document.instances.findIndex(x=>x.id===selectedId);
    const sameEntity=selectedEntity==='entity_'+previewIndex;
    const expanded=sameEntity ? [...content.querySelectorAll('.component-state.expanded')].map(x=>x.id) : [];
    const focused=sameEntity && content.contains(document.activeElement) ? document.activeElement : null;
    const focusState=focused ? {id:focused.id,start:focused.selectionStart,end:focused.selectionEnd} : null;
    selectedEntity='entity_'+previewIndex;
    if(previewIndex>=0) updateInspector(selectedEntity);
    else {
      content.replaceChildren();
      const hint=document.createElement('p');hint.textContent='Save draft to build the component preview. No simulation is executed.';content.append(hint);
    }
    if(!runtimeMode) $('inspector-title').textContent=draft.instances[index].params.name;
    const item=draft.instances[index];
    $('inspector-subtitle').textContent=(runtimeMode?'Current runtime':'Initial definition')+' · '+item.prefab+' · '+item.id;
    if(!runtimeMode){
      const fields=document.createElement('section');fields.setAttribute('aria-label','Editable initial fields');
      const entry=catalog().find(x=>x.instance.prototype===item.prototype);
      const specs=structuredClone(entry?.inspector || state().definition.inspector[item.id] || {});
      for(const [field,role] of Object.entries(entry?.references || {})) {
        specs[field]={...specs[field],choices:draft.instances.filter(x=>x.role===role).map(x=>({value:x.id,label:x.params.name}))};
      }
      for(const [key,value] of Object.entries(item.params)) field(fields,key,value,v=>item.params[key]=v,'editor-'+item.id+'-'+key,specs[key]);
      content.prepend(fields);
    } else {
      content.querySelectorAll('.dynamic-input').forEach(input=>{
        const key=envelope.references.run_id+':'+selectedId+':'+input.id;
        if(runtimeDrafts.has(key)) input.value=runtimeDrafts.get(key).value;
        input.oninput=()=>{runtimeDrafts.set(key,{value:input.value,envelope:runtimeDrafts.get(key)?.envelope || structuredClone(envelope)});};
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
        const idx=state().document.instances.findIndex(x=>x.id===item.id);
        const components=Object.keys(entityData['entity_'+idx]?.component_info?.context_components || {});
        if(![item.params.name,item.id,item.prefab,...components].some(x=>x.toLocaleLowerCase().includes(query)))continue;
        rows.push({item,components});
      }
      if(!rows.length)continue;
      const title=document.createElement('h3');title.textContent={entity:'Actors',game_master:'Game masters',initializer:'Initializers'}[role];hierarchy.append(title);
      for(const {item,components} of rows){
        matches++;
        const b=button(item.params.name,()=>choose(item.id),hierarchy);b.dataset.instanceId=item.id;
        b.setAttribute('aria-pressed',String(item.id===selectedId));
        for(const component of components) {
          const c=button(component,()=>choose(item.id,component),hierarchy);c.classList.add('editor-component');
        }
      }
    }
    if(!matches){const empty=document.createElement('p');empty.textContent='No matching instances or components.';hierarchy.append(empty);}
  }
  function render(){
    if(!draft) return;
    const s=state(), view=runtimeMode?s.runtime:s.definition;
    const signature=JSON.stringify([runtimeMode,view]);
    if(signature!==renderedView){
      renderedView=signature;
      for(const key of Object.keys(entityData)) delete entityData[key];
      Object.assign(entityData,view?.entities || s.definition.entities);
      // SVG comes only from the standard escaped server renderer, never imported HTML.
      document.querySelector('.svg-container').innerHTML=view?.svg || s.definition.svg;
      const chosen=prototypePicker.value;prototypePicker.replaceChildren();
      for(const entry of catalog()) {
        const option=document.createElement('option');option.value=entry.instance.prototype;
        option.textContent=entry.instance.prefab+' · '+entry.instance.role+' · '+entry.instance.params.name;
        option.title=entry.description;prototypePicker.append(option);
      }
      if(catalog().some(x=>x.instance.prototype===chosen)) prototypePicker.value=chosen;
      renderHierarchy();
      inspect();
    }
    if(loggedRun!==envelope.references.run_id){loggedRun=envelope.references.run_id;loggedSteps=0;$('console-output').replaceChildren();}
    for(const entry of s.steps.slice(loggedSteps)) logConsole(`Step ${entry.step} · ${entry.acting_entity}\n${entry.action}`,'info');
    loggedSteps=s.steps.length;
    const latest=s.steps.at(-1);summary.textContent=latest?`Step ${latest.step} · ${latest.acting_entity}\n${latest.action}`:'Edit the initial definition, save, then Run.';
    controls();
  }
  document.addEventListener('click',event=>{
    const card=event.target.closest('.entity-card');if(!card || !draft)return;
    event.stopImmediatePropagation();const index=Number(card.dataset.entityId.split('_')[1]);
    const id=state().document.instances[index]?.id;
    if(draft.instances.some(x=>x.id===id))choose(id);
  },true);
  saveComponentState=async (_entity,component,_key,inputId)=>{
    const key=envelope.references.run_id+':'+selectedId+':'+inputId, saved=runtimeDrafts.get(key);
    if(await dispatch('runtime.edit',{instance_id:selectedId,component,value:$(inputId).value},saved?.envelope)){
      runtimeDrafts.delete(key);renderedView='';await refresh();
    }
  };
  function receive(next){
    if(envelope && next.references.session_id===envelope.references.session_id && next.revision<envelope.revision)return;
    const newSession=envelope && next.references.session_id!==envelope.references.session_id;
    envelope=next;connected=true;
    if(newSession && dirty){
      draftRevision=-1;renderedView='';
      report(Error('The server session changed. Unsaved fields are kept; review before Reload saved.'));
    } else if(!draft || newSession)adopt();
    else if(draftRevision!==state().revision){
      if(dirty)report(Error('Definition changed in another tab. Your fields are kept. Review before Reload saved.'));
      else adopt();
    }
    render();
  }
  async function refresh(){
    try {const response=await fetch('/api/state');if(!response.ok)throw Error('Cannot read editor state.');receive(await response.json());}
    catch(e){connected=false;controls();report(e);}
  }
  async function dispatch(operation,args={},source){
    if(!connected || sending)return false;
    sending=true;controls();error.textContent='';
    const from=source || envelope;
    try{
      const response=await fetch('/api/dispatch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({operation,arguments:args,revision:from.revision,references:from.references,retry_key:crypto.randomUUID()})});
      const result=await response.json();if(!response.ok)throw Error(result.error.message);
      if(operation==='project.run'){runtimeMode=true;mode.value='runtime';tab('simulation');}
      await refresh();return true;
    }catch(e){report(e);await refresh();return false;}
    finally{sending=false;controls();}
  }
  let events, timer, polling=false;
  function connect(){
    events=new EventSource('/api/events');
    events.onmessage=e=>receive(JSON.parse(e.data));
    events.onerror=()=>{connected=false;controls();};
    // Boundary acknowledgement changes without a completed-step event.
    timer=setInterval(async()=>{if(!polling){polling=true;try{await refresh();}finally{polling=false;}}},1000);
    refresh();
  }
  window.addEventListener('pagehide',()=>{clearInterval(timer);events.close();});
  window.addEventListener('pageshow',event=>{if(event.persisted)connect();});
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh();});
  connect();
  window.addEventListener('beforeunload',e=>{if(dirty || runtimeDrafts.size){e.preventDefault();e.returnValue='';}});
  refresh();controls();
})();
</script>
"""
