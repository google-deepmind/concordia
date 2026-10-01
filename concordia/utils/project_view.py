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
    const owned=(document.components || []).filter(x=>x.instance===sourceId);
    if((document.components?.length || 0)+owned.length>100) throw Error('At most 100 authored components.');
    document.instances.push(item);
    if(sourceId && document.components) {
      for(const component of owned) document.components.push({...structuredClone(component),id:crypto.randomUUID(),instance:id});
    }
    return id;
  }
  static remove(document,catalog,id) {
    const item=document.instances.find(x=>x.id===id);
    if((document.groups || []).some(x=>x.participants.includes(id)) ||
       (document.scenes || []).some(x=>x.participants.includes(id)) ||
       (document.scene_types || []).some(x=>x.game_master===id))
      throw Error('This instance is referenced by a participant group or scene. Update those references first.');
    if(!item) throw Error('Select an instance.');
    if(['entity','game_master'].includes(item.role) && document.instances.filter(x=>x.role===item.role).length===1)
      throw Error('Keep at least one actor and one game master.');
    for(const other of document.instances.filter(x=>x.id!==id)) {
      const entry=catalog.find(x=>x.instance.prototype===other.prototype);
      for(const field of Object.keys(entry?.references || {}))
        if(other.params[field]===id) throw Error(other.params.name+' references this instance through '+field+'. Change that reference first.');
    }
    document.instances=document.instances.filter(x=>x.id!==id);
    if(document.components)document.components=document.components.filter(x=>x.instance!==id);
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
class ProjectSceneOperations {
  static locate(document, selection) {
    const [kind,id]=String(selection).split(':');
    if(!['groups','scene_types','scenes','components'].includes(kind)) return null;
    const index=document[kind]?.findIndex(x=>x.id===id) ?? -1;
    return index<0 ? null : {kind,index,item:document[kind][index]};
  }
  static add(document,catalog,kind,id) {
    if(!document[kind] || document[kind].length>=100) throw Error('At most 100 records per section.');
    const actors=document.instances.filter(x=>x.role==='entity').map(x=>x.id);
    const masters=document.instances.filter(x=>catalog.some(c=>c.instance.prototype===x.prototype && c.accepts_scenes));
    let item;
    if(kind==='groups') item={id,name:'Participant group',participants:actors};
    if(kind==='scene_types') {
      if(!masters.length || !document.groups.length) throw Error('Create a scene-aware GM and participant group first.');
      item={id,name:'Scene type',game_master:masters[0].id,group:document.groups[0].id,premise:''};
    }
    if(kind==='scenes') {
      const type=document.scene_types[0], group=document.groups.find(x=>x.id===type?.group);
      if(!type || !group) throw Error('Create a scene type and participant group first.');
      item={id,name:'Scene',scene_type:type.id,participants:[...group.participants],num_rounds:1,premise:null};
    }
    if(!item) throw Error('Unknown scene structure.');
    const base=item.name;let n=2;while(document[kind].some(x=>x.name===item.name))item.name=base+' '+n++;
    document[kind].push(item);return kind+':'+id;
  }
  static remove(document,selection) {
    const found=this.locate(document,selection);
    if(!found) throw Error('Select a scene, type or group.');
    const {kind,item}=found;
    if(kind!=='components' && document[kind].length===1) throw Error('Keep at least one '+kind.replaceAll('_',' ')+'.');
    if(kind==='groups' && document.scene_types.some(x=>x.group===item.id)) throw Error('This group is used by a scene type. Change that reference first.');
    if(kind==='scene_types' && document.scenes.some(x=>x.scene_type===item.id)) throw Error('This scene type is used by a scene. Change that reference first.');
    document[kind].splice(found.index,1);return kind==='components'?item.instance:kind+':'+document[kind][0].id;
  }
  static duplicate(document,selection,id) {
    const found=this.locate(document,selection);
    if(!found) throw Error('Selection no longer exists.');
    const {kind,item}=found;
    ProjectComponentOperations.checkId(document[kind],id);
    if(document[kind].length>=100) throw Error('At most 100 records per section.');
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
    const root=/^\$\.(premise|max_steps): /.exec(message);
    if(root) return {selection:'simulation',field:root[1]==='premise'?'editor-premise':'editor-max-steps'};
    const match=/^\$\.(instances|components|groups|scene_types|scenes)\[([A-Za-z0-9_-]{1,128})\](?:\.([A-Za-z0-9_.-]+))?: /.exec(message);
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
    const fields=kind==='scene_types'?{game_master:'world-game-master',group:'world-group',premise:'world-premise'}:
      kind==='scenes'?{scene_type:'world-type',num_rounds:'world-rounds',premise:item.premise===null?'world-inherit':'world-premise'}:{};
    if(Object.hasOwn(fields,path)) field=fields[path];
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
      for(const kind of ['groups','scenes']) for(const item of document[kind] || [])
        if(item.participants.includes(instance.id)) add(kind+':'+item.id,'participants',item.name+' · participants');
      for(const item of document.scene_types || [])
        if(item.game_master===instance.id) add('scene_types:'+item.id,'game_master',item.name+' · game master');
      for(const item of document.components || [])
        if(item.instance===instance.id) add('components:'+item.id,'instance',item.name+' · owned component',true);
    } else {
      const found=ProjectSceneOperations.locate(document,selection);
      if(found?.kind==='groups') for(const item of document.scene_types || [])
        if(item.group===found.item.id) add('scene_types:'+item.id,'group',item.name+' · participant group');
      if(found?.kind==='scene_types') for(const item of document.scenes || [])
        if(item.scene_type===found.item.id) add('scenes:'+item.id,'scene_type',item.name+' · scene type');
    }
    return uses;
  }
  static candidates(document,catalog,id) {
    const source=document.instances.find(x=>x.id===id);
    if(!source || !catalog.some(x=>x.instance.prototype===source.prototype)) return [];
    const needsScenes=(document.scene_types || []).some(x=>x.game_master===id);
    return document.instances.filter(x=>x.id!==id && x.role===source.role &&
      catalog.some(c=>c.instance.prototype===x.prototype && c.instance.role===x.role && (!needsScenes || c.accepts_scenes)));
  }
  static replace(document,catalog,id,targetId) {
    if(!this.candidates(document,catalog,id).some(x=>x.id===targetId))
      throw Error('Choose an existing compatible reference target. Scene types require a scene-aware game master.');
    const uses=this.uses(document,catalog,id).filter(x=>!x.owned);
    if(!uses.length) throw Error('This instance has no replaceable references.');
    // Plan against a copy so a stale/invalid request never partially edits a draft.
    const next=structuredClone(document);
    for(const use of uses){
      const instance=next.instances.find(x=>x.id===use.selection);
      const owner=instance?.params || ProjectSceneOperations.locate(next,use.selection)?.item;
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

"""

EDITOR_SCRIPT = '<script>\n' + DRAFT_HISTORY_SCRIPT + r"""
(() => {
  const $ = id => document.getElementById(id);
  const layout = document.querySelector('.layout');
  const toolbar = document.createElement('div'); toolbar.id = 'editor-toolbar';
  document.querySelector('.header').append(toolbar);
  const heading=document.createElement('strong'); heading.id='editor-heading';
  heading.textContent=document.querySelector('.header h1').textContent; toolbar.append(heading);
  let envelope, draft, draftRevision, draftDefinition, draftBaseDocument, selectedId, connected = false, sending = false;
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
    if(ProjectSceneOperations.locate(next,selectedId)) return ProjectSceneOperations.duplicate(next,selectedId,crypto.randomUUID());
    const source=next.instances.find(x=>x.id===selectedId);
    if(!source) throw Error('Select an instance to duplicate.');
    return ProjectDraftOperations.add(next,catalog(),source.prototype,crypto.randomUUID(),source.id);
  }));
  const remove=button('Remove',()=>structural(next=>ProjectSceneOperations.locate(next,selectedId)?ProjectSceneOperations.remove(next,selectedId):ProjectDraftOperations.remove(next,catalog(),selectedId)));
  const up=button('Move earlier',()=>structural(next=>{moveSelection(next,-1);return selectedId;}));
  const down=button('Move later',()=>structural(next=>{moveSelection(next,1);return selectedId;}));
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
  function report(e){
    error.textContent=e.message || String(e);
    if(e.validationTarget){
      const target=e.validationTarget, submitted=JSON.stringify(draft);
      const link=button(target.field?'Show invalid field':'Show invalid item',()=>{
        if(runtimeMode) return;
        if(JSON.stringify(draft)!==submitted){report(Error('Draft changed since validation. Save again to locate the current error.'));return;}
        search.value='';choose(target.selection);
        const input=target.field && $(target.field);
        if(input){input.focus();input.scrollIntoView({block:'center'});}
      },error);
      link.dataset.validationAction='true';
    }
  }
  function catalog(){return draftDefinition?.catalog || [];}
  function moveSelection(next,offset){
    if(ProjectSceneOperations.locate(next,selectedId)) ProjectSceneOperations.move(next,selectedId,offset);
    else ProjectDraftOperations.move(next,selectedId,offset);
  }
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
    draftDefinition=state().definition;draftBaseDocument=state().document;
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
    const world=draft && ProjectSceneOperations.locate(draft,selectedId);
    if(world){
      duplicate.disabled=remove.disabled=structureDisabled;
      up.disabled=structureDisabled || world.index===0;
      down.disabled=structureDisabled || world.index===draft[world.kind].length-1;
      if(world.kind==='components'){
        up.disabled=structureDisabled || !draft.components.slice(0,world.index).some(x=>x.instance===world.item.instance);
        down.disabled=structureDisabled || !draft.components.slice(world.index+1).some(x=>x.instance===world.item.instance);
      }
    }
    document.querySelectorAll('[data-author-action]').forEach(x=>x.disabled=structureDisabled);
    document.querySelectorAll('[data-validation-action]').forEach(x=>x.disabled=runtimeMode || !draft);
    exportButton.disabled=!s;
    document.querySelectorAll('[data-definition-field]').forEach(x=>x.disabled=unavailable || active || runtimeMode);
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
      const description=document.createElement('p');description.textContent=spec.description+' Requires: '+(spec.dependencies.length?'the owner’s memory':'no other components')+'. Name is an editor label; context label controls the text supplied to the actor or GM. Save to rebuild the preview. Runtime edits are separate.';content.append(description);
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
    const hint=document.createElement('p');hint.textContent=kind==='groups'?'Reusable possible participants, not a simulated institution.':kind==='scene_types'?'A standard scene type selects its GM and possible participants.':'Ordered scene rounds and participants. Runtime scheduling uses standard SceneTracker.';content.append(hint);
    field(content,'Name',item.name,v=>item.name=v,'world-name');
    const choices=items=>items.map(x=>({value:x.id,label:x.name ?? x.params.name}));
    if(kind==='scene_types'){
      const masters=draft.instances.filter(x=>catalog().some(c=>c.instance.prototype===x.prototype && c.accepts_scenes));
      field(content,'Game master',item.game_master,v=>item.game_master=v,'world-game-master',{choices:choices(masters)});
      field(content,'Participant group',item.group,v=>item.group=v,'world-group',{choices:choices(draft.groups)});
      field(content,'Default premise for each participant',item.premise,v=>item.premise=v,'world-premise');
    }
    if(kind==='scenes'){
      field(content,'Scene type',item.scene_type,v=>item.scene_type=v,'world-type',{choices:choices(draft.scene_types),refresh:true});
      field(content,'Number of rounds',item.num_rounds,v=>item.num_rounds=v,'world-rounds');
      field(content,'Use scene type premise',item.premise===null,v=>item.premise=v?null:'','world-inherit',{refresh:true});
      if(item.premise!==null)field(content,'Premise override for each participant',item.premise,v=>item.premise=v,'world-premise');
    }
    if(kind==='groups' || kind==='scenes'){
      const heading=document.createElement('h3');heading.textContent='Participants';content.append(heading);
      const group=kind==='scenes'?draft.groups.find(x=>x.id===draft.scene_types.find(t=>t.id===item.scene_type)?.group):null;
      for(const actor of draft.instances.filter(x=>x.role==='entity')){
        const label=actor.params.name+(group && !group.participants.includes(actor.id)?' (outside selected group)':'');
        field(content,label,item.participants.includes(actor.id),v=>{
          item.participants=v?[...item.participants,actor.id]:item.participants.filter(x=>x!==actor.id);
        },'world-participant-'+actor.id);
      }
    }
    inspectReferences(content,selectedId);controls();
  }
  function inspect(){
    if(!draft) return;
    const world=ProjectSceneOperations.locate(draft,selectedId);
    if(world){inspectWorld(world);return;}
    const index=draft.instances.findIndex(x=>x.id===selectedId);
    const content=$('inspector-content');content.style.display='block';$('inspector-empty').style.display='none';
    if(index<0){
      $('inspector-title').textContent='Simulation';$('inspector-subtitle').textContent='Initial definition';content.replaceChildren();
      field(content,'Initial premise',draft.premise,v=>draft.premise=v,'editor-premise');
      field(content,'Maximum steps (1–1000)',draft.max_steps,v=>draft.max_steps=v,'editor-max-steps');
      controls();return;
    }
    const previewIndex=(runtimeMode?state().document:draftBaseDocument).instances.findIndex(x=>x.id===selectedId);
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
      const specs=structuredClone(entry?.inspector || draftDefinition.inspector[item.id] || {});
      for(const [field,role] of Object.entries(entry?.references || {})) {
        specs[field]={...specs[field],choices:draft.instances.filter(x=>x.role===role).map(x=>({value:x.id,label:x.params.name})),refresh:true};
      }
      for(const [key,value] of Object.entries(item.params)) field(fields,key,value,v=>item.params[key]=v,'editor-'+item.id+'-'+key,specs[key]);
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
    for(const [kind,title] of [['scenes','Scene sequence'],['scene_types','Scene types'],['groups','Participant groups']]){
      if(!draft[kind])continue;
      const heading=document.createElement('h3');heading.textContent=title;hierarchy.append(heading);
      const action=button('Add '+({scenes:'scene',scene_types:'scene type',groups:'group'}[kind]),()=>structural(next=>ProjectSceneOperations.add(next,catalog(),kind,crypto.randomUUID())),hierarchy);action.dataset.authorAction='true';
      draft[kind].forEach((item,index)=>{
        if(![item.name,item.id].some(x=>x.toLocaleLowerCase().includes(query)))return;
        matches++;
        const label=kind==='scenes'?`${index+1}. ${item.name} · ${item.num_rounds} rounds`:item.name;
        const b=button(label,()=>choose(kind+':'+item.id),hierarchy);b.dataset.worldId=kind+':'+item.id;
        b.setAttribute('aria-pressed',String(selectedId===kind+':'+item.id));
      });
    }
    for(const role of ['entity','game_master','initializer']){
      const rows=[];
      for(const item of draft.instances.filter(x=>x.role===role)) {
        const idx=(runtimeMode?state().document:draftBaseDocument).instances.findIndex(x=>x.id===item.id);
        const components=Object.keys(entityData['entity_'+idx]?.component_info?.context_components || {});
        if(![item.params.name,item.id,item.prefab,...components,...(draft.components || []).filter(c=>c.instance===item.id).flatMap(c=>[c.name,c.type])].some(x=>x.toLocaleLowerCase().includes(query)))continue;
        rows.push({item,components});
      }
      if(!rows.length)continue;
      const title=document.createElement('h3');title.textContent={entity:'Actors',game_master:'Game masters',initializer:'Initializers'}[role];hierarchy.append(title);
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
    if(!matches){const empty=document.createElement('p');empty.textContent='No matching instances or components.';hierarchy.append(empty);}
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
    const id=(runtimeMode?state().document:draftBaseDocument).instances[index]?.id;
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
      const result=await response.json();
      if(!response.ok){
        const failure=Error(result.error.message);
        failure.validationTarget=ProjectValidation.forSave(draft,operation,args,result.error);
        throw failure;
      }
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
