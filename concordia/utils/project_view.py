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
