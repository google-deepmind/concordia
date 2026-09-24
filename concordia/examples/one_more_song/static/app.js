'use strict';
const $ = id => document.getElementById(id);
let request = null, busy = false, lastRevision = -1;
function alertText(id, text) { $(id).textContent = text; $(id).hidden = !text; }
function render(state) {
  request = state.pending;
  const g = state.game;
  $('mode').hidden = g.mode !== 'fixture';
  $('status').textContent = state.status;
  $('phase').textContent = state.finished ? 'Conversation complete.' : request ?
    (g.turn === 3 ? 'Turn 3 of 3 · Make your final proposal. Their votes follow.' : `Turn ${g.turn} of 3 · Listen, then speak in your own words.`) :
    'Waiting for the next voice. Your draft stays here; no need to resend.';
  $('send').disabled = !request || busy;
  $('send').textContent = request ? 'Say it' : 'Waiting…';
  $('form').hidden = state.finished;
  if (state.revision !== lastRevision) {
    $('conversation').replaceChildren();
    for (const e of g.events) {
      const item = document.createElement('article'); item.className = 'entry' + (e.actor === 'You' ? ' player' : '');
      const who = document.createElement('strong'); who.textContent = e.actor;
      const text = document.createElement('p'); text.textContent = e.text.startsWith(e.actor + ':') ? e.text.slice(e.actor.length + 1).trim() : e.text;
      item.append(who, text); $('conversation').append(item);
    }
    lastRevision = state.revision;
  }
  $('journal').hidden = !g.events.length;
  $('result').hidden = !g.ending;
  if (g.ending) { $('ending').textContent = g.ending; $('votes').textContent = Object.entries(g.votes).map(([name, vote]) => `${name}: ${vote}`).join(' · '); }
}
async function poll() {
  try {
    const response = await fetch('api/state', {cache:'no-store'});
    if (!response.ok) throw new Error('Connection unavailable');
    render(await response.json()); alertText('network', '');
  } catch (_) {
    request = null; $('send').disabled = true;
    alertText('network', 'Connection lost. Your draft is kept. Reconnecting…');
  } finally { setTimeout(poll, 600); }
}
$('form').addEventListener('submit', async event => {
  event.preventDefault(); if (!request || busy) return;
  const draft = $('reply').value.trim();
  if (!draft) { alertText('error', 'Write something to say first.'); $('reply').focus(); return; }
  const id = request.id; busy = true; $('send').disabled = true;
  try {
    const response = await fetch('api/action', {method:'POST', headers:{'Content-Type':'application/json','X-Astral-Client':'1'}, body:JSON.stringify({request_id:id,response:draft})});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The reply could not be accepted.');
    if ($('reply').value.trim() === draft) $('reply').value = '';
    request = null; alertText('error','');
  } catch (error) { alertText('error', error.message + ' Your draft is kept.'); }
  finally { busy = false; }
});
document.querySelectorAll('[data-draft]').forEach(button => button.addEventListener('click', () => { $('reply').value = button.dataset.draft; $('reply').focus(); }));
poll();
