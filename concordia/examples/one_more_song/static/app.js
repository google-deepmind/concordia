import {safeStorage} from '../shared/browser-storage.js';
'use strict';
const $ = id => document.getElementById(id);
let request = null, busy = false, lastRevision = -1;
let waitingSince = null, lastEventCount = -1;
const storage = safeStorage('sessionStorage');
let draftKey = null;
function saveDraft() {
  if (!draftKey) return;
  if ($('reply').value) storage.set(draftKey, $('reply').value);
  else storage.remove(draftKey);
}
function alertText(id, text) { $(id).textContent = text; $(id).hidden = !text; }
function render(state) {
  request = state.pending;
  const g = state.game;
  const nextDraftKey = `one-more-song:${location.pathname}:${g.session_id}`;
  if (nextDraftKey !== draftKey) {
    // Each new host run has its own drafts; never carry an old offer into it.
    const typedBeforeConnect = draftKey === null ? $('reply').value : '';
    draftKey = nextDraftKey;
    $('reply').value = storage.get(draftKey) || typedBeforeConnect;
    saveDraft();
    lastRevision = -1;
    $('conversation').replaceChildren();
  }
  $('mode').hidden = g.mode !== 'fixture';
  $('ai-explainer').hidden = g.mode === 'fixture';
  $('status').textContent = state.status;
  if (request || state.finished) waitingSince = null;
  else if (waitingSince === null || g.events.length !== lastEventCount) waitingSince = Date.now();
  lastEventCount = g.events.length;
  const waitingSeconds = waitingSince === null ? 0 : Math.floor((Date.now() - waitingSince) / 1000);
  $('phase').textContent = state.finished ?
    (g.ending ? 'Conversation complete.' : 'Conversation stopped before the ending. Your recorded dialogue is available below.') : request ?
    (g.turn === 3 ? 'Turn 3 of 3 · Make your final proposal. Their votes follow.' : `Turn ${g.turn} of 3 · Listen, then speak in your own words.`) :
    `Waiting for the next voice (${waitingSeconds}s). Slower models can take a minute or more. Your draft stays here; no need to resend.`;
  $('send').disabled = !request || busy;
  $('send').textContent = request ? 'Say it' : 'Waiting…';
  $('form').hidden = state.finished;
  document.querySelector('.start-link').hidden = !!g.events.length || state.finished;
  if (state.revision !== lastRevision) {
    // Recorded events are append-only. Keep existing nodes so a screen reader
    // announces only new dialogue and reading/selection isn't reset.
    const transcript = $('conversation');
    if (g.events.length < transcript.children.length) transcript.replaceChildren();
    for (const e of g.events.slice(transcript.children.length)) {
      const item = document.createElement('article'); item.id = `event-${e.step}`; item.tabIndex = -1; item.className = 'entry' + (e.actor === 'You' ? ' player' : '');
      const who = document.createElement('strong'); who.textContent = e.actor;
      const text = document.createElement('p'); text.textContent = e.text.startsWith(e.actor + ':') ? e.text.slice(e.actor.length + 1).trim() : e.text;
      item.append(who, text); $('conversation').append(item);
    }
    lastRevision = state.revision;
  }
  const lastHuman = g.events.findLastIndex(e => e.actor === 'You');
  const latestReply = g.events.slice(lastHuman + 1).find(e => e.actor !== 'You' && e.step < 8);
  $('latest').hidden = !latestReply || state.finished;
  if (latestReply) $('latest-link').href = `#event-${latestReply.step}`;
  $('journal').hidden = !g.events.length;
  $('result').hidden = !g.ending;
  if (g.ending) {
    $('ending').textContent = g.ending;
    const offer = g.events[lastHuman];
    $('final-offer').textContent = offer ? offer.text.replace(/^You:\s*/, '') : '';
    $('votes').textContent = Object.entries(g.votes).map(([name, vote]) => `${name}: ${vote === 'ACCEPT' ? 'accepted' : 'declined'}`).join(' · ');
  }
}
async function poll() {
  try {
    const response = await fetch('api/state', {cache:'no-store', signal:AbortSignal.timeout(12000)});
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
    const response = await fetch('api/action', {method:'POST', headers:{'Content-Type':'application/json','X-Astral-Client':'1'}, body:JSON.stringify({request_id:id,response:draft}), signal:AbortSignal.timeout(15000)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The reply could not be accepted.');
    if ($('reply').value.trim() === draft) { $('reply').value = ''; saveDraft(); }
    request = null; alertText('error','');
  } catch (error) { alertText('error', (error.name === 'TimeoutError' || error instanceof TypeError ? 'Could not confirm submission. Check the conversation before trying again.' : error.message) + ' Your draft is kept.'); }
  finally { busy = false; }
});
document.querySelectorAll('[data-draft]').forEach(button => button.addEventListener('click', () => { $('reply').value = button.dataset.draft; saveDraft(); $('reply').focus(); }));
$('reply').addEventListener('input', saveDraft);
poll();
