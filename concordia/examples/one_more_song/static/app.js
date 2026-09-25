import {safeStorage} from '../shared/browser-storage.js';
'use strict';
const $ = id => document.getElementById(id);
let request = null, busy = false, lastRevision = -1;
let waitingSince = null, lastEventCount = -1;
const storage = safeStorage('sessionStorage');
let draftKey = null, uncertainSubmission = null, previousDraft = null;
function keepPreviousDraft(text, source = 'suggestion') {
  previousDraft = text || null;
  $('restore-draft').hidden = previousDraft === null;
  $('restore-draft').textContent = source === 'previous-game' ? 'Use draft from previous game' : 'Restore previous draft';
  if (!draftKey) return;
  if (previousDraft) storage.set(draftKey + ':previous', JSON.stringify({text:previousDraft, source}));
  else storage.remove(draftKey + ':previous');
}
function saveSubmission() {
  if (!draftKey) return;
  if (uncertainSubmission) storage.set(draftKey + ':submission', JSON.stringify(uncertainSubmission));
  else storage.remove(draftKey + ':submission');
}
function saveDraft() {
  if (!draftKey) return;
  if ($('reply').value) storage.set(draftKey, $('reply').value);
  else storage.remove(draftKey);
}
function alertText(id, text) { if ($(id).textContent !== text) $(id).textContent = text; $(id).hidden = !text; }
function render(state) {
  request = state.pending;
  const g = state.game;
  const nextDraftKey = `one-more-song:${location.pathname}:${g.session_id}`;
  if (nextDraftKey !== draftKey) {
    // Each new host run has its own drafts; never carry an old offer into it.
    const priorGameDraft = draftKey === null ? '' : ($('reply').value || previousDraft || '');
    const typedBeforeConnect = draftKey === null ? $('reply').value : '';
    draftKey = nextDraftKey;
    $('reply').value = storage.get(draftKey) || typedBeforeConnect;
    saveDraft();
    uncertainSubmission = null;
    try {
      const saved = JSON.parse(storage.get(draftKey + ':submission'));
      if (saved && typeof saved.id === 'string' && typeof saved.response === 'string') uncertainSubmission = saved;
    } catch {}
    let undo = null;
    try { undo = JSON.parse(storage.get(draftKey + ':previous')); } catch {}
    keepPreviousDraft(priorGameDraft || (undo && typeof undo.text === 'string' ? undo.text : ''), priorGameDraft ? 'previous-game' : undo?.source);
    alertText('error', uncertainSubmission ? 'A previous send was not confirmed in this tab. Check last send before taking another turn. Your draft is kept.' : priorGameDraft ? 'The host started a fresh game. Your previous draft is available below if you want to use it; review it for this new conversation.' : '');
    lastRevision = -1;
    $('conversation').replaceChildren();
  }
  $('mode').hidden = g.mode !== 'fixture';
  $('ai-explainer').hidden = g.mode === 'fixture';
  if ($('status').textContent !== state.status) $('status').textContent = state.status;
  if (request || state.finished) waitingSince = null;
  else if (waitingSince === null || g.events.length !== lastEventCount) waitingSince = Date.now();
  lastEventCount = g.events.length;
  const waitingSeconds = waitingSince === null ? 0 : Math.floor((Date.now() - waitingSince) / 1000);
  $('phase').textContent = state.finished ?
    (g.ending ? 'Conversation complete.' : 'Conversation stopped before the ending. Use the conversation download to keep the dialogue recorded so far.') : request ?
    (g.turn === 3 ? 'Turn 3 of 3 · Make your final proposal. Their votes follow.' : `Turn ${g.turn} of 3 · Listen, then speak in your own words.`) :
    `Waiting for the next voice (${waitingSeconds}s). Slower models can take a minute or more. Your draft stays here; no need to resend.`;
  $('send').disabled = (!request && !uncertainSubmission) || busy;
  const sendLabel = busy ? 'Sending…' : uncertainSubmission ? 'Check last send' : request ? 'Say it' : 'Waiting…';
  if ($('send').textContent !== sendLabel) $('send').textContent = sendLabel;
  $('form').hidden = state.finished;
  // The last turn must contain an offer: don't suggest spending it on another
  // opening question. Suggestions still only fill an editable draft.
  const suggestions = document.querySelectorAll('[data-draft]');
  suggestions[0].hidden = g.turn === 3;
  const finalTurn = g.turn === 3;
  const suggestionLabel = finalTurn ? 'Suggest a final offer' : 'Suggest a compromise';
  if (suggestions[1].textContent !== suggestionLabel) suggestions[1].textContent = suggestionLabel;
  suggestions[1].dataset.draft = finalTurn ?
    'My final offer: one quiet, unamplified song lasting at most two minutes, then silence. Do you both agree?' :
    'Could we agree to one quiet, unamplified song with a firm end time?';
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
  event.preventDefault(); if ((!request && !uncertainSubmission) || busy) return;
  const attempt = uncertainSubmission || {id:request.id, response:$('reply').value.trim()};
  if (!attempt.response) { alertText('error', 'Write something to say first.'); $('reply').focus(); return; }
  // The existing HumanSession inbox makes retries of this exact ID and response
  // idempotent. Never retry an uncertain send using the next turn's request ID.
  uncertainSubmission = attempt; saveSubmission();
  busy = true; $('send').disabled = true;
  try {
    const response = await fetch('api/action', {method:'POST', headers:{'Content-Type':'application/json','X-Astral-Client':'1'}, body:JSON.stringify({request_id:attempt.id,response:attempt.response}), signal:AbortSignal.timeout(15000)});
    const data = await response.json();
    if (!response.ok) {
      uncertainSubmission = null; saveSubmission();
      alertText('error', (typeof data.detail === 'string' ? data.detail : 'The reply could not be accepted.') + ' Review your draft before taking another turn.');
      return;
    }
    uncertainSubmission = null; saveSubmission();
    keepPreviousDraft(null);
    if ($('reply').value.trim() === attempt.response) { $('reply').value = ''; saveDraft(); }
    request = null; alertText('error','');
  } catch (_) {
    alertText('error', 'Could not confirm submission. Use Check last send to check the same turn safely. Your draft is kept.');
  } finally { busy = false; }
});
document.querySelectorAll('[data-draft]').forEach(button => button.addEventListener('click', () => {
  const before = $('reply').value;
  if (before === button.dataset.draft) { $('reply').focus(); return; }
  keepPreviousDraft(before);
  $('reply').value = button.dataset.draft; saveDraft(); $('reply').focus();
}));
$('restore-draft').addEventListener('click', () => {
  if (previousDraft === null) return;
  $('reply').value = previousDraft; keepPreviousDraft(null);
  if ($('error').textContent.startsWith('The host started a fresh game.')) alertText('error', '');
  saveDraft(); $('reply').focus();
});
$('reply').addEventListener('input', () => {
  keepPreviousDraft(null); saveDraft();
});
poll();
