// Copyright 2026 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The omen composer: where the Steward may speak, the move sheet opens on
// a sign every people will see and the words it will be taken to say. The
// answer goes back as "<sign>: <words>", Concordia's free action.

import {escapeHtml} from './util.js';

export const OMEN_ASK = 'Send the peoples a sign, or make one move.';

/**
 * The composer as HTML (`omen` from the question): the signs to pick from,
 * a box for the words and the send button, or why it cannot be spoken now.
 * @param {?Object} omen
 * @param {number=} omega What the Steward holds now.
 * @return {string}
 */
export function omenHtml(omen, omega) {
  if (!omen) return '';
  const signs = omen.signs.map((s, i) => `<button class="sign${i ? '' : ' on'}" data-sign="${escapeHtml(s)}"`
    + ` aria-pressed="${i === 0}">${escapeHtml(s)}</button>`).join('');
  const why = omen.ready ? '' : (omega != null && omega < omen.cost
    ? `It needs ${omen.cost} Ω; you have ${Math.floor(omega * 10) / 10}.`
    : 'The sky is still quiet from the last one: next turn.');
  return `<div class="omen"><p class="movelabel">Speak an omen · ${omen.cost} Ω</p>`
    + `<div class="signs" role="group" aria-label="The sign">${signs}</div>`
    + `<textarea id="omenwords" aria-label="What the sign will be taken to say"`
    + ` placeholder="What the sign will be taken to say (up to ${omen.max_words} words)"${omen.ready ? '' : ' disabled'}></textarea>`
    + `<div class="row"><span class="small muted" id="omencount">0 of ${omen.max_words} words</span>`
    + `<button class="primary" id="omensend"${omen.ready ? '' : ' disabled'}>Send the omen</button></div>`
    + `<p class="small muted">${why ? escapeHtml(why) + ' ' : ''}Every people sees the sign;`
    + ' the Gazette\'s printers will tell them what it meant, a turn later.</p></div>'
    + '<p class="movelabel">Or a move</p>';
}

/**
 * The words as sent: at most `maxWords`, single-spaced.
 * @param {string} text
 * @param {number} maxWords
 * @return {string}
 */
export function omenWords(text, maxWords) {
  return String(text || '').trim().split(/\s+/).filter(Boolean).slice(0, maxWords).join(' ');
}

/**
 * The free answer for a sign and its words: "<sign>: <words>".
 * @param {string} sign
 * @param {string} words
 * @return {string}
 */
export function omenResponse(sign, words) {
  return `${sign}: ${words}`.trim();
}
