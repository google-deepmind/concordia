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

// Moves planned ahead: a move queued for a later turn,
// sent by the server on that turn (or the player is asked, and told why);
// and the limited rewind back a turn.

import {escapeHtml, shortMove} from './util.js';

// How far ahead a move may be planned (session.PLAN_AHEAD).
export const PLAN_AHEAD = 5;

/**
 * The sheet's 'When' row: now, or 1 to PLAN_AHEAD turns ahead.
 * @param {number} selected
 * @param {number=} ahead
 * @return {string}
 */
export function whenHtml(selected, ahead = PLAN_AHEAD) {
  const turns = [0, ...Array.from({length: ahead}, (_, i) => i + 1)];
  return '<span class="movelabel">When</span>' + turns.map((n) =>
    `<button data-when="${n}" aria-pressed="${n === selected}"${n === selected ? ' class="on"' : ''}`
    + ` aria-label="${n ? `In ${n} turn${n === 1 ? '' : 's'}` : 'Now'}">${n ? `+${n}` : 'Now'}</button>`).join('');
}

/**
 * The plans as chips, soonest first, each with a button to cancel it.
 * @param {?Array<!Object>} plans
 * @return {string}
 */
export function plansHtml(plans) {
  return (plans || []).map((p) => `<span class="plan">⏱ <b>turn ${p.tick}</b>`
    + ` <span class="muted">(in ${p.in})</span> ${escapeHtml(shortMove(p.move))}`
    + `<button data-cancel="${p.tick}" aria-label="Cancel the plan for turn ${p.tick}">✕</button></span>`).join('');
}

/**
 * The next plan in a few words, for the dock: e.g. 'Planned: Seed bright
 * clouds in 2 turns'.
 * @param {?Array<!Object>} plans
 * @return {string}
 */
export function nextPlanWords(plans) {
  const next = (plans || [])[0];
  if (!next) return '';
  const when = next.in <= 0 ? 'this turn' : `in ${next.in} turn${next.in === 1 ? '' : 's'}`;
  const more = plans.length > 1 ? ` (+${plans.length - 1} more)` : '';
  return `Planned: ${shortMove(next.move)} ${when}${more}`;
}

/**
 * Where the chart marks planned moves: the tick each will act on.
 * @param {?Object} forecast
 * @return {!Array<number>}
 */
export function plannedTicks(forecast) {
  if (!forecast) return [];
  return (forecast.planned || []).map(([tick]) => tick + forecast.acts_in);
}

/**
 * The rewind button: back to the previous turn, while rewinds last. A first
 * tap arms it (`armed`), a second goes back; empty when it cannot.
 * @param {?Object} rewind
 * @param {boolean=} armed
 * @return {string}
 */
export function rewindHtml(rewind, armed = false) {
  if (!rewind) return '';
  const left = `${rewind.left} of 3 left`;
  if (!rewind.can) {
    return rewind.left > 0 ? '' : '<p class="rewind muted">↶ No rewinds left</p>';
  }
  const label = armed ? `Tap again: back to turn ${rewind.to}` : `↶ Back to turn ${rewind.to}`;
  return `<button class="rewind${armed ? ' armed' : ''}" id="rewindbtn" aria-label="${escapeHtml(label)}, ${left}">`
    + `${escapeHtml(label)} <small>${left}</small></button>`;
}


/**
 * What came of a course just held, in the player's own
 * readings: e.g. 'You held turns 2–5. Your thermometer: 29.6 → 21.2 °C ·
 * Ω 6.2 → 10.8 · biodiversity 0.63, steady.' and why it stopped, if it did.
 * @param {?Object} run
 * @return {string}
 */
export function heldWords(run) {
  if (!run) return '';
  const fmt = (v, d) => (v == null ? '?' : Number(v).toFixed(d));
  const move = ([a, b], d, unit = '') => (a != null && b != null && Math.abs(b - a) < 0.5 * 10 ** -d
    ? `${fmt(b, d)}${unit}, steady` : `${fmt(a, d)} → ${fmt(b, d)}${unit}`);
  const parts = [];
  if (run.read_c && run.read_c[1] != null) parts.push(`your thermometer ${move(run.read_c, 1, ' °C')}`);
  parts.push(`Ω ${move(run.omega, 1)}`);
  if (run.life && run.life[1] != null) parts.push(`biodiversity ${move(run.life, 2)}`);
  const text = `You held turns ${run.from}–${run.to}: ${parts.join(' · ')}.`;
  return run.stopped ? `${text} Stopped: ${run.stopped}.` : text;
}
