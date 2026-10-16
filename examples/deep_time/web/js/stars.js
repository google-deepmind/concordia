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

// Stars: the goal, and harder ways to reach it.

import {escapeHtml} from './util.js';

/**
 * The stars as a list: each rule with a star, gold once earned, grey once
 * missed, plain while the game is on (`met` null).
 * @param {?Array<!Object>} stars
 * @return {string}
 */
export function starsHtml(stars) {
  return (stars || []).map((star) => {
    const cls = star.met === true ? 'met' : star.met === false ? 'missed' : '';
    return `<li class="${cls}"><b>${star.met === true ? '★' : '☆'}</b>${escapeHtml(star.text)}</li>`;
  }).join('');
}

/**
 * E.g. '★★☆' for two of three earned.
 * @param {?Array<!Object>} stars
 * @return {string}
 */
export function starLine(stars) {
  return (stars || []).map((star) => (star.met ? '★' : '☆')).join('');
}

/**
 * E.g. '2 of 3 stars'.
 * @param {?Array<!Object>} stars
 * @return {string}
 */
export function starWords(stars) {
  const all = (stars || []).length, met = (stars || []).filter((s) => s.met).length;
  return `${met} of ${all} star${all === 1 ? '' : 's'}`;
}

// Side objectives: optional tasks, each paid in Ω when met.
// `side` is the state's rules (no `state`), the Ω reading's (each as last
// read), or the recap's (verdicts on the true world); `tick` (the reading's
// or the game's) says how long a task still has to go.
const SIDE_MARK = {open: '◇', met: '✓', missed: '✗'};

/**
 * Renders side objectives as HTML list items.
 * @param {?Array<!Object>} side
 * @param {?number=} tick
 * @return {string}
 */
export function sideHtml(side, tick) {
  return (side || []).map((task) => {
    const state = task.state || 'open';
    let when = '';
    if (state === 'open' && tick != null) {
      const left = Math.max(0, task.until - tick);
      when = tick < task.from - 1 ? ` · from turn ${task.from}` : ` · ${left} turn${left === 1 ? '' : 's'} to go`;
    }
    return `<li class="side ${state === 'open' ? '' : state}"><b>${SIDE_MARK[state] || '◇'}</b>${escapeHtml(task.text)}`
      + ` <span class="reward">+${task.reward} Ω${when}</span></li>`;
  }).join('');
}

/**
 * E.g. '1 of 1 side objective met (+6 Ω)'; '' with none.
 * @param {?Array<!Object>} side
 * @return {string}
 */
export function sideWords(side) {
  const all = (side || []).length;
  if (!all) return '';
  const met = side.filter((task) => task.state === 'met');
  const gain = met.reduce((sum, task) => sum + task.reward, 0);
  return `${met.length} of ${all} side objective${all === 1 ? '' : 's'} met${gain ? ` (+${gain} Ω)` : ''}`;
}
