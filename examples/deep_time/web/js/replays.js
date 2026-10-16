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

// Replays: decided games played with the mock's minds,
// watched again exactly, from the start screen or the recap.

import {escapeHtml} from './util.js';

/**
 * One line of a replay: the world, how it ended, its stars, its length.
 * @param {!Object} r
 * @return {string}
 */
export function replayWords(r) {
  const stars = '★'.repeat(r.stars || 0) + '☆'.repeat(Math.max(0, 3 - (r.stars || 0)));
  return `${r.title || r.scenario} · ${r.outcome || 'undecided'} ${stars} · ${r.ticks} turns`;
}

/**
 * The start screen's list: a button per replay, newest first.
 * @param {?Array<!Object>} replays
 * @return {string}
 */
export function replaysHtml(replays) {
  if (!replays || !replays.length) return '';
  return '<h2>Watch again</h2>' + replays.map((r) =>
    `<button class="replay" data-game="${escapeHtml(r.game_id)}">▶ ${escapeHtml(replayWords(r))}`
    + `<small>${escapeHtml(r.when || '')}</small></button>`).join('');
}
