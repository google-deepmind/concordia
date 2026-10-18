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

// The story so far: the Advisor's briefings, the Gazette's
// broadsheets, the news a role has heard and, once revealed, the
// Chronicler's chapters, newest first, each with its turn.

import {escapeHtml} from './util.js';

export const STORY_ICONS = {advisor: '🧭', press: '📰', news: '📣', chronicle: '📜', war: '⚔️'};

/**
 * The feed as HTML (`items` from the game's state, newest first). Long
 * entries are clamped; a tap opens them (the page toggles `open`).
 * @param {?Array<!Object>} items
 * @return {string}
 */
export function storyHtml(items) {
  if (!items || !items.length) return '<p class="none">Nothing has happened yet.</p>';
  return items.map((item) => `<div class="storyrow ${escapeHtml(item.kind)}">`
    + `<span class="turnchip">t${Number(item.tick) || 0}</span>`
    + `<span class="sicon" aria-hidden="true">${STORY_ICONS[item.kind] || '·'}</span>`
    + `<p><b>${escapeHtml(item.who)}</b> ${escapeHtml(item.text)}</p></div>`).join('');
}
