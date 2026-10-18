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

// Scenario cards: the start screen's worlds as a strip of
// cards to swipe through, each with its map at the start, its age, premise,
// and the stars won or the lock of the campaign.

import {escapeHtml} from './util.js';

/**
 * The world's map as pixels, one a tile: {w, h, data} (RGBA), from the
 * biome codes the server sends (north first) and their colours.
 * @param {!Object} thumb
 * @param {boolean=} dark
 * @return {!Object}
 */
export function thumbPixels(thumb, dark = false) {
  const rows = thumb.rows || [];
  const h = rows.length, w = h ? rows[0].length : 0;
  const colour = {};
  for (const b of thumb.legend || []) colour[b.code] = dark ? b.dark : b.light;
  const data = new Uint8ClampedArray(w * h * 4);
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x += 1) {
      const c = colour[row[x]] || [128, 128, 128];
      const i = 4 * (y * w + x);
      data[i] = c[0]; data[i + 1] = c[1]; data[i + 2] = c[2]; data[i + 3] = 255;
    }
  });
  return {w, h, data};
}

/**
 * The first sentence of a premise, for a card.
 * @param {string} text
 * @return {string}
 */
export function firstSentence(text) {
  const m = /^.*?[.!?](\s|$)/.exec(text || '');
  return (m ? m[0] : text || '').trim();
}

/**
 * One card: the map (a canvas the page paints), the title and age, the
 * premise's first sentence, and the stars won. Any world may be played from
 * here, so a chapter the campaign has not opened yet shows no lock.
 * @param {!Object} s
 * @param {?Object} chapter
 * @param {boolean=} selected
 * @return {string}
 */
export function scenarioCardHtml(s, chapter, selected = false) {
  const stars = chapter && chapter.open ? '★'.repeat(chapter.stars) + '☆'.repeat(3 - chapter.stars) : '';
  return `<button class="scard${selected ? ' on' : ''}" data-scenario="${escapeHtml(s.key)}"`
    + ` aria-pressed="${selected}" aria-label="${escapeHtml(s.title)}">`
    + `<canvas class="sthumb" data-thumb="${escapeHtml(s.key)}" width="40" height="30" aria-hidden="true"></canvas>`
    + `<span class="stitle"><b>${escapeHtml(s.title)}</b>${chapter ? `<small>${escapeHtml(chapter.when)}</small>` : ''}</span>`
    + `<span class="spremise">${escapeHtml(firstSentence(s.premise))}</span>`
    + `<span class="sfoot"><span>${(s.roles || []).length} roles${s.coop ? ' · 👥 co-op' : ''}</span>`
    + `<span class="cstars">${stars}</span></span></button>`;
}

/**
 * The co-op button's words for a world two phones play together, or ''.
 * @param {?Object} s
 * @return {string}
 */
export function coopWords(s) {
  const players = (s && s.coop && s.coop.players) || [];
  return players.length === 2 ? `👥 Play together: ${players[0]} + ${players[1]}` : '';
}

