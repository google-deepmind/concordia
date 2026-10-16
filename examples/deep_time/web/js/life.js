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

// Life on the map: a tile's species drawn as patches
// in proportion to their cover, not as one averaged colour (white and black
// daisies averaged read a muddy grey).

import {hash2, valueNoise} from './util.js';

/**
 * The species a point of a tile shows, for a uniform draw `u` in [0, 1):
 * walking the cover shares in order. Shares sum to the tile's cover (at
 * most 1); a draw beyond them is bare ground (-1).
 * @param {!Array<number>} shares
 * @param {number} u
 * @return {number}
 */
export function pickShare(shares, u) {
  let reached = 0;
  for (let i = 0; i < shares.length; i++) {
    reached += shares[i];
    if (u < reached) return i;
  }
  return -1;
}

/**
 * A patchy uniform draw for pixel (x, y) of a map `width` pixels wide:
 * cells of `cell` pixels with wobbling edges, so flowers grow in clumps.
 * @param {number} x
 * @param {number} y
 * @param {number} width
 * @param {number} cell
 * @return {number}
 */
export function patchDraw(x, y, width, cell) {
  const scale = cell * 1.5, period = Math.max(2, Math.round(width / scale));
  const jx = 2.2 * cell * (valueNoise(x / scale, y / scale, period, 7) - 0.5);
  const jy = 2.2 * cell * (valueNoise(x / scale, y / scale, period, 11) - 0.5);
  const cells = Math.max(1, Math.round(width / cell));
  const cx = ((Math.floor((x + jx) / cell) % cells) + cells) % cells;
  return hash2(cx, Math.floor((y + jy) / cell));
}

/**
 * When the life on a player's map was seen: their census's
 * turn and age. E.g. 'as your census saw it on turn 6 · 2 turns ago'; ''
 * for the truth (watching, or the game decided).
 * @param {?Object} seen
 * @return {string}
 */
export function seenWords(seen) {
  if (!seen) return '';
  const age = seen.age <= 0 ? 'this turn' : `${seen.age} turn${seen.age === 1 ? '' : 's'} ago`;
  return `as your census saw it on turn ${seen.tick} · ${age}`;
}

/**
 * A latitude's temperature from a census's bands ([[degrees, °C], ...],
 * south first): straight between band centres, flat beyond the outermost.
 * @param {?Array<!Array<number>>} bands
 * @param {number} lat
 * @return {?number}
 */
export function bandTemp(bands, lat) {
  if (!bands || !bands.length) return null;
  if (lat <= bands[0][0]) return bands[0][1];
  for (let i = 1; i < bands.length; i++) {
    const [la, ta] = bands[i - 1], [lb, tb] = bands[i];
    if (lat <= lb) return ta + (tb - ta) * (lat - la) / (lb - la);
  }
  return bands[bands.length - 1][1];
}
