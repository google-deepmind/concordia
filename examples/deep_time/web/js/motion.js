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

// Motion that explains: the player's moves acting this
// turn, laid out on the map in tile coordinates (rows north first), for the
// page to play over the turn's glide. Places are drawn deterministically
// from the turn, so a redraw shows the same sky.

import {shortMove} from './util.js';

const LAND = 'clh', SHELF = 's';

/**
 * A number in [0, 1) from integers (a small, stable hash).
 * @param {number} a
 * @param {number} b
 * @param {number=} c
 * @return {number}
 */
export function hash01(a, b, c = 0) {
  let h = Math.imul(a + 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x632be59b, 0xc2b2ae35) ^ Math.imul(c + 7, 0x27d4eb2f);
  h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d); h ^= h >>> 12;
  return (h >>> 0) / 4294967296;
}

/**
 * Tiles of a class ('clh' land, 's' shelf), or by a test (r, c) -> bool.
 * @param {!Object} map
 * @param {function(number, number, string): boolean} test
 * @return {!Array<!Array<number>>}
 */
function tilesWhere(map, test) {
  const out = [];
  (map.terrain || []).forEach((row, r) => {
    for (let c = 0; c < row.length; c++) if (test(r, c, row[c])) out.push([r, c]);
  });
  return out;
}

/**
 * `n` of `tiles` picked by the turn's hash, near `at` (within `radius`
 * tiles, longitude wrapping) when it was aimed.
 * @param {!Array<!Array<number>>} tiles
 * @param {number} n
 * @param {number} tick
 * @param {number} salt
 * @param {?Array<number>} at
 * @param {number} radius
 * @param {number} cols
 * @return {!Array<!Array<number>>}
 */
function pick(tiles, n, tick, salt, at, radius, cols) {
  const near = at ? tiles.filter(([r, c]) => {
    const dc = Math.min(Math.abs(c - at[1]), cols - Math.abs(c - at[1]));
    return Math.hypot(r - at[0], dc) <= radius;
  }) : tiles;
  const pool = near.length ? near : tiles;
  const out = [];
  for (let i = 0; i < n && pool.length; i++) {
    const [r, c] = pool[Math.floor(hash01(tick, salt, i) * pool.length)];
    out.push([r + hash01(tick, salt + 1, i), c + hash01(tick, salt + 2, i), hash01(tick, salt + 3, i)]);
  }
  return out;
}

/**
 * Each acting move as {motion, colour, label, ...}: points ([row, col,
 * phase]) for particles, a centre and radius for an aimed veil, the place
 * for a ring, the herd kind or the people for theirs.
 * @param {?Array<!Object>} acting
 * @param {!Object} map
 * @param {number} tick
 * @return {!Array<!Object>}
 */
export function motionSprites(acting, map, tick) {
  const cols = map.cols, rows = map.rows;
  const land = tilesWhere(map, (r, c, ch) => LAND.includes(ch));
  const shelf = tilesWhere(map, (r, c, ch) => ch === SHELF);
  const row = (r) => map.terrain[r] || '';
  const coast = shelf.filter(([r, c]) => [[r - 1, c], [r + 1, c], [r, (c + 1) % cols], [r, (c - 1 + cols) % cols]]
    .some(([rr, cc]) => LAND.includes(row(rr)[cc] || '')));
  return (acting || []).map((a, i) => {
    const salt = 17 * (i + 1), at = a.target || null;
    const base = {motion: a.motion, colour: a.colour, label: shortMove(a.move || ''), at};
    switch (a.motion) {
      case 'veil':  // The sky changes: over the world, or a circle where aimed.
        return {...base, radius: at ? 5 : null, points: at
          ? pick(land.concat(shelf), 10, tick, salt, at, 5, cols)
          : Array.from({length: 14}, (_, j) => [hash01(tick, salt, j) * rows, hash01(tick, salt + 1, j) * cols, hash01(tick, salt + 2, j)])};
      case 'sprout': return {...base, points: pick(land, 18, tick, salt, at, 3.5, cols)};
      case 'rain': case 'dust':
        return {...base, points: pick(land, 40, tick, salt, at, 5, cols)};
      case 'embers': return {...base, points: pick(land, 6, tick, salt, at, 6, cols)};
      case 'seas': return {...base, points: pick(coast.length ? coast : shelf, 16, tick, salt, at, 8, cols)};
      case 'coasts': return {...base, points: pick(coast, 18, tick, salt, at, 8, cols)};
      case 'gather': return {...base, points: pick(land, 16, tick, salt, at, 6, cols), to: [rows * 0.06, cols / 2]};
      case 'ring': {
        const site = a.site && map.site ? map.site : null;
        return {...base, place: at || site, lat: a.lat == null ? null : a.lat};
      }
      case 'herd': return {...base, herd: a.herd};
      case 'works': {
        const p = (map.polities || []).findIndex((x) => x.name === a.people);
        const towns = (map.settlements || []).filter((s) => s[0] === p).map((s) => [s[2], s[3], s[4]]);
        return {...base, people: a.people, towns};
      }
      default: return {...base};  // 'sweep': the whole map.
    }
  });
}
