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

// Small pure helpers of the page: escaping, numbers, colours, noise.

/**
 * Escapes HTML special characters in `text`.
 * @param {*} text
 * @return {string}
 */
export function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
}

/**
 * Strips the trailing bracketed cost from a move option.
 * @param {string} option
 * @return {string}
 */
export const moveWords = (option) => option.replace(/\s*\[[^\]]*\]\s*$/, '');  // Drop the cost.

/**
 * Returns the CSS freshness class for a reading of `age` turns.
 * @param {number} age
 * @return {string}
 */
export const ageClass = (age) => (age === 0 ? 'fresh' : age <= 2 ? 'late' : 'stale');

/**
 * Formats a reading's age in turns.
 * @param {number} age
 * @return {string}
 */
export const ageWords = (age) => (age === 0 ? 'fresh' : `${age} turn${age === 1 ? '' : 's'} old`);

/**
 * Formats a fraction in [0, 1] as a rounded percentage string.
 * @param {number} x
 * @return {string}
 */
export const pct = (x) => `${Math.round(100 * x)}%`;

/**
 * Formats a number to `d` decimal places, or '–' when nullish.
 * @param {?number|undefined} x
 * @param {number=} d
 * @return {string}
 */
export const fmt = (x, d = 1) => (x == null ? '–' : Number(x).toFixed(d));

/**
 * Returns the active color scheme ('dark' or 'light').
 * @return {string}
 */
export const themeNow = () => (typeof matchMedia === 'function'
  && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

/**
 * Linearly interpolates between two RGB arrays `a` and `b` by `t` in [0, 1].
 * @param {!Array<number>} a
 * @param {!Array<number>} b
 * @param {number} t
 * @return {!Array<number>}
 */
export function mix(a, b, t) {
  t = Math.max(0, Math.min(1, t));
  return a.map((x, i) => Math.round(x + (b[i] - x) * t));
}

/**
 * Formats an RGB triplet and optional alpha as a CSS rgb/rgba string.
 * @param {!Array<number>} c
 * @param {number=} a
 * @return {string}
 */
export const rgb = (c, a = 1) => a === 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${a})`;

/**
 * Samples a multi-stop RGB color ramp at `t` in [0, 1].
 * @param {!Array<!Array<number>>} stops
 * @param {number} t
 * @return {!Array<number>}
 */
export function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  return mix(stops[i], stops[i + 1], t - i);
}

/**
 * Hashes integer coordinates `(x, y)` to a number in [0, 1).
 * @param {number} x
 * @param {number} y
 * @return {number}
 */
export function hash2(x, y) {
  let h = Math.imul(x, 374761393) + Math.imul(y, 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

/**
 * Smooth value noise on a unit lattice, wrapping every `period` cells in x.
 * @param {number} x
 * @param {number} y
 * @param {number} period
 * @param {number} seed
 * @return {number}
 */
export function valueNoise(x, y, period, seed) {
  // Smooth noise on a unit lattice, wrapping every `period` cells in x.
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const h = (i, j) => hash2((((i % period) + period) % period) + seed * 7919, j + seed * 104729);
  const a = h(x0, y0), b = h(x0 + 1, y0), c = h(x0, y0 + 1), d = h(x0 + 1, y0 + 1);
  return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
}

/**
 * Quadratic ease-in-out on `k` in [0, 1].
 * @param {number} k
 * @return {number}
 */
export const easeInOut = (k) => (k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2);

export const NEUTRAL_MOVES = /^(wait and save omega|keep the current balance of labor|hold steady|listen and wait|browse lightly)$/;

/**
 * Shortens and capitalizes a move string for UI labels.
 * @param {string} move
 * @return {string}
 */
export const shortMove = (move) => {
  const words = moveWords(move).replace(/\s*\([^)]*\)\s*$/, '');
  return words.charAt(0).toUpperCase() + words.slice(1);
};
