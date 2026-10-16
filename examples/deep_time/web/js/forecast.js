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

// The Steward's forecast: its late readings reckoned forward
// with the moves it made since, drawn as a cone past the last reading.

import {escapeHtml, shortMove} from './util.js';

// Turns of the course drawn past the cone, at least.
export const COURSE_TURNS = 3;

/**
 * Turns of the course to draw past the cone: three, or on to just past the
 * last planned move, as far as the course goes.
 * @param {?Object} f
 * @return {number}
 */
export function courseTurns(f) {
  if (!f) return 0;
  const room = Math.max(0, (f.course || []).length - f.acts_in);
  const last = Math.max(0, ...(f.planned || []).map(([tick]) => tick - f.tick + 1));
  return Math.min(room, Math.max(COURSE_TURNS, last));
}

/**
 * The lowest and highest temperatures the forecast draws, for the chart.
 * @param {?Object} f
 * @return {!Array<number>}
 */
export function forecastRange(f) {
  if (!f) return [];
  const course = (f.course || []).slice(f.acts_in, f.acts_in + courseTurns(f));
  return [f.now - f.now_spread, f.now + f.now_spread,
          f.ahead - f.ahead_spread, f.ahead + f.ahead_spread, ...course];
}

/**
 * The cone and the course as SVG points, given the chart's scales (X of a
 * tick, Y of a temperature): the likely range now widening to the turn a
 * move acts, and past it the course as the moves made fade (one line).
 * @param {?Object} f
 * @param {function(number): number} X
 * @param {function(number): number} Y
 * @return {?Object}
 */
export function coneShape(f, X, Y) {
  if (!f) return null;
  const at = (tick, value) => `${X(tick).toFixed(1)},${Y(value).toFixed(1)}`;
  const t0 = f.tick, t1 = f.tick + f.acts_in;
  const cone = [at(t0, f.now + f.now_spread), at(t1, f.ahead + f.ahead_spread),
                at(t1, f.ahead - f.ahead_spread), at(t0, f.now - f.now_spread)].join(' ');
  const mid = [at(t0, f.now), at(t1, f.ahead)].join(' ');
  const later = (f.course || []).slice(f.acts_in, f.acts_in + courseTurns(f));
  const course = later.length
    ? [at(t1, f.ahead), ...later.map((v, i) => at(t1 + i + 1, v))].join(' ') : '';
  return {cone, mid, course};
}

/**
 * The dead reckoning on the chart: an arrow from the last reading down (or
 * up) to the forecast of now, the moves' effect the readings have not shown
 * yet; null when it is under half a degree.
 * @param {?Object} f
 * @param {?number} lastSeen
 * @param {function(number): number} X
 * @param {function(number): number} Y
 * @return {?Object}
 */
export function reckonShape(f, lastSeen, X, Y) {
  if (!f || lastSeen == null || Math.abs(f.unseen) < 0.5) return null;
  const x = X(f.tick), y0 = Y(lastSeen), y1 = Y(f.now);
  const head = y1 > y0 ? -4 : 4;  // The arrowhead points along the arrow.
  return {
    line: `${x.toFixed(1)},${y0.toFixed(1)} ${x.toFixed(1)},${y1.toFixed(1)}`,
    head: `M${(x - 4).toFixed(1)},${(y1 + head).toFixed(1)} L${x.toFixed(1)},${y1.toFixed(1)}`
      + ` L${(x + 4).toFixed(1)},${(y1 + head).toFixed(1)}`,
  };
}

/**
 * The chart's key for the forecast, e.g. 'forecast 17 ± 2 °C now, 15 ± 3
 * next turn'.
 * @param {?Object} f
 * @return {string}
 */
export function forecastKey(f) {
  if (!f) return '';
  const turn = f.acts_in === 1 ? 'next turn' : `in ${f.acts_in} turns`;
  return `forecast ${Math.round(f.now)} ± ${Math.round(f.now_spread)} °C now,`
    + ` ${Math.round(f.ahead)} ± ${Math.round(f.ahead_spread)} ${turn}`;
}

/**
 * What the moves made since the last reading have done, which the readings
 * do not show yet (empty when it is under half a degree).
 * @param {?Object} f
 * @return {string}
 */
export function unseenWords(f) {
  if (!f || Math.abs(f.unseen) < 0.5) return '';
  const sign = f.unseen > 0 ? '+' : '−';
  return `your moves since the last reading: ${sign}${Math.abs(f.unseen).toFixed(0)} °C, not read yet`;
}

/**
 * When the course leaves the band, if it does within the forecast.
 * @param {?Object} f
 * @return {string}
 */
export function leavesWords(f) {
  if (!f || f.leaves_in == null) return '';
  const turns = f.leaves_in === 1 ? 'turn' : 'turns';
  return `as your moves fade it leaves the band by the ${f.edge} in about ${f.leaves_in} ${turns}`;
}

/**
 * Where the course stands once the planned moves act, e.g. 'with your plan,
 * about 14 °C on turn 13' (the last plan's turn).
 * @param {?Object} f
 * @return {string}
 */
export function planCourseWords(f) {
  const planned = (f && f.planned) || [];
  if (!planned.length) return '';
  const acts = Math.max(...planned.map(([tick]) => tick)) + f.acts_in;
  const value = (f.course || [])[acts - f.tick - 1];
  if (value == null) return '';
  const plan = planned.length === 1 ? 'your plan' : 'your plans';
  return `with ${plan}, about ${Math.round(value)} °C on turn ${acts}`;
}

/**
 * After the game: how far off the forecasts were, against the raw readings
 * (`whose`: 'Your' for the Steward's player, else e.g. "The Steward's").
 * @param {?Object} r
 * @param {string=} whose
 * @return {string}
 */
export function trackWords(r, whose = 'Your') {
  if (!r || r.now_error_c == null) return '';
  const better = r.now_error_c < r.raw_error_c;
  return `${whose} forecasts of the planet now were off by ${r.now_error_c.toFixed(1)} °C on average,`
    + ` the raw readings by ${r.raw_error_c.toFixed(1)} °C${better ? '' : ' (the forecast did not help here)'};`
    + ` ${Math.round(r.inside_share * 100)}% of the true temperatures fell inside its range.`;
}

/**
 * A move's what-if: what the Steward's forecast says the
 * move brings, from its own late readings. E.g. {text: 'forecast: about
 * 12 ± 4 °C next turn', warn: false} or {text: 'forecast: about −33 ± 19 °C
 * next turn · too cold at once', warn: true}.
 * @param {?Object} w
 * @return {!Object}
 */
export function whatIfWords(w) {
  if (!w) return {text: '', warn: false};
  const spread = w.spread ? ` ± ${Math.round(w.spread)}` : '';
  let text = `forecast: about ${Math.round(w.then)}${spread} °C next turn`, warn = false;
  if (w.leaves_in != null) {
    warn = true;
    const side = w.edge === 'top' ? 'too hot' : 'too cold';
    text += w.leaves_in <= 1 ? ` · ${side} at once` : ` · ${side} in ${w.leaves_in} turns`;
  } else if (w.returns_in != null) {
    text += ` · back in the band in ${w.returns_in} turn${w.returns_in === 1 ? '' : 's'}`;
  }
  return {text: text.replace('-', '−'), warn};
}

/**
 * A move that takes the world out of its band within two turns, by the
 * forecast, asks for a second tap: the first says what it will do.
 * @param {?Object} w
 * @return {boolean}
 */
export function asksTwice(w) {
  return !!w && w.leaves_in != null && w.leaves_in <= 2;
}

// The recap's line, told: what the player's thermometer
// read (faint dots, late and noisy against the truth) and where each move
// acted, ▲ warming, ▼ cooling, ● other. `X` maps a tick, `Y` a °C; `top`
// is where the marks sit. Returns SVG, and the key in words.
export const MARK_WAYS = {warm: ['▲', '#c2412d'], cool: ['▼', '#2b6cb0'], other: ['●', '#6b7280']};

/**
 * Renders the recap's thermometer readings and move markers as SVG.
 * @param {!Object} recap
 * @param {function(number): number} X
 * @param {function(number): number} Y
 * @param {number} top
 * @return {string}
 */
export function recapMarksSvg(recap, X, Y, top) {
  let svg = '';
  for (const [tick, v] of recap.seen || []) {
    svg += `<circle cx="${X(tick).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="1.7" fill="var(--muted)" opacity=".55"/>`;
  }
  for (const [tick, move, way] of recap.marks || []) {
    const [glyph, colour] = MARK_WAYS[way] || MARK_WAYS.other, x = X(tick).toFixed(1);
    svg += `<line x1="${x}" x2="${x}" y1="${top + 6}" y2="${Y(Math.min(...(recap.temps || [0])))}" stroke="${colour}" stroke-width="1" opacity=".35"/>`
      + `<text x="${x}" y="${top + 5}" text-anchor="middle" font-size="8" fill="${colour}"><title>${escapeHtml(`turn ${tick}: ${move}`)}</title>${glyph}</text>`;
  }
  return svg;
}

/**
 * Returns the legend key for the recap's move markers.
 * @param {!Object} recap
 * @return {string}
 */
export function recapKey(recap) {
  const ways = new Set((recap.marks || []).map((m) => m[2]));
  const parts = ['warm', 'cool', 'other'].filter((w) => ways.has(w))
    .map((w) => `${MARK_WAYS[w][0]} ${w === 'warm' ? 'a move that warmed' : w === 'cool' ? 'a move that cooled' : 'another move'}`);
  if ((recap.seen || []).length) parts.push('· your thermometer');
  return parts.join('  ');
}

// The game's turning points in the recap: where the world
// swung hardest, and the move of the player's that came before, with what
// they read, what their forecast said and what was true when they moved.
const TURN_WAYS = {left: 'left the band', across: 'went across the band', back: 'came back into the band', swing: ''};

/**
 * Formats a degree reading with a minus sign or em-dash.
 * @param {?number} v
 * @return {string}
 */
const deg = (v) => (v == null ? '—' : `${v < 0 ? '−' : ''}${Math.abs(Math.round(v * 10) / 10)}`);

/**
 * Renders the game's turning points in the recap as an HTML list.
 * @param {?Array<!Object>} points
 * @return {string}
 */
export function turningHtml(points) {
  if (!points || !points.length) return '';
  return '<ul class="turning">' + points.map((p) => {
    const d = Math.round((p.to_c - p.from_c) * 10) / 10;
    const way = TURN_WAYS[p.way] || '';
    const bad = p.way === 'left' || p.way === 'across';
    const head = `<b>Turns ${p.from}–${p.to}: ${deg(p.from_c)} → ${deg(p.to_c)} °C</b> `
      + `<span class="${bad ? 'hungry' : 'muted'}">(${d > 0 ? '+' : ''}${deg(d)} °C${way ? `, ${way}` : ''})</span>`;
    const body = p.move
      ? `${escapeHtml(shortMove(p.move))} on turn ${p.move_tick}: you read ${deg(p.read_c)} °C`
        + (p.forecast_c != null ? `, your forecast said ${deg(p.forecast_c)}` : '')
        + `; it was ${deg(p.true_c)} °C.`
      : `It came while you waited (you read ${deg(p.read_c)} °C).`;
    return `<li>${head}<br>${body}</li>`;
  }).join('') + '</ul>';
}
