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

// Tests of forecast.js: `node --test` (run by web/js_test.py).
import assert from 'node:assert/strict';
import {test} from 'node:test';

import {asksTwice, coneShape, courseTurns, forecastKey, forecastRange, leavesWords, planCourseWords, recapKey, recapMarksSvg, reckonShape, trackWords, turningHtml, unseenWords, whatIfWords} from './forecast.js';

// As session.py sends it (forecast.Forecast.to_dict) after a cloud seeding.
const cooled = {
  tick: 12, now: 17.2, now_spread: 2.4, ahead: 19.6, ahead_spread: 3.1, acts_in: 1,
  unseen: -4.2, readings: 4, course: [19.6, 21.3, 22.2, 22.8, 23.2, 23.4],
  leaves_in: 3, edge: 'top',
};

test('the cone widens from now to the turn a move acts, then the course runs on', () => {
  const X = (tick) => tick * 10, Y = (v) => 100 - v;
  const shape = coneShape(cooled, X, Y);
  assert.equal(shape.cone, '120.0,80.4 130.0,77.3 130.0,83.5 120.0,85.2');
  assert.equal(shape.mid, '120.0,82.8 130.0,80.4');
  assert.equal(shape.course, '130.0,80.4 140.0,78.7 150.0,77.8 160.0,77.2');
  assert.equal(coneShape(null, X, Y), null);
});

test('an arrow carries the last reading to the forecast of now', () => {
  const X = (tick) => tick * 10, Y = (v) => 100 - v;
  const arrow = reckonShape(cooled, 21.4, X, Y);
  assert.equal(arrow.line, '120.0,78.6 120.0,82.8');
  assert.equal(arrow.head, 'M116.0,78.8 L120.0,82.8 L124.0,78.8');  // Pointing down.
  assert.equal(reckonShape({...cooled, unseen: 0.2}, 21.4, X, Y), null);
  assert.equal(reckonShape(cooled, null, X, Y), null);
});

test('the course is drawn on past the last planned move', () => {
  assert.equal(courseTurns(cooled), 3);
  assert.equal(courseTurns({...cooled, planned: [[16, 'dark_haze']]}), 5);  // Acts at 17.
  assert.equal(courseTurns({...cooled, planned: [[13, 'dark_haze']]}), 3);
  const X = (tick) => tick * 10, Y = (v) => 100 - v;
  const shape = coneShape({...cooled, planned: [[16, 'dark_haze']]}, X, Y);
  assert.equal(shape.course.split(' ').length, 6);
  assert.equal(courseTurns(null), 0);
});

test('the course says where the plans leave the planet', () => {
  assert.equal(planCourseWords({...cooled, planned: [[13, 'bright_clouds']]}), 'with your plan, about 21 °C on turn 14');
  assert.equal(planCourseWords({...cooled, planned: [[13, 'a'], [14, 'b']]}), 'with your plans, about 22 °C on turn 15');
  assert.equal(planCourseWords(cooled), '');
  assert.equal(planCourseWords({...cooled, planned: [[30, 'a']]}), '');  // Past the course.
});

test('the chart makes room for the whole forecast', () => {
  const range = forecastRange(cooled);
  assert.equal(Math.min(...range).toFixed(1), '14.8');
  assert.equal(Math.max(...range).toFixed(1), '22.8');
  assert.deepEqual(forecastRange(null), []);
});

test('the forecast in words: now, next turn, the unseen, the band', () => {
  assert.equal(forecastKey(cooled), 'forecast 17 ± 2 °C now, 20 ± 3 next turn');
  assert.equal(unseenWords(cooled), 'your moves since the last reading: −4 °C, not read yet');
  assert.equal(unseenWords({...cooled, unseen: 0.3}), '');
  assert.equal(leavesWords(cooled), 'as your moves fade it leaves the band by the top in about 3 turns');
  assert.equal(leavesWords({...cooled, leaves_in: null}), '');
  assert.equal(forecastKey({...cooled, acts_in: 3}), 'forecast 17 ± 2 °C now, 20 ± 3 in 3 turns');
});

test('after the game, the track record against the raw readings', () => {
  const r = {turns: 30, raw_error_c: 2.3, now_error_c: 1.6, ahead_error_c: 2.1, inside_share: 0.83};
  assert.equal(trackWords(r), 'Your forecasts of the planet now were off by 1.6 °C on average,'
    + ' the raw readings by 2.3 °C; 83% of the true temperatures fell inside its range.');
  assert.match(trackWords({...r, now_error_c: 2.5}), /did not help here/);
  assert.match(trackWords(r, "The Steward's"), /^The Steward's forecasts of the planet now/);
  assert.equal(trackWords(null), '');
});

test('a move says what the forecast says it brings, and a wreck asks twice', () => {
  const clouds = {then: 12.3, spread: 4.2, leaves_in: null, edge: '', returns_in: null};
  assert.deepEqual(whatIfWords(clouds), {text: 'forecast: about 12 ± 4 °C next turn', warn: false});
  assert.equal(asksTwice(clouds), false);
  const ash = {then: -32.8, spread: 18.6, leaves_in: 1, edge: 'bottom', returns_in: null};
  assert.deepEqual(whatIfWords(ash), {text: 'forecast: about −33 ± 19 °C next turn · too cold at once', warn: true});
  assert.equal(asksTwice(ash), true);
  const later = {then: 30, spread: 3, leaves_in: 4, edge: 'top', returns_in: null};
  assert.equal(whatIfWords(later).text, 'forecast: about 30 ± 3 °C next turn · too hot in 4 turns');
  assert.equal(asksTwice(later), false);  // A warning, not a wall.
  const back = {then: 36, spread: 3, leaves_in: null, edge: '', returns_in: 2};
  assert.equal(whatIfWords(back).text, 'forecast: about 36 ± 3 °C next turn · back in the band in 2 turns');
  assert.deepEqual(whatIfWords(null), {text: '', warn: false});
});

test('the recap marks where each move acted and what the thermometer read', () => {
  const recap = {temps: [10, 12, 15, 14], seen: [[1, 11.5], [2, 13]],
                 marks: [[2, 'Release a dark haze', 'warm'], [3, 'Seed white daisies', 'other']]};
  const svg = recapMarksSvg(recap, (t) => t * 10, (v) => 100 - v, 4);
  assert.match(svg, /<circle cx="10.0" cy="88.5"/);  // A reading, where it was read.
  assert.match(svg, /fill="#c2412d"><title>turn 2: Release a dark haze<\/title>▲<\/text>/);
  assert.match(svg, /<title>turn 3: Seed white daisies<\/title>●/);
  assert.equal(recapKey(recap), '▲ a move that warmed  ● another move  · your thermometer');
  assert.equal(recapKey({marks: [], seen: []}), '');
});

test('the recap tells the turning points and what was read before them', () => {
  const html = turningHtml([
    {from: 28, to: 30, from_c: 35.4, to_c: -42.2, way: 'across', move: 'bury the sky in ash (a volcanic winter, for years) [6 omega]',
     move_tick: 29, read_c: 46.0, forecast_c: 35.4, true_c: 24.2},
    {from: 21, to: 23, from_c: 19.3, to_c: 34.6, way: 'swing', move: null, read_c: 17.4},
  ]);
  assert.match(html, /Turns 28–30: 35.4 → −42.2 °C<\/b> <span class="hungry">\(−77.6 °C, went across the band\)/);
  assert.match(html, /Bury the sky in ash on turn 29: you read 46 °C, your forecast said 35.4; it was 24.2 °C\./);
  assert.match(html, /<span class="muted">\(\+15.3 °C\)<\/span>/);
  assert.match(html, /It came while you waited \(you read 17.4 °C\)/);
  assert.equal(turningHtml([]), '');
  assert.equal(turningHtml(null), '');
});
