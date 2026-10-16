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

// Tests of plans.js: `node --test` (run by web/js_test.py).
import assert from 'node:assert/strict';
import {test} from 'node:test';

import {heldWords, nextPlanWords, plannedTicks, plansHtml, rewindHtml, whenHtml} from './plans.js';

// As session.py sends them: soonest first.
const plans = [
  {tick: 14, move: 'seed bright clouds (cools the planet) [4 Ω]', in: 2},
  {tick: 16, move: 'dark haze (warms the planet) [4 Ω]', in: 4},
];

test('the When row offers now and the turns ahead, one pressed', () => {
  const html = whenHtml(2);
  assert.equal((html.match(/data-when=/g) || []).length, 6);
  assert.match(html, /data-when="2" aria-pressed="true" class="on" aria-label="In 2 turns">\+2</);
  assert.match(html, /data-when="0" aria-pressed="false" aria-label="Now">Now</);
  assert.equal((whenHtml(0, 3).match(/data-when=/g) || []).length, 4);
});

test('each plan is a chip with its turn and a cancel button', () => {
  const html = plansHtml(plans);
  assert.match(html, /⏱ <b>turn 14<\/b> <span class="muted">\(in 2\)<\/span> Seed bright clouds<button data-cancel="14"/);
  assert.match(html, /data-cancel="16" aria-label="Cancel the plan for turn 16"/);
  assert.equal(plansHtml([]), '');
});

test('the dock names the next plan', () => {
  assert.equal(nextPlanWords(plans), 'Planned: Seed bright clouds in 2 turns (+1 more)');
  assert.equal(nextPlanWords([{...plans[0], in: 1}]), 'Planned: Seed bright clouds in 1 turn');
  assert.equal(nextPlanWords([]), '');
});

test('planned moves are marked on the turn they act', () => {
  assert.deepEqual(plannedTicks({acts_in: 1, planned: [[14, 'bright_clouds'], [16, 'dark_haze']]}), [15, 17]);
  assert.deepEqual(plannedTicks(null), []);
});

test('the rewind button arms on a first tap and says what is left', () => {
  const rewind = {left: 2, can: true, to: 11};
  assert.match(rewindHtml(rewind), /id="rewindbtn" aria-label="↶ Back to turn 11, 2 of 3 left">↶ Back to turn 11 <small>2 of 3 left/);
  assert.match(rewindHtml(rewind, true), /class="rewind armed".*Tap again: back to turn 11/);
  assert.equal(rewindHtml({left: 2, can: false, to: 11}), '');  // No turn kept yet.
  assert.match(rewindHtml({left: 0, can: false, to: 11}), /No rewinds left/);
  assert.equal(rewindHtml(null), '');
});


test('a held course says what came of it, in the player\'s own readings', () => {
  const run = {from: 2, to: 5, turns: 4, read_c: [29.6, 21.2], omega: [6.2, 10.8], life: [0.63, 0.63], stopped: ''};
  assert.equal(heldWords(run),
    'You held turns 2–5: your thermometer 29.6 → 21.2 °C · Ω 6.2 → 10.8 · biodiversity 0.63, steady.');
  assert.equal(heldWords({...run, stopped: 'a crisis needs you'}).slice(-28), 'Stopped: a crisis needs you.');
  assert.equal(heldWords({...run, read_c: [null, null]}).includes('thermometer'), false);  // No reading yet.
  assert.equal(heldWords(null), '');
});
