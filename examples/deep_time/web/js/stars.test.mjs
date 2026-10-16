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

// Tests of stars.js: `node --test` (run by web/js_test.py).

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {sideHtml, sideWords, starLine, starWords, starsHtml} from './stars.js';

const decided = [
  {text: 'Reach the goal', met: true},
  {text: 'Break the ice by turn 16', met: true},
  {text: 'Break the ice by turn 13', met: false},
];

test('earned stars are gold, missed ones grey, open ones plain', () => {
  const html = starsHtml(decided);
  assert.match(html, /<li class="met"><b>★<\/b>Reach the goal/);
  assert.match(html, /<li class="missed"><b>☆<\/b>Break the ice by turn 13/);
  assert.match(starsHtml([{text: 'Reach the goal', met: null}]), /<li class=""><b>☆<\/b>/);
});

test('a decided game reads its stars at a glance', () => {
  assert.equal(starLine(decided), '★★☆');
  assert.equal(starWords(decided), '2 of 3 stars');
  assert.equal(starWords([]), '0 of 0 stars');
});

const tasks = [
  {key: 'methanogens', text: 'Keep the methanogens on 2% of the planet until turn 20', from: 1, until: 20, reward: 6},
  {key: 'woods', text: 'Raise woods on 4% of the planet between turns 12 and 28', from: 12, until: 28, reward: 4},
];

test('side objectives show their reward and how long they have to go', () => {
  const html = sideHtml(tasks, 16);
  assert.match(html, /<b>◇<\/b>Keep the methanogens on 2% of the planet until turn 20 <span class="reward">\+6 Ω · 4 turns to go\b/);
  assert.match(html, /\+4 Ω · 12 turns to go\b/);
  assert.match(sideHtml(tasks, 5), /\+4 Ω · from turn 12/);
  assert.match(sideHtml(tasks), /\+6 Ω<\/span>/);  // The goal card: no clock.
});

test('a side objective met or missed says so, and the recap adds them up', () => {
  const judged = [{...tasks[0], state: 'met'}, {...tasks[1], state: 'missed'}];
  const html = sideHtml(judged, 30);
  assert.match(html, /<li class="side met"><b>✓<\/b>Keep the methanogens/);
  assert.match(html, /<li class="side missed"><b>✗<\/b>Raise woods/);
  assert.doesNotMatch(html, /to go\b/);
  assert.equal(sideWords(judged), '1 of 2 side objectives met (+6 Ω)');
  assert.equal(sideWords([{...tasks[0], state: 'missed'}]), '0 of 1 side objective met');
  assert.equal(sideWords([]), '');
});
