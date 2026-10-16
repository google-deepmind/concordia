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

// Tests of feel.js: `node --test` (run by web/js_test.py).
import assert from 'node:assert/strict';
import {test} from 'node:test';

import {CUES, cueFor, feelIcon, feelLabel, nextFeel} from './feel.js';

const asked = (id) => ({question: {request_id: id}, crises: []});

test('the game decided is cued once, as won or lost', () => {
  assert.equal(cueFor(asked('a'), {outcome: 'victory'}), 'won');
  assert.equal(cueFor({outcome: 'victory'}, {outcome: 'victory'}), null);
  assert.equal(cueFor(asked('a'), {outcome: 'defeat'}), 'lost');
  assert.equal(cueFor(asked('a'), {outcome: 'defeat', spectator: true}), null);
});

test('a crisis at its last turn is cued when it becomes so', () => {
  const hot = {kind: 'band', title: 'Too hot: about 41 ± 2 °C', turns: 1};
  const calm = {...asked('a'), crises: [{...hot, turns: 4}]};
  assert.equal(cueFor(calm, {...asked('b'), crises: [hot]}), 'urgent');
  assert.equal(cueFor({...asked('a'), crises: [hot]}, {...asked('b'), crises: [hot]}), null);
  assert.equal(cueFor(calm, {...asked('b'), crises: [{...hot, turns: null}]}), null);
});

test('the turn coming back is cued only after a real wait', () => {
  const waiting = {question: null, crises: []};
  assert.equal(cueFor(waiting, asked('b'), 5000), 'turn');
  assert.equal(cueFor(waiting, asked('b'), 900), null);  // A quick world: no buzz each turn.
  assert.equal(cueFor(asked('a'), asked('a'), 9000), null);  // Still the same question.
});

test('the toggle cycles and says what it does', () => {
  assert.equal(nextFeel('all'), 'buzz');
  assert.equal(nextFeel('buzz'), 'off');
  assert.equal(nextFeel('off'), 'all');
  assert.equal(feelIcon('buzz'), '📳');
  assert.equal(feelLabel('off'), 'Sound and vibration off');
  assert.ok(CUES.won.notes.length > CUES.turn.notes.length);
});
