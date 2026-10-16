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

// Tests of life.js: `node --test` (run by web/js_test.py).

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {bandTemp, patchDraw, pickShare, seenWords} from './life.js';

test('a point shows each species in proportion to its cover', () => {
  assert.equal(pickShare([0.5, 0.3], 0.1), 0);
  assert.equal(pickShare([0.5, 0.3], 0.6), 1);
  assert.equal(pickShare([0.5, 0.3], 0.9), -1);  // Bare ground.
  assert.equal(pickShare([], 0.2), -1);
});

test('patches share out a mixed tile fairly and in clumps', () => {
  const counts = [0, 0, 0];
  let same = 0, last = null;
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 384; x++) {
      const s = pickShare([0.45, 0.35], patchDraw(x, y, 384, 4));
      counts[s + 1]++;
      if (s === last) same++;
      last = s;
    }
  }
  const n = 64 * 384;
  assert.ok(Math.abs(counts[1] / n - 0.45) < 0.05, `white ${counts[1] / n}`);
  assert.ok(Math.abs(counts[2] / n - 0.35) < 0.05, `black ${counts[2] / n}`);
  assert.ok(Math.abs(counts[0] / n - 0.20) < 0.05, `bare ${counts[0] / n}`);
  assert.ok(same / n > 0.6, 'clumped, not salt and pepper');
  assert.equal(patchDraw(5, 9, 384, 4), patchDraw(5, 9, 384, 4));
});

test('a player\'s life map says when their census saw it', () => {
  assert.equal(seenWords({tick: 6, age: 2}), 'as your census saw it on turn 6 · 2 turns ago');
  assert.equal(seenWords({tick: 9, age: 1}), 'as your census saw it on turn 9 · 1 turn ago');
  assert.equal(seenWords({tick: 9, age: 0}), 'as your census saw it on turn 9 · this turn');
  assert.equal(seenWords(null), '');  // The truth: no age to tell.
});

test('a census\'s band temperatures shade every latitude', () => {
  const bands = [[-45, 20], [-15, 28], [15, 30], [45, 22]];
  assert.equal(bandTemp(bands, -15), 28);
  assert.equal(bandTemp(bands, 0), 29);  // Halfway between 15°S and 15°N.
  assert.equal(bandTemp(bands, -80), 20);  // Beyond the last band: its own.
  assert.equal(bandTemp(bands, 80), 22);
  assert.equal(bandTemp(null, 0), null);
});
