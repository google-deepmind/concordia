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

// Tests of util.js: `node --test` (run by web/js_test.py).

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {escapeHtml, fmt, mix, pct, ramp, shortMove, valueNoise} from './util.js';

test('escapes what HTML would read', () => {
  assert.equal(escapeHtml('<b a="1">&</b>'), '&lt;b a=&quot;1&quot;&gt;&amp;&lt;/b&gt;');
});

test('numbers read short', () => {
  assert.equal(fmt(1.25, 1), '1.3');
  assert.equal(fmt(null), '–');
  assert.equal(pct(0.456), '46%');
});

test('a move reads as its words, without its cost or aside', () => {
  assert.equal(shortMove('seed bright clouds (cools the planet) [4 omega]'), 'Seed bright clouds');
});

test('colours mix and ramp within their stops', () => {
  assert.deepEqual(mix([0, 0, 0], [100, 200, 50], 0.5), [50, 100, 25]);
  assert.deepEqual(ramp([[0, 0, 0], [10, 10, 10], [20, 20, 20]], 1), [20, 20, 20]);
  assert.deepEqual(ramp([[0, 0, 0], [10, 10, 10]], -1), [0, 0, 0]);
});

test('noise is smooth, seeded and in [0, 1]', () => {
  for (let i = 0; i < 50; i++) {
    const v = valueNoise(i * 0.37, i * 0.71, 8, 3);
    assert.ok(v >= 0 && v <= 1);
    assert.equal(v, valueNoise(i * 0.37, i * 0.71, 8, 3));
  }
  assert.equal(valueNoise(0.5, 0.5, 4, 1), valueNoise(4.5, 0.5, 4, 1));  // Wraps.
});
