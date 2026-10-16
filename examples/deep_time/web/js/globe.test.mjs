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

// Tests of globe.js: `node --test` (run by web/js_test.py).
import assert from 'node:assert/strict';
import {test} from 'node:test';

import {globePixels} from './globe.js';

// Two columns, sea then land; ice on the top and bottom rows.
const thumb = {
  rows: ['ii', 'sl', 'sl', 'ii'],
  legend: [
    {code: 's', name: 'sea', light: [0, 0, 200], dark: [0, 0, 100]},
    {code: 'l', name: 'land', light: [0, 200, 0], dark: [0, 100, 0]},
    {code: 'i', name: 'ice', light: [250, 250, 250], dark: [200, 200, 200]},
  ],
};
const at = (g, x, y) => [...g.data.slice(4 * (y * g.size + x), 4 * (y * g.size + x) + 4)];

test('outside the disc is clear; the middle shows the world', () => {
  const g = globePixels(thumb, 40, 0, 0);
  assert.equal(at(g, 0, 0)[3], 0);
  assert.equal(at(g, 20, 20)[3], 255);
});

test('turning the globe brings the other half round', () => {
  const front = globePixels(thumb, 40, 0, 0), back = globePixels(thumb, 40, 0.5, 0);
  const green = (p) => p[1] > p[2], blue = (p) => p[2] > p[1];
  // Longitude 0 at the centre: the map's first column (sea); half a turn
  // brings its second (land) round.
  assert.ok(blue(at(front, 21, 20)));
  assert.ok(green(at(back, 21, 20)));
});

test('tilting toward the north shows its ice near the top', () => {
  const g = globePixels(thumb, 40, 0, 0.6);
  const top = at(g, 20, 3);
  assert.ok(top[0] > 150 && top[1] > 150 && top[2] > 150);  // Ice, lit.
  assert.equal(globePixels({rows: []}, 10).data.length, 400);
});
