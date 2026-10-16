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

// Tests of motion.js: `node --test` (run by web/js_test.py).

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {hash01, motionSprites} from './motion.js';

// A small world: land on the left, a shelf, then deep sea.
const map = {
  rows: 4, cols: 8,
  terrain: ['llsddddd', 'hlsddddd', 'llsdddds', 'clsddddd'],
  polities: [{name: 'Amber folk'}], settlements: [[0, 1, 1, 0, 1], [0, 2, 3, 1, 0]],
  site: [2, 6],
};

test('the same turn lays out the same sky; another turn another', () => {
  assert.equal(hash01(3, 4, 5), hash01(3, 4, 5));
  assert.notEqual(hash01(3, 4, 5), hash01(4, 4, 5));
  const one = motionSprites([{motion: 'sprout', colour: [1, 2, 3], move: 'seed white daisies [2 omega]'}], map, 7);
  assert.deepEqual(one, motionSprites([{motion: 'sprout', colour: [1, 2, 3], move: 'seed white daisies [2 omega]'}], map, 7));
  assert.notDeepEqual(one, motionSprites([{motion: 'sprout', colour: [1, 2, 3], move: 'seed white daisies [2 omega]'}], map, 8));
});

test('each motion goes where its move acts', () => {
  const [sprout, iron, marsh, clouds, aimed, works, ring, herd] = motionSprites([
    {motion: 'sprout', colour: [0, 0, 0], move: 'seed white daisies [2 omega]', target: [1, 1]},
    {motion: 'seas', colour: [0, 0, 0], move: 'wash iron into the seas [4 omega]'},
    {motion: 'coasts', colour: [0, 0, 0], move: 'restore the coastal marshes [6 omega]'},
    {motion: 'veil', colour: [0, 0, 0], move: 'seed bright clouds (cools the planet) [4 omega]'},
    {motion: 'veil', colour: [0, 0, 0], move: 'release a dark haze [4 omega]', target: [2, 1]},
    {motion: 'works', colour: [0, 0, 0], move: 'put more labor into: farm the land', people: 'Amber folk'},
    {motion: 'ring', colour: [0, 0, 0], move: 'raise the domes', site: true},
    {motion: 'herd', colour: [0, 0, 0], move: 'thin the wolf packs', herd: 'predator'},
  ], map, 3);
  const onLand = ([r, c]) => 'clh'.includes(map.terrain[Math.floor(r)][Math.floor(c)]);
  assert.equal(sprout.points.length, 18);
  assert.ok(sprout.points.every(onLand), 'sprouts grow on land');
  assert.ok(iron.points.every(([r, c]) => map.terrain[Math.floor(r)][Math.floor(c)] === 's'), 'iron on the shelf seas');
  const besideLand = ([r, c]) => {
    const [rr, cc] = [Math.floor(r), Math.floor(c)], at = (y, x) => (map.terrain[y] || '')[(x + 8) % 8] || '';
    return at(rr, cc) === 's' && [at(rr - 1, cc), at(rr + 1, cc), at(rr, cc - 1), at(rr, cc + 1)].some((t) => 'clh'.includes(t) && t);
  };
  assert.ok(marsh.points.every(besideLand), 'marshes on shelf seas beside land, across the seam too');
  assert.equal(clouds.radius, null);  // Over the whole world.
  assert.equal(clouds.points.length, 14);
  assert.equal(aimed.radius, 5);
  assert.deepEqual(works.towns, [[1, 0, 1], [3, 1, 0]]);
  assert.deepEqual(ring.place, [2, 6]);
  assert.equal(herd.herd, 'predator');
  assert.equal(sprout.label, 'Seed white daisies');
});
