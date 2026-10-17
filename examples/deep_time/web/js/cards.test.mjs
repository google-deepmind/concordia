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

// Tests of cards.js: `node --test` (run by web/js_test.py).
import assert from 'node:assert/strict';
import {test} from 'node:test';

import {coopWords, firstSentence, scenarioCardHtml, thumbPixels} from './cards.js';

// As /api/thumb sends it: biome codes per tile, north first, and colours.
const thumb = {
  rows: ['~~g', '~fd'],
  legend: [
    {code: '~', name: 'deep ocean', light: [20, 40, 90], dark: [10, 20, 50]},
    {code: 'g', name: 'grassland', light: [150, 180, 90], dark: [80, 100, 50]},
    {code: 'f', name: 'forest', light: [40, 110, 50], dark: [20, 60, 30]},
  ],
};

test('the map becomes a pixel a tile, in the light or dark colours', () => {
  const px = thumbPixels(thumb);
  assert.deepEqual([px.w, px.h], [3, 2]);
  assert.deepEqual([...px.data.slice(8, 12)], [150, 180, 90, 255]);  // North-east grass.
  assert.deepEqual([...px.data.slice(20, 24)], [128, 128, 128, 255]);  // An unknown code.
  assert.deepEqual([...thumbPixels(thumb, true).data.slice(0, 4)], [10, 20, 50, 255]);
});

test('a card shows the world, its age, its premise and its stars', () => {
  const s = {key: 'archipelago', title: 'Archipelago', premise: 'A warm sea strewn with islands. Green them.',
             roles: [{name: 'Gaia'}, {name: 'Grasslands'}, {name: 'Forests'}]};
  const html = scenarioCardHtml(s, {when: '9,000 years ago', stars: 2, open: true}, true);
  assert.match(html, /class="scard on" data-scenario="archipelago" aria-pressed="true"/);
  assert.match(html, /<b>Archipelago<\/b><small>9,000 years ago<\/small>/);
  assert.match(html, /<span class="spremise">A warm sea strewn with islands.<\/span>/);
  assert.match(html, /3 roles.*★★☆/);
  // Playable from here: a chapter not yet opened shows no lock and no stars.
  const unopened = scenarioCardHtml(s, {when: 'x', stars: 0, open: false});
  assert.match(unopened, /class="scard" data-scenario="archipelago"/);
  assert.doesNotMatch(unopened, /🔒|☆/);
  assert.equal(firstSentence('One. Two.'), 'One.');
});

test('a world for two phones says so, and how to play it together', () => {
  const s = {key: 'light_delay', title: 'Light Delay', premise: 'A colony.', roles: [{}, {}],
             coop: {players: ['Mission Control', 'Colonists of New Ararat'], text: 'Two phones.'}};
  assert.match(scenarioCardHtml(s, null), /2 roles · 👥 co-op/);
  assert.equal(coopWords(s), '👥 Play together: Mission Control + Colonists of New Ararat');
  assert.equal(coopWords({key: 'x'}), '');
});

