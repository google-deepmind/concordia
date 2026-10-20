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

// Tests of panels.js: `node --test` (run by web/js_test.py).
import assert from 'node:assert/strict';
import {test} from 'node:test';

import {PANELS, panelAt, tabsHtml} from './panels.js';

test('every card has one panel', () => {
  const cards = PANELS.flatMap((p) => p.cards);
  assert.equal(new Set(cards).size, cards.length);
  assert.ok(cards.includes('movescard') && cards.includes('instcard'));
});

test('tabs show the panels with something in them, the active one pressed', () => {
  const html = tabsHtml('moves', {now: true, readings: true, moves: true, story: false, world: true});
  assert.equal((html.match(/role="tab"/g) || []).length, 4);
  assert.match(html, /data-panel="moves" aria-selected="true" class="on">Past moves</);
  assert.doesNotMatch(html, /Story/);
});

test('a swipe comes to rest on the nearest panel', () => {
  const keys = ['now', 'readings', 'moves'];
  assert.equal(panelAt(0, 400, keys), 'now');
  assert.equal(panelAt(390, 400, keys), 'readings');
  assert.equal(panelAt(5000, 400, keys), 'moves');
  assert.equal(panelAt(10, 0, keys), 'now');
});
