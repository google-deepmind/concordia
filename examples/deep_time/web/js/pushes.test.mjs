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

// Tests of pushes.js: `node --test` (run by web/js_test.py).

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {adviceTag, laborBrief, strongPush} from './pushes.js';

test('a push of 15 °C or more is strong', () => {
  assert.ok(strongPush('cools about 57 °C next turn, lasting'));
  assert.ok(strongPush('cools about 6 to 16 °C next turn, lasting'));
  assert.ok(!strongPush('cools about 6 °C next turn, fading over a few turns'));
  assert.ok(!strongPush('greens bare land; hardly changes the warmth'));
});

test('a labor chip shows the main good and rising smoke', () => {
  assert.deepEqual(
    laborBrief('iron +4.6, slag +4.4 a generation; ore −7.7, charcoal −6.1; smoke +3.2, slag +4.4'),
    {gain: 'iron +4.6', smoke: 'smoke +3.2'});
  assert.deepEqual(laborBrief('grain +13.5 a generation; research −0.6; smoke −0.4'),
    {gain: 'grain +13.5', smoke: ''});
  assert.deepEqual(laborBrief(''), {gain: '', smoke: ''});
});

test('the Advisor\'s tag says how old its counsel is', () => {
  assert.equal(adviceTag({move: 'seed bright clouds (cools the planet) [4 omega]', age: 3}), 'Advisor · 3 turns old');
  assert.equal(adviceTag({move: 'wait and save omega', age: 1}), 'Advisor · 1 turn old');
  assert.equal(adviceTag({move: 'wait and save omega', age: 0}), 'Advisor');
  assert.equal(adviceTag(null), '');
});
