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

import assert from 'node:assert/strict';
import test from 'node:test';

import {omenHtml, omenResponse, omenWords} from './omens.js';

const OMEN = {option: 'speak an omen [2 omega]', signs: ['a red dawn', 'a comet', 'ash on the snow'],
              max_words: 25, cost: 2, ready: true};

test('the composer offers each sign, the first picked, and a box for the words', () => {
  const html = omenHtml(OMEN, 5);
  assert.match(html, /Speak an omen · 2 Ω/);
  assert.match(html, /<button class="sign on" data-sign="a red dawn" aria-pressed="true">a red dawn<\/button>/);
  assert.match(html, /data-sign="ash on the snow" aria-pressed="false"/);
  assert.match(html, /up to 25 words/);
  assert.doesNotMatch(html, /\bdisabled\b/);
  assert.match(html, /Or a move<\/p>$/);
  assert.equal(omenHtml(null), '');
});

test('an omen that cannot be spoken says why', () => {
  const poor = omenHtml({...OMEN, ready: false}, 1.26);
  assert.match(poor, /It needs 2 Ω; you have 1\.2\./);
  assert.match(poor, /id="omensend" disabled/);
  assert.match(omenHtml({...OMEN, ready: false}, 9), /still quiet from the last one/);
});

test('the words are cut to the limit and sent after the sign', () => {
  assert.equal(omenWords('  burn   less\ncoal now ', 3), 'burn less coal');
  assert.equal(omenWords('', 25), '');
  assert.equal(omenResponse('a comet', 'spend it slowly'), 'a comet: spend it slowly');
  assert.equal(omenResponse('a comet', ''), 'a comet:');
});
