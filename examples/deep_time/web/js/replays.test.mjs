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

// Tests of replays.js: `node --test` (run by web/js_test.py).
import assert from 'node:assert/strict';
import {test} from 'node:test';

import {replayWords, replaysHtml} from './replays.js';

// As /api/campaign lists them (session.list_replays).
const won = {game_id: 'snowball-2-1f', scenario: 'snowball', title: 'Snowball Earth', seed: 2,
             players: ['Gaia'], outcome: 'victory', stars: 2, ticks: 16, when: '2026-10-15 07:10'};

test('a replay is one line: world, ending, stars, length', () => {
  assert.equal(replayWords(won), 'Snowball Earth · victory ★★☆ · 16 turns');
  assert.equal(replayWords({scenario: 'herds', outcome: 'defeat', stars: 0, ticks: 31}), 'herds · defeat ☆☆☆ · 31 turns');
});

test('the start screen lists replays to watch again', () => {
  const html = replaysHtml([won]);
  assert.match(html, /<h2>Watch again<\/h2><button class="replay" data-game="snowball-2-1f">▶ Snowball Earth/);
  assert.match(html, /<small>2026-10-15 07:10<\/small>/);
  assert.equal(replaysHtml([]), '');
});
