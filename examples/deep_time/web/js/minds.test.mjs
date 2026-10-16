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

// Tests of minds.js: `node --test` (run by web/js_test.py).

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {mindsBadge, mindsName, mindsOffer, mindsTable, mindsWords} from './minds.js';

test('each choice of minds says what it costs', () => {
  const offer = {options: ['mock', 'together', 'ollama'], left_usd: 11.2, per_game_usd: 0.4};
  assert.equal(mindsOffer('together', offer), 'about $0.40 a game · $11.20 left');
  assert.match(mindsOffer('mock', offer), /free/);
  assert.match(mindsOffer('ollama', offer), /this Mac/);
});

test('a game says what its minds did and cost', () => {
  const spend = {minds: 'together', label: 'LIVE (a / b)', calls: 212, tokens: 381400, usd: 0.3814};
  assert.equal(mindsName(spend), 'Together');
  assert.equal(mindsWords(spend), 'Minds: Together: 212 calls, 381k tokens, about $0.38');
  assert.match(mindsWords({...spend, label: 'MOCK (fallback: no Together key)'}), /the mock played/);
  assert.equal(mindsBadge(spend), 'Together $0.38');
  assert.equal(mindsBadge({...spend, label: 'MOCK (fallback: Ollama unreachable)', minds: 'ollama'}),
    'Ollama unreachable · mock');
});

test('measured minds say what a game costs, how fast and how often it was won', () => {
  const offer = {left_usd: 1.5, per_game_usd: 0.5, measured: {
    mock: {usd_per_game: 0, seconds_per_turn: 0.05, wins: 0.81, games: 176},
    together: {usd_per_game: 0.2, seconds_per_turn: 3.1, wins: 0.55, games: 11},
    ollama: {usd_per_game: 0, seconds_per_turn: 22.8, wins: 0.67, games: 3},
  }};
  assert.equal(mindsOffer('mock', offer), 'free · instant · won 81% of 176');
  assert.equal(mindsOffer('together', offer), '≈ $0.20 a game · 3 s a turn · won 55% of 11 · $1.50 left');
  assert.equal(mindsOffer('ollama', offer), 'free · 23 s a turn · won 2 of 3');
});


test('the minds read as measured in the picked world, side by side', () => {
  const offer = {
    left_usd: 1.5, per_game_usd: 0.2,
    measured: {mock: {usd_per_game: 0, seconds_per_turn: 0.05, wins: 0.81, games: 176},
      together: {usd_per_game: 0.2, seconds_per_turn: 3.1, wins: 0.55, games: 11}},
    by_scenario: {herds: {
      mock: {usd_per_game: 0, seconds_per_turn: 0.07, wins: 0.5625, games: 16},
      together: {usd_per_game: 0.263, seconds_per_turn: 3.1, wins: 1, games: 3},
      ollama: {usd_per_game: 0, seconds_per_turn: null, wins: 0, games: 1},
    }},
  };
  assert.equal(mindsOffer('together', offer, 'herds'), '≈ $0.26 a game · 3 s a turn · won 3 of 3 here · $1.50 left');
  assert.equal(mindsOffer('mock', offer, 'herds'), 'free · instant · won 56% of 16 here');
  assert.equal(mindsOffer('ollama', offer, 'herds'), 'free · speed unmeasured · won 0 of 1 here');
  assert.equal(mindsOffer('together', offer, 'snowball'), '≈ $0.20 a game · 3 s a turn · won 55% of 11 · $1.50 left');
  const table = mindsTable(offer, 'herds', 'Herds');
  assert.match(table, /<summary>Compare minds in Herds<\/summary>/);
  assert.match(table, /<th scope="row">Together<\/th><td>≈ \$0.26 a game<\/td><td>3 s a turn<\/td><td>3 of 3<\/td>/);
  assert.match(table, /<th scope="row">Mock minds<\/th><td>free<\/td><td>instant<\/td><td>9 of 16<\/td>/);
  assert.equal(mindsTable(offer, 'snowball', 'Snowball'), '');  // Nothing to compare.
});
