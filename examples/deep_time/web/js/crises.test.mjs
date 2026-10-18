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

// Tests of crises.js: `node --test` (run by web/js_test.py).
import assert from 'node:assert/strict';
import {test} from 'node:test';

import {askWords, countdown, crisesHtml, crisisKey, crisisMarks, pressingTag, sharedHtml} from './crises.js';

// As session.py sends them (crises.steward_crises), from a watched Daisyworld.
const hot = {kind: 'band', title: 'Too hot: about 43 ± 14 °C', detail: 'Out of the band 10–35 °C; strain 1 of 8 breaks the world.',
             turns: 7, answers: ['seed white daisies [2 omega]', 'seed bright clouds (cools the planet) [4 omega]'], urgency: 3.1};
const coming = {kind: 'band', title: 'Heat coming', detail: 'Your forecast leaves the band by the top as your moves fade.',
                turns: 2, answers: [], urgency: 2.5};

test('countdowns say what is left, or when trouble comes', () => {
  assert.equal(countdown(hot), '7 turns left');
  assert.equal(countdown(coming), 'in 2 turns');
  assert.equal(countdown({kind: 'deadline', title: '1 turn left', turns: 1}), '1 turn left');
  assert.equal(countdown({kind: 'famine', title: 'Cedar league go hungry', turns: null}), '');
  assert.equal(countdown({kind: 'famine', title: 'The granary is nearly empty', turns: 0}), 'now');
});

test('each crisis shows its icon, countdown, detail and answers', () => {
  const html = crisesHtml([hot, coming]);
  assert.match(html, /<div class="crisis band"><span class="cicon" aria-hidden="true">🌡<\/span>/);
  assert.match(html, /<b>Too hot: about 43 ± 14 °C<\/b> <span class="due">7 turns left<\/span>/);
  assert.match(html, /data-move="seed bright clouds \(cools the planet\) \[4 omega\]">Seed bright clouds</);
  assert.match(html, /<div class="crisis band urgent">/);  // Two turns or fewer.
  assert.equal(crisesHtml([]), '');
});

test('crises show where their role can place them on the map', () => {
  const map = {
    rows: 30, cols: 40, site: [12, 30],
    polities: [{name: 'Amber folk'}, {name: 'Cedar league'}],
    settlements: [[0, 1, 5, 6, 1], [1, 2, 20, 33, 0], [1, 3, 22, 35, 1]],
    species: [{kind: 'grazer'}, {kind: 'flock'}],
    herds: [[1, 7, 10, 39, 2], [1, 8, 12, 1, 2], [0, 9, 3, 3, 1]],
  };
  const crises = [
    {kind: 'band', title: 'Too hot: about 43 ± 14 °C', turns: 2, where: {world: 'hot'}},
    {kind: 'species', title: 'Aerobes dwindling', turns: null, where: {lat: -15}},
    {kind: 'unrest', title: 'Smoke sickens the land of the Cedar league', turns: null, where: {people: 'Cedar league'}},
    {kind: 'herds', title: 'Our herds shrink', turns: null, where: {herd: 'flock'}},
    {kind: 'famine', title: 'Stores run low', turns: 3, where: {site: true}},
    {kind: 'deadline', title: '3 turns left', turns: 3, where: null},
  ];
  const [edge, band, ring, herd, site, ...rest] = crisisMarks(crises, map);
  assert.equal(rest.length, 0);  // A deadline is nowhere in particular.
  assert.deepEqual([edge.type, edge.hot, edge.urgent, edge.icon], ['edge', true, true, '🌡']);
  assert.deepEqual([band.type, band.row, band.half], ['band', 17.5, 2.5]);  // 15°S: below the equator.
  assert.deepEqual([ring.type, ring.row, ring.col], ['ring', 22, 35]);  // The capital, not the town.
  assert.equal(herd.type, 'ring');
  assert.equal(herd.row, 11);
  assert.ok(herd.col > 39 || herd.col < 1, `the flocks straddle the seam: ${herd.col}`);
  assert.deepEqual([site.row, site.col], [12, 30]);
  assert.ok(edge.fresh);
  const again = crisisMarks(crises, map, new Set(crises.map(crisisKey)));
  assert.ok(again.every((m) => !m.fresh));  // Shown last turn: no arrival.
  assert.equal(crisisKey({kind: 'band', title: 'Too hot: about 44 °C', where: {world: 'hot'}}), crisisKey(crises[0]));
});

test('a people sees what the age asks of it, and its numbers against the floor', () => {
  const iron = {kind: 'iron_age', good: 'iron', verb: 'forge', label: 'iron forged by all peoples', made: 34, need: 110,
                by: 40, left: 20, pace: 3.8, last: 1.6, floor: 0.4, numbers: 0.55};
  const html = sharedHtml(iron);
  assert.match(html, /iron forged by all peoples/);
  assert.match(html, /34 of 110 by turn 40/);
  assert.match(html, /76 to go in 20 turns: about 3.8 a turn among all peoples/);
  assert.match(html, /; last turn 1.6\.<\/p>/);  // Behind, but far from the end.
  assert.match(sharedHtml({...iron, left: 8}), /last turn <span class="hungry">1.6<\/span>/);
  assert.match(html, /class="small hungry">Our numbers: 55% of the start. A people under 40% breaks/);
  const ahead = sharedHtml({...iron, last: 5, left: 8, numbers: 0.95});
  assert.match(ahead, /last turn 5\./);
  assert.doesNotMatch(ahead, /hungry/);
  assert.doesNotMatch(sharedHtml({...iron, made: 120}), /to go\b/);
  const ship = sharedHtml({kind: 'seed_ship', by: 50, left: 30, min_biodiversity: 0.17, floor: 0.4, numbers: 1});
  assert.match(ship, /A seed ship must leave from a world of biodiversity 0.17 or more by turn 50 \(30 turns left\)/);
  assert.equal(sharedHtml(null), '');
  assert.match(sharedHtml({...iron, next_tech: {label: 'shaft mining', does: 'mine ore +40%', cost: 8, research: 2.5}}),
               /📜 Next for our iron: <b>shaft mining<\/b> \(mine ore \+40%\) · 2.5 of 8 research/);
  const held = sharedHtml({...iron, held: {good: 'iron', short: ['charcoal', 'wood'], recipe: 'cut wood'}});
  assert.match(held, /🔗 Held back: iron ← charcoal ← wood · more hands to <b>cut wood<\/b>/);
  assert.match(sharedHtml({...iron, held: {good: 'iron', short: ['ore'], recipe: null}}), /we make no ore: buy it/);
  assert.match(sharedHtml({...iron, held: {good: 'iron', short: [], recipe: 'smelt iron in a bloomery'}}),
               /🔗 Hardly anyone works to <b>smelt iron in a bloomery<\/b>, so we make little iron/);
});

test('the moves that matter most are tagged', () => {
  assert.equal(pressingTag({kind: 'link', move: 'put more labor into: cut wood'}), '🔗 weakest link');
  assert.equal(pressingTag({kind: 'learn', move: 'learn shaft mining (8 research)'}), '📜 worth learning');
  assert.equal(pressingTag({kind: 'famine', move: 'put more labor into: farm the land'}), '🌾 feeds us');
  assert.equal(pressingTag({kind: 'unrest', move: 'x'}), '🦗 answers it');
  assert.equal(pressingTag({kind: 'mystery', move: 'x'}), '⚠ answers it');
  assert.equal(pressingTag(null), '');
});

test('the sheet heads with the question alone', () => {
  const base = 'You are Basalt clans, a people of the early iron age. Choose where to put more of your labor this generation.';
  assert.equal(askWords(base + ' Most pressing now: The granary is nearly empty; farm answers it. Our weakest link: wood.'), base);
  assert.equal(askWords(base + ' Our weakest link: wood.'), base);
  // A free question that may be aimed lists its moves; the header does not.
  assert.equal(askWords(base + '\n(a) seed white daisies\n(b) wait\nTo aim a move ...'), base);
  assert.equal(askWords(base), base);
  assert.equal(askWords(undefined), '');
});
