/**
 * @license
 * Copyright 2026 DeepMind Technologies Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @fileoverview Tests of the page's pure helpers: node --test. */

'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {
  askMove, cameBy, chartSeries, crowdedSpot, domainsFor, kinName, leverChoices,
  peopleColour, releaseMove, appendNew, entryMeta, entryShown, fateChoices,
  fateMove, reachNote, recentDisasters, releaseStatus, savedNote,
  transmissionText, transmissionTree, twinNote, variantNote, withinReach,
} = require('./app.js');

test('a release is written in the form the World reads', () => {
  assert.equal(
      releaseMove('belief', 'sacred', 'song', ' 8, 9 ', ' The moon is kind. '),
      'release belief sacred as a song at 8,9: The moon is kind.');
});

test('a people keeps its colour', () => {
  assert.equal(peopleColour('Hill people'), peopleColour('Hill people'));
  assert.notEqual(peopleColour('Hill people'), peopleColour('River people'));
  assert.match(peopleColour('River people'), /^hsl\(\d+, 65%, 40%\)$/);
});

test('a question is written in the form the World reads', () => {
  assert.equal(askMove('Ada', ' Why share? '), 'ask Ada: Why share?');
});

test('the footer says where a world was restored and last saved', () => {
  assert.equal(savedNote({saved_year: null, restored_year: null}), '');
  assert.equal(savedNote({saved_year: 3, restored_year: null}),
      '· saved at year 3');
  assert.equal(savedNote({saved_year: 9, restored_year: 0}),
      '· restored at year 0, saved at year 9');
});

test('the page says what a twin is and where it is', () => {
  assert.equal(twinNote(null), 'No twin.');
  assert.equal(
      twinNote({lever: 'richness', value: 0.8, year: 40, year_now: 52,
        people: 61, running: true, error: ''}),
      'The twin (richness 0.8, forked in year 40) is in year 52: 61 people.');
  assert.equal(
      twinNote({lever: 'without', value: 'story', year: 3, year_now: 3,
        people: 12, running: false, error: ''}),
      'The twin (without story, forked in year 3) is in year 3: 12 people' +
      ' (stopped).');
});

test('the fork form offers each lever, then each channel to go without', () => {
  assert.deepEqual(
      leverChoices({levers: {richness: [0.05, 1, 0.2]}, channels: ['story']}),
      [['richness', 'richness (0.05–1; now 0.2)'],
        ['without story', 'without story']]);
});

test('a custom says how it came and from whom', () => {
  assert.equal(cameBy({via: 'observation', from: 'Mara'}),
      'came by observation from Mara');
  assert.equal(cameBy({via: 'birth', from: ''}), 'came by birth');
});

test('the dead are named with their years, the forgotten as such', () => {
  assert.equal(kinName({name: 'Ada', living: true, known: true}), 'Ada');
  assert.equal(kinName({name: 'Tone', living: false, known: true, born: -4,
    died: 63}), 'Tone (died 63, aged 67)');
  assert.equal(kinName({name: 'Elor', living: false, known: false}),
      'Elor (forgotten)');
});

test('holders nest under those they had a custom from', () => {
  const roots = transmissionTree([
    {name: 'Bo', via: 'observation', from: 'Ada'},
    {name: 'Ada', via: 'upbringing', from: 'Mara'},
    {name: 'Cy', via: 'reading', from: ''},
    {name: 'Di', via: 'observation', from: 'Bo'},
  ]);
  assert.equal(transmissionText(roots),
      'Ada (upbringing from Mara) > [Bo (observation from Ada) > ' +
      '[Di (observation from Bo)]], Cy (reading)');
});

test('learning from one another makes no loop', () => {
  const roots = transmissionTree([
    {name: 'Ada', via: 'observation', from: 'Bo'},
    {name: 'Bo', via: 'observation', from: 'Ada'},
  ]);
  assert.equal(roots.length, 1);
  assert.equal(transmissionText(roots),
      'Bo (observation from Ada) > [Ada (observation from Bo)]');
});

test('a variant says how it arose and who holds it', () => {
  const variant = {id: 'c52', origin: 'mutation', year: 9, living: true,
    holders: [{name: 'Ada'}]};
  assert.equal(variantNote(variant), 'c52 · mutation, year 9 · held by 1');
  assert.equal(variantNote({...variant, holders: []}),
      'c52 · mutation, year 9 · held by nobody, kept in a place');
  assert.equal(variantNote({...variant, living: false}),
      'c52 · mutation, year 9 · forgotten: nobody holds it now');
});

test('the notebook filters what the person wrote, moves and changes', () => {
  const entries = ['hypothesis', 'observation', 'note', 'move', 'world',
    'event'].map((kind) => ({kind}));
  const kinds = (filter) => entries.filter((e) => entryShown(e, filter))
      .map((e) => e.kind);
  assert.deepEqual(kinds('all').length, 6);
  assert.deepEqual(kinds('mine'), ['hypothesis', 'observation', 'note']);
  assert.deepEqual(kinds('moves'), ['move', 'world']);
  assert.deepEqual(kinds('events'), ['event']);
  assert.equal(entryMeta({year: 40, kind: 'world'}), 'Year 40 · the World');
  assert.equal(entryMeta({year: 3, kind: 'hypothesis'}),
      'Year 3 · hypothesis');
});

test('a release says where it went and whether it lives', () => {
  const release = {id: 'c30', how: 'song', tile: [6, 13], year: 4, heard: 2,
    holders: [[4, 0], [5, 3], [14, 8], [15, 7]], peak: 8, peak_year: 14,
    variants: 2, died: null};
  assert.equal(releaseStatus(release),
      'c30 · song at 6,13, year 4 · 2 heard it · held by 7 now (most: 8,' +
      ' year 14) · 2 variants');
  assert.equal(releaseStatus({...release, how: 'text', variants: 1}),
      'c30 · text at 6,13, year 4 · on a shelf · held by 7 now (most: 8,' +
      ' year 14)');
  assert.equal(releaseStatus({...release, heard: 0, peak: 0, died: 5}),
      'c30 · song at 6,13, year 4 · 0 heard it · nobody took it up;' +
      ' forgotten in year 5');
  assert.equal(releaseStatus({...release, peak: 0}),
      'c30 · song at 6,13, year 4 · 2 heard it · nobody has taken it up yet');
  assert.equal(releaseStatus({...release, died: 40}),
      'c30 · song at 6,13, year 4 · 2 heard it · died out in year 40' +
      ' (at most 8 held it)');
});

test('a song reaches as many as the World counts at a place', () => {
  const world = {rows: 16, cols: 24, reach: [[0, 0, 2], [1, 0, 1]]};
  assert.equal(withinReach(world, [0, 0]), 2);
  assert.equal(withinReach(world, [16, 24]), 2);  // Across both edges.
  assert.equal(withinReach(world, [8, 12]), 0);
  assert.equal(withinReach({rows: 16, cols: 24}, [0, 0]), 0);
});

test('disasters and migrants are written in the form the World reads', () => {
  assert.equal(fateMove('fire', [5, 6]), 'send a fire at 5,6');
  assert.equal(fateMove('migrants', [0, 23]), 'bring migrants at 0,23');
  assert.equal(fateMove('found school', [1, 2]), 'found a school at 1,2');
  assert.equal(fateMove('close library', [1, 2]), 'close the library at 1,2');
  assert.deepEqual(fateChoices({disaster_kinds: ['fire']}), ['fire',
    'migrants', 'found library', 'found school', 'found shrine',
    'close library', 'close school', 'close shrine']);
});

test('a release says how it fares in the twin', () => {
  const release = {id: 'c30', how: 'song', tile: [6, 13], year: 4, heard: 2,
    holders: [[4, 0], [5, 3]], peak: 3, peak_year: 5, variants: 1,
    died: null, twin: {id: 'c41', heard: 2, now: 0, peak: 0, died: 6}};
  assert.equal(releaseStatus(release),
      'c30 · song at 6,13, year 4 · 2 heard it · held by 3 now (most: 3,' +
      ' year 5) · in the twin: died out');
  assert.equal(releaseStatus({...release, twin: {...release.twin, died: null,
    now: 5}}).endsWith(' · in the twin: held by 5'), true);
});

test('the map shows the disasters of the last twenty years', () => {
  const rows = [
    {year: 3, disasters: [{kind: 'fire', tile: [1, 1]}]},
    {year: 20, disasters: []},
    {year: 25, disasters: [{kind: 'plague', tile: [5, 5]}]},
    {year: 26},
  ];
  assert.deepEqual(recentDisasters(rows, 40),
      [{kind: 'plague', tile: [5, 5], age: 15}]);
});

test('the ring says the chance that someone takes a song up', () => {
  const world = {rows: 16, cols: 24, song_uptake: 0.25, reach: [[0, 0, 3]]};
  assert.equal(reachNote(world, [0, 0]), 'The ring (0,0): 3 people would' +
      ' hear a song here; the chance at least one takes it up is 58%.');
  assert.equal(reachNote(world, [8, 12]),
      'The ring (8,12): nobody would hear a song here.');
});

test('what the page has already is never appended twice', () => {
  const kept = [{id: 1}, {id: 2}];
  assert.deepEqual(appendNew(kept, [{id: 2}, {id: 3}], 'id', 2, 10),
      [{id: 1}, {id: 2}, {id: 3}]);
  assert.deepEqual(appendNew(kept, [{id: 3}], 'id', 2, 2), [{id: 2}, {id: 3}]);
});

test('the census charts peoples, one people halfway up', () => {
  const rows = [
    {year: 0, population: 20, skill_worked: 1, tongues: 2,
     peoples: {peoples: 1, sizes: [20]}},
    {year: 1, population: 30, skill_worked: 2, tongues: 3,
     peoples: {peoples: 1, sizes: [30]}},
  ];

  const series = chartSeries(rows, 5);

  const byName = Object.fromEntries(series.map((line) => [line.name, line]));
  assert.deepEqual(Object.keys(byName),
                   ['population', 'skill_worked', 'tongues', 'peoples',
                     'fields']);
  assert.equal(byName.fields.most, 1);
  assert.equal(byName.population.most, 30);
  assert.equal(byName.skill_worked.most, 5);
  assert.equal(byName.tongues.most, 3);
  assert.equal(byName.peoples.most, 2);
  assert.equal(byName.peoples.value(rows[1]), 1);
  assert.equal(byName.peoples.value({year: 2}), 0);
  rows.push({year: 2, population: 30, skill_worked: 2, tongues: 3,
             peoples: {peoples: 3, sizes: [12, 10, 8]}});
  assert.equal(
      chartSeries(rows, 5).find((line) => line.name === 'peoples').most, 3);
});

test('a marriage rule may say whom it allows; other kinds may not', () => {
  assert.equal(
      releaseMove('marriage', 'kinship', 'song', '5, 9', 'We marry kin.',
                  'tongue'),
      'release marriage kinship within tongue as a song at 5,9: We marry kin.');
  assert.equal(
      releaseMove('belief', 'sacred', 'text', '1,1', 'Hm.', 'tongue'),
      'release belief sacred as a text at 1,1: Hm.');
  assert.equal(
      releaseMove('marriage', 'kinship', 'song', '1,1', 'Love.'),
      'release marriage kinship as a song at 1,1: Love.');
});

test('a kind offers only its domains', () => {
  const world = {domains: ['kinship', 'food', 'language'],
                 kind_domains: {marriage: ['kinship'], norm: ['kinship',
                   'food', 'language']}};
  assert.deepEqual(domainsFor(world, 'marriage'), ['kinship']);
  assert.deepEqual(domainsFor(world, 'norm'), world.domains);
  assert.deepEqual(domainsFor({domains: ['food']}, 'marriage'), ['food']);
});

test('a release goes where a song would reach the most', () => {
  const world = {rows: 20, cols: 20, reach: [[10, 10, 2], [0, 0, 1]]};

  assert.deepEqual(crowdedSpot(world), [10, 10]);
  assert.equal(crowdedSpot({...world, reach: []}), null);
  assert.equal(crowdedSpot({rows: 20, cols: 20}), null);
});
