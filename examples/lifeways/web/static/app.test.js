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
  askMove, cameBy, kinName, leverChoices, peopleColour, releaseMove,
  entryMeta, entryShown, fateMove, releaseStatus, savedNote,
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

test('a song reaches those old enough within sight, across the edges', () => {
  const world = {rows: 16, cols: 24, sight: 2, child_acts_age: 5, people: [
    {tile: [0, 0], age: 30},
    {tile: [15, 23], age: 30},  // Two steps away, across both edges.
    {tile: [1, 1], age: 3},  // Too young.
    {tile: [0, 3], age: 30},  // Three steps away.
  ]};
  assert.equal(withinReach(world, [0, 0]), 2);
  assert.equal(withinReach(world, [8, 12]), 0);
});

test('disasters and migrants are written in the form the World reads', () => {
  assert.equal(fateMove('fire', [5, 6]), 'send a fire at 5,6');
  assert.equal(fateMove('migrants', [0, 23]), 'bring migrants at 0,23');
});
