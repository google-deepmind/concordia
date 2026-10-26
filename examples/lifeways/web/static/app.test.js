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
  askMove, leverChoices, peopleColour, releaseMove, savedNote, twinNote,
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
