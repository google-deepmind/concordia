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
  askMove, cameBy, chartSeries, crowdedSpot, customKind, domainsFor, goneNote,
  kinName, worldName, yearText,
  leverChoices, lineageGroups,
  peopleColour, releaseMove, appendNew, entryMeta, entryShown, fateChoices,
  fateMove, influenceNote, lifeAnswer, lifeCustoms, lifeMemory, lifeMoves,
  lifeNote, lifeQuestion, lifeStatus, lifeSuggested, livesNote, reachNote,
  recentDisasters, releaseStatus, savedNote,
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
      '· picked up again at year 0, saved at year 9');
});

test('the page says what a twin is and where it is', () => {
  assert.equal(twinNote(null), 'No twin.');
  assert.equal(
      twinNote({lever: 'richness', value: 0.8, year: 40, year_now: 52,
        people: 61, running: true, error: ''}),
      'The twin (how rich the land is: 0.8; forked in year 40) is in year' +
      ' 52: 61 people alive.');
  assert.equal(
      twinNote({lever: 'without', value: 'story', year: 3, year_now: 3,
        people: 12, running: false, error: ''}),
      'The twin (without stories; forked in year 3) is in year 3: 12 people' +
      ' alive (stopped).');
});

test('the fork form offers each lever, then each channel to go without', () => {
  assert.deepEqual(
      leverChoices({levers: {richness: [0.05, 1, 0.2]}, channels: ['story']}),
      [['richness', 'How rich the land is (0.05–1; now 0.2)'],
        ['without story', 'Without stories']]);
});

test('a custom says how it came and from whom', () => {
  assert.equal(cameBy({via: 'observation', from: 'Mara'}),
      'saw it done by Mara');
  assert.equal(cameBy({via: 'birth', from: ''}), 'was born with it');
  assert.equal(cameBy({via: 'upbringing', from: 'Ada'}),
      'was taught it as a child by Ada');
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
      'Ada was taught it as a child by Mara; Bo saw it done by Ada; ' +
      'Di saw it done by Bo; Cy read it in a text');
});

test('learning from one another makes no loop', () => {
  const roots = transmissionTree([
    {name: 'Ada', via: 'observation', from: 'Bo'},
    {name: 'Bo', via: 'observation', from: 'Ada'},
  ]);
  assert.equal(roots.length, 1);
  assert.equal(transmissionText(roots),
      'Bo saw it done by Ada; Ada saw it done by Bo');
});

test('a variant says how it arose and who holds it', () => {
  const variant = {id: 'c52', origin: 'mutation', year: 9, living: true,
    holders: [{name: 'Ada'}]};
  assert.equal(variantNote(variant),
      'held by 1 person · changed by a copying error, year 9');
  assert.equal(variantNote({...variant, holders: []}),
      'nobody holds it, but a library or shrine keeps it · changed by a ' +
      'copying error, year 9');
  assert.equal(variantNote({...variant, living: false, year: 28607}),
      'nobody holds this version any more · changed by a copying error, ' +
      'year 28,607');
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
      'sung at 6,13 in year 4 · 2 heard it · held by 7 now (most: 8, in' +
      ' year 14) · 2 versions of it, as copies changed it');
  assert.equal(releaseStatus({...release, how: 'text', variants: 1}),
      'left as a text on a library shelf at 6,13 in year 4 · held by 7 now' +
      ' (most: 8, in year 14)');
  assert.equal(releaseStatus({...release, heard: 0, peak: 0, died: 5}),
      'sung at 6,13 in year 4 · 0 heard it · nobody took it up; gone by' +
      ' year 5');
  assert.equal(releaseStatus({...release, peak: 0}),
      'sung at 6,13 in year 4 · 2 heard it · nobody has taken it up yet');
  assert.equal(releaseStatus({...release, died: 40, year: 29904}),
      'sung at 6,13 in year 29,904 · 2 heard it · died out in year 40' +
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
      'sung at 6,13 in year 4 · 2 heard it · held by 3 now (most: 3, in' +
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
  assert.equal(reachNote(world, [0, 0]), 'The white ring on the map (row 0,' +
      ' column 0) is where your next release goes: 3 people would hear a' +
      ' song here; the chance at least one takes it up is 58%.');
  assert.equal(reachNote(world, [8, 12]), 'The white ring on the map (row' +
      ' 8, column 12) is where your next release goes: nobody would hear a' +
      ' song here.');
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

const ASKED_MOVE = {
  id: 'r1',
  kind: 'move',
  question: 'It is year 0. What does Kaka do this year? Choose one move:\n' +
      '- gather food\n- rest\n- court Hara\nAnswer with one of these ' +
      'moves, exactly as written, then the ids of the customs Kaka follows ' +
      'in doing it, if any.',
  context: 'Who I am:\nKaka.\n\n' +
      'A custom of ours (kinship) [c1]: We raise a child only when fed.\n' +
      'Something we know how to do (food) [c4]: Smoking fish. (level 2)\n' +
      'A custom of ours (kinship) [c1]: We raise a child only when fed.\n',
  year: 0,
  seconds_left: 90,
};

test('a played person chooses among the moves its question lists', () => {
  assert.deepEqual(lifeMoves(ASKED_MOVE.question),
      ['gather food', 'rest', 'court Hara']);
  assert.equal(lifeQuestion(ASKED_MOVE),
      'It is year 0. What does Kaka do this year?');
  assert.deepEqual(lifeCustoms(ASKED_MOVE.context), [
    {id: 'c1', label: 'A custom of ours', domain: 'kinship',
     text: 'We raise a child only when fed.'},
    {id: 'c4', label: 'Something we know how to do', domain: 'food',
     text: 'Smoking fish. (level 2)'},
  ]);
});

test('a played person answers as the World reads answers', () => {
  assert.equal(lifeAnswer('move', 'rest', []), 'rest');
  assert.equal(lifeAnswer('move', 'gather food', ['c4']), 'gather food c4');
  assert.equal(lifeAnswer('move', '', ['c4']), '');
  assert.equal(lifeAnswer('pass_on', '', ['c1', 'c4']), 'pass on c1 c4');
  assert.equal(lifeAnswer('interview', '', [], '  Fish.  '), 'Fish.');
  assert.equal(lifeQuestion({kind: 'pass_on', question:
    'Kaka\'s child is about to be born. Which customs? Answer "pass on" and ' +
    'the ids.'}), 'Kaka\'s child is about to be born. Which customs?');
});

test('the life note says what waits, and when a life ends', () => {
  assert.equal(lifeNote(null), '');
  assert.equal(lifeNote({name: 'Kaka', alive: true, asked: ASKED_MOVE}),
      'Year 0: the year waits for you. Answer within 90\u00a0s, or you gather ' +
      'food.');
  assert.match(lifeNote({name: 'Kaka', alive: true,
    asked: {...ASKED_MOVE, kind: 'pass_on', year: -1}}),
  /^The year waits.*pass on all you hold\.$/);
  assert.equal(lifeNote({name: 'Kaka', alive: true, asked: null}),
      'The others are choosing; the year turns.');
  assert.match(lifeNote({name: 'Kaka', alive: false, asked: null}),
      /^Kaka has died.*born again/);
  assert.equal(livesNote({lives: ['Kaka']}), '');
  assert.equal(livesNote({lives: ['Kaka', 'Teo']}), 'Your lives: Kaka → Teo');
});

test('the played person sees how its life stands, as it last saw it', () => {
  const context = '[observation] Year 1. Hara is 20 years old.\n' +
      '[observation] Year 1. Kaka is 30 years old. Partner: none.\n' +
      '[observation] Year 2. Kaka is 31 years old. Partner: Hara.\n\nA ...';
  assert.equal(lifeStatus(context, 'Kaka'),
      'Year 2. Kaka is 31 years old. Partner: Hara.');
  assert.equal(lifeStatus(context, 'Raan'), '');
});

test('the played person reads its memory as a person would', () => {
  const context = '[observation] Kaka sees Sane gather food: [custom id=c4 ' +
      'kind=skill domain=food level=2 via=observation from=Sane stores=11.0]' +
      ' Smoking fish over green alder. |params={"level": 2} Kaka sees Hara ' +
      'gather food: [custom id=c7 kind=habit domain=food via=observation ' +
      'from=Hara stores=11.0] In lean years, gather first. Year 1. Kaka is ' +
      '30 years old.\n\nA custom of ours (kinship) [c1]: Feed the child.';
  assert.equal(lifeMemory(context, 'Kaka'),
      '[observation] Kaka sees Sane gather food: Smoking fish over ' +
      'green alder.\nKaka sees Hara gather food: In lean years, gather ' +
      'first.\nYear 1. Kaka is 30 years old.\n\nA custom of ours (kinship):' +
      ' Feed the child.');
});

test('a move is done by the know-how it needs', () => {
  const customs = [
    {id: 'c4', label: 'Something we know how to do', domain: 'food'},
    {id: 'c7', label: 'A habit of ours', domain: 'food'},
    {id: 'c11', label: 'Something we know how to do', domain: 'farming'},
    {id: 'c9', label: 'Something we know how to do', domain: 'learning'},
  ];
  assert.deepEqual(lifeSuggested('gather food', customs), ['c4']);
  assert.deepEqual(lifeSuggested(
      'gather food, sparing the fields of others', customs), ['c4']);
  assert.deepEqual(lifeSuggested('sow a field', customs), ['c11']);
  assert.deepEqual(lifeSuggested('read at the library', customs), ['c9']);
  assert.deepEqual(lifeSuggested('rest', customs), []);
});

test('the player sees how far its ways have reached', () => {
  assert.equal(influenceNote({}),
      'Nobody living has learned a custom from you yet.');
  assert.equal(influenceNote({reach: {people: 1, customs: 1}}),
      '1 living person has learned a custom from you.');
  assert.equal(influenceNote({reach: {people: 4, customs: 2}}),
      '4 living people have learned 2 customs from you.');
});

test('the fold of dead branches says what it holds', () => {
  assert.equal(goneNote(1, 1), '1 other version nobody holds any more ' +
      '(copying errors that died out)');
  assert.equal(goneNote(386, 30), '386 other versions nobody holds any ' +
      'more (copying errors that died out); the latest 30 are listed');
});

test('a lineage groups versions by how they live on', () => {
  const v = (id, living, holders) => ({id, living, holders});
  const groups = lineageGroups([
    v('c1', false, []), v('c2', true, ['a']), v('c3', true, []),
    v('c4', true, ['a', 'b']),
  ]);
  assert.deepEqual(groups.held.map((x) => x.id), ['c4', 'c2']);
  assert.deepEqual(groups.kept.map((x) => x.id), ['c3']);
});

test('kinds, worlds and years read as words', () => {
  assert.equal(customKind({kind: 'norm', domain: 'kinship',
    params: {form: 'internalized'}}),
  'a rule they keep as their own (about kinship)');
  assert.equal(customKind({kind: 'norm', domain: 'sharing',
    params: {form: 'known'}}), 'a rule others expect of them (about sharing)');
  assert.equal(customKind({kind: 'skill', domain: 'food',
    params: {level: 9}}), 'know-how, level 9 (about food)');
  assert.equal(customKind({kind: 'marriage', domain: 'kinship',
    params: {within: 'descent'}}),
  'whom to marry, only within their own people (about kinship)');
  assert.equal(customKind({kind: 'bias', domain: 'learning', params: {}}),
      'whom to learn from (about learning)');
  assert.equal(worldName('sandbox'), 'The Sandbox');
  assert.equal(worldName('first_farmers'), 'First Farmers');
  assert.equal(yearText(30193), '30,193');
});
