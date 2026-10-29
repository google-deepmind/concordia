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
  anatomyLines, answerText, askedNote, askMove, awayLines, cameBy, foundText,
  fairnessSeries, keptSeries, keptText, kinLayout, peoplesSeries,
  labLines, labStatus, prevalenceSeries, previewText, spreadLabels,
  surveyMove, surveyText, treeLayout, twinReport, wordChange,
  letGoText, mapText,
  tileText,
  overlayKey, parseFolded, personColour, placeText, queuedNote,
  tongueRanks,
  statusLines,
  goneFromPanel, paceNote,
  peoplesText,
  chartSeries,
  crowdedSpot, customKind,
  domainsFor, goneNote,
  kinName, worldName, yearText, entryText, moveWords,
  leverChoices, moreNote,
  peopleColour, releaseMove, appendNew, checkNote, entryMeta, entryShown,
  fateChoices,
  fateMove, influenceNote, lifeAnswer, lifeCustoms, lifeMemory, lifeMoves,
  lifeNote, lifeQuestion, lifeSuggested, livesNote, reachNote,
  recentDisasters, releaseStatus, savedNote,
  holdersLabel, holdersText, namesText, twinNote, variantNote, withinReach,
} = require('./app.js');

test('a release is written in the form the World reads', () => {
  assert.equal(
      releaseMove('belief', 'sacred', 'song', ' 8, 9 ', ' The moon is kind. '),
      'release belief sacred as a song at 8,9: The moon is kind.');
});

test('a people keeps its colour', () => {
  assert.equal(peopleColour('Hill people'), peopleColour('Hill people'));
  assert.notEqual(peopleColour('Hill people'), peopleColour('River people'));
  assert.notEqual(peopleColour('West people'), peopleColour('East people'));
  assert.match(peopleColour('River people'), /^#[0-9a-f]{6}$/);
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
      [['richness',
        'How rich the land is (from 0.05, least, to 1, most; now 0.2)'],
        ['without story', 'Without stories']]);
});

test('a custom says how it came and from whom', () => {
  assert.equal(cameBy({via: 'observation', from: 'Mara'}),
      'saw it done by Mara');
  assert.equal(cameBy({via: 'birth', from: ''}), 'had it from the start');
  assert.equal(cameBy({via: 'birth', from: 'Mara'}), 'inherited it from Mara');
  assert.equal(cameBy({via: 'upbringing', from: 'Ada'}),
      'was taught it as a child by Ada');
});

test('the peoples say whom they count, and are named', () => {
  assert.equal(peoplesText({peoples: 2, sizes: [26, 17],
    descents: ['Hill people', 'River people'], grown: 43}, 63),
  'Of the 63 alive, 43 are grown; they form 2 peoples (groups whose ' +
      'marriage rules let them marry one another): 26 mostly of the Hill ' +
      'people, 17 mostly of the River people. Children belong to none ' +
      'until grown.');
  assert.equal(peoplesText({peoples: 1, sizes: [9]}, 12),
      'Of the 12 alive, 9 are grown; they form one people (groups whose ' +
      'marriage rules let them marry one another): 9. Children belong to ' +
      'none until grown.');
  assert.equal(peoplesText({peoples: 0, sizes: [], grown: 0}, 3),
      'None of the 3 alive is grown, so no peoples.');
});

test('the pace says when the valley cannot keep the pace asked', () => {
  const seen = [[0, 100], [5000, 102], [20000, 108]];
  assert.equal(paceNote(seen, 10, false), 'running at about 0.4 a second: ' +
      'as fast as this valley can go now');
  assert.equal(paceNote(seen, 10, true), 'running at about 0.4 a second: ' +
      'as fast as this valley can go now (a twin beside it halves that)');
  assert.equal(paceNote(seen, 0.25, false), '');  // It keeps up.
  assert.equal(paceNote([[0, 1], [5000, 2]], 10, false), '');  // Too soon.
  assert.equal(paceNote([[0, 0], [10000, 30]], 10, false),
      'running at about 3 a second: as fast as this valley can go now');
});

test('an ask says when and where its answer comes', () => {
  assert.equal(askedNote('Solu', 122, false), 'Asked in year 122. Solu ' +
      'answers when the year turns; the answer shows under Interviews.');
  assert.equal(askedNote('Solu', 122, true), 'Asked in year 122. Solu ' +
      'answers when the year turns (press Play: the world is paused); the ' +
      'answer shows under Interviews.');
});

test('moves waiting say one happens a year', () => {
  assert.equal(queuedNote([]), '');
  assert.equal(queuedNote(['send a fire at 1,1']),
      'Next year: Send a fire at row 1, column 1.');
  assert.equal(queuedNote(['send a fire at 1,1', 'bring migrants at 2,2',
    'stop the world']), 'Next year: Send a fire at row 1, column 1. Then, ' +
      'one a year: Bring newcomers to row 2, column 2; End this world.');
});

test('a custom let go says what pushed it out, and why', () => {
  const record = {text: 'Sowing wild barley.', kind: 'skill',
    domain: 'farming', for_text: 'We take nothing from another\'s field.',
    why: 'domain'};
  assert.equal(letGoText(record), '"Sowing wild barley." (know-how about ' +
      'farming) gave way to "We take nothing from another\'s field.": two ' +
      'about farming are the most one holds.');
  assert.equal(letGoText({...record, kind: 'habit', why: 'locus'}),
      '"Sowing wild barley." (a habit about farming) gave way to "We take ' +
      'nothing from another\'s field.": one holds only one of each kind ' +
      'about a thing.');
});

test('a status is a line a sentence', () => {
  assert.deepEqual(statusLines('Year 0. Kaka is 29 years old; the land ' +
      'around holds 10.0 food. Food stored: 8.0 (a year needs 1.0). 2 other ' +
      'people are within sight.'), ['Year 0.', 'Kaka is 29 years old; the ' +
      'land around holds 10.0 food.', 'Food stored: 8.0 (a year needs 1.0).',
  '2 other people are within sight.']);
  assert.deepEqual(statusLines(''), []);
});

test('an entity says how Concordia builds it, customs counted once', () => {
  assert.deepEqual(anatomyLines({class: 'EntityAgentWithLogging',
    acting: {class: 'ConcatActComponent', source: 'concordia'},
    components: [
      {key: '__memory__', class: 'AssociativeMemory', source: 'concordia'},
      {key: 'learner', class: 'Learner', source: 'lifeways'},
      {key: 'custom c1', class: 'Custom', source: 'lifeways'},
      {key: 'custom c2', class: 'Custom', source: 'lifeways'},
    ]}), [
    'EntityAgentWithLogging (stock Concordia): the entity',
    'AssociativeMemory (stock Concordia): remembers what it has seen',
    'Learner (this example): takes up customs it is offered, and lets old ' +
        'ones go',
    'Custom × 2 (this example): a custom: a text it reads when it decides',
    'It acts through ConcatActComponent (stock Concordia): asks its mind (a ' +
        'language model, or the simple rules) what it does, in one ' +
        'document of all the parts above',
  ]);
});

test('the map colours people by the overlay chosen, and keys it', () => {
  const people = [
    {name: 'A', descent: 'River people', speech: 'c1', food: 0.5, age: 30,
      people: 0},
    {name: 'B', descent: 'River people', speech: 'c1', food: 7, age: 60,
      people: 0},
    {name: 'C', descent: 'Hill people', speech: 'c2', food: 2, age: 5,
      people: -1},
  ];
  const world = {people, grown_age: 16};
  const tongues = tongueRanks(people);
  assert.equal(tongues.get('c1'), 0);
  assert.equal(personColour(people[0], 'food', world, tongues), '#d62828');
  assert.equal(personColour(people[1], 'food', world, tongues), '#ffffff');
  assert.equal(personColour(people[2], 'people', world, tongues), '#9a9a9a');
  assert.deepEqual(overlayKey(world, 'age').map(([, words]) => words),
      ['grown (1)', 'elders (1)', 'children (1)']);
  assert.deepEqual(overlayKey(world, 'people').map(([, words]) => words),
      ['a people of 2, mostly of the River people',
        'children and newcomers, of no people yet (1)']);
  assert.deepEqual(overlayKey(world, 'tongue').map(([, words]) => words),
      ['a tongue spoken by 2, mostly of the River people',
        'a tongue spoken by 1, mostly of the Hill people']);
  assert.deepEqual(overlayKey(world, 'descent').map(([, words]) => words),
      ['the River people (2)', 'the Hill people (1)']);
});

test('a place says who founded and keeps it, and if it is a ruin', () => {
  const place = {kind: 'court', tile: [4, 7], founder: 'Experimenter',
    founded: 12, keeper: 'Ada', kept: 30, used: 30, ruined: false};
  assert.deepEqual(placeText(place), {
    title: 'A court at row 4, column 7',
    lines: ['Founded by you in year 12.', 'Last kept by Ada, in year 30.',
      'Last used in year 30.'],
  });
  assert.deepEqual(placeText({...place, keeper: '', kept: -1, ruined: true,
    founder: 'Bo'}), {
    title: 'A court at row 4, column 7 (a ruin)',
    lines: ['Founded by Bo in year 12.', 'Unused since year 30: a ruin. ' +
        'What it holds will crumble.'],
  });
  // Kept by its founder as it was founded: said once.
  assert.deepEqual(placeText({...place, founder: 'Ada', founded: 30}).lines,
      ['Founded by Ada in year 30.', 'Last used in year 30.']);
});

test('folded sections are remembered, and only known ones', () => {
  assert.deepEqual([...parseFolded('["notebook", "built", "nowhere"]')],
      ['notebook', 'built']);
  assert.deepEqual([...parseFolded('not json')], []);
  assert.deepEqual([...parseFolded('{"notebook": 1}')], []);
});

test('the map is said in words: who is where, what stands there', () => {
  const world = {rows: 16, cols: 24, fields: [[1, 1]],
    people: [{name: 'Ada', tile: [2, 3]}, {name: 'Bo', tile: [2, 3]},
      {name: 'Cy', tile: [12, 20]}],
    places: [{kind: 'library', tile: [2, 3]}, {kind: 'court', tile: [9, 9]},
      {kind: 'library', tile: [4, 4]}]};
  assert.equal(mapText(world), 'On the map: 3 people, most in the ' +
      'north-west (2); 2 libraries, 1 court; 1 field.');
  assert.equal(tileText(world, [2, 3]),
      'Row 2, column 3: Ada and Bo; a library.');
  assert.equal(tileText(world, [0, 0]), 'Row 0, column 0: nobody.');
});

test('a return says what happened since, the latest first', () => {
  const entries = [
    {id: 1, year: 3, kind: 'event', text: 'Only 9 people are left.'},
    {id: 2, year: 4, kind: 'move', text: 'You queue a move: stop the world'},
    {id: 3, year: 9, kind: 'event', text: 'Kaka built a court at row 1, ' +
        'column 2.'},
  ];
  assert.deepEqual(awayLines(entries, 1),
      ['Year 9: Kaka built a court at row 1, column 2.']);
  assert.deepEqual(awayLines(entries, 0, 1),
      ['Year 9: Kaka built a court at row 1, column 2.']);
  assert.deepEqual(awayLines(entries, 3), []);
});

test('a custom found says what it is and how many hold it', () => {
  assert.equal(foundText({text: 'We share.', kind: 'norm', domain: 'sharing',
    params: {form: 'known'}, holders: 12}), '"We share." (a rule others ' +
      'expect of them (about sharing)) · held by 12');
  assert.equal(foundText({text: 'Hills.', kind: 'belief', domain: 'sacred',
    holders: 0}), '"Hills." (a belief (about sacred)) · held by nobody now ' +
      '(a place keeps it)');
});

test('a custom\'s tree is laid out by depth, sized by holders', () => {
  const {height, nodes} = treeLayout([
    {id: 'c1', parent: '', depth: 0, living: true, holders: [{}, {}, {}, {}],
      text: 'We share.'},
    {id: 'c2', parent: 'c1', depth: 1, living: false, holders: [],
      text: 'We share, always and always and always, with everyone here.'},
    {id: 'c3', parent: 'c1', depth: 1, living: true, holders: [],
      text: 'We.'},
  ]);
  assert.equal(height, 105);
  assert.deepEqual(nodes.map((n) => [n.id, n.state, n.x > nodes[0].x]),
      [['c1', 'held', false], ['c2', 'gone', true], ['c3', 'kept', true]]);
  assert.equal(nodes[0].label, '4 · We share.');
  assert.equal(nodes[2].label, '−share');
  assert.equal(wordChange('In lean years, gather.', 'In lean years, always ' +
      'gather, always.'), '+always +always');
  assert.equal(wordChange('We share.', 'We share!'), 'the same words');
  assert.deepEqual([nodes[1].px, nodes[1].py], [nodes[0].x, nodes[0].y]);
  assert.equal(nodes[0].px, null);
});

test('a family is laid out in rows, a few names a row', () => {
  const kin = {person: {name: 'Ada', living: true},
    parents: [{name: 'Bo', living: false}, {name: 'Cy', living: true}],
    children: ['D', 'E', 'F', 'G', 'H', 'I'].map((name) =>
      ({name, living: true}))};
  const {boxes, links} = kinLayout(kin, 340);
  assert.deepEqual(boxes.filter((b) => b.self).map((b) => b.name), ['Ada']);
  assert.deepEqual(boxes.filter((b) => b.more).map((b) => b.name), ['+2']);
  assert.equal(links.length, 2);  // Parents to Ada, Ada to children.
  assert.ok(boxes.every((b) => b.x >= 0 && b.x + b.w <= 340));
});

test('a release says who would hear it and what they would let go', () => {
  assert.equal(previewText({hearers: 5, room: 1, push: 3, no: 1, pushed: [
    {text: 'Children are a blessing.', why: 'locus', count: 3}]}, 'kinship'),
  'Of the 5 who would hear it: 1 has room for it; 3 would let a custom go ' +
      'to take it up; 1 would not take it (they know as well already, or ' +
      'hold their own). They would let go: "Children are a blessing." (3: ' +
      'one holds one of each kind about kinship).');
  assert.equal(previewText({hearers: 0, room: 0, push: 0, no: 0,
    pushed: []}, 'food'), 'Nobody would hear it there now.');
});

test('fairness lines: inequality on 0 to 1, the rest on their own', () => {
  const rows = [{gini: 0.4, shares: 3, breaches: 1}, {shares: 6}];
  const lines = fairnessSeries(rows);
  assert.deepEqual(lines.map((l) => [l.name, l.most]),
      [['gini', 1], ['shares', 6], ['breaches', 1], ['judged', 1]]);
  assert.equal(lines[0].value(rows[1]), 0);  // Older rows lack the field.
});

test('norms kept: shares over ten years, by form and sight', () => {
  const rows = [
    {year: 1},  // Older rows lack the measure.
    {year: 2, norms_kept: {own_seen: [1, 4], own_unseen: [2, 2],
      known_seen: [1, 1], known_unseen: [0, 3]}},
    {year: 12, norms_kept: {own_seen: [3, 4], own_unseen: [0, 0],
      known_seen: [0, 0], known_unseen: [0, 1200]}},
  ];
  const lines = keptSeries(rows);
  assert.deepEqual(lines.map((l) => l.label), ['own, seen', 'own, unseen',
    'expected, seen', 'expected, unseen']);
  assert.equal(lines[0].value(rows[0]), null);
  assert.equal(lines[0].value(rows[1]), 0.25);
  assert.equal(lines[0].value(rows[2]), 0.75);  // Year 2 is ten years past.
  assert.equal(lines[1].value(rows[2]), null);  // None held in those years.
  assert.equal(lines[0].format(0.25), '25%');
  assert.equal(keptText(rows), 'Years 2–12. Unseen: own norms, 100% kept ' +
      '(2 of 2); expected, 0% kept (0 of 1,203). Seen: own, 50% kept (4 of ' +
      '8); expected, 100% kept (1 of 1).');
  assert.equal(keptText([{year: 1}, {year: 2, norms_kept: {
    own_seen: [0, 0], own_unseen: [0, 0], known_seen: [0, 0],
    known_unseen: [0, 0]}}]), '');
  assert.equal(keptText([{year: 3, norms_kept: {own_seen: [1, 2]}}]),
      'Years 3–3. Unseen: own norms, none held; expected, none held. ' +
      'Seen: own, 50% kept (1 of 2); expected, none held.');
});

test('a survey is asked in the World\'s form and read in words', () => {
  assert.equal(surveyMove(' Why share? ', 0), 'survey: Why share?');
  assert.equal(surveyMove('Why share?', 10), 'survey 10: Why share?');
  assert.equal(moveWords('survey: Why share?'), 'Ask everyone: “Why share?”');
  assert.equal(moveWords('survey 10: Why share?'),
      'Ask 10 people: “Why share?”');
  assert.deepEqual(surveyText({year: 1200, question: 'Why share?', asked: 30,
    answered: 4}), {head: 'Year 1,200: “Why share?”, asked of 30; 4 ' +
    'answered so far.', lines: []});
  assert.deepEqual(surveyText({year: 12, question: 'Why share?', asked: 30,
    answered: 20, tally: [['c3', 'We share our food.', 15]], none: 5}), {
    head: 'Year 12: “Why share?”, answered by 20 of 30.',
    lines: ['“We share our food.”: named by 15 (75%)',
      'Named none of their customs: 5 (25%)']});
  assert.deepEqual(surveyText({year: 12, question: 'Why?', asked: 3,
    answered: 0, tally: [], none: 0}).lines, []);
});

test('the twin report reads each measure in both worlds', () => {
  const row = (year, population, fields, kept) => ({year, population, fields,
    tongues: 2, gini: 0.3, norms_kept: {own_unseen: kept}});
  const rows = [row(9, 30, 4, [0, 0]), row(10, 40, 4, [1, 4]),
    row(11, 40, 4, [1, 4])];
  const twin = [row(10, 50, 0, [3, 4]), row(11, 70, 0, [3, 4]),
    row(12, 99, 0, [0, 0])];
  assert.deepEqual(twinReport(rows, twin), {
    // A tie (50% and 50 points) keeps the measures' order.
    head: 'Over the 2 years 10–11, a year on average; the twin differs most ' +
        'in fields, alive, own norms kept unseen.',
    lines: [
      'Fields: 4 here, 0 in the twin (−100%).',
      'Alive: 40 here, 60 in the twin (+50%).',
      'Own norms kept unseen: 25% here, 75% in the twin (+50 points).',
      'The same in both (within 1%): tongues, inequality.',
    ],
  });
  const one = twinReport([row(10, 100, 1, [0, 0])],
      [row(10, 100.5, 1, [1, 2])]);
  assert.equal(one.head, 'Over year 10, a year on average: the same in ' +
      'every measure both worlds have.');
  assert.deepEqual(one.lines, [
    'The same in both (within 1%): alive, fields, tongues, inequality.',
    'Own norms kept unseen: none held here, 50% in the twin.']);
  assert.equal(twinReport(rows, [row(12, 1, 1, [0, 0])]), null);
  assert.equal(twinReport([row(10, 5, 1, [0, 0])],
      [row(10, 5, 1, [0, 0])]).head,
  'Over year 10, a year on average: the same in every measure.');
});

test('a hypothesis says what the census will check, then how it came out',
    () => {
      const check = {measure: 'fields', direction: 'rise', due: 1019,
        from: 6, from_years: [4, 8]};
      const words = {fields: 'fields sown'};
      assert.equal(checkNote(check, words), 'The census checks it in year ' +
          '1,019: fields sown, 6 on average (years 4–8), to rise.');
      assert.equal(checkNote({...check, result: {held: false, to: 5.125,
        to_years: [1015, 1019]}}, words), 'Did not hold (year 1,019): ' +
          'fields sown went from 6 to 5.13 on average (years 1,015–1,019); ' +
          'you said rise.');
    });

test('the lab says what it plays, and what came of it', () => {
  assert.equal(labStatus(null), 'No experiment yet.');
  const compare = {kind: 'compare', lever: ['richness', 0.5], seeds: [0, 4],
    years: 100, runs: 10, done: 3, state: 'playing'};
  assert.equal(labStatus(compare), 'Playing how rich the land is set to ' +
      '0.5, against this world\'s settings: 3 of 10 runs done (seeds 0–4, ' +
      '100 years each)…');
  const done = {...compare, state: 'done', seconds: 95, result: {seeds:
    [0, 1, 2, 3, 4], measures: {living: {control: [40, 2.5],
      treatment: [52.25, 3], difference: [12.25, 1], above: 5, below: 0}}}};
  assert.equal(labStatus(done), 'How rich the land is set to 0.5, against ' +
      'this world\'s settings (seeds 0–4, 100 years each), done in 95 s:');
  assert.deepEqual(labLines(done), ['Alive: 40 ± 2.5 as set, 52.25 ± 3 ' +
      'with the lever (+12.25; more in 5 of 5 seeds, less in 0).']);
  const again = {kind: 'replicate', lever: null, seeds: [1, 1], years: 50,
    state: 'done', seconds: 9, here: {living: 70, fields: 3, tongues: 0.999},
    result: {seeds: [1], measures: {
      living: {mean: 40, sd: 0, least: 40, most: 40},
      fields: {mean: 3, sd: 0, least: 2, most: 4},
      tongues: {mean: 1, sd: 0, least: 1, most: 1}}}};
  assert.equal(labStatus(again), 'This world\'s settings on other seeds ' +
      '(seed 1, 50 years each), done in 9 s:');
  assert.deepEqual(labLines(again), [
    'Alive: 40 ± 0 (from 40 to 40); this world, 70, above every seed.',
    'Fields sown: 3 ± 0 (from 2 to 4); this world, 3, within them.',
    'Tongues: 1 ± 0 (from 1 to 1); this world, 1, within them.']);
  assert.deepEqual(labLines({...again, here: undefined,
    here_note: 'This world has not lived 50 years yet.'}).slice(-1),
  ['This world has not lived 50 years yet.']);
  assert.equal(labStatus({...compare, lever: ['without', 'story'],
    state: 'failed', error: 'ValueError: no'}), 'The experiment failed: ' +
      'ValueError: no');
});

test('labels at the same height spread up from the bottom', () => {
  const labels = [{at: 100}, {at: 100}, {at: 100}, {at: 20}];
  spreadLabels(labels, 13, 7, 100);
  assert.deepEqual(labels.map((l) => l.at), [20, 74, 87, 100]);
  const apart = [{at: 10}, {at: 60}];
  spreadLabels(apart, 13, 7, 100);
  assert.deepEqual(apart.map((l) => l.at), [10, 60]);
});

test('what most hold, and each people, over the years', () => {
  const rows = [
    {held: {top: [['c1', 0.9], ['c2', 0.5]]},
      peoples: {sizes: [10, 4], descents: ['River people', 'Hill people']}},
    {held: {top: [['c2', 0.6], ['c3', 0.2]]},
      peoples: {sizes: [12], descents: ['River people']}},
  ];
  const held = prevalenceSeries(rows, {c2: {text: 'We share our food.'}});
  assert.deepEqual(held.map((l) => [l.id, l.label]), [['c2', 'We share our…'],
    ['c1', 'c1'], ['c3', 'c3']]);
  assert.equal(held[0].value(rows[1]), 0.6);
  assert.equal(held[1].value(rows[1]), null);  // Not among the five then.
  assert.equal(held[0].format(0.6), '60%');
  const peoples = peoplesSeries(rows);
  assert.deepEqual(peoples.map((l) => [l.label, l.most]),
      [['River people', 12], ['Hill people', 12]]);
  assert.equal(peoples[1].value(rows[1]), 0);
});

test('a panel says when its person is no longer alive', () => {
  assert.equal(goneFromPanel('Solu', 122, 1137), 'Solu is no longer alive ' +
      '(year 1,137). Below is how they were in year 122.');
});

test('the dead are named with their years, the forgotten as such', () => {
  assert.equal(kinName({name: 'Ada', living: true, known: true}), 'Ada');
  assert.equal(kinName({name: 'Tone', living: false, known: true, born: -4,
    died: 63}), 'Tone (died in year 63, aged 67)');
  assert.equal(kinName({name: 'Elor', living: false, known: false}),
      'Elor (forgotten)');
});

test('holders are grouped by how they came by a custom', () => {
  const holders = [
    {name: 'Bo', via: 'observation', from: 'Ada'},
    {name: 'Ada', via: 'upbringing', from: 'Mara'},
    {name: 'Cy', via: 'upbringing', from: 'Mara'},
    {name: 'Di', via: 'upbringing', from: 'Mara'},
    {name: 'Eo', via: 'sanction', from: ''},
    {name: 'Fa', via: 'sanction', from: ''},
  ];
  assert.equal(holdersText(holders),
      'Ada, Cy and Di were taught it as a child by Mara; Eo and Fa were ' +
      'shamed into it; Bo saw it done by Ada.');
  assert.equal(holdersText(holders, 2), 'Ada, Cy and 1 more were taught ' +
      'it as a child by Mara; Eo and Fa were shamed into it; Bo saw it ' +
      'done by Ada.');
  assert.equal(holdersLabel(holders), '6 hold it: Bo, Ada, Cy and 3 more');
  assert.equal(holdersLabel(holders.slice(0, 1)), '1 holds it: Bo');
  assert.equal(namesText([], 3), '');
  assert.equal(namesText(['Ada', 'Bo'], 3), 'Ada and Bo');
});

test('a variant says tersely who holds it and when it arose', () => {
  const variant = {id: 'c52', origin: 'mutation', year: 9, living: true,
    holders: [{name: 'Ada'}]};
  assert.equal(variantNote(variant, 'held'),
      'held by 1 person · since year 9');
  assert.equal(variantNote({...variant, holders: []}, 'kept'),
      'since year 9');
  assert.equal(variantNote({...variant, holders: []}, 'tree'),
      'only in a library or shrine · since year 9');
  const dead = {...variant, holders: [], living: false, year: 28607};
  assert.equal(variantNote(dead, 'gone'), 'arose in year 28,607');
  assert.equal(variantNote({...dead, returned: true}, 'tree'),
      'no longer held · arose in year 28,607 · its words came back in a ' +
      'later copy');
  assert.equal(variantNote({...dead, again: [22, 107]}, 'gone'),
      'arose in year 28,607 · the same words arose again in years 22, 107');
  assert.equal(variantNote({...variant, origin: 'founder'}, 'held'),
      'held by 1 person · since year 9 · brought by the first settlers');
  assert.equal(variantNote({...variant, kind: 'skill', params: {level: 3}},
      'held'), 'level 3 · held by 1 person · since year 9');
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
  assert.equal(entryMeta({year: 40, kind: 'world'}),
      'Year 40 · what came of it');
  assert.equal(entryMeta({year: 3, kind: 'hypothesis'}),
      'Year 3 · hypothesis');
});

test('a release says where it went and whether it lives', () => {
  const release = {id: 'c30', how: 'song', tile: [6, 13], year: 4, heard: 2,
    holders: [[4, 0], [5, 3], [14, 8], [15, 7]], peak: 8, peak_year: 14,
    variants: 2, died: null};
  assert.equal(releaseStatus(release),
      'sung at row 6, column 13 in ' +
      'year 4 · 2 heard it · held by 7 now (most: 8, in' +
      ' year 14) · 2 versions of it, as copies changed it');
  assert.equal(releaseStatus({...release, how: 'text', variants: 1}),
      'left as a text on a library shelf at row 6, column 13 in ' +
      'year 4 · held by 7 now' +
      ' (most: 8, in year 14)');
  assert.equal(releaseStatus({...release, heard: 0, peak: 0, died: 5}),
      'sung at row 6, column 13 in ' +
      'year 4 · 0 heard it · nobody took it up; gone by' +
      ' year 5');
  assert.equal(releaseStatus({...release, peak: 0}),
      'sung at row 6, column 13 in ' +
      'year 4 · 2 heard it · nobody has taken it up yet');
  assert.equal(releaseStatus({...release, died: 40, year: 29904}),
      'sung at row 6, column 13 in ' +
      'year 29,904 · 2 heard it · died out in year 40' +
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
    'found court', 'close library', 'close school', 'close shrine',
    'close court']);
});

test('a release says how it fares in the twin', () => {
  const release = {id: 'c30', how: 'song', tile: [6, 13], year: 4, heard: 2,
    holders: [[4, 0], [5, 3]], peak: 3, peak_year: 5, variants: 1,
    died: null, twin: {id: 'c41', heard: 2, now: 0, peak: 0, died: 6}};
  assert.equal(releaseStatus(release),
      'sung at row 6, column 13 in ' +
      'year 4 · 2 heard it · held by 3 now (most: 3, in' +
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
      ' song here now; the chance at least one takes it up is 58%. People' +
      ' move, so this can change by next year.');
  assert.equal(reachNote(world, [8, 12]), 'The white ring on the map (row' +
      ' 8, column 12) is where your next release goes: nobody would hear a' +
      ' song here now.');
  assert.equal(reachNote(world, [0, 0], 'text'), 'The white ring on the map' +
      ' (row 0, column 0) is where your next release goes: a text is left' +
      ' on a library\'s shelf here (a new library if none stands near), for' +
      ' whoever comes to read it.');
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
  name: 'Kaka',
  unanswered: 'gathers food',
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
      'Year 0: the year waits for you. Answer within 90\u00a0s, or Kaka, by ' +
      'habit, gathers food.');
  assert.match(lifeNote({name: 'Kaka', alive: true,
    asked: {...ASKED_MOVE, kind: 'pass_on', year: -1,
      unanswered: 'passes on all they hold'}}),
  /^The year waits.*by habit, passes on all they hold\.$/);
  assert.equal(lifeNote({name: 'Kaka', alive: true, asked: null}),
      'The others are choosing; the year turns.');
  assert.match(lifeNote({name: 'Kaka', alive: false, asked: null}),
      /^Kaka has died.*born again/);
  assert.equal(livesNote({lives: ['Kaka']}), '');
  assert.equal(livesNote({lives: ['Kaka', 'Teo']}), 'Your lives: Kaka → Teo');
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
  assert.equal(goneNote(1, 1), '1 other version that died out: nobody ' +
      'holds it, and no library or shrine keeps it');
  assert.equal(goneNote(386, 30), '386 other versions that died out: ' +
      'nobody holds them, and no library or shrine keeps them; the latest ' +
      '30 are listed');
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

test('moves read as a person would say them', () => {
  assert.equal(moveWords('release norm kinship as a song at 8,9: Feed all.'),
      'Sing “Feed all.” at row 8, column 9: a rule (about kinship)');
  assert.equal(moveWords(
      'release marriage kinship within tongue as a text at 1,2: Wed well.'),
  'Leave as a text “Wed well.” at row 1, column 2: whom to marry, only ' +
      'one who speaks as we do (about kinship)');
  assert.equal(moveWords('send a flood at 3,4'),
      'Send a flood at row 3, column 4');
  assert.equal(moveWords('bring migrants at 3,4'),
      'Bring newcomers to row 3, column 4');
  assert.equal(moveWords('close the shrine at 3,4'),
      'Close the shrine nearest row 3, column 4');
  assert.equal(moveWords('ask Kaka: Why?'), 'Ask Kaka: “Why?”');
  assert.equal(moveWords('stop the world'), 'End this world');
  assert.equal(entryText({kind: 'move',
    text: 'You queue a move: send a fire at 1,1'}),
  'Your move for next year: Send a fire at row 1, column 1');
  assert.equal(entryText({kind: 'world', text: 'A fire.'}), 'A fire.');
  assert.equal(answerText({name: 'Kaka', answer: 'Kaka says: "We sow."'}),
      'Kaka says: "We sow."');
  assert.equal(answerText({name: 'Kaka', answer: 'I would rather not say.'}),
      '"I would rather not say."');
  assert.equal(answerText({name: 'Kaka', answer: 'Kakaru sows.'}),
      '"Kakaru sows."');
  assert.equal(answerText({name: 'Kaka', answer: 'Kaka, smiling, sows.'}),
      'Kaka, smiling, sows.');
  assert.equal(answerText({name: 'Kaka', answer: 'Kaka-ru sows.'}),
      '"Kaka-ru sows."');
  assert.equal(answerText({name: 'Kaka', answer: 'Kaka Because we must.'}),
      '"Because we must."');
  assert.equal(answerText({name: 'Kaka', answer: 'Kaka: gather food'}),
      '"gather food"');
  assert.equal(entryText({kind: 'move',
    text: 'You fork a twin in year 40: seed_bank 0.3.'}),
  'You fork a twin in year 40: how fast bare land greens again set to 0.3.');
  assert.equal(entryText({kind: 'move',
    text: 'You fork a twin in year 3: without sanction.'}),
  'You fork a twin in year 3: without shaming those who break a rule.');
});

test('a list says what it leaves out', () => {
  assert.equal(moreNote(80, 80), '');
  assert.equal(moreNote(80, 95), 'the first 80 of 95 are listed');

});

test('memory and notes read as said; old places in words', () => {
  const context = '[observation] Hara teaches: Honour whoever builds a' +
      ' shrine at 2,2.\nA custom of ours (sacred) [c9]: Year 1. Kaka is 30' +
      ' years old and lives on high land at 1,2; always.';
  const memory = lifeMemory(context, 'Kaka');
  assert.ok(memory.includes('builds a shrine at 2,2.'), memory);
  assert.ok(memory.includes('lives on high land at 1,2; always.'), memory);
  assert.equal(entryText({kind: 'note', text: 'It grew to 1,200 people.'}),
      'It grew to 1,200 people.');
  assert.equal(entryText({kind: 'event', text: 'A fire struck at 8,1: 3' +
      ' lived within reach, 0 died.'}), 'A fire struck at row 8, column 1:' +
      ' 3 lived within reach, 0 died.');
  assert.equal(entryText({kind: 'world', text: 'Your release "Meet at 1,000' +
      ' hands; 5 hear it." is sung at 8,13; 3 hear it.'}), 'Your release' +
      ' "Meet at 1,000 hands; 5 hear it." is sung at row 8, column 13; 3' +
      ' hear it.');
  assert.equal(entryText({kind: 'world', text: 'Rest at 1,000 "of 2,3".'}),
      'Rest at 1,000 "of 2,3".');
  assert.equal(entryText({kind: 'world', text: 'c33984 is sung at 15,12;' +
      ' 2 hear it.'}), 'Your release is sung at row 15, column 12; 2 hear' +
      ' it.');
  assert.equal(entryText({kind: 'world', text: 'c10859 is written at the' +
      ' library at 9,9.'}), 'Your release is written at the library at row' +
      ' 9, column 9.');
  assert.equal(entryText({kind: 'event', text: 'Your release c33984 died' +
      ' out: "We respect our elders" (at most 0 held it): nobody took it' +
      ' up.'}), 'Your release "We respect our elders" died out (at most 0' +
      ' held it): nobody took it up.');
  assert.equal(entryText({kind: 'event', text: 'c30662 is the most widely' +
      ' held custom for the first time (100% hold it): "In lean."'}),
  'A custom is the most widely held for the first time (100% of the living' +
      ' hold it): "In lean."');
  assert.equal(entryText({kind: 'world', text: 'c9 is sung at 1,2; 1 hear' +
      ' it.'}), 'Your release is sung at row 1, column 2; 1 hears it.');
  assert.equal(entryText({kind: 'event', text: 'Your release c5 died out:' +
      ' "We say "hi" daily" (at most 0 held it): nobody was within hearing.'}),
  'Your release "We say "hi" daily" died out (at most 0 held it): nobody' +
      ' was within hearing.');
  // A move the World echoes, as typed, ends like a report: left as typed.
  assert.equal(entryText({kind: 'world', text: 'The release is not' +
      ' understood: release a song is sung at 3,4; 5 hear it.'}),
  'The release is not understood: release a song is sung at 3,4; 5 hear it.');
  assert.equal(entryText({kind: 'world', text: 'Your release "Sow." is' +
      ' written at the library at 2,5.'}), 'Your release "Sow." is written' +
      ' at the library at row 2, column 5.');
});

test('a skill is released with how good it is', () => {
  assert.equal(releaseMove('skill', 'food', 'song', '3,4', 'Dry fish.', '', 4),
      'release skill food level 4 as a song at 3,4: Dry fish.');
  assert.equal(releaseMove('norm', 'food', 'song', '3,4', 'Share.', '', 4),
      'release norm food as a song at 3,4: Share.');
  assert.equal(moveWords('release skill food level 4 as a song at 3,4: Dry' +
      ' fish.'), 'Sing “Dry fish.” at row 3, column 4: know-how, level 4' +
      ' (about food)');
});
