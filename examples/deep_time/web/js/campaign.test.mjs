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

// Tests of campaign.js: `node --test` (run by web/js_test.py).

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {campaignHtml, campaignTotal, chapterHtml, dailyHtml, nextChapterHtml, savedHtml, whatsNewHtml} from './campaign.js';

const campaign = {
  chapters: [
    {scenario: 'daisyworld', title: 'A young sun', when: '3 billion years ago', stars: 2, open: true},
    {scenario: 'herds', title: 'Grazers', when: '3 billion years ago', stars: 0, open: true},
    {scenario: 'great_oxidation', title: 'The first breath', when: '2.6 billion years ago', stars: 0, open: false},
  ],
  daily: {date: '2026-10-15', scenario: 'snowball', title: 'The world in ice', seed: 417, stars: 1},
};

test('open chapters show their best stars; closed ones are locked', () => {
  const html = campaignHtml(campaign);
  const named = campaignHtml({chapters: [{scenario: 'great_acceleration', title: 'Coal and steam',
                                          when: '230 years ago', world: 'Great Acceleration', stars: 0, open: true}]});
  assert.match(named, /<small>230 years ago · Great Acceleration<\/small>/);
  assert.match(html, /data-scenario="daisyworld">.*A young sun.*★★☆/);
  assert.match(html, /data-scenario="herds">.*☆☆☆/);
  assert.match(html, /class="chapter locked" disabled aria-label="The first breath, locked">/);
  assert.doesNotMatch(html, /data-scenario="great_oxidation"/);  // Not playable yet.
});

test('the daily challenge names its world and best', () => {
  const html = dailyHtml(campaign, {snowball: 'Snowball'});
  assert.match(html, /data-scenario="snowball" data-seed="417"/);
  assert.match(html, /Snowball · world #417/);
  assert.match(html, /best ★☆☆/);
  assert.match(dailyHtml({daily: {...campaign.daily, stars: 0}}, {}), /not yet played/);
});

test('the campaign counts its stars', () => {
  assert.deepEqual(campaignTotal(campaign), {stars: 2, of: 9});
});

test('a saved game is offered to play on', () => {
  const saved = {game_id: 'daisyworld-3-1f', scenario: 'daisyworld', tick: 12, max_ticks: 40,
                 players: ['Gaia'], minds: 'together', saved_at: '2026-10-15 03:40'};
  const html = savedHtml(saved, {daisyworld: 'Daisyworld'});
  assert.match(html, /data-game="daisyworld-3-1f" data-seat="Gaia"/);
  assert.match(html, /▶ Play on<\/b><span>Daisyworld · turn 13 of 40 · as Gaia<\/span>/);
  assert.match(html, /saved 2026-10-15 03:40 · together minds/);
  assert.equal(savedHtml(null, {}), '');
});


test('a recap tells its chapter, the best stars and the way on', () => {
  const chapter = {number: 1, of: 11, title: 'A young sun', when: '3 billion years ago', stars: 2, best: 2,
                   new_best: true, daily: false, total: 2,
                   next: {scenario: 'herds', title: 'Grazers', when: '3 billion years ago', open: true, opened: true}};
  const html = chapterHtml(chapter);
  assert.match(html, /Chapter 1 of 11: A young sun/);
  assert.match(html, /a new best · 2 of 33 stars in all/);
  assert.match(html, /★★☆/);
  const opened = nextChapterHtml(chapter);
  assert.match(opened, /class="nextchapter primary"/);
  assert.match(opened, /data-scenario="herds"/);
  assert.match(opened, /🔓 Opened: chapter 2, Grazers/);
  const again = {...chapter, new_best: false, daily: true, next: {...chapter.next, opened: false}};
  assert.match(chapterHtml(again), /best ★★☆ · 2 of 33 stars in all · today's challenge/);
  assert.match(nextChapterHtml(again), /^<button class="nextchapter" .*Next: chapter 2, Grazers/);
  const shut = {...chapter, best: 0, next: {...chapter.next, open: false, opened: false}};
  assert.match(nextChapterHtml(shut), /🔒 A star opens chapter 2: Grazers/);
  assert.doesNotMatch(nextChapterHtml(shut), /<button/);
  assert.match(nextChapterHtml({...chapter, next: null}), /last chapter/);
  assert.equal(chapterHtml(null), '');
  assert.equal(nextChapterHtml(null), '');
});

test('the start screen says what is new, with a way to play it, and the build', () => {
  const changes = [{when: '18 Oct', world: 'great_acceleration', text: 'The Steward speaks.'},
                   {when: '17 Oct', text: 'A clearer start.'}];
  const html = whatsNewHtml(changes, {commit: 'abc1234', when: '18 Oct 03:23'},
                            {great_acceleration: 'Great Acceleration'});
  assert.match(html, /What's new<\/h2><p class="small"><b>18 Oct\.<\/b> The Steward speaks\.<\/p>/);
  assert.match(html, /<button class="primary" data-world="great_acceleration">Play Great Acceleration<\/button>/);
  assert.match(html, /Build abc1234 · 18 Oct 03:23/);
  assert.match(html, /A clearer start/);  // The two newest.
  assert.doesNotMatch(whatsNewHtml([...changes, {when: '16 Oct', text: 'Older.'}], null), /Older/);
  assert.equal(whatsNewHtml([], null), '');
  assert.match(whatsNewHtml(null, {commit: 'abc1234', when: 'now'}), /^<p class="small muted">Build abc1234/);
});

test("a change about a people's role is played as that role", () => {
  const html = whatsNewHtml(
    [{when: '19 Oct', world: 'fire_and_iron', role: 'Basalt clans', text: 'War comes.'}],
    null, {fire_and_iron: 'Fire and Iron'});
  assert.match(html, /data-role="Basalt clans">Play Fire and Iron as the Basalt clans</);
});
