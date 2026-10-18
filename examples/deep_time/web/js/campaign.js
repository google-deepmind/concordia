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

// The campaign through deep time and the daily challenge.

import {escapeHtml} from './util.js';

/**
 * Renders `n` of 3 stars.
 * @param {number} n
 * @return {string}
 */
const line = (n) => '★'.repeat(n) + '☆'.repeat(Math.max(0, 3 - n));

/**
 * The chapters as buttons: open ones with their best stars, closed ones
 * locked until the chapter before has a star.
 * @param {?Object} campaign
 * @return {string}
 */
export function campaignHtml(campaign) {
  return (campaign && campaign.chapters || []).map((c, i) => {
    const head = `<span class="cnum">${i + 1}</span><span class="ctext"><b>${escapeHtml(c.title)}</b>`
      + `<small>${escapeHtml(c.when)}${c.world ? ' · ' + escapeHtml(c.world) : ''}</small></span>`;
    if (!c.open) return `<button class="chapter locked" disabled aria-label="${escapeHtml(c.title)}, locked">${head}<span class="cstars">🔒</span></button>`;
    return `<button class="chapter" data-scenario="${escapeHtml(c.scenario)}">${head}`
      + `<span class="cstars">${line(c.stars)}</span></button>`;
  }).join('');
}

/**
 * Today's challenge as a button (it plays that world, seed and all).
 * @param {?Object} campaign
 * @param {?Object<string, string>=} titles
 * @return {string}
 */
export function dailyHtml(campaign, titles) {
  const d = campaign && campaign.daily;
  if (!d) return '';
  const name = (titles && titles[d.scenario]) || d.title;
  return `<button class="daily" data-scenario="${escapeHtml(d.scenario)}" data-seed="${Number(d.seed)}">`
    + `<b>Today's challenge</b><span>${escapeHtml(name)} · world #${Number(d.seed)}</span>`
    + `<span class="cstars">${d.stars ? 'best ' + line(d.stars) : 'not yet played'}</span></button>`;
}

/**
 * How many stars of how many the campaign holds.
 * @param {?Object} campaign
 * @return {!Object}
 */
export function campaignTotal(campaign) {
  const all = (campaign && campaign.chapters) || [];
  return {stars: all.reduce((a, c) => a + c.stars, 0), of: 3 * all.length};
}

/**
 * A saved game to play on, as a button at the top of the
 * start screen; empty without one.
 * @param {?Object} saved
 * @param {?Object<string, string>=} titles
 * @return {string}
 */
export function savedHtml(saved, titles) {
  if (!saved || !saved.game_id) return '';
  const name = (titles && titles[saved.scenario]) || saved.scenario;
  const who = (saved.players || []).join(' and ');
  return `<button class="resume" data-game="${escapeHtml(saved.game_id)}" data-seat="${escapeHtml((saved.players || [])[0] || '')}">`
    + `<b>▶ Play on</b><span>${escapeHtml(name)} · turn ${Number(saved.tick) + 1} of ${Number(saved.max_ticks)}`
    + `${who ? ` · as ${escapeHtml(who)}` : ''}</span>`
    + `<small>saved ${escapeHtml(saved.saved_at || '')}${saved.minds && saved.minds !== 'mock' ? ` · ${escapeHtml(saved.minds)} minds` : ''}</small></button>`;
}

/**
 * A decided game's place in the campaign, for its recap:
 * its chapter, the best stars there, and the stars in all. Empty for a
 * game no chapter has (a watched one).
 * @param {?Object} chapter
 * @return {string}
 */
export function chapterHtml(chapter) {
  if (!chapter) return '';
  const best = chapter.new_best ? 'a new best' : `best ${line(chapter.best)}`;
  return `<div class="recapchapter"><span class="cnum">${Number(chapter.number)}</span>`
    + `<span class="ctext"><b>Chapter ${Number(chapter.number)} of ${Number(chapter.of)}: ${escapeHtml(chapter.title)}</b>`
    + `<small>${best} · ${Number(chapter.total)} of ${3 * Number(chapter.of)} stars in all`
    + `${chapter.daily ? " · today's challenge" : ''}</small></span>`
    + `<span class="cstars">${line(chapter.best)}</span></div>`;
}

/**
 * The way on from a recap: the next chapter as a button once it is open
 * (marked when this game's star opened it), or what opens it.
 * @param {?Object} chapter
 * @return {string}
 */
export function nextChapterHtml(chapter) {
  const next = chapter && chapter.next;
  if (!chapter) return '';
  if (!next) return '<p class="small muted">The last chapter of deep time.</p>';
  const n = Number(chapter.number) + 1;
  if (!next.open) return `<p class="small muted">🔒 A star opens chapter ${n}: ${escapeHtml(next.title)}.</p>`;
  return `<button class="nextchapter${next.opened ? ' primary' : ''}" id="recapnext" data-scenario="${escapeHtml(next.scenario)}">`
    + `${next.opened ? '🔓 Opened: ' : 'Next: '}chapter ${n}, ${escapeHtml(next.title)} ▶</button>`;
}

/**
 * What changed for players lately, with a button to the world it changed,
 * and which build this is (`changes` and `build` from api/campaign).
 * @param {?Array<!Object>} changes Newest first: {when, text, world?}.
 * @param {?Object} build {commit, when}.
 * @param {?Object<string, string>=} titles Scenario key -> title.
 * @return {string}
 */
export function whatsNewHtml(changes, build, titles) {
  const shown = (changes || []).slice(0, 2);
  const stamp = build && build.commit
    ? `<p class="small muted">Build ${escapeHtml(build.commit)} · ${escapeHtml(build.when)}</p>` : '';
  if (!shown.length) return stamp;
  const key = (shown.find((c) => c.world && titles && titles[c.world]) || {}).world;
  return '<h2>What\'s new</h2>' + shown.map((c) =>
    `<p class="small"><b>${escapeHtml(c.when)}.</b> ${escapeHtml(c.text)}</p>`).join('')
    + (key ? `<button class="primary" data-world="${escapeHtml(key)}">Play ${escapeHtml(titles[key])}</button>` : '')
    + stamp;
}
