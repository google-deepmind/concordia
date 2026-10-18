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

// Crises with countdowns: trouble the role can see coming,
// from its own readings, with the moves that answer it.

import {escapeHtml, shortMove} from './util.js';

export const CRISIS_ICONS = {band: '🌡', life: '🌿', species: '🌼', herds: '🐾', famine: '🌾', unrest: '🦗', deadline: '⏳', side: '◇'};

/**
 * The countdown chip: '3 turns left', 'in 2 turns' (trouble ahead), or ''.
 * @param {!Object} c
 * @return {string}
 */
export function countdown(c) {
  if (c.turns == null) return '';
  if (c.turns <= 0) return 'now';
  const n = `${c.turns} turn${c.turns === 1 ? '' : 's'}`;
  return c.kind === 'band' && !/^Too /.test(c.title) ? `in ${n}` : `${n} left`;
}

/**
 * The strip of crises, the most pressing first; each answer is a button
 * carrying the move (the page opens the move sheet at it).
 * @param {?Array<!Object>} crises
 * @return {string}
 */
export function crisesHtml(crises) {
  return (crises || []).map((c) => {
    const due = countdown(c);
    const answers = (c.answers || []).map((a) =>
      `<button class="answer" data-move="${escapeHtml(a)}">${escapeHtml(shortMove(a))}</button>`).join('');
    return `<div class="crisis ${escapeHtml(c.kind)}${c.turns != null && c.turns <= 2 ? ' urgent' : ''}">`
      + `<span class="cicon" aria-hidden="true">${CRISIS_ICONS[c.kind] || '⚠'}</span>`
      + `<div class="cbody"><b>${escapeHtml(c.title)}</b>${due ? ` <span class="due">${escapeHtml(due)}</span>` : ''}`
      + `<small>${escapeHtml(c.detail)}</small>${answers ? `<div class="answers">${answers}</div>` : ''}</div></div>`;
  }).join('');
}

/**
 * Crises on the map: where each shows, as its role can know
 * it. A people's at its capital, the colony's at its site, a herd's around
 * its herds, a life form's along the latitude its census last counted most
 * of it, the whole world's as a glow around the map's edge. `map` is the
 * map payload's part the page keeps (rows, polities, settlements, herds,
 * site); `seen` the crises' keys shown last turn, so a new one can arrive.
 * @param {?Array<!Object>} crises
 * @param {!Object} map
 * @param {!Set<string>=} seen
 * @return {!Array<!Object>}
 */
export function crisisMarks(crises, map, seen = new Set()) {
  const marks = [];
  for (const c of crises || []) {
    const w = c.where;
    if (!w) continue;
    const key = crisisKey(c), base = {key, kind: c.kind, icon: CRISIS_ICONS[c.kind] || '⚠', title: c.title,
      urgent: c.turns != null && c.turns <= 2, fresh: !seen.has(key)};
    if (w.world) {
      marks.push({...base, type: 'edge', hot: w.world === 'hot'});
    } else if (w.lat != null) {
      // Rows count from the north pole (0) to the south (rows).
      marks.push({...base, type: 'band', row: (90 - w.lat) / 180 * map.rows, half: map.rows / 12});
    } else if (w.people) {
      const p = (map.polities || []).findIndex((x) => x.name === w.people);
      const capital = (map.settlements || []).find((s) => s[0] === p && s[4]);
      if (capital) marks.push({...base, type: 'ring', row: capital[2], col: capital[3]});
    } else if (w.site && map.site) {
      marks.push({...base, type: 'ring', row: map.site[0], col: map.site[1]});
    } else if (w.herd) {
      const kinds = (map.species || []).map((s) => s.kind);
      const herds = (map.herds || []).filter((h) => kinds[h[0]] === w.herd);
      if (herds.length) {  // The herds' middle, the short way round in longitude.
        const cols = map.cols, angle = (c) => 2 * Math.PI * c / cols;
        const sx = herds.reduce((a, h) => a + Math.cos(angle(h[3])), 0);
        const sy = herds.reduce((a, h) => a + Math.sin(angle(h[3])), 0);
        const col = ((Math.atan2(sy, sx) / (2 * Math.PI)) * cols + cols) % cols;
        const row = herds.reduce((a, h) => a + h[2], 0) / herds.length;
        marks.push({...base, type: 'ring', row, col});
      }
    }
  }
  return marks;
}

/**
 * A crisis's identity from turn to turn: its kind and place, not its words
 * (the numbers in a title change every turn).
 * @param {!Object} c
 * @return {string}
 */
export function crisisKey(c) {
  return `${c.kind}:${JSON.stringify(c.where || null)}`;
}

/**
 * What the age asks of a people, for its card: the good
 * all peoples make together as a bar, the pace the turns left ask for and
 * last turn's making, and the people's numbers against the floor
 * under which it breaks and ends the age. Empty without a shared goal.
 * @param {?Object} sh
 * @return {string}
 */
export function sharedHtml(sh) {
  if (!sh) return '';
  let html = '';
  if (sh.good) {
    html += `<div class="hud one"><div class="row"><span class="label">${escapeHtml(sh.label || sh.good)}</span>`
      + `<span>${sh.made} of ${sh.need} by turn ${sh.by}</span><span class="bar${sh.made >= sh.need ? '' : ' off'}">`
      + `<i style="width:${Math.min(100, Math.round(100 * sh.made / sh.need))}%"></i></span></div></div>`;
    if (sh.made < sh.need && sh.pace != null) {
      // Behind, near the end: last turn's making short of the pace needed.
      const late = sh.last != null && sh.last < sh.pace && sh.left <= 10;
      html += `<p class="small">${Math.round((sh.need - sh.made) * 10) / 10} to go in ${sh.left} turns: about ${sh.pace} a turn among all peoples`
        + (sh.last != null ? `; last turn ${late ? `<span class="hungry">${sh.last}</span>` : sh.last}` : '')
        + '.</p>';
    }
  } else if (sh.kind === 'seed_ship') {
    html += `<p class="small">A seed ship must leave from a world of biodiversity ${sh.min_biodiversity} or more by turn ${sh.by} (${sh.left} turns left).</p>`;
  }
  if (sh.held && !sh.held.short.length) {  // Hardly any hands make the good.
    html += `<p class="small road">🔗 Hardly anyone works to <b>${escapeHtml(sh.held.recipe)}</b>, so we make little ${escapeHtml(sh.held.good)}</p>`;
  } else if (sh.held) {  // Last generation's weakest link in the chain.
    const chain = [sh.held.good, ...sh.held.short].map(escapeHtml).join(' ← ');
    html += `<p class="small road">🔗 Held back: ${chain} · `
      + (sh.held.recipe ? `more hands to <b>${escapeHtml(sh.held.recipe)}</b>` : `we make no ${escapeHtml(sh.held.short.at(-1))}: buy it`)
      + '</p>';
  }
  if (sh.next_tech) {  // The tech that would most help, while research is short.
    const t = sh.next_tech;
    html += `<p class="small road">📜 Next for our ${escapeHtml(sh.good || 'goal')}: <b>${escapeHtml(t.label)}</b> (${escapeHtml(t.does)}) · `
      + `${t.research} of ${t.cost} research: study and teach bring it nearer</p>`;
  }
  if (sh.floor) {
    const pct = Math.round(100 * sh.numbers), floor = Math.round(100 * sh.floor);
    const close = sh.numbers < sh.floor + 0.2;
    html += `<p class="small${close ? ' hungry' : ''}">Our numbers: ${pct}% of the start. A people under ${floor}% breaks, and the age ends for all.</p>`;
  }
  return html;
}

/**
 * A tag for a move that matters most this turn: the answer
 * to the most urgent crisis, or the weakest link of a people's chain.
 * @param {?Object} p
 * @return {string}
 */
export function pressingTag(p) {
  if (!p) return '';
  if (p.kind === 'link') return '🔗 weakest link';
  if (p.kind === 'learn') return '📜 worth learning';
  if (p.kind === 'famine') return `${CRISIS_ICONS.famine} feeds us`;
  return `${CRISIS_ICONS[p.kind] || '⚠'} answers it`;
}

/**
 * The question as the sheet heads it: a people's question ends with what
 * matters most for a mind reading it; on the phone the moves carry
 * that as tags, so the heading keeps the question alone.
 * @param {?string} callToAction
 * @return {string}
 */
export function askWords(callToAction) {
  return String(callToAction || '').split(/ (?=Most pressing now: |Our weakest link: )/)[0];
}
