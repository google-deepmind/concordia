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

// Instrument widgets: each reading drawn from the data
// the role received, beside its words.

import {escapeHtml, fmt, pct, rgb, themeNow} from './util.js';
import {MAP_COLORS, TRADE_PALETTE, tempRgb} from './palette.js';
import {sideHtml} from './stars.js';

/**
 * Formats the change between two readings as an HTML chip.
 * @param {string} label
 * @param {?number|undefined} now
 * @param {?number|undefined} then
 * @param {string=} unit
 * @param {number=} digits
 * @param {boolean=} flip
 * @return {string}
 */
export function trendOf(label, now, then, unit = '', digits = 1, flip = false) {
  if (then == null || now == null) return '';
  const d = now - then;
  if (Math.abs(d) < Math.pow(10, -digits) / 2) return '<small class="flat">steady</small>';
  const up = d > 0;
  return `<small class="${up !== flip ? 'up' : 'down'}">${up ? '▲' : '▼'} ${Math.abs(d).toFixed(digits)}${unit}</small>`;
}

/**
 * Renders a horizontal range gauge as HTML.
 * @param {?number|undefined} value
 * @param {number} lo
 * @param {number} hi
 * @param {?Array<number>=} band
 * @param {?Array<number>=} ticks
 * @return {string}
 */
export function gaugeHtml(value, lo, hi, band, ticks) {
  const at = (v) => Math.max(0, Math.min(100, 100 * (v - lo) / (hi - lo)));
  return '<div class="gauge">' + (band ? `<i style="left:${at(band[0])}%;width:${at(band[1]) - at(band[0])}%"></i>` : '')
    + (value == null ? '' : `<b style="left:${at(value)}%"></b>`)
    + (ticks || []).map((t) => `<span style="left:${at(t)}%">${t}</span>`).join('') + '</div>';
}

/**
 * Renders a labeled progress meter as HTML.
 * @param {string} label
 * @param {?number|undefined} value
 * @param {number} max
 * @param {string} colour
 * @param {?number=} mark
 * @param {?Array<number>=} zone
 * @param {?number=} cap
 * @return {string}
 */
export function meterHtml(label, value, max, colour, mark, zone, cap) {
  const w = Math.max(0, Math.min(100, 100 * (value || 0) / max));
  const z = zone ? `<s style="left:${100 * zone[0] / max}%;width:${100 * (zone[1] - zone[0]) / max}%"></s>` : '';
  return `<div class="meter"><span>${escapeHtml(label)}</span><span class="bar"><i style="width:${w}%;background:${colour}"></i>${z}`
    + (mark != null ? `<u style="left:${100 * mark / max}%"></u>` : '')
    + (cap != null ? `<u class="cap" style="left:${Math.min(99, 100 * cap / max)}%"></u>` : '') + `</span><span>${fmt(value, 2)}</span></div>`;
}

export const INSTRUMENTS = {
  'climate.global'(d, prev) {
    const band = d.band_c, t = d.temp_c;
    const lo = Math.min(band[0] - 15, t - 3), hi = Math.max(band[1] + 15, t + 3);
    const lats = (d.bands || []).slice().reverse().map((b) => {  // North first.
      const c = tempRgb(b.temp_c);
      return `<span style="background:${rgb(c)}" title="${escapeHtml(b.band)}">${escapeHtml(b.band)}<b>${Math.round(b.temp_c)}°</b>${b.ice > 0.05 ? `❄ ${pct(b.ice)}` : ''}</span>`;
    }).join('');
    const out = t < band[0] || t > band[1];
    const chips = [`sun ×${fmt(d.luminosity, 2)}`, `albedo ${fmt(d.albedo, 2)}`]
      .concat(d.co2 != null ? [`CO₂ ×${fmt(d.co2, 2)}`] : [], d.ice > 0.005 ? [`ice ${pct(d.ice)}`] : [],
              d.oxygen_pal != null ? [`O₂ ${fmt(d.oxygen_pal, 3)}`] : []);
    return {wide: true, html: `<div class="big"${out ? ' style="color:#c2412d"' : ''}>${fmt(t)} °C${trendOf('t', t, prev && prev.temp_c, '°')}</div>`
      + gaugeHtml(t, lo, hi, band, [band[0], band[1]])
      + `<div class="lats">${lats}</div><div class="ichips">${chips.map((c) => `<span>${c}</span>`).join('')}</div>`};
  },
  'bio.census'(d, prev) {
    const theme = themeNow(), look = d.look || {}, keys = Object.keys(d.cover || {});
    const colour = (k) => (look[k] ? rgb(look[k][theme] || look[k].light) : '#888');
    const bare = rgb(MAP_COLORS[theme].bare);
    const stack = (cover, rest) => '<div class="stack">' + keys.map((k) => `<i style="width:${100 * (cover[k] || 0)}%;background:${colour(k)}"></i>`).join('')
      + (rest != null ? `<i style="width:${100 * rest}%;background:${bare}"></i>` : '') + '</div>';
    const rest = (cover) => Math.max(0, 1 - keys.reduce((a, k) => a + (cover[k] || 0), 0));
    const rows = (d.bands || []).slice().reverse().map((b) => `<span>${escapeHtml(b.band)}</span>${stack(b.cover || {}, rest(b.cover || {}))}<span>${Math.round(b.temp_c)}°</span>`).join('');
    const chips = keys.map((k) => {
      const g = (d.growth || {})[k];
      const arrow = g == null ? '' : g > 1.0 ? ' <b class="grow">▲</b>' : g < 0.97 ? ' <b class="shrink">▼</b>' : '';
      return `<span><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${colour(k)};margin-right:4px"></i>`
        + `${escapeHtml((look[k] && look[k].name) || k)} ${pct(d.cover[k])}${arrow}</span>`;
    }).concat([`<span>bare ${pct(d.bare || 0)}</span>`]);
    return {wide: true, html: stack(d.cover || {}, d.bare) + `<div class="ichips">${chips.join('')}</div>`
      + (rows ? `<div class="latrows">${rows}</div>` : '')};
  },
  'gaia.indices'(d, prev) {
    return {html: `<div class="big">${fmt(d.omega)} Ω${trendOf('o', d.omega, prev && prev.omega, '', 1)}</div>`
      + meterHtml('biodiversity', d.biodiversity, 1, 'var(--good)')
      + meterHtml('stability', d.stability, 1, '#2b8fd9')
      // The arks' seed vault, against the goal's mark.
      + (d.vault ? meterHtml('seed vault', d.vault.held, Math.max(0.3, 1.5 * d.vault.need), '#b07a2a', d.vault.need) : '')
      // Side objectives, as of this reading.
      + (d.side && d.side.length ? `<ul class="stars">${sideHtml(d.side, d.tick)}</ul>` : '')
      + (d.instruments ? `<p class="none">${escapeHtml(d.instruments)}</p>` : '')};
  },
  'cycles.carbon'(d, prev) {
    // A frozen world's tipping zone: from where a dark haze may tip the ice
    // to where it surely does (the scenario's measured `tip_co2`).
    // A goal's cap on the air (`cap_co2`): a red line, and how far to it.
    const tip = d.tipping_co2, co2 = d.relative_co2, cap = d.cap_co2;
    const top = tip ? Math.max(2, tip[1] * 1.2, co2) : 2;
    let verdict = !tip ? '' : co2 >= tip[1] ? '<span class="tip go">a haze should tip the ice · not under ash</span>'
      : co2 >= tip[0] ? `<span class="tip">a haze may tip the ice · surer from ×${fmt(tip[1], 1)}</span>`
      : `<span class="tip">too thin to tip · a haze may from ×${fmt(tip[0], 1)}</span>`;
    if (cap != null) {
      verdict += co2 >= cap ? `<span class="tip go">over the cap ×${fmt(cap, 2)}</span>`
        : `<span class="tip">cap ×${fmt(cap, 2)} · ${fmt(cap - co2, 2)} to go</span>`;
    }
    return {wide: !!tip || cap != null, html: `<div class="big">CO₂ ×${fmt(co2, 2)}${trendOf('c', co2, prev && prev.relative_co2, '', 2)}</div>`
      + meterHtml('vs the start', co2, top, co2 > (cap || 1.3) ? '#c2412d' : '#d98a2b', 1, tip, cap)
      + `<div class="ichips">${verdict}<span>greenhouse ${fmt(d.forcing_w_m2)} W/m²</span></div>`};
  },
  'ocean.global'(d) {
    const chips = [`sea ice ${pct(d.sea_ice || 0)}`, `sea ${fmt(d.mean_sst_c)} °C`, `pH ${fmt(d.ph, 2)}`]
      .concat(d.dead_zone > 0.005 ? [`dead zone ${pct(d.dead_zone)}`] : []);
    return {html: `<div class="big">${fmt(d.mean_sst_c)} °C <small class="flat">at sea</small></div>`
      + meterHtml('sea ice', d.sea_ice || 0, 1, '#8fb8d8') + `<div class="ichips">${chips.map((c) => `<span>${c}</span>`).join('')}</div>`};
  },
  'civ.census'(d) {
    const rows = Object.values(d.polities || {}).map((p) => {
      const fed = p.food_ratio == null ? 1 : p.food_ratio, s = p.stocks || {};
      return `<span>${escapeHtml(p.name)}</span><span>${fmt(p.population)}</span>`
        + `<span class="bar"><i style="width:${Math.min(100, 100 * fed)}%;background:${fed < 0.95 ? '#c2412d' : 'var(--good)'}"></i></span>`
        + `<span>${Math.round(s.grain || 0)}</span>`;
    }).join('');
    return {wide: true, html: `<div class="prow"><small class="flat"></small><small class="flat">people</small><small class="flat">fed</small><small class="flat">grain</small>${rows}</div>`};
  },
  'econ.prices'(d, prev) {
    const was = (prev && prev.prices) || {};
    const chips = Object.entries(d.prices || {}).map(([k, v]) => {
      const t = was[k] == null || Math.abs(v - was[k]) < 0.005 ? '' : v > was[k] ? ' <b class="up">▲</b>' : ' <b class="down">▼</b>';
      return `<span>${escapeHtml(k)} ${fmt(v, v < 10 ? 1 : 0)}${t}</span>`;
    });
    return {wide: true, html: `<div class="ichips">${chips.join('')}</div>`};
  },
  'civ.treaties'(d) {
    // Peace pacts (where conquest is played) cap nothing: a dove, no cap.
    const caps = Object.entries(d.caps || {}).filter(([k]) => k !== 'peace').map(([k, v]) => `<span>${escapeHtml(k)} cap ${fmt(v, 1)}</span>`);
    const live = (d.treaties || []).map((t) => `<span>${t.kind === 'peace' ? '🕊️' : '🤝'} ${escapeHtml(typeof t === 'string' ? t : (t.members || t.parties || []).join(' & ') || JSON.stringify(t))}</span>`);
    const embargoes = (d.embargoes || []).map((e) => `<span class="up">⛔ ${escapeHtml(Array.isArray(e) ? e.join(' & ') : String(e))}</span>`);
    return {html: `<div class="big">${(d.treaties || []).length} <small class="flat">in force</small></div>`
      + `<div class="ichips">${live.concat(embargoes, caps).join('')}</div>`};
  },
  'media.gazette'(d) {
    return {html: '<p class="none">consent for coal</p>' + Object.entries(d.consent || {}).map(([k, v]) => meterHtml(k, v, 1, v < 0.85 ? '#d98a2b' : 'var(--good)')).join('')
      + `<div class="ichips"><span>${(d.broadsheets || []).length} broadsheets</span></div>`};
  },
  news(d) {  // A digest: each item with its turn and a sign of its kind.
    const sign = (t) => (/fire|burn/i.test(t) ? '🔥' : /erupt|volcan|ash/i.test(t) ? '🌋' : /ice|snow|froz/i.test(t) ? '❄'
      : /disease|plague|sicken|pest/i.test(t) ? '🦠' : /town|found|settle/i.test(t) ? '🏠' : /famine|hungry|starv/i.test(t) ? '🥣'
      : /treaty|pact|embargo/i.test(t) ? '🤝' : /spread|retreat|evolv|cover|bloom/i.test(t) ? '🌿' : '•');
    return {wide: true, html: (d.items || []).slice().reverse().map((n) =>
      `<div class="newsrow"><span class="turnchip">${n.tick == null ? '' : `t${n.tick}`}</span><span>${sign(n.text)}</span><span>${escapeHtml(n.text)}</span></div>`).join('')};
  },
  genome(d) {
    const chips = [`thrives at ${fmt(d.optimum_c)} °C`]
      .concat(d.albedo != null ? [`petals reflect ${pct(d.albedo)}`] : [], d.white_diet != null ? [`eats white petals ${pct(d.white_diet)}`] : [],
              (d.evolving || []).map((e) => `evolving: ${e}`));
    return {html: `<div class="ichips">${chips.map((c) => `<span>${escapeHtml(c)}</span>`).join('')}</div>`};
  },
  'bio.pacts'(d) {
    const pacts = (d.pacts || []).map((p) => `<span>🤝 ${escapeHtml(typeof p === 'string' ? p : (p.members || p.species || []).join(' & ') || JSON.stringify(p))}</span>`);
    return {html: `<div class="big">${(d.pacts || []).length} <small class="flat">pacts · ${d.pending ? d.pending.length : 0} offered</small></div>`
      + (pacts.length ? `<div class="ichips">${pacts.join('')}</div>` : '')};
  },
  'colony.status'(d, prev) {
    const t = d.temp_c, goal = d.target_temp_c;
    const chips = Object.entries(d.stocks || {}).map(([k, v]) => `<span>${escapeHtml(k)} ${fmt(v)}</span>`);
    const labor = Object.entries(d.labor || {}).sort((a, b) => b[1] - a[1]);
    return {wide: true, html: `<div class="big">${fmt(t)} °C${trendOf('s', t, prev && prev.temp_c, '°')} <small class="flat">sky · ${fmt(d.colonists)} colonists</small></div>`
      + gaugeHtml(t, Math.min(t, -70) - 5, Math.max(goal, t) + 15, [goal, goal + 15], [goal])
      + '<div class="stack">' + labor.map(([k, v], i) => `<i style="width:${100 * v}%;background:${rgb(TRADE_PALETTE[i % TRADE_PALETTE.length])}" title="${escapeHtml(k)}"></i>`).join('') + '</div>'
      + `<div class="ichips">${labor.map(([k, v]) => `<span>${escapeHtml(k.replace('_', ' '))} ${pct(v)}</span>`).concat(chips).join('')}</div>`};
  },
};
