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

// Minds: who plays the roles no person plays, as the page
// offers them and says what they cost.

// Name and blurb of each kind of minds.
import {escapeHtml} from './util.js';

export const MINDS_OFFER = {
  mock: ['Mock minds', 'free · instant · the same game every time'],
  together: ['Live · Together', 'language models online'],
  ollama: ['Live · Ollama', 'small models on this Mac · slower'],
};

/**
 * The blurb under a choice of minds (`offer` from api/minds): what a game
 * costs, how fast a turn goes and how often such games were won, as
 * measured; the budget left for priced minds.
 * @param {string} minds
 * @param {?Object} offer
 * @param {?string=} scenario
 * @return {string}
 */
export function mindsOffer(minds, offer, scenario) {
  const here = scenario && offer && offer.by_scenario && (offer.by_scenario[scenario] || {})[minds];
  const m = here || (offer && offer.measured && offer.measured[minds]);
  if (!m) {
    if (minds === 'together' && offer) {
      return `about $${offer.per_game_usd.toFixed(2)} a game · $${offer.left_usd.toFixed(2)} left`;
    }
    return (MINDS_OFFER[minds] || [minds, ''])[1];
  }
  const left = minds === 'together' ? ` · $${offer.left_usd.toFixed(2)} left` : '';
  return `${costWords(m)} · ${paceWords(m)} · ${wonWords(m, !!here)}${left}`;
}

/**
 * Formats per-game cost in words.
 * @param {!Object} m
 * @return {string}
 */
function costWords(m) {
  return m.usd_per_game > 0 ? `≈ $${m.usd_per_game.toFixed(2)} a game` : 'free';
}

/**
 * Formats per-turn pace in words.
 * @param {!Object} m
 * @return {string}
 */
function paceWords(m) {
  if (m.seconds_per_turn == null) return 'speed unmeasured';
  return m.seconds_per_turn < 1 ? 'instant' : `${Math.round(m.seconds_per_turn)} s a turn`;
}

/**
 * Few games read as a count ('won 2 of 3 here'), many as a share.
 * @param {!Object} m
 * @param {boolean} here
 * @return {string}
 */
function wonWords(m, here) {
  const won = m.games < 10 ? `${Math.round(m.wins * m.games)} of ${m.games}` : `${Math.round(m.wins * 100)}% of ${m.games}`;
  return `won ${won}${here ? ' here' : ''}`;
}

/**
 * The minds side by side in one world: what a game costs,
 * how long a turn takes and how often the game was won with them there.
 * @param {?Object} offer
 * @param {string} scenario
 * @param {?string=} title
 * @return {string}
 */
export function mindsTable(offer, scenario, title) {
  const here = (offer && offer.by_scenario && offer.by_scenario[scenario]) || {};
  const rows = Object.keys(MINDS_OFFER).filter((k) => here[k]).map((k) => {
    const m = here[k];
    return `<tr><th scope="row">${MINDS_OFFER[k][0].replace('Live · ', '')}</th><td>${costWords(m)}</td>`
      + `<td>${paceWords(m)}</td><td>${Math.round(m.wins * m.games)} of ${m.games}</td></tr>`;
  });
  if (rows.length < 2) return '';
  return `<details class="mindscmp"><summary>Compare minds in ${escapeHtml(title || scenario)}</summary>`
    + '<table><thead><tr><th></th><th>A game</th><th>A turn</th><th>Won</th></tr></thead>'
    + `<tbody>${rows.join('')}</tbody></table>`
    + '<p class="small muted">The other roles played by each; your role by the skilled bot or a person.</p></details>';
}

/**
 * The short name of a game's minds (`spend` from the game's state).
 * @param {!Object} spend
 * @return {string}
 */
export function mindsName(spend) {
  return (MINDS_OFFER[spend.minds] || [spend.minds])[0].replace('Live · ', '');
}

/**
 * A game's minds in a few words, for its title: what they cost so far, or
 * that live minds were out of reach and the mock is playing.
 * @param {!Object} spend
 * @return {string}
 */
export function mindsBadge(spend) {
  if (/fallback/i.test(spend.label || '')) return `${mindsName(spend)} unreachable · mock`;
  return `${mindsName(spend)} $${spend.usd.toFixed(2)}`;
}

/**
 * What a game's minds did and cost, in words.
 * @param {!Object} spend
 * @return {string}
 */
export function mindsWords(spend) {
  const tokens = spend.tokens >= 1000 ? `${Math.round(spend.tokens / 1000)}k` : `${spend.tokens}`;
  const fell = /fallback/i.test(spend.label || '') ? ' (unreachable: the mock played)' : '';
  return `Minds: ${mindsName(spend)}${fell}: ${spend.calls} calls, ${tokens} tokens, about $${spend.usd.toFixed(2)}`;
}
