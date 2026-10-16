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

// Pushes: what a move does, as the move sheet shows
// it.

/**
 * A push that moves the climate 15 °C or more (as measured on this world),
 * or wrecks it: drawn in red.
 * @param {string} push
 * @return {boolean}
 */
export function strongPush(push) {
  if (/freezes|boils|all life|swings/.test(push)) return true;
  const degrees = [...push.matchAll(/(\d+(?:\.\d+)?) °C/g)].map((m) => Number(m[1]));
  return Math.max(0, ...degrees) >= 15;
}

/**
 * A labor move's push in brief, for its chip: the main good it makes, and
 * smoke or soot if they rise. E.g. {gain: 'iron +4.6', smoke: 'smoke +3.2'}.
 * @param {string} push
 * @return {!Object}
 */
export function laborBrief(push) {
  if (!push) return {gain: '', smoke: ''};
  const parts = push.split('; ');
  const gain = parts[0].replace(/ a generation$/, '').split(', ')[0];
  const smoke = parts.slice(1).join(', ').split(', ')
    .find((p) => /smoke|soot/.test(p) && p.includes('+')) || '';
  return {gain, smoke};
}

/**
 * The Advisor's tag on the move it recommends: its
 * counsel reads the world as it stood when the Advisor was asked, a slow
 * mind's turns ago, so the tag says how old. E.g. 'Advisor · 2 turns old'.
 * @param {?Object} advice
 * @return {string}
 */
export function adviceTag(advice) {
  if (!advice) return '';
  return advice.age > 0 ? `Advisor · ${advice.age} turn${advice.age === 1 ? '' : 's'} old` : 'Advisor';
}
