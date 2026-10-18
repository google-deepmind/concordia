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

// The game page's panels: below the map, the cards gathered
// into panels to swipe between, with a tab for each panel that has
// something to show.

import {escapeHtml} from './util.js';

export const PANELS = [
  {key: 'now', label: 'Now', cards: ['goalcard', 'civcard', 'holdcard', 'waiting', 'briefcard', 'peoplecard']},
  {key: 'readings', label: 'Readings', cards: ['instcard', 'lagcard', 'readcard', 'ctxcard']},
  {key: 'moves', label: 'Moves', cards: ['movescard']},
  {key: 'story', label: 'Story', cards: ['storycard', 'newscard', 'chroncard']},
  {key: 'world', label: 'World', cards: ['climcard', 'colonycard', 'invite']},
];

/**
 * The tabs: one per panel with a visible card (`shown`: key -> bool), the
 * active one pressed.
 * @param {string} active
 * @param {!Object<string, boolean>} shown
 * @return {string}
 */
export function tabsHtml(active, shown) {
  return PANELS.filter((p) => shown[p.key]).map((p) =>
    `<button role="tab" data-panel="${escapeHtml(p.key)}" aria-selected="${p.key === active}"`
    + `${p.key === active ? ' class="on"' : ''}>${escapeHtml(p.label)}</button>`).join('');
}

/**
 * Which panel a swipe has come to rest on, from the strip's scroll offset.
 * @param {number} scrollLeft
 * @param {number} width
 * @param {!Array<string>} keys
 * @return {string|undefined}
 */
export function panelAt(scrollLeft, width, keys) {
  if (!keys.length || width <= 0) return keys[0];
  const i = Math.round(scrollLeft / width);
  return keys[Math.max(0, Math.min(keys.length - 1, i))];
}
