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

// Feel: a vibration and a few synthesized notes when the
// game needs the player, with a toggle; nothing plays unless the player
// has touched the page (as browsers require).

// Each cue: a vibration pattern (ms on, off, on…) and notes (hertz, seconds).
export const CUES = {
  turn: {vibrate: [25], notes: [[660, 0.08]]},
  urgent: {vibrate: [60, 40, 60], notes: [[440, 0.1], [330, 0.16]]},
  won: {vibrate: [80, 50, 80, 50, 160], notes: [[523, 0.12], [659, 0.12], [784, 0.26]]},
  lost: {vibrate: [220], notes: [[392, 0.2], [311, 0.34]]},
};

// A turn comes back after at least this long a wait: worth a nudge.
export const TURN_WAIT_MS = 4000;

/**
 * Titles of crises with at most 1 turn remaining.
 * @param {?Object} s
 * @return {!Set<string>}
 */
const urgentTitles = (s) => new Set(((s && s.crises) || [])
  .filter((c) => c.turns != null && c.turns <= 1).map((c) => c.title));

/**
 * Which cue a change of state deserves, if any: the game decided (won or
 * lost), a crisis newly at its last turn, or the player's turn back after a
 * wait of `waitedMs`.
 * @param {?Object} prev
 * @param {?Object} state
 * @param {number=} waitedMs
 * @return {?string}
 */
export function cueFor(prev, state, waitedMs = 0) {
  if (!state || state.spectator) return null;
  if (state.outcome && !(prev && prev.outcome)) {
    return state.outcome === 'victory' ? 'won' : 'lost';
  }
  if (!prev || state.outcome) return null;
  const before = urgentTitles(prev);
  if ([...urgentTitles(state)].some((t) => !before.has(t))) return 'urgent';
  const asked = state.question && (!prev.question || prev.question.request_id !== state.question.request_id);
  if (asked && !prev.question && waitedMs >= TURN_WAIT_MS) return 'turn';
  return null;
}

// The toggle: sound and vibration, vibration only, or nothing.
export const FEEL_MODES = ['all', 'buzz', 'off'];

/**
 * Cycles to the next feel mode.
 * @param {string} mode
 * @return {string}
 */
export function nextFeel(mode) {
  return FEEL_MODES[(FEEL_MODES.indexOf(mode) + 1) % FEEL_MODES.length];
}

/**
 * Returns the icon for the given feel mode.
 * @param {string} mode
 * @return {string}
 */
export function feelIcon(mode) {
  return {all: '🔔', buzz: '📳', off: '🔕'}[mode] || '🔔';
}

/**
 * Returns the accessible label for the given feel mode.
 * @param {string} mode
 * @return {string}
 */
export function feelLabel(mode) {
  return {all: 'Sound and vibration on', buzz: 'Vibration only', off: 'Sound and vibration off'}[mode]
    || 'Sound and vibration on';
}
