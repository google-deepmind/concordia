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

// Tests of instruments.js: `node --test` (run by web/js_test.py, which
// passes payloads published by real games in DEEP_TIME_PAYLOADS).

import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';

import {INSTRUMENTS, meterHtml} from './instruments.js';

const carbon = (co2, extra = {}) => INSTRUMENTS['cycles.carbon'](
  {relative_co2: co2, forcing_w_m2: 2.0, ...extra}, null).html;

test('the air gauge says where a haze tips a frozen world', () => {
  const zone = {tipping_co2: [1.3, 2.0]};
  assert.match(carbon(1.1, zone), /too thin to tip/);
  assert.match(carbon(1.5, zone), /a haze may tip the ice/);
  assert.match(carbon(2.1, zone), /a haze should tip the ice · not under ash/);
  assert.match(carbon(1.5, zone), /<s style="left:/);  // The zone, hatched.
});

test('the air gauge shows a goal\'s cap and the way to it', () => {
  assert.match(carbon(1.11, {cap_co2: 1.3}), /cap ×1\.30 · 0\.19 to go\b/);
  assert.match(carbon(1.4, {cap_co2: 1.3}), /over the cap ×1\.30/);
  assert.match(carbon(1.1, {cap_co2: 1.3}), /class="cap"/);
  assert.equal(INSTRUMENTS['cycles.carbon']({relative_co2: 1, forcing_w_m2: 0}, null).wide, false);
});

test('a meter keeps its fill within the bar', () => {
  assert.match(meterHtml('x', 9, 2, 'red'), /width:100%/);
  assert.match(meterHtml('x', -1, 2, 'red'), /width:0%/);
});

test('every widget draws every payload real games publish', () => {
  const path = process.env.DEEP_TIME_PAYLOADS;
  if (!path) return;  // Run alone: nothing published to draw.
  const payloads = JSON.parse(readFileSync(path, 'utf8'));
  let drawn = 0;
  for (const [topic, list] of Object.entries(payloads)) {
    const draw = INSTRUMENTS[topic];
    if (!draw) continue;
    for (const d of list) {
      const w = draw(d, d);
      assert.ok(w && typeof w.html === 'string' && w.html.length > 0, topic);
      assert.doesNotMatch(w.html, /undefined|NaN/, topic);
      drawn++;
    }
  }
  assert.ok(drawn >= 8, `only ${drawn} payloads drawn`);
});

test('a peace pact shows as a dove, with no cap', () => {
  const html = INSTRUMENTS['civ.treaties']({
    treaties: [{kind: 'peace', parties: ['Amber folk', 'Basalt clans'], ticks_left: 9},
               {kind: 'coal', parties: ['Amber folk', 'Cedar league'], ticks_left: 3}],
    caps: {coal: 4, peace: 0},
  }, null).html;
  assert.match(html, /🕊️ Amber folk &amp; Basalt clans/);
  assert.match(html, /🤝 Amber folk &amp; Cedar league/);
  assert.doesNotMatch(html, /peace cap/);
});
