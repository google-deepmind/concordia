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

// The page's colours: map palettes and ramps, one scale
// of temperature for every world, and the trades' colours.

import {ramp} from './util.js';

// The map: 40 × 30 tiles over the 48 regions, north at the top. Terrain is
// painted once per game (and theme) to an offscreen canvas; ice and the
// chosen layer are composited over it on every refresh.
export const MAP_COLORS = {
  light: {deep: [22, 58, 108], sea: [40, 98, 156], shelf: [70, 142, 176], sand: [214, 200, 150],
          low: [112, 152, 84], high: [146, 126, 92], rock: [178, 172, 160], ice: [238, 244, 250], seaice: [176, 206, 230],
          temps: [[34, 62, 150], [112, 164, 222], [238, 234, 204], [236, 150, 68], [168, 36, 40]],
          char: [58, 46, 40], ember: [255, 120, 40], bloom: [124, 118, 48],
          smog: [92, 76, 58], plume: [98, 168, 84],
          bare: [208, 190, 142], green: [36, 128, 60]},
  dark: {deep: [10, 30, 62], sea: [22, 58, 100], shelf: [36, 92, 116], sand: [142, 130, 92],
         low: [70, 104, 56], high: [104, 90, 66], rock: [128, 124, 116], ice: [190, 202, 214], seaice: [96, 132, 168],
         temps: [[40, 64, 140], [86, 132, 186], [190, 184, 150], [204, 124, 56], [160, 44, 44]],
         char: [24, 18, 16], ember: [255, 110, 36], bloom: [104, 98, 36],
         smog: [176, 150, 112], plume: [96, 170, 84],
         bare: [126, 114, 84], green: [56, 148, 80]},
};

// One physical scale for every world (blues below freezing, reds in the
// heat), so a glance compares a snowball with a hothouse.
export const TEMP_SCALE = [-40, 45];

export const FIRE_RAMP = [[250, 222, 120], [238, 150, 48], [200, 56, 30], [120, 20, 24]];

export const FLAME = [[196, 44, 24], [240, 104, 32], [255, 176, 64], [255, 230, 150]];

// The seasons layer: each tile's annual swing, from mild (maritime) to wild
// (continental), and a year playing over it (snow and sea ice coming and
// going with each hemisphere's winter).
export const SEASON_RAMP = [[38, 112, 132], [186, 160, 84], [128, 58, 132]];

export const SEASON_MAX = 40;  // °C of annual range at the far end of the ramp.

// News spreading from where it happened: rings that grow at
// the couriers' pace and fade, so who has heard what can be read off the
// map. Fires and herds' news in ember, the peoples' in gold.
export const RIPPLE_COLOURS = {bio: [255, 140, 60], civ: [255, 208, 96], geo: [214, 164, 255], shift: [150, 205, 160]};

// A people's player's own people: numbers and food, where its
// hands go as one stacked bar, and its stores.
export const LABOR_COLOURS = {farm: [122, 170, 72], woodcut: [92, 128, 64], charcoal: [70, 64, 60],
                       mine: [150, 120, 96], bloomery: [214, 112, 52], smithy: [120, 132, 150],
                       study: [96, 134, 196]};

// A trade's colour: the iron age's own, any later trade one of a palette
// picked by its name (so it keeps its colour from turn to turn).
export const TRADE_PALETTE = [[78, 160, 140], [196, 150, 60], [150, 96, 170], [210, 98, 120],
                       [96, 170, 210], [170, 180, 70], [120, 110, 200], [220, 140, 90]];

/**
 * Converts a temperature in °C to an RGB triplet.
 * @param {number} t
 * @return {!Array<number>}
 */
export function tempRgb(t) {
  return ramp(MAP_COLORS.light.temps, (t - TEMP_SCALE[0]) / (TEMP_SCALE[1] - TEMP_SCALE[0]));
}
