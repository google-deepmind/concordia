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

// The title screen's globe: the picked world's map at its
// start (`/api/thumb`, one biome a tile, north first) wrapped on a sphere
// and seen from space, turning slowly.

/**
 * The globe as pixels: a `size` x `size` RGBA image of the sphere turned by
 * `turn` (0..1 of a rotation), tilted toward the viewer by `tilt` radians,
 * lit from the upper left; outside the disc transparent.
 * @param {!Object} thumb
 * @param {number} size
 * @param {number=} turn
 * @param {number=} tilt
 * @param {boolean=} dark
 * @return {!Object}
 */
export function globePixels(thumb, size, turn = 0, tilt = 0.35, dark = false) {
  const rows = thumb.rows || [];
  const h = rows.length, w = h ? rows[0].length : 0;
  const colour = {};
  for (const b of thumb.legend || []) colour[b.code] = dark ? b.dark : b.light;
  const data = new Uint8ClampedArray(size * size * 4);
  if (!w) return {size, data};
  const r = size / 2, ct = Math.cos(tilt), st = Math.sin(tilt);
  for (let py = 0; py < size; py += 1) {
    for (let px = 0; px < size; px += 1) {
      const x = (px + 0.5 - r) / r, y = (r - py - 0.5) / r, d = x * x + y * y;
      if (d > 1) continue;
      const z = Math.sqrt(1 - d);
      // Tip the view toward the north pole, then read latitude, longitude.
      const y2 = y * ct + z * st, z2 = -y * st + z * ct;
      const lat = Math.asin(Math.max(-1, Math.min(1, y2)));
      const lon = Math.atan2(x, z2) / (2 * Math.PI) + turn;
      const col = ((Math.floor((lon - Math.floor(lon)) * w) % w) + w) % w;
      const row = Math.min(h - 1, Math.max(0, Math.floor((0.5 - lat / Math.PI) * h)));
      const c = colour[rows[row][col]] || [128, 128, 128];
      const light = 0.55 + 0.45 * Math.max(0, -0.45 * x + 0.45 * y + 0.77 * z);
      const i = 4 * (py * size + px);
      data[i] = c[0] * light; data[i + 1] = c[1] * light; data[i + 2] = c[2] * light;
      data[i + 3] = d > 0.97 ? 255 * (1 - d) / 0.03 : 255;  // A soft limb.
    }
  }
  return {size, data};
}
