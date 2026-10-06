# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared geometry helpers for atlas generators.

Atlas terrain is hand-traced from real geography at a small number of control
points. These helpers turn those sparse points into smooth closed paths so the
result reads as surveyed coastline rather than as a polygon chain. Everything
here is deterministic: the same inputs always produce byte-identical output.
"""

import math
import zlib


def sid(text):
  """Stable string hash.

  The builtin hash() is salted per process, so using it as a shape seed would
  silently move the map on every run.

  Args:
    text: String to compute stable 32-bit hash for.

  Returns:
    Non-negative 31-bit integer hash.
  """
  return zlib.crc32(text.encode()) & 0x7FFFFFFF


def rng(seed):
  """Tiny deterministic LCG."""
  state = [seed & 0xFFFFFFFF]

  def nxt():
    state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
    return state[0] / 0x7FFFFFFF

  return nxt


def closed_spline(pts):
  """Closed uniform Catmull-Rom through pts, emitted as cubic beziers."""
  n = len(pts)
  d = [f'M {pts[0][0]:.1f},{pts[0][1]:.1f}']
  for i in range(n):
    p0, p1, p2, p3 = (
        pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
    )
    c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
    c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
    d.append(
        f'C {c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} '
        f'{p2[0]:.1f},{p2[1]:.1f}'
    )
  d.append('Z')
  return ' '.join(d)


def open_spline(pts, close=True):
  """Catmull-Rom through pts in order; optionally closes the figure."""
  n = len(pts)
  d = [f'M {pts[0][0]:.1f},{pts[0][1]:.1f}']
  for i in range(n - 1):
    p0 = pts[i - 1] if i > 0 else pts[0]
    p1, p2 = pts[i], pts[i + 1]
    p3 = pts[i + 2] if i + 2 < n else pts[n - 1]
    c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
    c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
    d.append(
        f'C {c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} '
        f'{p2[0]:.1f},{p2[1]:.1f}'
    )
  if close:
    d.append('Z')
  return ' '.join(d)


def blob(cx, cy, rx, ry, seed, n=9, jitter=0.26, squash=0.0):
  """A closed organic shape centred on (cx, cy). Never axis-aligned."""
  rnd = rng(seed)
  pts = []
  phase = rnd() * math.tau
  for i in range(n):
    a = phase + (i / n) * math.tau
    r = 1.0 + (rnd() - 0.5) * 2 * jitter
    pts.append((
        cx + math.cos(a) * rx * r,
        cy + math.sin(a) * ry * r + math.cos(a) * squash * ry,
    ))
  return closed_spline(pts)


def parcel(cx, cy, w, h, seed, rot=0.0, jitter=2.0):
  """A straight-edged quadrilateral field, centred on (cx, cy).

  The counterpart to blob(). Natural land cover is organic, but *reclaimed*
  land is surveyed: paddy polders, orchards and allotments are bounded by
  built bunds and dykes that run straight. Drawing them with blob() makes
  farmland read as a random green smudge rather than as worked fields, so
  this emits hard corners and straight edges instead of a spline.

  Args:
    cx: centre x.
    cy: centre y.
    w: width before rotation.
    h: height before rotation.
    seed: deterministic shape seed.
    rot: rotation in radians; a small value keeps a grid of parcels from
      looking like a UI table.
    jitter: max corner displacement in map units. Kept small on purpose --
      the edges are supposed to be surveyed, just not machine-perfect.

  Returns:
    An SVG path string.
  """
  rnd = rng(seed)
  hw, hh = w / 2, h / 2
  ca, sa = math.cos(rot), math.sin(rot)
  pts = []
  for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
    lx += (rnd() - 0.5) * 2 * jitter
    ly += (rnd() - 0.5) * 2 * jitter
    pts.append((cx + lx * ca - ly * sa, cy + lx * sa + ly * ca))
  head = f'M {pts[0][0]:.1f},{pts[0][1]:.1f}'
  rest = ' '.join(f'L {x:.1f},{y:.1f}' for x, y in pts[1:])
  return f'{head} {rest} Z'


def ribbon(spine, half_width):
  """A closed path for a river/canal of varying width.

  Args:
    spine: list of (x, y, w); w scales the local half width.
    half_width: base half width in map units.

  Returns:
    An SVG path string tracing the ribbon.
  """
  left, right = [], []
  n = len(spine)
  for i, (x, y, w) in enumerate(spine):
    nxt = spine[min(i + 1, n - 1)]
    prv = spine[max(i - 1, 0)]
    dx, dy = nxt[0] - prv[0], nxt[1] - prv[1]
    mag = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / mag, dx / mag
    hw = half_width * w
    left.append((x + nx * hw, y + ny * hw))
    right.append((x - nx * hw, y - ny * hw))
  return open_spline(left + list(reversed(right)) + [left[0]])
