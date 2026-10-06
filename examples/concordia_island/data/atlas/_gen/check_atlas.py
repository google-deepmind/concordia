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

"""Sanity-checks atlas geometry without needing a browser.

Catches the failure modes that make a map look wrong but still validate:
  * a place sitting in open water or out at sea
  * a place stranded far from any road, canal or ferry route
  * districts whose label anchor is nowhere near their own places
  * axis-aligned geometry, which is the signature of a placeholder shape

Usage:
  python3 check_atlas.py [atlas_id ...]
"""

import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # data/atlas/_gen
_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
UI = os.path.join(_PROJECT, 'ui')
sys.path.insert(0, UI)
import atlas_loader as al  # pylint: disable=g-import-not-at-top

NUM = re.compile(r'-?\d+(?:\.\d+)?')


def path_points(d: str) -> list[tuple[float, float]]:
  """Every coordinate pair in an SVG path, ignoring command semantics.

  Good enough for point-in-polygon on our own generated paths because every
  command we emit is absolute and the bezier hull tracks the outline closely.

  Args:
    d: SVG path string.

  Returns:
    List of (x, y) coordinate pairs.
  """
  vals = [float(v) for v in NUM.findall(d)]
  return list(zip(vals[0::2], vals[1::2]))


def inside(pt, poly):
  x, y = pt
  hit = False
  n = len(poly)
  for i in range(n):
    x0, y0 = poly[i]
    x1, y1 = poly[(i + 1) % n]
    if (y0 > y) != (y1 > y):
      xx = (x1 - x0) * (y - y0) / (y1 - y0) + x0
      if x < xx:
        hit = not hit
  return hit


def dist_to_segment(p, a, b):
  px, py = p
  ax, ay = a
  bx, by = b
  dx, dy = bx - ax, by - ay
  if dx == 0 and dy == 0:
    return math.hypot(px - ax, py - ay)
  denom = dx * dx + dy * dy
  t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
  return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def axis_aligned_fraction(d: str) -> float:
  """Fraction of consecutive point pairs that share an x or a y exactly.

  Args:
    d: SVG path string.

  Returns:
    Float fraction in [0.0, 1.0].
  """
  pts = path_points(d)
  if len(pts) < 3:
    return 0.0
  aligned = sum(
      1 for i in range(len(pts) - 1)
      if pts[i][0] == pts[i + 1][0] or pts[i][1] == pts[i + 1][1]
  )
  return aligned / (len(pts) - 1)


def check(atlas_id: str) -> tuple[list[str], list[str]]:
  """Validates an atlas for road connectivity, water overlap, and geometry."""
  a = al.load_atlas(atlas_id)
  problems = []
  notes = []

  water = [f for f in a.terrain if f['class'] in ('ocean', 'water')]
  # Terrain is painted back to front, so a later land feature can legitimately
  # sit on top of water. Only flag places over water that nothing covers.
  land_after = {}
  for i, f in enumerate(a.terrain):
    land_after[f['id']] = [
        g for g in a.terrain[i + 1:]
        if g['class'] not in ('ocean', 'water', 'wetland')
    ]

  water_polys = [(f['id'], path_points(f['path'])) for f in water]
  cover_polys = {
      f['id']: [(g['id'], path_points(g['path'])) for g in land_after[f['id']]]
      for f in water
  }

  segments = []
  for r in a.roads:
    g = r['geometry']
    for i in range(len(g) - 1):
      segments.append((tuple(g[i]), tuple(g[i + 1])))

  for p in a.places:
    xy = tuple(p['xy'])
    # A jetty, a pier or a fish landing is *supposed* to be over water --
    # that is the whole point of the category. Exempt it, but report it as a
    # note rather than staying silent, so an accidental `waterfront` never
    # hides a genuinely misplaced venue.
    waterfront = p.get('category') == 'waterfront'
    for wid, poly in water_polys:
      if inside(xy, poly):
        covered = any(inside(xy, cp) for _, cp in cover_polys[wid])
        if covered:
          pass
        elif waterfront:
          notes.append(
              f'place {p["id"]} sits on water feature {wid!r} '
              '(allowed: category is waterfront)'
          )
        else:
          problems.append(
              f'place {p["id"]} at {list(xy)} sits in water feature {wid!r}'
          )
        break
    if segments:
      d = min(dist_to_segment(xy, s[0], s[1]) for s in segments)
      if d > 70:
        problems.append(
            f'place {p["id"]} at {list(xy)} is {d:.0f} units from the nearest '
            'route -- agents would appear to walk across country'
        )

  # District anchors should sit among their own members.
  members = {}
  for p in a.places:
    if p.get('district'):
      members.setdefault(p['district'], []).append(p['xy'])
  for d in a.districts:
    pts = members.get(d['id'])
    if not pts:
      notes.append(f'district {d["id"]} has no places assigned to it')
      continue
    cx = sum(q[0] for q in pts) / len(pts)
    cy = sum(q[1] for q in pts) / len(pts)
    ax, ay = d['label_anchor']
    off = math.hypot(cx - ax, cy - ay)
    if off > 90:
      problems.append(
          f'district {d["id"]} label anchor {d["label_anchor"]} is {off:.0f} '
          f'units from its own places at [{cx:.0f}, {cy:.0f}]'
      )

  x0, y0, x1, y1 = a.meta['bounds']
  full_area = (x1 - x0) * (y1 - y0)
  for f in a.terrain:
    pts = path_points(f['path'])
    if pts:
      bw = max(p[0] for p in pts) - min(p[0] for p in pts)
      bh = max(p[1] for p in pts) - min(p[1] for p in pts)
      # A rectangle covering essentially the whole map is the deliberate base
      # wash every atlas starts with, not a lazy placeholder.
      if bw * bh >= 0.95 * full_area:
        continue
    frac = axis_aligned_fraction(f['path'])
    if frac > 0.6:
      problems.append(
          f'terrain {f["id"]} is {frac * 100:.0f}% axis-aligned -- that reads '
          'as a placeholder rectangle, not geography'
      )

  return problems, notes


def main():
  ids = sys.argv[1:] or al.available_atlases()
  failed = False
  for atlas_id in ids:
    problems, notes = check(atlas_id)
    print(f'=== {atlas_id}')
    for n in notes:
      print(f'  note: {n}')
    for p in problems:
      print(f'  PROBLEM: {p}')
    if problems:
      failed = True
    else:
      print('  clean')
  sys.exit(1 if failed else 0)


if __name__ == '__main__':
  main()
