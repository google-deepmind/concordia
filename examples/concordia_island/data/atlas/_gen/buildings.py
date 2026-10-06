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

"""Generates building stock for an atlas.

A town drawn as roads plus a handful of labelled pins looks like a diagram.
What makes a zoomed-in map look like a *place* is the building stock: rows of
houses set back from residential streets, shopfronts pressed up against the
arterial through downtown, bigger slabs where the offices are. None of that is
simulation data -- no agent is ever placed in a generated building, and nothing
keys off them -- it is the cartographic equivalent of hatching.

The layout rule is the one real towns follow: buildings line the streets. We
walk each road, step along it, and drop a footprint on each side at a setback
from the centreline, rotated to face the road. Density and size come from the
road class and from how close we are to a town centre.

Everything here is deterministic. Seeds are CRC32 of stable strings (see
geom.sid), never Python's salted hash(), so regenerating an atlas twice
produces byte-identical output.
"""

import math

from geom import sid


# Per-road-class frontage rules.
#
#   spacing  distance between buildings along the street, map units
#   setback  distance from the road centreline to the building centre
#   size     (width along street, depth away from street)
#   cls      building palette class
FRONTAGE = {
    'arterial': {
        'spacing': 27.0, 'setback': 17.0, 'size': (17.0, 12.0),
        'cls': 'commercial',
    },
    'residential': {
        'spacing': 21.0, 'setback': 12.5, 'size': (12.0, 10.0),
        'cls': 'residential',
    },
}

# Downtown gets denser, deeper, commercial frontage. Within CORE_RADIUS of a
# core anchor the rules above are replaced by these.
CORE_FRONTAGE = {
    'spacing': 17.0, 'setback': 13.0, 'size': (14.0, 13.0), 'cls': 'commercial',
}


def _rng(seed):
  """Tiny deterministic LCG, matching geom.rng."""
  state = [seed & 0xFFFFFFFF]

  def nxt():
    state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
    return state[0] / 0x7FFFFFFF

  return nxt


def point_seg_dist(px, py, ax, ay, bx, by):
  """Distance from a point to a line segment."""
  dx, dy = bx - ax, by - ay
  denom = dx * dx + dy * dy
  if denom <= 1e-9:
    return math.hypot(px - ax, py - ay)
  t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
  return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def polyline_dist(px, py, pts):
  """Distance from a point to a polyline given as [(x, y), ...]."""
  best = float('inf')
  for i in range(len(pts) - 1):
    d = point_seg_dist(px, py, pts[i][0], pts[i][1],
                       pts[i + 1][0], pts[i + 1][1])
    if d < best:
      best = d
  return best


def rect_path(cx, cy, w, h, angle):
  """Closed SVG path for a rectangle centred at (cx, cy), rotated by angle."""
  cos, sin = math.cos(angle), math.sin(angle)
  corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
  pts = [
      (cx + dx * cos - dy * sin, cy + dx * sin + dy * cos)
      for dx, dy in corners
  ]
  body = ' '.join(f'L {x:.1f},{y:.1f}' for x, y in pts[1:])
  return f'M {pts[0][0]:.1f},{pts[0][1]:.1f} {body} Z'


def l_path(cx, cy, w, h, angle):
  """Closed SVG path for an L-shaped plan, for variety in the stock."""
  cos, sin = math.cos(angle), math.sin(angle)
  hw, hh = w / 2, h / 2
  corners = [
      (-hw, -hh), (hw, -hh), (hw, hh * 0.1), (hw * 0.1, hh * 0.1),
      (hw * 0.1, hh), (-hw, hh),
  ]
  pts = [
      (cx + dx * cos - dy * sin, cy + dx * sin + dy * cos)
      for dx, dy in corners
  ]
  body = ' '.join(f'L {x:.1f},{y:.1f}' for x, y in pts[1:])
  return f'M {pts[0][0]:.1f},{pts[0][1]:.1f} {body} Z'


def street_buildings(
    roads,
    district_centres,
    core_anchors=(),
    core_radius=110.0,
    town_radius=150.0,
    avoid_points=(),
    avoid_radius=16.0,
    avoid_polylines=(),
    frontage=None,
):
  """Lays out building footprints along street frontages.

  Args:
    roads: atlas road dicts, each with 'id', 'class' and 'geometry'.
    district_centres: [(x, y), ...] neighbourhood centres. Buildings only
      appear within `town_radius` of one, so highways crossing open country do
      not sprout a ribbon of houses.
    core_anchors: [(x, y), ...] downtown centres, which get denser commercial
      frontage.
    core_radius: how far the dense core rules extend.
    town_radius: how far from a district centre buildings are placed at all.
    avoid_points: [(x, y), ...] named place coordinates. Generated stock is
      kept clear of these so it does not collide with venue footprints.
    avoid_radius: clearance around each avoid point.
    avoid_polylines: [([(x, y), ...], clearance), ...] water courses, canals
      and the like that buildings must not be drawn on top of.
    frontage: optional override for the FRONTAGE table.

  Returns:
    A list of {'class': ..., 'path': ...} dicts, deduplicated by position.
  """
  rules = frontage or FRONTAGE
  out = []
  taken = []  # (x, y, clearance) of everything already placed

  def blocked(x, y, clearance):
    for ax, ay in avoid_points:
      if math.hypot(ax - x, ay - y) < avoid_radius:
        return True
    for pts, gap in avoid_polylines:
      if polyline_dist(x, y, pts) < gap:
        return True
    for tx, ty, tc in taken:
      if math.hypot(tx - x, ty - y) < max(clearance, tc):
        return True
    return False

  def near(x, y, anchors):
    if not anchors:
      return float('inf')
    return min(math.hypot(ax - x, ay - y) for ax, ay in anchors)

  for road in roads:
    rule = rules.get(road['class'])
    if not rule:
      continue
    geom = road['geometry']
    rnd = _rng(sid(road['id']) % 9973)

    # Walk the whole route as one continuous run so spacing does not reset at
    # every control point, which would bunch buildings up at the vertices.
    carry = rnd() * rule['spacing']
    for i in range(len(geom) - 1):
      ax, ay = geom[i]
      bx, by = geom[i + 1]
      seg = math.hypot(bx - ax, by - ay)
      if seg < 1e-6:
        continue
      ux, uy = (bx - ax) / seg, (by - ay) / seg
      nx, ny = -uy, ux           # left normal
      angle = math.atan2(uy, ux)

      t = carry
      while t < seg:
        mx, my = ax + ux * t, ay + uy * t
        in_core = near(mx, my, core_anchors) <= core_radius
        r = CORE_FRONTAGE if in_core else rule
        if near(mx, my, district_centres) <= town_radius:
          for side in (1, -1):
            # Deterministic wobble so the row is not a picket fence.
            jitter_along = (rnd() - 0.5) * 4.0
            jitter_out = (rnd() - 0.5) * 5.0
            offset = r['setback'] + jitter_out
            sx = mx + nx * side * offset + ux * jitter_along
            sy = my + ny * side * offset + uy * jitter_along
            w = r['size'][0] * (0.82 + rnd() * 0.36)
            h = r['size'][1] * (0.82 + rnd() * 0.36)
            clearance = max(w, h) * 0.62
            if blocked(sx, sy, clearance):
              continue
            skew = (rnd() - 0.5) * 0.16
            shape = l_path if rnd() < 0.22 else rect_path
            out.append({
                'class': r['cls'],
                'path': shape(sx, sy, w, h, angle + skew),
            })
            taken.append((sx, sy, clearance))
        t += r['spacing'] * (0.85 + rnd() * 0.3)
      carry = t - seg

  return out


def block_buildings(anchors, avoid_points=(), avoid_radius=14.0):
  """Large standalone slabs -- office parks, plazas, institutional campuses.

  Args:
    anchors: [(x, y, w, h, angle_deg, cls), ...] explicit slab placements.
    avoid_points: named place coordinates to keep clear of.
    avoid_radius: clearance around each avoid point.

  Returns:
    A list of {'class': ..., 'path': ...} dicts.
  """
  out = []
  for x, y, w, h, deg, cls in anchors:
    if any(math.hypot(ax - x, ay - y) < avoid_radius
           for ax, ay in avoid_points):
      continue
    out.append({'class': cls, 'path': rect_path(x, y, w, h,
                                                math.radians(deg))})
  return out
