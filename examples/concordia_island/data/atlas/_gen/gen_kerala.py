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

"""Regenerates the Alappuzha (Kerala) atlas.

The first draft laid the coast out as a vertical rectangle at x=150 and the
lake as another rectangle, which is exactly what a backwater town does not
look like. This rewrites the geometry from control points traced off the real
place: an NNW-SSE coastline, the beach and pier, the canal grid that earns
Alappuzha the "Venice of the East" tag, the Kuttanad polders inland, and
Vembanad / Punnamada lake to the east.

Place identity (ids, display names, categories, parents, interiors) is read
back out of the existing file and never invented here; only coordinates and
district membership are authored.

Run:
  python3 gen_kerala.py
"""

import functools
import math
import os

from buildings import block_buildings
from buildings import street_buildings
from geom import blob
from geom import open_spline
from geom import parcel
from geom import ribbon
from geom import sid
import yaml

ATLAS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kerala.yaml'
)

# --------------------------------------------------------------------------
# Alappuzha, stylised. x: 0 = Arabian Sea, 1000 = Vembanad lake. y: 0 = north.
#
# The coast trends NNW-SSE, so the shoreline slides east as you go south. The
# town is a narrow strip pinned between the beach and the backwaters, cut
# through by east-west canals that run from the old pier to the lake.
# --------------------------------------------------------------------------

COAST = [
    (118, -20), (126, 60), (140, 150), (152, 240), (168, 330),
    (182, 420), (198, 510), (210, 600), (224, 720),
]

# Sand strip runs just inland of the surf line.
BEACH_INNER = [(x + 46, y) for x, y in COAST]

LAKE_SHORE = [
    (700, -20), (676, 70), (700, 160), (664, 250), (690, 340),
    (656, 430), (684, 520), (650, 610), (672, 720),
]

# Canal routing is constrained by the venues, not the other way round: place
# coordinates are simulation identities that must stay put, so where a channel
# would otherwise run straight through the church, the bus stand, the office
# block or the apartments, the channel moves. `check_atlas.py` enforces this --
# it fails loudly on any non-waterfront place sitting inside a water polygon.
COMMERCIAL_CANAL = [
    (206, 280, 0.8), (280, 274, 0.9), (360, 286, 1.0), (440, 276, 1.0),
    (520, 288, 1.1), (600, 278, 1.2), (670, 290, 1.3),
]
VADAI_CANAL = [
    (222, 428, 0.7), (300, 420, 0.8), (382, 432, 0.9), (462, 422, 1.0),
    (544, 434, 1.1), (620, 424, 1.2), (676, 436, 1.2),
]
# A channel that stops in the middle of a dry field reads as an unfinished
# drawing, so every north-south cut runs between two east-west ones and every
# east-west cut runs from the pier to open water.
LINK_CANAL = [
    (298, 214, 0.6), (306, 300, 0.8), (314, 380, 0.9), (322, 430, 0.9),
    (332, 516, 0.8), (338, 568, 0.7),
]
# Alappuzha's grid is denser than three channels. These are the cross-cuts
# that make the "Venice of the East" description do any work: a north canal
# behind the temple ward, an eastern link running down to the paddy, and the
# thodu that drains Kuttanad into the lake.
NORTH_CANAL = [
    (238, 214, 0.6), (312, 206, 0.7), (392, 218, 0.8), (470, 208, 0.8),
    (548, 220, 0.9), (612, 212, 1.0), (668, 222, 1.0),
]
EAST_LINK_CANAL = [
    (512, 226, 0.6), (520, 302, 0.8), (528, 380, 0.9), (536, 452, 0.8),
    (544, 524, 0.7), (550, 568, 0.7),
]
KUTTANAD_THODU = [
    (330, 560, 0.7), (418, 574, 0.8), (506, 560, 0.9), (592, 578, 1.0),
    (664, 566, 1.1), (700, 540, 1.1),
]

# Half-widths, in map units. These were roughly double to begin with, which
# made each channel about 2% of the map's width -- wider than the arterial, so
# at region zoom the grid painted as six cyan highways rather than as dug
# canals. Alappuzha's channels are narrow enough that buildings crowd both
# banks, and the hierarchy (Commercial widest, the thodu and links narrower)
# is what carries the reading, not the absolute size.
CANALS_AUTHORED = [
    ('commercial_canal', 'Commercial Canal', COMMERCIAL_CANAL, 6.0),
    ('vadai_canal', 'Vadai Canal', VADAI_CANAL, 5.5),
    ('link_canal', 'Link Canal', LINK_CANAL, 4.5),
    ('north_canal', 'North Canal', NORTH_CANAL, 4.0),
    ('east_link_canal', 'East Link Canal', EAST_LINK_CANAL, 4.0),
    ('kuttanad_thodu', 'Kuttanad Thodu', KUTTANAD_THODU, 5.0),
]

# Clear water between a channel's edge and a venue's centre.
CANAL_CLEARANCE = 4.0


def _relax_spine(spine, half, obstacles, clearance=CANAL_CLEARANCE,
                 iterations=120):
  """Nudges a canal spine sideways until it clears every obstacle.

  Hand-placing canals around fixed venues does not converge: every manual
  shift that frees one venue drives the channel into another. So the authored
  spine is treated as an intent, and this pushes each control point along the
  local normal until no obstacle lies inside the channel, then re-smooths so
  the result still reads as a dug canal rather than a zigzag.

  Deterministic: no randomness, and the same input always yields the same
  output.

  Args:
    spine: List of (x, y, w) control points.
    half: Channel half-width in map units.
    obstacles: Iterable of (x, y) that must end up outside the channel.
    clearance: Extra margin beyond the channel edge.
    iterations: Relaxation steps.

  Returns:
    A new list of (x, y, w) control points.
  """
  need = half + clearance

  # Densify first. With only 5-7 authored control points a venue can sit
  # squarely between two of them: every control point clears it while the
  # segment joining them runs straight over it. Subdividing gives the
  # relaxation somewhere to actually bend.
  #
  # The subdivision count has to be driven by `need`, not fixed. Obstacles are
  # only ever tested at control points, so the spacing must stay well under the
  # clearance radius or a venue slips through the gap between two points that
  # both clear it -- which is exactly what happened to the mosque when the
  # channels were narrowed and `need` shrank with them. Subdividing until the
  # spacing is at most half the radius guarantees any point inside the channel
  # is within `need` of some control point.
  pts = [[float(x), float(y)] for x, y, _ in spine]
  widths = [p[2] for p in spine]
  while len(pts) < 512:
    longest = max(
        math.dist(a, b) for a, b in zip(pts, pts[1:])
    )
    if longest <= need / 2:
      break
    dense = [pts[0]]
    for a, b in zip(pts, pts[1:]):
      dense.append([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])
      dense.append(b)
    pts = dense
  # Interpolate the authored widths across the denser point list.
  n = len(pts)
  widths = [widths[min(len(widths) - 1, round(i * (len(widths) - 1) / (n - 1)))]
            for i in range(n)]
  for _ in range(iterations):
    moved = False
    for i, (px, py) in enumerate(pts):
      # Local direction, used to build the normal we are allowed to slide on.
      a = pts[max(0, i - 1)]
      b = pts[min(len(pts) - 1, i + 1)]
      dx, dy = b[0] - a[0], b[1] - a[1]
      length = (dx * dx + dy * dy) ** 0.5 or 1.0
      nx, ny = -dy / length, dx / length
      for ox, oy in obstacles:
        vx, vy = px - ox, py - oy
        dist = (vx * vx + vy * vy) ** 0.5
        if dist >= need:
          continue
        # Push along the normal, in whichever direction takes us away.
        sign = 1.0 if (vx * nx + vy * ny) >= 0 else -1.0
        push = (need - dist) * 0.6
        pts[i][0] += nx * sign * push
        pts[i][1] += ny * sign * push
        px, py = pts[i]
        moved = True
    # Gentle smoothing keeps the channel readable as a dug canal. Kept weak on
    # purpose: at 0.25 either side it simply undid each push and the relaxation
    # never converged. Endpoints stay pinned so the canal still meets the coast
    # and the lake where it was authored to.
    for i in range(1, len(pts) - 1):
      for k in (0, 1):
        pts[i][k] = 0.8 * pts[i][k] + 0.1 * (pts[i - 1][k] + pts[i + 1][k])
    if not moved:
      break

  return [(round(p[0], 1), round(p[1], 1), w) for p, w in zip(pts, widths)]


@functools.lru_cache(maxsize=1)
def canals():
  """The canal table with every channel relaxed clear of the fixed venues.

  Decorative waterfront places (a houseboat jetty, a fish landing) are
  deliberately NOT obstacles -- those belong on the water.

  Returns:
    A list of (id, name, spine, half_width) tuples.
  """
  decor_ids = {d[0] for d in DECOR_PLACES}
  obstacles = tuple((entry[0][0], entry[0][1])
                    for pid, entry in PLACE_XY.items()
                    if pid not in decor_ids)
  return [(cid, name, _relax_spine(spine, half, obstacles), half)
          for cid, name, spine, half in CANALS_AUTHORED]

DISTRICTS = [
    ('beach_ward', 'Beach Ward', (272, 250), '#38bdf8'),
    ('canal_town', 'Canal Town', (404, 348), '#ef4444'),
    ('backwater_edge', 'Backwater Edge', (612, 320), '#10b981'),
    ('punnamada', 'Punnamada', (792, 168), '#8b5cf6'),
    ('kuttanad', 'Kuttanad', (486, 566), '#f59e0b'),
]
DCENTRE = {d[0]: d[2] for d in DISTRICTS}


def water_west(shore, seed_edge=-40):
  """Closed path for everything seaward (west) of `shore`."""
  pts = [(seed_edge, -40)] + list(shore) + [(seed_edge, 740)]
  return open_spline(pts)


def water_east(shore, seed_edge=1040):
  """Closed path for everything east of `shore`."""
  pts = [(seed_edge, -40)] + list(reversed(shore)) + [(seed_edge, 740)]
  return open_spline(pts)


def strip(outer, inner):
  """Closed path between two roughly parallel shore lines."""
  return open_spline(list(outer) + list(reversed(inner)))


def terrain():
  """Builds the Alappuzha land cover, back to front.

  Order matters and is not the order you would list these features in prose.
  Built form goes down *before* the canals and the parks, because the
  renderer injects its built-up wash immediately ahead of the first built
  polygon -- so anything that must read over the town (every channel in a
  town whose whole identity is its channels) has to be authored after it.
  The previous ordering had the canal grid buried under the town core.

  Returns:
    An ordered list of terrain feature dicts.
  """
  t = []

  def add(cls, tid, path, **extra):
    feat = {'class': cls, 'id': tid, 'path': path}
    feat.update(extra)
    t.append(feat)

  # 1. Land base.
  add('grass', 'mainland', 'M 0,0 L 1000,0 L 1000,700 L 0,700 Z')

  # 2. Kuttanad: reclaimed paddy, much of it below sea level.
  #
  #    Drawn basin-first, then fields on top, because that is the order the
  #    landscape was actually made in: Kuttanad is a waterlogged backwater
  #    basin out of which individual polders were *drained* and ring-bunded.
  #    Painting it that way round also does the cartography for free -- the
  #    gaps the fields leave behind are the wet bund channels, so the
  #    patchwork is produced by the geometry rather than by hoping two
  #    similar greens read as different land cover.
  add(
      'wetland',
      'kuttanad_belt',
      open_spline([
          (250, 470), (360, 486), (470, 470), (580, 492), (660, 478),
          (690, 700), (250, 700),
      ]),
  )

  # The polder grid. Bunds are built, so they run straight and parallel;
  # the whole grid is rotated a few degrees off-axis so it reads as surveyed
  # land rather than as a UI table.
  bund_rot = -0.055
  ca, sa = math.cos(bund_rot), math.sin(bund_rot)
  polder_idx = 0
  for cy in (-66.0, 0.0, 66.0):
    for cx in (-165.0, -55.0, 55.0, 165.0):
      add(
          'farmland',
          f'polder_{polder_idx}',
          parcel(470 + cx * ca - cy * sa, 592 + cx * sa + cy * ca,
                 98, 58, 700 + polder_idx, rot=bund_rot, jitter=2.4),
      )
      polder_idx += 1

  # 3. Coconut groves.
  for i, (cx, cy, rx, ry) in enumerate([
      (300, 150, 88, 62), (470, 180, 74, 52), (588, 128, 66, 46),
      (250, 330, 54, 40), (600, 250, 58, 42), (300, 430, 50, 36),
      (640, 560, 62, 44),
  ]):
    add('forest', f'palm_grove_{i}', blob(cx, cy, rx, ry, 800 + i, n=9))

  # 4. Arabian Sea and the beach.
  add('ocean', 'arabian_sea', water_west(COAST))
  add('sand', 'alappuzha_beach', strip(COAST, BEACH_INNER), halo=True)

  # 5. Vembanad / Punnamada lake, with the usual scatter of low islands and
  #    the reed fringe where the backwater meets the land.
  add('water', 'vembanad_lake', water_east(LAKE_SHORE))
  add('wetland', 'punnamada_fringe',
      strip(LAKE_SHORE, [(x - 54, y) for x, y in LAKE_SHORE]))
  for i, (cx, cy, rx, ry) in enumerate([
      (782, 112, 42, 28), (826, 300, 34, 24), (770, 470, 38, 26),
      (812, 612, 30, 22),
  ]):
    add('grass', f'lake_island_{i}', blob(cx, cy, rx, ry, 900 + i, n=7))

  # 6. Built form. The town is a narrow strip pinned between the beach and the
  #    backwaters; the NH-66 frontage is the spine everything hangs off.
  add('urban_core', 'town_core', blob(404, 344, 92, 104, 41, n=11,
                                      jitter=0.16))
  add('commercial', 'nh66_frontage',
      strip([(406, 90), (398, 200), (410, 320), (402, 440), (414, 560)],
            [(452, 560), (444, 440), (456, 320), (448, 200), (452, 90)]))
  add('commercial', 'market_quarter', blob(432, 348, 44, 32, 61, n=8,
                                           jitter=0.16))
  add('commercial', 'pier_road_shops',
      strip([(200, 292), (260, 290), (320, 292), (370, 300)],
            [(372, 320), (322, 312), (262, 310), (202, 312)]))
  add('residential', 'beach_colony', blob(236, 246, 66, 96, 51, n=9))
  add('residential', 'north_colony', blob(320, 232, 56, 44, 52, n=8))
  add('residential', 'south_apartments', blob(334, 498, 58, 46, 53, n=8))
  add('residential', 'lakeside_estates', blob(730, 396, 62, 52, 54, n=8))
  add('residential', 'villa_quarter', blob(792, 176, 64, 50, 55, n=8))
  add('residential', 'temple_ward', blob(356, 378, 46, 40, 56, n=8))
  add('civic', 'office_campus', blob(636, 316, 48, 38, 57, n=7))
  add('industrial', 'boat_yard_ground', blob(704, 300, 34, 24, 58, n=7))

  # 7. The canal grid, over the town. This is the thing that makes Alappuzha
  #    Alappuzha, so nothing is allowed to be painted on top of it.
  for cid, _, spine, half in canals():
    add('water', cid, ribbon(spine, half))

  # 8. Open space.
  add('park', 'municipal_park', blob(492, 258, 46, 34, 59, n=8))
  add('park', 'boat_race_lawn', blob(742, 214, 40, 28, 60, n=8))
  return t


def roads():
  """The full road table: streets, canal routes, and the derived bridges."""
  streets = [
      {
          'id': 'nh_66',
          'name': 'NH-66',
          'class': 'arterial',
          'shield': {'type': 'us_state', 'state': 'NH', 'number': '66'},
          'geometry': [
              [396, 0], [404, 90], [396, 190], [408, 290], [400, 390],
              [412, 480], [404, 580], [418, 700],
          ],
      },
      # Beach Rd and Backwater Rd are arterials, not residential streets.
      # Residential is held back to street zoom, so with NH-66 as the only
      # arterial the region view had no road skeleton whatsoever -- canals
      # floating on a tan wash. These two give the coastal spine and the
      # lake connector that the town is actually organised around.
      {
          'id': 'beach_road',
          'name': 'Beach Rd',
          'class': 'arterial',
          'geometry': [
              [176, 60], [190, 150], [202, 240], [218, 330], [232, 420],
              [248, 510], [262, 600],
          ],
      },
      {
          'id': 'pier_road',
          'name': 'Pier Rd',
          'class': 'residential',
          'geometry': [[168, 300], [232, 304], [300, 300], [360, 308],
                       [404, 330]],
      },
      {
          'id': 'market_street',
          'name': 'Market St',
          'class': 'residential',
          'geometry': [[340, 350], [380, 344], [424, 352], [470, 344],
                       [520, 352]],
      },
      {
          'id': 'temple_road',
          'name': 'Temple Rd',
          'class': 'residential',
          'geometry': [[342, 400], [372, 378], [404, 362], [440, 372],
                       [472, 396]],
      },
      {
          'id': 'mosque_lane',
          'name': 'Mosque Ln',
          'class': 'residential',
          'geometry': [[330, 272], [368, 288], [404, 280], [444, 294]],
      },
      {
          'id': 'school_road',
          'name': 'School Rd',
          'class': 'residential',
          'geometry': [[318, 438], [356, 416], [396, 408], [434, 420],
                       [468, 444]],
      },
      {
          'id': 'backwater_road',
          'name': 'Backwater Rd',
          'class': 'arterial',
          'geometry': [[448, 320], [516, 308], [580, 318], [636, 312],
                       [690, 324], [736, 372]],
      },
      {
          'id': 'punnamada_road',
          'name': 'Punnamada Rd',
          'class': 'residential',
          'geometry': [[470, 206], [548, 192], [628, 184], [704, 172],
                       [780, 168]],
      },
      {
          'id': 'vembanad_ferry',
          'name': 'Vembanad Ferry',
          'class': 'ferry',
          'geometry': [[690, 330], [752, 292], [810, 240], [852, 186]],
      },
      {
          'id': 'punnamada_ferry',
          'name': 'Punnamada Ferry',
          'class': 'ferry',
          'geometry': [[712, 236], [768, 206], [820, 178], [864, 140]],
      },
      {
          'id': 'kuttanad_bund',
          'name': 'Kuttanad Bund',
          'class': 'trail',
          'geometry': [[286, 500], [368, 516], [456, 504], [546, 524],
                       [630, 512]],
      },
      {
          'id': 'polder_bund_walk',
          'name': 'Polder Bund Walk',
          'class': 'trail',
          'geometry': [[352, 596], [438, 608], [524, 594], [608, 612],
                       [688, 598]],
      },
  ]
  return streets + canal_roads() + bridges(streets)


def canal_roads():
  """The navigable canals, as routes rather than as water polygons.

  The terrain layer paints the channel; this is the *route* down it, which is
  what gets a name on the map and what a boat actually follows.

  Returns:
    A list of road dicts of class `canal`.
  """
  return [
      {
          'id': cid,
          'name': name,
          'class': 'canal',
          'geometry': [[p[0], p[1]] for p in spine],
      }
      for cid, name, spine, _ in canals()
  ]


# Bridges are not authored at all: they are wherever a road actually crosses a
# canal.
#
# This has been through two worse designs. First, hand-authored coordinate
# pairs, where every span was a fixed ~44 units long regardless of the channel
# it crossed, so decks overshot far onto dry land. Then authored `near_xy`
# hints naming a canal, which fixed the length but kept the deeper mistake:
# nothing checked that a road was *there*. Several decks ended up as grey slabs
# floating in open water with no street leading onto them, because the hint had
# been placed near a canal but not near a road.
#
# Deriving them from the true intersections makes the invariant hold by
# construction -- there is a bridge if and only if a road crosses a channel --
# and it means new roads and re-routed canals both get correct bridges for
# free.

# How far the deck runs onto each bank past the water's edge. Enough to read as
# landing on solid ground, not so much that it becomes a road in its own right.
BRIDGE_APPROACH = 6.0

# Road classes that carry a deck. Ferries are routes over open water and canals
# are the water, so neither bridges anything.
BRIDGEABLE = frozenset({'arterial', 'residential', 'trail'})


def _segment_intersection(p1, p2, p3, p4):
  """Returns the (x, y) where segment p1-p2 meets p3-p4, or None.

  Args:
    p1: Start of the first segment, as (x, y).
    p2: End of the first segment.
    p3: Start of the second segment.
    p4: End of the second segment.

  Returns:
    The intersection point, or None if the segments are parallel or do not
    overlap within their extents.
  """
  x1, y1 = p1
  x2, y2 = p2
  x3, y3 = p3
  x4, y4 = p4
  denom = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
  if abs(denom) < 1e-9:
    return None
  t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denom
  u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / denom
  if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
    return None
  return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def bridges(road_list):
  """Short spans carrying the street grid over the canals.

  Args:
    road_list: The non-bridge roads, as emitted by `roads()`. Only classes in
      BRIDGEABLE are considered.

  Returns:
    A list of road dicts with `bridge: True`, one per road/canal crossing.
  """
  out = []
  seen = set()
  for road in road_list:
    if road['class'] not in BRIDGEABLE:
      continue
    geom = [(float(x), float(y)) for x, y in road['geometry']]
    for cid, cname, spine, half in canals():
      pts = [(p[0], p[1]) for p in spine]
      for ra, rb in zip(geom, geom[1:]):
        hit = None
        for ca, cb in zip(pts, pts[1:]):
          hit = _segment_intersection(ra, rb, ca, cb)
          if hit:
            break
        if not hit:
          continue
        # One deck per road/canal pair. A wiggly road can clip the same
        # channel across two consecutive segments, which would otherwise
        # stack two decks on top of each other.
        key = (road['id'], cid)
        if key in seen:
          continue
        seen.add(key)

        # The deck runs along the road, not across the channel: that is the
        # direction traffic travels. An oblique crossing has to be longer to
        # span the same water, hence the 1/sin(angle) term, clamped so a very
        # shallow crossing does not generate an absurd span.
        rdx, rdy = rb[0] - ra[0], rb[1] - ra[1]
        rlen = math.hypot(rdx, rdy) or 1.0
        rdx, rdy = rdx / rlen, rdy / rlen
        ctan = _tangent_near(pts, hit)
        sin_theta = abs(rdx * ctan[1] - rdy * ctan[0])
        stretch = min(3.0, 1.0 / max(sin_theta, 1e-3))
        reach = (half * stretch) + BRIDGE_APPROACH
        out.append({
            'id': f'{road["id"]}_x_{cid}',
            'name': f'{road["name"]} Bridge' if 'Bridge' not in road['name']
                    else road['name'],
            'class': 'residential' if road['class'] == 'trail'
                     else road['class'],
            'bridge': True,
            'canal': cname,
            'geometry': [
                [
                    round(hit[0] - rdx * reach, 1),
                    round(hit[1] - rdy * reach, 1),
                ],
                [
                    round(hit[0] + rdx * reach, 1),
                    round(hit[1] + rdy * reach, 1),
                ],
            ],
        })
  return out


def _tangent_near(pts, at):
  """Unit tangent of the polyline `pts` on the segment nearest to `at`."""
  best = None
  for a, b in zip(pts, pts[1:]):
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg2 = dx * dx + dy * dy
    t = 0.0 if seg2 == 0 else (((at[0] - a[0]) * dx + (at[1] - a[1]) * dy)
                               / seg2)
    t = max(0.0, min(1.0, t))
    cx, cy = a[0] + t * dx, a[1] + t * dy
    d2 = (at[0] - cx) ** 2 + (at[1] - cy) ** 2
    if best is None or d2 < best[0]:
      length = (seg2 ** 0.5) or 1.0
      best = (d2, (dx / length, dy / length))
  return best[1]


def districts():
  """Generates district boundary geometry and metadata."""
  out = []
  shapes = {
      'beach_ward': (80, 150),
      'canal_town': (104, 124),
      'backwater_edge': (92, 108),
      'punnamada': (96, 86),
      'kuttanad': (150, 106),
  }
  for did, name, (cx, cy), color in DISTRICTS:
    rx, ry = shapes[did]
    out.append({
        'id': did,
        'name': name,
        'color': color,
        'path': blob(cx, cy, rx, ry, sid(did) % 9973, n=10, jitter=0.18),
        'label_anchor': [cx, cy],
    })
  return out


# Hand-placed. The economic gradient is deliberate: the chawl sits jammed
# against the canal in the dense town, villas and the waterfront estate face
# the lake.
PLACE_XY = {
    'town_square': ((402, 332), 'canal_town'),
    'market': ((426, 350), 'canal_town'),
    'tea_shop': ((386, 354), 'canal_town'),
    'general_store': ((376, 328), 'canal_town'),
    'temple': ((356, 374), 'canal_town'),
    'mosque': ((370, 288), 'canal_town'),
    'church': ((442, 298), 'canal_town'),
    'library': ((412, 286), 'canal_town'),
    'community_center': ((452, 324), 'canal_town'),
    'medical_clinic': ((420, 390), 'canal_town'),
    'school': ((378, 406), 'canal_town'),
    'restaurant': ((446, 366), 'canal_town'),
    'rooftop_lounge': ((398, 308), 'canal_town'),
    'bus_stand': ((330, 332), 'canal_town'),
    'park': ((492, 258), 'backwater_edge'),
    'toddy_shop': ((560, 508), 'kuttanad'),
    'office_building': ((636, 316), 'backwater_edge'),
    'office_cafeteria': ((630, 326), 'backwater_edge'),
    'office_floor_tech': ((636, 314), 'backwater_edge'),
    'office_floor_finance': ((644, 318), 'backwater_edge'),
    'office_floor_creative': ((632, 308), 'backwater_edge'),
    'sunset_apartments': ((342, 388), 'canal_town'),
    'sunset_apartments_common_room': ((348, 394), 'canal_town'),
    'coral_village': ((296, 244), 'beach_ward'),
    'palm_heights': ((334, 500), 'kuttanad'),
    'ocean_view_estates': ((734, 398), 'backwater_edge'),
    'paradise_point': ((792, 176), 'punnamada'),
    # See the Brecksville generator: the nightly marketplace GM reports this
    # for the whole population and no preset declares it.
    'marketplace': ((430, 336), 'canal_town'),

    # ---- Cartographic landmarks -------------------------------------------
    # Everything below has NO counterpart in sim/locations.py: no agent is
    # ever reported at one, and none is reachable through an alias. They are
    # here for the same reason a paper map of Alappuzha names the boat jetty
    # and the coir works -- a backwater town drawn without them is not
    # recognisably that town. Each carries `decor: true`, so a permanently
    # empty marker is self-evidently scenery rather than a data gap.
    'houseboat_jetty': ((716, 250), 'punnamada'),
    'nehru_trophy_point': ((748, 208), 'punnamada'),
    'boat_yard': ((704, 300), 'backwater_edge'),
    'ferry_jetty': ((686, 334), 'backwater_edge'),
    'kayal_homestay': ((762, 356), 'backwater_edge'),
    'paddy_view_homestay': ((616, 588), 'kuttanad'),
    'fish_landing': ((252, 298), 'beach_ward'),
    'coir_works': ((272, 358), 'beach_ward'),
    'sea_bridge_pier': ((186, 302), 'beach_ward'),
}

NEW_PLACES = {
    'marketplace': {'name': 'Market', 'category': 'retail'},
}

# Cartography-only venues. See the note in PLACE_XY.
DECOR_PLACES = {
    'houseboat_jetty': {'name': 'Punnamada Houseboat Jetty',
                        'category': 'waterfront', 'min_zoom': 0},
    'nehru_trophy_point': {'name': 'Nehru Trophy Finishing Point',
                           'category': 'landmark', 'min_zoom': 0},
    'boat_yard': {'name': 'Country Boat Yard', 'category': 'waterfront',
                  'min_zoom': 2},
    'ferry_jetty': {'name': 'Boat Jetty', 'category': 'transit',
                    'min_zoom': 1},
    'kayal_homestay': {'name': 'Kayal Homestay', 'category': 'home',
                       'min_zoom': 2},
    'paddy_view_homestay': {'name': 'Paddy View Homestay', 'category': 'home',
                            'min_zoom': 2},
    'fish_landing': {'name': 'Fish Landing Market', 'category': 'grocery',
                     'min_zoom': 1},
    'coir_works': {'name': 'Coir Cooperative', 'category': 'workplace',
                   'min_zoom': 1},
    'sea_bridge_pier': {'name': 'Alappuzha Pier', 'category': 'landmark',
                        'min_zoom': 0},
}

ALIASES = {
    # run.py hardcodes island date venues (`--first_dates`) for
    # every setting.
    'sunset_cafe': 'tea_shop',
    'lighthouse_hike': 'park',
    'beach': 'park',
}


def places(old_places):
  """Re-places every venue in the new layout.

  Place *identity* is read out of the existing atlas and never invented, with
  the single exception of DECOR_PLACES -- cartographic landmarks that are
  explicitly flagged so nothing downstream can mistake them for simulation
  locations.

  Args:
    old_places: the previous atlas's place list.

  Returns:
    The new place list.

  Raises:
    AssertionError: if the coordinate table and the place inventory disagree,
      which means a location was added or removed without updating this file.
  """
  by_id = {p['id']: p for p in old_places}
  for pid, spec in NEW_PLACES.items():
    if pid in by_id:
      by_id[pid].update(spec)
    else:
      by_id[pid] = dict(spec, id=pid)
  for pid, spec in DECOR_PLACES.items():
    by_id[pid] = dict(spec, id=pid, decor=True)
  missing = set(by_id) - set(PLACE_XY)
  extra = set(PLACE_XY) - set(by_id)
  assert not missing and not extra, f'missing={missing} extra={extra}'
  out = []
  for pid, (xy, dist) in PLACE_XY.items():
    src = by_id[pid]
    p = {
        'id': pid,
        'name': src['name'],
        'category': src['category'],
        'district': dist,
        'xy': [xy[0], xy[1]],
    }
    if src.get('decor'):
      p['decor'] = True
    for key in ('parent', 'floor', 'min_zoom'):
      if src.get(key) is not None:
        p[key] = src[key]
    if src.get('virtual'):
      p['virtual'] = True
    out.append(p)
  return out


# --------------------------------------------------------------------------
# Building stock.
# --------------------------------------------------------------------------

# Dense frontage applies along the canal-town strip and at the office campus.
CORE_ANCHORS = [
    (404, 344),   # town core
    (420, 348),   # market quarter
    (636, 316),   # office campus
]

# Standalone slabs: (x, y, w, h, angle_deg, class).
SLABS = [
    (636, 312, 44, 30, -3, 'commercial'),   # office building
    (652, 332, 28, 20, 5, 'commercial'),    # office annexe
    (378, 404, 46, 24, -2, 'civic'),        # government school
    (430, 350, 34, 26, 4, 'retail'),        # town market hall
    (330, 330, 24, 16, -6, 'industrial'),   # KSRTC bus stand
    (272, 358, 40, 22, 3, 'industrial'),    # coir works
    (704, 300, 30, 18, -5, 'industrial'),   # boat yard
    (252, 298, 26, 16, 2, 'retail'),        # fish landing
]


def buildings(place_list):
  """Generates the decorative building stock.

  Alappuzha is a narrow strip of dense frontage threaded between water on
  three sides, so the exclusion list does most of the work here: nothing may
  be drawn in a canal, in the lake fringe, or out on the paddy.

  Args:
    place_list: the atlas places, used to keep the stock clear of venues.

  Returns:
    A list of {'class', 'path'} dicts.
  """
  avoid = [(p['xy'][0], p['xy'][1]) for p in place_list]
  # Keep stock out of the water, but only just: Alappuzha's whole character is
  # buildings crowding right up to the canal bank, so a wide exclusion here
  # reads as a suburb with a drainage ditch rather than a canal town. An
  # earlier 7-unit pad (plus a tight town radius) left only 55 buildings for
  # the entire town.
  water = [([(x, y) for x, y, _ in spine], half + 4.0)
           for _, _, spine, half in canals()]
  water.append((list(COAST), 52.0))
  water.append((list(LAKE_SHORE), 60.0))

  out = street_buildings(
      roads(),
      district_centres=list(DCENTRE.values()),
      core_anchors=CORE_ANCHORS,
      core_radius=95.0,
      # Tighter than Brecksville: this is a compact town, not a township. But
      # 105 cut the stock off well inside the built-up wash, leaving visibly
      # empty tan. The water and paddy exclusions above are what actually stop
      # houses landing in Kuttanad, so this can afford to be generous.
      town_radius=135.0,
      avoid_points=avoid,
      avoid_radius=13.0,
      avoid_polylines=water,
  )
  out.extend(block_buildings(SLABS, avoid_points=avoid, avoid_radius=12.0))
  return out


def main():
  with open(ATLAS_PATH) as f:
    old = yaml.safe_load(f)

  place_list = places(old['places'])
  doc = {
      'meta': {
          'id': 'kerala',
          'display_name': 'Alappuzha, Kerala',
          'tagline': 'The Venice of the East',
          'bounds': [0, 0, 1000, 700],
          # Beach to the far side of Vembanad is roughly 14 km.
          'km_across': 14.0,
          'north_up': True,
      },
      'terrain': terrain(),
      'roads': roads(),
      'districts': districts(),
      'places': place_list,
      'buildings': buildings(place_list),
      'interiors': old.get('interiors') or {},
      'aliases': ALIASES,
  }

  header = (
      '# Alappuzha, Kerala -- hand-traced stylisation.\n'
      '#\n'
      '# Geometry is GENERATED by _gen/gen_kerala.py; edit that script rather\n'
      '# than this file so the layout stays reproducible.\n'
      '#\n'
      '# Place ids are the location names from sim/locations.py, EXCEPT those\n'
      '# marked `decor: true`, which are cartographic landmarks with no\n'
      '# simulation counterpart -- no agent is ever reported at one. Coords\n'
      '# are abstract map units, not lat/lon; meta.km_across carries the real\n'
      '# world scale. The `buildings` block is decorative stock, generated\n'
      '# from street frontage; it carries no simulation meaning at all.\n'
  )
  with open(ATLAS_PATH, 'w') as f:
    f.write(header)
    yaml.safe_dump(doc, f, sort_keys=False, width=100, default_flow_style=False)

  print('wrote', ATLAS_PATH)
  print('terrain', len(doc['terrain']), 'roads', len(doc['roads']),
        'districts', len(doc['districts']), 'places', len(doc['places']),
        'buildings', len(doc['buildings']))


if __name__ == '__main__':
  main()
