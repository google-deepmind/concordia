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

"""Regenerates the Brecksville atlas geometry.

The first pass at `brecksville.yaml` laid the town out as axis-aligned
rectangles on a 3x4 grid: every district was an identical 100x100 square, the
Cuyahoga was a zigzag, and Chippewa Creek was a straight band spanning the
whole map. That fails the realism bar, so this script rewrites the terrain,
road and district geometry from hand-chosen control points based on the real
town, and re-places every location inside the new layout.

Place *identity* is preserved exactly: ids, display names, categories, parents
and the interiors block are read back out of the existing file and never
invented here. Only coordinates and district membership are recomputed.

Run:
  python3 _gen/gen_brecksville.py
"""

import math
import os

from buildings import block_buildings
from buildings import street_buildings
from geom import blob
from geom import ribbon
from geom import sid
import yaml

ATLAS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'brecksville.yaml',
)

# --------------------------------------------------------------------------
# Real Brecksville, stylised. x: 0 = west, 1000 = east. y: 0 = north.
#
#   I-77           north-south through the west side
#   Ohio Turnpike  east-west across the far south
#   SR 21          "Brecksville Rd", north-south spine through downtown
#   SR 82          "Royalton Rd" west of 21, "Chippewa Rd" east of it
#   Downtown       the 21 x 82 intersection, roughly the town centre
#   Cuyahoga       the valley forming the whole eastern edge
#   Chippewa Creek west-to-east through the Metroparks gorge into the Cuyahoga
#   Reservation    Cleveland Metroparks, the dominant green mass NE/E
# --------------------------------------------------------------------------

DOWNTOWN = (516, 384)

# (id, display name, centre, colour)
DISTRICTS = [
    ('downtown', 'Downtown', (516, 384), '#ef4444'),
    ('snowville', 'Snowville', (132, 286), '#3b82f6'),
    ('oakes', 'Oakes', (300, 148), '#ec4899'),
    ('whitewood', 'Whitewood', (452, 128), '#6366f1'),
    ('chippewa', 'Chippewa', (646, 196), '#10b981'),
    ('fitzwater', 'Fitzwater', (810, 122), '#14b8a6'),
    ('riverview', 'Riverview', (818, 448), '#f59e0b'),
    ('highland', 'Highland', (664, 432), '#f43f5e'),
    ('parkside', 'Parkside', (632, 594), '#84cc16'),
    ('barr_road', 'Barr Road', (392, 574), '#8b5cf6'),
    ('stadium', 'Stadium', (228, 466), '#a855f7'),
]
DCENTRE = {d[0]: d[2] for d in DISTRICTS}

CUYAHOGA_SPINE = [
    (884, 0, 0.7), (872, 70, 0.8), (890, 140, 0.9), (876, 215, 1.0),
    (852, 285, 1.1), (866, 355, 1.0), (890, 420, 0.95), (872, 495, 1.05),
    (846, 560, 1.15), (860, 630, 1.0), (842, 700, 0.9),
]

CHIPPEWA_SPINE = [
    (262, 268, 0.45), (330, 286, 0.5), (398, 266, 0.55), (466, 288, 0.6),
    (534, 272, 0.7), (602, 296, 0.8), (668, 278, 0.9), (734, 300, 1.0),
    (800, 288, 1.05), (862, 306, 1.1),
]


def terrain():
  """Generates all terrain polygons for the Brecksville atlas."""
  t = []

  def add(cls, tid, path, **extra):
    feat = {'class': cls, 'id': tid, 'path': path}
    feat.update(extra)
    t.append(feat)

  # 1. Base. Brecksville is overwhelmingly green; grass is the correct wash.
  add('grass', 'base', 'M 0,0 L 1000,0 L 1000,700 L 0,700 Z')

  # 2. Working farmland survives in the far south-west of the township.
  add('farmland', 'southwest_fields', blob(96, 596, 108, 74, 11, n=8))
  add('farmland', 'barr_fields', blob(300, 648, 92, 52, 12, n=8))

  # 3. Cleveland Metroparks, Brecksville Reservation: the town's defining
  #    feature, ~3,400 acres wrapping the Chippewa gorge in the east.
  add('forest', 'brecksville_reservation', blob(700, 200, 200, 138, 21, n=11))
  add('forest', 'chippewa_gorge_woods', blob(452, 290, 210, 60, 22, n=10))
  # 4. Cuyahoga Valley National Park: the corridor down the eastern edge.
  add(
      'forest',
      'cuyahoga_valley',
      'M 770,0 C 830,30 812,110 828,190 C 844,270 800,330 812,410 '
      'C 824,490 786,545 800,620 C 810,668 792,690 786,700 '
      'L 1000,700 L 1000,0 Z',
  )
  add('forest', 'south_reservation', blob(672, 556, 128, 92, 23, n=10))
  add('forest', 'west_woodlot', blob(150, 154, 96, 70, 24, n=9))

  # 5. Built form. Residential tracts per neighbourhood, painted over the
  #    woodland because several of these streets genuinely run into the park.
  for did, _, (cx, cy), _ in DISTRICTS:
    if did == 'downtown':
      continue
    t.append({
        'class': 'residential',
        'id': f'{did}_tract',
        'path': blob(cx, cy, 96, 76, sid(did) % 9973, n=9, jitter=0.22),
    })

  # 7. Commercial ribbon along Route 82 and up Route 21, then the core.
  add(
      'commercial',
      'royalton_corridor',
      'M 214,350 C 300,336 380,344 456,336 L 456,428 '
      'C 380,420 300,428 214,414 C 190,402 190,362 214,350 Z',
  )
  add(
      'commercial',
      'chippewa_corridor',
      'M 576,338 C 650,326 716,332 778,344 C 800,356 800,404 778,416 '
      'C 716,428 650,422 576,410 Z',
  )
  add(
      'commercial',
      'brecksville_rd_north',
      'M 482,232 C 502,222 538,222 558,232 C 568,268 568,308 558,336 '
      'L 482,336 C 472,308 472,268 482,232 Z',
  )
  add('urban_core', 'town_centre', blob(516, 384, 78, 62, 41, n=10,
                                        jitter=0.16))
  # The four quadrants of the SR 21 x SR 82 crossroads. Downtown Brecksville
  # is the densest thing on the map and the previous single blob made it read
  # as no busier than a residential cul-de-sac.
  add('urban_core', 'downtown_nw', blob(482, 348, 34, 28, 141, n=8,
                                        jitter=0.14))
  add('urban_core', 'downtown_ne', blob(552, 350, 32, 26, 142, n=8,
                                        jitter=0.14))
  add('urban_core', 'downtown_sw', blob(484, 420, 30, 26, 143, n=8,
                                        jitter=0.14))
  add('urban_core', 'downtown_se', blob(552, 418, 34, 28, 144, n=8,
                                        jitter=0.14))
  add('commercial', 'downtown_north_block', blob(518, 300, 44, 30, 145, n=8,
                                                 jitter=0.16))
  add('commercial', 'downtown_south_block', blob(518, 464, 42, 30, 146, n=8,
                                                 jitter=0.16))

  # The office park / former VA campus sits east of I-77 on Route 82, and is
  # where a third of the working population spends its day -- so it gets a
  # commercial apron and a service strip rather than one civic blob.
  add('civic', 'office_campus', blob(276, 396, 66, 50, 42, n=8, jitter=0.18))
  add('commercial', 'office_park_apron', blob(276, 344, 58, 26, 147, n=8,
                                              jitter=0.16))
  add('commercial', 'office_park_south', blob(268, 448, 52, 24, 148, n=8,
                                              jitter=0.16))
  add('industrial', 'i77_service', blob(176, 400, 46, 34, 43, n=7))

  # 6. Water, drawn over the built form so the valley is never paved over.
  add('water', 'cuyahoga_river', ribbon(CUYAHOGA_SPINE, 26), halo=True)
  add('water', 'chippewa_creek', ribbon(CHIPPEWA_SPINE, 13))
  add('wetland', 'valley_flats', blob(840, 620, 74, 58, 31, n=8))

  # 7. Neighbourhood parks, one per tract, offset from the housing.
  for did, _, (cx, cy), _ in DISTRICTS:
    if did == 'downtown':
      continue
    px, py = park_anchor(did)
    t.append({
        'class': 'park',
        'id': f'{did}_green',
        'path': blob(px, py, 44, 34, (sid(did) // 7) % 9973, n=8),
    })
  return t


def park_anchor(did):
  cx, cy = DCENTRE[did]
  # Push the green space away from the town centre so parks sit on the
  # outside edge of each tract, which is how suburban platting works.
  dx, dy = cx - DOWNTOWN[0], cy - DOWNTOWN[1]
  mag = math.hypot(dx, dy) or 1.0
  return (cx + dx / mag * 40, cy + dy / mag * 32)


def roads():
  return [
      {
          'id': 'i77',
          'name': 'I-77',
          'class': 'interstate',
          'shield': {'type': 'interstate', 'number': '77'},
          'geometry': [
              [196, 0], [204, 90], [190, 180], [196, 270], [184, 360],
              [196, 450], [180, 540], [190, 630], [178, 700],
          ],
      },
      {
          'id': 'ohio_turnpike',
          'name': 'Ohio Turnpike',
          'class': 'interstate',
          'shield': {'type': 'interstate', 'number': '80'},
          'geometry': [
              [0, 676], [140, 662], [300, 670], [460, 656], [620, 664],
              [780, 650], [1000, 658],
          ],
      },
      {
          'id': 'route_82_west',
          'name': 'Royalton Rd (SR 82)',
          'class': 'arterial',
          'shield': {'type': 'us_state', 'state': 'OHIO', 'number': '82'},
          'geometry': [
              [0, 394], [90, 388], [196, 384], [300, 390], [400, 382],
              [516, 384],
          ],
      },
      {
          'id': 'route_82_east',
          'name': 'Chippewa Rd (SR 82)',
          'class': 'arterial',
          'shield': {'type': 'us_state', 'state': 'OHIO', 'number': '82'},
          'geometry': [
              [516, 384], [606, 378], [694, 390], [776, 378], [846, 388],
              [922, 378], [1000, 386],
          ],
      },
      {
          'id': 'route_21',
          'name': 'Brecksville Rd (SR 21)',
          'class': 'arterial',
          'shield': {'type': 'us_state', 'state': 'OHIO', 'number': '21'},
          'geometry': [
              [512, 0], [520, 90], [508, 180], [518, 270], [516, 384],
              [522, 470], [510, 560], [518, 650], [512, 700],
          ],
      },
      {
          'id': 'riverview_rd',
          'name': 'Riverview Rd',
          'class': 'residential',
          'geometry': [
              [846, 10], [834, 100], [850, 190], [820, 280], [834, 370],
              [812, 448], [826, 540], [806, 630], [818, 700],
          ],
      },
      {
          'id': 'chippewa_creek_dr',
          'name': 'Chippewa Creek Dr',
          'class': 'residential',
          'geometry': [
              [516, 384], [560, 340], [604, 300], [646, 262], [700, 240],
              [760, 252], [812, 280], [846, 306],
          ],
      },
      {
          'id': 'snowville_rd',
          'name': 'Snowville Rd',
          'class': 'residential',
          'geometry': [
              [132, 386], [124, 340], [132, 286], [120, 230], [140, 176],
              [170, 140],
          ],
      },
      {
          'id': 'oakes_rd',
          'name': 'Oakes Rd',
          'class': 'residential',
          'geometry': [
              [300, 388], [292, 310], [304, 230], [300, 148], [312, 80],
              [304, 20],
          ],
      },
      {
          'id': 'whitewood_rd',
          'name': 'Whitewood Rd',
          'class': 'residential',
          'geometry': [
              [300, 148], [370, 136], [452, 128], [516, 138], [512, 90],
          ],
      },
      {
          'id': 'fitzwater_rd',
          'name': 'Fitzwater Rd',
          'class': 'residential',
          'geometry': [
              [646, 196], [710, 170], [764, 140], [810, 122], [846, 100],
          ],
      },
      {
          'id': 'highland_dr',
          'name': 'Highland Dr',
          'class': 'residential',
          'geometry': [
              [560, 400], [612, 418], [664, 432], [724, 440], [790, 446],
              [818, 448],
          ],
      },
      {
          'id': 'parkside_dr',
          'name': 'Parkside Dr',
          'class': 'residential',
          'geometry': [
              [518, 470], [550, 520], [590, 560], [632, 594], [680, 630],
              [700, 664],
          ],
      },
      {
          'id': 'barr_rd',
          'name': 'Barr Rd',
          'class': 'residential',
          'geometry': [
              [190, 556], [278, 566], [392, 574], [470, 566], [516, 560],
          ],
      },
      {
          'id': 'stadium_dr',
          'name': 'Stadium Dr',
          'class': 'residential',
          'geometry': [
              [300, 390], [268, 424], [228, 466], [206, 510], [190, 556],
          ],
      },
      {
          'id': 'mill_rd',
          'name': 'Mill Rd',
          'class': 'residential',
          'geometry': [
              [196, 270], [300, 258], [400, 266], [472, 288], [516, 320],
          ],
      },
      {
          'id': 'valley_pkwy',
          'name': 'Valley Pkwy',
          'class': 'trail',
          'geometry': [
              [560, 250], [620, 222], [688, 206], [756, 198], [812, 216],
              [846, 250],
          ],
      },
  ]


def districts():
  out = []
  for did, name, (cx, cy), color in DISTRICTS:
    rx, ry = (94, 76) if did != 'downtown' else (86, 70)
    out.append({
        'id': did,
        'name': name,
        'color': color,
        'path': blob(cx, cy, rx, ry, (sid(did) + 5) % 9973, n=9, jitter=0.2),
        'label_anchor': [round(cx), round(cy)],
    })
  return out


# --------------------------------------------------------------------------
# Place layout.
# --------------------------------------------------------------------------

# Downtown, laid out around the SR 21 x SR 82 crossroads.
DOWNTOWN_XY = {
    'town_square': (516, 372),
    'city_hall': (494, 350),
    'library': (540, 348),
    'community_center': (556, 372),
    'fire_station': (484, 404),
    'medical_clinic': (552, 404),
    'church': (490, 322),
    'school': (548, 320),
    'cafe': (500, 388),
    'restaurant': (534, 392),
    'general_store': (478, 370),
    'market': (518, 410),
    'swell_bar': (546, 424),
    'rooftop_lounge': (488, 430),
    'giant_eagle': (612, 366),
    'cvs_pharmacy': (646, 384),
    'shopping_plaza': (692, 370),
    'bbh_high_school': (520, 246),
    'office_park': (276, 396),
    'office_cafeteria': (258, 410),
    'office_floor_tech': (268, 380),
    'office_floor_finance': (292, 392),
    'office_floor_creative': (290, 414),
    'park': (712, 214),
    'chippewa_trail': (604, 268),
    'blossom': (788, 618),
    'sunset_apartments_common_room': (356, 336),
    # Physical marketplace in downtown Brecksville.
    'marketplace': (534, 366),
}

ALIASES = {
    # run.py hardcodes island date venues (`--first_dates`) for
    # every setting, so a Brecksville run schedules dates at e.g.
    # `lighthouse_hike` or `rooftop_lounge`. Those places do not exist here;
    # map them to the nearest Brecksville equivalent. The UI flags every agent
    # placed this way.
    'sunset_cafe': 'cafe',
    'lighthouse_hike': 'park',
    'beach': 'chippewa_trail',
}

SIM_IDS = {
    # run.py `_PLACE_PREFIX_MAP` canonicalises Brecksville home
    # buildings onto the shared island ids before the simulation starts, so
    # runtime logs say `coral_village_unit_118` for a Brecksville Commons home.
    # Exact inverse of that map (same building, not an approximation).
    'sunset_apartments': 'millbrook_apts',
    'coral_village': 'brecksville_commons',
    'palm_heights': 'chippewa_ridge',
    'ocean_view_estates': 'riverview_estates',
    'paradise_point': 'timber_creek',
}

DOWNTOWN_DISTRICT = {
    'giant_eagle': 'downtown',
    'cvs_pharmacy': 'downtown',
    'shopping_plaza': 'downtown',
    'park': 'chippewa',
    'chippewa_trail': 'chippewa',
    'blossom': 'parkside',
    'sunset_apartments_common_room': 'stadium',
}

RESIDENTIAL_XY = {
    'millbrook_apts': ((352, 352), 'stadium'),
    'brecksville_commons': ((560, 452), 'downtown'),
    'chippewa_ridge': ((622, 250), 'chippewa'),
    'riverview_estates': ((790, 470), 'riverview'),
    'timber_creek': ((642, 540), 'parkside'),
}

NEIGHBOURHOODS = [
    'snowville', 'chippewa', 'riverview', 'barr_road', 'oakes', 'whitewood',
    'fitzwater', 'highland', 'parkside', 'stadium',
]

# Angle (degrees, 0 = east) and radius for each amenity within its tract. The
# three green uses cluster on the park side; the bus stop and gas station sit
# toward the road frontage.
AMENITY_LAYOUT = {
    'commons': (0, 0),
    'bar': (200, 34),
    'diner': (250, 38),
    'pizza': (300, 34),
    'salon': (340, 44),
    'daycare': (30, 46),
    'church': (70, 52),
    'gas_station': (150, 58),
    'bus_stop': (175, 44),
    'park': (None, None),
    'dog_park': (None, None),
    'playground': (None, None),
}


# The marketplace in downtown Brecksville. If agents visit during the
# day to socialize or roleplay, they can be placed here.
# Note: Nighttime marketplace trading is a virtual phase, not a physical trip.
NEW_PLACES = {
    'marketplace': {'name': 'Market', 'category': 'retail'},
}


def places(old_places):
  """Generates the full place dictionary with coordinates and zoom levels."""
  by_id = {p['id']: p for p in old_places}
  for pid, spec in NEW_PLACES.items():
    if pid in by_id:
      by_id[pid].update(spec)
    else:
      by_id[pid] = dict(spec, id=pid)
  out = []

  def emit(pid, xy, district, min_zoom=None):
    src = by_id[pid]
    p = {
        'id': pid,
        'name': src['name'],
        'category': src['category'],
        'xy': [round(xy[0]), round(xy[1])],
    }
    if district:
      p['district'] = district
    if src.get('parent'):
      p['parent'] = src['parent']
    if src.get('virtual'):
      p['virtual'] = True
    mz = min_zoom if min_zoom is not None else src.get('min_zoom')
    if mz:
      p['min_zoom'] = mz
    out.append(p)

  for pid, xy in DOWNTOWN_XY.items():
    emit(pid, xy, DOWNTOWN_DISTRICT.get(pid, 'downtown'))

  for pid, (xy, dist) in RESIDENTIAL_XY.items():
    emit(pid, xy, dist)

  for nbr in NEIGHBOURHOODS:
    cx, cy = DCENTRE[nbr]
    px, py = park_anchor(nbr)
    rot = sid(nbr) % 360
    for suffix, (ang, rad) in AMENITY_LAYOUT.items():
      pid = f'{nbr}_{suffix}'
      if ang is None:
        # Green uses live inside the park polygon.
        idx = ['park', 'dog_park', 'playground'].index(suffix)
        a = math.radians(rot + 120 * idx)
        xy = (px + math.cos(a) * 18, py + math.sin(a) * 14)
      else:
        a = math.radians(ang + rot * 0.15)
        xy = (cx + math.cos(a) * rad, cy + math.sin(a) * rad * 0.85)
      minor = suffix not in ('commons', 'park')
      emit(pid, xy, nbr, min_zoom=2 if minor else 0)
  return out


# --------------------------------------------------------------------------
# Building stock.
# --------------------------------------------------------------------------

# Where downtown density applies. The crossroads itself, plus the two places
# people actually commute to: the office park and the Route 82 retail strip.
CORE_ANCHORS = [
    DOWNTOWN,
    (276, 396),   # office park / former VA campus
    (650, 378),   # Chippewa Rd retail strip
]

# Standalone slabs: (x, y, w, h, angle_deg, class). These are the structures
# too big to be produced by street frontage -- the office towers, the school,
# the supermarket and its parking-lot neighbours.
SLABS = [
    (262, 380, 46, 30, -4, 'commercial'),    # office park, north building
    (292, 404, 40, 34, 6, 'commercial'),     # office park, south building
    (268, 428, 30, 20, -8, 'commercial'),    # office park annexe
    (520, 240, 62, 30, 2, 'civic'),          # BBH high school
    (612, 360, 48, 30, -3, 'retail'),        # Giant Eagle
    (694, 366, 54, 26, 2, 'retail'),         # shopping plaza
    (176, 398, 34, 24, 5, 'industrial'),     # I-77 service
    (500, 356, 26, 22, -6, 'civic'),         # city hall
    (540, 344, 24, 18, 4, 'civic'),          # library
]


def buildings(place_list):
  """Generates the decorative building stock.

  Args:
    place_list: the atlas places, used only to keep generated stock clear of
      the venues that get their own category-shaped footprint at zoom.

  Returns:
    A list of {'class', 'path'} dicts.
  """
  avoid = [(p['xy'][0], p['xy'][1]) for p in place_list]
  water = [
      ([(x, y) for x, y, _ in CUYAHOGA_SPINE], 40.0),
      ([(x, y) for x, y, _ in CHIPPEWA_SPINE], 24.0),
  ]
  out = street_buildings(
      roads(),
      district_centres=list(DCENTRE.values()),
      core_anchors=CORE_ANCHORS,
      core_radius=115.0,
      town_radius=145.0,
      avoid_points=avoid,
      avoid_radius=17.0,
      avoid_polylines=water,
  )
  out.extend(block_buildings(SLABS, avoid_points=avoid, avoid_radius=13.0))
  return out


def main():
  with open(ATLAS_PATH) as f:
    old = yaml.safe_load(f)

  place_list = places(old['places'])
  doc = {
      'meta': {
          'id': 'brecksville',
          'display_name': 'Brecksville',
          'tagline': 'Cuyahoga County, Ohio',
          'bounds': [0, 0, 1000, 700],
          # The map covers roughly the 5 x 3.5 mile township plus the valley.
          'km_across': 8.0,
          'north_up': True,
      },
      'terrain': terrain(),
      'roads': roads(),
      'districts': districts(),
      'places': place_list,
      'buildings': buildings(place_list),
      'interiors': old.get('interiors') or {},
      'aliases': ALIASES,
      'sim_ids': SIM_IDS,
  }

  header = (
      '# Brecksville, Ohio -- hand-traced stylisation.\n'
      '#\n'
      '# Geometry is GENERATED by _gen/gen_brecksville.py; edit that script\n'
      '# rather than this file so the layout stays reproducible.\n'
      '#\n'
      '# Place ids are the location names from sim/locations.py. Coordinates\n'
      '# are abstract map units, not lat/lon; meta.km_across carries the real\n'
      '# world scale.\n'
  )
  with open(ATLAS_PATH, 'w') as f:
    f.write(header)
    yaml.safe_dump(doc, f, sort_keys=False, width=100, default_flow_style=False)

  print('wrote', ATLAS_PATH)
  print(
      'terrain', len(doc['terrain']),
      'roads', len(doc['roads']),
      'districts', len(doc['districts']),
      'places', len(doc['places']),
      'buildings', len(doc['buildings']),
  )
  before = {p['id'] for p in old['places']} | set(NEW_PLACES)
  after = {p['id'] for p in doc['places']}
  assert before == after, (
      f'place id drift: missing={sorted(before - after)} '
      f'added={sorted(after - before)}'
  )
  print('place ids preserved exactly')


if __name__ == '__main__':
  main()
