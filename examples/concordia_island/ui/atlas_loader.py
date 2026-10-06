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

"""World Atlas loader for the Concordia map dashboard.

An *atlas* is a declarative description of the physical geography of one
simulation setting: terrain polygons, the road network, districts, and the
places agents can occupy. Atlases live in `data/atlas/<id>.yaml` and are keyed
by the same location IDs used by `sim/locations.py`, so the atlas is a pure
presentation join over the simulation's own vocabulary.

Coordinate system
-----------------
Each atlas declares `meta.bounds: [x0, y0, x1, y1]` in abstract *map units*.
All geometry in the file is expressed in those units. The client maps map units
to screen pixels via a single transform, so the same atlas renders correctly at
any viewport size or zoom level. Map units are not lat/lon; `meta.geo_anchor`
optionally records the real-world corners for scale-bar purposes.

Schema
------
meta:
  id, display_name, tagline, bounds, scale_bar_km
  geo_anchor: {nw: [lat, lon], se: [lat, lon]}   # optional
terrain:   ordered list of {class, id, path}     # painted back-to-front
roads:     list of {id, name, class, geometry, shield?}
districts: list of {id, name, color, path?, label_anchor}
places:    list of {id, name, category, district?, parent?, xy, min_zoom?}
buildings: optional list of {class, path}  -- decorative footprints
interiors: optional {building_id: {floors: [...]}}
aliases:   optional {runtime_location_id: place_id}  -- approximate venue
sim_ids:   optional {runtime_location_id: place_id}  -- exact renames

`places[].id` MUST match a location name from `sim/locations.py`. Validation is
strict: unknown or duplicate IDs raise rather than being silently placed at a
random coordinate (the behaviour of the previous dashboard).
"""

from __future__ import annotations

import dataclasses
import functools
import os
from typing import Any

import yaml


# Terrain classes understood by the renderer. Anything else is a hard error so
# that a typo in an atlas surfaces immediately instead of painting nothing.
TERRAIN_CLASSES = frozenset({
    'ocean',
    'water',
    'wetland',
    'sand',
    'grass',
    'park',
    'forest',
    'farmland',
    'urban_core',
    'residential',
    'commercial',
    'civic',
    'industrial',
})

ROAD_CLASSES = frozenset({
    'interstate',
    'arterial',
    'residential',
    'trail',
    'rail',
    'ferry',
    'canal',
})

# Individual building footprints. These are cartography, not simulation state:
# they say "this block is built up and here is roughly how", the same way a
# paper map shades a town. They carry no agent data and nothing keys off them.
BUILDING_CLASSES = frozenset({
    'generic',
    'residential',
    'commercial',
    'retail',
    'civic',
    'industrial',
})

# Place categories drive icon, marker colour and default min_zoom.
PLACE_CATEGORIES = frozenset({
    'home',
    'workplace',
    'grocery',
    'retail',
    'food',
    'bar',
    'cafe',
    'civic',
    'education',
    'health',
    'worship',
    'park',
    'nature',
    'transit',
    'waterfront',
    'landmark',
    'plaza',
})


class AtlasError(Exception):
  """Raised when an atlas file is missing, malformed, or inconsistent."""


@dataclasses.dataclass(frozen=True)
class Atlas:
  """A parsed, validated atlas."""

  meta: dict[str, Any]
  terrain: list[dict[str, Any]]
  roads: list[dict[str, Any]]
  districts: list[dict[str, Any]]
  places: list[dict[str, Any]]
  interiors: dict[str, Any]
  # Generated building stock. Purely decorative; see BUILDING_CLASSES.
  buildings: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  # Runtime location ids with no place of their own in this geography, mapped
  # onto the nearest real one. See `_validate` for why this exists.
  aliases: dict[str, str] = dataclasses.field(default_factory=dict)
  # Exact renames: the runner canonicalises setting-specific building ids onto
  # the shared island ids (e.g. `brecksville_commons` -> `coral_village`, see
  # `_PLACE_PREFIX_MAP` in run.py). These map them back. Unlike
  # `aliases` these are the same place, so the UI does not flag them.
  sim_ids: dict[str, str] = dataclasses.field(default_factory=dict)

  @property
  def id(self) -> str:
    return self.meta['id']

  @property
  def place_ids(self) -> frozenset[str]:
    return frozenset(p['id'] for p in self.places)

  def to_json_dict(self) -> dict[str, Any]:
    """Returns the wire format consumed by the browser."""
    return {
        'meta': self.meta,
        'terrain': self.terrain,
        'roads': self.roads,
        'districts': self.districts,
        'places': self.places,
        'buildings': self.buildings,
        'interiors': self.interiors,
        'aliases': self.aliases,
        'sim_ids': self.sim_ids,
    }


def _atlas_dir() -> str:
  """Locates data/atlas/ relative to this module, in source or runfiles."""
  here = os.path.dirname(os.path.abspath(__file__))
  candidate = os.path.join(os.path.dirname(here), 'data', 'atlas')
  if os.path.isdir(candidate):
    return candidate
  raise AtlasError(
      f'Atlas directory not found. Looked for {candidate!r}. If running under '
      'runner, ensure the atlas YAML filegroup is listed in the target\'s '
      'data = [...] attribute.'
  )


def available_atlases() -> list[str]:
  """Returns the ids of all atlases on disk, sorted."""
  try:
    names = os.listdir(_atlas_dir())
  except OSError as e:
    raise AtlasError(f'Cannot list atlas directory: {e}') from e
  return sorted(
      n[:-5] for n in names if n.endswith('.yaml') and not n.startswith('_')
  )


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise AtlasError(msg)


def _validate(raw: dict[str, Any], atlas_id: str) -> Atlas:
  """Validates a parsed YAML document and returns an Atlas."""
  _require(isinstance(raw, dict), f'{atlas_id}: top level must be a mapping')

  meta = raw.get('meta')
  _require(isinstance(meta, dict), f'{atlas_id}: missing "meta" section')
  for key in ('id', 'display_name', 'bounds'):
    _require(key in meta, f'{atlas_id}: meta.{key} is required')
  _require(
      meta['id'] == atlas_id,
      f'{atlas_id}: meta.id is {meta["id"]!r}, expected {atlas_id!r} to match '
      'the filename',
  )
  bounds = meta['bounds']
  _require(
      isinstance(bounds, list) and len(bounds) == 4,
      f'{atlas_id}: meta.bounds must be [x0, y0, x1, y1]',
  )
  _require(
      bounds[2] > bounds[0] and bounds[3] > bounds[1],
      f'{atlas_id}: meta.bounds must have positive extent, got {bounds}',
  )

  terrain = raw.get('terrain') or []
  for i, feat in enumerate(terrain):
    _require(
        feat.get('class') in TERRAIN_CLASSES,
        f'{atlas_id}: terrain[{i}] has unknown class {feat.get("class")!r}. '
        f'Valid: {sorted(TERRAIN_CLASSES)}',
    )
    _require(
        bool(feat.get('path')), f'{atlas_id}: terrain[{i}] is missing "path"'
    )

  roads = raw.get('roads') or []
  for i, road in enumerate(roads):
    _require(
        road.get('class') in ROAD_CLASSES,
        f'{atlas_id}: roads[{i}] has unknown class {road.get("class")!r}. '
        f'Valid: {sorted(ROAD_CLASSES)}',
    )
    geom = road.get('geometry')
    _require(
        isinstance(geom, list) and len(geom) >= 2,
        f'{atlas_id}: roads[{i}] ({road.get("id")}) needs >= 2 points',
    )

  districts = raw.get('districts') or []
  district_ids = set()
  for i, dist in enumerate(districts):
    did = dist.get('id')
    _require(bool(did), f'{atlas_id}: districts[{i}] is missing "id"')
    _require(did not in district_ids, f'{atlas_id}: duplicate district {did!r}')
    district_ids.add(did)

  places = raw.get('places') or []
  _require(bool(places), f'{atlas_id}: atlas has no places')
  seen: set[str] = set()
  for i, place in enumerate(places):
    pid = place.get('id')
    _require(bool(pid), f'{atlas_id}: places[{i}] is missing "id"')
    _require(pid not in seen, f'{atlas_id}: duplicate place id {pid!r}')
    seen.add(pid)
    xy = place.get('xy')
    _require(
        isinstance(xy, list) and len(xy) == 2,
        f'{atlas_id}: place {pid!r} needs xy: [x, y]',
    )
    _require(
        bounds[0] <= xy[0] <= bounds[2] and bounds[1] <= xy[1] <= bounds[3],
        f'{atlas_id}: place {pid!r} at {xy} falls outside bounds {bounds}',
    )
    _require(
        place.get('category') in PLACE_CATEGORIES,
        f'{atlas_id}: place {pid!r} has unknown category '
        f'{place.get("category")!r}. Valid: {sorted(PLACE_CATEGORIES)}',
    )
    dist = place.get('district')
    _require(
        dist is None or dist in district_ids,
        f'{atlas_id}: place {pid!r} references unknown district {dist!r}',
    )

  # Parents must themselves be places, so the interior tier can nest cleanly.
  for place in places:
    parent = place.get('parent')
    _require(
        parent is None or parent in seen,
        f'{atlas_id}: place {place["id"]!r} has parent {parent!r} which is not '
        'a place in this atlas',
    )

  buildings = raw.get('buildings') or []
  _require(
      isinstance(buildings, list), f'{atlas_id}: "buildings" must be a list'
  )
  for i, b in enumerate(buildings):
    _require(
        b.get('class') in BUILDING_CLASSES,
        f'{atlas_id}: buildings[{i}] has unknown class {b.get("class")!r}. '
        f'Valid: {sorted(BUILDING_CLASSES)}',
    )
    _require(
        bool(b.get('path')), f'{atlas_id}: buildings[{i}] is missing "path"'
    )

  # Aliases exist because the simulation emits location ids that no
  # `sim/locations.py` preset defines. Two known sources: the nightly
  # marketplace game master, and `run.py` hardcoding island
  # date venues (lighthouse_hike, rooftop_lounge, ...) for every setting. Rather
  # than rendering a third of the population as "unknown", an atlas may map
  # those ids onto its nearest real place -- but the mapping is declared in
  # data, validated here, and flagged in the UI, never applied invisibly.
  aliases = raw.get('aliases') or {}
  _require(
      isinstance(aliases, dict), f'{atlas_id}: "aliases" must be a mapping'
  )
  for src, dst in aliases.items():
    _require(
        src not in seen,
        f'{atlas_id}: alias {src!r} shadows a real place of the same name',
    )
    _require(
        dst in seen,
        f'{atlas_id}: alias {src!r} points at {dst!r}, which is not a place '
        'in this atlas',
    )

  sim_ids = raw.get('sim_ids') or {}
  _require(
      isinstance(sim_ids, dict), f'{atlas_id}: "sim_ids" must be a mapping'
  )
  for src, dst in sim_ids.items():
    _require(
        src not in seen and src not in aliases,
        f'{atlas_id}: sim_id {src!r} shadows a place or alias of the same name',
    )
    _require(
        dst in seen,
        f'{atlas_id}: sim_id {src!r} points at {dst!r}, which is not a place '
        'in this atlas',
    )

  return Atlas(
      meta=meta,
      terrain=terrain,
      roads=roads,
      districts=districts,
      places=places,
      buildings=buildings,
      interiors=raw.get('interiors') or {},
      aliases=aliases,
      sim_ids=sim_ids,
  )


@functools.lru_cache(maxsize=16)
def load_atlas(atlas_id: str) -> Atlas:
  """Loads and validates one atlas by id. Cached.

  Args:
    atlas_id: Basename of the YAML file, e.g. 'brecksville'.

  Returns:
    The validated Atlas.

  Raises:
    AtlasError: if the file is missing, unparseable, or fails validation.
  """
  path = os.path.join(_atlas_dir(), f'{atlas_id}.yaml')
  if not os.path.exists(path):
    raise AtlasError(
        f'No atlas named {atlas_id!r} at {path}. '
        f'Available: {available_atlases()}'
    )
  try:
    with open(path) as f:
      raw = yaml.safe_load(f)
  except yaml.YAMLError as e:
    raise AtlasError(f'{atlas_id}: YAML parse error: {e}') from e
  return _validate(raw, atlas_id)


# Maps a `sim/locations.py` setting preset name onto an atlas id. The presets
# `ohio_suburb` and `brecksville_1000` are both Brecksville; `ohio_suburb` is
# the deprecated earlier name.
SETTING_TO_ATLAS = {
    'island': 'concordia_island',
    'kerala': 'kerala',
    'ohio_suburb': 'brecksville',
    'brecksville_1000': 'brecksville',
}


def resolve_atlas_id(setting: str | None, personas_dir: str = '') -> str:
  """Chooses an atlas from a setting name, falling back to the personas path.

  The simulation itself selects a setting by substring-matching the personas
  date label (see run.py), so we mirror that here for runs where
  the setting is not passed explicitly.

  Args:
    setting: A `sim/locations.py` preset name, or None.
    personas_dir: local storage personas path, used as a fallback signal.

  Returns:
    An atlas id. Defaults to 'concordia_island'.
  """
  if setting and setting in SETTING_TO_ATLAS:
    return SETTING_TO_ATLAS[setting]
  haystack = (personas_dir or '').lower()
  # Order matters: 'brecksville' must win over the older 'ohio_suburb' label,
  # and both describe the same place.
  for needle, atlas_id in (
      ('brecksville', 'brecksville'),
      ('ohio_suburb', 'brecksville'),
      ('kerala', 'kerala'),
  ):
    if needle in haystack:
      return atlas_id
  return 'concordia_island'
