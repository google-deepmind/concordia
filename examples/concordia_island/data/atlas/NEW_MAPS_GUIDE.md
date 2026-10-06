# Atlas Authoring & Map Generation Guide

This guide describes how to create, test, and register new geographical maps
(atlases) for Concordia Island simulations and the map dashboard.

--------------------------------------------------------------------------------

## 1. Overview & Directory Structure

Map assets and generators live under:
`examples/concordia_island

```
data/atlas/
├── concordia_island.yaml   # Default tropical island atlas
├── brecksville.yaml        # Brecksville, OH suburban atlas (generated)
├── kerala.yaml             # Alappuzha, Kerala backwater/delta atlas (generated)
└── _gen/
    ├── geom.py             # Deterministic geometric primitives (splines, parcels, ribbons)
    ├── buildings.py        # Algorithmic building footprint generators
    ├── check_atlas.py      # Automated validation linter (road proximity, water collisions)
    ├── gen_brecksville.py  # Generator script for brecksville.yaml
    └── gen_kerala.py       # Generator script for kerala.yaml
```

Generated atlas YAML files are checked into version control and packaged into
the `:atlas_data` filegroup in `concordia_island/BUILD`.

--------------------------------------------------------------------------------

## 2. Atlas Schema (`<geography>.yaml`)

Each atlas YAML file consists of five top-level sections:

```yaml
meta:
  id: <geography_id>          # e.g. brecksville, kerala, lagos
  display_name: <Display Name> # e.g. "Brecksville, OH"
  tagline: <Short Tagline>     # e.g. "A suburb of Cleveland, OH"
  bounds: [0, 0, 1000, 700]    # [min_x, min_y, max_x, max_y] in map coordinate units
  km_across: 4.5              # Real-world scale in kilometers across the horizontal axis

terrain: [...]
roads: [...]
districts: [...]
places: [...]
buildings: [...]
aliases: {...}                # Optional
```

### Coordinate Space

*   Abstract integer map units (typically `1000 × 700` or `1200 × 800`).
*   `[0, 0]` is top-left; X increases east, Y increases south.
*   The frontend uses a zoom transform to map these units into canvas pixels at
    whatever device pixel ratio (DPR) is active.

### Terrain Classes

`ocean` · `water` · `wetland` · `sand` · `grass` · `park` · `forest` ·
`farmland` · `urban_core` · `residential` · `commercial` · `civic` ·
`industrial`

Each polygon is defined as an SVG path string (`M ... L ... Z` or Catmull-Rom
splines):

```yaml
terrain:
  - id: background
    class: grass
    path: "M 0,0 L 1000,0 L 1000,700 L 0,700 Z"
```

> **Rendering Order Note**: The renderer injects the built-up tan wash *before
> the first built terrain feature* (`urban_core`, `residential`, `commercial`,
> `civic`, `industrial`). Always place natural background cover (grass, ocean)
> first, followed by built areas, followed by water cutouts and local
> parks/groves.

### Road Classes

`interstate` · `arterial` · `residential` · `trail` · `rail` · `ferry` · `canal`

```yaml
roads:
  - id: route_82
    name: Route 82
    class: arterial
    geometry: [[0, 350], [500, 350], [1000, 350]]
    shield: {type: us_state, state: "OHIO", number: "82"}
```

*   Optional `bridge: true`: draws road parapets instead of casings (used where
    roads cross canals/rivers).

### Districts

Neighborhood or district chips shown at regional zoom (LOD 0–1):

```yaml
districts:
  - id: downtown
    name: Downtown
    color: "#ef4444"
    label_anchor: [516, 384]
    path: "M 450,300 L 550,300 L 550,400 L 450,400 Z" # Optional outline
```

### Places & Categories

Places correspond to simulation destination IDs emitted by `sim/locations.py`:

```yaml
places:
  - id: town_square
    name: Town Square
    category: plaza
    xy: [516, 372]
    district: downtown
    min_zoom: 0
```

Standard categories: `home`, `workplace`, `grocery`, `retail`, `food`, `cafe`,
`bar`, `civic`, `education`, `health`, `worship`, `park`, `nature`, `transit`,
`waterfront`, `landmark`, `plaza`.

### Buildings (Cartographic Footprints)

Decorative footprint polygons visible at block zoom (LOD 3) and interior zoom
(LOD 4):

```yaml
buildings:
  - id: b_001
    footprint: [[510, 360], [530, 360], [530, 380], [510, 380]]
    class: commercial
```

--------------------------------------------------------------------------------

## 3. Authoring Generators (`_gen/`)

For complex geographies, author a deterministic generator script (e.g.
`gen_<name>.py`) using the utilities in `_gen/`.

### Shared Geometric Helpers (`geom.py`)

*   `blob(cx, cy, rx, ry, seed, n_points=12, jitter=0.25)`: Organic closed
    polygon. Use for natural features (groves, forest blobs, water bodies).
*   `parcel(cx, cy, w, h, seed, rot=0.0, jitter=2.0)`: Straight-edged
    quadrilateral with slight corner perturbation. Use for surveyed/reclaimed
    land (agricultural polders, city blocks, building lots).
*   `ribbon(spine, half_width)`: Polyline corridor expanded into a closed SVG
    path with rounded or square end-caps. Use for wide canals and river
    corridors.
*   `closed_spline(pts)` / `open_spline(pts)`: Catmull-Rom smoothing for curved
    paths.
*   `sid(name)`: Stable CRC32 integer hashing for deterministic procedural
    generation without floating-point drift across platforms.

### Building Footprint Helpers (`buildings.py`)

*   `street_buildings(road_geom, offset, lot_width, lot_depth, ...)`: Generates
    rows of rectangular building lots along street frontages.
*   `block_buildings(bbox, rows, cols, ...)`: Generates grids of building
    footprints for downtown urban cores.

--------------------------------------------------------------------------------

## 4. Connecting Simulation Presets to Map IDs

1.  **Preset Definition**: Define your location identities in `sim/locations.py`
    (e.g. `_make_<name>_preset()`).
2.  **Preset Registration**: Register your preset factory in `SETTING_PRESETS`
    in `sim/locations.py`.
3.  **Atlas Resolver**: Map the simulation preset name to the atlas ID in
    `ui/atlas_loader.py`:

    ```python
    SETTING_TO_ATLAS = {
        'island': 'concordia_island',
        'kerala': 'kerala',
        'ohio_suburb': 'brecksville',
        'brecksville_1000': 'brecksville',
        '<new_setting>': '<new_atlas_id>',
    }
    ```
4.  **BUILD targets**: Ensure the new YAML is covered by `:atlas_data` in
    `examples/concordia_island

--------------------------------------------------------------------------------

## 5. Validation & Testing Protocol

Before deploying or checking in a new atlas:

### A. Run the Linter

```bash
python3 examples/concordia_island
```

Checks:

*   No land venues accidentally sit inside water polygons (except those
    explicitly marked `category: waterfront`).
*   All places sit within a reasonable distance of a road.
*   Coordinate boundaries and polygon closures are valid.

### B. Launch in Geography-Only Mode

Verify visual appearance without needing a running simulation:

```bash
/google/bin/releases/arca9-local-runner-cli/runner-for-agents run \
  //examples/concordia_island -- \
  --geography=<new_atlas_id> --port=8145
```

Open `http://localhost:8145/` (or `http://<hostname>.c.localhost:8145/`) in
your browser to inspect:

*   Region level (LOD 0): District chips and counts.
*   Street level (LOD 2): Road casing, street names, venue category glyphs.
*   Block level (LOD 3): Building footprints, enlarged category icons,
    sublabels.

### C. Server Caching Warning

> `atlas_loader.load_atlas()` caches atlas YAML files in memory using
> `@functools.lru_cache`. If you regenerate or edit an atlas YAML file, you must
> **restart the dashboard server** on that port to see the changes.
