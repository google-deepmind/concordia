# Concordia Map Dashboard

A multiscale, terrain-rendered cartographic dashboard for Concordia Island
simulations. Shows where every agent is, how they move, and lets you scrub
through the simulation timeline tick by tick.

## Quick start

```bash
# Replay the bundled Gemini 2.5 Flash ESA job-loss run from the paper
# (data/paper_runs/job_loss/gemini_2_5_flash_esa) on the concordia_island map
python -m examples.concordia_island.ui.map_dashboard

# View one of your own finished runs
python -m examples.concordia_island.ui.map_dashboard \
    --run_dir=./local_runs/my_run --geography=brecksville

# View just the map, no agent data (useful for checking cartography)
python -m examples.concordia_island.ui.map_dashboard --geography=brecksville

# Override the port (default 8080)
python -m examples.concordia_island.ui.map_dashboard --geography=kerala --port=8138
```

Then open `http://localhost:<port>/` in your browser. The unified Simulation
Studio (`ui/simulation_studio.py`) serves the same map under `/map`.

## Flags

| Flag                | Default                                                       | Description                             |
| ------------------- | ------------------------------------------------------------- | --------------------------------------- |
| `--run_dir`         | bundled `data/paper_runs/job_loss/gemini_2_5_flash_esa`       | A run's `--output_dir`                  |
:                     :                                                               : (`simulation_state.json`,               :
:                     :                                                               : `location_history.json`, ...) or a      :
:                     :                                                               : `data/paper_runs/` folder.              :
| `--id`              | (none)                                                        | Run looked up under `./local_runs/`     |
:                     :                                                               : when `--run_dir` is not given.          :
| `--geography`       | auto                                                          | Atlas id: `concordia_island`,           |
:                     :                                                               : `brecksville`, or `kerala`. Picked from :
:                     :                                                               : `--setting` / `--personas_dir`; else    :
:                     :                                                               : `concordia_island` for the paper runs   :
:                     :                                                               : and `brecksville` (run.py's default     :
:                     :                                                               : population) for other runs.             :
| `--setting`         | (none)                                                        | Simulation preset name (e.g.            |
:                     :                                                               : `brecksville_1000`, `island`,           :
:                     :                                                               : `kerala`). Used to auto-pick geography. :
| `--port`            | 8080                                                          | HTTP port.                              |
| `--expected_agents` | 100                                                           | Expected number of agents (for data     |
:                     :                                                               : quality reporting).                     :
| `--expected_ticks`  | 100                                                           | Expected number of simulation ticks.    |
| `--start_time`      | `Thursday, January 1st, 7:00 AM`                              | Simulation time at tick 1.              |
| `--tick_interval`   | 120                                                           | Minutes of simulated time per tick.     |

## Using the dashboard

### Navigation

-   **Scroll** to zoom in/out (pinch on trackpad)
-   **Click + drag** to pan
-   **+/−** buttons or keyboard `+`/`-` to zoom
-   **⏎ (fit)** button or `0` to reset view
-   Click a **district bubble** to zoom into that neighborhood
-   Click the **breadcrumb** to zoom back out

### URL parameters

The view is encoded in the URL hash, so a particular framing can be shared or
scripted. All are optional and combine with `&`.

| Param      | Meaning                                                         |
| ---------- | --------------------------------------------------------------- |
| `g`        | Geography id, e.g. `#g=kerala`                                  |
| `k`        | Zoom scale, e.g. `#k=2.5`                                       |
| `c`        | Centre in map units, e.g. `#c=516,384`                          |
| `nopoll=1` | Load once and stop polling. Use for screenshots and for pinning |
:            : a view while debugging.                                         :

```
http://localhost:8143/#g=kerala&k=3&c=404,344&nopoll=1
```

### Zoom levels (LOD)

The map has five level-of-detail tiers that answer different questions:

| LOD | Name     | Zoom    | What you see                                |
| --- | -------- | ------- | ------------------------------------------- |
| 0   | Region   | < 1.15× | District bubbles with agent counts          |
| 1   | District | 1.15–2× | District bubbles + major place markers      |
| 2   | Street   | 2–3.4×  | All place markers with geometric glyphs     |
| 3   | Block    | 3.4–6×  | **Enlarged emoji icons** per building type, |
:     :          :         : category sub-labels                         :
| 4   | Interior | > 6×    | Child places (rooms, floors) if defined     |

### Timeline

The footer timeline lets you scrub through the simulation:

-   **◀ ▶** step back/forward one tick
-   **⏮ ⏭** jump to first/latest tick
-   **▶ (play)** auto-advance through ticks
-   **Speed selector**: 0.5×, 1×, 2×, 4×
-   **Click/drag** on the timeline track to jump to any tick
-   The timeline shows **coverage bands** — bright segments where movement data
    exists, dim where agents are carried forward from earlier ticks

### Layer toggles

Open the **Layers** panel on the left to toggle:

| Layer                | Default | What it does                               |
| -------------------- | ------- | ------------------------------------------ |
| Terrain & land cover | ✅       | Background terrain blobs (grass, water,    |
:                      :         : urban, etc.)                               :
| Roads                | ✅       | Interstate shields, arterial roads, trails |
| Building footprints  | ✅       | Subtle grey footprint shapes behind place  |
:                      :         : markers                                    :
| Place markers        | ✅       | Venue markers (geometric glyphs at low     |
:                      :         : zoom, emoji icons at high zoom)            :
| Labels               | ✅       | Place and district names                   |
| Movement trails      | ❌       | Lines showing each agent's recent path     |
:                      :         : between places                             :
| Flow ribbons         | ❌       | Aggregate directional ribbons between      |
:                      :         : places                                     :
| Occupancy heatmap    | ❌       | Density overlay                            |

### Data quality pills

The top bar shows real-time data quality:

-   **Agents**: how many are visible out of expected
-   **Unknown** (red): agents at locations not in the atlas
-   **Approximate** (amber): agents mapped via alias to a nearby known place
-   **Carried forward** (grey): agents with no log at this tick, shown at their
    last known position

### Agent roster

The right panel lists all agents. Click one to:

-   Highlight it on the map
-   See its current location, places visited, color assignment
-   The map auto-centers on the selected agent

## Architecture

```
map_dashboard.py ─────── HTTP server (stdlib http.server, threaded)
├── atlas_loader.py ──── YAML atlas loading + validation
├── map_data.py ──────── Run-directory loading, movement log parsing, trajectory
│                         reconstruction with forward-fill and age tracking
└── static/ ──────────── Frontend (zero dependencies, vanilla ES modules)
    ├── index.html ────── App shell, layout, panels
    ├── map.css ───────── All styling
    ├── app.js ────────── Boot, fetch loop, timeline, roster, layer toggles
    ├── atlas.js ──────── View transform, categories, LOD thresholds, colors
    ├── terrain.js ────── Canvas terrain painter (Catmull-Rom splines)
    ├── markers.js ────── SVG place/district markers, emoji icons, labels
    └── agents.js ─────── Agent dot rendering, tweening, trails, flows, heatmap
```

### Rendering layers

Five visual layers share a single `View` transform:

1.  **`#layer-terrain`** (canvas) — terrain blobs, roads, highway shields,
    footprints
2.  **`#layer-vector`** (SVG) — place markers, district bubbles, hit testing
3.  **`#layer-agents`** (canvas) — agent dots, trails, flows, heatmap
4.  **`#tint`** (div) — time-of-day color overlay
5.  **`#layer-labels`** (HTML) — text labels with collision culling

### API endpoints

| Endpoint                    | Description                                    |
| --------------------------- | ---------------------------------------------- |
| `GET /`                     | Serves `index.html`                            |
| `GET /static/<path>`        | Serves frontend assets                         |
| `GET /api/config`           | Boot config: run id, geography list, tick      |
:                             : settings                                       :
| `GET /api/atlas?g=<id>`     | Full atlas data for rendering (terrain, roads, |
:                             : places)                                        :
| `GET /api/data`             | Live snapshot: agent positions, tick progress, |
:                             : sim clock                                      :
| `GET /api/location_history` | Full movement log: per-tick snapshots with     |
:                             : staleness tracking                             :

### Embedding in simulation_studio.py

The server is factored for reuse. To mount as a panel:

```python
from ...ui import map_dashboard

# Call once at startup
config = map_dashboard.api_config()

# Route API requests
status, body = map_dashboard.handle_api(path, query_params)

# Serve static files from
map_dashboard.STATIC_DIR
```

## Atlas system

### What is an atlas?

An atlas is a YAML file that describes the physical geography of one simulation
setting. It lives at `data/atlas/<id>.yaml` and contains:

-   **Terrain** — polygons painted as organic blobs (grass, water, forest,
    urban, etc.)
-   **Roads** — polylines with highway shields
-   **Districts** — named regions agents can occupy
-   **Places** — individual venues with category, coordinates, and zoom tier
-   **Aliases** — mappings for runtime location IDs that have no atlas entry

### Available atlases

| Id                 | File                    | Places | Description       |
| ------------------ | ----------------------- | ------ | ----------------- |
| `concordia_island` | `concordia_island.yaml` | 29     | Fictional island, |
:                    :                         :        : hand-authored     :
| `brecksville`      | `brecksville.yaml`      | 153    | Brecksville, Ohio |
:                    :                         :        : (generated)       :
| `kerala`           | `kerala.yaml`           | 28     | Alappuzha, Kerala |
:                    :                         :        : (generated)       :

### Atlas YAML schema

```yaml
meta:
  id: my_world               # Must match filename (my_world.yaml)
  display_name: My World     # Shown in the geography picker
  tagline: A test world       # Optional subtitle
  bounds: [0, 0, 1000, 700]  # [x0, y0, x1, y1] in map units
  km_across: 6.0             # Real-world width for the scale bar
  north_up: true             # Optional

# Terrain: painted back to front. Earlier entries are behind later ones.
terrain:
  - class: ocean              # See "Terrain classes" below
    id: sea
    path: "M 0,0 L 1000,0 L 1000,700 L 0,700 Z"   # SVG path

  - class: grass
    id: main_field
    path: "M 100,100 L 400,80 L 450,300 L 120,320 Z"

# Roads: rendered as polylines with casings and optional shields.
roads:
  - id: main_street
    name: Main Street
    class: arterial           # See "Road classes" below
    geometry: [[200, 150], [500, 150], [800, 300]]
  - id: i77
    name: I-77
    class: interstate
    geometry: [[50, 0], [50, 700]]
    shield: {type: interstate, number: "77"}

# Districts: named regions shown as bubbles at low zoom.
districts:
  - id: downtown
    name: Downtown
    color: "#ef4444"
    label_anchor: [500, 350]  # Where the bubble and label appear
    # path is optional: adds an organic blob behind the district

# Places: the venues agents can visit.
places:
  - id: town_hall             # Must match a sim/locations.py location id
    name: Town Hall
    category: civic           # See "Place categories" below
    district: downtown        # Optional: which district this belongs to
    xy: [500, 340]            # Position in map units
    min_zoom: 0               # 0 = always visible; higher = only when zoomed in
  - id: park_bench
    name: Park Bench
    category: park
    district: downtown
    xy: [520, 360]
    min_zoom: 2
    parent: town_hall         # Optional: only shown at interior zoom (LOD 4)
  - id: old_lighthouse
    name: Old Lighthouse
    category: landmark
    xy: [900, 120]
    decor: true               # Scenery only: see "Decor places" below

# Buildings: individual footprints, drawn once you zoom past the district
# tier. Purely cartographic: nothing references them and agents never stand
# in one. Generated by _gen/buildings.py, not hand-authored.
buildings:
  - class: commercial         # residential | commercial | retail | civic | industrial
    path: "M 480,330 L 512,326 L 516,352 L 484,356 Z"

# Aliases: map runtime location IDs onto real places.
# Use when the simulation emits location names that are not in this atlas.
aliases:
  marketplace: town_square    # "marketplace" events render at town_square
  sunset_cafe: harbor_cafe    # Island date venue → nearest real place
```

#### Terrain paint order is a contract, not a preference

Terrain is painted strictly in file order, and the renderer injects the built-up
wash **immediately before the first feature whose class is in
`atlas.js:BUILT_CLASSES`**. That wash is what makes a town read as one
contiguous settlement rather than a scatter of blobs.

The consequence for generators: **emit water and parks *after* built form.**
Both `gen_brecksville.py` and `gen_kerala.py` do. A global class-based z-order
was considered and rejected — `concordia_island.yaml` opens with `ocean` as a
base layer while `kerala.yaml` opens with a full-map `grass` rect and puts
`ocean` at index 16, and no single ordering satisfies both.

#### Decor places

`decor: true` marks a place as scenery. It is drawn and labelled like any other,
but it has **no simulation identity**: there is no matching id in
`sim/locations.py` and no agent can be reported there. Use it for the landmarks
that make a place look inhabited — a jetty, a boat yard, a fish landing —
without inventing venues the simulation does not have.

#### Bridge roads

A road may carry `bridge: true`, which makes the renderer draw parapets instead
of a normal casing. In `gen_kerala.py` these are **not authored**: they are
computed from the true intersections of the street network with the canal
spines, so a bridge exists if and only if a road actually crosses a channel. Two
earlier designs that authored them by hand both drifted out of sync with the
water and left decks floating in open canal.

### Terrain classes

`ocean` · `water` · `wetland` · `sand` · `grass` · `park` · `forest` ·
`farmland` · `urban_core` · `residential` · `commercial` · `civic` ·
`industrial`

Each maps to a fill color defined in `atlas.js:TERRAIN_FILL`. The renderer draws
each polygon as an organic blob with Catmull-Rom smoothed edges.

### Road classes

`interstate` · `arterial` · `residential` · `trail` · `rail` · `ferry` · `canal`

Each has a casing color, fill color, line width, and minimum zoom threshold
defined in `atlas.js:ROAD_STYLE`.

### Place categories

Each category has a color, geometric glyph (for lower zoom), and emoji icon (for
block zoom):

Category     | Icon | Color     | Default min_zoom
------------ | ---- | --------- | ----------------
`home`       | 🏠    | `#8b7fd4` | 1
`workplace`  | 🏢    | `#3b7dd8` | 0
`grocery`    | 🛒    | `#2fa36b` | 1
`retail`     | 🛍️   | `#c9832f` | 2
`food`       | 🍽️   | `#d9683e` | 2
`cafe`       | ☕    | `#a8703c` | 2
`bar`        | 🍸    | `#b0559b` | 2
`civic`      | 🏛️   | `#6b7ba8` | 1
`education`  | 🎓    | `#4a90c4` | 1
`health`     | 🏥    | `#d1495b` | 1
`worship`    | ⛪    | `#8a7ab0` | 1
`park`       | 🌳    | `#4e9e52` | 2
`nature`     | 🌿    | `#3d7a44` | 0
`transit`    | 🚌    | `#5e6b7a` | 2
`waterfront` | ⚓    | `#2b8ab0` | 0
`landmark`   | 🏰    | `#c2872c` | 0
`plaza`      | ⛲    | `#d4663f` | 0

## Building a new atlas

### Step 1: Understand the coordinate system

Atlas coordinates are abstract **map units**, not lat/lon. Declare whatever
bounds make sense for your world. A typical range is `[0, 0, 1000, 700]`. The
renderer maps these to screen pixels via a zoom transform; all geometry uses the
same coordinate space.

### Step 2: Create the YAML file

Create `data/atlas/<your_id>.yaml`. Start with the `meta` section:

```yaml
meta:
  id: lagos
  display_name: Lagos
  tagline: West African megacity
  bounds: [0, 0, 1200, 800]
  km_across: 15
```

### Step 3: Add terrain

Work back-to-front. Start with the background fill (ocean, grass, etc.), then
layer urban areas, parks, and water features on top. Each polygon is an SVG path
string.

**Tip:** For realistic-looking organic shapes, hand-trace rough outlines with
coordinates. The renderer applies Catmull-Rom spline smoothing, so you don't
need many control points — 6–12 per polygon is usually enough.

### Step 4: Add roads

Roads are polylines with at least 2 points. Add highways first (class
`interstate`), then main roads (`arterial`), then minor streets (`residential`).
Add shields for highways:

```yaml
- id: i77
  name: I-77
  class: interstate
  geometry: [[50, 0], [55, 200], [60, 400], [50, 700]]
  shield: {type: interstate, number: "77"}
```

### Step 5: Add districts

Districts are optional but help at the region zoom level. Each district needs a
`label_anchor` coordinate where its bubble appears. Optionally add a `path` for
a colored region blob, and a `color`.

### Step 6: Add places

Every location ID emitted by `sim/locations.py` for your setting should have a
place entry. Each needs:

-   `id` — **must exactly match** the simulation's location string
-   `name` — human-readable display name
-   `category` — one of the 17 categories above
-   `xy` — `[x, y]` within the atlas bounds
-   `min_zoom` — `0` for landmarks always visible, `2` for minor venues

### Step 7: Add aliases

If the simulation emits location IDs that don't correspond to a place in your
atlas (e.g. `marketplace`, `sunset_cafe`), add aliases mapping them to the
nearest real place.

### Step 8: Register in the atlas resolver

If your geography has a `sim/locations.py` preset, add the mapping to
`atlas_loader.py:SETTING_TO_ATLAS`:

```python
SETTING_TO_ATLAS = {
    'island': 'concordia_island',
    'kerala': 'kerala',
    'ohio_suburb': 'brecksville',
    'brecksville_1000': 'brecksville',
    'lagos_500': 'lagos',           # ← add your entry
}
```

### Step 9: Validate

Run the atlas checker:

```bash
python3 data/atlas/_gen/check_atlas.py
```

Or just start the server in geography-only mode:

```bash
python -m examples.concordia_island.ui.map_dashboard --geography=lagos --port=8142
```

It will crash with a clear error if anything is malformed.

### Step 10: Rebuild

The atlas YAML is loaded from the `data/atlas/` directory at startup. After
editing an atlas:

-   **JS/CSS changes** take effect on browser refresh (served live).
-   **YAML or Python changes** require a server restart.

> [!WARNING] `atlas_loader.load_atlas()` is `@functools.lru_cache`d, and
> `/api/atlas?g=` will serve **any** geography, not just the one the server was
> started with. So a server launched with `--geography=brecksville` still
> answers `?g=kerala` — from whatever `kerala.yaml` looked like when that
> process first got asked for it. If you have several dashboards running on
> different ports, a stale one will hand you an old map with no error and no
> warning. Restart *every* server that has been asked for a geography you just
> regenerated, and check which port a screenshot actually came from before
> trusting it.

## Using atlas generators

The Brecksville and Kerala atlases are generated by scripts in
`data/atlas/_gen/`. These are deterministic — no random seeds, all jitter comes
from CRC32 — and produce the full YAML including terrain polygons, roads,
districts, places, and building footprints.

Both scripts **write the atlas in place**; do not redirect stdout.

```bash
cd data/atlas/_gen

python3 gen_brecksville.py   # → ../brecksville.yaml
python3 gen_kerala.py        # → ../kerala.yaml
python3 check_atlas.py       # validate all three atlases
```

`check_atlas.py` is the gate. It fails loudly if a place sits inside a water
polygon or too far from any road. One exemption: places whose category is
`waterfront` are allowed on water and reported as a *note* rather than a problem
— a houseboat jetty belongs on the lake.

### Shared helpers

`geom.py` (deterministic geometry):

-   `blob(cx, cy, rx, ry, seed, ...)` — organic polygon, never axis-aligned
-   `parcel(cx, cy, w, h, seed, rot, jitter)` — straight-edged quadrilateral
-   `ribbon(spine, half_width)` — a channel or road corridor as a closed polygon
-   `open_spline` / `closed_spline` — Catmull-Rom smoothing
-   `sid(...)` — CRC32 id hashing; `rng(...)` — LCG

> [!TIP] Pick `blob()` vs `parcel()` by asking who made the edge. Natural cover
> (marsh, grove, dune) is organic, so it gets `blob()`. *Reclaimed* land — paddy
> polders, orchards, allotments — is bounded by built bunds and dykes that run
> straight, so it gets `parcel()`. Drawing worked fields with `blob()` is what
> makes farmland read as a random green smudge.

`buildings.py` (cartographic footprints):

-   `street_buildings(...)` — lots fronting a polyline, for ribbon development
-   `block_buildings(...)` — a filled block, for downtown cores

Neither has any simulation meaning; see "Buildings" in the schema above.

### Kerala: the canal relaxation pass

Place coordinates are simulation identities and must not move, so when an
authored canal would run straight through the church or the bus stand, **the
channel moves, not the venue**. Hand-nudging did not converge — every shift that
freed one venue drove the channel into another.

`gen_kerala.py:canals()` therefore treats `CANALS_AUTHORED` as *intent* and runs
`_relax_spine()` over it: densify the spine, push each control point along the
local normal until nothing lies within `half_width + CANAL_CLEARANCE`, then
weakly re-smooth so it still reads as a dug canal. Endpoints are pinned.
Deterministic, cached, and it resolves every conflict automatically.

Two things about it are easy to get wrong and are load-bearing:

-   **Densification must scale with the clearance radius.** Obstacles are only
    tested at control points, so if the spacing exceeds the clearance a venue
    slips through the gap between two points that both clear it. It is
    subdivided until spacing ≤ `need / 2`. A previously fixed 2× subdivision
    broke the moment the channels were narrowed.
-   **Smoothing must stay weak** (`0.8 / 0.1 / 0.1`). At `0.5 / 0.25 / 0.25` it
    simply undid each push and never converged.

## Run directory data model

`LocalFileSource` (`map_data.py`) reads a run's `--output_dir`:

| File                         | Used for                                       |
| ---------------------------- | ---------------------------------------------- |
| `simulation_state.json`      | Current/final tick, max ticks, sim clock        |
| `performance.json`           | Run metadata (agent count, timings)             |
| `location_history.json`      | Per-tick agent positions (plus `_part2.json`    |
:                              : when split)                                    :
| `entity_memories.json` /     | Fallback: trajectories reconstructed from the   |
: `simulation_structured.json` : `// <place> [<time>]` tags on observation      :
: / `*_memories.json`          : memories when no location history was written  :
:                              : (the `data/paper_runs/` folders store one       :
:                              : `<first>_<last>_memories.json` per agent)       :

Agents with no record at a given tick are
**forward-filled** from their last known position, with a staleness age tracked
per agent per tick.

## Modifying the frontend

### No build step

The frontend is vanilla ES modules with zero dependencies. No bundler, no npm,
no CDN. Edit the JS/CSS files directly and refresh the browser.

### Module structure

-   **`atlas.js`** — Categories, terrain fills, road styles, LOD thresholds,
    View transform, color palette. Edit here to add new categories, change
    colors, or adjust zoom thresholds.
-   **`terrain.js`** — Canvas painting: terrain polygons (Catmull-Rom smoothed),
    roads (with casings and shields), building footprints. Edit here to change
    how terrain looks.
-   **`markers.js`** — SVG place markers (glyphs at low zoom, emoji at high
    zoom), district bubbles, HTML labels with collision culling. Edit here to
    change how places appear.
-   **`agents.js`** — Agent dot rendering (with tweening between positions),
    trails, flow ribbons, heatmap. Edit here to change how agents look.
-   **`app.js`** — Glues everything together: boot, data fetch loop, timeline
    controls, roster, layer toggles, panels. The main state object is `S`.

### Adding a new place category

1.  Add to `PLACE_CATEGORIES` in `atlas_loader.py`
2.  Add to `CATEGORIES` in `atlas.js` with `label`, `color`, `minZoom`, `icon`
    (emoji), and `glyph` (SVG path in a 24×24 box)
3.  Use the new category in your atlas YAML

### Adjusting LOD thresholds

Edit `lodFor()` in `atlas.js`. The thresholds must stay in sync with
`zoomThresholdFor()` in `markers.js`.

### Syntax checking

The ES modules can be syntax-checked with any JavaScript parser (for example
`node --check static/app.js`, or esprima from Python).
