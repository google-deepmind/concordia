/* disableFinding(Lint) */
/**
 * @fileoverview Core map model: category metadata and the view transform.
 * @suppress {lintChecks}
 *
 * Coordinate spaces
 * -----------------
 *   map units  -- what the atlas YAML is authored in (meta.bounds)
 *   screen px  -- what we draw
 *
 * A single View owns the map->screen transform. Every layer (terrain canvas,
 * vector SVG, agent canvas, HTML labels) consumes the same View, which is what
 * keeps them in registration. Zoom is a real transform, not a CSS scale on a
 * DOM tree, so text and stroke widths stay under our control at every level.
 */

/* ------------------------------------------------------------------------ *
 * Place categories
 *
 * `glyph` is SVG path data in a 24x24 box, drawn centred inside the marker.
 * Deliberately geometric and monochrome so the family stays consistent --
 * emoji render differently on every platform and cannot be tinted.
 * ------------------------------------------------------------------------ */
export const CATEGORIES = {
  home:       {label: 'Home',       color: '#8b7fd4', minZoom: 1,
               icon: '\u{1F3E0}',
               glyph: 'M4 11 L12 4 L20 11 L20 20 L14 20 L14 14 L10 14 L10 20 L4 20 Z'},
  workplace:  {label: 'Workplace',  color: '#3b7dd8', minZoom: 0,
               icon: '\u{1F3E2}',
               glyph: 'M5 8 L19 8 L19 20 L5 20 Z M9 4 L15 4 L15 8 L9 8 Z M8 11 H11 V14 H8 Z M13 11 H16 V14 H13 Z'},
  grocery:    {label: 'Grocery',    color: '#2fa36b', minZoom: 1,
               icon: '\u{1F6D2}',
               glyph: 'M4 7 H8 L10 16 H18 L20 9 H9 M10 20 a1.4 1.4 0 1 0 0.01 0 M17 20 a1.4 1.4 0 1 0 0.01 0'},
  retail:     {label: 'Retail',     color: '#c9832f', minZoom: 2,
               icon: '\u{1F6CD}\uFE0F',
               glyph: 'M5 9 L7 5 L17 5 L19 9 L19 20 L5 20 Z M9 12 H15 V20 H9 Z'},
  food:       {label: 'Food',       color: '#d9683e', minZoom: 2,
               icon: '\u{1F37D}\uFE0F',
               glyph: 'M8 4 V12 M6 4 V9 a2 2 0 0 0 4 0 V4 M16 4 c-2 2 -2 6 0 8 V20 M8 12 V20'},
  cafe:       {label: 'Cafe',       color: '#a8703c', minZoom: 2,
               icon: '\u2615',
               glyph: 'M5 8 H17 V14 a4 4 0 0 1 -4 4 H9 a4 4 0 0 1 -4 -4 Z M17 9 h2 a2 2 0 0 1 0 4 h-2 M6 4 v2 M10 4 v2 M14 4 v2'},
  bar:        {label: 'Bar',        color: '#b0559b', minZoom: 2,
               icon: '\u{1F378}',
               glyph: 'M5 5 H19 L12 13 Z M12 13 V20 M8 20 H16'},
  civic:      {label: 'Civic',      color: '#6b7ba8', minZoom: 1,
               icon: '\u{1F3DB}\uFE0F',
               glyph: 'M12 4 L21 9 H3 Z M5 11 V18 M9.5 11 V18 M14.5 11 V18 M19 11 V18 M3 20 H21'},
  education:  {label: 'Education',  color: '#4a90c4', minZoom: 1,
               icon: '\u{1F393}',
               glyph: 'M12 5 L22 10 L12 15 L2 10 Z M6 12 V17 c0 2 12 2 12 0 V12'},
  health:     {label: 'Health',     color: '#d1495b', minZoom: 1,
               icon: '\u{1F3E5}',
               glyph: 'M10 4 H14 V10 H20 V14 H14 V20 H10 V14 H4 V10 H10 Z'},
  worship:    {label: 'Worship',    color: '#8a7ab0', minZoom: 1,
               icon: '\u26EA',
               glyph: 'M10 3 H14 V7 H18 V11 H14 V21 H10 V11 H6 V7 H10 Z'},
  park:       {label: 'Park',       color: '#4e9e52', minZoom: 2,
               icon: '\u{1F333}',
               glyph: 'M12 3 L18 12 H14 L18 18 H6 L10 12 H6 Z M12 18 V21'},
  nature:     {label: 'Nature',     color: '#3d7a44', minZoom: 0,
               icon: '\u{1F33F}',
               glyph: 'M7 14 a5 5 0 1 1 10 0 a4 4 0 0 1 -10 0 M12 14 V21 M12 17 L9 15 M12 18 L15 16'},
  transit:    {label: 'Transit',    color: '#5e6b7a', minZoom: 2,
               icon: '\u{1F68C}',
               glyph: 'M6 5 H18 V15 H6 Z M6 15 V19 H9 V15 M15 15 V19 H18 V15 M8 8 H16 V12 H8 Z'},
  waterfront: {label: 'Waterfront', color: '#2b8ab0', minZoom: 0,
               icon: '\u2693',
               glyph: 'M12 3 a2 2 0 1 0 0.01 0 M12 6 V20 M7 10 H17 M4 15 c2 4 14 4 16 0'},
  landmark:   {label: 'Landmark',   color: '#c2872c', minZoom: 0,
               icon: '\u{1F3F0}',
               glyph: 'M12 3 L14 9 H10 Z M10 9 H14 L15 20 H9 Z M7 20 H17'},
  plaza:      {label: 'Town Square', color: '#d4663f', minZoom: 0,
               icon: '\u26F2',
               glyph: 'M12 3 L21 12 L12 21 L3 12 Z M12 8 L16 12 L12 16 L8 12 Z'},
};

export const DEFAULT_CATEGORY = {label: 'Place', color: '#7c8698', minZoom: 2,
                                 icon: '\u{1F4CD}',
                                 glyph: 'M12 4 a8 8 0 1 0 0.01 0'};

export function categoryOf(place) {
  return CATEGORIES[place.category] || DEFAULT_CATEGORY;
}

/* ------------------------------------------------------------------------ *
 * Building footprints per category
 *
 * A map where every venue is the same grey square tells you nothing at block
 * zoom. These give each category a plausible plan: houses are small and
 * gabled, offices and schools are large rectangular blocks, shopfronts are
 * wide and shallow against the street, churches get a nave and transept.
 *
 *   w, h  extent in map units
 *   kind  which plan generator to use (see terrain.js:footprintPath)
 *   cls   building palette entry
 * ------------------------------------------------------------------------ */
export const FOOTPRINT = {
  home:       {w: 13, h: 11, kind: 'house',  cls: 'residential'},
  workplace:  {w: 30, h: 22, kind: 'tower',  cls: 'commercial'},
  grocery:    {w: 26, h: 18, kind: 'shed',   cls: 'retail'},
  retail:     {w: 17, h: 11, kind: 'shop',   cls: 'retail'},
  food:       {w: 14, h: 11, kind: 'shop',   cls: 'retail'},
  cafe:       {w: 12, h: 10, kind: 'shop',   cls: 'retail'},
  bar:        {w: 13, h: 10, kind: 'shop',   cls: 'retail'},
  civic:      {w: 24, h: 18, kind: 'civic',  cls: 'civic'},
  education:  {w: 34, h: 20, kind: 'wing',   cls: 'civic'},
  health:     {w: 24, h: 18, kind: 'wing',   cls: 'civic'},
  worship:    {w: 18, h: 14, kind: 'church', cls: 'civic'},
  transit:    {w: 12, h: 8,  kind: 'shed',   cls: 'industrial'},
  landmark:   {w: 18, h: 16, kind: 'civic',  cls: 'civic'},
  waterfront: {w: 16, h: 10, kind: 'shed',   cls: 'industrial'},
  // park / nature / plaza deliberately absent: open space has no building.
};

/** Categories that get the heavier "this is a destination" marker. */
export const KEY_VENUES = new Set(
    ['cafe', 'bar', 'food', 'workplace', 'home', 'grocery']);

export function footprintOf(place) {
  if (place.footprint === false) return null;
  return FOOTPRINT[place.category] || null;
}

/* Terrain fill colours, keyed to the CSS custom properties in map.css so the
 * canvas and the DOM never disagree about what "grass" means.
 *
 * The palette is sampled from a Google Maps regional view rather than picked
 * by eye: open country is a desaturated mint (#d3f8e2 in the reference),
 * built-up land is a near-white warm tan (#f7f7f7 there, warmed slightly here
 * so the town reads as *settlement* against the green), and water is a light
 * cyan. The single most important property is that built and unbuilt land
 * differ in hue, not just in lightness -- that is what makes a town's shape
 * legible at a glance on a real map. */
export const TERRAIN_FILL = {
  ocean: '#8ecfe8',
  water: '#9fdbef',
  /* Wetland, grass and farmland are the three greens that have to survive
   * being next to each other -- paddy country is literally all three at once.
   * They are separated by *hue* (damp teal / neutral mint / crop yellow-green)
   * rather than by lightness, because a lightness-only ramp collapses under
   * the simulation-clock tint and reads as one flat smudge. */
  wetland: '#c2e3dc',
  sand: '#f2e7c9',
  grass: '#d6f1de',
  park: '#c4edd3',
  forest: '#b4e6c6',
  farmland: '#e6efc6',
  urban_core: '#eae3d6',
  residential: '#f4f0e7',
  commercial: '#f0e9db',
  civic: '#eee9e1',
  industrial: '#e6e0d4',
};

/* Classes that count as "built up". Painted as one contiguous tan mass with
 * the road-corridor wash, so a town reads as a single settlement rather than
 * as a scatter of disconnected blobs. */
export const BUILT_CLASSES = new Set(
    ['urban_core', 'residential', 'commercial', 'civic', 'industrial']);

/* Individual building footprints, drawn from the atlas `buildings` block once
 * zoomed past the district tier.
 *
 * These must be clearly darker than the built-up wash they sit on. An earlier
 * pass matched Google's very subtle building/background delta, but Google's
 * background is near-white (#f7f7f7) whereas ours is a much warmer tan, so the
 * same delta made the footprints disappear entirely. These values keep the warm
 * hue but drop enough luminance that a block of buildings is legible at a
 * glance, with a distinctly darker edge to hold the shape. */
export const BUILDING_FILL = {
  generic:     {fill: '#ddd5c6', stroke: '#bfb4a0'},
  residential: {fill: '#e0d9cb', stroke: '#c3b8a4'},
  commercial:  {fill: '#d6cdba', stroke: '#b9ad96'},
  retail:      {fill: '#dbd1bc', stroke: '#bfb29a'},
  civic:       {fill: '#d9d3c8', stroke: '#bdb5a6'},
  industrial:  {fill: '#d1c9ba', stroke: '#b4ab99'},
};

export function buildingFill(cls) {
  return BUILDING_FILL[cls] || BUILDING_FILL.generic;
}

/* Road styling. `w` values are widths in *map units* so roads thicken
 * naturally as you zoom, clamped in the painter to stay legible.
 *
 * Arterials are the blue-grey of the reference rather than white: against a
 * tan town and a green hinterland, white arterials disappear into the
 * built-up wash, which is why the previous palette made Brecksville read as
 * an undifferentiated beige smear. Residential streets stay white, so the
 * hierarchy is carried by hue at every zoom. */
export const ROAD_STYLE = {
  interstate:  {casing: '#e3b055', fill: '#f8cd77', w: 9,   minZoom: 0},
  arterial:    {casing: '#9fb6d3', fill: '#bdcee4', w: 6.5, minZoom: 0},
  residential: {casing: '#dde2e8', fill: '#ffffff', w: 3.4, minZoom: 1.4},
  trail:       {casing: null,      fill: '#b6a884', w: 1.6, minZoom: 2.0, dash: [5, 4]},
  rail:        {casing: '#9ca3af', fill: '#ffffff', w: 2.4, minZoom: 1.6, dash: [7, 7]},
  ferry:       {casing: null,      fill: '#79b6cf', w: 1.8, minZoom: 1.0, dash: [3, 5]},
  // The terrain layer already paints the channel itself. This entry is only
  // the *route* down it -- what carries the canal's name and what a boat
  // follows. It therefore gets no casing: with one, the darker edge read as a
  // road casing and turned Alappuzha's canals into a grid of cyan highways at
  // region zoom. Held back to street zoom for the same reason.
  canal:       {casing: null,      fill: '#8ecfe8', w: 2.0, minZoom: 2.2},
};

/* Road classes that a settlement grows along. Used by the built-up wash. */
export const URBAN_ROAD_CLASSES = new Set(
    ['interstate', 'arterial', 'residential']);

/* ------------------------------------------------------------------------ *
 * Level of detail
 *
 * Tiers answer different questions rather than showing the same thing bigger:
 *   0 region   what is this place
 *   1 district where is everyone
 *   2 street   what is around here
 *   3 block    who is here
 *   4 interior what is happening inside
 * ------------------------------------------------------------------------ */
export const LOD_NAMES = ['region', 'district', 'street', 'block', 'interior'];

export function lodFor(zoom) {
  if (zoom < 1.15) return 0;
  if (zoom < 2.0) return 1;
  if (zoom < 3.4) return 2;
  if (zoom < 6.0) return 3;
  return 4;
}

/* ------------------------------------------------------------------------ *
 * View transform
 * ------------------------------------------------------------------------ */

const MIN_K = 0.45;
const MAX_K = 14;

export class View {
  constructor() {
    this.k = 1;       // scale: screen px per map unit, relative to fitted base
    this.tx = 0;      // translate, screen px
    this.ty = 0;
    this.base = 1;    // px per map unit at zoom 1 (set by fit())
    this.w = 0;
    this.h = 0;
    this.bounds = [0, 0, 1000, 700];
    this._anim = null;
  }

  resize(w, h) {
    const hadFit = this.w > 0;
    const cx = hadFit ? (this.w / 2 - this.tx) / this.scale() : 0;
    const cy = hadFit ? (this.h / 2 - this.ty) / this.scale() : 0;
    this.w = w;
    this.h = h;
    this._computeBase();
    if (hadFit) this.centerOn(cx, cy);
    else this.fit();
  }

  setBounds(bounds) {
    this.bounds = bounds.slice();
    this._computeBase();
  }

  _computeBase() {
    const [x0, y0, x1, y1] = this.bounds;
    if (!this.w || !this.h) return;
    // Cover, not contain: the map should bleed to the edges rather than sit in
    // letterbox bars, which looks unfinished for a full-bleed canvas.
    this.base = Math.max(this.w / (x1 - x0), this.h / (y1 - y0));
  }

  scale() { return this.base * this.k; }

  /** Map units -> screen px. */
  toScreen(x, y) {
    const s = this.scale();
    return [x * s + this.tx, y * s + this.ty];
  }

  /** Screen px -> map units. */
  toMap(px, py) {
    const s = this.scale();
    return [(px - this.tx) / s, (py - this.ty) / s];
  }

  centerOn(mx, my) {
    const s = this.scale();
    this.tx = this.w / 2 - mx * s;
    this.ty = this.h / 2 - my * s;
    this.clamp();
  }

  fit() {
    this.k = 1;
    const [x0, y0, x1, y1] = this.bounds;
    this.centerOn((x0 + x1) / 2, (y0 + y1) / 2);
  }

  /** Keeps the map from being panned off into empty space. */
  clamp() {
    const s = this.scale();
    const [x0, y0, x1, y1] = this.bounds;
    const wMap = (x1 - x0) * s;
    const hMap = (y1 - y0) * s;
    const minTx = this.w - x1 * s;
    const maxTx = -x0 * s;
    const minTy = this.h - y1 * s;
    const maxTy = -y0 * s;
    this.tx = wMap <= this.w ? (this.w - wMap) / 2 - x0 * s
                             : Math.min(maxTx, Math.max(minTx, this.tx));
    this.ty = hMap <= this.h ? (this.h - hMap) / 2 - y0 * s
                             : Math.min(maxTy, Math.max(minTy, this.ty));
  }

  /** Zooms by a factor while holding the given screen point fixed. */
  zoomAt(px, py, factor) {
    const before = this.toMap(px, py);
    this.k = Math.min(MAX_K, Math.max(MIN_K, this.k * factor));
    const after = this.toMap(px, py);
    const s = this.scale();
    this.tx += (after[0] - before[0]) * s;
    this.ty += (after[1] - before[1]) * s;
    this.clamp();
  }

  panBy(dx, dy) {
    this.tx += dx;
    this.ty += dy;
    this.clamp();
  }

  /** Animated move to a target centre and zoom. */
  flyTo(mx, my, k, ms, onFrame, onDone) {
    if (this._anim) cancelAnimationFrame(this._anim);
    const s0 = {k: this.k, tx: this.tx, ty: this.ty};
    const cur = {k: this.k, tx: this.tx, ty: this.ty};
    this.k = Math.min(MAX_K, Math.max(MIN_K, k));
    this.centerOn(mx, my);
    const s1 = {k: this.k, tx: this.tx, ty: this.ty};
    Object.assign(this, cur);

    const reduce = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce || ms <= 0) {
      Object.assign(this, s1);
      this.clamp();
      onFrame && onFrame();
      onDone && onDone();
      return;
    }

    const t0 = performance.now();
    const step = (now) => {
      const u = Math.min(1, (now - t0) / ms);
      const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      this.k = s0.k + (s1.k - s0.k) * e;
      this.tx = s0.tx + (s1.tx - s0.tx) * e;
      this.ty = s0.ty + (s1.ty - s0.ty) * e;
      onFrame && onFrame();
      if (u < 1) {
        this._anim = requestAnimationFrame(step);
      } else {
        this._anim = null;
        this.clamp();
        onDone && onDone();
      }
    };
    this._anim = requestAnimationFrame(step);
  }

  /** Frames an arbitrary map-unit rect with padding. */
  flyToRect(x0, y0, x1, y1, pad, ms, onFrame, onDone) {
    const w = Math.max(1, x1 - x0) + pad * 2;
    const h = Math.max(1, y1 - y0) + pad * 2;
    const k = Math.min(this.w / (w * this.base), this.h / (h * this.base));
    this.flyTo((x0 + x1) / 2, (y0 + y1) / 2, k, ms, onFrame, onDone);
  }
}

/* ------------------------------------------------------------------------ *
 * Geometry helpers
 * ------------------------------------------------------------------------ */

/** Bounding box of an SVG path's coordinate pairs, or null. */
export function pathBBox(d) {
  const nums = d.match(/-?\d+(\.\d+)?/g);
  if (!nums || nums.length < 4) return null;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i + 1 < nums.length; i += 2) {
    const x = +nums[i], y = +nums[i + 1];
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  return [x0, y0, x1, y1];
}

/** Stable pseudo-random in [0,1) from a string — used for deterministic
 *  per-agent jitter so agents at one place do not stack into a single dot. */
export function hash01(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000;
}

/** Distinct, colourblind-conscious agent palette. */
export const AGENT_COLORS = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4',
  '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9a6324',
  '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
  '#ff6d01', '#1f78b4', '#b15928', '#6a3d9a',
];

export function agentColor(name) {
  return AGENT_COLORS[Math.floor(hash01(name) * AGENT_COLORS.length)];
}

export function initials(name) {
  const parts = String(name).trim().split(/\s+/);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}
