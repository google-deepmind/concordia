/* disableFinding(Lint) */
/**
 * @fileoverview Terrain and road painter.
 * @suppress {lintChecks}
 *
 * Everything here draws onto a single canvas in *map unit* space: we set the
 * canvas transform once from the View and then fill cached Path2D objects
 * built straight from the atlas path strings. That keeps per-frame work to
 * pure rasterisation and means zooming never re-parses geometry.
 *
 * Roads are drawn in two passes -- all casings, then all fills -- which is the
 * standard cartographic trick that makes intersections read correctly instead
 * of showing seams where one road crosses another.
 */

import {TERRAIN_FILL, ROAD_STYLE, BUILT_CLASSES, URBAN_ROAD_CLASSES,
        buildingFill, footprintOf, pathBBox, hash01} from './atlas.js';

const _pathCache = new WeakMap();

function cachedPaths(atlas) {
  let c = _pathCache.get(atlas);
  if (c) return c;
  const terrain = atlas.terrain.map((f) => ({
    feat: f,
    path: new Path2D(f.path),
    bbox: pathBBox(f.path),
    built: BUILT_CLASSES.has(f.class),
  }));
  c = {
    terrain,
    // Where the built-up wash is injected into the paint order: immediately
    // before the first authored built-form polygon, so natural cover sits
    // under the town and water/parks (which generators emit last) sit over it.
    washAt: Math.max(0, terrain.findIndex((t) => t.built)),
    hasBuilt: terrain.some((t) => t.built),
    buildings: (atlas.buildings || []).map((b) => ({
      feat: b,
      path: new Path2D(b.path),
    })),
    wash: buildUrbanWash(atlas),
    footprints: buildFootprints(atlas),
    halo: null,
  };
  const haloFeat = atlas.terrain.find(
      (f) => f.halo || f.id === 'shoreline' || f.id === 'land');
  if (haloFeat) c.halo = new Path2D(haloFeat.path);
  _pathCache.set(atlas, c);
  return c;
}

/* ------------------------------------------------------------------------ *
 * Built-up wash
 *
 * On a real map a town is not a scatter of coloured blobs: it is one
 * continuous pale mass that grows along its roads and thins out into the
 * countryside. That silhouette -- the star of built-up land radiating from a
 * centre down each highway -- is most of what makes a Google Maps regional
 * view legible, and it is the single biggest thing the previous palette was
 * missing.
 *
 * We synthesise it by walking the road network and stamping a disc at each
 * sample, with a radius that tapers to nothing as you get further from the
 * nearest district centre. Roads far from any neighbourhood therefore cross
 * open green, exactly as they should, while roads through town fatten into
 * the settlement. The whole thing is one cached Path2D filled once per frame.
 * ------------------------------------------------------------------------ */

/** Max corridor half-width, in map units, per road class. */
const WASH_RADIUS = {
  interstate: 15,
  arterial: 27,
  residential: 21,
};

// Full width within NEAR units of a district centre, nothing beyond FAR.
const WASH_NEAR = 70;
const WASH_FAR = 190;

function buildUrbanWash(atlas) {
  const anchors = atlas.districts
      .map((d) => d.label_anchor)
      .filter((a) => Array.isArray(a) && a.length === 2);
  if (!anchors.length) return null;

  const path = new Path2D();
  let stamped = false;

  const falloff = (x, y) => {
    let best = Infinity;
    for (const a of anchors) {
      const d = Math.hypot(a[0] - x, a[1] - y);
      if (d < best) best = d;
    }
    if (best >= WASH_FAR) return 0;
    if (best <= WASH_NEAR) return 1;
    const u = (best - WASH_NEAR) / (WASH_FAR - WASH_NEAR);
    // Smoothstep, so the edge of town is a soft taper rather than a step.
    return 1 - u * u * (3 - 2 * u);
  };

  const STEP = 7;   // map units between stamps; small enough to read as solid
  for (const road of atlas.roads) {
    const maxR = WASH_RADIUS[road.class];
    if (!maxR || !URBAN_ROAD_CLASSES.has(road.class)) continue;
    const g = road.geometry;
    for (let i = 0; i + 1 < g.length; i++) {
      const [x0, y0] = g[i];
      const [x1, y1] = g[i + 1];
      const len = Math.hypot(x1 - x0, y1 - y0);
      const n = Math.max(1, Math.round(len / STEP));
      for (let j = 0; j <= n; j++) {
        const t = j / n;
        const x = x0 + (x1 - x0) * t;
        const y = y0 + (y1 - y0) * t;
        const f = falloff(x, y);
        if (f <= 0.02) continue;
        path.moveTo(x + maxR * f, y);
        path.arc(x, y, maxR * f, 0, 6.2832);
        stamped = true;
      }
    }
  }
  return stamped ? path : null;
}

/* ------------------------------------------------------------------------ *
 * Place footprints
 *
 * Each place gets a plan appropriate to what it is. Built once per atlas in
 * map-unit space and cached, because the geometry never changes -- only the
 * view transform does.
 * ------------------------------------------------------------------------ */

function buildFootprints(atlas) {
  const out = [];
  for (const p of atlas.places) {
    if (p.parent || p.virtual) continue;
    const fp = footprintOf(p);
    if (!fp) continue;
    // Deterministic per-place rotation and size wobble. Without it a street of
    // shops is a row of identical rectangles, which no real block ever is.
    const h = hash01(p.id);
    const rot = (h - 0.5) * 0.34;                     // +/- ~10 degrees
    const scale = 0.86 + hash01(p.id + '#s') * 0.3;
    const w = (p.footprint_size || fp.w) * scale;
    const d = (p.footprint_size ? p.footprint_size * fp.h / fp.w : fp.h) * scale;
    out.push({
      place: p,
      cls: fp.cls,
      minZoom: fp.kind === 'house' ? 3.0 : 2.0,
      path: footprintPath(fp.kind, p.xy[0], p.xy[1], w, d, rot),
    });
  }
  return out;
}

/** Rotates (dx, dy) about a centre and returns absolute map coordinates. */
function rp(cx, cy, dx, dy, cos, sin) {
  return [cx + dx * cos - dy * sin, cy + dx * sin + dy * cos];
}

/**
 * Builds one building plan.
 *
 * @param {string} kind house|shop|shed|tower|wing|civic|church
 * @param {number} cx centre x, map units
 * @param {number} cy centre y, map units
 * @param {number} w  width, map units
 * @param {number} h  depth, map units
 * @param {number} rot rotation in radians
 * @return {!Path2D}
 */
function footprintPath(kind, cx, cy, w, h, rot) {
  const p = new Path2D();
  const cos = Math.cos(rot);
  const sin = Math.sin(rot);
  const W = w / 2;
  const H = h / 2;
  const poly = (pts) => {
    let first = true;
    for (const [dx, dy] of pts) {
      const [x, y] = rp(cx, cy, dx, dy, cos, sin);
      if (first) { p.moveTo(x, y); first = false; } else { p.lineTo(x, y); }
    }
    p.closePath();
  };

  switch (kind) {
    case 'house':
      // Simple gabled plan: a rectangle with a clipped porch corner.
      poly([[-W, -H], [W, -H], [W, H], [-W * 0.35, H], [-W * 0.35, H * 0.35],
            [-W, H * 0.35]]);
      break;
    case 'shop':
      // Wide, shallow, presenting its frontage to the street.
      poly([[-W, -H * 0.7], [W, -H * 0.7], [W, H], [-W, H]]);
      break;
    case 'shed':
      poly([[-W, -H], [W, -H], [W, H], [-W, H]]);
      break;
    case 'tower':
      // Office block with a setback wing, so it does not read as a bare box.
      poly([[-W, -H], [W * 0.3, -H], [W * 0.3, -H * 0.2], [W, -H * 0.2],
            [W, H], [-W, H]]);
      break;
    case 'wing':
      // School / hospital: a long spine with a perpendicular wing.
      poly([[-W, -H * 0.45], [W, -H * 0.45], [W, H * 0.45],
            [W * 0.25, H * 0.45], [W * 0.25, H], [-W * 0.3, H],
            [-W * 0.3, H * 0.45], [-W, H * 0.45]]);
      break;
    case 'church':
      // Nave plus transept.
      poly([[-W * 0.32, -H], [W * 0.32, -H], [W * 0.32, -H * 0.15],
            [W, -H * 0.15], [W, H * 0.3], [W * 0.32, H * 0.3],
            [W * 0.32, H], [-W * 0.32, H], [-W * 0.32, H * 0.3],
            [-W, H * 0.3], [-W, -H * 0.15], [-W * 0.32, -H * 0.15]]);
      break;
    case 'civic':
    default:
      // Squared-off block with a recessed entrance bay.
      poly([[-W, -H], [W, -H], [W, H], [W * 0.28, H], [W * 0.28, H * 0.55],
            [-W * 0.28, H * 0.55], [-W * 0.28, H], [-W, H]]);
      break;
  }
  return p;
}

/** Applies the view transform so subsequent draws are in map units. */
function applyView(ctx, view) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const s = view.scale();
  ctx.setTransform(s * dpr, 0, 0, s * dpr, view.tx * dpr, view.ty * dpr);
  return s;
}

/* ------------------------------------------------------------------------ *
 * Terrain
 * ------------------------------------------------------------------------ */

export function paintTerrain(ctx, atlas, view, opts) {
  const o = opts || {};
  const cache = cachedPaths(atlas);
  applyView(ctx, view);

  // Base wash so any gap in the authored geometry reads as water rather than
  // as a hole in the map.
  const [x0, y0, x1, y1] = atlas.meta.bounds;
  ctx.fillStyle = TERRAIN_FILL.ocean;
  ctx.fillRect(x0 - 2000, y0 - 2000, (x1 - x0) + 4000, (y1 - y0) + 4000);

  // Soft depth halos hugging the coast. Cheap, and it is most of what makes a
  // coastline look like a real map rather than a filled blob.
  if (cache.halo && o.halos !== false) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.20)';
    ctx.lineJoin = 'round';
    for (let i = 4; i >= 1; i--) {
      ctx.lineWidth = (i * 9) / 1;
      ctx.globalAlpha = 0.16;
      ctx.stroke(cache.halo);
    }
    ctx.restore();
  }

  // Terrain in authored order, with the built-up wash injected immediately
  // before the first built-form polygon. Generators emit water and parks
  // after built form, so canals and greens still read over the town.
  for (let i = 0; i < cache.terrain.length; i++) {
    if (i === cache.washAt && cache.hasBuilt && cache.wash &&
        o.urbanWash !== false) {
      ctx.fillStyle = TERRAIN_FILL.residential;
      ctx.fill(cache.wash);
    }
    const item = cache.terrain[i];
    const fill = TERRAIN_FILL[item.feat.class];
    if (!fill) continue;
    ctx.fillStyle = fill;
    ctx.fill(item.path);
  }

  // Forest stipple: a scatter of small darker dots, clipped to forest polys.
  // Only worth the cost once zoomed in enough to perceive it.
  if (view.k >= 1.4 && o.texture !== false) {
    ctx.save();
    ctx.fillStyle = 'rgba(82,146,106,0.16)';
    for (const item of cache.terrain) {
      if (item.feat.class !== 'forest' || !item.bbox) continue;
      ctx.save();
      ctx.clip(item.path);
      const [bx0, by0, bx1, by1] = item.bbox;
      const step = 13;
      for (let y = by0; y < by1; y += step) {
        for (let x = bx0; x < bx1; x += step) {
          // Deterministic offset so the texture does not crawl when panning.
          const jx = ((x * 37 + y * 17) % 11) - 5;
          const jy = ((x * 23 + y * 41) % 11) - 5;
          ctx.beginPath();
          ctx.arc(x + jx, y + jy, 1.5, 0, 6.2832);
          ctx.fill();
        }
      }
      ctx.restore();
    }
    ctx.restore();
  }

  // Neighbourhood extent is carried entirely by the cartography -- built-up
  // tan against open green -- and labelled by the name plus the occupancy
  // chip. Nothing is stroked around a district.
  //
  // This went through two earlier rounds: a 14%-alpha colour wash, then a
  // dashed hairline in the district colour. Both turned every neighbourhood
  // into a coloured bubble, which is precisely what a real map does not do.
  // Google Maps draws no boundary for a neighbourhood at all; you infer it
  // from where the buildings stop. Resist adding a third variant.


  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

/* ------------------------------------------------------------------------ *
 * Roads
 * ------------------------------------------------------------------------ */

/**
 * Traces a route through `pts` as a centripetal-ish Catmull-Rom spline.
 *
 * Atlas route geometry is hand-traced at a handful of control points. Joining
 * them with straight segments makes every road read as a polygon chain, which
 * is the clearest possible tell that a map is synthetic. Interpolating instead
 * costs one pass and makes sparse input look surveyed.
 *
 * Two consecutive identical points, or a two-point route, degrade gracefully to
 * a straight line, which is what you want for an interstate segment.
 */
function tracePolyline(ctx, pts) {
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  if (pts.length === 2) {
    ctx.lineTo(pts[1][0], pts[1][1]);
    return;
  }
  const n = pts.length;
  for (let i = 0; i < n - 1; i++) {
    const p0 = pts[i > 0 ? i - 1 : 0];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2 < n ? i + 2 : n - 1];
    // Standard uniform Catmull-Rom to cubic Bezier control points. The 1/6
    // factor is what keeps the curve passing exactly through p1 and p2.
    ctx.bezierCurveTo(
        p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6,
        p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6,
        p2[0], p2[1]);
  }
}

export function paintRoads(ctx, atlas, view) {
  const s = applyView(ctx, view);
  const visible = atlas.roads.filter((r) => {
    const st = ROAD_STYLE[r.class];
    return st && view.k >= (st.minZoom || 0);
  });
  if (!visible.length) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return;
  }

  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // Widths are authored in map units but clamped in screen px, so a residential
  // street never becomes a hairline at low zoom nor a runway at high zoom.
  const widthFor = (st, casing) => {
    const wMap = st.w * (casing ? 1.5 : 1);
    const px = Math.max(1, Math.min(wMap * s, casing ? 26 : 18));
    return px / s;
  };

  // Pass 1: casings.
  for (const r of visible) {
    const st = ROAD_STYLE[r.class];
    // A bridge gets a dark parapet instead of the usual soft casing, and
    // butt caps so the deck ends squarely at the bank rather than bulging
    // out over the water.
    if (r.bridge) {
      ctx.lineCap = 'butt';
      ctx.strokeStyle = '#8a8f96';
      ctx.lineWidth = widthFor(st, true) * 1.15;
      ctx.setLineDash([]);
      tracePolyline(ctx, r.geometry);
      ctx.stroke();
      ctx.lineCap = 'round';
      continue;
    }
    if (!st.casing) continue;
    ctx.strokeStyle = st.casing;
    ctx.lineWidth = widthFor(st, true);
    ctx.setLineDash([]);
    tracePolyline(ctx, r.geometry);
    ctx.stroke();
  }

  // Pass 2: fills.
  for (const r of visible) {
    const st = ROAD_STYLE[r.class];
    ctx.lineCap = r.bridge ? 'butt' : 'round';
    ctx.strokeStyle = st.fill;
    ctx.lineWidth = widthFor(st, false);
    ctx.setLineDash(st.dash ? st.dash.map((d) => d / s) : []);
    tracePolyline(ctx, r.geometry);
    ctx.stroke();
  }
  ctx.lineCap = 'round';

  // Pass 3: lane divider on the interstate, once it is wide enough to see.
  if (view.k >= 1.8) {
    for (const r of visible) {
      if (r.class !== 'interstate') continue;
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 1 / s;
      ctx.setLineDash([8 / s, 8 / s]);
      tracePolyline(ctx, r.geometry);
      ctx.stroke();
    }
  }

  ctx.setLineDash([]);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

/* ------------------------------------------------------------------------ *
 * Route shields
 *
 * Drawn in screen space so they stay a constant readable size, positioned at
 * the midpoint of the road they belong to.
 * ------------------------------------------------------------------------ */

export function paintShields(ctx, atlas, view) {
  for (const r of atlas.roads) {
    if (!r.shield) continue;
    const g = r.geometry;
    // Anchor at whichever vertex is nearest the centre of the viewport, so the
    // shield stays on screen while panning along a long route.
    let best = null;
    let bestD = Infinity;
    for (const p of g) {
      const [sx, sy] = view.toScreen(p[0], p[1]);
      if (sx < 40 || sy < 40 || sx > view.w - 40 || sy > view.h - 40) continue;
      const d = Math.hypot(sx - view.w / 2, sy - view.h / 2);
      if (d < bestD) { bestD = d; best = [sx, sy]; }
    }
    if (!best) continue;
    drawShield(ctx, best[0], best[1], r.shield);
  }
}

function drawShield(ctx, x, y, shield) {
  const num = String(shield.number || '');
  ctx.save();
  ctx.font = '700 12px Inter, system-ui, sans-serif';
  const w = Math.max(26, ctx.measureText(num).width + 14);
  const h = 26;
  const rx = x - w / 2;
  const ry = y - h / 2;

  ctx.shadowColor = 'rgba(0,0,0,0.25)';
  ctx.shadowBlur = 3;
  ctx.shadowOffsetY = 1;

  if (shield.type === 'interstate') {
    ctx.fillStyle = '#3b6db5';
    roundRect(ctx, rx, ry, w, h, 5);
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    roundRect(ctx, rx + 1.5, ry + 1.5, w - 3, h - 3, 4);
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.font = '600 6.5px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('INTERSTATE', x, ry + 9);
    ctx.font = '700 13px Inter, system-ui, sans-serif';
    ctx.fillText(num, x, ry + 21);
  } else {
    ctx.fillStyle = '#fff';
    roundRect(ctx, rx, ry, w, h, 4);
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 2;
    roundRect(ctx, rx + 1, ry + 1, w - 2, h - 2, 3);
    ctx.stroke();
    ctx.fillStyle = '#6b7280';
    ctx.font = '600 6px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(String(shield.state || '').toUpperCase(), x, ry + 8.5);
    ctx.fillStyle = '#111827';
    ctx.font = '700 13px Inter, system-ui, sans-serif';
    ctx.fillText(num, x, ry + 21);
  }
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/* ------------------------------------------------------------------------ *
 * Buildings
 *
 * Two layers, both in map units:
 *
 *   paintBuildings   the atlas's own generated building stock -- house rows,
 *                    downtown blocks, office slabs. This is what makes a
 *                    zoomed-in town look inhabited rather than like a road
 *                    network with pins floating over it.
 *   paintFootprints  one plan per *place*, shaped by its category, drawn over
 *                    the stock so a named venue always has a building of the
 *                    right kind under its marker.
 *
 * Both fade in at the street tier (k >= 2) rather than the block tier, which
 * is when there is finally enough screen space per building for them to be
 * anything other than noise.
 * ------------------------------------------------------------------------ */

const BUILDINGS_MIN_K = 2.0;

export function paintBuildings(ctx, atlas, view) {
  if (view.k < BUILDINGS_MIN_K) return;
  const cache = cachedPaths(atlas);
  if (!cache.buildings.length) return;
  const s = applyView(ctx, view);

  // Fade in over the first half-step of zoom so buildings arrive rather than
  // pop. Below ~1px of stroke there is no point drawing outlines at all.
  const alpha = Math.min(1, (view.k - BUILDINGS_MIN_K) / 0.8);
  const strokePx = 0.7;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineWidth = strokePx / s;
  ctx.lineJoin = 'round';

  // Group by class so we are not thrashing fillStyle per building.
  const byClass = new Map();
  for (const b of cache.buildings) {
    const cls = b.feat.class || 'generic';
    if (!byClass.has(cls)) byClass.set(cls, []);
    byClass.get(cls).push(b);
  }
  // Outline as soon as the stroke is worth about a pixel. Waiting until k>=3
  // left the whole street tier as flat untextured tan; the edge is what makes a
  // row of buildings legible as separate buildings.
  const outline = view.k >= 2.2;
  for (const [cls, list] of byClass) {
    const pal = buildingFill(cls);
    ctx.fillStyle = pal.fill;
    ctx.strokeStyle = pal.stroke;
    for (const b of list) {
      ctx.fill(b.path);
      if (outline) ctx.stroke(b.path);
    }
  }
  ctx.restore();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

export function paintFootprints(ctx, atlas, view) {
  if (view.k < BUILDINGS_MIN_K) return;
  const cache = cachedPaths(atlas);
  if (!cache.footprints.length) return;
  const s = applyView(ctx, view);

  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineWidth = 0.9 / s;

  // Named venues sit slightly proud of the generated stock: a soft drop
  // shadow at block zoom is enough to separate "the cafe" from "a building".
  const shadow = view.k >= 3.4;
  if (shadow) {
    ctx.shadowColor = 'rgba(90, 80, 64, 0.22)';
    ctx.shadowBlur = 3 / s;
    ctx.shadowOffsetY = 1.2 / s;
  }

  for (const f of cache.footprints) {
    if (view.k < f.minZoom) continue;
    const pal = buildingFill(f.cls);
    ctx.fillStyle = pal.fill;
    ctx.fill(f.path);
  }

  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.shadowOffsetY = 0;

  if (view.k >= 2.6) {
    for (const f of cache.footprints) {
      if (view.k < f.minZoom) continue;
      ctx.strokeStyle = buildingFill(f.cls).stroke;
      ctx.stroke(f.path);
    }
  }
  ctx.restore();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

/* ------------------------------------------------------------------------ *
 * Minimap
 * ------------------------------------------------------------------------ */

export function paintMinimap(ctx, atlas, view, w, h) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const [x0, y0, x1, y1] = atlas.meta.bounds;
  const s = Math.min(w / (x1 - x0), h / (y1 - y0));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.setTransform(s * dpr, 0, 0, s * dpr, -x0 * s * dpr, -y0 * s * dpr);
  const cache = cachedPaths(atlas);
  ctx.fillStyle = TERRAIN_FILL.ocean;
  ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  for (const item of cache.terrain) {
    const fill = TERRAIN_FILL[item.feat.class];
    if (!fill) continue;
    ctx.fillStyle = fill;
    ctx.fill(item.path);
  }
  ctx.restore();

  // Viewport rectangle.
  const tl = view.toMap(0, 0);
  const br = view.toMap(view.w, view.h);
  ctx.save();
  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(
      (tl[0] - x0) * s, (tl[1] - y0) * s,
      (br[0] - tl[0]) * s, (br[1] - tl[1]) * s);
  ctx.restore();
}
