/* disableFinding(Lint) */
/**
 * @fileoverview Agent layer: positioning, movement animation, trails and aggregate flows.
 * @suppress {lintChecks}
 *
 * Honesty note
 * ------------
 * The simulation records *arrivals*, not journeys: an agent is at A on tick t
 * and at B on tick t+1, with no transit state and no intermediate coordinates.
 * Everything drawn between two anchors here is therefore interpolation, not
 * data. Three rules follow from that and are enforced below:
 *
 *   1. A tick with no record for an agent renders as a dimmed, dashed
 *      "unknown" marker -- we never interpolate across a gap as if we knew.
 *   2. The daily 07:00 reset (the simulation clock teleports everyone home at
 *      the day boundary) is a mechanic, not a commute. When a majority
 *      of agents relocate on a single tick we cross-fade instead of animating
 *      a mass sprint across town.
 *   3. The scrubber only ever stops on real ticks. Intermediate frames are
 *      presentation, and are labelled as such in the legend.
 */

import {agentColor, hash01, lodFor} from './atlas.js';

const TWEEN_MS = 750;
const STAGGER_MS = 160;
const TRAIL_TICKS = 6;
const RESET_FRACTION = 0.55;  // share of agents moving that implies a reset
// Forward-fill confidence decay. A position carried this many ticks past its
// last confirming log is drawn dashed; at STALE_FULL_TICKS it reaches maximum
// fade. Agents normally re-log every tick or two, so anything beyond these is
// a genuine gap in the record rather than ordinary jitter.
const STALE_DASH_TICKS = 3;
const STALE_FULL_TICKS = 8;

function easeInOut(u) {
  return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
}

export class AgentLayer {
  constructor(atlas) {
    this.setAtlas(atlas);
    this.sprites = new Map();
    this.showTrails = false;
    this.showFlows = false;
    this.showHeat = false;
    this.selected = null;
    this.filterCategory = null;
    this._flows = [];
    this._resetting = false;
  }

  setAtlas(atlas) {
    this.atlas = atlas;
    this.byId = new Map(atlas.places.map((p) => [p.id, p]));
    // Unit suffixes are a naming convention in the sim rather than real data,
    // so we resolve them back to their building here.
    this.buildingIds = atlas.places
        .filter((p) => p.category === 'home' && !p.parent)
        .map((p) => p.id);
    this.aliases = new Map(Object.entries(atlas.aliases || {}));
    // Exact renames of runner-canonicalised building ids (see atlas_loader).
    this.simIds = new Map(Object.entries(atlas.sim_ids || {}));
    this._occCache = null;
  }

  /**
   * Resolves a simulation location string to an atlas place.
   *
   * Returns `{place, approx}` or null. `approx` is true when the match came
   * from the atlas `aliases` table, i.e. the simulation named a venue this
   * geography does not have and we substituted the nearest real one. Callers
   * MUST surface that rather than drawing it like a known position.
   *
   * Returns null when the name is genuinely unknown -- callers must render
   * that as an explicit unknown state rather than inventing a position.
   */
  resolvePlace(loc) {
    if (!loc) return null;
    const direct = this.byId.get(loc);
    if (direct) return {place: direct, approx: false};
    // "<building>_unit_12" -> "<building>"
    const m = /^(.*)_unit_\d+$/.exec(loc);
    if (m && this.byId.has(m[1])) {
      return {place: this.byId.get(m[1]), approx: false};
    }
    const base = m ? m[1] : loc;
    const renamed = this.simIds.get(base);
    if (renamed && this.byId.has(renamed)) {
      return {place: this.byId.get(renamed), approx: false};
    }
    for (const b of this.buildingIds) {
      if (loc.startsWith(b + '_')) {
        return {place: this.byId.get(b), approx: false};
      }
    }
    const alias = this.aliases.get(loc);
    if (alias && this.byId.has(alias)) {
      return {place: this.byId.get(alias), approx: true};
    }
    return null;
  }

  /** Where an agent standing at `place` should be drawn, at the current LOD. */
  anchorFor(place, agentName, lod) {
    // Below the block tier, children collapse into their parent building.
    let p = place;
    if (lod < 3 && p.parent && this.byId.has(p.parent)) {
      p = this.byId.get(p.parent);
    }
    // Deterministic jitter so co-located agents form a readable cluster
    // instead of stacking into one dot, and so they do not jump on re-render.
    const a = hash01(agentName + '|' + p.id) * Math.PI * 2;
    const r = 3 + hash01(p.id + '|' + agentName) * 7;
    return [p.xy[0] + Math.cos(a) * r, p.xy[1] + Math.sin(a) * r];
  }

  /**
   * Installs a new per-tick snapshot and starts the tween toward it.
   *
   * `row.age`, when present, is the number of ticks since the movement logs
   * last actually confirmed that agent's location (0 == confirmed this tick).
   * Positions with age > 0 are forward-filled reconstructions, and are drawn
   * as provisional so a run that stopped logging cannot masquerade as a run
   * whose agents all stood still.
   *
   * @param {!Array<{agent: string, location: string, age: (number|undefined)}>}
   *     rows
   * @param {number} tick
   * @param {number} lod
   * @param {boolean} animate
   */
  setSnapshot(rows, tick, lod, animate) {
    const now = performance.now();
    const seen = new Set();
    let moved = 0;
    let total = 0;

    for (const row of rows) {
      const name = row.agent;
      if (!name) continue;
      seen.add(name);
      total++;
      const hit = this.resolvePlace(row.location);
      const place = hit && hit.place;
      let sp = this.sprites.get(name);
      if (!sp) {
        sp = {
          name,
          color: agentColor(name),
          loc: null,
          place: null,
          pos: null,
          from: null,
          to: null,
          t0: 0,
          delay: 0,
          unknown: false,
          approx: false,
          stale: 0,
          trail: [],
        };
        this.sprites.set(name, sp);
      }

      sp.stale = Number.isFinite(row.age) ? row.age : 0;

      if (!place) {
        // Unknown location: hold last known position but mark it uncertain.
        sp.unknown = true;
        sp.approx = false;
        sp.loc = row.location || null;
        continue;
      }

      const target = this.anchorFor(place, name, lod);
      const changed = sp.place && sp.place.id !== place.id;
      if (changed) moved++;

      if (!sp.pos) {
        sp.pos = target.slice();
        sp.from = target.slice();
        sp.to = target.slice();
      } else {
        sp.from = sp.pos.slice();
        sp.to = target;
      }
      sp.t0 = now;
      sp.delay = animate && changed ? hash01(name) * STAGGER_MS : 0;
      sp.unknown = false;
      // The simulation named a venue this geography does not have; we are
      // showing the atlas's declared stand-in, not a surveyed position.
      sp.approx = hit.approx;
      sp.animate = !!animate && changed;
      sp.place = place;
      sp.loc = row.location;

      if (changed || !sp.trail.length) {
        sp.trail.push({x: target[0], y: target[1], tick});
        if (sp.trail.length > TRAIL_TICKS) sp.trail.shift();
      }
    }

    // Agents absent from this snapshot are unknown for this tick, not gone.
    // If rows was completely empty, preserve existing state rather than marking all unknown.
    if (seen.size > 0) {
      for (const [name, sp] of this.sprites) {
        if (!seen.has(name)) sp.unknown = true;
      }
    }

    // Mass relocation means the day-boundary reset fired; cross-fade instead
    // of animating everyone across the map at once.
    this._resetting = total > 0 && moved / total >= RESET_FRACTION;
    if (this._resetting) {
      for (const sp of this.sprites.values()) {
        sp.animate = false;
        if (sp.to) sp.pos = sp.to.slice();
      }
    }

    this._occCache = null;
    this._recomputeFlows();
  }

  /** True while any sprite is still in motion. */
  advance(now) {
    let busy = false;
    for (const sp of this.sprites.values()) {
      if (!sp.pos || !sp.to) continue;
      if (!sp.animate) {
        sp.pos[0] = sp.to[0];
        sp.pos[1] = sp.to[1];
        continue;
      }
      const t = now - sp.t0 - sp.delay;
      if (t <= 0) { busy = true; continue; }
      const u = Math.min(1, t / TWEEN_MS);
      const e = easeInOut(u);
      // A gentle perpendicular arc reads as travel rather than as a ruler
      // line, without pretending to know the actual route taken.
      const dx = sp.to[0] - sp.from[0];
      const dy = sp.to[1] - sp.from[1];
      const dist = Math.hypot(dx, dy);
      const bow = Math.min(dist * 0.13, 26) * Math.sin(Math.PI * e);
      const nx = dist > 0.001 ? -dy / dist : 0;
      const ny = dist > 0.001 ? dx / dist : 0;
      const side = hash01(sp.name) > 0.5 ? 1 : -1;
      sp.pos[0] = sp.from[0] + dx * e + nx * bow * side;
      sp.pos[1] = sp.from[1] + dy * e + ny * bow * side;
      if (u < 1) busy = true;
      else sp.animate = false;
    }
    return busy;
  }

  /** place id -> array of sprites currently there. */
  occupancy() {
    if (this._occCache) return this._occCache;
    const m = new Map();
    for (const sp of this.sprites.values()) {
      if (!sp.place || sp.unknown) continue;
      const id = sp.place.id;
      if (!m.has(id)) m.set(id, []);
      m.get(id).push(sp);
    }
    this._occCache = m;
    return m;
  }

  /** Occupancy rolled up to districts, for the region/district tiers. */
  districtOccupancy() {
    const m = new Map();
    for (const sp of this.sprites.values()) {
      if (!sp.place || sp.unknown) continue;
      const d = sp.place.district;
      if (!d) continue;
      if (!m.has(d)) m.set(d, []);
      m.get(d).push(sp);
    }
    return m;
  }

  unknownCount() {
    let n = 0;
    for (const sp of this.sprites.values()) if (sp.unknown) n++;
    return n;
  }

  /** How many agents are drawn at an alias stand-in rather than a real place. */
  approxCount() {
    let n = 0;
    for (const sp of this.sprites.values()) {
      if (!sp.unknown && sp.approx) n++;
    }
    return n;
  }

  /** The distinct simulation location names currently being aliased. */
  approxLocations() {
    const s = new Set();
    for (const sp of this.sprites.values()) {
      if (!sp.unknown && sp.approx && sp.loc) s.add(sp.loc);
    }
    return [...s].sort();
  }

  /** The distinct simulation location names we could not place at all. */
  unknownLocations() {
    const s = new Set();
    for (const sp of this.sprites.values()) {
      if (sp.unknown && sp.loc) s.add(sp.loc);
    }
    return [...s].sort();
  }

  /**
   * How many agents are shown at a position the logs stopped confirming at
   * least STALE_DASH_TICKS ago, plus the worst such gap.
   * @return {{n: number, max: number}}
   */
  staleCount() {
    let n = 0;
    let max = 0;
    for (const sp of this.sprites.values()) {
      if (sp.unknown) continue;
      if (sp.stale >= STALE_DASH_TICKS) n++;
      if (sp.stale > max) max = sp.stale;
    }
    return {n, max};
  }

  _recomputeFlows() {
    // Aggregate origin -> destination pairs over the retained trail window.
    const pairs = new Map();
    for (const sp of this.sprites.values()) {
      for (let i = 1; i < sp.trail.length; i++) {
        const a = sp.trail[i - 1];
        const b = sp.trail[i];
        if (Math.abs(a.x - b.x) < 0.5 && Math.abs(a.y - b.y) < 0.5) continue;
        const key = `${a.x.toFixed(0)},${a.y.toFixed(0)}>${b.x.toFixed(0)},${b.y.toFixed(0)}`;
        const cur = pairs.get(key);
        if (cur) cur.n++;
        else pairs.set(key, {a, b, n: 1});
      }
    }
    this._flows = [...pairs.values()].sort((p, q) => q.n - p.n).slice(0, 60);
  }

  /* ---------------------------------------------------------------------- *
   * Painting
   * ---------------------------------------------------------------------- */

  paint(ctx, view) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const lod = lodFor(view.k);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, view.w, view.h);

    if (this.showHeat) this._paintHeat(ctx, view);
    if (this.showFlows) this._paintFlows(ctx, view);
    if (this.showTrails) this._paintTrails(ctx, view);

    const r = Math.max(2.8, Math.min(7, 2.2 * Math.sqrt(view.k) + 1.6));
    for (const sp of this.sprites.values()) {
      if (!sp.pos) continue;
      if (this.filterCategory && sp.place &&
          sp.place.category !== this.filterCategory) continue;
      const [x, y] = view.toScreen(sp.pos[0], sp.pos[1]);
      if (x < -20 || y < -20 || x > view.w + 20 || y > view.h + 20) continue;

      const isSel = this.selected === sp.name;

      if (sp.unknown) {
        // Explicitly uncertain: hollow, dashed, dimmed.
        ctx.save();
        ctx.globalAlpha = 0.45;
        ctx.setLineDash([2.5, 2.5]);
        ctx.strokeStyle = sp.color;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 6.2832);
        ctx.stroke();
        ctx.restore();
        continue;
      }

      if (isSel) {
        ctx.beginPath();
        ctx.arc(x, y, r + 6, 0, 6.2832);
        ctx.fillStyle = 'rgba(2,132,199,0.22)';
        ctx.fill();
      }

      if (sp.approx) {
        // Positioned via an atlas alias, not a real place in this geography.
        // Amber dotted halo so the substitution is legible at a glance.
        ctx.save();
        ctx.setLineDash([1.8, 2.2]);
        ctx.strokeStyle = 'rgba(217,119,6,0.85)';
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(x, y, r + 3.2, 0, 6.2832);
        ctx.stroke();
        ctx.restore();
      }

      // Forward-filled positions are a reconstruction, not a recording. Fade
      // them out as the last confirmed observation recedes, and drop the solid
      // outline once the position is more assertion than evidence.
      const staleF = Math.min(1, Math.max(0, sp.stale / STALE_FULL_TICKS));
      ctx.save();
      if (staleF > 0) ctx.globalAlpha = 1 - 0.6 * staleF;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 6.2832);
      ctx.fillStyle = sp.color;
      ctx.fill();
      ctx.lineWidth = isSel ? 2.4 : 1.5;
      if (sp.stale >= STALE_DASH_TICKS && !isSel) {
        ctx.setLineDash([2.2, 2.2]);
        ctx.strokeStyle = 'rgba(255,255,255,0.95)';
      } else {
        ctx.strokeStyle = isSel ? '#0f172a' : 'rgba(255,255,255,0.95)';
      }
      ctx.stroke();
      ctx.restore();
    }

    // Names only once there is room for them.
    if (lod >= 3) {
      ctx.font = '600 10px Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(255,255,255,0.92)';
      ctx.fillStyle = '#334155';
      for (const sp of this.sprites.values()) {
        if (!sp.pos || sp.unknown) continue;
        const [x, y] = view.toScreen(sp.pos[0], sp.pos[1]);
        if (x < 0 || y < 0 || x > view.w || y > view.h) continue;
        const first = sp.name.split(' ')[0];
        ctx.strokeText(first, x, y - r - 4);
        ctx.fillText(first, x, y - r - 4);
      }
    }
  }

  _paintTrails(ctx, view) {
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const sp of this.sprites.values()) {
      if (sp.trail.length < 2) continue;
      if (this.selected && this.selected !== sp.name) continue;
      // Comet tail: older segments thinner and fainter.
      for (let i = 1; i < sp.trail.length; i++) {
        const a = view.toScreen(sp.trail[i - 1].x, sp.trail[i - 1].y);
        const b = view.toScreen(sp.trail[i].x, sp.trail[i].y);
        const age = (sp.trail.length - i) / sp.trail.length;
        ctx.globalAlpha = 0.42 * (1 - age) + 0.06;
        ctx.strokeStyle = sp.color;
        ctx.lineWidth = 1 + 2.4 * (1 - age);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }
      // Live segment from the last anchor to where the sprite is now.
      if (sp.pos) {
        const last = sp.trail[sp.trail.length - 1];
        const a = view.toScreen(last.x, last.y);
        const b = view.toScreen(sp.pos[0], sp.pos[1]);
        ctx.globalAlpha = 0.5;
        ctx.strokeStyle = sp.color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  _paintFlows(ctx, view) {
    if (!this._flows.length) return;
    const max = this._flows[0].n || 1;
    ctx.save();
    ctx.lineCap = 'round';
    for (const f of this._flows) {
      const a = view.toScreen(f.a.x, f.a.y);
      const b = view.toScreen(f.b.x, f.b.y);
      const mx = (a[0] + b[0]) / 2;
      const my = (a[1] + b[1]) / 2;
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const d = Math.hypot(dx, dy) || 1;
      // Bow the ribbon so opposing flows between the same pair stay distinct.
      const cx = mx + (-dy / d) * d * 0.16;
      const cy = my + (dx / d) * d * 0.16;
      ctx.globalAlpha = 0.1 + 0.4 * (f.n / max);
      ctx.strokeStyle = '#0369a1';
      ctx.lineWidth = 1 + 7 * (f.n / max);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.quadraticCurveTo(cx, cy, b[0], b[1]);
      ctx.stroke();
    }
    ctx.restore();
  }

  _paintHeat(ctx, view) {
    const occ = this.occupancy();
    if (!occ.size) return;
    let max = 1;
    for (const list of occ.values()) max = Math.max(max, list.length);
    ctx.save();
    for (const [id, list] of occ) {
      const p = this.byId.get(id);
      if (!p) continue;
      const [x, y] = view.toScreen(p.xy[0], p.xy[1]);
      const intensity = list.length / max;
      const radius = 28 + 70 * intensity * Math.min(2, view.k);
      const g = ctx.createRadialGradient(x, y, 0, x, y, radius);
      g.addColorStop(0, `rgba(239,108,42,${0.36 * intensity + 0.08})`);
      g.addColorStop(1, 'rgba(239,108,42,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 6.2832);
      ctx.fill();
    }
    ctx.restore();
  }
}
