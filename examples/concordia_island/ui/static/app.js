/* disableFinding(Lint) */
/**
 * @fileoverview Application shell: data loading, render loop, interaction, timeline, panels.
 * @suppress {lintChecks}
 *
 * Data flow
 *   /api/config           once at boot -- run id, geographies, clock parameters
 *   /api/atlas?g=<id>     once per geography -- static, cached by the browser
 *   /api/data             polled -- live tick, agent locations, metrics
 *   /api/location_history polled less often -- per-tick snapshots for replay
 */

import {AgentLayer} from './agents.js';
import {CATEGORIES, categoryOf, initials, LOD_NAMES, lodFor, View} from './atlas.js';
import {MarkerLayer} from './markers.js';
import {paintBuildings, paintFootprints, paintMinimap, paintRoads, paintShields, paintTerrain} from './terrain.js';

/* ------------------------------------------------------------------------ *
 * State
 * ------------------------------------------------------------------------ */

const S = {
  config: null,
  atlas: null,
  view: new View(),
  agents: null,
  markers: null,
  ticks: [],
  snapshots: {},
  clock: null,  // last SimulationClock row, or null
  idx: -1,
  mode: 'live',  // 'live' | 'replay'
  playing: false,
  playTimer: null,
  latest: [],
  selectedAgent: null,
  historyLoaded: false,
  dirty: true,
  rafPending: false,
  layers: {
    terrain: true,
    roads: true,
    places: true,
    labels: true,
    trails: false,
    flows: false,
    heat: false,
    footprints: true,
  },
  dayNight: true,
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------------ *
 * Error surface -- loud, never silent
 * ------------------------------------------------------------------------ */

function showError(title, detail) {
  $('err-title').textContent = title;
  $('err-body').textContent =
      typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2);
  $('errbar').classList.add('show');
  console.error(title, detail);
}
function clearError() {
  $('errbar').classList.remove('show');
}
$('err-close').addEventListener('click', clearError);

async function getJSON(url) {
  const r = await fetch(url);
  const text = await r.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch (e) {
    throw new Error(
        `${url} returned non-JSON (HTTP ${r.status}):\n` + text.slice(0, 600));
  }
  if (!r.ok) {
    throw new Error(
        `${url} failed (HTTP ${r.status}):\n` +
        (body.error || text.slice(0, 600)));
  }
  return body;
}

/* ------------------------------------------------------------------------ *
 * Simulation clock
 *
 * Real date arithmetic from an ISO base supplied by the server, so a 40-day
 * run does not render as "January 40th" the way the old dashboard did.
 * ------------------------------------------------------------------------ */

function simTimeFor(tick) {
  const c = S.config;
  if (!c || !c.sim_start_iso) return null;
  const base = new Date(c.sim_start_iso);
  if (isNaN(base)) return null;
  const d = new Date(
      base.getTime() + (tick - 1) * (c.tick_interval_min || 120) * 60000);
  return d;
}

function fmtSimTime(d) {
  if (!d) return {full: '\u2014', short: '\u2014', hour: 12};
  const opts = {weekday: 'short', month: 'short', day: 'numeric'};
  const date = d.toLocaleDateString('en-US', opts);
  const time =
      d.toLocaleTimeString('en-US', {hour: 'numeric', minute: '2-digit'});
  return {
    full: `${date}, ${time}`,
    short: time,
    date,
    hour: d.getHours() + d.getMinutes() / 60,
  };
}

/* How hard the day/night wash is applied, 0 = off, 1 = the literal stop
 * colour below.
 *
 * At full strength the night stop multiplies terrain by roughly
 * (0.49, 0.55, 0.77), which darkens the map by about half and obliterates the
 * tan-versus-green distinction that the whole cartography depends on -- you
 * could no longer tell a town from a field at 5am. The wash only needs to say
 * "it is night", not simulate night, so it is pulled back toward white.
 *
 * Kept as one knob with the stop table left describing real light colours,
 * rather than hand-desaturating each stop, so this stays tunable in one place.
 */
const TINT_STRENGTH = 0.5;

/** Multiply-blend wash approximating the light at a given hour. */
function tintFor(hour) {
  const stops = [
    [0, [126, 140, 196]],  // night
    [5, [126, 140, 196]],
    [7, [255, 217, 176]],  // dawn
    [9, [255, 255, 255]],  // day
    [17, [255, 255, 255]],
    [19, [255, 196, 168]],  // dusk
    [21, [126, 140, 196]],  // night
    [24, [126, 140, 196]],
  ];
  let a = stops[0], b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (hour >= stops[i][0] && hour <= stops[i + 1][0]) {
      a = stops[i];
      b = stops[i + 1];
      break;
    }
  }
  const span = b[0] - a[0] || 1;
  const u = Math.min(1, Math.max(0, (hour - a[0]) / span));
  const c = [0, 1, 2].map((i) => {
    const lit = a[1][i] + (b[1][i] - a[1][i]) * u;
    // Lerp toward white (255 = multiply no-op).
    return Math.round(255 - (255 - lit) * TINT_STRENGTH);
  });
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function applyTint() {
  const d = simTimeFor(currentTick());
  const f = fmtSimTime(d);
  $('tod-readout').textContent = d ? f.full : '\u2014';
  $('tint').style.background =
      (S.dayNight && d) ? tintFor(f.hour) : 'transparent';
}

/* ------------------------------------------------------------------------ *
 * Canvas plumbing
 * ------------------------------------------------------------------------ */

const terrainCv = $('layer-terrain');
const agentCv = $('layer-agents');
const tctx = terrainCv.getContext('2d');
const actx = agentCv.getContext('2d');
const miniCv = $('minimap-canvas');
const mctx = miniCv.getContext('2d');

function sizeCanvases() {
  const stage = $('stage');
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  for (const [cv, ctx] of [[terrainCv, tctx], [agentCv, actx]]) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
  }
  miniCv.width = Math.round(150 * dpr);
  miniCv.height = Math.round(105 * dpr);
  mctx.setTransform(1, 0, 0, 1, 0, 0);
  mctx.scale(dpr, dpr);
  S.view.resize(w, h);
  markDirty();
}

function markDirty() {
  S.dirty = true;
  if (!S.rafPending) {
    S.rafPending = true;
    requestAnimationFrame(frame);
  }
}

function frame(now) {
  S.rafPending = false;
  const moving = S.agents ? S.agents.advance(now) : false;
  if (S.dirty) {
    drawStatic();
    S.dirty = false;
  }
  if (S.agents) S.agents.paint(actx, S.view);
  if (S.markers) {
    S.markers.update(
        S.view, S.agents.occupancy(), S.agents.districtOccupancy());
  }
  if (moving) {
    S.rafPending = true;
    requestAnimationFrame(frame);
  }
}

function drawStatic() {
  if (!S.atlas) return;
  tctx.setTransform(1, 0, 0, 1, 0, 0);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  tctx.scale(dpr, dpr);
  tctx.clearRect(0, 0, S.view.w, S.view.h);

  if (S.layers.terrain) {
    paintTerrain(tctx, S.atlas, S.view, {});
  }
  if (S.layers.roads) {
    paintRoads(tctx, S.atlas, S.view);
  }
  // Buildings sit *over* the road network: drawn underneath, every street
  // would slice through the venue it runs past.
  if (S.layers.footprints) {
    paintBuildings(tctx, S.atlas, S.view);
    paintFootprints(tctx, S.atlas, S.view);
  }
  if (S.layers.roads) {
    tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    paintShields(tctx, S.atlas, S.view);
  }

  $('layer-vector').style.display = S.layers.places ? '' : 'none';
  $('layer-labels').style.display = S.layers.labels ? '' : 'none';

  paintMinimap(mctx, S.atlas, S.view, 150, 105);
  updateZoomReadout();
  updateScaleBar();
  updateBreadcrumb();
}

function updateZoomReadout() {
  $('zoom-readout').textContent = Math.round(S.view.k * 100) + '%';
}

function updateScaleBar() {
  const m = S.atlas && S.atlas.meta;
  // `km_across` is the real-world distance the full map width represents. It
  // is the only thing that makes the bar meaningful, so an atlas that omits it
  // gets no scale bar at all rather than a fabricated one.
  if (!m || !m.km_across) {
    $('scale-label').textContent = '';
    $('scale-bar').style.width = '0px';
    return;
  }
  const kmPerUnit = m.km_across / (m.bounds[2] - m.bounds[0]);
  const pxPerUnit = S.view.scale();
  const targetPx = 80;
  const km = (targetPx / pxPerUnit) * kmPerUnit;
  const nice = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20];
  let chosen = nice[0];
  for (const n of nice)
    if (n <= km) chosen = n;
  const px = (chosen / kmPerUnit) * pxPerUnit;
  $('scale-bar').style.width = Math.max(28, Math.min(150, px)) + 'px';
  $('scale-label').textContent =
      chosen < 1 ? `${Math.round(chosen * 1000)} m` : `${chosen} km`;
}

function updateBreadcrumb() {
  const bc = $('breadcrumb');
  const lod = lodFor(S.view.k);
  const parts =
      [`<button data-act="fit">${S.atlas.meta.display_name}</button>`];
  if (lod >= 1) {
    const c = S.view.toMap(S.view.w / 2, S.view.h / 2);
    const d = nearestDistrict(c[0], c[1]);
    if (d) {
      parts.push('<span class="sep">&rsaquo;</span>');
      parts.push(
          `<button data-act="district" data-id="${d.id}">${d.name}</button>`);
    }
  }
  parts.push('<span class="sep">&rsaquo;</span>');
  parts.push(`<span class="cur">${LOD_NAMES[lod]}</span>`);
  bc.innerHTML = parts.join(' ');
}

function nearestDistrict(x, y) {
  let best = null, bd = Infinity;
  for (const d of S.atlas.districts) {
    if (!d.label_anchor) continue;
    const dd = Math.hypot(d.label_anchor[0] - x, d.label_anchor[1] - y);
    if (dd < bd) {
      bd = dd;
      best = d;
    }
  }
  return best;
}

$('breadcrumb').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  if (b.dataset.act === 'fit') {
    S.view.flyTo(
        (S.atlas.meta.bounds[0] + S.atlas.meta.bounds[2]) / 2,
        (S.atlas.meta.bounds[1] + S.atlas.meta.bounds[3]) / 2, 1, 520,
        markDirty, markDirty);
  } else if (b.dataset.act === 'district') {
    const d = S.atlas.districts.find((x) => x.id === b.dataset.id);
    if (d) flyToDistrict(d);
  }
});

function flyToDistrict(d) {
  const a = d.label_anchor || [500, 350];
  S.view.flyTo(a[0], a[1], 2.4, 620, markDirty, markDirty);
}

/* ------------------------------------------------------------------------ *
 * Interaction
 * ------------------------------------------------------------------------ */

const stage = $('stage');

stage.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = stage.getBoundingClientRect();
  S.view.zoomAt(
      e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.18 : 1 / 1.18);
  markDirty();
}, {passive: false});

let drag = null;
stage.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  drag = {x: e.clientX, y: e.clientY, moved: false};
  stage.classList.add('dragging');
});
window.addEventListener('mousemove', (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x;
  const dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
  S.view.panBy(dx, dy);
  drag.x = e.clientX;
  drag.y = e.clientY;
  markDirty();
});
window.addEventListener('mouseup', () => {
  drag = null;
  stage.classList.remove('dragging');
});

stage.addEventListener('dblclick', (e) => {
  const r = stage.getBoundingClientRect();
  S.view.zoomAt(e.clientX - r.left, e.clientY - r.top, 1.9);
  markDirty();
});

$('zoom-in').addEventListener('click', () => {
  S.view.zoomAt(S.view.w / 2, S.view.h / 2, 1.35);
  markDirty();
});
$('zoom-out').addEventListener('click', () => {
  S.view.zoomAt(S.view.w / 2, S.view.h / 2, 1 / 1.35);
  markDirty();
});
$('zoom-fit').addEventListener('click', () => {
  S.view.fit();
  markDirty();
});

window.addEventListener('keydown', (e) => {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  switch (e.key) {
    case '+':
    case '=':
      S.view.zoomAt(S.view.w / 2, S.view.h / 2, 1.35);
      markDirty();
      break;
    case '-':
    case '_':
      S.view.zoomAt(S.view.w / 2, S.view.h / 2, 1 / 1.35);
      markDirty();
      break;
    case '0':
      S.view.fit();
      markDirty();
      break;
    case 'Escape':
      S.view.zoomAt(S.view.w / 2, S.view.h / 2, 1 / 1.9);
      markDirty();
      break;
    case 'ArrowLeft':
      e.preventDefault();
      tlStep(-1);
      break;
    case 'ArrowRight':
      e.preventDefault();
      tlStep(1);
      break;
    case ' ':
      e.preventDefault();
      tlToggle();
      break;
  }
});

/* Tooltip */
const tip = $('tip');
function showTip(html, e) {
  tip.innerHTML = html;
  tip.classList.add('show');
  const r = stage.getBoundingClientRect();
  const x = e.clientX - r.left + 14;
  const y = e.clientY - r.top + 14;
  tip.style.left = Math.min(x, r.width - tip.offsetWidth - 10) + 'px';
  tip.style.top = Math.min(y, r.height - tip.offsetHeight - 10) + 'px';
}
function hideTip() {
  tip.classList.remove('show');
}

/* ------------------------------------------------------------------------ *
 * Panels
 * ------------------------------------------------------------------------ */

function bindPanelToggles() {
  const setup = (side) => {
    const panel = $('panel-' + side);
    const btn = $('toggle-' + side);
    const reposition = () => {
      const collapsed = panel.classList.contains('collapsed');
      const w = side === 'left' ? 272 : 300;
      btn.style[side] = collapsed ? '0px' : w + 'px';
      btn.innerHTML = (side === 'left') ? (collapsed ? '&#9654;' : '&#9664;') :
                                          (collapsed ? '&#9664;' : '&#9654;');
    };
    btn.addEventListener('click', () => {
      panel.classList.toggle('collapsed');
      reposition();
    });
    reposition();
  };
  setup('left');
  setup('right');
}

const LAYER_DEFS = [
  ['terrain', 'Terrain &amp; land cover'],
  ['roads', 'Roads'],
  ['footprints', 'Building footprints'],
  ['places', 'Place markers'],
  ['labels', 'Labels'],
  ['trails', 'Movement trails'],
  ['flows', 'Flow ribbons (aggregate)'],
  ['heat', 'Occupancy heatmap'],
];

function buildLayerToggles() {
  const host = $('layer-toggles');
  host.innerHTML = LAYER_DEFS
                       .map(([k, label]) => `
    <label class="check">
      <input type="checkbox" data-layer="${k}" ${S.layers[k] ? 'checked' : ''}>
      ${label}
    </label>`).join('');
  host.addEventListener('change', (e) => {
    const k = e.target.dataset.layer;
    if (!k) return;
    S.layers[k] = e.target.checked;
    if (S.agents) {
      S.agents.showTrails = S.layers.trails;
      S.agents.showFlows = S.layers.flows;
      S.agents.showHeat = S.layers.heat;
    }
    markDirty();
  });
}

function buildLegend() {
  const used = new Set(S.atlas.places.map((p) => p.category));
  $('legend').innerHTML = [...used]
                              .sort()
                              .map((c) => {
                                const cat = CATEGORIES[c];
                                if (!cat) return '';
                                return `<div class="legend-item">
      <span class="mk" style="background:${cat.color}"></span>${
                                    cat.label}</div>`;
                              })
                              .join('');
}

function buildCategoryFilter() {
  const used = [...new Set(S.atlas.places.map((p) => p.category))].sort();
  const host = $('cat-filter');
  host.innerHTML =
      `<label class="check"><input type="radio" name="catf" value="" checked> All types</label>` +
      used.map((c) => {
            const cat = CATEGORIES[c];
            if (!cat) return '';
            return `<label class="check"><input type="radio" name="catf" value="${
                c}">
          <span class="swatch" style="background:${cat.color}"></span>${
                cat.label}</label>`;
          })
          .join('');
  host.addEventListener('change', (e) => {
    const v = e.target.value || null;
    S.markers.filterCategory = v;
    S.agents.filterCategory = v;
    markDirty();
  });
}

/* --- Roster --- */

function buildRoster() {
  const q = ($('roster-search').value || '').toLowerCase();
  const rows = [...S.agents.sprites.values()]
                   .filter((sp) => !q || sp.name.toLowerCase().includes(q))
                   .sort((a, b) => a.name.localeCompare(b.name));
  $('roster-count').textContent = `(${S.agents.sprites.size})`;
  $('roster').innerHTML =
      rows.map((sp) => {
            const loc = sp.unknown ? '<span style="opacity:.6">unknown</span>' :
                                     (sp.place ? sp.place.name : '\u2014');
            const sel = S.selectedAgent === sp.name ? ' sel' : '';
            return `<div class="roster-row${sel}" data-agent="${
                escAttr(sp.name)}">
      <span class="av" style="background:${sp.color}">${
                initials(sp.name)}</span>
      <span class="nm">${esc(sp.name)}</span>
      <span class="loc">${loc}</span>
    </div>`;
          })
          .join('') ||
      '<div class="empty">No agents match</div>';
}

$('roster-search').addEventListener('input', buildRoster);

$('roster').addEventListener('click', (e) => {
  const row = e.target.closest('[data-agent]');
  if (!row) return;
  selectAgent(row.dataset.agent);
});

function selectAgent(name) {
  S.selectedAgent = (S.selectedAgent === name) ? null : name;
  S.agents.selected = S.selectedAgent;
  buildRoster();
  buildDetail();
  if (S.selectedAgent) {
    const sp = S.agents.sprites.get(S.selectedAgent);
    if (sp && sp.pos) {
      S.view.flyTo(
          sp.pos[0], sp.pos[1], Math.max(S.view.k, 3.2), 560, markDirty,
          markDirty);
    }
  }
  markDirty();
}

function buildDetail() {
  const host = $('detail');
  const sp = S.selectedAgent && S.agents.sprites.get(S.selectedAgent);
  if (!sp) {
    host.innerHTML =
        '<div class="empty"><span class="ico">&#128100;</span>Select an agent</div>';
    return;
  }
  const here = sp.place ? (S.agents.occupancy().get(sp.place.id) || []) : [];
  const others = here.filter((o) => o.name !== sp.name).map((o) => o.name);
  host.innerHTML = `
    <div class="hdr">
      <span class="av" style="background:${sp.color}">${
      initials(sp.name)}</span>
      <span class="nm">${esc(sp.name)}</span>
    </div>
    <div class="kv"><span class="k">Location</span><span class="v">${
      sp.unknown ? '<i>not recorded</i>' :
                   esc(sp.place ? sp.place.name : '\u2014')}</span></div>
    <div class="kv"><span class="k">Raw id</span><span class="v" style="font-size:11px;opacity:.7">${
      esc(sp.loc || '\u2014')}</span></div>
    <div class="kv"><span class="k">District</span><span class="v">${
      esc(districtName(sp.place))}</span></div>
    <div class="kv"><span class="k">Also here</span><span class="v">${
      others.length ? esc(others.slice(0, 4).join(', ')) +
              (others.length > 4 ? ` +${others.length - 4}` : '') :
                      'nobody'}</span></div>
    <div class="kv"><span class="k">Places visited</span><span class="v">${
      sp.trail.length}</span></div>`;
}

function districtName(place) {
  if (!place || !place.district) return '\u2014';
  const d = S.atlas.districts.find((x) => x.id === place.district);
  return d ? d.name : place.district;
}

const esc = (s) => String(s).replace(
    /[&<>]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]));
const escAttr = (s) => String(s).replace(/"/g, '&quot;');

/* ------------------------------------------------------------------------ *
 * Timeline
 * ------------------------------------------------------------------------ */

function currentTick() {
  return S.idx >= 0 && S.idx < S.ticks.length ? S.ticks[S.idx] : 0;
}

function tlSetMode(mode) {
  S.mode = mode;
  const p = $('pill-mode');
  p.textContent = mode === 'live' ? 'LIVE' : 'REPLAY';
  p.className = 'pill ' + (mode === 'live' ? 'live' : 'replay');
}

function tlGoTo(idx, animate) {
  if (!S.ticks.length) return;
  idx = Math.max(0, Math.min(S.ticks.length - 1, idx));
  S.idx = idx;
  const tick = S.ticks[idx];
  if (idx < S.ticks.length - 1) tlSetMode('replay');

  let rows = S.snapshots[String(tick)];
  if ((!rows || !rows.length) && idx === S.ticks.length - 1) rows = S.latest;
  if (!rows) {
    // Walk back to the most recent tick that has data rather than silently
    // showing nothing.
    for (let i = idx; i >= 0; i--) {
      const r = S.snapshots[String(S.ticks[i])];
      if (r && r.length) {
        rows = r;
        break;
      }
    }
  }
  if ((!rows || !rows.length) && S.latest && S.latest.length) {
    rows = S.latest;
  }
  S.agents.setSnapshot(rows || [], tick, lodFor(S.view.k), animate !== false);
  tlRender();
  applyTint();
  buildRoster();
  buildDetail();
  // The counts describe the tick on screen, so they have to move with it.
  updateDataQualityPills();
  markDirty();
}

function tlStep(d) {
  if (!S.ticks.length) return;
  const next = S.idx + d;
  if (next >= S.ticks.length) {
    tlGoTo(S.ticks.length - 1);
    tlSetMode('live');
    return;
  }
  tlGoTo(next);
}

function tlPlay() {
  if (!S.ticks.length) return;
  S.playing = true;
  $('tl-play').innerHTML = '&#9208;';
  const speed = +$('tl-speed').value || 900;
  clearInterval(S.playTimer);
  S.playTimer = setInterval(() => {
    if (S.idx < S.ticks.length - 1)
      tlStep(1);
    else
      tlPause();
  }, speed);
}
function tlPause() {
  S.playing = false;
  $('tl-play').innerHTML = '&#9654;';
  clearInterval(S.playTimer);
  S.playTimer = null;
}
function tlToggle() {
  S.playing ? tlPause() : tlPlay();
}

$('tl-play').addEventListener('click', tlToggle);
$('tl-prev').addEventListener('click', () => tlStep(-1));
$('tl-next').addEventListener('click', () => tlStep(1));
$('tl-first').addEventListener('click', () => tlGoTo(0));
$('tl-last').addEventListener('click', () => {
  tlGoTo(S.ticks.length - 1);
  tlSetMode('live');
});
$('tl-speed').addEventListener('change', () => {
  if (S.playing) tlPlay();
});

const track = $('tl-track');
function scrubFromEvent(e) {
  const r = track.getBoundingClientRect();
  const u = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  tlGoTo(Math.round(u * (S.ticks.length - 1)));
}
track.addEventListener('mousedown', (e) => {
  scrubFromEvent(e);
  const move = (ev) => scrubFromEvent(ev);
  const up = () => {
    window.removeEventListener('mousemove', move);
    window.removeEventListener('mouseup', up);
  };
  window.addEventListener('mousemove', move);
  window.addEventListener('mouseup', up);
});
track.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    tlStep(-1);
  }
  if (e.key === 'ArrowRight') {
    e.preventDefault();
    tlStep(1);
  }
});

/** Paints day/night banding and tick marks behind the scrub head. */
function tlRenderTrack() {
  const n = S.ticks.length;
  if (!n) return;
  const bands = [];
  const marks = [];
  for (let i = 0; i < n; i++) {
    const left = (i / Math.max(1, n - 1)) * 100;
    const w = 100 / Math.max(1, n - 1);
    const d = simTimeFor(S.ticks[i]);
    const hour = d ? d.getHours() + d.getMinutes() / 60 : 12;
    const night = hour < 6 || hour >= 20;
    const dusk = (hour >= 18 && hour < 20) || (hour >= 6 && hour < 8);
    const col = night ? 'rgba(51,65,85,.22)' :
        dusk          ? 'rgba(251,146,60,.20)' :
                        'rgba(255,255,255,0)';
    bands.push(`<div style="position:absolute;left:${left}%;width:${
        w}%;top:0;bottom:0;background:${col}"></div>`);
    if (n <= 60 || i % Math.ceil(n / 60) === 0) {
      marks.push(`<div style="position:absolute;left:${
          left}%;bottom:0;width:1px;height:6px;background:rgba(100,116,139,.45)"></div>`);
    }
    // Day boundary: a stronger rule where the sim resets everyone home.
    if (d && d.getHours() === 7 && d.getMinutes() === 0 && i > 0) {
      marks.push(`<div style="position:absolute;left:${
          left}%;top:0;bottom:0;width:1px;background:rgba(2,132,199,.5)"></div>`);
    }
  }
  $('tl-bands').innerHTML = bands.join('');
  $('tl-ticks').innerHTML = marks.join('');
}

function tlRender() {
  const n = S.ticks.length;
  const pct = n > 1 ? (S.idx / (n - 1)) * 100 : 0;
  $('tl-head').style.left = pct + '%';
  const tick = currentTick();
  const f = fmtSimTime(simTimeFor(tick));
  $('tl-time').textContent = f.full;
  // The scrubber can only span ticks the movement logs actually cover. If the
  // simulation clock ran past that, say so outright -- otherwise a run that
  // stopped logging at tick 29 of 112 reads as a complete 29-tick run.
  const lastLogged = n ? S.ticks[n - 1] : 0;
  const clockMax = S.clock && S.clock.max_ticks ? S.clock.max_ticks : 0;
  const el = $('tl-tick');
  if (!n) {
    el.textContent = '\u2014';
    el.title = '';
  } else if (clockMax > lastLogged) {
    el.textContent =
        `Tick ${tick} of ${lastLogged} logged \u00b7 clock ran to ${clockMax}`;
    el.title =
        `The movement logs for this run end at tick ${lastLogged}, but the ` +
        `simulation clock advanced to ${clockMax}. There is no position data ` +
        'to replay for the remainder, so the timeline stops where the ' +
        'evidence does.';
  } else {
    el.textContent = `Tick ${tick} of ${lastLogged}`;
    el.title = '';
  }
  track.setAttribute('aria-valuenow', String(tick));
  track.setAttribute('aria-valuemax', String(lastLogged));
  $('pill-clock').innerHTML = `&#128340; <b>${f.full}</b>`;
}

/* ------------------------------------------------------------------------ *
 * Data
 * ------------------------------------------------------------------------ */

async function loadAtlas(id) {
  const data = await getJSON('/api/atlas?g=' + encodeURIComponent(id));
  S.atlas = data;
  S.view.setBounds(data.meta.bounds);
  S.view.fit();
  S.agents = new AgentLayer(data);
  S.agents.showTrails = S.layers.trails;
  S.agents.showFlows = S.layers.flows;
  S.agents.showHeat = S.layers.heat;
  S.markers = new MarkerLayer($('layer-vector'), $('layer-labels'), data, {
    onPlaceClick: (place) => {
      S.markers.selectedPlace =
          S.markers.selectedPlace === place.id ? null : place.id;
      S.view.flyTo(
          place.xy[0], place.xy[1], Math.max(S.view.k, 3.6), 560, markDirty,
          markDirty);
    },
    onPlaceHover: (place, e) => {
      if (!place) {
        hideTip();
        return;
      }
      const occ = S.agents.occupancy().get(place.id) || [];
      const names = occ.slice(0, 6).map((s) => esc(s.name)).join(', ');
      showTip(
          `<b>${esc(place.name)}</b><br>` +
              `<span class="muted">${esc(categoryOf(place).label)}` +
              (place.district ? ` &middot; ${esc(districtName(place))}` : '') +
              `</span><br>` +
              (occ.length ? `${occ.length} here: ${names}${
                                occ.length > 6 ? '&hellip;' : ''}` :
                            '<span class="muted">empty</span>'),
          e);
    },
    onDistrictClick: (d) => flyToDistrict(d),
  });
  buildLegend();
  buildCategoryFilter();
  markDirty();
}

async function refresh() {
  let d;
  try {
    d = await getJSON('/api/data');
  } catch (e) {
    setStatus('bad', 'Offline');
    showError('Failed to load /api/data', e.message);
    return;
  }

  // The server reports per-query failures explicitly instead of returning an
  // empty list that would look like "no agents".
  if (d.error) {
    setStatus('bad', 'Query error');
    showError('Data query failed', d.error);
  } else if (d.partial_errors && d.partial_errors.length) {
    setStatus('warn', 'Degraded');
    showError('Some queries failed', d.partial_errors.join('\n\n'));
  } else {
    clearError();
  }

  $('pill-run-id').textContent = 'Run ' + (d.run_id || '\u2014');
  $('pill-agents').innerHTML = `&#128101; <b>${d.unique_agents || 0}</b>/${
      d.expected_agents || '?'} agents`;

  S.clock = d.sim_clock || null;

  if (!d.error) {
    const done = d.sim_clock && d.expected_ticks &&
        d.sim_clock.current_tick >= d.expected_ticks;
    setStatus(done ? 'ok' : 'live', done ? 'Complete' : 'Running');
  }

  // Grow the tick list from reported progress.
  const tp = d.tick_progress || [];
  if (tp.length) {
    const ticks = tp.map((t) => t.step).sort((a, b) => a - b);
    if (ticks.length !== S.ticks.length) {
      S.ticks = ticks;
      tlRenderTrack();
    }
  }

  if (d.agent_locations && d.agent_locations.length) {
    S.latest = d.agent_locations;
    if (S.ticks.length) {
      S.snapshots[String(S.ticks[S.ticks.length - 1])] = d.agent_locations;
    }
  }

  if (S.mode === 'live' && !S.playing) {
    if (S.ticks.length)
      tlGoTo(S.ticks.length - 1);
    else {
      S.agents.setSnapshot(S.latest, 0, lodFor(S.view.k), true);
      buildRoster();
      markDirty();
    }
  }

  updateDataQualityPills();
}

/**
 * Reports, in the topbar, every way the map is not a literal read of the data:
 * agents whose location could not be parsed at all, agents standing at an
 * atlas alias rather than a place this geography actually has, and agents
 * whose position is forward-filled from a log entry several ticks old.
 */
function updateDataQualityPills() {
  const unk = S.agents ? S.agents.unknownCount() : 0;
  const up = $('pill-unknown');
  if (unk > 0) {
    up.style.display = '';
    up.className = 'pill warn';
    up.textContent = `${unk} location${unk === 1 ? '' : 's'} not recorded`;
    const names = S.agents.unknownLocations();
    up.title = 'These agents had no parseable location on this tick. Their ' +
        'markers are drawn hollow and dashed rather than guessed at.' +
        (names.length ? '\n\nUnrecognised: ' + names.join(', ') : '');
  } else {
    up.style.display = 'none';
  }

  const approx = S.agents ? S.agents.approxCount() : 0;
  const ap = $('pill-approx');
  if (approx > 0) {
    const names = S.agents.approxLocations();
    ap.style.display = '';
    ap.className = 'pill warn';
    ap.textContent = `${approx} approximate`;
    ap.title = 'The simulation placed these agents at a venue ' +
        `${S.atlas.meta.display_name} does not have, so the atlas maps it to ` +
        'the nearest real place. Drawn with an amber dotted halo.' +
        (names.length ? '\n\nSubstituted: ' + names.join(', ') : '');
  } else {
    ap.style.display = 'none';
  }

  const stale = S.agents ? S.agents.staleCount() : {n: 0, max: 0};
  const sp = $('pill-stale');
  if (sp) {
    if (stale.n > 0) {
      sp.style.display = '';
      sp.className = 'pill warn';
      sp.textContent = `${stale.n} carried forward`;
      sp.title =
          'The movement logs stopped confirming these agents\u2019 locations. ' +
          'Their positions are held over from an earlier tick, not observed ' +
          `on this one (largest gap: ${stale.max} tick` +
          `${stale.max === 1 ? '' : 's'}). Drawn faded with a dashed outline.`;
    } else {
      sp.style.display = 'none';
    }
  }
}

async function loadHistory() {
  try {
    const h = await getJSON('/api/location_history');
    if (h.error) {
      showError('Location history unavailable', h.error);
      return;
    }
    if (h.ticks && h.ticks.length) {
      S.ticks = h.ticks;
      S.snapshots = Object.assign({}, h.snapshots, S.snapshots);
      S.historyLoaded = true;
      tlRenderTrack();
      tlGoTo(S.idx >= 0 ? S.idx : S.ticks.length - 1, false);
    }
  } catch (e) {
    showError('Failed to load /api/location_history', e.message);
  }
}

function setStatus(kind, text) {
  const p = $('pill-status');
  p.className = 'pill ' + kind;
  p.innerHTML =
      `<span class="dot${kind === 'live' ? ' pulse' : ''}"></span>${text}`;
}

/* ------------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------------ */

async function boot() {
  bindPanelToggles();
  buildLayerToggles();
  tlSetMode('live');

  try {
    S.config = await getJSON('/api/config');
  } catch (e) {
    showError('Failed to load /api/config', e.message);
    return;
  }

  const sel = $('geo-select');
  sel.innerHTML =
      S.config.geographies
          .map((g) => `<option value="${g.id}">${esc(g.display_name)}</option>`)
          .join('');
  sel.value = S.config.geography;
  sel.addEventListener('change', async () => {
    try {
      await loadAtlas(sel.value);
      // Re-seat the current snapshot into the new atlas's coordinate space.
      if (S.ticks.length)
        tlGoTo(S.idx >= 0 ? S.idx : S.ticks.length - 1, false);
      buildRoster();
      location.hash = 'g=' + sel.value;
    } catch (e) {
      showError('Failed to load atlas ' + sel.value, e.message);
    }
  });

  // The hash carries the whole view, not just the geography:
  //   #g=brecksville&k=3.2&c=516,384
  // so a particular reading of the map ("downtown at block zoom") is a link
  // you can send someone, and so the headless screenshot harness can frame a
  // shot without driving synthetic wheel events.
  const hash = location.hash || '';
  const hashGeo = /[#&]g=([a-z0-9_]+)/.exec(hash);
  const geo =
      (hashGeo && S.config.geographies.some((g) => g.id === hashGeo[1])) ?
      hashGeo[1] :
      S.config.geography;
  sel.value = geo;

  try {
    await loadAtlas(geo);
  } catch (e) {
    showError('Failed to load atlas ' + geo, e.message);
    return;
  }

  sizeCanvases();

  const hashK = /[#&]k=([0-9.]+)/.exec(hash);
  const hashC = /[#&]c=(-?[0-9.]+),(-?[0-9.]+)/.exec(hash);
  if (hashK || hashC) {
    if (hashK) {
      const k = parseFloat(hashK[1]);
      if (isFinite(k) && k > 0) S.view.k = k;
    }
    if (hashC) {
      S.view.centerOn(parseFloat(hashC[1]), parseFloat(hashC[2]));
    } else {
      S.view.clamp();
    }
    markDirty();
  }

  $('opt-daynight').addEventListener('change', (e) => {
    S.dayNight = e.target.checked;
    applyTint();
  });

  await refresh();
  loadHistory();

  // `#nopoll=1` freezes the data layer after the first fetch. Two uses: a
  // finished run does not need to be re-queried every ten seconds, and
  // headless capture tools fast-forward timers, so a page that polls forever
  // never reaches network idle and the screenshot never fires.
  if (!/[#&]nopoll=1/.test(hash)) {
    setInterval(refresh, 10000);
    setInterval(() => {
      if (!S.historyLoaded) loadHistory();
    }, 60000);
  }
}

window.addEventListener('resize', sizeCanvases);

/**
 * Debug handle.
 *
 * Exposing the live state is what makes the map testable: the headless
 * screenshot harness in ui/scratch drives zoom and centre through this rather
 * than synthesising wheel events, and it is the fastest way to inspect the
 * atlas from a console when a feature does not appear where you expect.
 */
window.__mapApp = {
  state: S,
  get view() {
    return S.view;
  },
  get atlas() {
    return S.atlas;
  },
  markDirty,
  lodFor,
};

boot();
