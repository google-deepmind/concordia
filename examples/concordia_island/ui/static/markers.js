/* disableFinding(Lint) */
/**
 * @fileoverview Place markers (SVG) and map labels (HTML).
 * @suppress {lintChecks}
 *
 * Markers are persistent keyed SVG elements that we reposition each frame
 * rather than re-serialising, so hover state, focus and CSS transitions
 * survive updates -- the previous dashboard rebuilt the whole map with
 * innerHTML on every poll, which destroyed all three.
 *
 * Labels live in an HTML overlay so text stays crisp and we can run a real
 * collision pass over measured boxes. Without culling, a 150-place map is an
 * unreadable pile of overlapping names at low zoom.
 */

import {categoryOf, lodFor, KEY_VENUES} from './atlas.js';

const SVGNS = 'http://www.w3.org/2000/svg';

function el(tag, attrs) {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}

/**
 * Teardrop pin, anchored at (0, 0) with its head centred at (0, -PIN_HEAD_Y).
 * Used at block zoom for the venues people actually go to, so a cafe reads as
 * a destination rather than as one more dot in a field of dots.
 */
const PIN_HEAD_Y = 19;
const PIN_PATH =
    'M 0,0 C -5.5,-8.5 -11,-13 -11,-19 A 11,11 0 1 1 11,-19 ' +
    'C 11,-13 5.5,-8.5 0,0 Z';

export class MarkerLayer {
  constructor(svg, labelHost, atlas, handlers) {
    this.svg = svg;
    this.labelHost = labelHost;
    this.handlers = handlers || {};
    this.nodes = new Map();      // place id -> {g, circle}
    this.districtNodes = new Map();
    this.hoverId = null;
    this.selectedPlace = null;
    this.filterCategory = null;
    this.setAtlas(atlas);
  }

  setAtlas(atlas) {
    this.atlas = atlas;
    this.svg.replaceChildren();
    this.labelHost.replaceChildren();
    this.nodes.clear();
    this.districtNodes.clear();

    this.gDistricts = el('g', {});
    this.gPlaces = el('g', {});
    this.svg.append(this.gDistricts, this.gPlaces);

    for (const p of atlas.places) {
      this.nodes.set(p.id, this._makePlaceNode(p));
    }
    for (const d of atlas.districts) {
      this.districtNodes.set(d.id, this._makeDistrictNode(d));
    }
  }

  _makePlaceNode(place) {
    const cat = categoryOf(place);
    const isKey = KEY_VENUES.has(place.category);
    const g = el('g', {class: 'marker', 'data-id': place.id, tabindex: '0',
                       role: 'button', 'aria-label': place.name});
    g.style.cursor = 'pointer';

    const halo = el('circle', {r: 15, fill: 'rgba(2,132,199,0)'});

    // Teardrop pin, shown at block zoom for key venues only.
    const pin = el('path', {
      d: PIN_PATH, fill: cat.color, stroke: '#ffffff', 'stroke-width': 2,
      'stroke-linejoin': 'round', style: 'display:none;',
    });

    // Google-style POI dot: the fill carries the category, a white rim lifts
    // it off the tan built-up wash, and the glyph is knocked out in white.
    // The previous white-disc-with-coloured-ring treatment vanished against
    // pale terrain and made every category look the same at a glance.
    const disc = el('circle', {
      r: 9, fill: cat.color, stroke: '#ffffff', 'stroke-width': 2,
    });
    const glyph = el('path', {
      d: cat.glyph, fill: 'none', stroke: '#ffffff', 'stroke-width': 2.4,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round',
      transform: 'translate(-6,-6) scale(0.5)',
    });

    // Enlarged emoji icon — an alternative to the glyph at interior zoom,
    // where there is room for it to read properly.
    const iconText = el('text', {
      'text-anchor': 'middle', 'dominant-baseline': 'central',
      'font-size': '18', y: '0.5',
      'font-family': '"Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif',
      style: 'display:none; pointer-events:none;',
    });
    iconText.textContent = cat.icon || '';

    // Count badge for occupied places.
    const badge = el('g', {class: 'badge', visibility: 'hidden'});
    const badgeBg = el('circle', {
      cx: 9, cy: -9, r: 7.5, fill: '#ef4444', stroke: '#fff', 'stroke-width': 1.6,
    });
    const badgeTx = el('text', {
      x: 9, y: -6.2, 'text-anchor': 'middle', fill: '#fff',
      'font-size': '9.5', 'font-weight': '700',
      'font-family': 'Inter, system-ui, sans-serif',
    });
    badge.append(badgeBg, badgeTx);

    g.append(halo, pin, disc, glyph, iconText, badge);

    const fire = (name) => (e) => {
      e.stopPropagation();
      const h = this.handlers[name];
      if (h) h(place, e);
    };
    g.addEventListener('click', fire('onPlaceClick'));
    g.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') fire('onPlaceClick')(e);
    });
    g.addEventListener('mouseenter', (e) => {
      this.hoverId = place.id;
      const h = this.handlers.onPlaceHover;
      if (h) h(place, e);
    });
    g.addEventListener('mouseleave', (e) => {
      this.hoverId = null;
      const h = this.handlers.onPlaceHover;
      if (h) h(null, e);
    });

    this.gPlaces.append(g);
    return {g, pin, disc, glyph, iconText, badge, badgeBg, badgeTx, place, cat,
            isKey};
  }

  /**
   * District marker: a small solid count chip, nothing more.
   *
   * The previous design drew a 22-54px translucent ring in the district
   * colour around every neighbourhood. At region zoom that is eleven coloured
   * bullseyes over a map -- it hides the geography it is supposed to
   * annotate, and it looks nothing like how a real map labels a
   * neighbourhood. A real map gives you a name and, if it has a number to
   * report, a small chip. The built-up tan now shows you where the
   * neighbourhood physically *is*.
   */
  _makeDistrictNode(dist) {
    const g = el('g', {class: 'district-marker', 'data-id': dist.id,
                       tabindex: '0', role: 'button', 'aria-label': dist.name});
    g.style.cursor = 'pointer';
    const chip = el('circle', {
      r: 13, fill: dist.color || '#64748b', stroke: '#ffffff',
      'stroke-width': 2.2,
    });
    const tx = el('text', {
      'text-anchor': 'middle', y: 4.4, fill: '#ffffff',
      'font-size': '13', 'font-weight': '700',
      'font-family': 'Inter, system-ui, sans-serif',
      style: 'pointer-events:none;',
    });
    g.append(chip, tx);
    g.addEventListener('click', (e) => {
      e.stopPropagation();
      const h = this.handlers.onDistrictClick;
      if (h) h(dist, e);
    });
    this.gDistricts.append(g);
    return {g, chip, tx, dist};
  }

  /**
   * Repositions and restyles everything for the current view.
   * @param {!Object} view
   * @param {!Map<string, !Array>} occupancy place id -> sprites
   * @param {!Map<string, !Array>} districtOcc district id -> sprites
   */
  update(view, occupancy, districtOcc) {
    const lod = lodFor(view.k);
    const showDistricts = lod <= 1;
    const showPlaces = lod >= 1;

    this.gDistricts.style.display = showDistricts ? '' : 'none';
    this.gPlaces.style.display = showPlaces ? '' : 'none';

    if (showDistricts) this._updateDistricts(view, districtOcc, lod);
    if (showPlaces) this._updatePlaces(view, occupancy, lod);

    this._updateLabels(view, occupancy, districtOcc, lod);
  }

  _updateDistricts(view, districtOcc, lod) {
    for (const [id, node] of this.districtNodes) {
      const d = node.dist;
      const anchor = d.label_anchor || [0, 0];
      const [x, y] = view.toScreen(anchor[0], anchor[1]);
      const list = districtOcc.get(id) || [];
      const n = list.length;
      node.g.setAttribute('transform', `translate(${x},${y})`);

      // An empty neighbourhood gets no chip at all. Drawing a grey zero over
      // every district is how the map ends up looking like a dashboard
      // instead of a map; the name label still tells you it is there.
      if (n === 0) {
        node.g.style.display = 'none';
        continue;
      }

      // The chip grows only slightly with the count -- enough to register
      // relative busyness, not enough to swallow the terrain underneath.
      const r = Math.max(12, Math.min(20, 11 + Math.sqrt(n) * 1.9));
      node.chip.setAttribute('r', r);
      node.tx.textContent = n > 999 ? '999+' : String(n);
      node.tx.setAttribute('font-size',
                           String(Math.max(11, Math.min(15, r * 0.95))));
      node.tx.setAttribute('y', String(r * 0.34));
      node.g.style.opacity = 1;
      node.g.style.display =
          (x < -80 || y < -80 || x > view.w + 80 || y > view.h + 80) ? 'none' : '';
    }
  }

  _updatePlaces(view, occupancy, lod) {
    const k = view.k;
    for (const [id, node] of this.nodes) {
      const p = node.place;

      // Per-feature LOD: minor places only appear once zoomed in, and children
      // (office floors, common rooms) only at the interior tier.
      const minZoom = p.min_zoom !== undefined ? p.min_zoom : node.cat.minZoom;
      let visible = k >= zoomThresholdFor(minZoom);
      if (p.parent && lod < 4) visible = false;
      if (p.virtual && lod >= 4) visible = false;
      if (this.filterCategory && p.category !== this.filterCategory) visible = false;

      const [x, y] = view.toScreen(p.xy[0], p.xy[1]);
      if (x < -60 || y < -60 || x > view.w + 60 || y > view.h + 60) visible = false;

      node.g.style.display = visible ? '' : 'none';
      if (!visible) continue;

      const occ = occupancy.get(id);
      const n = occ ? occ.length : 0;
      const isSel = this.selectedPlace === id;

      node.g.setAttribute('transform', `translate(${x},${y})`);

      // Three stages, matching how a real map escalates a point of interest:
      //
      //   district zoom  a plain coloured dot -- "something is here"
      //   street zoom    a POI dot with the category glyph knocked out
      //   block zoom     key venues become a teardrop pin with an emoji, so
      //                  the cafe you are looking for is findable at a glance
      //
      // Only key venues get promoted to a pin. Promoting everything would
      // just reproduce the old wall-of-identical-markers at a larger size.
      const usePin = lod >= 3 && node.isKey;
      const useEmoji = lod >= 3 && !!node.cat.icon;
      const tiny = lod <= 1;

      if (usePin) {
        node.pin.style.display = '';
        node.disc.style.display = 'none';
        const scale = isSel ? 1.18 : 1;
        node.pin.setAttribute(
            'transform', scale === 1 ? '' : `scale(${scale})`);
        node.pin.setAttribute('stroke', isSel ? '#0f172a' : '#ffffff');
        node.pin.setAttribute('stroke-width', isSel ? 2.6 : 2);
        const hy = -PIN_HEAD_Y * scale;
        if (useEmoji) {
          node.glyph.style.display = 'none';
          node.iconText.style.display = '';
          node.iconText.setAttribute('font-size', String(13 * scale));
          node.iconText.setAttribute('y', String(hy + 0.5));
        } else {
          node.glyph.style.display = '';
          node.iconText.style.display = 'none';
          node.glyph.setAttribute(
              'transform', `translate(-7.2,${hy - 7.2}) scale(0.6)`);
        }
        node.badge.setAttribute('transform', `translate(6,${hy + 8})`);
        node.badgeBg.setAttribute('cx', 4);
        node.badgeBg.setAttribute('cy', -4);
      } else {
        node.pin.style.display = 'none';
        node.disc.style.display = '';

        // Occupied places read a little larger; that is the one piece of
        // live data the marker itself is allowed to encode.
        const r = tiny ? (n > 0 ? 7 : 5)
                       : (lod >= 3 ? (n > 0 ? 14 : 12) : (n > 0 ? 10 : 8));
        node.disc.setAttribute('r', r);
        node.disc.setAttribute('fill', node.cat.color);
        node.disc.setAttribute('stroke', isSel ? '#0f172a' : '#ffffff');
        node.disc.setAttribute('stroke-width', isSel ? 3 : 2);

        // Below the street tier there is no room for a glyph inside a 5px
        // dot; showing one just muddies the colour that carries the category.
        const showGlyph = !tiny && r >= 7;
        if (useEmoji && r >= 11) {
          node.glyph.style.display = 'none';
          node.iconText.style.display = '';
          node.iconText.setAttribute('font-size', String(r * 1.15));
          node.iconText.setAttribute('y', '0.5');
        } else if (showGlyph) {
          node.glyph.style.display = '';
          node.iconText.style.display = 'none';
          const gs = (r * 2 * 0.62) / 24;
          node.glyph.setAttribute(
              'transform', `translate(${-12 * gs},${-12 * gs}) scale(${gs})`);
        } else {
          node.glyph.style.display = 'none';
          node.iconText.style.display = 'none';
        }
        node.badge.setAttribute('transform', '');
        node.badgeBg.setAttribute('cx', r - 1);
        node.badgeBg.setAttribute('cy', -(r - 1));
      }

      // Unoccupied venues recede but stay legible: a map that hides its
      // empty places is useless for asking "where could they have gone".
      node.g.style.opacity = n > 0 ? 1 : 0.55;

      if (n > 0 && !tiny) {
        node.badge.setAttribute('visibility', 'visible');
        node.badgeTx.textContent = n > 99 ? '99+' : String(n);
        node.badgeBg.setAttribute('r', n > 9 ? 8.5 : 7.5);
      } else {
        node.badge.setAttribute('visibility', 'hidden');
      }
    }
  }

  /* -------------------------------------------------------------------- *
   * Labels with greedy collision culling
   * -------------------------------------------------------------------- */

  _updateLabels(view, occupancy, districtOcc, lod) {
    const host = this.labelHost;
    const wanted = [];

    if (lod <= 1) {
      for (const d of this.atlas.districts) {
        const a = d.label_anchor;
        if (!a) continue;
        const [x, y] = view.toScreen(a[0], a[1]);
        // Clears the count chip, which is at most ~20px in radius. The old
        // 42px offset was sized for the ring that no longer exists and left
        // the name floating in the middle of nowhere.
        const occupied = (districtOcc.get(d.id) || []).length > 0;
        wanted.push({
          key: 'd:' + d.id, text: d.name, cls: 'district',
          x, y: y + (occupied ? 27 : 0), prio: 100,
          w: d.name.length * 7.6 + 8, h: 14,
        });
      }
    }

    if (lod >= 1) {
      for (const p of this.atlas.places) {
        if (p.parent && lod < 4) continue;
        if (p.virtual && lod >= 4) continue;
        if (this.filterCategory && p.category !== this.filterCategory) continue;
        const minZoom = p.min_zoom !== undefined ? p.min_zoom : categoryOf(p).minZoom;
        if (view.k < zoomThresholdFor(minZoom)) continue;
        const [x, y] = view.toScreen(p.xy[0], p.xy[1]);
        if (x < -40 || y < -40 || x > view.w + 40 || y > view.h + 40) continue;
        const occ = occupancy.get(p.id);
        const n = occ ? occ.length : 0;
        const major = minZoom <= 0;
        const cat = categoryOf(p);
        // A pin is anchored at the point and rises above it, so its label
        // needs far less clearance than a disc centred on the same point.
        const pinned = lod >= 3 && KEY_VENUES.has(p.category);
        const yOff = pinned ? 10 : (lod >= 3 ? 20 : 15);
        const text = lod >= 3 ? p.name + '\n' + cat.label : p.name;
        wanted.push({
          key: 'p:' + p.id, text,
          cls: 'place' + (major ? ' major' : '') + (lod >= 3 ? ' detailed' : ''),
          x, y: y + yOff, prio: (major ? 50 : 10) + Math.min(20, n) +
              (pinned ? 15 : 0),
          w: p.name.length * (major ? 7.1 : 6.4) + 10, h: lod >= 3 ? 28 : 14,
        });
      }
    }

    // Greedy placement, most important first, into a coarse spatial grid.
    wanted.sort((a, b) => b.prio - a.prio);
    const CELL = 40;
    const grid = new Map();
    const placed = [];
    for (const lab of wanted) {
      const x0 = lab.x - lab.w / 2, x1 = lab.x + lab.w / 2;
      const y0 = lab.y - lab.h / 2, y1 = lab.y + lab.h / 2;
      let clash = false;
      const cx0 = Math.floor(x0 / CELL), cx1 = Math.floor(x1 / CELL);
      const cy0 = Math.floor(y0 / CELL), cy1 = Math.floor(y1 / CELL);
      outer:
      for (let cy = cy0; cy <= cy1 && !clash; cy++) {
        for (let cx = cx0; cx <= cx1; cx++) {
          const bucket = grid.get(cx + ':' + cy);
          if (!bucket) continue;
          for (const o of bucket) {
            if (x0 < o.x1 && x1 > o.x0 && y0 < o.y1 && y1 > o.y0) {
              clash = true;
              break outer;
            }
          }
        }
      }
      if (clash) continue;
      const box = {x0, y0, x1, y1};
      for (let cy = cy0; cy <= cy1; cy++) {
        for (let cx = cx0; cx <= cx1; cx++) {
          const key = cx + ':' + cy;
          if (!grid.has(key)) grid.set(key, []);
          grid.get(key).push(box);
        }
      }
      placed.push(lab);
    }

    // Keyed DOM reuse so we are not churning nodes every frame.
    const live = new Set();
    for (const lab of placed) {
      live.add(lab.key);
      let n = host.querySelector(`[data-k="${CSS.escape(lab.key)}"]`);
      if (!n) {
        n = document.createElement('div');
        n.dataset.k = lab.key;
        host.append(n);
      }
      if (n.className !== 'map-label ' + lab.cls) {
        n.className = 'map-label ' + lab.cls;
      }
      // Multi-line labels (name + category) use innerHTML for the sub-label.
      const lines = lab.text.split('\n');
      const desired = lines.length > 1
          ? lines[0] + '<span class="sub-label">' + lines[1] + '</span>'
          : lines[0];
      if (n.innerHTML !== desired) n.innerHTML = desired;
      n.style.left = lab.x + 'px';
      n.style.top = lab.y + 'px';
    }
    for (const n of [...host.children]) {
      if (!live.has(n.dataset.k)) n.remove();
    }
  }
}

/**
 * Maps an atlas `min_zoom` tier (0..4) onto a View.k threshold.
 * Kept in one place so the tier boundaries in lodFor() and the per-feature
 * thresholds cannot drift apart.
 */
function zoomThresholdFor(tier) {
  switch (tier) {
    case 0: return 0;
    case 1: return 1.15;
    case 2: return 2.0;
    case 3: return 3.4;
    default: return 6.0;
  }
}
