/**
 * @license
 * Copyright 2026 DeepMind Technologies Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @fileoverview The Lifeways page: draws the valley and the census, shows a
 * person's customs, and sends the Experimenter's moves. It only reads the
 * world and sends what the person chooses; the server runs the simulation.
 */

'use strict';

/** Terrain colours, by kind. */
const TERRAIN_COLOURS = {
  grass: '#9bc472',
  forest: '#4f7d3f',
  river: '#5fa8d3',
  coast: '#8fc9e8',
  hills: '#a89077',
  marsh: '#7f9c6a',
};

/** Glyphs of places, by kind. */
const PLACE_GLYPHS = {library: 'L', school: 'S', shrine: '✦'};

/** The page's state. */
const page = {
  /** @type {?Object} */ world: null,
  /** @type {!Array<!Object>} */ rows: [],
  /** @type {!Array<!Object>} The twin's census, if there is a twin. */
  twinRows: [],
  twinSince: -1,
  /** @type {?string} The twin drawn: its lever, value and year. */
  twin: null,
  /** @type {!Array<!Object>} The notebook's entries, oldest first. */
  entries: [],
  entrySince: 0,
  /** @type {?string} The person shown, if any. */ person: null,
  since: -1,
  tile: 1,
};

/**
 * Fetches JSON from the server (relative to the page).
 * @param {string} path The path under the page's base.
 * @param {?Object=} body A JSON body to POST, if any.
 * @return {!Promise<?>} The response's JSON.
 */
async function api(path, body = null) {
  const response = await fetch(path, body === null ? {} : {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let reason = '';
    try {
      reason = (await response.json()).error || '';
    } catch (error) {
      // Not JSON: say the status.
    }
    throw new Error(reason || `${path}: ${response.status}`);
  }
  return response.json();
}

/**
 * A colour for a people, the same every time.
 * @param {string} name The people's name.
 * @return {string} An HSL colour.
 */
function peopleColour(name) {
  let hash = 0;
  for (const char of name) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return `hsl(${hash % 360}, 65%, 40%)`;
}

/**
 * Draws the valley: terrain shaded by food, places, people.
 * @param {!Object} world The server's state.
 */
function drawMap(world) {
  const canvas = /** @type {!HTMLCanvasElement} */ (
      document.getElementById('map'));
  const width = canvas.clientWidth || 480;
  page.tile = Math.max(4, Math.floor(width / world.cols));
  canvas.width = world.cols * page.tile;
  canvas.height = world.rows * page.tile;
  const ctx = canvas.getContext('2d');
  const most = Math.max(1, ...world.food.flat());
  for (let r = 0; r < world.rows; r++) {
    for (let c = 0; c < world.cols; c++) {
      ctx.fillStyle = TERRAIN_COLOURS[world.kinds[world.terrain[r][c]]];
      ctx.fillRect(c * page.tile, r * page.tile, page.tile, page.tile);
      ctx.fillStyle = `rgba(40, 30, 20, ${0.45 * (1 - world.food[r][c] / most)})`;
      ctx.fillRect(c * page.tile, r * page.tile, page.tile, page.tile);
    }
  }
  ctx.font = `${Math.floor(page.tile * 0.8)}px system-ui`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const place of world.places) {
    const [r, c] = place.tile;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.fillRect(c * page.tile, r * page.tile, page.tile, page.tile);
    ctx.fillStyle = '#222';
    ctx.fillText(PLACE_GLYPHS[place.kind] || '?', (c + 0.5) * page.tile,
                 (r + 0.5) * page.tile);
  }
  const crowd = {};
  for (const person of world.people) {
    const key = person.tile.join(',');
    const n = (crowd[key] = (crowd[key] || 0) + 1) - 1;
    const angle = n * 2.4;
    const spread = Math.min(0.35, 0.12 * Math.sqrt(n));
    person.x = (person.tile[1] + 0.5 + spread * Math.cos(angle)) * page.tile;
    person.y = (person.tile[0] + 0.5 + spread * Math.sin(angle)) * page.tile;
    ctx.beginPath();
    ctx.arc(person.x, person.y, person.age < world.grown_age ? page.tile * 0.14 :
            page.tile * 0.22, 0, 2 * Math.PI);
    ctx.fillStyle = peopleColour(person.descent || 'valley');
    ctx.fill();
  }
}

/**
 * Draws the census: the living and the mean skill, year by year; a twin's
 * census, if any, dashed over its world's, on the same scales.
 * @param {!Array<!Object>} rows The census rows.
 * @param {!Array<!Object>=} twinRows The twin's rows after its fork.
 */
function drawChart(rows, twinRows = []) {
  const canvas = /** @type {!HTMLCanvasElement} */ (
      document.getElementById('chart'));
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (rows.length < 2) return;
  const both = rows.concat(twinRows);
  const series = [
    {key: 'population', colour: '#2d6a4f', most: Math.max(...both.map(
        (row) => row.population), 1)},
    {key: 'skill_worked', colour: '#bc6c25', most: page.world.max_level},
    {key: 'tongues', colour: '#5a189a', most: Math.max(...both.map(
        (row) => row.tongues || 0), 1)},
  ];
  const first = rows[0].year;
  const lastYear = Math.max(...both.map((row) => row.year));
  const x = (year) => 4 + ((year - first) / Math.max(1, lastYear - first)) *
      (canvas.width - 8);
  for (const [drawn, dash] of [[rows, []], [twinRows, [5, 4]]]) {
    for (const {key, colour, most} of series) {
      ctx.beginPath();
      drawn.forEach((row, i) => {
        const y = canvas.height - 4 - ((row[key] || 0) / most) *
            (canvas.height - 8);
        if (i === 0) ctx.moveTo(x(row.year), y); else ctx.lineTo(x(row.year), y);
      });
      ctx.setLineDash(dash);
      ctx.strokeStyle = colour;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
  ctx.setLineDash([]);
  const last = rows[rows.length - 1];
  document.getElementById('legend').textContent =
      `Green: the living (${last.population}). Brown: the skill people work` +
      ` by (${last.skill_worked}). Purple: tongues spoken (${last.tongues}).` +
      ` Years ${rows[0].year}–${last.year}.` +
      (twinRows.length ? ' Dashed: the twin.' : '');
  const channels = Object.entries(last.held.by_channel)
      .map(([channel, n]) => `${channel} ${n}`).join(', ');
  document.getElementById('channels').textContent =
      `Customs held, by how they came: ${channels}. Peoples: ` +
      `${last.peoples.peoples}.`;
}

/**
 * The Experimenter's release move, in the form the World reads.
 * @param {string} kind The custom's kind.
 * @param {string} domain Its domain.
 * @param {string} how 'song' or 'text'.
 * @param {string} at The place, 'row,col'.
 * @param {string} says What the custom says.
 * @return {string} The move.
 */
function releaseMove(kind, domain, how, at, says) {
  return `release ${kind} ${domain} as a ${how} at ` +
      `${at.replace(/\s/g, '')}: ${says.trim()}`;
}

/**
 * What the page says of a twin.
 * @param {?Object} twin The state's twin, or null.
 * @return {string} The line.
 */
function twinNote(twin) {
  if (!twin) return 'No twin.';
  const pulled = twin.lever === 'without' ?
      `without ${twin.value}` : `${twin.lever} ${twin.value}`;
  return `The twin (${pulled}, forked in year ${twin.year}) is in year ` +
      `${twin.year_now}: ${twin.people} people` +
      (twin.running ? '.' : ' (stopped).') +
      (twin.error ? ` ${twin.error}` : '');
}

/**
 * The lever's choices for the fork form: each lever, then each channel the
 * world could go without.
 * @param {!Object} world The state.
 * @return {!Array<!Array<string>>} [value, label] pairs.
 */
function leverChoices(world) {
  return Object.entries(world.levers)
      .map(([lever, [low, high, now]]) =>
        [lever, `${lever} (${low}–${high}; now ${now})`])
      .concat(world.channels.map((channel) =>
        [`without ${channel}`, `without ${channel}`]));
}

/** What the person writes in the notebook. */
const NOTE_KINDS = ['hypothesis', 'observation', 'note'];

/**
 * Whether the notebook shows an entry under a filter.
 * @param {!Object} entry An entry.
 * @param {string} filter 'all', 'mine', 'moves' or 'events'.
 * @return {boolean}
 */
function entryShown(entry, filter) {
  if (filter === 'mine') return NOTE_KINDS.includes(entry.kind);
  if (filter === 'moves') return ['move', 'world'].includes(entry.kind);
  if (filter === 'events') return entry.kind === 'event';
  return true;
}

/**
 * What the notebook says above an entry.
 * @param {!Object} entry An entry.
 * @return {string} E.g. 'Year 40 · hypothesis'.
 */
function entryMeta(entry) {
  const kind = {move: 'your move', world: 'the World', event: 'the valley'};
  return `Year ${entry.year} · ${kind[entry.kind] || entry.kind}`;
}

/** Lists the notebook, newest first, as the filter allows. */
function drawNotebook() {
  const filter = document.getElementById('note-filter').value;
  const shown = page.entries.filter((e) => entryShown(e, filter))
      .slice(-100).reverse();
  document.getElementById('entries').replaceChildren(...shown.map((entry) => {
    const item = document.createElement('li');
    if (NOTE_KINDS.includes(entry.kind)) item.className = 'mine';
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = entryMeta(entry);
    const text = document.createElement('div');
    text.textContent = entry.text;
    item.append(meta, text);
    return item;
  }));
}

/**
 * The Experimenter's question to a person, in the form the World reads.
 * @param {string} name The person's name.
 * @param {string} question The question.
 * @return {string} The move.
 */
function askMove(name, question) {
  return `ask ${name}: ${question.trim()}`;
}

/**
 * Lists the latest interviews, the newest first.
 * @param {!Object} world The server's state.
 */
function drawTranscripts(world) {
  const pending = Object.entries(world.interviews).map(([name, question]) => {
    const item = document.createElement('li');
    item.textContent = `${name} is asked: "${question}" (answers this year)`;
    return item;
  });
  const answered = [...world.transcripts].reverse().map((transcript) => {
    const item = document.createElement('li');
    item.textContent = `"${transcript.answer}"`;
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = `${transcript.name}, year ${transcript.year}, asked` +
        ` "${transcript.question}"`;
    item.append(meta);
    return item;
  });
  document.getElementById('transcripts').replaceChildren(
      ...pending, ...answered);
}

/**
 * Shows a person's customs.
 * @param {string} name The person's name.
 */
async function showPerson(name) {
  const [person, kin] = await Promise.all([
    api(`api/person?name=${encodeURIComponent(name)}`),
    api(`api/kin?name=${encodeURIComponent(name)}`),
  ]);
  document.getElementById('person-name').textContent = person.name;
  const age = page.world.year - person.born;
  document.getElementById('person-body').textContent =
      `${age} years old, of the ${person.descent || 'valley'}; food ` +
      `${person.food}.`;
  document.getElementById('person-kin').replaceChildren(...KIN.map(
      ([key, label]) => kinRow(label, [].concat(kin[key] || []))));
  const list = document.getElementById('person-customs');
  list.replaceChildren(...person.customs.map((custom) => {
    const item = document.createElement('li');
    item.textContent = custom.text;
    const meta = document.createElement('div');
    meta.className = 'meta';
    const detail = Object.entries(custom.params)
        .filter(([key]) => key !== 'words')
        .map(([key, value]) => `${key} ${value}`).join(', ');
    meta.textContent = `${custom.kind}, ${custom.domain}` +
        (detail ? `, ${detail}` : '') + ` · ${cameBy(custom)} · `;
    const lineage = document.createElement('button');
    lineage.type = 'button';
    lineage.className = 'link';
    lineage.textContent = `lineage of ${custom.id}`;
    lineage.addEventListener('click', () => showLineage(custom.id));
    meta.append(lineage);
    item.append(meta);
    return item;
  }));
  document.getElementById('person').hidden = false;
  page.person = person.name;
}

/** The family the inspector shows: [key in the kin reply, label]. */
const KIN = [
  ['grandparents', 'Grandparents'],
  ['parents', 'Parents'],
  ['partner', 'Partner'],
  ['siblings', 'Brothers and sisters'],
  ['children', 'Children'],
  ['grandchildren', 'Grandchildren'],
];

/**
 * How a person came by a custom, and from whom.
 * @param {!Object} custom A custom of the inspector's person.
 * @return {string} E.g. 'came by observation from Mara'.
 */
function cameBy(custom) {
  return `came by ${custom.via || 'birth'}` +
      (custom.from ? ` from ${custom.from}` : '');
}

/**
 * A kin's name as the family list shows it: the dead with when they died.
 * @param {!Object} kin An entry of the kin reply.
 * @return {string} E.g. 'Vira', 'Tone (died 63, aged 59)', 'Elor (forgotten)'.
 */
function kinName(kin) {
  if (kin.living) return kin.name;
  if (!kin.known) return `${kin.name} (forgotten)`;
  return `${kin.name} (died ${kin.died}, aged ${kin.died - kin.born})`;
}

/**
 * What a lineage says of one variant: its id, how and when it arose, who
 * holds it.
 * @param {!Object} variant A variant of the lineage reply.
 * @return {string} E.g. 'c52 · mutation, year 9 · held by 3'.
 */
function variantNote(variant) {
  const held = !variant.living ? 'forgotten: nobody holds it now' :
      variant.holders.length ? `held by ${variant.holders.length}` :
      'held by nobody, kept in a place';
  return `${variant.id} · ${variant.origin}, year ${variant.year} · ${held}`;
}

/**
 * One line of the family: the living are links to their own inspector.
 * @param {string} label What they are.
 * @param {!Array<!Object>} kin Their entries.
 * @return {!HTMLElement} The line.
 */
function kinRow(label, kin) {
  const row = document.createElement('p');
  row.append(`${label}: `);
  if (!kin.length) row.append('none');
  kin.forEach((one, i) => {
    if (i) row.append(', ');
    if (one.living) {
      const link = document.createElement('button');
      link.type = 'button';
      link.className = 'link';
      link.textContent = one.name;
      link.addEventListener('click', () => showPerson(one.name));
      row.append(link);
    } else {
      row.append(kinName(one));
    }
  });
  return row;
}

/**
 * Nests a variant's holders under those they had it from (who learned from
 * whom); a holder whose source holds it no more, or is a place, is a root.
 * @param {!Array<!Object>} holders {name, via, from} of each holder.
 * @return {!Array<!Object>} The roots, each with its `learners`.
 */
function transmissionTree(holders) {
  const byName = new Map(holders.map(
      (h) => [h.name, {...h, learners: []}]));
  const roots = [];
  for (const node of byName.values()) {
    const teacher = node.from && node.from !== node.name ?
        byName.get(node.from) : undefined;
    if (teacher && !isUnder(teacher, node)) {
      teacher.learners.push(node);
    } else {
      roots.push(node);
    }
  }
  return roots;
}

/**
 * Whether `node` is `under` or among its learners (no cycles in the tree).
 * @param {!Object} node A holder.
 * @param {!Object} under Another.
 * @return {boolean}
 */
function isUnder(node, under) {
  if (node === under) return true;
  return under.learners.some((learner) => isUnder(node, learner));
}

/**
 * Writes a transmission tree as text: 'Mara > Ada (observation), Bo'.
 * @param {!Array<!Object>} roots The tree's roots.
 * @return {string} The text.
 */
function transmissionText(roots) {
  return roots.map((node) => {
    const how = node.from ? `${node.via} from ${node.from}` : node.via;
    const own = `${node.name} (${how})`;
    return node.learners.length ?
        `${own} > [${transmissionText(node.learners)}]` : own;
  }).join(', ');
}

/**
 * Shows a custom's lineage: its variants, who holds each and from whom.
 * @param {string} id A variant's id.
 */
async function showLineage(id) {
  const lineage = await api(`api/lineage?id=${encodeURIComponent(id)}`);
  document.getElementById('lineage-title').textContent =
      `The lineage of ${id}` +
      (lineage.lost_parent ? ` (its ancestor ${lineage.lost_parent} is ` +
        'forgotten)' : '');
  const items = lineage.variants.map((variant) => {
    const item = document.createElement('li');
    item.style.marginLeft = `${Math.min(variant.depth, 8) * 12}px`;
    const text = document.createElement('div');
    text.textContent = variant.text;
    if (variant.id === id) text.style.fontWeight = '600';
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = variantNote(variant);
    const holders = document.createElement('div');
    holders.className = 'holders';
    holders.textContent =
        transmissionText(transmissionTree(variant.holders));
    item.append(text, meta, holders);
    return item;
  });
  if (lineage.more) {
    const more = document.createElement('li');
    more.textContent = `…and ${lineage.more} more variants.`;
    items.push(more);
  }
  document.getElementById('lineage-variants').replaceChildren(...items);
  const panel = document.getElementById('lineage');
  panel.hidden = false;
  panel.scrollIntoView({behavior: 'smooth', block: 'start'});
}

/**
 * Handles a tap on the map: a person, or the place for a release.
 * @param {!MouseEvent} event The click.
 */
function onMapTap(event) {
  if (!page.world) return;
  const canvas = event.currentTarget;
  const scale = canvas.width / canvas.clientWidth;
  const x = event.offsetX * scale;
  const y = event.offsetY * scale;
  const row = Math.floor(y / page.tile);
  const col = Math.floor(x / page.tile);
  document.getElementById('at').value = `${row},${col}`;
  let nearest = null;
  let best = page.tile * page.tile;
  for (const person of page.world.people) {
    const d = (person.x - x) ** 2 + (person.y - y) ** 2;
    if (d < best) {
      best = d;
      nearest = person;
    }
  }
  if (nearest) showPerson(nearest.name);
}

/**
 * What the footer says of the world's saves.
 * @param {!Object} world The state.
 * @return {string} The note.
 */
function savedNote(world) {
  const notes = [];
  if (world.restored_year != null) {
    notes.push(`restored at year ${world.restored_year}`);
  }
  if (world.saved_year != null) notes.push(`saved at year ${world.saved_year}`);
  return notes.length ? `· ${notes.join(', ')}` : '';
}

/** Polls the world and the census, and redraws. */
async function refresh() {
  try {
    const world = await api('api/state');
    page.world = world;
    document.getElementById('year').textContent =
        `${world.scenario}, year ${world.year}: ${world.people.length} people` +
        (world.running ? '' : ' (the world has stopped)');
    document.getElementById('play').textContent =
        world.pace.paused ? 'Play' : 'Pause';
    document.getElementById('queued').textContent = world.pace.queued.length ?
        `Next: ${world.pace.queued.join('; ')}` : '';
    document.getElementById('error').textContent = world.error || '';
    document.getElementById('saved').textContent = savedNote(world);
    drawMap(world);
    drawTranscripts(world);
    document.getElementById('twin').textContent = twinNote(world.twin);
    document.getElementById('unfork').hidden = !world.twin;
    const twin = world.twin ?
        `${world.twin.lever} ${world.twin.value} ${world.twin.year}` : null;
    if (twin !== page.twin) {
      page.twin = twin;
      page.twinRows = [];
      // The twin's own rows begin with the year it was forked in.
      page.twinSince = world.twin ? world.twin.year - 1 : -1;
    }
    const rows = await api(`api/census?since=${page.since}`);
    const twinRows = world.twin ?
        await api(`api/census?since=${page.twinSince}&world=twin`) : [];
    if (twinRows.length) {
      page.twinRows = page.twinRows.concat(twinRows).slice(-world.rows_kept);
      page.twinSince = twinRows[twinRows.length - 1].year;
    }
    if (rows.length) {
      page.rows = page.rows.concat(rows).slice(-world.rows_kept);
      page.since = rows[rows.length - 1].year;
    }
    if (rows.length || twinRows.length) drawChart(page.rows, page.twinRows);
    const entries = await api(`api/notebook?since=${page.entrySince}`);
    if (entries.length) {
      page.entries = page.entries.concat(entries).slice(-1000);
      page.entrySince = entries[entries.length - 1].id;
      drawNotebook();
    }
  } catch (error) {
    document.getElementById('error').textContent = String(error);
  }
}

/**
 * Fills a select with options.
 * @param {string} id The select's id.
 * @param {!Array<string>} values The options.
 */
function fill(id, values) {
  const select = document.getElementById(id);
  select.replaceChildren(...values.map((value) => new Option(value, value)));
}

/** Wires the page. */
async function start() {
  const world = await api('api/state');
  fill('kind', world.custom_kinds);
  fill('domain', world.domains);
  document.getElementById('lever').replaceChildren(...leverChoices(world).map(
      ([value, label]) => new Option(label, value)));
  const build = await api('api/build');
  document.getElementById('build').textContent =
      `build ${build.commit || '?'}`;
  document.getElementById('map').addEventListener('click', onMapTap);
  document.getElementById('person-close').addEventListener('click', () => {
    document.getElementById('person').hidden = true;
  });
  document.getElementById('lineage-close').addEventListener('click', () => {
    document.getElementById('lineage').hidden = true;
  });
  document.getElementById('play').addEventListener('click', async () => {
    await api('api/pace', {paused: !page.world.pace.paused});
    refresh();
  });
  document.getElementById('speed').addEventListener('change', async (event) => {
    await api('api/pace', {years_per_second: Number(event.target.value)});
  });
  document.getElementById('stop').addEventListener('click', async () => {
    if (confirm('Stop the world for good?')) await api('api/stop', {});
  });
  document.getElementById('ask').addEventListener('submit', async (event) => {
    event.preventDefault();
    const question = document.getElementById('question').value.trim();
    if (!question || !page.person) return;
    await api('api/move', {move: askMove(page.person, question)});
    document.getElementById('question').value = '';
    refresh();
  });
  document.getElementById('release').addEventListener('submit', async (event) => {
    event.preventDefault();
    const says = document.getElementById('says').value.trim();
    if (!says) return;
    const move = releaseMove(
        document.getElementById('kind').value,
        document.getElementById('domain').value,
        document.getElementById('how').value,
        document.getElementById('at').value, says);
    await api('api/move', {move});
    document.getElementById('says').value = '';
    refresh();
  });
  document.getElementById('fork').addEventListener('submit', async (event) => {
    event.preventDefault();
    const lever = document.getElementById('lever').value;
    const value = document.getElementById('lever-value').value.trim();
    const [name, channel] = lever.startsWith('without ') ?
        ['without', lever.slice('without '.length)] : [lever, Number(value)];
    const said = document.getElementById('fork-error');
    said.textContent = '';
    document.getElementById('twin').textContent = 'Forking at the year\'s end…';
    try {
      await api('api/twin', {lever: name, value: channel});
    } catch (error) {
      said.textContent = `No twin: ${error.message}`;
    }
    refresh();
  });
  document.getElementById('note').addEventListener('submit', async (event) => {
    event.preventDefault();
    const said = document.getElementById('note-error');
    said.textContent = '';
    const text = document.getElementById('note-text').value.trim();
    if (!text) return;
    try {
      await api('api/note',
          {kind: document.getElementById('note-kind').value, text});
    } catch (error) {
      said.textContent = `Not written: ${error.message}`;
      return;
    }
    document.getElementById('note-text').value = '';
    refresh();
  });
  document.getElementById('note-filter').addEventListener(
      'change', drawNotebook);
  document.getElementById('unfork').addEventListener('click', async () => {
    document.getElementById('fork-error').textContent = '';
    await api('api/twin', {end: true});
    refresh();
  });
  refresh();
  setInterval(refresh, 1500);
}

if (typeof document !== 'undefined') {
  start();
} else if (typeof module !== 'undefined') {
  module.exports = {
    askMove, cameBy, kinName, leverChoices, peopleColour, releaseMove,
    entryMeta, entryShown, savedNote, transmissionText, transmissionTree,
    twinNote, variantNote,
  };
}
