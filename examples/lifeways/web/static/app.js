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

/** The colour of a disaster's outline on the map (amber, unlike people). */
const DISASTER_COLOUR = '#ffb703';

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
  /** @type {?string} The played person's question shown, if any. */
  asked: null,
  /** @type {?Array<number>} The place chosen for a release: [row, col]. */
  at: null,
  /** Whether the person chose the place; until then it follows the crowd. */
  atChosen: false,
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
  // Fields: golden furrows across the sown land.
  ctx.strokeStyle = '#e9c46a';
  ctx.lineWidth = Math.max(1, page.tile / 8);
  for (const [r, c] of world.fields || []) {
    for (const furrow of [0.3, 0.7]) {
      ctx.beginPath();
      ctx.moveTo(c * page.tile, (r + furrow) * page.tile);
      ctx.lineTo((c + 1) * page.tile, (r + furrow) * page.tile);
      ctx.stroke();
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
  for (const disaster of recentDisasters(page.rows, world.year)) {
    // The land a disaster struck, fading over the years after.
    const [r, c] = disaster.tile;
    const reach = (world.disaster_radius + 0.5) * page.tile;
    const x = (c + 0.5) * page.tile;
    const y = (r + 0.5) * page.tile;
    ctx.globalAlpha = Math.max(0.2, 1 - disaster.age / RECENT_YEARS);
    ctx.beginPath();
    ctx.moveTo(x, y - reach);
    ctx.lineTo(x + reach, y);
    ctx.lineTo(x, y + reach);
    ctx.lineTo(x - reach, y);
    ctx.closePath();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = DISASTER_COLOUR;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = DISASTER_COLOUR;
    ctx.fillText(disaster.kind, x, y - reach - page.tile * 0.4);
    ctx.globalAlpha = 1;
  }
  if (page.at) {  // The place chosen for a release: a ring.
    const [r, c] = page.at;
    ctx.beginPath();
    ctx.arc((c + 0.5) * page.tile, (r + 0.5) * page.tile, page.tile * 0.45, 0,
        2 * Math.PI);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
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
    if (world.played && person.name === world.played.name) {  // You.
      ctx.beginPath();
      ctx.arc(person.x, person.y, page.tile * 0.4, 0, 2 * Math.PI);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 2]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

/**
 * The census's lines: what each reads from a row, its colour, and its top.
 * Peoples are drawn so that one people sits halfway up and a split shows.
 * @param {!Array<!Object>} rows Every row drawn (a world's and its twin's).
 * @param {number} maxLevel The highest skill level.
 * @return {!Array<{name: string, colour: string, value: function(!Object):
 *     number, most: number}>} The lines.
 */
function chartSeries(rows, maxLevel) {
  const lines = [
    {name: 'population', colour: '#2d6a4f',
     value: (row) => row.population || 0},
    {name: 'skill_worked', colour: '#bc6c25',
     value: (row) => row.skill_worked || 0},
    {name: 'tongues', colour: '#5a189a', value: (row) => row.tongues || 0},
    {name: 'peoples', colour: '#9d0208',
     value: (row) => (row.peoples && row.peoples.peoples) || 0},
    {name: 'fields', colour: '#e9c46a', value: (row) => row.fields || 0},
  ];
  const floor = {population: 1, skill_worked: maxLevel, tongues: 1,
                 peoples: 2, fields: 1};
  return lines.map((line) => ({...line, most: line.name === 'skill_worked' ?
      maxLevel : Math.max(...rows.map(line.value), floor[line.name])}));
}

/**
 * Draws the census: the living, the skill worked by, tongues, peoples and
 * fields,
 * year by year; a twin's census, if any, dashed over its world's, on the
 * same scales.
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
  const series = chartSeries(both, page.world.max_level);
  const first = rows[0].year;
  const lastYear = Math.max(...both.map((row) => row.year));
  const x = (year) => 4 + ((year - first) / Math.max(1, lastYear - first)) *
      (canvas.width - 8);
  for (const [drawn, dash] of [[rows, []], [twinRows, [5, 4]]]) {
    for (const {colour, value, most} of series) {
      ctx.beginPath();
      drawn.forEach((row, i) => {
        const y = canvas.height - 4 - (value(row) / most) *
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
      ` Red: peoples (${last.peoples.peoples}). Gold: fields` +
      ` (${last.fields || 0}).` +
      ` Years ${rows[0].year}–${last.year}.` +
      (twinRows.length ? ' Dashed: the twin.' : '');
  const channels = Object.entries(last.held.by_channel)
      .map(([channel, n]) => `${channel} ${n}`).join(', ');
  document.getElementById('channels').textContent =
      `Customs held, by how they came: ${channels}. Peoples, by size: ` +
      `${last.peoples.sizes.join(', ') || 'none'}.`;
}

/**
 * The Experimenter's release move, in the form the World reads.
 * @param {string} kind The custom's kind.
 * @param {string} domain Its domain.
 * @param {string} how 'song' or 'text'.
 * @param {string} at The place, 'row,col'.
 * @param {string} says What the custom says.
 * @param {string=} within Whom a marriage rule allows: 'tongue', 'descent',
 *     or '' for anyone.
 * @return {string} The move.
 */
function releaseMove(kind, domain, how, at, says, within = '') {
  const allows = kind === 'marriage' && within ? ` within ${within}` : '';
  return `release ${kind} ${domain}${allows} as a ${how} at ` +
      `${at.replace(/\s/g, '')}: ${says.trim()}`;
}

/**
 * The domains a custom of a kind may be of (a marriage rule is of kinship).
 * @param {!Object} world The state.
 * @param {string} kind The kind.
 * @return {!Array<string>} The domains.
 */
function domainsFor(world, kind) {
  return (world.kind_domains || {})[kind] || world.domains;
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

/**
 * What the page says of a release: where and how it went, and its fate.
 * @param {!Object} release A release the session follows.
 * @return {string} E.g. 'c30 · song at 6,13, year 4 · 2 heard it · held by 8
 *     now (most: 8, year 14) · 2 variants'.
 */
function releaseStatus(release) {
  const heard = release.how === 'song' ? ` · ${release.heard} heard it` :
      ' · on a shelf';
  const head = `${release.id} · ${release.how} at ${release.tile.join(',')},` +
      ` year ${release.year}${heard}`;
  if (release.died !== null) {
    return `${head} · ` + (release.peak ?
        `died out in year ${release.died} (at most ${release.peak} held it)` :
        `nobody took it up; forgotten in year ${release.died}`) +
        twinFate(release.twin);
  }
  const last = release.holders[release.holders.length - 1];
  const now = last ? last[1] : 0;
  if (!release.peak) {
    return `${head} · nobody has taken it up yet${twinFate(release.twin)}`;
  }
  const variants = release.variants > 1 ?
      ` · ${release.variants} variants` : '';
  return `${head} · held by ${now} now (most: ${release.peak}, year ` +
      `${release.peak_year})${variants}${twinFate(release.twin)}`;
}

/**
 * How a release fares in the twin, if it reached one.
 * @param {?Object|undefined} twin The release's `twin`, if any.
 * @return {string} E.g. ' · in the twin: held by 3'.
 */
function twinFate(twin) {
  if (!twin) return '';
  if (twin.died !== null) return ' · in the twin: died out';
  return ` · in the twin: held by ${twin.now}`;
}

/**
 * Draws how many held a release, year by year.
 * @param {!HTMLCanvasElement} canvas A small canvas.
 * @param {!Array<!Array<number>>} holders [year, holders] pairs.
 */
function drawSpark(canvas, holders) {
  canvas.width = 160;
  canvas.height = 28;
  const ctx = canvas.getContext('2d');
  if (holders.length < 2) return;
  const most = Math.max(1, ...holders.map(([, n]) => n));
  ctx.beginPath();
  holders.forEach(([, n], i) => {
    const x = 1 + (i / (holders.length - 1)) * 158;
    const y = 27 - (n / most) * 25;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = getComputedStyle(document.body)
      .getPropertyValue('--accent') || '#2d6a4f';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

/**
 * Lists the person's releases, the latest first.
 * @param {!Object} world The state.
 */
function drawReleases(world) {
  if (!world.releases.length) return;
  document.getElementById('releases').replaceChildren(
      ...[...world.releases].reverse().map((release) => {
        const item = document.createElement('li');
        const text = document.createElement('div');
        text.textContent = `"${release.text}"`;
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = releaseStatus(release);
        const spark = document.createElement('canvas');
        drawSpark(spark, release.holders);
        item.append(text, meta, spark);
        return item;
      }));
}

/**
 * How many would hear a song sung at a place, as the World counts them
 * (`reach`: [row, col, how many] for each tile some would hear at).
 * @param {!Object} world The state.
 * @param {!Array<number>} at [row, col], wrapped at the map's edges.
 * @return {number}
 */
function withinReach(world, at) {
  const wrap = (i, n) => ((i % n) + n) % n;
  const [row, col] = [wrap(at[0], world.rows), wrap(at[1], world.cols)];
  const hit = (world.reach || []).find(([r, c]) => r === row && c === col);
  return hit ? hit[2] : 0;
}

/**
 * Where a song would reach the most people (the World's `reach`, most
 * first): the release's place until the person picks one.
 * @param {!Object} world The state.
 * @return {?Array<number>} [row, col], or null if a song would reach nobody.
 */
function crowdedSpot(world) {
  const most = (world.reach || [])[0];
  return most ? [most[0], most[1]] : null;
}

/**
 * What the page says of the ring: who would hear a song there, and the
 * chance that at least one of them takes it up.
 * @param {!Object} world The state.
 * @param {!Array<number>} at [row, col].
 * @return {string}
 */
function reachNote(world, at) {
  const n = withinReach(world, at);
  const head = `The ring (${at.join(',')}): `;
  if (!n) return `${head}nobody would hear a song here.`;
  const someone = Math.round(100 * (1 - (1 - world.song_uptake) ** n));
  return `${head}${n} ${n === 1 ? 'person' : 'people'} would hear a song ` +
      `here; the chance at least one takes it up is ${someone}%.`;
}

/**
 * Says where a release would go, and who would hear a song there.
 * @param {!Object} world The state.
 */
function drawAt(world) {
  const note = document.getElementById('at-note');
  if (!page.at) {
    note.textContent = '';
    return;
  }
  note.textContent = reachNote(world, page.at) + (page.atChosen ? '' :
      ' (Where the most would hear, until you tap the map.)');
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

/** How many years the map shows a disaster after it struck. */
const RECENT_YEARS = 20;

/**
 * The disasters of the last few years, from the census rows.
 * @param {!Array<!Object>} rows The census rows.
 * @param {number} year The world's year.
 * @return {!Array<!Object>} Each disaster with its `age` in years.
 */
function recentDisasters(rows, year) {
  return rows.filter((row) => year - row.year <= RECENT_YEARS)
      .flatMap((row) => (row.disasters || []).map(
          (disaster) => ({...disaster, age: year - row.year})));
}

/** What the Experimenter may found or close. */
const INSTITUTIONS = ['library', 'school', 'shrine'];

/**
 * The choices of "Shape the valley": disasters, migrants, institutions.
 * @param {!Object} world The state.
 * @return {!Array<string>}
 */
function fateChoices(world) {
  return world.disaster_kinds.concat(['migrants'],
      INSTITUTIONS.map((kind) => `found ${kind}`),
      INSTITUTIONS.map((kind) => `close ${kind}`));
}

/**
 * The Experimenter's move to shape the valley, in the form the World reads.
 * @param {string} kind A disaster's kind, 'migrants', or 'found <kind>' or
 *     'close <kind>'.
 * @param {!Array<number>} at [row, col].
 * @return {string} The move.
 */
function fateMove(kind, at) {
  const where = at.join(',');
  if (kind === 'migrants') return `bring migrants at ${where}`;
  const [act, place] = kind.split(' ');
  if (act === 'found') return `found a ${place} at ${where}`;
  if (act === 'close') return `close the ${place} at ${where}`;
  return `send a ${kind} at ${where}`;
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
  page.at = [row, col];
  page.atChosen = true;
  drawMap(page.world);
  drawAt(page.world);
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

/** The refresh under way, if any, and whether another is wanted after it. */
const refreshing = {/** @type {?Promise<void>} */ now: null, again: false};

/**
 * Polls the world and redraws, one refresh at a time: a refresh asked for
 * while one is under way runs once it has ended, so none fetches the same
 * new rows twice.
 * @return {!Promise<void>}
 */
async function refresh() {
  if (refreshing.now) {
    refreshing.again = true;
    return refreshing.now;
  }
  refreshing.now = (async () => {
    do {
      refreshing.again = false;
      await refreshOnce();
    } while (refreshing.again);
  })();
  try {
    await refreshing.now;
  } finally {
    refreshing.now = null;
  }
}

/**
 * Appends what is newer than what the page has.
 * @param {!Array<!Object>} kept What the page has, oldest first.
 * @param {!Array<!Object>} fresh What the server sent, oldest first.
 * @param {string} key The field that orders them ('year' or 'id').
 * @param {number} since The last the page has.
 * @param {number} most How many to keep.
 * @return {!Array<!Object>}
 */
function appendNew(kept, fresh, key, since, most) {
  return kept.concat(fresh.filter((item) => item[key] > since)).slice(-most);
}

/** Polls the world and the census, and redraws. */
async function refreshOnce() {
  try {
    const world = await api('api/state');
    page.world = world;
    document.getElementById('year').textContent =
        `${world.scenario}, year ${world.year}: ${world.people.length} people` +
        (world.running ? '' : ' (the world has stopped)');
    document.getElementById('anew-box').hidden = world.running;
    document.getElementById('stop').hidden = !world.running;
    document.getElementById('play').textContent =
        world.pace.paused ? 'Play' : 'Pause';
    document.getElementById('queued').textContent = world.pace.queued.length ?
        `Next: ${world.pace.queued.join('; ')}` : '';
    document.getElementById('error').textContent = world.error || '';
    document.getElementById('saved').textContent = savedNote(world);
    document.getElementById('minds').textContent = world.minds === 'mock' ?
        'MOCK minds' : `LIVE minds (${world.minds})`;
    if (!page.atChosen) {
      page.at = crowdedSpot(world);
      document.getElementById('at').value = page.at ? page.at.join(',') : '';
    }
    drawMap(world);
    drawAt(world);
    drawLife(world);
    drawTranscripts(world);
    drawReleases(world);
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
      page.twinRows = appendNew(page.twinRows, twinRows, 'year',
          page.twinSince, world.rows_kept);
      page.twinSince = twinRows[twinRows.length - 1].year;
    }
    if (rows.length) {
      page.rows = appendNew(page.rows, rows, 'year', page.since,
          world.rows_kept);
      page.since = rows[rows.length - 1].year;
    }
    if (rows.length || twinRows.length) drawChart(page.rows, page.twinRows);
    const entries = await api(`api/notebook?since=${page.entrySince}`);
    if (entries.length) {
      page.entries = appendNew(page.entries, entries, 'id', page.entrySince,
          1000);
      page.entrySince = entries[entries.length - 1].id;
      drawNotebook();
    }
  } catch (error) {
    document.getElementById('error').textContent = String(error);
  }
}

/** What a played person does if the page does not answer, by question. */
const LIFE_UNASKED = {
  move: 'gather food',
  pass_on: 'pass on all you hold',
  interview: 'keep your counsel',
};

/** A custom in a person's context: "<label> (<domain>) [c3]: <text>". */
const LIFE_CUSTOM = /^(.*?) \(([^)\n]+)\) \[(c\d+)\]: (.*)$/gm;

/**
 * The moves a played person may make: its question's "- " lines.
 * @param {string} question The World's question.
 * @return {!Array<string>} The moves, in order.
 */
function lifeMoves(question) {
  return question.split('\n').filter((line) => line.startsWith('- '))
      .map((line) => line.slice(2).trim()).filter(Boolean);
}

/**
 * The customs a played person holds, as its context lists them.
 * @param {string} context The person's context.
 * @return {!Array<!Object>} Each custom once: {id, label, domain, text}.
 */
function lifeCustoms(context) {
  const found = new Map();
  for (const [, label, domain, id, text] of context.matchAll(LIFE_CUSTOM)) {
    if (!found.has(id)) found.set(id, {id, label, domain, text});
  }
  return [...found.values()];
}

/**
 * How the played person's life stands: its latest status line, as observed.
 * @param {string} context The person's context.
 * @param {string} name The person.
 * @return {string} The line ('' if none is in sight yet).
 */
function lifeStatus(context, name) {
  const lines = [...context.matchAll(/Year -?\d+\. (\S+) is [^\n]*/g)]
      .filter((found) => found[1] === name);
  return lines.length ? lines[lines.length - 1][0].trim() : '';
}

/**
 * What the played person knows and remembers, for a person to read: the
 * tags by which minds learn a custom become its id, and each thing the
 * person saw, and each year's status line, begins a line.
 * @param {string} context The person's context.
 * @param {string} name The person.
 * @return {string} The context, readable.
 */
function lifeMemory(context, name) {
  const who = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return context
      .replace(/ \|params=\{.*?\}(?= [A-Z[]|\s*$)/gm, '')
      .replace(/\[custom id=(c\d+)[^\]\n]*\] /g, '($1) ')
      .replace(new RegExp(
          `([.!?)])(?<!Year -?\\d+\\.) ` +
          `(?=Year -?\\d+\\. ${who} |${who} (?!is ))`, 'g'),
      '$1\n');
}

/**
 * The question, without the World's instructions on how to answer it.
 * @param {!Object} asked The question (`played.asked` in the state).
 * @return {string} The question to show.
 */
function lifeQuestion(asked) {
  if (asked.kind === 'move') {
    return asked.question.split('\n')[0].replace(/ Choose one move:$/, '');
  }
  if (asked.kind === 'pass_on') {
    return asked.question.replace(/ Answer "pass on"[^]*$/, '');
  }
  return asked.question;
}

/**
 * The answer the World reads, from what the person chose on the page.
 * @param {string} kind The question's kind ('move', 'pass_on', 'interview').
 * @param {string} move The move chosen (for a move).
 * @param {!Array<string>} ids The customs chosen.
 * @param {string=} said What the person says (for an interview).
 * @return {string} The answer ('' if there is nothing to send).
 */
function lifeAnswer(kind, move, ids, said = '') {
  if (kind === 'interview') return said.trim();
  if (kind === 'pass_on') return ['pass on', ...ids].join(' ');
  return move ? [move, ...ids].join(' ') : '';
}

/**
 * A line on the played person's life: what waits for it, or that it ended.
 * @param {?Object} played The state's `played`.
 * @return {string} The line.
 */
function lifeNote(played) {
  if (!played) return '';
  if (!played.alive) {
    return `${played.name} has died; what they taught lives on in others. ` +
        'Stop the world and begin a new one to live again.';
  }
  const asked = played.asked;
  if (!asked) return 'The others are choosing; the year turns.';
  const year = asked.year >= 0 ? `Year ${asked.year}: the` : 'The';
  return `${year} year waits for you. Answer within ` +
      `${asked.seconds_left}\u00a0s, or you ${LIFE_UNASKED[asked.kind] ||
      LIFE_UNASKED.move}.`;
}

/**
 * A radio button or checkbox with its label.
 * @param {string} type 'radio' or 'checkbox'.
 * @param {string} name The group's name.
 * @param {string} value The value sent.
 * @param {string} text The label.
 * @param {boolean} checked Whether it starts chosen.
 * @param {string=} meta A muted line under the label, if any.
 * @return {!HTMLElement} The label holding the input.
 */
function choiceInput(type, name, value, text, checked, meta = '') {
  const label = document.createElement('label');
  const input = document.createElement('input');
  Object.assign(input, {type, name, value, checked});
  const words = document.createElement('span');
  words.textContent = text;
  if (meta) {
    const small = document.createElement('span');
    small.className = 'meta';
    small.textContent = meta;
    words.append(document.createElement('br'), small);
  }
  label.append(input, words);
  return label;
}

/**
 * Shows the played person's life: its question, kept as chosen until the
 * question changes.
 * @param {!Object} world The state.
 */
function drawLife(world) {
  const played = world.played;
  document.getElementById('life').hidden = !played;
  if (!played) return;
  document.getElementById('life-name').textContent = played.name;
  document.getElementById('life-note').textContent = lifeNote(played);
  const asked = played.alive ? played.asked : null;
  document.getElementById('life-form').hidden = !asked;
  if (!asked) {
    page.asked = null;
    return;
  }
  const key = `${asked.id} ${asked.error || ''}`;
  if (key === page.asked) return;
  page.asked = key;
  document.getElementById('life-error').textContent = asked.error || '';
  document.getElementById('life-status').textContent =
      lifeStatus(asked.context, played.name);
  document.getElementById('life-question').textContent = lifeQuestion(asked);
  document.getElementById('life-context').textContent =
      lifeMemory(asked.context, played.name);
  const passing = asked.kind === 'pass_on';
  const interview = asked.kind === 'interview';
  document.getElementById('life-moves').hidden = asked.kind !== 'move';
  document.getElementById('life-customs').hidden = interview;
  document.getElementById('life-say-box').hidden = !interview;
  document.getElementById('life-say').value = '';
  document.getElementById('life-move-list').replaceChildren(
      ...lifeMoves(asked.question).map(
          (move, i) => choiceInput('radio', 'life-move', move, move, i === 0)));
  document.getElementById('life-customs-legend').textContent = passing ?
      'The customs you pass on to your child' :
      'The customs you follow in it, if any';
  document.getElementById('life-custom-list').replaceChildren(
      ...lifeCustoms(asked.context).map((custom) => choiceInput(
          'checkbox', 'life-custom', custom.id, custom.text, passing,
          `${custom.label} (${custom.domain}) ${custom.id}`)));
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
  fill('domain', domainsFor(world, document.getElementById('kind').value));
  document.getElementById('kind').addEventListener('change', (event) => {
    const kind = event.target.value;
    fill('domain', domainsFor(page.world || world, kind));
    document.getElementById('within-box').hidden = kind !== 'marriage';
  });
  fill('fate-kind', fateChoices(world));
  fill('preset', world.presets);
  document.getElementById('preset').value = world.scenario;
  document.getElementById('lever').replaceChildren(...leverChoices(world).map(
      ([value, label]) => new Option(label, value)));
  const build = await api('api/build');
  document.getElementById('build').textContent =
      `build ${build.commit || '?'}`;
  document.getElementById('map').addEventListener('click', onMapTap);
  document.getElementById('at').addEventListener('change', (event) => {
    const at = event.target.value.split(',').map((n) => parseInt(n, 10));
    page.at = at.length === 2 && at.every(Number.isFinite) ? at : null;
    page.atChosen = page.at !== null;
    drawMap(page.world);
    drawAt(page.world);
  });
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
  document.getElementById('anew').addEventListener('click', async () => {
    const scenario = document.getElementById('preset').value;
    if (!confirm(`Begin a new world (${scenario})? This one is set aside.`)) {
      return;
    }
    await api('api/new',
        {scenario, play_as: document.getElementById('play-as').checked});
    Object.assign(page, {rows: [], since: -1, twinRows: [], twinSince: -1,
      twin: null, entries: [], entrySince: 0, person: null, at: null,
      asked: null});
    refresh();
  });
  document.getElementById('life-form').addEventListener('submit',
      async (event) => {
        event.preventDefault();
        const played = page.world && page.world.played;
        const asked = played && played.asked;
        if (!asked) return;
        const move = document.querySelector('input[name="life-move"]:checked');
        const ids = [...document.querySelectorAll(
            'input[name="life-custom"]:checked')].map((input) => input.value);
        const answer = lifeAnswer(asked.kind, move ? move.value : '', ids,
            document.getElementById('life-say').value);
        const said = document.getElementById('life-error');
        if (!answer) {
          said.textContent = 'Choose or say something first.';
          return;
        }
        said.textContent = '';
        try {
          await api('api/me', {id: asked.id, answer});
          document.getElementById('life-form').hidden = true;
        } catch (error) {
          said.textContent = `Not taken: ${error.message}`;
        }
        refresh();
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
        document.getElementById('at').value, says,
        document.getElementById('within').value);
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
    if (name !== 'without' && (value === '' || !Number.isFinite(channel))) {
      said.textContent = 'No twin: give the lever a value, a number.';
      return;
    }
    document.getElementById('twin').textContent = 'Forking at the year\'s end…';
    try {
      await api('api/twin', {lever: name, value: channel});
    } catch (error) {
      said.textContent = `No twin: ${error.message}`;
    }
    refresh();
  });
  document.getElementById('fate').addEventListener('submit', async (event) => {
    event.preventDefault();
    const said = document.getElementById('fate-error');
    if (!page.at || !page.atChosen) {
      said.textContent = 'Tap the map first, to choose the place.';
      return;
    }
    said.textContent = '';
    await api('api/move', {
      move: fateMove(document.getElementById('fate-kind').value, page.at),
    });
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
    askMove, cameBy, chartSeries, crowdedSpot, domainsFor, kinName,
    leverChoices, peopleColour, releaseMove, appendNew, entryMeta, entryShown,
    fateChoices, fateMove, lifeAnswer, lifeCustoms, lifeMoves, lifeNote,
    lifeMemory, lifeQuestion, lifeStatus, reachNote, recentDisasters, releaseStatus, savedNote,
    transmissionText, transmissionTree, twinNote, variantNote, withinReach,
  };
}
