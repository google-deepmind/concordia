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
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
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
 * Draws the census: the living and the mean skill, year by year.
 * @param {!Array<!Object>} rows The census rows.
 */
function drawChart(rows) {
  const canvas = /** @type {!HTMLCanvasElement} */ (
      document.getElementById('chart'));
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (rows.length < 2) return;
  const series = [
    {key: 'population', colour: '#2d6a4f', most: Math.max(...rows.map(
        (row) => row.population), 1)},
    {key: 'skill_worked', colour: '#bc6c25', most: page.world.max_level},
    {key: 'tongues', colour: '#5a189a', most: Math.max(...rows.map(
        (row) => row.tongues || 0), 1)},
  ];
  const x = (i) => 4 + (i / (rows.length - 1)) * (canvas.width - 8);
  for (const {key, colour, most} of series) {
    ctx.beginPath();
    rows.forEach((row, i) => {
      const y = canvas.height - 4 - ((row[key] || 0) / most) *
          (canvas.height - 8);
      if (i === 0) ctx.moveTo(x(i), y); else ctx.lineTo(x(i), y);
    });
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  const last = rows[rows.length - 1];
  document.getElementById('legend').textContent =
      `Green: the living (${last.population}). Brown: the skill people work` +
      ` by (${last.skill_worked}). Purple: tongues spoken (${last.tongues}).` +
      ` Years ${rows[0].year}–${last.year}.`;
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
  const person = await api(`api/person?name=${encodeURIComponent(name)}`);
  document.getElementById('person-name').textContent = person.name;
  const age = page.world.year - person.born;
  document.getElementById('person-body').textContent =
      `${age} years old, of the ${person.descent || 'valley'}; food ` +
      `${person.food}; partner: ${person.partner || 'none'}; parents: ` +
      `${person.parents.join(', ') || 'none'}.`;
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
        (detail ? `, ${detail}` : '') + ` · came by ${custom.via}`;
    item.append(meta);
    return item;
  }));
  document.getElementById('person').hidden = false;
  page.person = person.name;
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
    drawMap(world);
    drawTranscripts(world);
    const rows = await api(`api/census?since=${page.since}`);
    if (rows.length) {
      page.rows = page.rows.concat(rows).slice(-world.rows_kept);
      page.since = rows[rows.length - 1].year;
      drawChart(page.rows);
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
  const build = await api('api/build');
  document.getElementById('build').textContent =
      `build ${build.commit || '?'}`;
  document.getElementById('map').addEventListener('click', onMapTap);
  document.getElementById('person-close').addEventListener('click', () => {
    document.getElementById('person').hidden = true;
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
  refresh();
  setInterval(refresh, 1500);
}

if (typeof document !== 'undefined') {
  start();
} else if (typeof module !== 'undefined') {
  module.exports = {askMove, peopleColour, releaseMove};
}
