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
const PLACE_GLYPHS = {library: 'L', school: 'S', shrine: '✦', court: 'C'};

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
  /** The year the person's panel shows (it follows the world's). */
  personYear: -1,
  /** @type {?string} The lineage shown (a variant's id), if any. */
  lineage: null,
  /** The year the lineage shows (it is redrawn only when asked). */
  lineageYear: -1,
  /** @type {!Array<!Array<number>>} [ms, year] seen while running. */
  paceSeen: [],
  /** How the map colours people (`personColour`), remembered if it may. */
  overlay: remembered('lifeways.overlay', 'descent'),
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
 * The peoples' colours: none green, blue or brown, so a dot never melts
 * into the land (the presets' peoples get the first four apart: River red,
 * Hill amber, West white, East magenta).
 */
const PEOPLE_COLOURS = [
  '#fcbf49', '#f77f00', '#ffffff', '#d62828', '#1a1a1a', '#d81b9a', '#7b2cbf',
];

/**
 * A colour for a people, the same every time.
 * @param {string} name The people's name.
 * @return {string} A colour of `PEOPLE_COLOURS`.
 */
function peopleColour(name) {
  let hash = 0;
  for (const char of name) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return PEOPLE_COLOURS[hash % PEOPLE_COLOURS.length];
}

/**
 * A choice this browser remembers, if it may (private windows may not).
 * @param {string} key The storage key.
 * @param {string} fallback The value if none is kept.
 * @return {string} The value.
 */
function remembered(key, fallback) {
  try {
    return (typeof localStorage !== 'undefined' &&
        localStorage.getItem(key)) || fallback;
  } catch (error) {
    return fallback;
  }
}

/**
 * Keeps a choice in this browser, if it may.
 * @param {string} key The storage key.
 * @param {string} value The value.
 */
function remember(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (error) {
    // Not kept: the page works the same without it.
  }
}

/**
 * Sizes a canvas's pixels to the screen's (sharp on any phone) and returns
 * its context, drawing in CSS pixels.
 * @param {!HTMLCanvasElement} canvas The canvas.
 * @param {number} width Its width in CSS pixels.
 * @param {number} height Its height in CSS pixels.
 * @return {!CanvasRenderingContext2D} The context.
 */
function crisp(canvas, width, height) {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return ctx;
}

/**
 * A colour the stylesheet sets for the page's theme.
 * @param {string} name The custom property, e.g. '--chart-population'.
 * @param {string} fallback The colour if it sets none.
 * @return {string} The colour.
 */
function themeColour(name, fallback) {
  const colour = getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
  return colour || fallback;
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
  const ctx = crisp(canvas, world.cols * page.tile, world.rows * page.tile);
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
  const tongues = tongueRanks(world.people);
  for (const person of world.people) {
    const key = person.tile.join(',');
    const n = (crowd[key] = (crowd[key] || 0) + 1) - 1;
    const angle = n * 2.4;
    const spread = Math.min(0.35, 0.12 * Math.sqrt(n));
    person.x = (person.tile[1] + 0.5 + spread * Math.cos(angle)) * page.tile;
    person.y = (person.tile[0] + 0.5 + spread * Math.sin(angle)) * page.tile;
    ctx.beginPath();
    ctx.arc(person.x, person.y,
        page.tile * (person.age < world.grown_age ? 0.2 : 0.3), 0, 2 * Math.PI);
    ctx.fillStyle = personColour(person, page.overlay, world, tongues);
    ctx.fill();
    // A dark ring and a light one: a dot shows on any land, light or dark.
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.lineWidth = 1;
    ctx.stroke();
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
  // A place's letter stays readable over anyone standing there.
  ctx.font = `bold ${Math.floor(page.tile * 0.85)}px system-ui`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
  ctx.fillStyle = '#222';
  for (const place of world.places) {
    const [r, c] = place.tile;
    const glyph = PLACE_GLYPHS[place.kind] || '?';
    ctx.strokeText(glyph, (c + 0.5) * page.tile, (r + 0.5) * page.tile);
    ctx.fillText(glyph, (c + 0.5) * page.tile, (r + 0.5) * page.tile);
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
  const width = canvas.clientWidth || 480;
  const height = Math.round(width * 0.375);
  const ctx = crisp(canvas, width, height);
  if (rows.length < 2) return;
  const both = rows.concat(twinRows);
  const series = chartSeries(both, page.world.max_level).map((line) => ({
    ...line, colour: themeColour(`--chart-${line.name}`, line.colour)}));
  const first = rows[0].year;
  const lastYear = Math.max(...both.map((row) => row.year));
  const labels = 78;  // Room at the right for each line's value and name.
  const x = (year) => 4 + ((year - first) / Math.max(1, lastYear - first)) *
      (width - 8 - labels);
  const y = (value, most) => height - 6 - (value / most) * (height - 14);
  for (const [drawn, dash] of [[rows, []], [twinRows, [5, 4]]]) {
    for (const {colour, value, most} of series) {
      ctx.beginPath();
      drawn.forEach((row, i) => {
        const at = y(value(row), most);
        if (i === 0) {
          ctx.moveTo(x(row.year), at);
        } else {
          ctx.lineTo(x(row.year), at);
        }
      });
      ctx.setLineDash(dash);
      ctx.strokeStyle = colour;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
  ctx.setLineDash([]);
  // Each line's value now, at its end, kept apart so none hides another.
  const ends = series.map(({name, colour, value, most}) => ({
    colour, text: `${value(rows[rows.length - 1])} ${CHART_NAMES[name]}`,
    at: y(value(rows[rows.length - 1]), most)}))
      .sort((a, b) => a.at - b.at);
  ends.forEach((end, i) => {
    if (i) end.at = Math.max(end.at, ends[i - 1].at + 13);
  });
  ctx.font = 'bold 12px system-ui';
  ctx.textBaseline = 'middle';
  for (const end of ends) {
    ctx.fillStyle = end.colour;
    ctx.fillText(end.text, width - labels + 2, Math.min(end.at, height - 6));
  }
  const last = rows[rows.length - 1];
  document.getElementById('legend').textContent =
      `Years ${yearText(rows[0].year)}–${yearText(last.year)}, each line` +
      ' labelled with its value now. Each has its own scale: its top is the' +
      ' most it reached in these years. Skill: the mean level of the' +
      ' know-how people work by (the top is the highest level there is).' +
      ' Peoples: groups who would marry one another. Fields: fields sown.' +
      (twinRows.length ? ' Dashed: the twin.' : '');
  const channels = Object.entries(last.held.by_channel)
      .sort(([, a], [, b]) => b - a)
      .map(([channel, n]) => `${HELD_BY[channel] || channel} ${n}`)
      .join(', ');
  document.getElementById('channels').textContent =
      `How the living came by the customs they hold: ${channels}. ` +
      peoplesText(last.peoples, last.population);
}

/**
 * The peoples in words: who is counted, and each named by its descent.
 * @param {!Object} peoples The census's peoples: `peoples`, `sizes`, and
 *     (since cycle 37) `descents` and `grown`.
 * @param {number} alive How many are alive.
 * @return {string} E.g. 'Of the 63 alive, 43 are grown; they form 2
 *     peoples (groups whose marriage rules let them marry one another):
 *     26 mostly of the Hill people, 17 mostly of the River people.
 *     Children belong to none until grown.'
 */
function peoplesText(peoples, alive) {
  const grown = peoples.grown ?? peoples.sizes.reduce((a, b) => a + b, 0);
  if (!grown) return `None of the ${alive} alive is grown, so no peoples.`;
  const each = peoples.sizes.map((n, i) => peoples.descents ?
    `${n} mostly of the ${peoples.descents[i] || 'valley'}` : `${n}`);
  return `Of the ${alive} alive, ${grown} ${grown === 1 ? 'is' : 'are'} ` +
      `grown; they form ${peoples.peoples === 1 ? 'one people' :
        `${peoples.peoples} peoples`} (groups whose marriage rules let them ` +
      `marry one another): ${each.join(', ')}. Children belong to none ` +
      'until grown.';
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
 * @param {number=} level How good a skill is (0: not said, the World's 1).
 * @return {string} The move.
 */
function releaseMove(kind, domain, how, at, says, within = '', level = 0) {
  const good = kind === 'skill' && level ? ` level ${level}` : '';
  const allows = kind === 'marriage' && within ? ` within ${within}` : '';
  return `release ${kind} ${domain}${good}${allows} as a ${how} at ` +
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
      `without ${CHANNEL_WORDS[twin.value] || twin.value}` :
      `${(LEVER_WORDS[twin.lever] || twin.lever).toLowerCase()}: ` +
        `${twin.value}`;
  return `The twin (${pulled}; forked in year ${yearText(twin.year)}) is ` +
      `in year ${yearText(twin.year_now)}: ${twin.people} people alive` +
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
        [lever, `${LEVER_WORDS[lever] || lever} (from ${low}, least, to ` +
          `${high}, most; now ${now})`])
      .concat(world.channels.map((channel) =>
        [`without ${channel}`,
          `Without ${CHANNEL_WORDS[channel] || channel}`]));
}

/**
 * What the page says of a release: where and how it went, and its fate.
 * @param {!Object} release A release the session follows.
 * @return {string} E.g. 'sung at 6,13 in year 4 · 2 heard it · held by 8 now
 *     (most: 8, in year 14)'.
 */
function releaseStatus(release) {
  const at = `row ${release.tile[0]}, column ${release.tile[1]}`;
  const head = release.how === 'song' ?
      `sung at ${at} in year ${yearText(release.year)} · ${release.heard}` +
        ` heard it` :
      `left as a text on a library shelf at ${at} in year ` +
        yearText(release.year);
  if (release.died !== null) {
    return `${head} · ` + (release.peak ?
        `died out in year ${yearText(release.died)} (at most ` +
          `${release.peak} held it)` :
        `nobody took it up; gone by year ${yearText(release.died)}`) +
        twinFate(release.twin);
  }
  const last = release.holders[release.holders.length - 1];
  const now = last ? last[1] : 0;
  if (!release.peak) {
    return `${head} · nobody has taken it up yet${twinFate(release.twin)}`;
  }
  const variants = release.variants > 1 ?
      ` · ${release.variants} versions of it, as copies changed it` : '';
  return `${head} · held by ${now} now (most: ${release.peak}, in year ` +
      `${yearText(release.peak_year)})${variants}${twinFate(release.twin)}`;
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

/** What each census line counts, as its end is labelled. */
const CHART_NAMES = {
  population: 'alive',
  skill_worked: 'skill',
  tongues: 'tongues',
  peoples: 'peoples',
  fields: 'fields',
};

/**
 * Draws how many held a release, year by year.
 * @param {!HTMLCanvasElement} canvas A small canvas.
 * @param {!Array<!Array<number>>} holders [year, holders] pairs.
 */
function drawSpark(canvas, holders) {
  const ctx = crisp(canvas, 160, 28);
  if (holders.length < 2) return;
  const most = Math.max(1, ...holders.map(([, n]) => n));
  ctx.beginPath();
  holders.forEach(([, n], i) => {
    const x = 1 + (i / (holders.length - 1)) * 158;
    const y = 27 - (n / most) * 25;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = themeColour('--accent', '#2d6a4f');
  ctx.lineWidth = 2;
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
        item.append(text, meta);
        if (release.holders.length > 1) {
          const label = document.createElement('div');
          label.className = 'meta';
          label.textContent = 'How many held it, year by year:';
          const spark = document.createElement('canvas');
          drawSpark(spark, release.holders);
          item.append(label, spark);
        }
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
 * @param {string=} how 'song' or 'text'.
 * @return {string}
 */
function reachNote(world, at, how = 'song') {
  const head = `The white ring on the map (row ${at[0]}, column ${at[1]}) ` +
      'is where your next release goes: ';
  if (how === 'text') {
    return `${head}a text is left on a library's shelf here (a new ` +
        'library if none stands near), for whoever comes to read it.';
  }
  const n = withinReach(world, at);
  if (!n) return `${head}nobody would hear a song here now.`;
  const someone = Math.round(100 * (1 - (1 - world.song_uptake) ** n));
  return `${head}${n} ${n === 1 ? 'person' : 'people'} would hear a song ` +
      `here now; the chance at least one takes it up is ${someone}%. ` +
      'People move, so this can change by next year.';
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
  note.textContent = reachNote(world, page.at,
      document.getElementById('how').value) + (page.atChosen ? '' :
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
  const kind = {move: 'your move', world: 'what came of it',
    event: 'the valley'};
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
    text.textContent = entryText(entry);
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
const INSTITUTIONS = ['library', 'school', 'shrine', 'court'];

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
  const line = (text) => {
    const item = document.createElement('li');
    item.textContent = text;
    return item;
  };
  const queued = world.pace.queued
      .map((move) => move.match(/^ask ([^:]+): (.+)$/))
      .filter(Boolean)
      .map(([, name, question]) =>
        line(`${name} will be asked next year: "${question}"`));
  const pending = Object.entries(world.interviews).map(([name, question]) =>
    line(`${name} is asked: "${question}" (answers this year)`));
  const answered = [...world.transcripts].reverse().map((transcript) => {
    const item = document.createElement('li');
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = `${transcript.name}, year ${transcript.year}, was ` +
        `asked "${transcript.question}":`;
    const answer = document.createElement('div');
    answer.textContent = answerText(transcript);
    item.append(meta, answer);
    return item;
  });
  document.getElementById('transcripts').replaceChildren(
      ...queued, ...pending, ...answered);
  document.getElementById('mock-answers').hidden = world.minds !== 'mock';
}

/**
 * Your moves waiting: one happens each year, in the order made.
 * @param {!Array<string>} queued The moves, as the World reads them.
 * @return {string} E.g. 'Next year: Send a fire at row 1, column 1. Then,
 *     one a year: Bring newcomers to row 2, column 2.', or ''.
 */
function queuedNote(queued) {
  if (!queued.length) return '';
  const [next, ...then] = queued.map(moveWords);
  return `Next year: ${next}.` +
      (then.length ? ` Then, one a year: ${then.join('; ')}.` : '');
}

/**
 * A person's status, a line a sentence (it is one long paragraph).
 * @param {string} status The status the World gives.
 * @return {!Array<string>} Its sentences.
 */
function statusLines(status) {
  return status.split(/(?<=[.)])\s+(?=[A-Z0-9])/).map((s) => s.trim())
      .filter(Boolean);
}

/** What each Concordia part does here, in words, by its class. */
const PART_WORDS = {
  AssociativeMemory: 'remembers what it has seen',
  ObservationToMemory: 'writes each thing it sees into memory',
  LastNObservations: 'keeps the latest things it saw in mind',
  Constant: 'a fixed text it reads: who it is',
  Learner: 'takes up customs it is offered, and lets old ones go',
  Custom: 'a custom: a text it reads when it decides',
  ConcatActComponent: 'asks its mind (a language model, or the simple ' +
      'rules) what it does, in one document of all the parts above',
  PlayedAct: 'you answer for this person; unanswered, it acts by habit',
  HumanActComponent: 'a person answers',
  RuleAct: 'a rule decides',
  SwitchAct: 'sends each of the engine\'s questions to the part that ' +
      'answers it',
  WorldComponent: 'the valley itself: land, people, places, culture, rules',
  PlayerRoster: 'people join the game when born and leave when they die',
  TurnGate: 'who acts next; closes the year when everyone has acted',
  Questions: 'what each person is asked this year',
  Answers: 'what each answer does in the world',
  MakeObservation: 'tells each person what they see',
  Terminate: 'ends the world when you end it',
};

/** Where a part comes from, in words. */
const SOURCE_WORDS = {
  concordia: 'stock Concordia',
  contrib: 'Concordia contrib',
  lifeways: 'this example',
};

/**
 * How Concordia builds an entity, a line a part (customs counted as one).
 * @param {!Object} anatomy The anatomy reply: {class, acting, components}.
 * @return {!Array<string>} E.g. 'Learner (this example): takes up ...'.
 */
function anatomyLines(anatomy) {
  const line = (part, count = 1) => `${part.class}` +
      (count > 1 ? ` × ${count}` : '') +
      ` (${SOURCE_WORDS[part.source] || part.source}): ` +
      (PART_WORDS[part.class] || 'a part of this entity');
  const counts = new Map();
  const first = new Map();
  for (const part of anatomy.components) {
    counts.set(part.class, (counts.get(part.class) || 0) + 1);
    if (!first.has(part.class)) first.set(part.class, part);
  }
  return [
    `${anatomy.class} (${SOURCE_WORDS.concordia}): the entity`,
    ...[...first.values()].map((part) => line(part, counts.get(part.class))),
    `It acts through ${line(anatomy.acting)}`,
  ];
}

/**
 * Fills a list with how Concordia builds an entity (fetched when shown).
 * @param {string} listId The list's id.
 * @param {string} name The entity's name ('' for the World).
 */
async function showAnatomy(listId, name) {
  const anatomy = await api(`api/anatomy?name=${encodeURIComponent(name)}`);
  document.getElementById(listId).replaceChildren(
      ...anatomyLines(anatomy).map((text) => {
        const item = document.createElement('li');
        item.textContent = text;
        return item;
      }));
}

/**
 * What the ask form says once a question is asked.
 * @param {string} name Who is asked.
 * @param {number} year The year it is asked in.
 * @param {boolean} paused Whether the world is paused.
 * @return {string} The note.
 */
function askedNote(name, year, paused) {
  return `Asked in year ${yearText(year)}. ${name} answers when the year ` +
      'turns' + (paused ? ' (press Play: the world is paused)' : '') +
      '; the answer shows under Interviews.';
}

/**
 * Shows a person's customs.
 * @param {string} name The person's name.
 */
async function showPerson(name) {
  if (name !== page.person) {
    document.getElementById('ask-note').textContent = '';
  }
  const [person, kin] = await Promise.all([
    api(`api/person?name=${encodeURIComponent(name)}`),
    api(`api/kin?name=${encodeURIComponent(name)}`),
  ]);
  document.getElementById('person-name').textContent = person.name;
  const age = page.world.year - person.born;
  document.getElementById('person-body').textContent =
      `${age} years old, born of the ${person.descent || 'valley'}; food ` +
      `stored ${person.food} (a year's eating is 1).`;
  document.getElementById('person-kin').replaceChildren(...KIN.map(
      ([key, label]) => kinRow(label, [].concat(kin[key] || []))));
  const list = document.getElementById('person-customs');
  list.replaceChildren(...person.customs.map((custom) => {
    const item = document.createElement('li');
    item.textContent = custom.text +
        (custom.kind === 'dialect' ? ' (a few of its words)' : '');
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = `${customKind(custom)} · ${name} ${cameBy(custom)} · `;
    const lineage = document.createElement('button');
    lineage.type = 'button';
    lineage.className = 'link';
    lineage.textContent = 'how it has changed';
    lineage.addEventListener('click', () => showLineage(custom.id));
    meta.append(lineage);
    item.append(meta);
    return item;
  }));
  const parts = document.getElementById('person-anatomy');
  if (parts.open) showAnatomy('person-anatomy-list', person.name);
  document.getElementById('person-anatomy-label').textContent =
      `How Concordia builds ${person.name}`;
  const letGo = person.let_go || [];
  document.getElementById('person-let-go-box').hidden = !letGo.length;
  document.getElementById('person-let-go').replaceChildren(
      ...[...letGo].reverse().map((record) => {
        const item = document.createElement('li');
        item.textContent = letGoText(record);
        return item;
      }));
  document.getElementById('person').hidden = false;
  page.person = person.name;
  page.personYear = page.world.year;
}

/**
 * A custom a person let go, and why, in words.
 * @param {!Object} record {text, kind, domain, for_text, why} (`why`:
 *     'locus', 'domain' or 'capacity').
 * @return {string} E.g. '"Sowing wild barley." (know-how about farming) gave
 *     way to "We take nothing from another's field.": two about farming
 *     are the most one holds.'
 */
function letGoText(record) {
  const what = `${KIND_WORDS[record.kind] || record.kind} about ` +
      record.domain;
  const why = {
    locus: 'one holds only one of each kind about a thing',
    domain: `two about ${record.domain} are the most one holds`,
    capacity: 'one holds only so many customs in all',
  }[record.why] || record.why;
  return `"${record.text}" (${what}) gave way to "${record.for_text}": ` +
      `${why}.`;
}

/**
 * Keeps the open panels true to the year: the person's is redrawn each
 * year, and says when the person is no longer alive; the lineage's (whose
 * folds a redraw would close) offers to show it as of now.
 * @param {!Object} world The world's state.
 */
function followYear(world) {
  if (page.person && !document.getElementById('person').hidden &&
      page.personYear !== world.year && page.personYear >= 0) {
    if (world.people.some((p) => p.name === page.person)) {
      showPerson(page.person).catch(() => {});
    } else {
      document.getElementById('person-body').textContent =
          goneFromPanel(page.person, page.personYear, world.year);
      page.personYear = -1;  // Shown as gone; nothing more to follow.
    }
  }
  document.getElementById('lineage-update').hidden =
      !page.lineage || page.lineageYear >= world.year;
}

/**
 * The pace the world keeps, when it is short of the pace asked: a valley
 * runs only as fast as its people answer, and a twin shares the machine.
 * @param {!Array<!Array<number>>} seen [ms, year] seen while running, at
 *     least ten seconds of them.
 * @param {number} asked The years a second asked for.
 * @param {boolean} twin Whether a twin runs beside it.
 * @return {string} E.g. 'running at about 0.4 a second: as fast as this
 *     valley can go now (a twin beside it halves that)', or ''.
 */
function paceNote(seen, asked, twin) {
  if (seen.length < 2) return '';
  const [ms0, year0] = seen[0];
  const [ms1, year1] = seen[seen.length - 1];
  if (ms1 - ms0 < 10000) return '';
  const rate = (year1 - year0) / ((ms1 - ms0) / 1000);
  if (rate >= 0.8 * asked) return '';
  const shown = rate < 1 ? rate.toFixed(1) : String(Math.round(rate));
  return `running at about ${shown} a second: as fast as this valley can ` +
      'go now' + (twin ? ' (a twin beside it halves that)' : '');
}

/**
 * What the person's panel says once its person has died.
 * @param {string} name The person.
 * @param {number} shown The year the panel last showed them alive.
 * @param {number} year The world's year.
 * @return {string} The line.
 */
function goneFromPanel(name, shown, year) {
  return `${name} is no longer alive (year ${yearText(year)}). ` +
      `Below is how they were in year ${yearText(shown)}.`;
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
  const [alone, withWhom] = LEARNED_BY[custom.via || 'birth'] ||
      [`came by ${custom.via}`, `came by ${custom.via} from`];
  return custom.from ? `${withWhom} ${custom.from}` : alone;
}

/** An example of what a released custom of each kind might say. */
const EXAMPLE_SAYS = {
  norm: 'We give the first fish to the river.',
  value: 'We do not forget a debt.',
  belief: 'Storms come when the shrine is neglected.',
  skill: 'Smoking fish over green alder.',
  habit: 'In lean years, gather before anything else.',
  marriage: 'We marry only our own people.',
  bias: 'Learn from the eldest.',
};

/** What each kind of custom is, in words. */
const KIND_WORDS = {
  norm: 'a rule',
  value: 'something held dear',
  belief: 'a belief',
  skill: 'know-how',
  habit: 'a habit',
  bias: 'whom to learn from',
  dialect: 'a way of speaking',
  marriage: 'whom to marry',
};

/** The world's levers, in words. */
const LEVER_WORDS = {
  richness: 'How rich the land is',
  regrowth: 'How fast food grows back',
  seed_bank: 'How fast bare land greens again',
  subsistence: 'Food found while doing other things',
  mutation: 'How often copying a custom errs',
  dialect_drift: 'How fast tongues change',
  garble: 'How hard a strange tongue is to follow',
  child_cost: 'Food a child costs its mother',
};

/** Each way customs pass between people, as a twin goes without it. */
const CHANNEL_WORDS = {
  apprenticeship: 'apprenticeship',
  arrival: 'newcomers',
  birth: 'customs given at birth',
  observation: 'learning by watching others',
  practice: 'learning by practice',
  preaching: 'preaching',
  reading: 'reading',
  release: 'your releases',
  rite: 'rites',
  sanction: 'shaming those who break a rule',
  school: 'schools',
  story: 'stories',
  upbringing: 'parents teaching their children',
};

/** How the living came by the customs they hold, by channel. */
const HELD_BY = {
  apprenticeship: 'as apprentices',
  arrival: 'brought in',
  birth: 'inherited them',
  observation: 'saw them done',
  practice: 'by practice',
  preaching: 'heard them preached',
  reading: 'read them',
  release: 'from your releases',
  rite: 'at rites',
  sanction: 'shamed into them',
  school: 'at school',
  story: 'heard in stories',
  upbringing: 'taught as children',
};

/**
 * What kind of custom one is, in words.
 * @param {!Object} custom A custom: its kind, domain and params.
 * @return {string} E.g. 'a rule they keep as their own (about kinship)'.
 */
function customKind(custom) {
  const params = custom.params || {};
  let what = KIND_WORDS[custom.kind] || custom.kind;
  if (custom.kind === 'norm') {
    what = params.form === 'known' ? 'a rule others expect of them' :
        'a rule they keep as their own';
  } else if (custom.kind === 'skill' && params.level) {
    const most = page.world && page.world.max_level;
    what = `know-how, level ${params.level}` + (most ? ` of ${most}` : '');
  } else if (custom.kind === 'marriage' && params.within) {
    what += params.within === 'tongue' ?
        ', only one who speaks as they do' : ', only within their own people';
  }
  return `${what} (about ${custom.domain})`;
}

/**
 * A year, as the page writes it.
 * @param {number} year The year.
 * @return {string} E.g. '30,193'.
 */
function yearText(year) {
  return Number(year).toLocaleString('en');
}

/**
 * The world's name, as the page writes it.
 * @param {string} scenario The scenario or preset.
 * @return {string} E.g. 'The Sandbox', 'First Farmers'.
 */
function worldName(scenario) {
  if (scenario === 'sandbox') return 'The Sandbox';
  return scenario.split('_')
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

/** How a custom came to someone, by channel: alone, and from someone. */
const LEARNED_BY = {
  birth: ['had it from the start', 'inherited it from'],
  arrival: ['brought it into the valley', 'brought it in with'],
  upbringing: ['was taught it as a child', 'was taught it as a child by'],
  observation: ['saw it done', 'saw it done by'],
  apprenticeship: ['learned it as an apprentice',
    'learned it as an apprentice of'],
  sanction: ['was shamed into it', 'was shamed into it by'],
  practice: ['came to it by practice', 'came to it by practice with'],
  reading: ['read it in a text', 'read it in a text by'],
  school: ['learned it at school', 'learned it at school from'],
  rite: ['learned it at a rite', 'learned it at a rite led by'],
  story: ['heard it in a story', 'heard it in a story from'],
  preaching: ['heard it preached', 'heard it preached by'],
  release: ['had it from your release', 'had it from your release, via'],
};

/**
 * A kin's name as the family list shows it: the dead with when they died.
 * @param {!Object} kin An entry of the kin reply.
 * @return {string} E.g. 'Vira', 'Tone (died 63, aged 59)', 'Elor (forgotten)'.
 */
function kinName(kin) {
  if (kin.living) return kin.name;
  if (!kin.known) return `${kin.name} (forgotten)`;
  return `${kin.name} (died in year ${kin.died}, aged ` +
      `${kin.died - kin.born})`;
}

/**
 * What a lineage says of one version, tersely: the panel says once that a
 * new version is a copying error, and each list says what its rows are.
 * @param {!Object} variant A variant of the lineage reply.
 * @param {string} where 'held', 'kept', 'gone' (the lists, which say
 *     whether their versions are held) or 'tree' (which does not).
 * @return {string} E.g. 'held by 3 people · since year 9'; 'arose in year
 *     22 · the same words arose again in year 107'.
 */
function variantNote(variant, where) {
  const n = variant.holders.length;
  const year = variant.year.toLocaleString('en');
  const level = (variant.params || {}).level;
  const parts = variant.kind === 'skill' && level ? [`level ${level}`] : [];
  if (where === 'gone') {
    parts.push(`arose in year ${year}`);
  } else if (where === 'kept') {
    parts.push(`since year ${year}`);
  } else if (n) {
    parts.push(`held by ${n} ${n === 1 ? 'person' : 'people'}`,
        `since year ${year}`);
  } else if (variant.living) {
    parts.push('only in a library or shrine', `since year ${year}`);
  } else {
    parts.push('no longer held', `arose in year ${year}`);
  }
  if (variant.origin !== 'mutation') {
    parts.push(ARISEN_BY[variant.origin] || variant.origin);
  }
  const again = variant.again || [];
  if (again.length) {
    parts.push(`the same words arose again in ${again.length === 1 ?
      'year' : 'years'} ${again.map((y) => y.toLocaleString('en'))
        .join(', ')}`);
  }
  if (where === 'tree' && variant.returned) {
    parts.push('its words came back in a later copy');
  }
  return parts.join(' · ');
}

/**
 * A list of names, bounded: 'Ada', 'Ada and Bo', 'Ada, Bo and 3 more'.
 * @param {!Array<string>} names The names.
 * @param {number} shown How many to name at most.
 * @return {string} The list.
 */
function namesText(names, shown) {
  if (names.length > shown) {
    return `${names.slice(0, shown).join(', ')} and ` +
        `${names.length - shown} more`;
  }
  if (names.length < 2) return names.join('');
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

/**
 * Who holds a version, grouped by how they came by it (largest first).
 * @param {!Array<!Object>} holders {name, via, from} of each holder.
 * @param {number=} shown How many names each group lists at most.
 * @return {string} E.g. 'Ada and Bo inherited it from Mara; Cy saw it done
 *     by Ada.'
 */
function holdersText(holders, shown = 8) {
  const groups = new Map();
  for (const holder of holders) {
    const how = cameBy(holder);
    if (!groups.has(how)) groups.set(how, []);
    groups.get(how).push(holder.name);
  }
  return [...groups].sort((a, b) => b[1].length - a[1].length)
      .map(([how, names]) => `${namesText(names, shown)} ` +
          (names.length > 1 ? how.replace(/^was /, 'were ') : how))
      .join('; ') + '.';
}

/**
 * The label of a version's holders' fold: how many, and the first names.
 * @param {!Array<!Object>} holders {name, ...} of each holder.
 * @return {string} E.g. '9 hold it: Masa, Arvi, Luli and 6 more'.
 */
function holdersLabel(holders) {
  const n = holders.length;
  return `${n === 1 ? '1 holds' : `${n} hold`} it: ` +
      namesText(holders.map((h) => h.name), 3);
}

/**
 * What a list leaves out, if anything.
 * @param {number} listed How many are listed.
 * @param {number} count How many there are.
 * @return {string} E.g. 'the first 80 of 95 are listed', or ''.
 */
function moreNote(listed, count) {
  return listed < count ? `the first ${listed} of ${count} are listed` : '';
}

/**
 * The fold of a lineage's dead branches.
 * @param {number} count How many versions nobody holds any more.
 * @param {number} listed How many of them are listed (the latest).
 * @return {string} The fold's label.
 */
function goneNote(count, listed) {
  const versions = count === 1 ? '1 other version' : `${count} other versions`;
  return `${versions} that died out: nobody holds ${count === 1 ? 'it' :
    'them'}, and no library or shrine keeps ${count === 1 ? 'it' : 'them'}` +
      (listed < count ? `; the latest ${listed} are listed` : '');
}

/** How a version of a custom arose, by its origin. */
const ARISEN_BY = {
  founder: 'brought by the first settlers',
  arrival: 'brought in by newcomers',
  release: 'released by you',
  mutation: 'changed by a copying error',
  improvement: 'improved by a careful learner',
  known: 'came to be only expected of people, not held as their own',
  internalized: 'came to be held as people\'s own',
};

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
 * Shows a custom's lineage: its variants, who holds each and from whom.
 * @param {string} id A variant's id.
 * @param {boolean=} scroll Whether to bring the panel into view.
 */
async function showLineage(id, scroll = true) {
  const lineage = await api(`api/lineage?id=${encodeURIComponent(id)}`);
  page.lineage = id;
  page.lineageYear = page.world.year;
  document.getElementById('lineage-asof').textContent =
      `As of year ${yearText(page.world.year)}.`;
  document.getElementById('lineage-update').hidden = true;
  document.getElementById('lineage-title').textContent =
      `How a custom has changed: “${lineage.text}”`;
  const item = (variant, where) => {
    const li = document.createElement('li');
    if (where === 'tree') {
      li.style.marginLeft = `${Math.min(variant.depth, 6) * 12}px`;
    }
    const text = document.createElement('div');
    text.textContent = variant.text;
    if (variant.id === id) text.style.fontWeight = '600';
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = variantNote(variant, where) +
        (variant.id === id ? ' · the version you opened' : '');
    li.append(text, meta);
    if (variant.holders.length && where === 'held') {
      const who = document.createElement('details');
      const label = document.createElement('summary');
      label.textContent = holdersLabel(variant.holders);
      const names = document.createElement('div');
      names.className = 'holders';
      names.textContent = holdersText(variant.holders);
      who.append(label, names);
      li.append(who);
    }
    return li;
  };
  const fill = (key, variants, label) => {
    document.getElementById(`lineage-${key}`).hidden = !variants.length;
    document.getElementById(`lineage-${key}-label`).textContent = label;
    document.getElementById(`lineage-${key}-list`).replaceChildren(
        ...variants.map((variant) => item(variant, key)));
  };
  document.getElementById('lineage-held-none').hidden =
      lineage.held.length > 0;
  document.getElementById('lineage-held-list').replaceChildren(
      ...lineage.held.map((variant) => item(variant, 'held')));
  const heldMore = moreNote(lineage.held.length, lineage.held_count);
  document.getElementById('lineage-held-more').textContent =
      heldMore ? `${sentence(heldMore)}.` : '';
  fill('kept', lineage.kept, `Kept only in libraries or shrines ` +
      `(${lineage.kept_count}): nobody holds these, but someone may read or ` +
      'hear them there' + (lineage.kept.length < lineage.kept_count ?
        `; ${moreNote(lineage.kept.length, lineage.kept_count)}` : ''));
  const shown = lineage.variants.length;
  fill('tree', lineage.variants, `Its family tree (${shown}` +
      (lineage.more ? ` of ${shown + lineage.more} versions shown` :
        ' versions') + ': those held or kept, and the ones they came from)' +
      ': each is indented under the one it was copied from');
  fill('gone', lineage.gone,
      goneNote(lineage.gone_count, lineage.gone.length));
  document.getElementById('lineage-older').hidden = !lineage.lost_parent;
  const panel = document.getElementById('lineage');
  panel.hidden = false;
  if (scroll) panel.scrollIntoView({behavior: 'smooth', block: 'start'});
}

/**
 * Handles a tap on the map: a person, or the place for a release.
 * @param {!MouseEvent} event The click.
 */
function onMapTap(event) {
  if (!page.world) return;
  const canvas = event.currentTarget;  // Drawn in CSS pixels, tile by tile.
  const scale = (page.world.cols * page.tile) / canvas.clientWidth;
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
  // A place on the tile tapped, unless the tap was nearer a person on it.
  const place = page.world.places.find(
      (p) => p.tile[0] === row && p.tile[1] === col);
  const centre = ((col + 0.5) * page.tile - x) ** 2 +
      ((row + 0.5) * page.tile - y) ** 2;
  if (place && (!nearest || centre <= best)) {
    showPlace(place.id);
  } else if (nearest) {
    showPerson(nearest.name);
  }
}

/** What each kind of place holds, in words: [heading, none yet]. */
const PLACE_HOLDS = {
  library: ['Its shelf: texts anyone near may read', 'Nothing is on its ' +
      'shelf yet.'],
  school: ['What it teaches: its teacher\'s curriculum', 'Nobody has ' +
      'taught here yet.'],
  shrine: ['Its rite: what the rite is led by', 'Nobody has led its rite ' +
      'yet.'],
  court: ['Its law: the rules it last judged by', 'It has judged nobody ' +
      'yet.'],
};

/**
 * A place's title and what the panel says of it, in words.
 * @param {!Object} place The place reply: kind, tile, founder, founded,
 *     keeper, kept, used, ruined.
 * @return {!Object} {title, lines}.
 */
function placeText(place) {
  const at = `row ${place.tile[0]}, column ${place.tile[1]}`;
  const who = (name) => name === 'Experimenter' ? 'you' : name;
  const lines = [`Founded by ${who(place.founder)} in year ` +
      `${yearText(place.founded)}.`];
  if (place.keeper && place.kept >= 0 &&
      !(place.keeper === place.founder && place.kept === place.founded)) {
    lines.push(`Last kept by ${who(place.keeper)}, in year ` +
        `${yearText(place.kept)}.`);
  }
  lines.push(place.ruined ?
      `Unused since year ${yearText(place.used)}: a ruin. What it holds ` +
        'will crumble.' :
      `Last used in year ${yearText(place.used)}.`);
  return {
    title: `A ${place.kind} at ${at}${place.ruined ? ' (a ruin)' : ''}`,
    lines,
  };
}

/**
 * Shows a place: who founded and keeps it, and what it holds.
 * @param {string} id The place's id.
 */
async function showPlace(id) {
  const place = await api(`api/place?id=${encodeURIComponent(id)}`);
  const {title, lines} = placeText(place);
  document.getElementById('place-title').textContent = title;
  document.getElementById('place-body').replaceChildren(...lines.map((t) => {
    const item = document.createElement('li');
    item.textContent = t;
    return item;
  }));
  const [heading, none] = PLACE_HOLDS[place.kind] || ['What it holds', ''];
  document.getElementById('place-holds-label').textContent = heading;
  document.getElementById('place-holds-none').textContent = none;
  document.getElementById('place-holds-none').hidden = place.holds.length > 0;
  document.getElementById('place-holds').replaceChildren(
      ...place.holds.map((custom) => {
        const item = document.createElement('li');
        item.textContent = custom.text;
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = customKind(custom) + ' · ';
        const lineage = document.createElement('button');
        lineage.type = 'button';
        lineage.className = 'link';
        lineage.textContent = 'how it has changed';
        lineage.addEventListener('click', () => showLineage(custom.id));
        meta.append(lineage);
        item.append(meta);
        return item;
      }));
  const panel = document.getElementById('place');
  panel.hidden = false;
  panel.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

/**
 * What the footer says of the world's saves.
 * @param {!Object} world The state.
 * @return {string} The note.
 */
function savedNote(world) {
  const notes = [];
  if (world.restored_year != null) {
    notes.push(`picked up again at year ${yearText(world.restored_year)}`);
  }
  if (world.saved_year != null) {
    notes.push(`saved at year ${yearText(world.saved_year)}`);
  }
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
    followYear(world);
    if (world.pace.paused || !world.running) page.paceSeen = [];
    else page.paceSeen = [...page.paceSeen, [Date.now(), world.year]];
    page.paceSeen = page.paceSeen.filter(([ms]) => Date.now() - ms < 20000);
    document.getElementById('pace-real').textContent = paceNote(
        page.paceSeen, world.pace.years_per_second, Boolean(world.twin));
    document.getElementById('year').textContent =
        `${worldName(world.scenario)}, year ${yearText(world.year)}: ` +
        `${world.people.length} people alive` +
        (world.running ? '' : ' (the world has stopped)');
    document.getElementById('anew-box').hidden = world.running;
    document.getElementById('play-as-box').hidden = !world.playable;
    document.getElementById('stop').hidden = !world.running;
    document.getElementById('play').textContent =
        world.pace.paused ? 'Play' : 'Pause';
    for (const id of ['queued', 'fate-queued']) {
      document.getElementById(id).textContent = queuedNote(world.pace.queued);
    }
    document.getElementById('error').textContent = world.error || '';
    document.getElementById('saved').textContent = savedNote(world);
    document.getElementById('minds').textContent = world.minds === 'mock' ?
        'people think by simple rules (no language model)' :
        `people think with a language model (${world.minds})`;
    if (!page.atChosen) {
      page.at = crowdedSpot(world);
      document.getElementById('at').value = page.at ? page.at.join(',') : '';
    }
    drawMap(world);
    drawPeopleKey(world);
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

/** The label a person's context gives its skills (components/culture.py). */
const SKILL_LABEL = 'Something we know how to do';

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
 * What the played person knows and remembers, for a person to read: the
 * tags by which minds learn a custom, and customs' ids, are taken out, and
 * each thing the person saw, and each year's status line, begins a line.
 * @param {string} context The person's context.
 * @param {string} name The person.
 * @return {string} The context, readable.
 */
function lifeMemory(context, name) {
  const who = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return context
      .replace(/ \|params=\{.*?\}(?= [A-Z[]|\s*$)/gm, '')
      .replace(/\[custom id=c\d+[^\]\n]*\] /g, '')
      .replace(/ \[c\d+\]:/g, ':')
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
 * The know-how a move is done by, to follow in it: food skills make
 * gathering yield more, farming know-how makes a sown field take root, and
 * reading and writing need the skill of it.
 * @param {string} move The move.
 * @param {!Array<!Object>} customs The person's customs (`lifeCustoms`).
 * @return {!Array<string>} The ids of the skills the move is done by.
 */
function lifeSuggested(move, customs) {
  const domain = move.startsWith('gather food') ? 'food' :
      move === 'sow a field' ? 'farming' :
      / at the library$/.test(move) ? 'learning' : '';
  return customs.filter(
      (custom) => custom.label === SKILL_LABEL && custom.domain === domain)
      .map((custom) => custom.id);
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
        'You will be born again as the next child born of their line; if ' +
        'their line is gone, of their people; if they are gone too, as the ' +
        'next child born in the valley.';
  }
  const asked = played.asked;
  if (!asked) return 'The others are choosing; the year turns.';
  const year = asked.year >= 0 ? `Year ${yearText(asked.year)}: the` : 'The';
  return `${year} year waits for you. Answer within ` +
      `${asked.seconds_left}\u00a0s, or ${asked.name || played.name}, by ` +
      `habit, ${asked.unanswered}.`;
}

/**
 * The lives the player has lived, if more than one.
 * @param {!Object} played The state's `played`.
 * @return {string} The lives, oldest first ('' for a first life).
 */
function livesNote(played) {
  const lives = played.lives || [];
  return lives.length > 1 ? `Your lives: ${lives.join(' → ')}` : '';
}

/**
 * How far the player's ways have reached: the living who learned a custom
 * from one of its lives.
 * @param {!Object} played The state's `played`.
 * @return {string} The line.
 */
function influenceNote(played) {
  const reach = played.reach || {people: 0, customs: 0};
  if (!reach.people) return 'Nobody living has learned a custom from you yet.';
  const people = reach.people === 1 ? '1 living person has' :
      `${reach.people} living people have`;
  const customs = reach.customs === 1 ? 'a custom' : `${reach.customs} customs`;
  return `${people} learned ${customs} from you.`;
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
  document.getElementById('life-lives').textContent = livesNote(played);
  document.getElementById('life-reach').textContent = influenceNote(played);
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
  document.getElementById('life-status').replaceChildren(
      ...statusLines(played.status || '').map((line) => {
        const item = document.createElement('li');
        item.textContent = line;
        return item;
      }));
  document.getElementById('life-question').textContent = lifeQuestion(asked);
  document.getElementById('life-context').textContent =
      lifeMemory(asked.context, played.name);
  const passing = asked.kind === 'pass_on';
  const interview = asked.kind === 'interview';
  document.getElementById('life-moves').hidden = asked.kind !== 'move';
  document.getElementById('life-customs').hidden = interview;
  document.getElementById('life-say-box').hidden = !interview;
  document.getElementById('life-say').value = '';
  const moves = lifeMoves(asked.question);
  const customs = lifeCustoms(asked.context);
  const suggested = new Set(passing ? customs.map((c) => c.id) :
      lifeSuggested(moves[0] || '', customs));
  document.getElementById('life-move-list').replaceChildren(
      ...moves.map(
          (move, i) => choiceInput('radio', 'life-move', move, move, i === 0)));
  document.getElementById('life-customs-legend').textContent = passing ?
      'The customs you pass on to your child' :
      'The customs you follow in it, if any. Your know-how makes the work ' +
      'go better (a field takes root only by farming know-how), and those ' +
      'who see you may take up what you follow.';
  document.getElementById('life-custom-list').replaceChildren(
      ...customs.map((custom) => choiceInput(
          'checkbox', 'life-custom', custom.id, custom.text,
          suggested.has(custom.id),
          `${custom.label} (about ${custom.domain})`)));
}

/**
 * Follows the know-how a newly chosen move is done by (`lifeSuggested`).
 * @param {!Event} event The move's change.
 */
function onLifeMove(event) {
  const asked = page.world && page.world.played && page.world.played.asked;
  if (!asked || event.target.name !== 'life-move') return;
  const suggested = new Set(
      lifeSuggested(event.target.value, lifeCustoms(asked.context)));
  for (const box of document.querySelectorAll('input[name="life-custom"]')) {
    box.checked = suggested.has(box.value);
  }
}

/**
 * The map's key: each kind of land, and what its shading means.
 * @param {!Array<string>} kinds The kinds of land in this world.
 */
function drawMapKey(kinds) {
  const key = document.getElementById('map-key');
  key.replaceChildren('Land:');
  for (const kind of kinds) {
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = TERRAIN_COLOURS[kind] || '#999';
    key.append(swatch, kind);
  }
  key.append('. The darker the land, the less food it holds now.');
}

/**
 * The key to the people's colours: each descent among the living, most
 * first, with how many; and whether the map marks you.
 * @param {!Object} world The state.
 */
function drawPeopleKey(world) {
  const key = document.getElementById('people-key');
  key.replaceChildren();
  overlayKey(world, page.overlay).forEach(([colour, label], i) => {
        const swatch = document.createElement('span');
        swatch.className = 'swatch';
        swatch.style.background = colour;
        key.append(i ? ', ' : '', swatch, label);
      });
  document.getElementById('key-you').hidden = !world.played;
}

/** Food stored, in bands, for the food overlay: [below, colour, words]. */
const FOOD_BANDS = [
  [1, '#d62828', 'starving (under 1)'],
  [3, '#f77f00', 'short (1 to 3)'],
  [6, '#fcbf49', 'enough (3 to 6)'],
  [Infinity, '#ffffff', 'plenty (6 or more)'],
];
/** The age from which a person is an elder (the World's `ELDER_AGE`). */
const ELDER_AGE = 56;

/**
 * Ranks the tongues spoken, most speakers first.
 * @param {!Array<!Object>} people The living, with `speech`.
 * @return {!Map<string, number>} Each tongue's rank.
 */
function tongueRanks(people) {
  const counts = new Map();
  for (const p of people) counts.set(p.speech, (counts.get(p.speech) || 0) + 1);
  return new Map([...counts].sort((a, b) => b[1] - a[1])
      .map(([speech], i) => [speech, i]));
}

/**
 * A person's colour on the map, by the overlay chosen.
 * @param {!Object} person The person, as the state gives them.
 * @param {string} overlay 'descent', 'people', 'tongue', 'food' or 'age'.
 * @param {!Object} world The state (its `grown_age`).
 * @param {!Map<string, number>} tongues The tongues' ranks.
 * @return {string} The colour.
 */
function personColour(person, overlay, world, tongues) {
  const ranked = (i) => i < 0 ? '#9a9a9a' :
    PEOPLE_COLOURS[i % PEOPLE_COLOURS.length];
  if (overlay === 'people') return ranked(person.people ?? -1);
  if (overlay === 'tongue') return ranked(tongues.get(person.speech) ?? -1);
  if (overlay === 'food') {
    return FOOD_BANDS.find(([below]) => person.food < below)[1];
  }
  if (overlay === 'age') {
    return person.age < world.grown_age ? '#ffffff' :
        person.age >= ELDER_AGE ? '#1a1a1a' : '#d62828';
  }
  return peopleColour(person.descent || 'valley');
}

/**
 * The key to the map's colours for the overlay chosen.
 * @param {!Object} world The state.
 * @param {string} overlay As `personColour`'s.
 * @return {!Array<!Array<string>>} [colour, words] of each, most first.
 */
function overlayKey(world, overlay) {
  const tongues = tongueRanks(world.people);
  const groups = new Map();
  for (const p of world.people) {
    const colour = personColour(p, overlay, world, tongues);
    if (!groups.has(colour)) groups.set(colour, []);
    groups.get(colour).push(p);
  }
  const mostly = (people) => {
    const counts = new Map();
    for (const p of people) {
      counts.set(p.descent, (counts.get(p.descent) || 0) + 1);
    }
    return [...counts].sort((a, b) => b[1] - a[1])[0][0] || 'valley';
  };
  const label = (colour, people) => {
    const n = people.length;
    const one = people[0];
    if (overlay === 'people') {
      return one.people < 0 ?
        `children and newcomers, of no people yet (${n})` :
        `a people of ${n}, mostly of the ${mostly(people)}`;
    }
    if (overlay === 'tongue') {
      return `a tongue spoken by ${n}, mostly of the ${mostly(people)}`;
    }
    if (overlay === 'food') {
      return `${FOOD_BANDS.find(([, c]) => c === colour)[2]} (${n})`;
    }
    if (overlay === 'age') {
      return `${one.age < world.grown_age ? 'children' :
        one.age >= ELDER_AGE ? 'elders' : 'grown'} (${n})`;
    }
    return `the ${one.descent || 'valley'} (${n})`;
  };
  return [...groups].sort((a, b) => b[1].length - a[1].length)
      .map(([colour, people]) => [colour, label(colour, people)]);
}

/**
 * A move, in words: the page's moves are written in the form the World
 * reads ("release norm kinship as a song at 8,9: ..."); the page shows them
 * as a person would say them.
 * @param {string} move A move as the World reads it.
 * @return {string} The move in words (a move it does not know, as it is).
 */
function moveWords(move) {
  const at = (r, c) => `row ${r}, column ${c}`;
  const release = new RegExp('^release (\\w+) (\\w+)(?: level (\\d+))?' +
      '(?: within (\\w+))? as a (text|song) at (\\d+),(\\d+): (.+)$');
  let m = move.match(release);
  if (m) {
    const [, kind, domain, level, within, how, r, c, says] = m;
    const allows = within === 'tongue' ? ', only one who speaks as we do' :
        within === 'descent' ? ', only our own people' : '';
    const good = level ? `, level ${level}` : '';
    const as = how === 'song' ? 'Sing' : 'Leave as a text';
    return `${as} “${says}” at ${at(r, c)}: ` +
        `${KIND_WORDS[kind] || kind}${good}${allows} (about ${domain})`;
  }
  m = move.match(/^send a (\w+) at (\d+),(\d+)$/);
  if (m) return `Send a ${m[1]} at ${at(m[2], m[3])}`;
  m = move.match(/^bring migrants at (\d+),(\d+)$/);
  if (m) return `Bring newcomers to ${at(m[1], m[2])}`;
  m = move.match(/^(found|close) (?:a|the) (\w+) at (\d+),(\d+)$/);
  if (m) {
    return `${m[1] === 'found' ? 'Found a' : 'Close the'} ${m[2]} ` +
        `${m[1] === 'found' ? 'at' : 'nearest'} ${at(m[3], m[4])}`;
  }
  m = move.match(/^ask ([^:]+): (.+)$/);
  if (m) return `Ask ${m[1]}: “${m[2]}”`;
  if (move === 'stop the world') return 'End this world';
  return move;
}

/**
 * An interview answer as the page shows it: a mind's answer tells what the
 * person says ("Kaka says: ..."), so it stands as written; anything else,
 * a player's words or a mind's new sentence after its name ("Kaka Because
 * we must."), is what the person says, quoted without the name. As the
 * World's `said_by` and `speech`.
 * @param {!Object} transcript The answer, with who gave it (`name`).
 * @return {string} Its text.
 */
function answerText(transcript) {
  const answer = transcript.answer.trim();
  const name = transcript.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  if (new RegExp(`^${name}(?:\\s+[a-z]|,)`).test(answer)) return answer;
  const said = answer.replace(new RegExp(`^${name}(?![\\w-])[\\s:]*`), '');
  return `"${said.trim()}"`;
}

/**
 * A notebook entry's text as the page shows it: a move or a twin in words,
 * the World's and the valley's entries with any place of an old save in
 * words (`OLD_PLACES`), and the person's own notes as written.
 * @param {!Object} entry The entry.
 * @return {string} Its text.
 */
function entryText(entry) {
  if (entry.kind === 'world' || entry.kind === 'event') {
    return OLD_PLACES.reduce((text, [form, words]) =>
      text.replace(form, words), entry.text);  // Already in words since.
  }
  if (entry.kind !== 'move') return entry.text;
  let m = entry.text.match(new RegExp(
      '^(?:Your move for next year(?:, as the World reads it)?|' +
      'You queue a move): (.+)$'));
  if (m) return `Your move for next year: ${moveWords(m[1])}`;
  m = entry.text.match(new RegExp(
      '^You fork a twin in year (\\d+): (?:without (\\w+)|(\\w+) (\\S+))\\.$'));
  if (m) {
    const pulled = m[2] ?
        `without ${CHANNEL_WORDS[m[2]] || m[2]}` :
        `${(LEVER_WORDS[m[3]] || m[3]).toLowerCase()} set to ${m[4]}`;
    return `You fork a twin in year ${yearText(m[1])}: ${pulled}.`;
  }
  return entry.text;
}

/**
 * The World's and the session's own sentences that named places "r,c", or
 * customs by id, before they named them in words, as notebooks saved then
 * hold them: each anchored to its form, so a quoted text is never read.
 */
const OLD_PLACES = [
  // The oldest forms named a release by its id.
  [/^c\d+ is sung at (\d+),(\d+); (\d+) hears? it\.$/,
    (_, r, c, n) => `Your release is sung at row ${r}, column ${c}; ${n} ` +
      `${n === '1' ? 'hears' : 'hear'} it.`],
  [/^c\d+ (is written at the library at) (\d+),(\d+)\.$/,
    'Your release $1 row $2, column $3.'],
  [/^Your release c\d+ died out: (".*") (\(at most \d+ held it\))/,
    'Your release $1 died out $2'],
  [/^c\d+ is the most widely held custom for the first time \((\d+)% hold it\)/,
    'A custom is the most widely held for the first time ($1% of the living' +
      ' hold it)'],
  [/^(A \w+ str(?:uck|ikes) at) (\d+),(\d+):/, '$1 row $2, column $3:'],
  [/^(Your release ".*" is sung at) (\d+),(\d+)(; \d+ hears? it\.)$/,
    '$1 row $2, column $3$4'],
  [/^(Your release ".*" is written at the library at) (\d+),(\d+)\.$/,
    '$1 row $2, column $3.'],
  [/^(\d+ of the .+ come to) (\d+),(\d+):/, '$1 row $2, column $3:'],
  [/^(A \w+ is founded at) (\d+),(\d+);/, '$1 row $2, column $3;'],
  [/^(A \w+ stands within \d+ steps of) (\d+),(\d+)\.$/,
    '$1 row $2, column $3.'],
  [/^(No \w+ stands at) (\d+),(\d+)\.$/, '$1 row $2, column $3.'],
  [/^(The \w+ at) (\d+),(\d+)( is closed;)/, '$1 row $2, column $3$4'],
];

/**
 * Words begun as a sentence.
 * @param {string} words E.g. 'a rule'.
 * @return {string} E.g. 'A rule'.
 */
function sentence(words) {
  return words.charAt(0).toUpperCase() + words.slice(1);
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
  document.getElementById('kind').replaceChildren(...world.custom_kinds.map(
      (kind) => new Option(sentence(KIND_WORDS[kind] || kind), kind)));
  fill('domain', domainsFor(world, document.getElementById('kind').value));
  document.getElementById('kind').addEventListener('change', (event) => {
    const kind = event.target.value;
    const domain = document.getElementById('domain');
    const about = domain.value;  // Kept if the new kind can be about it.
    fill('domain', domainsFor(page.world || world, kind));
    if ([...domain.options].some((option) => option.value === about)) {
      domain.value = about;
    }
    document.getElementById('says').placeholder =
        EXAMPLE_SAYS[kind] || EXAMPLE_SAYS.norm;
    document.getElementById('within-box').hidden = kind !== 'marriage';
    document.getElementById('level-box').hidden = kind !== 'skill';
  });
  drawMapKey(world.kinds);
  document.getElementById('level').replaceChildren(
      ...Array.from({length: world.max_level}, (_, i) => new Option(
          String(i + 1), String(i + 1), false, i + 1 === 3)));
  fill('fate-kind', fateChoices(world));
  document.getElementById('preset').replaceChildren(...world.presets.map(
      (preset) => new Option(worldName(preset), preset)));
  document.getElementById('preset').value = world.scenario;
  const presetNote = () => {
    document.getElementById('preset-note').textContent =
        (world.preset_notes || {})[document.getElementById('preset').value] ||
        '';
  };
  document.getElementById('preset').addEventListener('change', presetNote);
  presetNote();
  document.getElementById('lever').replaceChildren(...leverChoices(world).map(
      ([value, label]) => new Option(label, value)));
  const leverTo = () => {  // A channel goes without; it takes no value.
    document.getElementById('lever-to').hidden =
        document.getElementById('lever').value.startsWith('without ');
  };
  leverTo();
  document.getElementById('lever').addEventListener('change', leverTo);
  document.getElementById('how').addEventListener('change', () => {
    if (page.world) drawAt(page.world);
  });
  const build = await api('api/build');
  document.getElementById('build').textContent =
      `Lifeways, build ${build.commit || '?'} (the code's version)`;
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
  document.getElementById('place-close').addEventListener('click', () => {
    document.getElementById('place').hidden = true;
  });
  document.getElementById('lineage-close').addEventListener('click', () => {
    document.getElementById('lineage').hidden = true;
    page.lineage = null;
  });
  document.getElementById('person-anatomy').addEventListener('toggle',
      (event) => {
        if (event.target.open && page.person) {
          showAnatomy('person-anatomy-list', page.person);
        }
      });
  document.getElementById('world-anatomy').addEventListener('toggle',
      (event) => {
        if (event.target.open) showAnatomy('world-anatomy-list', '');
      });
  const overlay = document.getElementById('overlay');
  overlay.value = page.overlay;
  overlay.addEventListener('change', () => {
    page.overlay = overlay.value;
    remember('lifeways.overlay', page.overlay);
    if (page.world) {
      drawMap(page.world);
      drawPeopleKey(page.world);
    }
  });
  document.getElementById('lineage-update').addEventListener('click', () => {
    if (page.lineage) showLineage(page.lineage, false);
  });
  document.getElementById('play').addEventListener('click', async () => {
    await api('api/pace', {paused: !page.world.pace.paused});
    refresh();
  });
  document.getElementById('speed').addEventListener('change', async (event) => {
    page.paceSeen = [];
    await api('api/pace', {years_per_second: Number(event.target.value)});
  });
  document.getElementById('stop').addEventListener('click', async () => {
    if (confirm('End this world for good?')) await api('api/stop', {});
  });
  document.getElementById('anew').addEventListener('click', async () => {
    const scenario = document.getElementById('preset').value;
    if (!confirm(`Begin a new world (${worldName(scenario)})? This one ` +
        'is set aside.')) {
      return;
    }
    const said = document.getElementById('anew-error');
    said.textContent = '';
    try {
      await api('api/new',
          {scenario, play_as: document.getElementById('play-as').checked});
    } catch (error) {
      said.textContent = `No new world: ${error.message}`;
      return;
    }
    Object.assign(page, {rows: [], since: -1, twinRows: [], twinSince: -1,
      twin: null, entries: [], entrySince: 0, person: null, at: null,
      asked: null});
    refresh();
  });
  document.getElementById('life-moves').addEventListener('change', onLifeMove);
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
    const note = document.getElementById('ask-note');
    if (!page.person) return;
    if (!question) {
      note.textContent = 'Write a question first.';
      return;
    }
    await api('api/move', {move: askMove(page.person, question)});
    document.getElementById('question').value = '';
    note.textContent = askedNote(page.person, page.world.year,
        page.world.pace.paused);
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
        document.getElementById('within').value,
        Number(document.getElementById('level').value));
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
      said.textContent = 'Not forked: give the lever a value, a number.';
      return;
    }
    document.getElementById('twin').textContent = 'Forking at the year\'s end…';
    try {
      await api('api/twin', {lever: name, value: channel});
    } catch (error) {
      said.textContent = `Not forked: ${error.message}`;
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
    anatomyLines, answerText, askedNote, askMove, cameBy, letGoText,
    overlayKey, personColour, placeText, queuedNote, tongueRanks,
    statusLines,
    goneFromPanel, paceNote,
    peoplesText,
    chartSeries,
    crowdedSpot, customKind,
    domainsFor, kinName, entryText, moveWords, worldName, yearText,
    goneNote, leverChoices, moreNote, peopleColour, releaseMove,
    appendNew, entryMeta, entryShown,
    fateChoices, fateMove, influenceNote, lifeAnswer, lifeCustoms, lifeMemory,
    lifeMoves, lifeNote, lifeQuestion, lifeSuggested, livesNote,
    reachNote, recentDisasters, releaseStatus, savedNote,
    holdersLabel, holdersText, namesText, twinNote, variantNote, withinReach,
  };
}
