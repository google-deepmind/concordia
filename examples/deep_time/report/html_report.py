# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A self-contained HTML report for a Deep Time run.

No external scripts or stylesheets: charts are inline SVG with a small hover
script, so the file opens anywhere. Sections: outcome and LLM economy, time
series (one measure per chart), cognition meters for every slow mind, the
Chronicler's chapters, the Advisor's briefings, the event log, the topic graph
and the raw data table.
"""

# pylint: disable=g-complex-comprehension,missing-function-docstring

from collections.abc import Mapping, Sequence
import dataclasses
import html
import json
import math
from typing import Any

from examples.deep_time import game as game_lib

# Categorical slots in fixed order (light, dark), validated as a set.
# Reference palette slots 1-8 in their validated order: (light, dark).
SLOTS = (
    ('#2a78d6', '#3987e5'),
    ('#eb6834', '#d95926'),
    ('#1baf7a', '#199e70'),
    ('#eda100', '#c98500'),
    ('#e87ba4', '#d55181'),
    ('#008300', '#008300'),
    ('#4a3aa7', '#9085e9'),
    ('#e34948', '#e66767'),
)


def _slot_vars(mode: int) -> str:
  return ';'.join(f'--s{i + 1}:{pair[mode]}' for i, pair in enumerate(SLOTS))


STATUS = {'victory': ('#0ca30c', '✓'), 'defeat': ('#d03b3b', '✕')}

_CSS = """
.dt-report{color-scheme:light;--surface:#fcfcfb;--surface-2:#f3f2ef;
--text:#0b0b0b;--text-2:#52514e;--muted:#8a8984;--grid:#e4e3df;
--band:#e9f2e9;@LIGHT@;
font:15px/1.5 system-ui,-apple-system,sans-serif;color:var(--text);
background:var(--surface);max-width:980px;margin:0 auto;padding:16px}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"]))
.dt-report{color-scheme:dark;--surface:#1a1a19;--surface-2:#242422;
--text:#fff;--text-2:#c3c2b7;--muted:#8f8e86;--grid:#33332f;--band:#1f2d20;
@DARK@}
.dt-report .cell{fill:var(--fd)}.dt-report .cell-label{fill:var(--td)}}
:root[data-theme="dark"] .dt-report .cell{fill:var(--fd)}
:root[data-theme="dark"] .dt-report .cell-label{fill:var(--td)}
:root[data-theme="dark"] .dt-report{color-scheme:dark;--surface:#1a1a19;
--surface-2:#242422;--text:#fff;--text-2:#c3c2b7;--muted:#8f8e86;
--grid:#33332f;--band:#1f2d20;@DARK@}
body{margin:0;background:var(--surface)}
.dt-report h1{font-size:24px;margin:8px 0 4px}
.dt-report h2{font-size:18px;margin:28px 0 8px}
.dt-report .sub{color:var(--text-2);margin:0 0 12px}
.badge{display:inline-flex;gap:6px;align-items:center;padding:2px 10px;
border-radius:999px;background:var(--surface-2);font-weight:600}
.badge i{font-style:normal;color:var(--status)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
gap:8px;margin:12px 0}
.tile{background:var(--surface-2);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:22px}.tile span{color:var(--text-2);font-size:13px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
gap:16px}
figure{margin:0;position:relative}
figcaption{font-weight:600;margin-bottom:4px}
.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--text-2);
font-size:13px;margin:2px 0 4px}
.legend span::before{content:"";display:inline-block;width:12px;height:3px;
border-radius:2px;background:var(--c);margin-right:6px;vertical-align:middle}
svg text{fill:var(--text-2);font-size:11px}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .band{fill:var(--band)}
svg .cross{stroke:var(--muted);stroke-width:1;visibility:hidden}
svg .cell{fill:var(--fl)}svg .cell-label{fill:var(--tl);font-size:10px}
.scale{height:10px;border-radius:2px;margin:4px 0 2px;max-width:320px;
background:linear-gradient(90deg,var(--gl))}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"]))
.scale{background:linear-gradient(90deg,var(--gd))}}
:root[data-theme="dark"] .scale{background:linear-gradient(90deg,var(--gd))}
.scale-labels{display:flex;justify-content:space-between;max-width:320px;
color:var(--text-2);font-size:12px}
.tip{position:absolute;pointer-events:none;background:var(--surface);
border:1px solid var(--grid);border-radius:6px;padding:6px 8px;
font-size:12px;color:var(--text);visibility:hidden;white-space:nowrap;
box-shadow:0 2px 6px rgba(0,0,0,.12)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid);
vertical-align:top}
th{color:var(--text-2);font-weight:600}
.scroll{overflow-x:auto}
ol.chapters li{margin-bottom:8px}
details{margin:8px 0}summary{cursor:pointer;color:var(--text-2)}
pre{background:var(--surface-2);padding:8px;border-radius:6px;overflow-x:auto;
font-size:12px}
""".replace('@LIGHT@', _slot_vars(0)).replace('@DARK@', _slot_vars(1))

_SCRIPT = """
document.querySelectorAll('figure[data-series]').forEach(function(fig){
  var d=JSON.parse(fig.dataset.series),svg=fig.querySelector('svg'),
      tip=fig.querySelector('.tip'),cross=svg.querySelector('.cross'),
      hit=svg.querySelector('.hit');
  function show(ev){
    var r=svg.getBoundingClientRect(),vb=svg.viewBox.baseVal;
    var x=(ev.clientX-r.left)*vb.width/r.width;
    var i=Math.round((x-d.x0)/d.dx);i=Math.max(0,Math.min(d.ticks.length-1,i));
    var px=d.x0+i*d.dx;cross.setAttribute('x1',px);cross.setAttribute('x2',px);
    cross.style.visibility='visible';
    var rows=['<b>tick '+d.ticks[i]+'</b>'];
    d.series.forEach(function(s){rows.push(s.label+': '+s.values[i]);});
    tip.innerHTML=rows.join('<br>');tip.style.visibility='visible';
    var left=(px*r.width/vb.width)+12;
    if(left>r.width-140){left-=160;}
    tip.style.left=left+'px';tip.style.top='28px';
  }
  hit.addEventListener('mousemove',show);
  hit.addEventListener('mouseleave',function(){
    cross.style.visibility='hidden';tip.style.visibility='hidden';});
});
"""


# Diverging temperature scale: cold blue <-> neutral <-> hot red
# (reference palette: blue ramp, neutral midpoint, red slot), per mode.
DIVERGING = {
    'light': ('#184f95', '#f0efec', '#e34948'),
    'dark': ('#5598e7', '#383835', '#e66767'),
}
TEMP_SCALE = (-10.0, 22.5, 55.0)  # cold end, daisy optimum, hot end (C)


def _rgb(color: str) -> tuple[int, int, int]:
  return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))


def _mix(a: str, b: str, t: float) -> str:
  ra, rb = _rgb(a), _rgb(b)
  return '#%02x%02x%02x' % tuple(round(x + (y - x) * t) for x, y in zip(ra, rb))


def diverging(value: float, mode: str = 'light') -> str:
  """Colour of a temperature on the diverging scale."""
  cold, mid, hot = DIVERGING[mode]
  low, centre, high = TEMP_SCALE
  if value <= centre:
    t = (min(max(value, low), centre) - low) / (centre - low)
    return _mix(cold, mid, t)
  t = (min(value, high) - centre) / (high - centre)
  return _mix(mid, hot, t)


def _ink(fill: str) -> str:
  r, g, b = (c / 255 for c in _rgb(fill))
  luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
  return '#0b0b0b' if luminance > 0.55 else '#ffffff'


def _cell(
    x: float,
    y: float,
    w: float,
    h: float,
    value: float,
    title: str,
    label: str = '',
) -> str:
  light, dark = diverging(value, 'light'), diverging(value, 'dark')
  style = f'--fl:{light};--fd:{dark};--tl:{_ink(light)};--td:{_ink(dark)}'
  text = ''
  if label:
    text = (
        f'<text class="cell-label" x="{x + w / 2:.1f}" y="{y + h / 2 + 3:.1f}"'
        f' text-anchor="middle" style="{style}">{_esc(label)}</text>'
    )
  return (
      f'<rect class="cell" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}"'
      f' height="{h:.1f}" style="{style}"><title>{_esc(title)}</title></rect>'
      + text
  )


def temperature_legend() -> str:
  low, centre, high = TEMP_SCALE
  return (
      '<div class="scale" style="--gl:{};--gd:{}"></div>'.format(
          ','.join(DIVERGING['light']), ','.join(DIVERGING['dark'])
      )
      + f'<div class="scale-labels"><span>{low:g} °C</span><span>{centre:g} °C'
      f' (daisy optimum)</span><span>{high:g} °C</span></div>'
  )


def latitude_heatmap(
    rows: Sequence[Mapping[str, Any]], band_names: Sequence[str]
) -> str:
  """Temperature by latitude band (north at top) against time."""
  bands = len(band_names)
  width, left, top = 460, 44, 4
  cell_h = 18
  plot_w = width - left - 8
  cell_w = plot_w / max(1, len(rows))
  height = top + bands * cell_h + 20
  parts = [
      f'<svg viewBox="0 0 {width} {height}" width="100%" role="img"'
      ' aria-label="Temperature by latitude over time">'
  ]
  for row_index, band in enumerate(reversed(range(bands))):
    y = top + row_index * cell_h
    parts.append(
        f'<text x="{left - 6}" y="{y + cell_h / 2 + 4:.1f}"'
        f' text-anchor="end">{_esc(band_names[band])}</text>'
    )
    for i, row in enumerate(rows):
      value = float(row[f'temp_band{band}'])
      parts.append(
          _cell(
              left + i * cell_w,
              y,
              cell_w + 0.5,
              cell_h - 1,
              value,
              f'tick {row["tick"]} · {band_names[band]} · {value:.1f} °C',
          )
      )
  for k in range(5):
    i = round(k * (len(rows) - 1) / 4)
    parts.append(
        f'<text x="{left + (i + 0.5) * cell_w:.1f}" y="{height - 4}"'
        f' text-anchor="middle">{rows[i]["tick"]}</text>'
    )
  parts.append('</svg>')
  return (
      '<figure><figcaption>Temperature by latitude over time</figcaption>'
      + ''.join(parts)
      + '</figure>'
  )


def planet_map(context: game_lib.GameContext) -> str:
  """Final state of every region: temperature fill, dominant cover label."""
  planet = context.planet
  grid = planet.grid
  daisies = planet.daisies
  temps = daisies.temperatures_c()
  ice = daisies.ice_map()
  covers = {s: daisies.cover_map(s) for s in daisies.species()}
  width, left, top = 460, 44, 4
  cell_w = (width - left - 8) / grid.sectors
  cell_h = 30
  height = top + grid.bands * cell_h + 4
  parts = [
      f'<svg viewBox="0 0 {width} {height}" width="100%" role="img"'
      ' aria-label="Planet map at the final tick">'
  ]
  for row_index, band in enumerate(reversed(range(grid.bands))):
    y = top + row_index * cell_h
    parts.append(
        f'<text x="{left - 6}" y="{y + cell_h / 2 + 4:.1f}"'
        f' text-anchor="end">{_esc(grid.band_name(band))}</text>'
    )
    for sector in range(grid.sectors):
      r = grid.region(band, sector)
      dominant = max(covers, key=lambda s, r=r: covers[s][r])
      share = covers[dominant][r]
      land = daisies.land_fraction()[r]
      if ice[r] > 0.5:
        label = 'ice'
      elif land < 0.25:
        label = 'sea'
      elif share >= 0.05:
        label = f'{dominant[0].upper()}{share * 100:.0f}'
      else:
        label = 'bare'
      title = (
          f'{grid.region_name(r)} · land {land:.0%} · {temps[r]:.1f} °C ·'
          f' ice {ice[r]:.0%} · '
          + ', '.join(f'{s} {covers[s][r]:.0%}' for s in covers)
      )
      parts.append(
          _cell(
              left + sector * cell_w,
              y,
              cell_w - 2,
              cell_h - 2,
              float(temps[r]),
              title,
              label,
          )
      )
  parts.append('</svg>')
  return (
      '<figure><figcaption>Planet map, final tick (W/B = dominant daisy,'
      ' % of region; sea; ice)</figcaption>'
      + ''.join(parts)
      + '</figure>'
  )


@dataclasses.dataclass(frozen=True)
class Series:
  key: str
  label: str
  slot: int  # 1-based categorical slot; colour follows the entity
  short: str = ''  # direct label at the line end (defaults to label)


def nice_axis(low: float, high: float, intervals: int = 4) -> list[float]:
  """Returns evenly spaced 1-2-2.5-5 ticks covering [low, high]."""
  if high <= low:
    high = low + 1.0
  raw = (high - low) / intervals
  magnitude = 10 ** math.floor(math.log10(raw))
  step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
  start = math.floor(low / step + 1e-9) * step
  end = math.ceil(high / step - 1e-9) * step
  count = int(round((end - start) / step))
  return [start + i * step for i in range(count + 1)]


def _esc(value: Any) -> str:
  return html.escape(str(value))


def _fmt(value: float) -> str:
  return f'{value:.3g}' if abs(value) < 1000 else f'{value:,.0f}'


def _segments(
    values: Sequence[float | None],
) -> list[list[tuple[int, float]]]:
  """Splits a series into runs of present (index, value) points."""
  runs: list[list[tuple[int, float]]] = [[]]
  for i, v in enumerate(values):
    if v is None:
      if runs[-1]:
        runs.append([])
    else:
      runs[-1].append((i, v))
  return [run for run in runs if run]


def line_chart(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    series: Sequence[Series],
    *,
    unit: str = '',
    band: tuple[float, float] | None = None,
    width: int = 460,
    height: int = 200,
) -> str:
  """Returns an SVG line chart (one y-axis) with a hover layer."""
  ticks = [int(r['tick']) for r in rows]
  # Missing values (a species before it was founded, a belief before the
  # first reading) leave gaps instead of drawing false zeros.
  values = {
      s.key: [None if r.get(s.key) is None else float(r[s.key]) for r in rows]
      for s in series
  }
  all_values = [v for vs in values.values() for v in vs if v is not None]
  if not all_values:
    all_values = [0.0]
  if band:
    all_values += list(band)
  axis = nice_axis(min(all_values), max(all_values))
  y_min, y_max = axis[0], axis[-1]
  left, right, top, bottom = 44, 52, 10, 24
  plot_w, plot_h = width - left - right, height - top - bottom
  span = max(1, ticks[-1] - ticks[0]) if ticks else 1
  dx = plot_w / max(1, len(ticks) - 1)

  def x_of(i: int) -> float:
    return left + i * dx

  def y_of(v: float) -> float:
    return top + plot_h * (1 - (v - y_min) / (y_max - y_min))

  parts = [
      f'<svg viewBox="0 0 {width} {height}" width="100%" role="img"'
      f' aria-label="{_esc(title)}">'
  ]
  if band:
    y_hi, y_lo = y_of(band[1]), y_of(band[0])
    parts.append(
        f'<rect class="band" x="{left}" y="{y_hi:.1f}" width="{plot_w}"'
        f' height="{y_lo - y_hi:.1f}"/>'
    )
  for v in axis:
    y = y_of(v)
    parts.append(
        f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}"'
        f' y2="{y:.1f}"/><text x="{left - 6}" y="{y + 4:.1f}"'
        f' text-anchor="end">{_fmt(v)}</text>'
    )
  for k in range(5):
    i = round(k * (len(ticks) - 1) / 4) if ticks else 0
    if ticks:
      parts.append(
          f'<text x="{x_of(i):.1f}" y="{height - 6}"'
          f' text-anchor="middle">{ticks[i]}</text>'
      )
  for s in series:
    for segment in _segments(values[s.key]):
      points = ' '.join(f'{x_of(i):.1f},{y_of(v):.1f}' for i, v in segment)
      parts.append(
          f'<polyline class="line" style="stroke:var(--s{s.slot})"'
          f' points="{points}"/>'
      )
    present = [v for v in values[s.key] if v is not None]
    if len(series) > 1 and present:
      end = present[-1]
      parts.append(
          f'<text x="{left + plot_w + 6}" y="{y_of(end) + 4:.1f}">'
          f'{_esc(s.short or s.label)}</text>'
      )
  parts.append(
      f'<line class="cross" x1="0" x2="0" y1="{top}" y2="{top + plot_h}"/>'
      f'<rect class="hit" x="{left}" y="{top}" width="{plot_w}"'
      f' height="{plot_h}" fill="transparent"/></svg>'
  )
  data = {
      'ticks': ticks,
      'x0': left,
      'dx': dx if len(ticks) > 1 else span,
      'series': [
          {
              'label': s.label,
              'values': [
                  '–' if v is None else f'{v:.3g}{unit}' for v in values[s.key]
              ],
          }
          for s in series
      ],
  }
  legend = ''
  if len(series) > 1:
    legend = (
        '<div class="legend">'
        + ''.join(
            f'<span style="--c:var(--s{s.slot})">{_esc(s.label)}</span>'
            for s in series
        )
        + '</div>'
    )
  return (
      f"<figure data-series='{_esc(json.dumps(data))}'>"
      f'<figcaption>{_esc(title)}</figcaption>{legend}{"".join(parts)}'
      '<div class="tip"></div></figure>'
  )


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
  head = ''.join(f'<th>{_esc(h)}</th>' for h in headers)
  body = ''.join(
      '<tr>' + ''.join(f'<td>{_esc(c)}</td>' for c in row) + '</tr>'
      for row in rows
  )
  return (
      f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
      f'<tbody>{body}</tbody></table></div>'
  )


def _messages(
    context: game_lib.GameContext, topic: str
) -> list[Mapping[str, Any]]:
  channel = 'topic/' + topic
  if channel not in context.measurements.available_channels():
    return []
  return list(context.measurements.get_channel(channel))


def build_report(
    context: game_lib.GameContext, summary: Mapping[str, Any]
) -> str:
  """Returns the full HTML report for a finished (or partial) run."""
  rows = context.planet.history
  species = context.planet.daisies.species()
  outcome = str(summary.get('outcome', 'unfinished'))
  color, icon = STATUS.get(outcome, ('#8a8984', '…'))
  llm = summary.get('llm', {})
  calls = llm.get('calls', {}).get('llm_calls_total', 0)
  final = rows[-1]
  low, high = context.planet.band()

  tiles = [
      (final['tick'], 'ticks'),
      (summary.get('sim_time', ''), 'simulated time'),
      (f'{final["temp_c"]:.1f} °C', 'final temperature'),
      (f'{final["omega"]:.1f}', 'omega left'),
      (calls, 'language model calls'),
      (f'{llm.get("total_tokens", 0):,}', 'tokens (estimated)'),
  ]
  if context.automation is not None:
    tiles.append(
        (f'{context.automation.ratio():.0%}', 'orders placed by reflexes')
    )
  gaps = [r['epistemic_gap_c'] for r in rows if 'epistemic_gap_c' in r]
  if gaps:
    tiles.append((f'{sum(gaps) / len(gaps):.1f} °C', 'mean epistemic gap'))
  slots = {name: i + 1 for i, name in enumerate(species)}
  noun = context.planet.species_noun
  names = {s: context.planet.species_name(s) for s in species}
  shorts = {s: names[s].split()[0].lower() for s in species}
  charts = [
      line_chart(
          f'Global temperature (habitable band {low:g}–{high:g} °C)',
          rows,
          [Series('temp_c', 'temperature', 1)],
          unit=' °C',
          band=(low, high),
      ),
      line_chart(
          f'{noun.capitalize()} cover (fraction of the planet)',
          rows,
          [Series(f'cover_{s}', names[s], slots[s], shorts[s]) for s in species]
          + [Series('bare', 'bare ground', len(species) + 1, 'bare')],
      ),
      line_chart(
          'Evolved growth optimum (°C)',
          rows,
          [
              Series(f'optimum_{s}', names[s], slots[s], shorts[s])
              for s in species
          ],
          unit=' °C',
      ),
      line_chart('Steward energy (omega)', rows, [Series('omega', 'omega', 1)]),
  ]
  if 'grazers' in rows[0]:
    charts.append(
        line_chart('Grazer biomass', rows, [Series('grazers', 'grazers', 1)])
    )
    charts.append(
        line_chart(
            'Grazer diet (share of white petals eaten)',
            rows,
            [Series('grazer_diet_white', 'white-petal preference', 1)],
        )
    )
  if any('believed_temp_c' in r for r in rows):
    charts.insert(
        1,
        line_chart(
            'What the Steward read vs the true temperature',
            rows,
            [
                Series('temp_c', 'true temperature', 1, 'truth'),
                Series('believed_temp_c', "Steward's reading", 2, 'reading'),
            ],
            unit=' °C',
            band=(low, high),
        ),
    )
  if context.planet.carbon is not None:
    charts.append(
        line_chart(
            'Atmospheric CO₂ (× start)',
            rows,
            [Series('co2', 'CO₂', 1)],
            unit='×',
        )
    )
  if context.planet.chemistry is not None:
    charts.append(
        line_chart(
            'Atmospheric oxygen (× present level)',
            rows,
            [Series('oxygen', 'oxygen', 1)],
            unit='×',
        )
    )
    charts.append(
        line_chart(
            'Atmospheric methane (relative)',
            rows,
            [Series('methane', 'methane', 1)],
        )
    )
  meters = context.cognition_meters()
  chapters = [m['payload'] for m in _messages(context, 'chronicle')]
  briefings = _messages(context, 'gaia.briefings')
  events = sorted(
      [
          (m['time'], m['topic'], m['payload'])
          for topic in ('geo.events', 'bio.events', 'civ.events', 'gaia.omens')
          for m in _messages(context, topic)
          if not (topic == 'gaia.omens' and ' waits.' in m['payload'])
      ],
      key=lambda e: (e[0] if e[0] is not None else -1),
  )
  topology = context.bus.topology()

  sections = [
      '<div class="dt-report">',
      f'<h1>Deep Time · {_esc(context.scenario.get("title", ""))}</h1>',
      (
          f'<p class="sub">{_esc(summary.get("models", ""))} ·'
          f' {_esc(context.scenario_name)} scenario</p>'
      ),
      (
          f'<span class="badge" style="--status:{color}"><i>{icon}</i>'
          f'{_esc(outcome)}</span> <span class="sub">'
          f'{_esc(summary.get("outcome_reason", ""))}</span>'
      ),
      '<div class="tiles">'
      + ''.join(
          f'<div class="tile"><b>{_esc(v)}</b><span>{_esc(k)}</span></div>'
          for v, k in tiles
      )
      + '</div>',
      '<h2>World</h2><div class="charts">' + ''.join(charts) + '</div>',
      '<h2>Planet</h2>',
      temperature_legend(),
      '<div class="charts">'
      + latitude_heatmap(
          rows,
          [
              context.planet.grid.band_name(b)
              for b in range(context.planet.grid.bands)
          ],
      )
      + planet_map(context)
      + '</div>',
      '<h2>Cognition meters (slow minds)</h2>',
      (
          '<p class="sub">Each slow mind is a <code>pubsub.Publisher</code>.'
          ' Latency is in ticks (sim-time mode); a busy mind skips'
          ' triggers.</p>'
      ),
      _table(
          ['owner', 'mind', 'topic', 'latency', 'computations', 'busy'],
          [
              [
                  m['owner'],
                  m['mind'],
                  m['topic'],
                  m['latency_ticks'],
                  m['computations'],
                  m['busy'],
              ]
              for m in meters
          ],
      ),
      '<h2>Chronicle</h2><ol class="chapters">'
      + ''.join(f'<li>{_esc(c)}</li>' for c in chapters)
      + '</ol>',
      '<h2>Advisor briefings</h2>',
      _table(
          ['tick', 'briefing'],
          [[_fmt(b['time']), b['payload']] for b in briefings],
      ),
      '<h2>Event log</h2>',
      _table(
          ['tick', 'topic', 'event'],
          [
              [_fmt(t) if t is not None else '', topic, text]
              for t, topic, text in events
          ],
      ),
      '<h2>Topics</h2>',
      _table(
          ['topic', 'retention', 'publishers', 'subscribers', 'messages'],
          [
              [
                  name,
                  info['retention'],
                  ', '.join(info['publishers']),
                  ', '.join(info['subscribers']),
                  info['messages'],
              ]
              for name, info in topology.items()
          ],
      ),
      '<details><summary>Topic graph (Mermaid)</summary><pre>'
      + _esc(context.bus.to_mermaid())
      + '</pre></details>',
      '<details><summary>Data table (every tick)</summary>'
      + _table(list(rows[0]), [list(r.values()) for r in rows])
      + '</details>',
      f'<script>{_SCRIPT}</script>',
      '</div>',
  ]
  return (
      '<!doctype html><html lang="en"><head><meta charset="utf-8">'
      '<meta name="viewport" content="width=device-width,initial-scale=1">'
      f'<title>Deep Time report</title><style>{_CSS}</style></head><body>'
      + ''.join(sections)
      + '</body></html>'
  )
