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

"""A world on one page, to keep or share: its census, releases and notes.

`page` writes a self-contained HTML page (no scripts, no requests) from what
the session already keeps: the census rows, the notebook, the releases it
follows, the World's surveys and the lab's last experiment. It reads; it
never touches the world.
"""

from collections.abc import Mapping, Sequence
import html
from typing import Any

# The census lines drawn, each a small chart: its words, and how a row
# gives it.
LINES: Sequence[tuple[str, Any]] = (
    ('Alive', lambda row: row.get('population')),
    ('Fields sown', lambda row: row.get('fields')),
    ('Peoples', lambda row: (row.get('peoples') or {}).get('peoples')),
    ('Tongues', lambda row: row.get('tongues')),
    (
        'Different versions held',
        lambda row: (row.get('held') or {}).get('distinct'),
    ),
    ('Inequality (0 to 1)', lambda row: row.get('gini')),
)
# What the person writes in the notebook (`session.NOTE_KINDS`).
NOTE_KINDS = ('hypothesis', 'observation', 'note')
# The latest of the valley's own events the page lists.
EVENTS_SHOWN = 30
WIDTH, HEIGHT = 320, 64

_STYLE = """
:root { --ink: #1f2420; --paper: #f6f3ea; --muted: #4f534b; --line: #d8d2c2;
  --accent: #2d6a4f; --lean: #e7e0cc; }
@media (prefers-color-scheme: dark) { :root { --ink: #e8e6df;
  --paper: #1b1d1a; --muted: #b4b7ae; --line: #3a3d37; --accent: #74c69d;
  --lean: #33362f; } }
body { margin: 0 auto; max-width: 720px; padding: 16px; background:
  var(--paper); color: var(--ink); font: 16px/1.5 system-ui, sans-serif; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 24px 0 8px; }
.meta { color: var(--muted); font-size: 0.95rem; }
.charts { display: grid; grid-template-columns: repeat(auto-fill,
  minmax(260px, 1fr)); gap: 12px; }
figure { margin: 0; }
figcaption { font-size: 0.95rem; }
svg { width: 100%; height: auto; display: block; border: 1px solid
  var(--line); border-radius: 6px; }
ul { padding-left: 20px; }
li { margin: 4px 0; }
"""


def _e(text: Any) -> str:
  return html.escape(str(text))


def _number(value: float) -> str:
  return f'{value:.2f}'.rstrip('0').rstrip('.')


def chart(rows: Sequence[Mapping[str, Any]], value: Any) -> str:
  """A census line as a small inline SVG, lean years shaded behind it.

  Args:
    rows: The census rows, oldest first.
    value: How a row gives the line's value (None: not known that year).

  Returns:
    The SVG ('' if no row has a value).
  """
  points = [
      (int(row['year']), float(v))
      for row in rows
      if isinstance(v := value(row), (int, float))
  ]
  if not points:
    return ''
  first, last = rows[0]['year'], rows[-1]['year']
  most = max(max(v for _, v in points), 1e-9)
  span = max(1, last - first)

  def x(year: int) -> float:
    return 4 + (year - first) / span * (WIDTH - 8)

  def y(v: float) -> float:
    return HEIGHT - 4 - v / most * (HEIGHT - 8)

  step = (WIDTH - 8) / span
  bands = ''.join(
      f'<rect x="{x(row["year"]) - step / 2:.1f}" y="0"'
      f' width="{max(1.0, step):.1f}" height="{HEIGHT}"'
      ' fill="var(--lean)"/>'
      for row in rows
      if row.get('lean')
  )
  line = ' '.join(f'{x(t):.1f},{y(v):.1f}' for t, v in points)
  return (
      f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img">{bands}'
      f'<polyline points="{line}" fill="none" stroke="var(--accent)"'
      ' stroke-width="2"/></svg>'
  )


def _fate(release: Mapping[str, Any]) -> str:
  """What became of a release, in words."""
  holders = release.get('holders') or []
  now = holders[-1][1] if holders else 0
  said = (
      f'Year {release["year"]}: a {release["kind"]} about'
      f' {release["domain"]}, "{release["text"]}". Heard by'
      f' {release.get("heard", 0)}; at most {release.get("peak", 0)} held it'
  )
  if release.get('died') is not None:
    return f'{said}; it died out in year {release["died"]}.'
  return f'{said}; {now} hold it now.'


def page(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    notebook: Sequence[Mapping[str, Any]],
    releases: Sequence[Mapping[str, Any]],
    surveys: Sequence[Mapping[str, Any]],
    lab: Mapping[str, Any] | None,
    written: str,
    top_text: str = '',
) -> str:
  """The world on one page.

  Args:
    title: The world, as the page names it (scenario, seed, year).
    rows: The census rows kept, oldest first.
    notebook: The notebook's entries, oldest first.
    releases: The releases followed (`Session.releases`).
    surveys: The World's surveys.
    lab: The lab's last experiment (`Session.lab_status`), if any.
    written: When the page was written, in words.
    top_text: The words of the custom most widely held now.

  Returns:
    The HTML.
  """
  out = [
      '<!doctype html><html lang="en"><head><meta charset="utf-8">',
      '<meta name="viewport" content="width=device-width, initial-scale=1">',
      f'<title>{_e(title)}</title><style>{_STYLE}</style></head><body>',
      f'<h1>{_e(title)}</h1>',
      (
          f'<p class="meta">A Lifeways world, written {_e(written)}. Lifeways'
          ' is an example of Concordia: each person is an entity with its'
          ' customs as components; a World game master holds the valley.</p>'
      ),
  ]
  if rows:
    last = rows[-1]
    top = ((last.get('held') or {}).get('top') or [[None, 0]])[0]
    out.append('<h2>Now</h2><ul>')
    out.append(
        f'<li>{_e(last.get("population", 0))} alive in year {_e(last["year"])};'
        f' {_e((last.get("peoples") or {}).get("peoples", 0))} peoples,'
        f' {_e(last.get("tongues", 0))} tongues, {_e(last.get("fields", 0))}'
        ' fields.</li>'
    )
    if top[0]:
      out.append(
          f'<li>The most widely held custom, "{_e(top_text or top[0])}", is'
          f' held by {round(100 * top[1])}% of the living.</li>'
      )
    out.append('</ul><h2>Over the years</h2>')
    out.append(
        f'<p class="meta">Years {_e(rows[0]["year"])}–{_e(last["year"])};'
        ' each chart its own scale, from 0 to the most it reached. Shaded:'
        ' lean years.</p><div class="charts">'
    )
    for name, value in LINES:
      drawn = chart(rows, value)
      if drawn:
        now = value(last)
        shown = _number(now) if isinstance(now, (int, float)) else '—'
        out.append(
            f'<figure>{drawn}<figcaption>{_e(name)}: {_e(shown)}'
            ' now</figcaption></figure>'
        )
    out.append('</div>')
  if releases:
    out.append('<h2>Your releases</h2><ul>')
    out.extend(f'<li>{_e(_fate(r))}</li>' for r in releases)
    out.append('</ul>')
  mine = [e for e in notebook if e['kind'] in NOTE_KINDS]
  if mine:
    out.append('<h2>Your notes</h2><ul>')
    for entry in mine:
      check = entry.get('check') or {}
      result = check.get('result')
      if result:
        outcome = (
            f' (Checked in year {_e(check["due"])}:'
            f' {"held" if result["held"] else "did not hold"}.)'
        )
      elif check:
        outcome = f' (The census checks it in year {_e(check["due"])}.)'
      else:
        outcome = ''
      out.append(
          f'<li>Year {_e(entry["year"])}, {_e(entry["kind"])}:'
          f' {_e(entry["text"])}{outcome}</li>'
      )
    out.append('</ul>')
  events = [e for e in notebook if e['kind'] == 'event'][-EVENTS_SHOWN:]
  if events:
    out.append('<h2>What the valley did</h2><ul>')
    out.extend(
        f'<li>Year {_e(e["year"])}: {_e(e["text"])}</li>' for e in events
    )
    out.append('</ul>')
  tallied = [s for s in surveys if s.get('tally') is not None]
  if tallied:
    out.append('<h2>Surveys</h2><ul>')
    for survey in tallied:
      named = '; '.join(f'"{text}" ({n})' for _, text, n in survey['tally'][:3])
      out.append(
          f'<li>Year {_e(survey["year"])}, "{_e(survey["question"])}":'
          f' {_e(len(survey["answers"]))} of {_e(len(survey["asked"]))}'
          f' answered. Most named: {_e(named or "none")}.</li>'
      )
    out.append('</ul>')
  if lab and lab.get('state') == 'done' and lab.get('result'):
    out.append('<h2>The last experiment</h2>')
    lever = lab.get('lever')
    what = (
        f'{lever[0]} set to {lever[1]}, seed by seed'
        if lab['kind'] == 'compare' and lever
        else "this world's settings on other seeds"
    )
    seeds = lab['seeds']
    out.append(
        f'<p class="meta">{_e(what)}; seeds {_e(seeds[0])}–{_e(seeds[1])},'
        f' {_e(lab["years"])} years each (MOCK).</p><ul>'
    )
    for name, m in lab['result']['measures'].items():
      if lab['kind'] == 'compare':
        said = (
            f'{m["control"][0]} as set, {m["treatment"][0]} with the lever'
            f' (more in {m["above"]} seeds, less in {m["below"]})'
        )
      else:
        said = f'{m["mean"]} ± {m["sd"]} (from {m["least"]} to {m["most"]})'
      out.append(f'<li>{_e(name)}: {_e(said)}.</li>')
    out.append('</ul>')
  out.append('</body></html>')
  return '\n'.join(out) + '\n'
