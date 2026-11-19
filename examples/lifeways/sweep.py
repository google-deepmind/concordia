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

"""A sweep: one lever over seeds, against a control (the experiment's runs).

For each seed, `play` runs the control (the scenario's settings) and the
treatment (`scenarios.with_lever`) in MOCK tick mode, which is exact, and
`summarize` compares the measures over their later years, seed by seed;
`spread` reads one setting played on several seeds (the page's lab). Minds
compared (`minds.py`): `play_mind` plays a scenario with one mind, and
`minds_report` sets minds side by side. The harness only builds, plays and
reads simulations: each run is the stock simulation of
`simulation.build_simulation`, read through its census.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import statistics
import time
from typing import Any

from examples.lifeways import scenarios
from examples.lifeways import simulation
from examples.lifeways.world import world as world_lib

# The measures compared, each read from a census row.
MEASURES: Mapping[str, Any] = {
    'living': lambda row: row['population'],
    'distinct customs held': lambda row: row['held']['distinct'],
    'diversity of customs': lambda row: row['held']['diversity'],
    'share holding the top custom': lambda row: (
        row['held']['top'][0][1] if row['held']['top'] else 0.0
    ),
    'peoples': lambda row: row['peoples']['peoples'],
    'tongues': lambda row: row.get('tongues', 0),
    'skill worked': lambda row: row.get('skill_worked', 0.0),
    'fields': lambda row: row.get('fields', 0),
    'inequality': lambda row: row.get('gini', 0.0),
    'food shares given': lambda row: row.get('shares', 0),
}


@dataclasses.dataclass(frozen=True)
class Condition:
  """One run: a seed, and whether the lever is pulled."""

  seed: int
  treated: bool


def parse_lever(text: str) -> tuple[str, Any]:
  """A lever and its value from `<lever>=<value>`.

  Args:
    text: E.g. `richness=0.5` or `without=story`.

  Returns:
    The lever, and its value (a number, or a channel's name).

  Raises:
    ValueError: If the text is not of that form.
  """
  lever, sep, value = text.partition('=')
  if not sep or not lever or not value:
    raise ValueError(f'A lever is <lever>=<value>, not {text!r}.')
  if lever == scenarios.WITHOUT:
    return lever, value
  try:
    return lever, float(value)
  except ValueError as error:
    raise ValueError(f'{lever} is a number, not {value!r}.') from error


def play(
    scenario: str,
    lever: tuple[str, Any] | None,
    condition: Condition,
    years: int,
    overrides: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
  """Plays one run and returns its census rows.

  Args:
    scenario: The scenario or preset.
    lever: The lever and its value (None: no treatment).
    condition: The seed, and whether the lever is pulled.
    years: How many years.
    overrides: Optional. Settings that replace the scenario's (a world's
      own, to play it again).

  Returns:
    A census row a year (`Run.census_row`).
  """
  settings = scenarios.settings(
      scenario,
      **{**(overrides or {}), 'seed': condition.seed, 'max_years': years},
  )
  if condition.treated and lever is not None:
    settings = scenarios.with_lever(settings, *lever)
  run = simulation.build_simulation(
      settings, archive=simulation.Archive(keep=0)
  )
  rows: list[dict[str, Any]] = []
  run.lifecycle.year_closed = lambda year: rows.append(run.census_row(year))
  run.play(log_retention=0)
  return rows


def summarize(
    rows: Mapping[Condition, Sequence[Mapping[str, Any]]],
    burn_in: float = 0.5,
) -> dict[str, Any]:
  """Compares the treatment with the control, measure by measure.

  Each run's measure is its mean over the later years (after the first
  `burn_in` share). For each measure: the control's and the treatment's
  mean and standard deviation over seeds, and the paired difference
  (treatment minus control, seed by seed): its mean, standard deviation,
  and in how many seeds the treatment was above, below or equal.

  Args:
    rows: Each run's census rows, by condition.
    burn_in: The share of the years left out at the start.

  Returns:
    The comparison, by measure, and the seeds.
  """
  seeds = sorted({c.seed for c in rows})

  def sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0

  out: dict[str, Any] = {'seeds': seeds, 'measures': {}}
  for name, measure in MEASURES.items():
    control = [
        late_mean(rows[Condition(s, False)], measure, burn_in) for s in seeds
    ]
    treated = [
        late_mean(rows[Condition(s, True)], measure, burn_in) for s in seeds
    ]
    diffs = [t - c for t, c in zip(treated, control)]
    out['measures'][name] = {
        'control': [
            round(statistics.fmean(control), 3),
            round(sd(control), 3),
        ],
        'treatment': [
            round(statistics.fmean(treated), 3),
            round(sd(treated), 3),
        ],
        'difference': [
            round(statistics.fmean(diffs), 3),
            round(sd(diffs), 3),
        ],
        'above': sum(d > 0 for d in diffs),
        'below': sum(d < 0 for d in diffs),
        'equal': sum(d == 0 for d in diffs),
        'by_seed': {
            str(s): [round(c, 3), round(t, 3)]
            for s, c, t in zip(seeds, control, treated)
        },
    }
  return out


def late_mean(
    run: Sequence[Mapping[str, Any]], measure: Any, burn_in: float = 0.5
) -> float:
  """A run's measure: its mean over the later years (after `burn_in`)."""
  late = run[int(len(run) * burn_in) :] or run
  return statistics.fmean(float(measure(row)) for row in late)


def spread(
    runs: Mapping[int, Sequence[Mapping[str, Any]]], burn_in: float = 0.5
) -> dict[str, Any]:
  """How a measure varies over seeds: one setting, played again.

  Args:
    runs: Each run's census rows, by seed.
    burn_in: The share of the years left out at the start.

  Returns:
    The seeds, and by measure: the mean, standard deviation, least and most
    of the runs' measures, and each seed's.
  """
  seeds = sorted(runs)
  out: dict[str, Any] = {'seeds': seeds, 'measures': {}}
  for name, measure in MEASURES.items():
    values = [late_mean(runs[s], measure, burn_in) for s in seeds]
    out['measures'][name] = {
        'mean': round(statistics.fmean(values), 3),
        'sd': round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        'least': round(min(values), 3),
        'most': round(max(values), 3),
        'by_seed': {str(s): round(v, 3) for s, v in zip(seeds, values)},
    }
  return out


def report(summary: Mapping[str, Any], title: str) -> str:
  """The comparison as a Markdown table."""
  lines = [
      f'# {title}',
      '',
      (
          f'Seeds {", ".join(map(str, summary["seeds"]))}; each measure is a'
          " run's mean over its later half, then mean ± sd over seeds."
      ),
      '',
      '| Measure | Control | Treatment | Difference | Seeds above/below |',
      '|---|---|---|---|---|',
  ]
  for name, m in summary['measures'].items():
    lines.append(
        f'| {name} | {m["control"][0]} ± {m["control"][1]} |'
        f' {m["treatment"][0]} ± {m["treatment"][1]} |'
        f' {m["difference"][0]:+} ± {m["difference"][1]} |'
        f' {m["above"]}/{m["below"]} |'
    )
  return '\n'.join(lines) + '\n'


# Minds compared: the table's columns, a key of `mind_measures` and its
# heading.
MIND_COLUMNS = (
    ('answers', 'answers'),
    ('unread', 'not read'),
    ('cited', 'named a custom'),
    ('sow', '"sow a field"'),
    ('fields', 'fields grown'),
    ('barren', 'grew nothing'),
    ('deaths', 'deaths'),
    ('living', 'alive at the end'),
    ('own_unseen', 'own norms kept unseen'),
    ('known_unseen', 'expected norms kept unseen'),
    ('seconds_a_year', 'seconds a year'),
    ('calls', 'model calls'),
)


def mind_measures(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  """A run's measures, from its census rows.

  Args:
    rows: A census row a year (`Run.census_row`).

  Returns:
    Answers, those not read and those that named a custom; "sow a field"
    moves, fields grown and sowings that grew nothing; deaths; the living
    at the end; own and expected norms kept unseen (shares, None if none
    were held).
  """
  moves = [row.get('moves', {}) for row in rows]

  def kept(key: str) -> float | None:
    k = sum(row.get('norms_kept', {}).get(key, [0, 0])[0] for row in rows)
    n = sum(row.get('norms_kept', {}).get(key, [0, 0])[1] for row in rows)
    return round(k / n, 3) if n else None

  return {
      'answers': sum(sum(m.values()) for m in moves),
      'unread': sum(m.get('', 0) for m in moves),
      'cited': sum(row.get('cited', 0) for row in rows),
      'sow': sum(m.get(world_lib.SOW, 0) for m in moves),
      'fields': sum(row.get('sown', 0) for row in rows),
      'barren': sum(row.get('barren', 0) for row in rows),
      'deaths': sum(row.get('deaths', 0) for row in rows),
      'living': rows[-1]['population'] if rows else 0,
      'own_unseen': kept('own_unseen'),
      'known_unseen': kept('known_unseen'),
  }


def minds_report(
    results: Mapping[str, Sequence[Mapping[str, Any]]], title: str
) -> str:
  """The minds side by side, as a Markdown table.

  Args:
    results: Each mind's runs' measures (`mind_measures`, with `seconds_a_year`
      and `calls`).
    title: The table's title.

  Returns:
    The table: each measure's mean over a mind's runs (shares as %, a run
    with none held left out).
  """
  lines = [
      f'# {title}',
      '',
      '| minds | runs | ' + ' | '.join(h for _, h in MIND_COLUMNS) + ' |',
      '|---|---|' + '---|' * len(MIND_COLUMNS),
  ]
  for mind, runs in results.items():
    cells = []
    for key, _ in MIND_COLUMNS:
      values = [r[key] for r in runs if r.get(key) is not None]
      if not values:
        cells.append('–')
      elif key in ('own_unseen', 'known_unseen'):
        cells.append(f'{round(100 * statistics.fmean(values))}%')
      else:
        mean = statistics.fmean(values)
        cells.append(f'{mean:.1f}'.removesuffix('.0'))
    lines.append(f'| {mind} | {len(runs)} | ' + ' | '.join(cells) + ' |')
  return '\n'.join(lines) + '\n'


def play_mind(
    mind: str,
    scenario: str,
    seed: int,
    years: int,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """One run by one mind: its measures, seconds a year and model calls."""
  model, counts = simulation.make_model(mind)
  settings = scenarios.settings(
      scenario, **{**(overrides or {}), 'seed': seed, 'max_years': years}
  )
  run = simulation.build_simulation(
      settings, model=model, archive=simulation.Archive(keep=0)
  )
  rows: list[dict[str, Any]] = []
  run.lifecycle.year_closed = lambda year: rows.append(run.census_row(year))
  start = time.monotonic()
  run.play(log_retention=0)
  seconds = time.monotonic() - start
  out = mind_measures(rows)
  out['seconds_a_year'] = round(seconds / max(1, len(rows)), 2)
  out['calls'] = (
      len(counts.get_stats()['values'].get('llm_prompt_tokens', ()))
      if counts is not None
      else 0
  )
  out['rows'] = rows
  return out
