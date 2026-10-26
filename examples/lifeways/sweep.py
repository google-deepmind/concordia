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
`summarize` compares the measures over their later years, seed by seed. The
harness only builds, plays and reads simulations: each run is the stock
simulation of `simulation.build_simulation`, read through its census.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import statistics
from typing import Any

from examples.lifeways import scenarios
from examples.lifeways import simulation

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
    lever: tuple[str, Any],
    condition: Condition,
    years: int,
) -> list[dict[str, Any]]:
  """Plays one run and returns its census rows.

  Args:
    scenario: The scenario or preset.
    lever: The lever and its value.
    condition: The seed, and whether the lever is pulled.
    years: How many years.

  Returns:
    A census row a year (`Run.census_row`).
  """
  settings = scenarios.settings(scenario, seed=condition.seed, max_years=years)
  if condition.treated:
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

  def late_mean(run: Sequence[Mapping[str, Any]], measure: Any) -> float:
    late = run[int(len(run) * burn_in) :] or run
    return statistics.fmean(float(measure(row)) for row in late)

  def spread(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0

  out: dict[str, Any] = {'seeds': seeds, 'measures': {}}
  for name, measure in MEASURES.items():
    control = [late_mean(rows[Condition(s, False)], measure) for s in seeds]
    treated = [late_mean(rows[Condition(s, True)], measure) for s in seeds]
    diffs = [t - c for t, c in zip(treated, control)]
    out['measures'][name] = {
        'control': [
            round(statistics.fmean(control), 3),
            round(spread(control), 3),
        ],
        'treatment': [
            round(statistics.fmean(treated), 3),
            round(spread(treated), 3),
        ],
        'difference': [
            round(statistics.fmean(diffs), 3),
            round(spread(diffs), 3),
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
