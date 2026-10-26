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

r"""Draws a run's census (`census.jsonl` from `run.py`) as graphs.

  python -m examples.lifeways.plot --census=/tmp/lifeways/sandbox-0/census.jsonl

Writes `census.png` beside the census: people, births, deaths and
arrivals; the skill people work by; the customs held, by the channel they
came by; the norms held, by form; and what people did with their year.
Yearly counts of events are drawn as means over `--window` years.
"""

from collections.abc import Mapping, Sequence
import json
import os
from typing import Any

from absl import app
from absl import flags
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot  # pylint: disable=g-import-not-at-top

_CENSUS = flags.DEFINE_string('census', '', "A run's census.jsonl.")
_OUT = flags.DEFINE_string('out', '', 'The image (default: beside it).')
_WINDOW = flags.DEFINE_integer('window', 10, 'Years events are averaged over.')


def _series(
    rows: Sequence[Mapping[str, Any]], *path: str
) -> dict[str, list[float]]:
  """Each key of the dict at `path`, as a series over the rows."""
  keys = sorted(
      {key for row in rows for key in _at(row, path) if isinstance(key, str)}
  )
  return {key: [_at(row, path).get(key, 0) for row in rows] for key in keys}


def _at(row: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any]:
  for key in path:
    row = row.get(key, {})
  return row


def _mean(values: Sequence[float], window: int) -> list[float]:
  """Each value's mean with the `window - 1` before it."""
  return [
      sum(values[max(0, i - window + 1) : i + 1])
      / (i + 1 - max(0, i - window + 1))
      for i in range(len(values))
  ]


def plot(rows: Sequence[Mapping[str, Any]], out: str, window: int = 10) -> None:
  """Draws the census rows to the image `out`."""
  years = [row['year'] for row in rows]
  figure, axes = pyplot.subplots(5, 1, figsize=(10, 16), sharex=True)

  people = axes[0]
  people.plot(years, [r['population'] for r in rows], label='living')
  for key in ('births', 'deaths', 'arrivals'):
    events = _mean([r.get(key, 0) for r in rows], window)
    people.plot(
        years,
        [e * window for e in events],
        label=f'{key} per {window} years',
        lw=0.8,
    )
  people.set_title('People')

  axes[1].plot(
      years, [r.get('skill_worked', 0) for r in rows], lw=0.5, alpha=0.4
  )
  axes[1].plot(years, _mean([r.get('skill_worked', 0) for r in rows], window))
  axes[1].set_title('Skill people work by (mean level of those gathering)')

  for title, axis, path in (
      (
          'Customs held, by the channel they came by',
          axes[2],
          ('held', 'by_channel'),
      ),
      ('Norms held, by form', axes[3], ('held', 'norms_by_form')),
      ('What people did with their year', axes[4], ('moves',)),
  ):
    for key, values in _series(rows, *path).items():
      if path == ('moves',):
        values = _mean(values, window)
      axis.plot(years, values, label=key or '(not understood)', lw=0.8)
    axis.set_title(title)

  for axis in axes:
    if axis.get_legend_handles_labels()[0]:
      axis.legend(loc='upper left', fontsize='small', ncol=4)
    axis.grid(alpha=0.3)
  axes[-1].set_xlabel('year')
  figure.tight_layout()
  figure.savefig(out, dpi=90)
  pyplot.close(figure)


def main(argv: Sequence[str]) -> None:
  del argv
  with open(_CENSUS.value) as f:
    rows = [json.loads(line) for line in f if line.strip()]
  out = _OUT.value or os.path.join(os.path.dirname(_CENSUS.value), 'census.png')
  plot(rows, out, _WINDOW.value)
  print(out)


if __name__ == '__main__':
  app.run(main)
