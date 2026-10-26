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

r"""An experiment: one lever pulled, over many seeds, against a control.

  python -m examples.lifeways.experiment --lever=richness=0.5 --seeds=5 \
      --years=200 --out_dir=/tmp/richer

A twin on the page compares one world with itself, one lever changed. An
experiment repeats that comparison over seeds: for each seed it plays the
control (the scenario's settings) and the treatment (`scenarios.with_lever`)
in MOCK tick mode, which is exact, so every difference is the lever's or the
seed's. It then compares the measures over the later years, seed by seed.

The runs and the comparison are `sweep`'s.
"""

from collections.abc import Sequence
from concurrent import futures
import json
import os

from absl import app
from absl import flags
from absl import logging
from examples.lifeways import scenarios
from examples.lifeways import simulation
from examples.lifeways import sweep

_SCENARIO = flags.DEFINE_string('scenario', 'sandbox', 'Scenario or preset.')
_LEVER = flags.DEFINE_string(
    'lever',
    None,
    'The lever pulled: <lever>=<value> (e.g. richness=0.5), or'
    ' without=<channel>.',
    required=True,
)
_SEEDS = flags.DEFINE_integer('seeds', 5, 'How many seeds (0, 1, ...).')
_YEARS = flags.DEFINE_integer('years', 200, 'Years each run plays.')
_WORKERS = flags.DEFINE_integer(
    'workers', 4, 'Runs played at once, each in a process of its own.'
)
_OUT_DIR = flags.DEFINE_string('out_dir', '', 'Where to write the results.')


def main(argv: Sequence[str]) -> None:
  del argv
  try:
    lever = sweep.parse_lever(_LEVER.value)
    scenarios.with_lever(scenarios.settings(_SCENARIO.value), *lever)
  except ValueError as error:
    raise app.UsageError(str(error)) from error
  conditions = [
      sweep.Condition(seed, treated)
      for seed in range(_SEEDS.value)
      for treated in (False, True)
  ]
  logging.info(
      'MOCK: %s, %s, %d seeds x %d years, %d runs',
      _SCENARIO.value,
      _LEVER.value,
      _SEEDS.value,
      _YEARS.value,
      len(conditions),
  )
  with simulation.quiet():
    if _WORKERS.value > 1:
      with futures.ProcessPoolExecutor(max_workers=_WORKERS.value) as pool:
        played = pool.map(
            sweep.play,
            *zip(
                *[(_SCENARIO.value, lever, c, _YEARS.value) for c in conditions]
            ),
        )
        rows = dict(zip(conditions, played))
    else:
      rows = {
          c: sweep.play(_SCENARIO.value, lever, c, _YEARS.value)
          for c in conditions
      }
  summary = sweep.summarize(rows)
  title = f'{_SCENARIO.value}: {_LEVER.value}, {_YEARS.value} years'
  table = sweep.report(summary, title)
  print(table)
  if _OUT_DIR.value:
    os.makedirs(_OUT_DIR.value, exist_ok=True)
    with open(os.path.join(_OUT_DIR.value, 'summary.json'), 'w') as f:
      json.dump({'title': title, **summary}, f, indent=1)
    with open(os.path.join(_OUT_DIR.value, 'report.md'), 'w') as f:
      f.write(table)
    with open(os.path.join(_OUT_DIR.value, 'census.jsonl'), 'w') as f:
      for condition, run in rows.items():
        for row in run:
          f.write(
              json.dumps({
                  'seed': condition.seed,
                  'treated': condition.treated,
                  **row,
              })
              + '\n'
          )


if __name__ == '__main__':
  app.run(main)
