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

r"""Runs a Lifeways world headless and writes what happened (MOCK by default).

  python -m examples.lifeways.run --years=200 --seed=0 --out_dir=/tmp/lifeways

Writes `census.jsonl` (a row a year: the world's census and what the living
hold), `dead.jsonl` (each person who died: the final state of the person's
entity and its params), `customs.jsonl` (each variant of a custom once nobody
living holds it: the long record of culture) and `summary.json`; with
`--soak`, also each fiftieth year's memory use and seconds per year.
"""

import contextlib
import io
import json
import os
import resource
import subprocess
import sys
import time

from absl import app
from absl import flags
from examples.lifeways import scenarios
from examples.lifeways import simulation

_SCENARIO = flags.DEFINE_string('scenario', 'sandbox', 'Scenario or preset.')
_YEARS = flags.DEFINE_integer(
    'years', 100, 'Years to run (the Sandbox has no end).'
)
_SEED = flags.DEFINE_integer('seed', 0, 'World seed.')
_OUT_DIR = flags.DEFINE_string('out_dir', '', 'Where to write the run.')
_SOAK = flags.DEFINE_bool('soak', False, 'Report memory and speed as it runs.')


def _max_rss_mb() -> float:
  usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return usage / 1e6 if os.uname().sysname == 'Darwin' else usage / 1e3


def _rss_mb() -> float:
  """This process's resident memory now, in MB."""
  out = subprocess.run(
      ['ps', '-o', 'rss=', '-p', str(os.getpid())],
      capture_output=True,
      text=True,
      check=True,
  )
  return int(out.stdout.strip()) / 1e3


def main(argv):
  del argv
  out = _OUT_DIR.value or f'/tmp/lifeways/{_SCENARIO.value}-{_SEED.value}'
  os.makedirs(out, exist_ok=True)
  print(f'MOCK: {_SCENARIO.value}, seed {_SEED.value}, {_YEARS.value} years')
  dead = open(os.path.join(out, 'dead.jsonl'), 'w')
  census = open(os.path.join(out, 'census.jsonl'), 'w')
  customs = open(os.path.join(out, 'customs.jsonl'), 'w')

  def bury(record) -> None:
    dead.write(json.dumps(record) + '\n')

  def forget(records) -> None:
    for record in records:
      customs.write(json.dumps(record) + '\n')

  run = simulation.build_simulation(
      scenarios.settings(
          _SCENARIO.value, seed=_SEED.value, max_years=_YEARS.value
      ),
      archive=simulation.Archive(keep=0, sink=bury),
  )
  start = last = time.monotonic()
  soak = []

  def year_closed(year: int) -> None:
    nonlocal last
    world = run.world()
    row = {**world.history[-1], 'held': run.customs_census()}
    census.write(json.dumps(row) + '\n')
    if _SOAK.value and year % 50 == 0:
      now = time.monotonic()
      row = {
          'year': year,
          'living': len(world.people),
          'entities': len(run.sim.entities),
          'variants': len(world.culture),
          'rss_mb': round(_rss_mb(), 1),
          'max_rss_mb': round(_max_rss_mb(), 1),
          'seconds_per_year': round((now - last) / 50, 3),
          'raw_log': len(run.sim.get_raw_log()),
      }
      last = now
      soak.append(row)
      print(json.dumps(row), file=sys.stderr, flush=True)

  run.lifecycle.year_closed = year_closed
  run.lifecycle.forgotten = forget
  with contextlib.redirect_stdout(io.StringIO()):
    run.play()
  dead.close()
  census.close()
  customs.close()
  world = run.world()
  summary = {
      'scenario': _SCENARIO.value,
      'seed': _SEED.value,
      'years': world.year,
      'living': len(world.people),
      'dead': len(world.dead),
      'variants': len(world.culture),
      'variants_ever': world.culture.created(),
      'seconds': round(time.monotonic() - start, 1),
      'soak': soak,
  }
  with open(os.path.join(out, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
  print(json.dumps({k: v for k, v in summary.items() if k != 'soak'}))


if __name__ == '__main__':
  app.run(main)
