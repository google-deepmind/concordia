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

r"""Balance sweeps: many seeds and Steward variants in mock mode.

  python -m examples.deep_time.sweep --scenario=daisyworld --seeds=10 \
      --variants=default,passive,no_advice

Prints one row per variant (win rate, mean ticks survived, mean final Ω and
temperature, eruptions) and optionally writes every run to a CSV, to spot
dominant strategies, degenerate equilibria and runaway states.
"""

# pylint: disable=g-complex-comprehension,g-long-ternary

from collections.abc import Sequence
import contextlib
import csv
import io
import statistics
from typing import Any

from absl import app
from absl import flags
from examples.deep_time import model_setup
from examples.deep_time import scenarios
from examples.deep_time import simulation
from examples.deep_time.world import planet as planet_lib


def define_flags() -> None:
  """Defines the command-line flags (only when run as a script)."""
  flags.DEFINE_enum(
      'scenario', 'daisyworld', sorted(scenarios.SCENARIOS), 'Scenario.'
  )
  flags.DEFINE_integer('seeds', 10, 'Number of seeds per variant.')
  flags.DEFINE_list(
      'variants', ['default', 'passive', 'no_advice'], 'Mock policy variants.'
  )
  flags.DEFINE_integer('ticks', None, 'Maximum ticks per run.')
  flags.DEFINE_string('csv', None, 'Optional CSV of every run.')


def run_one(
    scenario: str, variant: str, seed: int, ticks: int | None = None
) -> dict[str, Any]:
  """Plays one silent mock run and returns its outcome row."""
  models = model_setup.make_models('mock', mock_variant=variant)
  sim, context = simulation.build_simulation(scenario, models, ticks, seed=seed)
  with contextlib.redirect_stdout(io.StringIO()):
    sim.play(max_steps=ticks)
  planet = context.planet
  final = planet.history[-1]
  temps = [row['temp_c'] for row in planet.history[1:]]
  return {
      'variant': variant,
      'seed': seed,
      'outcome': planet.outcome or 'unfinished',
      'reason': planet.outcome_reason,
      'ticks': planet.tick,
      'final_temp_c': final['temp_c'],
      'max_temp_c': max(temps) if temps else final['temp_c'],
      'final_omega': final['omega'],
      'mean_biodiversity': round(
          statistics.fmean(r['biodiversity'] for r in planet.history), 4
      ),
      'eruptions': final['eruptions'],
      'iron_made': final.get('iron_made', 0.0),
      'energy_made': final.get('energy_made', 0.0),
      'mean_gap_c': (
          round(
              statistics.fmean(
                  r.get('epistemic_gap_c', 0.0) for r in planet.history[1:]
              ),
              3,
          )
          if len(planet.history) > 1
          else 0.0
      ),
  }


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
  """Aggregates runs per variant."""
  out = []
  for variant in dict.fromkeys(r['variant'] for r in rows):
    runs = [r for r in rows if r['variant'] == variant]
    out.append({
        'variant': variant,
        'runs': len(runs),
        'win_rate': round(
            sum(r['outcome'] == planet_lib.VICTORY for r in runs) / len(runs),
            3,
        ),
        'mean_ticks': round(statistics.fmean(r['ticks'] for r in runs), 2),
        'mean_final_omega': round(
            statistics.fmean(r['final_omega'] for r in runs), 2
        ),
        'mean_max_temp_c': round(
            statistics.fmean(r['max_temp_c'] for r in runs), 2
        ),
        'mean_biodiversity': round(
            statistics.fmean(r['mean_biodiversity'] for r in runs), 3
        ),
        'mean_eruptions': round(
            statistics.fmean(r['eruptions'] for r in runs), 2
        ),
        'mean_gap_c': round(statistics.fmean(r['mean_gap_c'] for r in runs), 2),
    })
  return out


def sweep(
    scenario: str,
    seeds: int,
    variants: Sequence[str],
    ticks: int | None = None,
) -> list[dict[str, Any]]:
  return [
      run_one(scenario, variant, seed, ticks)
      for variant in variants
      for seed in range(seeds)
  ]


def main(argv: Sequence[str]) -> None:
  del argv
  values = flags.FLAGS
  rows = sweep(values.scenario, values.seeds, values.variants, values.ticks)
  if values.csv:
    with open(values.csv, 'w', newline='', encoding='utf-8') as f:
      writer = csv.DictWriter(f, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)
  for row in summarize(rows):
    print(row)


if __name__ == '__main__':
  define_flags()
  app.run(main)
