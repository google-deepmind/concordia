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

r"""Runs a Deep Time scenario and writes a run folder.

Offline (deterministic mock model):

  python -m examples.deep_time.run --scenario=daisyworld --mode=mock

Live on Together AI (key from TOGETHER_API_KEY or TOGETHER_AI_API_KEY):

  python -m examples.deep_time.run --scenario=daisyworld --mode=together \
      --ticks=10 --max_calls=100

The run folder holds `report.html` (charts, cognition meters, chronicle,
briefings, events, topics), `summary.json`, `timeseries.csv`, `topics.mmd`
(the publisher -> topic -> subscriber graph), Concordia's `log.html` and,
when the scenario has media, `gazette.html` (the contrib forum's own page).
"""

from collections.abc import Sequence
import csv
import datetime
import json
import os
import time
from typing import Any

from absl import app
from absl import flags
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import model_setup
from examples.deep_time import scenarios
from examples.deep_time import simulation
from examples.deep_time.report import html_report


def define_flags() -> None:
  """Defines the command-line flags.

  Called only when this module runs as a script, so that importing it (for
  example from tests that also import other examples' runners) never
  registers flags twice in absl's global registry.
  """
  flags.DEFINE_enum(
      'scenario', 'daisyworld', sorted(scenarios.SCENARIOS), 'Scenario to run.'
  )
  flags.DEFINE_enum(
      'mode', 'mock', ['mock', 'together', 'ollama'], 'Language model mode.'
  )
  flags.DEFINE_integer(
      'ticks', None, 'Maximum ticks (defaults to the scenario length).'
  )
  flags.DEFINE_string(
      'out_dir', None, 'Run folder (defaults to a timestamped temp folder).'
  )
  flags.DEFINE_string(
      'fast_model', model_setup.DEFAULT_FAST_MODEL, 'Model for per-tick acts.'
  )
  flags.DEFINE_string(
      'slow_model', model_setup.DEFAULT_SLOW_MODEL, 'Model for slow minds.'
  )
  flags.DEFINE_enum(
      'time_mode',
      'sim',
      ['sim', 'async'],
      'sim: slow minds land a fixed number of ticks after their trigger'
      ' (reproducible); async: they run on threads and land whenever the model'
      ' answers.',
  )
  flags.DEFINE_integer('seed', 0, 'Seed of the planet (eruptions).')
  flags.DEFINE_enum(
      'mock_variant',
      'default',
      ['default', 'passive', 'random', 'no_advice', 'forecaster'],
      'Mock policies: default (the skilled bot); passive (the main roles'
      ' never act); random (the main roles act at random); no_advice (the'
      ' Steward ignores its Advisor); forecaster (the Steward acts on its'
      ' forecast, not its late reading).',
  )
  flags.DEFINE_integer(
      'max_calls', 300, 'Call cap per live model (budget governor).'
  )


def write_timeseries(path: str, rows: Sequence[dict[str, Any]]) -> None:
  fields: list[str] = []
  for row in rows:
    fields.extend(k for k in row if k not in fields)
  with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def llm_economy(models: model_setup.Models) -> dict[str, Any]:
  """Condenses `ProfiledLanguageModel` statistics into calls and tokens."""
  stats = models.profiler.get_stats()
  values = stats.get('values', {})
  latencies = values.get('llm_latency_seconds', [])
  return {
      'calls': dict(stats.get('counters', {})),
      'prompt_tokens': int(sum(values.get('llm_prompt_tokens', []))),
      'total_tokens': int(sum(values.get('llm_total_tokens', []))),
      'mean_latency_s': (
          round(sum(latencies) / len(latencies), 4) if latencies else 0.0
      ),
  }


def summarize(
    context: game_lib.GameContext,
    models: model_setup.Models,
    wall_seconds: float,
) -> dict[str, Any]:
  """Returns the JSON-serializable summary of a finished run."""
  planet = context.planet
  final = planet.history[-1]
  summary: dict[str, Any] = {
      'scenario': context.scenario_name,
      'seed': context.seed,
      'models': models.label,
      'ticks': planet.tick,
      'sim_time': context.clock.formatted_now(),
      'outcome': planet.outcome or 'unfinished',
      'outcome_reason': planet.outcome_reason,
      'final': final,
      'topics': context.bus.stats(),
      'wall_seconds': round(wall_seconds, 3),
      'seconds_per_tick': round(wall_seconds / max(1, planet.tick), 4),
      'llm': llm_economy(models),
      'minds': context.cognition_meters(),
      'chapters': context.bus.message_count('chronicle'),
      'acts': {name: a.acts for name, a in context.actors.items()},
  }
  decisions = {
      name: actor.decisions
      for name, actor in context.actors.items()
      if isinstance(actor, actors_lib.PolityActor)
  }
  if decisions:
    summary['polity_decisions'] = decisions
  gaps = [
      r['epistemic_gap_c'] for r in planet.history if 'epistemic_gap_c' in r
  ]
  if gaps:
    summary['epistemic_gap_c'] = {
        'mean': round(sum(gaps) / len(gaps), 3),
        'max': max(gaps),
    }
  summary['instruments'] = planet.instrument_text()
  if context.wakeups is not None:
    summary['wakeups'] = context.wakeups.woken_by
  if context.automation is not None:
    summary['automation'] = {
        'ratio': round(context.automation.ratio(), 3),
        'reflex_orders': context.automation.reflex_orders,
        'turn_orders': context.automation.turn_orders,
        'reflexes': context.automation.engine.names(),
    }
  if context.planet.trade is not None:
    summary['bullwhip'] = {
        'ratio': context.planet.trade.bullwhip(),
        'orders_while_in_transit': context.planet.trade.blind_reorders,
        'orders_on_stale_prices': context.planet.trade.stale_orders,
    }
  treaties = context.planet.treaties
  if treaties is not None and context.planet.civ is not None:
    names = {k: p.name for k, p in context.planet.civ.polities.items()}
    summary['treaties'] = {
        'signed': treaties.signed,
        **treaties.payload(names, context.planet.tick),
    }
  gazette = context.planet.gazette
  if gazette is not None and context.planet.civ is not None:
    stances = [b.stance for b in gazette.printed]
    summary['gazette'] = {
        'broadsheets': {s: stances.count(s) for s in sorted(set(stances))},
        'karma': gazette.forum.get_karma(),
        'consent': {
            p.name: round(p.consent, 3)
            for p in context.planet.civ.polities.values()
        },
    }
  if models.mock is not None:
    summary['mock_calls'] = models.mock.stats()
  return summary


def run(
    scenario: str = 'daisyworld',
    mode: str = 'mock',
    *,
    ticks: int | None = None,
    out_dir: str | None = None,
    fast_model: str = model_setup.DEFAULT_FAST_MODEL,
    slow_model: str = model_setup.DEFAULT_SLOW_MODEL,
    max_calls: int = 300,
    time_mode: str = 'sim',
    seed: int = 0,
    mock_variant: str = 'default',
) -> dict[str, Any]:
  """Runs a scenario and writes its run folder; returns the summary."""
  if out_dir is None:
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join('/tmp', 'deep_time_runs', f'{scenario}-{stamp}')
  os.makedirs(out_dir, exist_ok=True)
  models = model_setup.make_models(
      mode,
      fast_model=fast_model,
      slow_model=slow_model,
      max_calls=max_calls,
      mock_variant=mock_variant,
  )
  sim, context = simulation.build_simulation(
      scenario,
      models,
      ticks,
      time_mode=time_mode,
      seed=seed,
      # Live characters play without the playbook lines (a scenario flag).
      overrides=None if models.mock is not None else {'coached': False},
  )
  start = time.perf_counter()
  log = sim.play(max_steps=ticks)
  wall = time.perf_counter() - start
  summary = summarize(context, models, wall)

  with open(os.path.join(out_dir, 'log.html'), 'w', encoding='utf-8') as f:
    f.write(log.to_html())
  write_timeseries(
      os.path.join(out_dir, 'timeseries.csv'), context.planet.history
  )
  with open(os.path.join(out_dir, 'topics.mmd'), 'w', encoding='utf-8') as f:
    f.write(context.bus.to_mermaid())
  with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, default=str)
  with open(os.path.join(out_dir, 'report.html'), 'w', encoding='utf-8') as f:
    f.write(html_report.build_report(context, summary))
  if context.planet.gazette is not None:
    # The contrib forum renders its own page: the Gazette as printed.
    with open(
        os.path.join(out_dir, 'gazette.html'), 'w', encoding='utf-8'
    ) as f:
      f.write(context.planet.gazette.forum.to_html())
  context.bus.close()
  summary['out_dir'] = out_dir
  return summary


def main(argv: Sequence[str]) -> None:
  del argv
  values = flags.FLAGS
  summary = run(
      values.scenario,
      values.mode,
      ticks=values.ticks,
      out_dir=values.out_dir,
      fast_model=values.fast_model,
      slow_model=values.slow_model,
      max_calls=values.max_calls,
      time_mode=values.time_mode,
      seed=values.seed,
      mock_variant=values.mock_variant,
  )
  print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
  define_flags()
  app.run(main)
