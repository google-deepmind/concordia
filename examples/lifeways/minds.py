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

r"""Minds compared: one scenario, played by several minds, in one table.

  python -m examples.lifeways.minds --scenario=first_farmers --years=24 \
      --minds=mock,ollama:llama3.2:3b,ollama:qwen3:1.7b --seeds=1 \
      --out_dir=/tmp/minds

Each mind (`simulation.make_model`'s spec) plays the same scenario on the
same seeds, one run at a time (live minds share a model server). The table
gives, for each mind, over its runs:
- how many answers the World could not read as a move (`unread`), and how
  many moves named a custom the mover acts by (`cited`);
- "sow a field" moves, fields grown, and sowings that grew nothing;
- deaths, and the living at the end;
- own and expected norms kept when no one could see (`norms_kept`);
- seconds a year, and model calls.

The runs and the table are `sweep`'s. The harness only builds, plays and
reads stock simulations; every number comes from the census
(`Run.census_row`).
"""

from collections.abc import Sequence
import json
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
from examples.lifeways import simulation
from examples.lifeways import sweep

_SCENARIO = flags.DEFINE_string('scenario', 'first_farmers', 'Scenario.')
_YEARS = flags.DEFINE_integer('years', 24, 'Years each run plays.')
_SEEDS = flags.DEFINE_integer('seeds', 1, 'How many seeds.')
_FIRST_SEED = flags.DEFINE_integer('first_seed', 0, 'The first seed.')
_MINDS = flags.DEFINE_list(
    'minds',
    ['mock'],
    'Who people think with, each `mock` or <api>:<model> (e.g.'
    ' ollama:llama3.2:3b); a Together key comes from the environment.',
)
_SETTINGS = flags.DEFINE_string(
    'settings',
    '{}',
    'More settings, as JSON (e.g. {"situation_first": true}).',
)
_OUT_DIR = flags.DEFINE_string('out_dir', '', 'Where to write the results.')


def main(argv: Sequence[str]) -> None:
  del argv
  results: dict[str, list[dict[str, Any]]] = {}
  for mind in _MINDS.value:
    for seed in range(_FIRST_SEED.value, _FIRST_SEED.value + _SEEDS.value):
      label = 'MOCK' if mind == simulation.MOCK else f'LIVE ({mind})'
      logging.info(
          '%s: %s, seed %d, %d years',
          label,
          _SCENARIO.value,
          seed,
          _YEARS.value,
      )
      with simulation.quiet():
        played = sweep.play_mind(
            mind,
            _SCENARIO.value,
            seed,
            _YEARS.value,
            json.loads(_SETTINGS.value),
        )
      results.setdefault(mind, []).append(played)
      if _OUT_DIR.value:  # Each run as it ends: a long comparison keeps all.
        os.makedirs(_OUT_DIR.value, exist_ok=True)
        name = f'{mind.replace(":", "_").replace("/", "_")}_seed{seed}'
        with open(os.path.join(_OUT_DIR.value, f'{name}.json'), 'w') as f:
          json.dump({'mind': mind, 'seed': seed, **played}, f)
  title = f'{_SCENARIO.value}, {_YEARS.value} years, {_SEEDS.value} seed(s)'
  table = sweep.minds_report(
      {
          m: [{k: v for k, v in r.items() if k != 'rows'} for r in runs]
          for m, runs in results.items()
      },
      title,
  )
  print(table)
  if _OUT_DIR.value:
    with open(os.path.join(_OUT_DIR.value, 'report.md'), 'w') as f:
      f.write(table)


if __name__ == '__main__':
  app.run(main)
