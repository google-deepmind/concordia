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

r"""Plays one game of a crucible toy and writes a run folder.

  python -m examples.crucible.run --toy=claims --mode=mock --seat=skilled
  python -m examples.crucible.run --toy=claims --mode=mock --seat=human
  python -m examples.crucible.run --toy=claims --mode=ollama \
      --model_name=qwen3:8b --seat=model --max_calls=200

The folder holds `log.html` (Concordia's `SimulationLog`), `summary.json`
(the outcome, every move's receipt, model calls) and `moves.csv`.
"""

import json
import os
import time

from absl import app
from absl import flags

from examples.crucible import harness
from examples.crucible import model_setup

_TOY = flags.DEFINE_string('toy', 'claims', 'Which toy.')
_SCENARIO = flags.DEFINE_string('scenario', 'sandbox', 'Toy scenario.')
_MODE = flags.DEFINE_enum('mode', 'mock', ['mock', 'ollama', 'together'], '')
_MODEL_NAME = flags.DEFINE_string('model_name', None, 'Provider model name.')
_SEAT = flags.DEFINE_enum('seat', 'skilled', list(harness.SEATS), 'Seat.')
_SEED = flags.DEFINE_integer('seed', 0, 'Seed for the bots.')
_MAX_CALLS = flags.DEFINE_integer('max_calls', 400, 'Live model call cap.')
_OUT_DIR = flags.DEFINE_string('out_dir', '/tmp/crucible_runs', 'Run root.')


def main(argv: list[str]) -> None:
  del argv
  model = model_setup.build_model(
      _MODE.value, _MODEL_NAME.value, _MAX_CALLS.value
  )
  game = harness.build_game(
      _TOY.value, model, _SEAT.value, _SEED.value, _SCENARIO.value
  )
  label = 'MOCK' if _MODE.value == 'mock' else f'{_MODE.value.upper()}'
  print(f'{label}: {_TOY.value}/{_SCENARIO.value}, seat {_SEAT.value}')
  start = time.time()
  log = game.play()
  out_dir = os.path.join(
      _OUT_DIR.value, f'{_TOY.value}-{_SEAT.value}-{int(time.time())}'
  )
  summary = harness.write_run(game, log, out_dir)
  for r in summary['receipts']:
    print(f"- {r['summary']}")
  print(json.dumps(summary['outcome']))
  print(f'{time.time() - start:.1f} s; run folder: {out_dir}')


if __name__ == '__main__':
  app.run(main)
