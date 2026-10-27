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

r"""Serves a running Lifeways world to a page, phone first (MOCK by default).

  python -m examples.lifeways.web.server --port=8921 --base_path=/lifeways

The page draws the valley, its people and places, the census as graphs and
a person's customs, and lets the person playing the Experimenter pace the
years, release customs and stop the world. Everything else the server does
is read the simulation (`session.Session`). With `--save`, the world outlives
the server: it is saved as it plays and when the server is stopped (SIGTERM),
and the next server of the same world plays on from the save.

The page and its API are `serving`'s.
"""

from collections.abc import Sequence
import signal
from typing import Any

from absl import app
from absl import flags
from absl import logging
from examples.lifeways import simulation
from examples.lifeways.web import serving
from examples.lifeways.web import session as session_lib

_PORT = flags.DEFINE_integer('port', 8921, 'Port (127.0.0.1).')
_BASE = flags.DEFINE_string('base_path', '/lifeways', 'Path prefix.')
_SCENARIO = flags.DEFINE_string(
    'scenario',
    'sandbox',
    'Scenario or preset of a new world (a saved world keeps its own).',
)
_SEED = flags.DEFINE_integer('seed', 0, 'World seed.')
_PACE = flags.DEFINE_float('years_per_second', 1.0, 'Starting pace.')
_MODEL = flags.DEFINE_string(
    'model',
    simulation.MOCK,
    'Who people think with: mock, or <api>:<model> for live minds, e.g.'
    ' ollama:llama3.2:3b (Together reads TOGETHER_API_KEY).',
)
_FREE = flags.DEFINE_float(
    'free',
    0.0,
    'If above 0, free mode: a year also closes after so many seconds (for'
    ' live minds, which are slow).',
)
_SAVE = flags.DEFINE_string(
    'save', '', 'File the world is saved to and restored from (none if empty).'
)


def _stopped(signum: int, frame: Any) -> None:
  del frame
  # Once is enough: a second signal must not cut short the save.
  signal.signal(signum, signal.SIG_IGN)
  raise SystemExit(f'Stopped by signal {signum}.')


def main(argv: Sequence[str]) -> None:
  del argv
  try:
    model, _ = simulation.make_model(_MODEL.value)
  except ValueError as error:
    raise app.UsageError(str(error)) from error
  overrides: dict[str, Any] = {'free': _FREE.value} if _FREE.value > 0 else {}

  def world(scenario: str) -> session_lib.Session:
    return session_lib.Session(
        scenario,
        seed=_SEED.value,
        years_per_second=_PACE.value,
        save_path=_SAVE.value or None,
        model=model,
        minds=_MODEL.value,
        **overrides,
    )

  # A world begun from the page in another preset is restored as itself.
  served = serving.Served(
      world(session_lib.saved_scenario(_SAVE.value) or _SCENARIO.value),
      renew=world,
  )
  server = serving.serve(served, _PORT.value, _BASE.value)
  logging.info(
      '%s: %s, seed %d, at http://127.0.0.1:%d%s/ (%s)',
      'MOCK' if _MODEL.value == simulation.MOCK else f'LIVE ({_MODEL.value})',
      served.session.scenario,
      _SEED.value,
      server.server_address[1],
      _BASE.value,
      f'restored at year {served.session.restored_year}'
      if served.session.restored_year is not None
      else 'a new world',
  )
  # launchd and the test script stop the server with SIGTERM: save first.
  signal.signal(signal.SIGTERM, _stopped)
  # The server prints nothing: the engine's own prints, from every world and
  # until their last turns have ended, go to the null device.
  with simulation.quiet():
    served.session.start()
    try:
      server.serve_forever()
    finally:
      session = served.session
      if session.save_soon(timeout=3.0):
        logging.info('Saved at year %s.', session.saved_year)
      session.close()
      session_lib.await_engines()
      server.server_close()


if __name__ == '__main__':
  app.run(main)
