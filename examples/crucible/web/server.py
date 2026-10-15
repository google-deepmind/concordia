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

r"""A plain-text page to play any crucible toy from a phone.

The person sits in the toy's seat through Concordia's `HumanActComponent`;
the server is its reader (contrib `BlockingInput`). The page shows exactly
what a model in that seat would read, lets the person preview a draft move
(measured on a copy of the world), and sends the move. Every game is written
to the telemetry folder: each move, what was on screen, how long it took,
the previews tried, and at the end the outcome and the hidden state.

  python -m examples.crucible.web.server --port 8901 --base_path /crucible

It listens on the loopback interface only and has no authentication; reach
it through a private tunnel, never the open internet.
"""

import threading

from absl import app as absl_app
from absl import flags

from examples.crucible.web import app

_PORT = flags.DEFINE_integer('port', 8901, 'Port (loopback only).')
_BASE_PATH = flags.DEFINE_string('base_path', '', 'URL prefix, e.g. /crucible')
_MODE = flags.DEFINE_enum('mode', 'mock', ['mock', 'ollama', 'together'], '')
_MODEL_NAME = flags.DEFINE_string('model_name', None, 'Provider model name.')
_MAX_CALLS = flags.DEFINE_integer('max_calls', 300, 'Live call cap per game.')
_LIVE = flags.DEFINE_list(
    'live', [], 'Live minds the page may pick besides the mock, e.g. ollama.'
)
_TELEMETRY_DIR = flags.DEFINE_string(
    'telemetry_dir', '/tmp/crucible_telemetry', 'Where games are recorded.'
)


def main(argv: list[str]) -> None:
  del argv
  games = app.Games(
      _MODE.value,
      _MODEL_NAME.value,
      _MAX_CALLS.value,
      _TELEMETRY_DIR.value,
      _LIVE.value,
  )
  server = app.serve(games, _PORT.value, _BASE_PATH.value)
  print(
      f'Crucible on http://127.0.0.1:{server.server_address[1]}'
      f'{_BASE_PATH.value}/ ({_MODE.value})',
      flush=True,
  )
  try:
    threading.Event().wait()
  except KeyboardInterrupt:
    server.shutdown()


if __name__ == '__main__':
  absl_app.run(main)
