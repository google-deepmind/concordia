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

| Path (under `--base_path`) | What |
|---|---|
| `/` | the page |
| `/static/<file>` | its scripts and styles |
| `GET /api/state` | the world: map, people, places, pace |
| `GET /api/census?since=<year>[&world=twin]` | census rows after a year |
| `GET /api/person?name=<name>` | one person's customs and kin |
| `GET /api/build` | the commit served |
| `POST /api/move` `{"move": ...}` | queue the Experimenter's move |
| `POST /api/pace` `{"years_per_second": ..., "paused": ...}` | pace |
| `POST /api/stop` | stop the world |
| `POST /api/twin` `{"lever": ..., "value": ...}` | fork a twin |
| `POST /api/twin` `{"end": true}` | let the twin go |
"""

from collections.abc import Callable, Mapping, Sequence
import http.server
import json
import os
import signal
import subprocess
from typing import Any
import urllib.parse

from absl import app
from absl import flags
from absl import logging
from examples.lifeways.web import session as session_lib

_PORT = flags.DEFINE_integer('port', 8921, 'Port (127.0.0.1).')
_BASE = flags.DEFINE_string('base_path', '/lifeways', 'Path prefix.')
_SCENARIO = flags.DEFINE_string('scenario', 'sandbox', 'Scenario or preset.')
_SEED = flags.DEFINE_integer('seed', 0, 'World seed.')
_PACE = flags.DEFINE_float('years_per_second', 1.0, 'Starting pace.')
_SAVE = flags.DEFINE_string(
    'save', '', 'File the world is saved to and restored from (none if empty).'
)

STATIC = os.path.join(os.path.dirname(__file__), 'static')
_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
}


def build() -> str:
  """The commit the server runs, if it runs from a git checkout.

  Returns:
    The short commit hash, or '' outside a checkout.
  """
  try:
    out = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.dirname(__file__),
        timeout=5,
    )
    return out.stdout.strip()
  except (OSError, subprocess.SubprocessError):
    return ''


def handler(
    session: session_lib.Session, base_path: str, commit: str = ''
) -> type[http.server.BaseHTTPRequestHandler]:
  """A request handler serving one session under a path prefix.

  Args:
    session: The session served.
    base_path: The path prefix (e.g. /lifeways).
    commit: The build the page reports.

  Returns:
    The handler class.
  """
  base = '/' + base_path.strip('/') if base_path.strip('/') else ''

  class Handler(http.server.BaseHTTPRequestHandler):
    """Routes the page's requests to the session."""

    def log_message(self, format: str, *args: Any) -> None:  # pylint: disable=redefined-builtin
      logging.debug(format, *args)

    def _send(
        self, status: int, body: bytes, kind: str = 'application/json'
    ) -> None:
      self.send_response(status)
      self.send_header('Content-Type', kind)
      self.send_header('Content-Length', str(len(body)))
      self.send_header('Cache-Control', 'no-store')
      self.end_headers()
      self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
      self._send(status, json.dumps(data).encode())

    def _path(self) -> tuple[str, dict[str, list[str]]] | None:
      url = urllib.parse.urlparse(self.path)
      if not url.path.startswith(base):
        return None
      return url.path[len(base) :] or '/', urllib.parse.parse_qs(url.query)

    def do_GET(self) -> None:  # pylint: disable=invalid-name
      if base and urllib.parse.urlparse(self.path).path == base:
        # The page's links are relative: it lives at the prefix's folder.
        self.send_response(301)
        self.send_header('Location', base + '/')
        self.send_header('Content-Length', '0')
        self.end_headers()
        return
      found = self._path()
      if found is None:
        self._json(404, {'error': 'not found'})
        return
      path, query = found
      since = query.get('since', ['-1'])[0]
      if path == '/api/census' and not since.lstrip('-').isdigit():
        self._json(400, {'error': 'since is a year'})
        return
      world = query.get('world', [''])[0]
      if path == '/api/census' and world not in ('', 'twin'):
        self._json(400, {'error': 'world is the twin, or none'})
        return

      def census() -> list[dict[str, Any]]:
        if world != 'twin':
          return session.census(int(since))
        twin = session.twin
        return [] if twin is None else twin.census(int(since))

      routes: Mapping[str, Callable[[], Any]] = {
          '/api/state': session.state,
          '/api/census': census,
          '/api/build': lambda: {'commit': commit},
      }
      if path in routes:
        self._json(200, routes[path]())
      elif path == '/api/person':
        person = session.person(query.get('name', [''])[0])
        if person is None:
          self._json(404, {'error': 'nobody of that name lives'})
        else:
          self._json(200, person)
      elif path in ('/', '/index.html'):
        self._static('index.html')
      elif path.startswith('/static/'):
        self._static(path[len('/static/') :])
      else:
        self._json(404, {'error': 'not found'})

    def do_POST(self) -> None:  # pylint: disable=invalid-name
      found = self._path()
      if found is None:
        self._json(404, {'error': 'not found'})
        return
      path, _ = found
      length = int(self.headers.get('Content-Length', '0') or 0)
      try:
        body = json.loads(self.rfile.read(length) or b'{}')
      except json.JSONDecodeError:
        body = None
      if not isinstance(body, dict):
        self._json(400, {'error': 'send a JSON object'})
        return
      if path == '/api/move':
        session.move(str(body.get('move', '')))
        self._json(200, session.input.settings())
      elif path == '/api/pace':
        pace, paused = body.get('years_per_second'), body.get('paused')
        if (
            pace is not None
            and (isinstance(pace, bool) or not isinstance(pace, (int, float)))
        ) or (paused is not None and not isinstance(paused, bool)):
          self._json(400, {'error': 'a number of years a second; paused true'})
          return
        self._json(200, session.pace(pace, paused))
      elif path == '/api/stop':
        session.stop()
        self._json(200, session.input.settings())
      elif path == '/api/twin':
        self._twin(body)
      else:
        self._json(404, {'error': 'not found'})

    def _twin(self, body: Mapping[str, Any]) -> None:
      if body.get('end') is True:
        twin = session.end_twin()
        if twin is not None:
          twin.close()
        self._json(200, {'twin': None})
        return
      try:
        twin = session.fork(str(body.get('lever', '')), body.get('value'))
      except ValueError as error:
        self._json(400, {'error': str(error)})
        return
      except RuntimeError as error:
        self._json(409, {'error': str(error)})
        return
      self._json(200, {'twin': twin.twin_of})

    def _static(self, name: str) -> None:
      path = os.path.normpath(os.path.join(STATIC, name))
      if not path.startswith(STATIC + os.sep) or not os.path.isfile(path):
        self._json(404, {'error': 'not found'})
        return
      with open(path, 'rb') as f:
        body = f.read()
      kind = _TYPES.get(os.path.splitext(path)[1], 'application/octet-stream')
      self._send(200, body, kind)

  return Handler


def serve(
    session: session_lib.Session, port: int, base_path: str
) -> http.server.ThreadingHTTPServer:
  """A server for a session (call `serve_forever`).

  Args:
    session: The session served.
    port: The port on 127.0.0.1 (0: a free one).
    base_path: The path prefix.

  Returns:
    The server.
  """
  return http.server.ThreadingHTTPServer(
      ('127.0.0.1', port), handler(session, base_path, build())
  )


def _stopped(signum: int, frame: Any) -> None:
  del frame
  # Once is enough: a second signal must not cut short the save.
  signal.signal(signum, signal.SIG_IGN)
  raise SystemExit(f'Stopped by signal {signum}.')


def main(argv: Sequence[str]) -> None:
  del argv
  session = session_lib.Session(
      _SCENARIO.value,
      seed=_SEED.value,
      years_per_second=_PACE.value,
      save_path=_SAVE.value or None,
  )
  server = serve(session, _PORT.value, _BASE.value)
  logging.info(
      'MOCK: %s, seed %d, at http://127.0.0.1:%d%s/ (%s)',
      _SCENARIO.value,
      _SEED.value,
      server.server_address[1],
      _BASE.value,
      f'restored at year {session.restored_year}'
      if session.restored_year is not None
      else 'a new world',
  )
  # launchd and the test script stop the server with SIGTERM: save first.
  signal.signal(signal.SIGTERM, _stopped)
  session.start()
  try:
    server.serve_forever()
  finally:
    if session.save_soon(timeout=3.0):
      logging.info('Saved at year %s.', session.saved_year)
    session.close()
    session_lib.await_engines()
    server.server_close()


if __name__ == '__main__':
  app.run(main)
