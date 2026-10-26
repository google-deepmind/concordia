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
| `GET /api/kin?name=<name>` | a person's family, three generations |
| `GET /api/lineage?id=<custom>` | a custom's variants, who holds them, from whom |
| `GET /api/build` | the commit served |
| `GET /api/notebook?since=<id>` | the notebook's entries after an id |
| `GET /notebook.md` | the notebook, to keep |
| `POST /api/move` `{"move": ...}` | queue the Experimenter's move |
| `POST /api/pace` `{"years_per_second": ..., "paused": ...}` | pace |
| `POST /api/stop` | stop the world |
| `POST /api/note` `{"kind": ..., "text": ...}` | write in the notebook |
| `POST /api/new` | begin a new world, once this one has stopped |
| `POST /api/twin` `{"lever": ..., "value": ...}` | fork a twin |
| `POST /api/twin` `{"end": true}` | let the twin go |
"""

from collections.abc import Callable, Mapping, Sequence
import http.server
import json
import os
import re
import signal
import subprocess
import threading
from typing import Any
import urllib.parse

from absl import app
from absl import flags
from absl import logging
from examples.lifeways import simulation
from examples.lifeways.web import session as session_lib

_PORT = flags.DEFINE_integer('port', 8921, 'Port (127.0.0.1).')
_BASE = flags.DEFINE_string('base_path', '/lifeways', 'Path prefix.')
_SCENARIO = flags.DEFINE_string('scenario', 'sandbox', 'Scenario or preset.')
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

STATIC = os.path.join(os.path.dirname(__file__), 'static')
# The largest request body the server reads.
BODY_LIMIT = 64 * 1024
_YEAR = re.compile(r'-?[0-9]+')
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


class Served:
  """The session a server serves; once its world stops, a new one may begin."""

  def __init__(
      self,
      session: session_lib.Session,
      renew: Callable[[], session_lib.Session] | None = None,
  ):
    """Initializes the holder.

    Args:
      session: The session served.
      renew: Optional. Builds the session of a new world.
    """
    self.session = session
    self._renew = renew
    self._lock = threading.Lock()

  def begin_anew(self) -> session_lib.Session:
    """Begins a new world in place of a stopped one.

    The stopped world's save is set aside (`.stopped`); the new world is
    saved in its place as it plays.

    Returns:
      The new session, playing.

    Raises:
      RuntimeError: If the world still plays, or no new world can be built.
    """
    with self._lock:
      old = self.session
      if old.running:
        raise RuntimeError('The world still plays: stop it first.')
      if self._renew is None:
        raise RuntimeError('This server cannot begin a new world.')
      old.close()
      old.set_aside_save('a new world begins', '.stopped')
      self.session = self._renew()
      self.session.start()
      return self.session


def handler(
    served: Served, base_path: str, commit: str = ''
) -> type[http.server.BaseHTTPRequestHandler]:
  """A request handler serving a session under a path prefix.

  Args:
    served: The session served (which a new world may replace).
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
      session = served.session
      since = query.get('since', ['-1'])[0]
      if path == '/api/census' and not _YEAR.fullmatch(since):
        self._json(400, {'error': 'since is a year'})
        return
      after = query.get('since', ['0'])[0]
      if path == '/api/notebook' and not re.fullmatch(r'[0-9]+', after):
        self._json(400, {'error': 'since is an entry'})
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
          '/api/notebook': lambda: session.notebook(int(after)),
      }
      if path in routes:
        self._json(200, routes[path]())
      elif path == '/api/person':
        person = session.person(query.get('name', [''])[0])
        if person is None:
          self._json(404, {'error': 'nobody of that name lives'})
        else:
          self._json(200, person)
      elif path == '/api/kin':
        kin = session.kin(query.get('name', [''])[0])
        if kin is None:
          self._json(404, {'error': 'nobody of that name is remembered'})
        else:
          self._json(200, kin)
      elif path == '/api/lineage':
        lineage = session.lineage(query.get('id', [''])[0])
        if lineage is None:
          self._json(404, {'error': 'no such custom is remembered'})
        else:
          self._json(200, lineage)
      elif path == '/notebook.md':
        self._send(
            200,
            session.notebook_markdown().encode(),
            'text/markdown; charset=utf-8',
        )
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
      session = served.session
      length = self.headers.get('Content-Length', '0') or '0'
      if not re.fullmatch(r'[0-9]+', length) or int(length) > BODY_LIMIT:
        self.close_connection = True
        self._json(400, {'error': f'send at most {BODY_LIMIT} bytes'})
        return
      try:
        body = json.loads(self.rfile.read(int(length)) or b'{}')
      except (json.JSONDecodeError, UnicodeDecodeError):
        body = None
      if not isinstance(body, dict):
        self._json(400, {'error': 'send a JSON object'})
        return
      if path == '/api/move':
        try:
          session.move(str(body.get('move', '')))
        except ValueError as error:
          self._json(400, {'error': str(error)})
          return
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
        self._twin(session, body)
      elif path == '/api/new':
        try:
          renewed = served.begin_anew()
        except RuntimeError as error:
          self._json(409, {'error': str(error)})
          return
        self._json(200, {'year': renewed.state()['year']})
      elif path == '/api/note':
        try:
          entry = session.note(str(body.get('kind', '')), body.get('text', ''))
        except ValueError as error:
          self._json(400, {'error': str(error)})
          return
        self._json(200, entry)
      else:
        self._json(404, {'error': 'not found'})

    def _twin(
        self, session: session_lib.Session, body: Mapping[str, Any]
    ) -> None:
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
    session: session_lib.Session | Served, port: int, base_path: str
) -> http.server.ThreadingHTTPServer:
  """A server for a session (call `serve_forever`).

  Args:
    session: The session served, or a holder that may begin a new world.
    port: The port on 127.0.0.1 (0: a free one).
    base_path: The path prefix.

  Returns:
    The server.
  """
  served = session if isinstance(session, Served) else Served(session)
  return http.server.ThreadingHTTPServer(
      ('127.0.0.1', port), handler(served, base_path, build())
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

  def world() -> session_lib.Session:
    return session_lib.Session(
        _SCENARIO.value,
        seed=_SEED.value,
        years_per_second=_PACE.value,
        save_path=_SAVE.value or None,
        model=model,
        minds=_MODEL.value,
        **overrides,
    )

  served = Served(world(), renew=world)
  server = serve(served, _PORT.value, _BASE.value)
  logging.info(
      '%s: %s, seed %d, at http://127.0.0.1:%d%s/ (%s)',
      'MOCK' if _MODEL.value == simulation.MOCK else f'LIVE ({_MODEL.value})',
      _SCENARIO.value,
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
