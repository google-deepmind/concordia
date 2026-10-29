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

"""Serves a Lifeways session over HTTP: the page and its API.

The library of `server` (the binary, whose flags it does not define): the
holder of the session served (`Served`), the request handler and `serve`.
Everything it does with the world is read the simulation
(`session.Session`), or ask the session to queue the Experimenter's moves.

| Path (under the base path) | What |
|---|---|
| `/` | the page |
| `/static/<file>` | its scripts and styles |
| `GET /api/state` | the world: map, people, places, pace |
| `GET /api/census?since=<year>[&world=twin]` | census rows after a year |
| `GET /api/person?name=<name>` | one person's customs and kin |
| `GET /api/kin?name=<name>` | a person's family, three generations |
| `GET /api/lineage?id=<custom>` | a custom's variants, who holds them, from whom |
| `GET /api/anatomy[?name=<name>]` | how Concordia builds a person, or the World GM |
| `GET /api/place?id=<place>` | a place: who keeps it, what it holds |
| `GET /api/search?q=<words>` | the living and the customs that match |
| `GET /api/customs?ids=<id,id,...>` | customs' words by id |
| `GET /api/release_preview?kind=..&domain=..&at=r,c[&level=n]` | who a song would reach, and what they would let go for it |
| `GET /api/build` | the commit served |
| `GET /api/notebook?since=<id>` | the notebook's entries after an id |
| `GET /notebook.md` | the notebook, to keep |
| `GET /report.html` | the world on one page, to keep or share |
| `GET /data/census.csv` | the census rows kept, a row a year |
| `GET /data/people.csv` | who holds what now: a row per living person and custom |
| `GET /data/replay.json` | the world to play again: `run.py --replay` |
| `GET /data/surveys.csv` | each survey answer kept, with the customs it named |
| `POST /api/move` `{"move": ...}` | queue the Experimenter's move |
| `POST /api/pace` `{"years_per_second": ..., "paused": ...}` | pace |
| `POST /api/stop` | stop the world |
| `POST /api/note` `{"kind": ..., "text": ..., "check": ...}` | write in the notebook; a hypothesis may check itself (`{"measure", "direction", "years"}`) |
| `POST /api/new` `{"scenario": ..., "play_as": ...}` | begin a new world, once this one stops; play a founder in it if `play_as` |
| `POST /api/me` `{"id": ..., "answer": ...}` | answer the played person's question |
| `POST /api/twin` `{"lever": ..., "value": ...}` | fork a twin |
| `POST /api/twin` `{"end": true}` | let the twin go |
| `GET /api/lab` | the lab's experiment playing or last played |
| `POST /api/lab` `{"kind", "seeds", "years", "lever", "value"}` | play an experiment on this world's settings in the background: `compare` (a lever pulled) or `replicate` (other seeds) |
"""

from collections.abc import Callable, Mapping
import http.server
import json
import os
import re
import subprocess
import threading
from typing import Any
import urllib.parse

from absl import logging
from examples.lifeways import scenarios
from examples.lifeways.web import session as session_lib

STATIC = os.path.join(os.path.dirname(__file__), 'static')
# The largest request body the server reads.
BODY_LIMIT = 64 * 1024
_YEAR = re.compile(r'-?[0-9]+')
_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
}
# The data the page offers to keep, by path: the session's method that
# writes it.
_DATA = {
    '/data/census.csv': 'census_csv',
    '/data/people.csv': 'people_csv',
    '/data/surveys.csv': 'surveys_csv',
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
      renew: Callable[[str, bool], session_lib.Session] | None = None,
  ):
    """Initializes the holder.

    Args:
      session: The session served.
      renew: Optional. Builds the session of a new world of a scenario, in
        which a founder is played if asked (`Session(play_as=...)`).
    """
    self.session = session
    self._renew = renew
    self._lock = threading.Lock()

  def begin_anew(
      self, scenario: str = '', play_as: bool = False
  ) -> session_lib.Session:
    """Begins a new world in place of a stopped one.

    The stopped world's save is set aside (`.stopped`); the new world is
    saved in its place as it plays.

    Args:
      scenario: The new world's scenario (a preset); the stopped one's if
        empty.
      play_as: Whether the page plays a founder in the new world.

    Returns:
      The new session, playing.

    Raises:
      ValueError: If there is no such scenario, or a person is to be played
        where none can be (the stopped world is kept then).
      RuntimeError: If the world still plays, or no new world can be built.
    """
    with self._lock:
      old = self.session
      scenario = scenario or old.scenario
      if scenario not in scenarios.PRESETS:
        raise ValueError(f'No scenario {scenario!r}.')
      if old.running:
        raise RuntimeError('The world still plays: stop it first.')
      if self._renew is None:
        raise RuntimeError('This server cannot begin a new world.')
      if play_as and not old.playable:  # Before the old world is let go.
        raise ValueError(session_lib.NOT_PLAYABLE)
      old.close()
      old.set_aside_save('a new world begins', '.stopped')
      self.session = self._renew(scenario, play_as)
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
      path = url.path
      if base and (path == base or path.startswith(base + '/')):
        path = path[len(base) :]
      # Else a proxy that mounts the page at the prefix has stripped it
      # (`tailscale serve --set-path`): the path is the page's own.
      return path or '/', urllib.parse.parse_qs(url.query)

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
      elif path == '/api/release_preview':
        try:
          row, col = (int(x) for x in query.get('at', [''])[0].split(','))
          level = int(query.get('level', ['0'])[0] or 0)
        except ValueError:
          self._json(400, {'error': 'at is <row>,<col>; level a number'})
        else:
          self._json(
              200,
              session.release_preview(
                  query.get('kind', [''])[0],
                  query.get('domain', [''])[0],
                  (row, col),
                  level,
              ),
          )
      elif path == '/api/customs':
        ids = [i for i in query.get('ids', [''])[0].split(',') if i]
        self._json(200, session.customs(ids))
      elif path == '/api/search':
        self._json(200, session.search(query.get('q', [''])[0]))
      elif path == '/api/place':
        place = session.place(query.get('id', [''])[0])
        if place is None:
          self._json(404, {'error': 'no such place stands'})
        else:
          self._json(200, place)
      elif path == '/api/anatomy':
        anatomy = session.anatomy(query.get('name', [''])[0])
        if anatomy is None:
          self._json(404, {'error': 'nobody of that name lives'})
        else:
          self._json(200, anatomy)
      elif path == '/api/lineage':
        lineage = session.lineage(query.get('id', [''])[0])
        if lineage is None:
          self._json(404, {'error': 'no such custom is remembered'})
        else:
          self._json(200, lineage)
      elif path == '/api/lab':
        self._json(200, {'job': session.lab_status()})
      elif path == '/notebook.md':
        self._send(
            200,
            session.notebook_markdown().encode(),
            'text/markdown; charset=utf-8',
        )
      elif path == '/data/replay.json':
        self._json(200, session.replay())
      elif path == '/report.html':
        self._send(
            200, session.report_html().encode(), 'text/html; charset=utf-8'
        )
      elif path in _DATA:
        self._send(
            200,
            getattr(session, _DATA[path])().encode(),
            'text/csv; charset=utf-8',
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
      elif path == '/api/lab':
        try:
          job = session.lab_start(
              str(body.get('kind', '')),
              int(body.get('seeds', 0)),
              int(body.get('years', 0)),
              str(body.get('lever', '')),
              body.get('value'),
          )
        except (TypeError, ValueError) as error:
          self._json(400, {'error': str(error)})
          return
        self._json(200, job)
      elif path == '/api/new':
        try:
          renewed = served.begin_anew(
              str(body.get('scenario') or ''), body.get('play_as') is True
          )
        except ValueError as error:
          self._json(400, {'error': str(error)})
          return
        except RuntimeError as error:
          self._json(409, {'error': str(error)})
          return
        self._json(200, {'year': renewed.state()['year']})
      elif path == '/api/me':
        self._me(session, body)
      elif path == '/api/note':
        try:
          check = body.get('check')
          entry = session.note(
              str(body.get('kind', '')),
              body.get('text', ''),
              check if isinstance(check, Mapping) else None,
          )
        except ValueError as error:
          self._json(400, {'error': str(error)})
          return
        self._json(200, entry)
      else:
        self._json(404, {'error': 'not found'})

    def _me(
        self, session: session_lib.Session, body: Mapping[str, Any]
    ) -> None:
      asked, answer = body.get('id'), body.get('answer')
      if not isinstance(asked, str) or not isinstance(answer, str):
        self._json(400, {'error': 'an id and an answer, both text'})
        return
      try:
        taken = session.answer(asked, answer)
      except RuntimeError as error:
        self._json(409, {'error': str(error)})
        return
      except ValueError as error:
        self._json(400, {'error': str(error)})
        return
      if not taken:
        self._json(409, {'error': 'that question is no longer asked'})
        return
      self._json(200, {'answered': asked})

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
