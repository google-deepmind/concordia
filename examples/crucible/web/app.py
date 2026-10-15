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

"""The phone page's game sessions and HTTP handler (no flags; see server.py)."""

import dataclasses
import http.server
import json
import os
import threading
import time
from typing import Any
import uuid

from concordia.contrib.components.agent import blocking_input

from examples.crucible import harness
from examples.crucible import model_setup
from examples.crucible import toys

_HERE = os.path.dirname(__file__)


@dataclasses.dataclass
class Session:
  """One game in progress (the server holds at most one)."""

  game_id: str
  game: harness.Game
  reader: blocking_input.BlockingInput
  telemetry_path: str
  mode: str = 'mock'
  thread: threading.Thread | None = None
  finished: bool = False
  error: str = ''
  asked_at: float = 0.0
  last_request_id: str = ''
  previews: list[str] = dataclasses.field(default_factory=list)


class Games:
  """Starts, runs and records games; thread-safe."""

  def __init__(
      self,
      mode: str,
      model_name: str | None,
      max_calls: int,
      telemetry_dir: str,
      live: list[str] | None = None,
  ):
    self._mode = mode
    self.modes = ['mock', *(live or [])] if mode == 'mock' else [mode]
    self._model_name = model_name
    self._max_calls = max_calls
    self._telemetry_dir = telemetry_dir
    self._lock = threading.Lock()
    self.session: Session | None = None
    os.makedirs(telemetry_dir, exist_ok=True)

  def _record(self, session: Session, entry: dict[str, Any]) -> None:
    entry = {'t': round(time.time(), 3), **entry}
    with open(session.telemetry_path, 'a') as f:
      f.write(json.dumps(entry) + '\n')

  def start(
      self, toy: str, mode: str | None = None, scenario: str | None = None
  ) -> Session:
    """Ends any game in progress and starts a new one of `toy`."""
    self.quit()
    mode = mode if mode in self.modes else self.modes[0]
    name = self._model_name if mode == self._mode else None
    model = model_setup.build_model(mode, name, self._max_calls)
    reader = blocking_input.BlockingInput()
    data = toys.load_data(toy)
    if scenario not in data.get('scenarios', {}):
      scenario = None
    game = harness.build_game(
        toy,
        model,
        'human',
        seed=int(time.time()) % 1000,
        scenario=scenario,
        reader=reader,
    )
    game_id = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    session = Session(
        game_id=game_id,
        game=game,
        reader=reader,
        telemetry_path=os.path.join(self._telemetry_dir, f'{game_id}.jsonl'),
        mode=mode,
    )
    self._record(
        session,
        {
            'kind': 'start',
            'toy': toy,
            'scenario': getattr(game.toy, 'scenario', None),
            'mode': mode,
            'model': name,
        },
    )  # fmt: skip
    session.thread = threading.Thread(
        target=self._run, args=(session,), daemon=True
    )
    with self._lock:
      self.session = session
    session.thread.start()
    return session

  def _run(self, session: Session) -> None:
    try:
      session.game.play()
    except blocking_input.InputClosedError:
      session.error = 'Game abandoned.'
    except Exception as error:  # pylint: disable=broad-exception-caught
      session.error = f'{type(error).__name__}: {error}'
    session.finished = True
    self._record(
        session,
        {
            'kind': 'end',
            'outcome': session.game.outcome(),
            'error': session.error,
            'truth': session.game.toy.get_state(),
        },
    )

  def quit(self) -> None:
    with self._lock:
      session, self.session = self.session, None
    if session is not None and not session.finished:
      session.reader.close()

  def state(self) -> dict[str, Any]:
    """What the page shows."""
    out: dict[str, Any] = {
        'toys': [
            {
                'toy': name,
                'scenarios': list(toys.load_data(name).get('scenarios', {})),
            }
            for name in toys.registry()
        ],
        'modes': self.modes,
        'game': None,
    }
    with self._lock:
      session = self.session
    if session is None:
      return out
    request = session.reader.pending()
    if request is not None and request.request_id != session.last_request_id:
      session.last_request_id = request.request_id
      session.asked_at = time.time()
      session.previews = []
    gm = session.game.sim.get_game_masters()[0]
    results = getattr(gm.get_component('__resolution__'), 'results', [])  # pyrefly: ignore[missing-attribute]
    out['game'] = {
        'id': session.game_id,
        'toy': session.game.toy.name,
        'scenario': getattr(session.game.toy, 'scenario', None),
        'seat': session.game.kind.seat,
        'finished': session.finished,
        'error': session.error,
        'outcome': session.game.outcome() if session.finished else None,
        'receipts': [str(r) for r in results],
        'request': (
            None
            if request is None
            else {
                'id': request.request_id,
                'context': request.context,
                'question': request.action_spec.call_to_action,
                'error': request.error,
            }
        ),
    }
    return out

  def preview(self, text: str) -> dict[str, Any]:
    with self._lock:
      session = self.session
    if session is None or session.reader.pending() is None:
      return {'ok': False, 'summary': 'Nothing to preview now.'}
    toy = session.game.toy
    receipt = toy.preview(session.game.kind.seat, text)
    session.previews.append(text)
    forecast = ''
    if session.mode == 'mock' and hasattr(toy, 'forecast'):
      forecast = toy.forecast(text)  # pyrefly: ignore[missing-attribute]
    return {
        'ok': True,
        'summary': receipt.summary,
        'steps': receipt.steps,
        'forecast': forecast,
    }

  def answer(self, request_id: str, text: str) -> dict[str, Any]:
    with self._lock:
      session = self.session
    if session is None:
      return {'ok': False}
    request = session.reader.pending()
    ok = session.reader.submit(request_id, text)
    if ok and request is not None:
      self._record(
          session,
          {
              'kind': 'move',
              'request_id': request_id,
              'answer': text,
              'seconds': round(time.time() - session.asked_at, 1),
              'previews': list(session.previews),
              'question': request.action_spec.call_to_action,
          },
      )
    return {'ok': ok}


def make_handler(games: Games, base_path: str) -> type[Any]:
  """The request handler class, bound to a `Games` and a URL prefix."""
  with open(os.path.join(_HERE, 'index.html')) as f:
    page = f.read().encode()
  base = base_path.rstrip('/')

  class Handler(http.server.BaseHTTPRequestHandler):
    """Serves the page and its small JSON API."""

    def log_message(self, format: str, *args: Any) -> None:  # pylint: disable=redefined-builtin
      del format, args

    def _path(self) -> str:
      path = self.path.split('?', 1)[0]
      if base and path.startswith(base):
        path = path[len(base) :]
      return path or '/'

    def _send(self, status: int, body: bytes, kind: str) -> None:
      self.send_response(status)
      self.send_header('Content-Type', kind)
      self.send_header('Cache-Control', 'no-store')
      self.send_header('Content-Length', str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def _json(self, data: Any, status: int = 200) -> None:
      self._send(status, json.dumps(data).encode(), 'application/json')

    def do_GET(self) -> None:  # pylint: disable=invalid-name
      path = self._path()
      if path in ('/', '/index.html'):
        self._send(200, page, 'text/html; charset=utf-8')
      elif path == '/api/state':
        self._json(games.state())
      else:
        self._send(404, b'not found', 'text/plain')

    def do_POST(self) -> None:  # pylint: disable=invalid-name
      length = int(self.headers.get('Content-Length', 0) or 0)
      try:
        body = json.loads(self.rfile.read(length) or b'{}')
      except json.JSONDecodeError:
        self._json({'ok': False, 'error': 'bad json'}, 400)
        return
      path = self._path()
      if path == '/api/start':
        toy = str(body.get('toy', ''))
        if toy not in toys.registry():
          self._json({'ok': False, 'error': 'unknown toy'}, 400)
          return
        games.start(
            toy, str(body.get('mode', 'mock')), body.get('scenario') or None
        )
        self._json({'ok': True})
      elif path == '/api/preview':
        self._json(games.preview(str(body.get('text', ''))))
      elif path == '/api/answer':
        self._json(
            games.answer(str(body.get('id', '')), str(body.get('text', '')))
        )
      elif path == '/api/quit':
        games.quit()
        self._json({'ok': True})
      else:
        self._json({'ok': False, 'error': 'not found'}, 404)

  return Handler


def serve(
    games: Games, port: int, base_path: str = ''
) -> http.server.ThreadingHTTPServer:
  """Starts serving on 127.0.0.1 in a background thread."""
  server = http.server.ThreadingHTTPServer(
      ('127.0.0.1', port), make_handler(games, base_path)
  )
  threading.Thread(target=server.serve_forever, daemon=True).start()
  return server
