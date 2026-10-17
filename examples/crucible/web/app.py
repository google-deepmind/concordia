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
import datetime
import http.server
import json
import os
import threading
import time
from typing import Any
import uuid

from concordia.contrib.components.agent import blocking_input
from concordia.contrib.language_models import fallback as fallback_lib

from examples.crucible import campaign
from examples.crucible import harness
from examples.crucible import model_setup
from examples.crucible import toys

_HERE = os.path.dirname(__file__)


MAIN_TOY = 'passiton'  # the game the page leads with


def _optional_data(name: str) -> dict[str, Any]:
  try:
    return toys.load_data(name)
  except FileNotFoundError:
    return {}


@dataclasses.dataclass
class Session:
  """One game in progress (the server holds at most one)."""

  game_id: str
  game: harness.Game
  reader: blocking_input.BlockingInput
  telemetry_path: str
  mode: str = 'mock'
  daily: str = ''  # the date, for a daily challenge
  model: Any = None
  # Live minds: the mock answers a call the provider cannot (cycle 88).
  fallback: fallback_lib.FallbackLanguageModel | None = None
  thread: threading.Thread | None = None
  finished: bool = False
  abandoned: bool = False  # ended by a new game or quit, not played out
  error: str = ''
  asked_at: float = 0.0
  last_request_id: str = ''
  previews: list[str] = dataclasses.field(default_factory=list)
  # The board as it was when the pending order was asked: served while the
  # turn resolves, when the ledger is half applied (cycle 91).
  shown_view: Any = None
  shown_turns: int = -1  # receipts there were when it was taken


class Games:
  """Starts, runs and records games; thread-safe."""

  def __init__(
      self,
      mode: str,
      model_name: str | None,
      max_calls: int,
      telemetry_dir: str,
      live: list[str] | None = None,
      budget_usd: float = 0.0,
      price_per_mtok: float = 1.0,
  ):
    """Initializes the games.

    Args:
      mode: The default mind ('mock', 'ollama' or 'together').
      model_name: The default mind's model name.
      max_calls: The most model calls one game may make.
      telemetry_dir: Where games, campaign progress and spend are kept.
      live: Live minds a person may pick besides the mock.
      budget_usd: Paid minds ('together') are offered only while the spend
        recorded in this folder stays under this.
      price_per_mtok: Estimated dollars per million tokens of a paid mind.
    """
    self._mode = mode
    self._live_modes = ['mock', *(live or [])] if mode == 'mock' else [mode]
    self._budget_usd = budget_usd
    self._price = price_per_mtok
    self._model_name = model_name
    self._max_calls = max_calls
    self._telemetry_dir = telemetry_dir
    self._lock = threading.Lock()
    # Held across a whole start, so two quick starts cannot both run games
    # (cycle 88: the first was left blocked and its record unfinished).
    self._start_lock = threading.Lock()
    self.session: Session | None = None
    os.makedirs(telemetry_dir, exist_ok=True)
    self.progress = campaign.Progress(
        os.path.join(telemetry_dir, 'campaign.json')
    )
    self._spend_path = os.path.join(telemetry_dir, 'spend.json')

  def spent_usd(self) -> float:
    """The estimated spend of every paid game recorded so far."""
    if not os.path.exists(self._spend_path):
      return 0.0
    with open(self._spend_path) as f:
      return round(sum(row['usd'] for row in json.load(f)), 4)

  def _add_spend(self, row: dict[str, Any]) -> None:
    rows = []
    if os.path.exists(self._spend_path):
      with open(self._spend_path) as f:
        rows = json.load(f)
    rows.append(row)
    with open(self._spend_path, 'w') as f:
      json.dump(rows, f, indent=1)

  @property
  def modes(self) -> list[str]:
    """The minds a person may pick now (paid ones only within budget)."""
    return [
        m
        for m in self._live_modes
        if m != 'together' or self.spent_usd() < self._budget_usd
    ]

  def _record(self, session: Session, entry: dict[str, Any]) -> None:
    entry = {'t': round(time.time(), 3), **entry}
    with open(session.telemetry_path, 'a') as f:
      f.write(json.dumps(entry) + '\n')

  def start(
      self,
      toy: str,
      mode: str | None = None,
      scenario: str | None = None,
      daily: bool = False,
  ) -> Session:
    """Ends any game in progress and starts a new one of `toy`.

    Args:
      toy: The toy.
      mode: 'mock' or a live mind the server allows.
      scenario: The toy's scene; a campaign chapter must be open.
      daily: Play today's challenge (its scene and seed come from the date).

    Returns:
      The new session.

    Raises:
      ValueError: If the scene is a chapter not yet open.
    """
    with self._start_lock:
      return self._start(toy, mode, scenario, daily)

  def _start(
      self, toy: str, mode: str | None, scenario: str | None, daily: bool
  ) -> Session:
    data = toys.load_data(toy)
    seed = int(time.time()) % 1000
    today = ''
    if daily and data.get('daily'):
      challenge = campaign.daily(data, datetime.date.today())
      scenario, seed, today = (
          challenge['scenario'],
          challenge['seed'],
          challenge['date'],
      )
      # A daily is a late chapter's scene: it opens with that chapter
      # (cycle 96: the owner went from chapter 1 to a finale and lost).
      if not campaign.is_open(
          data, self.progress.snapshot(), challenge['opens_with']
      ):
        raise ValueError("Today's challenge opens with its chapter.")
    elif scenario and not campaign.is_open(
        data, self.progress.snapshot(), scenario
    ):
      raise ValueError(f'Chapter {scenario!r} is not open yet.')
    self.quit()
    mode = mode if mode in self.modes else self.modes[0]
    name = self._model_name if mode == self._mode else None
    model = model_setup.build_model(mode, name, self._max_calls)
    counter = spare = None
    if mode != 'mock':
      # A provider error (after retries) or a spent budget no longer ends
      # the game or fills it with empty readings: the mock, which reads by
      # the same laws, answers that call.
      counter = model_setup.UsageCounter(model)
      spare = fallback_lib.FallbackLanguageModel(
          counter, model_setup.mock_model(), max_primary_calls=self._max_calls
      )
      model = spare
    reader = blocking_input.BlockingInput()
    if scenario not in data.get('scenarios', {}) and not today:
      scenario = None  # a daily's scene may be made by the toy (Pass it on)
    game = harness.build_game(
        toy,
        model,
        'human',
        seed=seed,
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
        daily=today,
        model=counter or model,
        fallback=spare,
    )
    self._record(
        session,
        {
            'kind': 'start',
            'toy': toy,
            'scenario': getattr(game.toy, 'scenario', None),
            'daily': today,
            'seed': seed,
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
      session.abandoned = True
    except Exception as error:  # pylint: disable=broad-exception-caught
      session.error = f'{type(error).__name__}: {error}'
    session.finished = True
    usage = None
    if isinstance(session.model, model_setup.UsageCounter):
      tokens = session.model.tokens
      usd = tokens / 1e6 * self._price if session.mode == 'together' else 0.0
      usage = {'calls': session.model.calls, 'est_tokens': tokens,
               'usd': round(usd, 4)}  # fmt: skip
      if session.fallback is not None:
        usage['fallbacks'] = session.fallback.fallbacks
        usage['fallback_error'] = session.fallback.last_error
      if usd:
        self._add_spend(
            {'game': session.game_id, 'mode': session.mode, **usage}
        )
    outcome = session.game.outcome()
    scenario = getattr(session.game.toy, 'scenario', None)
    if scenario and not session.error and 'stars' in outcome:
      key = f'daily:{session.daily}' if session.daily else scenario
      self.progress.record(key, int(outcome['stars']))
    self._record(
        session,
        {
            'kind': 'end',
            'outcome': session.game.outcome(),
            'usage': usage,
            'error': session.error,
            'abandoned': session.abandoned,
            'truth': session.game.toy.get_state(),
        },
    )

  def visit(self) -> None:
    """Notes that the page was opened (a timestamp, nothing else)."""
    with self._lock:
      with open(os.path.join(self._telemetry_dir, 'visits.jsonl'), 'a') as f:
        f.write(json.dumps({'t': time.strftime('%Y-%m-%dT%H:%M:%S')}) + '\n')

  def quit(self) -> None:
    with self._lock:
      session, self.session = self.session, None
    if session is not None and not session.finished:
      session.reader.close()

  def state(self) -> dict[str, Any]:
    """What the page shows."""
    out: dict[str, Any] = {
        # The main game's words over the shared look (palette, glyphs).
        'look': {
            **toys.load_data('orders_look'),
            **_optional_data(f'{MAIN_TOY}_look'),
        },
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
    # Every toy's campaign; the page leads with MAIN_TOY's (cycle 98: Pass
    # it on), and keeps the others as earlier versions.
    out['campaigns'] = {}
    for name in toys.registry():
      data = toys.load_data(name)
      if data.get('campaign'):
        progress = self.progress.snapshot()
        chapters = {
            'toy': name,
            'title': data.get('title', name),
            'chapters': campaign.chapters(data, progress),
        }
        if data.get('daily'):
          challenge = campaign.daily(data, datetime.date.today())
          challenge['stars'] = progress.get(f'daily:{challenge["date"]}', 0)
          chapters['daily'] = challenge
        out['campaigns'][name] = chapters
    if out['campaigns']:
      out['campaign'] = out['campaigns'].get(
          MAIN_TOY, next(iter(out['campaigns'].values()))
      )
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
        'daily': session.daily,
        'seat': session.game.kind.seat,
        'finished': session.finished,
        'error': session.error,
        'outcome': session.game.outcome() if session.finished else None,
        'receipts': [str(r) for r in results],
        'view': self._view(session, request, len(results)),
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

  def _view(self, session: Session, request: Any, turns: int) -> Any:
    """The board, except while a turn resolves: then as it was asked.

    Args:
      session: The game.
      request: The order being asked, if any.
      turns: How many turns have a receipt.

    Returns:
      The toy's view, or None for a toy without one.
    """
    toy = session.game.toy
    if not hasattr(toy, 'view'):
      return None
    resolving = (
        request is None
        and not session.finished
        and turns == session.shown_turns
        and session.shown_view is not None
    )
    if not resolving:
      session.shown_view = toy.view()  # pyrefly: ignore[missing-attribute]
      session.shown_turns = turns
    return session.shown_view

  def preview(self, text: str) -> dict[str, Any]:
    with self._lock:
      session = self.session
    if session is None or session.reader.pending() is None:
      return {'ok': False, 'summary': 'Nothing to preview now.'}
    toy = session.game.toy
    session.previews.append(text)  # tried, even if the preview fails
    if hasattr(toy, 'rehearse'):  # Pass it on: the first hop only
      try:
        hop = toy.rehearse(text)  # pyrefly: ignore[missing-attribute]
      except Exception as error:  # pylint: disable=broad-exception-caught
        return {
            'ok': False,
            'summary': f'The rehearsal failed ({type(error).__name__}).',
        }
      if not hop:
        return {
            'ok': True,
            'summary': (
                'Nothing to rehearse: say something, and someone'
                ' must stand between you and the top.'
            ),
        }
      if 'error' in hop:
        return {
            'ok': False,
            'summary': f'Too long to rehearse: {hop["error"]}.',
        }
      return {
          'ok': True,
          'summary': f'{hop["who"]} would pass it on as: "{hop["said"]}"',
          'rehearsal': hop,
      }
    try:
      receipt, world = toy.simulate(session.game.kind.seat, text)
    except Exception as error:  # pylint: disable=broad-exception-caught
      return {
          'ok': False,
          'summary': f'The preview failed ({type(error).__name__}); try again.',
      }
    predicted = None
    log = list(getattr(world, 'log', ()))
    entry = log[-1] if log else {}
    if isinstance(entry, dict) and 'actions' in entry:
      predicted = entry
    forecast = ''
    if session.mode == 'mock' and hasattr(toy, 'forecast'):
      forecast = toy.forecast(text)  # pyrefly: ignore[missing-attribute]
    return {
        'ok': True,
        'summary': receipt.summary,
        'forecast': forecast,
        'predicted': predicted,
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
        games.visit()
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
        try:
          games.start(
              toy,
              str(body.get('mode', 'mock')),
              body.get('scenario') or None,
              bool(body.get('daily', False)),
          )
        except ValueError as error:
          self._json({'ok': False, 'error': str(error)}, 403)
          return
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
