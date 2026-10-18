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

"""A small web server to play or watch Deep Time from a phone.

Run it on the loopback interface and expose it to your own devices with a
private tunnel (for example `tailscale serve`), never to the open internet:
it has no authentication. The page uses relative URLs, so it also works
under a path (e.g. `tailscale serve --set-path /deep-time ...`). Pages:

* `/`: play any role (the Steward, a species, a people, Mission Control or
  the colonists), alone or with a friend on another phone (`/?seat=<role>`
  opens that seat), or watch the mock play, on any scenario; the page polls
  `/api/state` and posts answers to
  the role's pending `HumanInputRequest` to `/api/choose`.
* `/demo/`: run reports (`report.html`, `gazette.html`, `log.html`) of runs
  written by `run.py` into the demo folder.

  python -m examples.deep_time.web.server --port 8890 --demo_dir /tmp/demos
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-long-ternary,missing-function-docstring

from collections.abc import Sequence
import datetime
import gzip
import html
import http.server
import json
import os
import re
import subprocess
import threading
import time
from typing import Any
import urllib.parse

from absl import app
from absl import flags
from examples.deep_time import campaign as campaign_lib
from examples.deep_time import game as game_lib
from examples.deep_time import model_setup
from examples.deep_time import run as run_lib
from examples.deep_time import scenarios
from examples.deep_time.web import session as session_lib
from examples.deep_time.world import data

_WEB_DIR = os.path.dirname(__file__)
if not os.path.isfile(os.path.join(_WEB_DIR, 'index.html')):
  _WEB_DIR = os.path.join(
      os.path.dirname(os.path.dirname(os.path.dirname(_WEB_DIR))),
      'opensource',
      'examples',
      'deep_time',
      'web',
  )
PAGE = os.path.join(_WEB_DIR, 'index.html')
# The page's script modules, in dependency order. They are ES
# modules (so `node --test` can test them), inlined into the one page.
SCRIPTS = os.path.join(_WEB_DIR, 'js')
MODULES = (
    'util.js',
    'motion.js',
    'palette.js',
    'stars.js',
    'instruments.js',
    'pushes.js',
    'life.js',
    'minds.js',
    'story.js',
    'omens.js',
    'campaign.js',
    'forecast.js',
    'plans.js',
    'crises.js',
    'cards.js',
    'feel.js',
    'replays.js',
    'panels.js',
    'globe.js',
)
MODULES_MARK = '/*@modules*/'
# Timed live games needed before a ledger's speed replaces the measured one.
MIN_TIMED = 2
_IMPORT = re.compile(r'^import [^\n]*;\n', re.M)
_EXPORT = re.compile(
    r'^export (?=(?:async |const |function |let |class ))', re.M
)
# Minds a server may offer besides the mock, and about what
# a full live game costs (shown before choosing).
LIVE_MINDS = ('together', 'ollama')
PER_GAME_USD = 0.5
TURN_SECONDS = 180.0  # A player's time to answer, when several share a game.


def build() -> dict[str, str]:
  """The code this server runs: its commit and when it was made, if known."""
  try:
    out = subprocess.run(
        ['git', 'log', '-1', '--format=%h|%cd', '--date=format:%d %b %H:%M'],
        cwd=_WEB_DIR,
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout.strip()
    commit, when = out.split('|', 1)
  except (OSError, subprocess.SubprocessError, ValueError):
    return {'commit': '', 'when': ''}
  return {'commit': commit, 'when': when}


# Read once: a deploy restarts the server.
BUILD = build()
BUILD_MARK = '/*@build*/null'


def page() -> bytes:
  """The page, with its script modules inlined (one self-contained file).

  Each module's `import` lines go (the modules share the page's script, in
  `MODULES` order) and its `export`s become plain declarations.

  Returns:
    The page's HTML.
  """
  with open(PAGE, encoding='utf-8') as f:
    page_html = f.read()
  code = []
  for name in MODULES:
    with open(os.path.join(SCRIPTS, name), encoding='utf-8') as f:
      code.append(_EXPORT.sub('', _IMPORT.sub('', f.read())))
  return (
      page_html.replace(MODULES_MARK, '\n'.join(code), 1)
      .replace(BUILD_MARK, json.dumps(BUILD), 1)
      .encode('utf-8')
  )


class Games:
  """The one game being played (a single-player server).

  Live minds are offered only if the server allows them
  (`live_minds`), and only while the spend ledger's total is under the
  budget: every game's spend is added to the ledger when it ends.
  """

  def __init__(
      self,
      telemetry_dir: str | None = None,
      live_minds: Sequence[str] = (),
      budget_usd: float = 0.0,
      ledger: str | None = None,
      progress: str | None = None,
  ):
    self._lock = threading.Lock()
    self.current: session_lib.GameSession | None = None
    # Where played games write how they went, if anywhere.
    self.telemetry_dir = telemetry_dir
    self.live_minds = tuple(m for m in live_minds if m in LIVE_MINDS)
    self.budget_usd = float(budget_usd)
    self.ledger = ledger
    # The best stars of played games: the campaign and daily challenge.
    self.progress = campaign_lib.Progress(progress)

  def campaign(self) -> dict[str, Any]:
    """The campaign's chapters and today's challenge, with best stars.

    With them, the newest save of a played game not running now,
    for the start screen to offer to play on.
    """
    found = self.progress.campaign(datetime.date.today())
    # Which code this is, and what changed for players lately.
    found['build'] = BUILD
    found['changes'] = data.load('changes.yaml')['changes'][:2]
    found['saved'] = self.saved()
    found['replays'] = (
        session_lib.list_replays(self.telemetry_dir)
        if self.telemetry_dir
        else []
    )
    return found

  def saved(self) -> dict[str, Any] | None:
    """The newest save to play on (without its checkpoint), or None."""
    if not self.telemetry_dir:
      return None
    save = session_lib.latest_save(self.telemetry_dir)
    with self._lock:
      game = self.current
    if save is None or (
        game is not None
        and game.game_id == save['game_id']
        and not game.finished
    ):
      return None
    keep = ('game_id', 'scenario', 'tick', 'max_ticks', 'players', 'minds')
    return {k: save.get(k) for k in keep} | {'saved_at': save.get('saved_at')}

  def rewind(self) -> str | None:
    """Takes the current game back a turn: None, or why not.

    The game is played on from the save before the previous turn, in a new
    session that carries the rewinds, the earlier saves and all the spend;
    the old session leaves its save, spend and telemetry to it.
    """
    with self._lock:
      game = self.current
    if game is None:
      return 'No game is running.'
    save, why = game.rewind_save()
    if save is None:
      return why
    try:
      again = session_lib.GameSession(
          game.scenario,
          'play',
          game.seed,
          players=list(game.seats),
          turn_seconds=game.turn_seconds,
          telemetry_dir=self.telemetry_dir,
          minds=game.minds,
          on_end=self._ended,
          resume=save,
      )
    except (KeyError, ValueError, TypeError) as error:
      return f'The game could not go back: {error}'
    game.superseded = True
    with self._lock:
      if self.current is not game:
        return 'Another game started.'
      self.current = again
    game.cancel()
    again.start()
    return None

  def replay(self, game_id: str) -> str | None:
    """Watches a decided game again, exactly."""
    if not self.telemetry_dir or os.sep in game_id or not game_id:
      return 'No such replay.'
    record = session_lib.load_replay(
        os.path.join(self.telemetry_dir, session_lib.REPLAYS, f'{game_id}.json')
    )
    if record is None:
      return 'No such replay (yet).'
    try:
      game = session_lib.GameSession(
          str(record['scenario']),
          'watch',
          int(record['seed']),
          players=list(record['players']),
          replay=record,
      )
    except (KeyError, ValueError, TypeError) as error:
      return f'The game could not be replayed: {error}'
    with self._lock:
      old, self.current = self.current, game
    if old is not None:
      old.cancel()
    game.start()
    return None

  def resume(self, game_id: str) -> str | None:
    """Plays a saved game on: None, or why not."""
    if not self.telemetry_dir or os.sep in game_id or not game_id:
      return 'No such save.'
    save = session_lib.load_save(
        os.path.join(self.telemetry_dir, session_lib.SAVES, f'{game_id}.json')
    )
    if save is None:
      return 'No such save.'
    minds = str(save.get('minds', 'mock'))
    if minds not in self.minds()['options']:
      minds = 'mock'  # The live budget is spent: the mock plays on.
    try:
      game = session_lib.GameSession(
          str(save['scenario']),
          'play',
          int(save['seed']),
          players=list(save['players']),
          turn_seconds=save.get('turn_seconds'),
          telemetry_dir=self.telemetry_dir,
          minds=minds,
          on_end=self._ended,
          resume=save,
      )
    except (KeyError, ValueError, TypeError) as error:
      return f'The save could not be played on: {error}'
    with self._lock:
      old, self.current = self.current, game
    if old is not None:
      old.cancel()
    game.start()
    return None

  def _ended(self, game: session_lib.GameSession) -> None:
    """A game ended: log a live game's spend, keep a played game's stars."""
    self._log_spend(game)
    planet = game.context.planet
    if game.mode == 'play' and planet.outcome:
      today, stars = datetime.date.today(), planet.star_count()
      before = self.progress.record(game.scenario, game.seed, stars, today)
      # Its place in the campaign, for the recap: best stars, next chapter.
      game.chapter = self.progress.chapter(
          game.scenario, game.seed, stars, before, today
      )

  def spent(self) -> float:
    """Dollars the ledger says live games have cost so far."""
    if not self.ledger or not os.path.exists(self.ledger):
      return 0.0
    try:
      with open(self.ledger, encoding='utf-8') as f:
        return float(sum(row.get('usd', 0.0) for row in json.load(f)))
    except (OSError, ValueError, TypeError, AttributeError):
      return self.budget_usd  # An unreadable ledger spends nothing more.

  def minds(self) -> dict[str, Any]:
    """The minds a new game may have, and the live budget left."""
    left = max(0.0, self.budget_usd - self.spent())
    free = lambda m: not model_setup.USD_PER_MILLION_TOKENS.get(m, 1.0)
    return {
        # Priced minds while the budget lasts; free (local) ones always.
        'options': [
            'mock',
            *(m for m in self.live_minds if free(m) or left > 0.05),
        ],
        'left_usd': round(left, 2),
        'per_game_usd': PER_GAME_USD,
        # What each costs, how fast it plays and how often it wins.
        'measured': self.measured(),
        # And in each world: what the minds there cost and how they played.
        'by_scenario': self.measured_by_scenario(),
    }

  def _ledger_rows(self) -> list[dict[str, Any]]:
    """This server's spend ledger: one row per live game ([] if none)."""
    if not self.ledger or not os.path.exists(self.ledger):
      return []
    try:
      with open(self.ledger, encoding='utf-8') as f:
        return [r for r in json.load(f) if isinstance(r, dict)]
    except (OSError, ValueError, TypeError):
      return []

  @staticmethod
  def _measure(
      games: list[dict[str, Any]], seconds: float | None
  ) -> dict[str, Any]:
    """Cost a game, seconds a turn and wins over ledger rows."""
    timed = [r for r in games if r.get('seconds') and r.get('ticks')]
    if len(timed) < MIN_TIMED:  # Too few to say: the measured speed.
      timed = []
    return {
        'usd_per_game': round(
            sum(float(r.get('usd', 0)) for r in games) / len(games), 3
        ),
        'seconds_per_turn': (
            round(sum(r['seconds'] / r['ticks'] for r in timed) / len(timed), 1)
            if timed
            else seconds
        ),
        'wins': round(
            sum(r.get('outcome') == 'victory' for r in games) / len(games), 2
        ),
        'games': len(games),
        'note': 'games on this server',
    }

  def measured(self) -> dict[str, dict[str, Any]]:
    """Each kind of minds' cost, speed and wins: this ledger's, or the data's.

    A kind with three or more games in this server's ledger is measured
    from them; otherwise the programme's measurements (`data/minds.yaml`).
    """
    found = {k: dict(v) for k, v in data.load('minds.yaml')['measured'].items()}
    rows = self._ledger_rows()
    for kind in {str(r.get('minds')) for r in rows}:
      games = [r for r in rows if r.get('minds') == kind]
      if len(games) >= 3:
        found[kind] = self._measure(
            games, found.get(kind, {}).get('seconds_per_turn')
        )
    return found

  def measured_by_scenario(self) -> dict[str, dict[str, dict[str, Any]]]:
    """Each kind of minds measured in each world.

    A world's games in this server's ledger, where it has any; otherwise
    the programme's measurements of that world (`data/minds.yaml`). A
    speed not measured in a world is the kind's own.

    Returns:
      Scenario -> kind -> {usd_per_game, seconds_per_turn, wins, games}.
    """
    overall = self.measured()
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario, kinds in (
        data.load('minds.yaml').get('by_scenario') or {}
    ).items():
      found[scenario] = {}
      for kind, numbers in kinds.items():
        entry = dict(numbers)
        entry.setdefault(
            'usd_per_game', overall.get(kind, {}).get('usd_per_game')
        )
        entry.setdefault(
            'seconds_per_turn', overall.get(kind, {}).get('seconds_per_turn')
        )
        found[scenario][kind] = entry
    rows = self._ledger_rows()
    for scenario in {str(r.get('scenario')) for r in rows}:
      for kind in {str(r.get('minds')) for r in rows}:
        games = [
            r
            for r in rows
            if r.get('minds') == kind and r.get('scenario') == scenario
        ]
        if games:
          found.setdefault(scenario, {})[kind] = self._measure(
              games, overall.get(kind, {}).get('seconds_per_turn')
          )
    return found

  def _log_spend(self, game: session_lib.GameSession) -> None:
    """Adds a live game's spend to the ledger."""
    spend = game.spend()
    if not self.ledger or spend['minds'] == 'mock' or not spend['calls']:
      return
    with self._lock:
      rows = []
      if os.path.exists(self.ledger):
        try:
          with open(self.ledger, encoding='utf-8') as f:
            rows = list(json.load(f))
        except (OSError, ValueError, TypeError):
          rows = []
      rows.append({
          'when': time.strftime('%Y-%m-%d %H:%M'),
          'game_id': game.game_id,
          'scenario': game.scenario,
          'ticks': game.context.planet.tick,
          'outcome': game.context.planet.outcome or 'unfinished',
          'seconds': (
              round(time.monotonic() - game.started_at)
              if hasattr(game, 'started_at')
              else None
          ),
          **spend,
      })
      os.makedirs(os.path.dirname(os.path.abspath(self.ledger)), exist_ok=True)
      with open(self.ledger, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=1)

  def start(
      self,
      scenario: str,
      mode: str,
      seed: int,
      players: list[str] | None = None,
      minds: str = 'mock',
  ) -> str | None:
    """Starts a game: None, or why it was refused."""
    if minds not in self.minds()['options']:
      return f'minds {minds!r} are not available (or the live budget is spent)'
    # Several people share a turn: give each a few minutes to answer.
    turn = TURN_SECONDS if players and len(players) > 1 else None
    game = session_lib.GameSession(
        scenario,
        mode,
        seed,
        players=players,
        turn_seconds=turn,
        telemetry_dir=self.telemetry_dir,
        minds=minds,
        on_end=self._ended,
    )
    with self._lock:
      old, self.current = self.current, game
    if old is not None:
      old.cancel()
    game.start()
    return None

  def state(self, seat: str | None = None) -> dict[str, Any]:
    with self._lock:
      game = self.current
    return game.state(seat) if game else {'idle': True}

  def map(
      self, seat: str | None = None, full: bool = False, world: str = ''
  ) -> dict[str, Any]:
    with self._lock:
      game = self.current
    return game.map(seat, full, world) if game else {'idle': True}

  def choose(
      self,
      request_id: str,
      response: str,
      target: tuple[int, int] | None = None,
  ) -> str | None:
    """Answers the current game's request: None, or why it was refused."""
    with self._lock:
      game = self.current
    if game is None:
      return 'No game is running.'
    return game.choose(request_id, response, target)

  def hold_course(self, seat: str | None, turns: int) -> str | None:
    """A player holds course for some turns: None, or why it was refused."""
    with self._lock:
      game = self.current
    if game is None:
      return 'No game is running.'
    return game.hold_course(seat, turns)

  def look(self, seat: str | None, what: str) -> str | None:
    """A player looked at something on the page: None, or why not kept."""
    with self._lock:
      game = self.current
    if game is None:
      return 'No game is running.'
    return game.look(seat, what)

  def plan(
      self, seat: str | None, move: str, turns: int, cancel: int | None
  ) -> str | None:
    """A player plans a move ahead, or cancels a plan: None, or why not."""
    with self._lock:
      game = self.current
    if game is None:
      return 'No game is running.'
    if cancel is not None:
      return game.cancel_plan(seat, cancel)
    return game.plan(seat, move, turns)

  def set_speed(self, speed: str) -> str | None:
    """Watching: pause, play, fast or step; None if refused."""
    with self._lock:
      game = self.current
    if game is None or game.mode != 'watch':
      return None
    try:
      return game.set_speed(speed)
    except ValueError:
      return None

  def stop(self) -> None:
    with self._lock:
      game, self.current = self.current, None
    if game is not None:
      game.cancel()


def demo_index(demo_dir: str) -> str:
  """An HTML list of the run folders in `demo_dir` and their reports."""
  rows = []
  if os.path.isdir(demo_dir):
    for name in sorted(os.listdir(demo_dir)):
      folder = os.path.join(demo_dir, name)
      summary_path = os.path.join(folder, 'summary.json')
      if not os.path.isfile(summary_path):
        continue
      with open(summary_path, encoding='utf-8') as f:
        summary = json.load(f)
      links = ' · '.join(
          f'<a href="{urllib.parse.quote(name)}/{page}">{label}</a>'
          for page, label in (
              ('report.html', 'report'),
              ('gazette.html', 'gazette'),
              ('log.html', 'log'),
          )
          if os.path.isfile(os.path.join(folder, page))
      )
      rows.append(
          f'<li><b>{html.escape(name)}</b>:'
          f' {html.escape(str(summary.get("outcome", "")))}'
          f' — {html.escape(str(summary.get("outcome_reason", "")))}<br>{links}</li>'
      )
  body = (
      '<ul>' + ''.join(rows) + '</ul>' if rows else '<p>No demo runs yet.</p>'
  )
  return (
      '<!doctype html><meta charset="utf-8"><meta name="viewport"'
      ' content="width=device-width, initial-scale=1"><title>Deep Time'
      ' demos</title><style>body{font:16px'
      ' system-ui;margin:16px;line-height:1.5}li{margin:0 0'
      ' 12px}</style><h1>Deep Time demo runs</h1><p><a href="../">←'
      ' play</a></p>'
      + body
  )


def make_handler(games: Games, demo_dir: str, base_path: str = ''):
  """The request handler class, bound to a game registry and a demo folder."""
  viewports: list[dict[str, Any]] = []  # What phones reported, newest last.

  class Handler(http.server.BaseHTTPRequestHandler):
    """Serves the page, the game API and demo reports."""

    def log_message(self, format: str, *args: Any) -> None:  # pylint: disable=redefined-builtin
      del format, args  # Quiet: the page polls every second.

    def _local(self, path: str) -> str:
      """The path without the published prefix, if it still carries it."""
      base = base_path.rstrip('/')
      if base and (path == base or path.startswith(base + '/')):
        return path[len(base) :] or '/'
      return path

    def _send(self, status: int, body: bytes, kind: str) -> None:
      # Phones on the tailnet pay for every byte: compress what is worth it
      # (a tick's map shrinks about eightfold).
      zipped = len(body) > 1024 and 'gzip' in self.headers.get(
          'Accept-Encoding', ''
      )
      if zipped:
        body = gzip.compress(body, compresslevel=6)
      self.send_response(status)
      self.send_header('Content-Type', kind)
      if zipped:
        self.send_header('Content-Encoding', 'gzip')
        self.send_header('Vary', 'Accept-Encoding')
      self.send_header('Content-Length', str(len(body)))
      self.send_header('Cache-Control', 'no-store')
      self.end_headers()
      self.wfile.write(body)

    def _json(self, value: Any, status: int = 200) -> None:
      self._send(status, json.dumps(value).encode(), 'application/json')

    def do_GET(self) -> None:  # pylint: disable=invalid-name
      path = self._local(urllib.parse.urlparse(self.path).path)
      if path in ('/', '/index.html'):
        self._send(200, page(), 'text/html; charset=utf-8')
      elif path == '/api/state':
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        # With the build, so a page left open knows when it is out of date.
        self._json({
            **games.state((query.get('seat') or [None])[0]),
            'build': BUILD['commit'],
        })
      elif path == '/api/map':
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        self._json(
            games.map(
                (query.get('seat') or [None])[0],
                full=(query.get('full') or ['0'])[0] == '1',
                world=(query.get('world') or [''])[0],
            )
        )
      elif path == '/api/viewport':
        self._json(viewports[-1] if viewports else {})
      elif path == '/api/scenarios':
        self._json(session_lib.scenario_list())
      elif path == '/api/minds':
        self._json(games.minds())
      elif path == '/api/thumb':
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        key = (query.get('scenario') or [''])[0]
        if key not in scenarios.SCENARIOS:
          self._json({'error': 'unknown scenario'}, 404)
        else:
          self._json(thumb(key))
      elif path == '/api/campaign':
        self._json(games.campaign())
      elif path in ('/demo', '/demo/'):
        self._send(
            200, demo_index(demo_dir).encode(), 'text/html; charset=utf-8'
        )
      elif path.startswith('/demo/'):
        self._file(urllib.parse.unquote(path[len('/demo/') :]))
      else:
        self._send(404, b'Not found', 'text/plain')

    def _file(self, relative: str) -> None:
      root = os.path.realpath(demo_dir)
      try:
        target = os.path.realpath(os.path.join(root, relative))
      except ValueError:  # e.g. a NUL byte in the path.
        target = ''
      if not target.startswith(root + os.sep) or not os.path.isfile(target):
        self._send(404, b'Not found', 'text/plain')
        return
      kind = (
          'text/html; charset=utf-8'
          if target.endswith('.html')
          else (
              'application/json' if target.endswith('.json') else 'text/plain'
          )
      )
      with open(target, 'rb') as f:
        self._send(200, f.read(), kind)

    def do_POST(self) -> None:  # pylint: disable=invalid-name
      length = int(self.headers.get('Content-Length', 0) or 0)
      try:
        body = json.loads(self.rfile.read(length) or b'{}')
      except json.JSONDecodeError:
        body = None
      if not isinstance(body, dict):
        self._json({'error': 'expected a JSON object'}, 400)
        return
      path = self._local(urllib.parse.urlparse(self.path).path)
      if path == '/api/replay':
        error = games.replay(str(body.get('game_id', '')))
        self._json({'ok': error is None, 'error': error or ''})
        return
      if path == '/api/rewind':
        error = games.rewind()
        self._json({'ok': error is None, 'error': error or ''})
        return
      if path == '/api/resume':
        error = games.resume(str(body.get('game_id', '')))
        self._json({'ok': error is None, 'error': error or ''})
        return
      if path == '/api/new':
        scenario = str(body.get('scenario', 'daisyworld'))
        mode = str(body.get('mode', 'play'))
        if scenario not in scenarios.SCENARIOS or mode not in ('play', 'watch'):
          self._json({'error': 'unknown scenario or mode'}, 400)
          return
        # One person (`role`) or several (`players`), one seat each.
        players = body.get('players') or (
            [body['role']] if body.get('role') else None
        )
        if players is not None and (
            not isinstance(players, list)
            or len(set(map(str, players))) != len(players)
            or any(
                str(p) not in dict(session_lib.roles(scenario)) for p in players
            )
        ):
          self._json({'error': 'unknown or repeated role'}, 400)
          return
        try:
          seed = int(body.get('seed', 0))
        except (TypeError, ValueError):
          self._json({'error': 'seed must be an integer'}, 400)
          return
        refused = games.start(
            scenario,
            mode,
            seed,
            [str(p) for p in players] if players else None,
            minds=str(body.get('minds', 'mock')),
        )
        if refused:
          self._json({'error': refused}, 400)
          return
        self._json({'ok': True})
      elif path == '/api/viewport':
        # The phone reports its screen, so its true size can be designed for.
        # It also reports how fast it draws the map: the first draw after
        # loading and the slowest frame of a pan, in milliseconds.
        report = {
            k: body.get(k)
            for k in (
                'w',
                'h',
                'dpr',
                'ua',
                'first_draw_ms',
                'draw_ms',
                'pan_ms',
            )
            if isinstance(body.get(k), (int, float, str))
        }
        viewports.append(report)
        del viewports[:-20]
        print(f'viewport {json.dumps(report)}', flush=True)
        self._json({'ok': True})
      elif path == '/api/hold':
        try:
          turns = int(body.get('turns', 0))
        except (TypeError, ValueError, OverflowError):
          turns = 0
        seat = body.get('seat')
        error = games.hold_course(str(seat) if seat else None, turns)
        self._json({'ok': error is None, 'error': error or ''})
      elif path == '/api/look':
        # What the player looked at this turn, for play telemetry.
        seat = body.get('seat')
        error = games.look(
            str(seat) if seat else None, str(body.get('what', ''))
        )
        self._json({'ok': error is None, 'error': error or ''})
      elif path == '/api/plan':
        try:
          turns = int(body.get('turns', 0))
          cancel = (
              int(body['cancel']) if body.get('cancel') is not None else None
          )
        except (TypeError, ValueError, OverflowError):
          turns, cancel = 0, None
        seat = body.get('seat')
        error = games.plan(
            str(seat) if seat else None,
            str(body.get('move', '')),
            turns,
            cancel,
        )
        self._json({'ok': error is None, 'error': error or ''})
      elif path == '/api/speed':
        speed = games.set_speed(str(body.get('speed', '')))
        if speed is None:
          self._json({'error': 'no game being watched, or no such speed'}, 400)
        else:
          self._json({'ok': True, 'speed': speed})
      elif path == '/api/choose':
        target = body.get('target')
        where = None
        if (
            isinstance(target, list)
            and len(target) == 2
            and all(isinstance(v, int) for v in target)
        ):
          where = (int(target[0]), int(target[1]))
        error = games.choose(
            str(body.get('request_id', '')),
            str(body.get('response', '')),
            where,
        )
        self._json({'ok': error is None, 'error': error or ''})
      else:
        self._json({'error': 'not found'}, 404)

  return Handler


def serve(
    port: int,
    demo_dir: str,
    host: str = '127.0.0.1',
    games: Games | None = None,
    base_path: str = '',
) -> http.server.ThreadingHTTPServer:
  """Starts the server (call `serve_forever` on it).

  Args:
    port: Port to listen on.
    demo_dir: Folder of run folders for `/demo/`.
    host: Interface to listen on (loopback by default).
    games: The game registry (a new one by default).
    base_path: A path the page is published under (e.g. '/deep-time'); a request
      still carrying it is routed as if it did not (a proxy may or may not strip
      it).

  Returns:
    The server.
  """
  return http.server.ThreadingHTTPServer(
      (host, port), make_handler(games or Games(), demo_dir, base_path)
  )


def make_demos(demo_dir: str, seed: int = 0) -> None:
  """Writes one MOCK run per scenario into `demo_dir` (skips existing ones)."""
  for scenario in sorted(scenarios.SCENARIOS):
    out = os.path.join(demo_dir, scenario)
    if os.path.isfile(os.path.join(out, 'summary.json')):
      continue
    run_lib.run(scenario, 'mock', out_dir=out, seed=seed)


_THUMBS: dict[str, dict[str, Any]] = {}
_THUMBS_LOCK = threading.Lock()


def thumb(scenario: str) -> dict[str, Any]:
  """A scenario's world at its start, for its card.

  The biome of every tile (codes, north first) and the biomes' colours, as
  the map draws them; worked out once per scenario.
  """
  with _THUMBS_LOCK:
    found = _THUMBS.get(scenario)
  if found is None:
    planet = game_lib.new_game(scenario, seed=0).planet
    found = {'rows': planet.biomes.rows(), 'legend': planet.biomes.legend()}
    with _THUMBS_LOCK:
      _THUMBS[scenario] = found
  return found


def telemetry_folder(demo_dir: str | None, flag: str | None) -> str | None:
  """Where played games are written: the flag, else beside the run folders.

  Never inside `demo_dir`, which the server publishes under /demo/.

  Args:
    demo_dir: The folder of run folders.
    flag: `--telemetry_dir`: None for the default, '' for none.

  Returns:
    The folder, or None to write no telemetry.
  """
  if flag is not None:
    return flag or None
  if not demo_dir:
    return None
  return os.path.normpath(demo_dir) + '-telemetry'


def main(argv: Sequence[str]) -> None:
  del argv
  demo_dir = flags.FLAGS.demo_dir
  if flags.FLAGS.make_demos:
    make_demos(demo_dir)
  telemetry = telemetry_folder(demo_dir, flags.FLAGS.telemetry_dir)
  games = Games(
      telemetry_dir=telemetry,
      live_minds=flags.FLAGS.live_minds,
      budget_usd=flags.FLAGS.live_budget_usd,
      # Live games' spend, beside the telemetry (outside the demo folder).
      ledger=os.path.join(telemetry, 'spend.json') if telemetry else None,
      progress=os.path.join(telemetry, 'progress.json') if telemetry else None,
  )
  server = serve(
      flags.FLAGS.port, demo_dir, games=games, base_path=flags.FLAGS.base_path
  )
  print(f'Deep Time on http://127.0.0.1:{flags.FLAGS.port}/', flush=True)
  try:
    server.serve_forever()
  finally:
    games.stop()


if __name__ == '__main__':
  flags.DEFINE_integer('port', 8890, 'Loopback port to serve on.')
  flags.DEFINE_string(
      'demo_dir', '/tmp/deep_time_demos', 'Folder of run folders.'
  )
  flags.DEFINE_bool('make_demos', False, 'Write a MOCK run per scenario first.')
  flags.DEFINE_list(
      'live_minds',
      [],
      'Live minds a player may choose besides the mock: together, ollama.',
  )
  flags.DEFINE_float(
      'live_budget_usd',
      0.0,
      'Dollars live games may spend in all (the ledger, spend.json).',
  )
  flags.DEFINE_string(
      'telemetry_dir',
      None,
      'Where played games write how they went (default: <demo_dir>-telemetry;'
      " '' for none).",
  )
  flags.DEFINE_string(
      'base_path', '', 'Path the page is published under (e.g. /deep-time).'
  )
  app.run(main)
