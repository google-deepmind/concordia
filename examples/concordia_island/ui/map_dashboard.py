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

r"""Concordia map dashboard (Hermetic Third-Party Server & Mountable API).

A multiscale, terrain-rendered map of a running (or finished) Concordia
simulation, showing where every agent is and how they move over time.

Can be run standalone or mounted directly inside `simulation_studio.py` via
`init_source()`, `handle_api()`, and `STATIC_DIR`.

Usage:
  python -m examples.concordia_island.ui.map_dashboard \
    --run_dir=examples/concordia_island/data/paper_runs/job_loss/gemini_2_5_flash_esa \
    --geography=concordia_island \
    --port=8080
"""

from collections.abc import Sequence
import http.server
import json
import mimetypes
import os
import posixpath
import socketserver
import sys
import threading
from typing import Any
import urllib.parse

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if _REPO_ROOT not in sys.path:
  sys.path.append(_REPO_ROOT)
import concordia  # pylint: disable=g-import-not-at-top
import concordia.contrib  # pylint: disable=g-import-not-at-top
_OPEN_CONCORDIA = os.path.join(_REPO_ROOT, 'concordia')
if os.path.isdir(_OPEN_CONCORDIA) and _OPEN_CONCORDIA not in concordia.__path__:
  concordia.__path__.append(_OPEN_CONCORDIA)
_OPEN_CONTRIB = os.path.join(_OPEN_CONCORDIA, 'contrib')
if (
    os.path.isdir(_OPEN_CONTRIB)
    and _OPEN_CONTRIB not in concordia.contrib.__path__
):
  concordia.contrib.__path__.append(_OPEN_CONTRIB)

# pylint: disable=g-import-not-at-top,g-bad-import-order
from absl import app
from absl import flags
from absl import logging
from examples.concordia_island.ui import atlas_loader
from examples.concordia_island.ui import map_data
# pylint: enable=g-import-not-at-top,g-bad-import-order

FLAGS = flags.FLAGS

if 'id' not in flags.FLAGS:
  _RUN_ID = flags.DEFINE_string(
      'id', '', 'Run ID to monitor (looked up under ./local_runs/).'
  )
else:
  _RUN_ID = flags.FLAGS['id']

if 'run_dir' not in flags.FLAGS:
  _RUN_DIR = flags.DEFINE_string(
      'run_dir',
      '',
      'Path to local simulation run directory (defaults to the bundled'
      ' data/paper_runs/job_loss/gemini_2_5_flash_esa run).',
  )
else:
  _RUN_DIR = flags.FLAGS['run_dir']

if 'port' not in flags.FLAGS:
  _PORT = flags.DEFINE_integer('port', 8080, 'HTTP port.')
else:
  _PORT = flags.FLAGS['port']

if 'geography' not in flags.FLAGS:
  _GEOGRAPHY = flags.DEFINE_string(
      'geography',
      '',
      'Atlas id to render (concordia_island, brecksville, kerala). Empty picks'
      ' one from --setting, --personas_dir or the run directory.',
  )
else:
  _GEOGRAPHY = flags.FLAGS['geography']

if 'setting' not in flags.FLAGS:
  _SETTING = flags.DEFINE_string(
      'setting',
      '',
      'sim/locations.py setting preset name, used to pick a geography.',
  )
else:
  _SETTING = flags.FLAGS['setting']

if 'personas_dir' not in flags.FLAGS:
  _PERSONAS_DIR = flags.DEFINE_string(
      'personas_dir',
      '',
      'Personas path; used as a fallback signal for geography selection.',
  )
else:
  _PERSONAS_DIR = flags.FLAGS['personas_dir']

if 'expected_agents' not in flags.FLAGS:
  _EXPECTED_AGENTS = flags.DEFINE_integer(
      'expected_agents', 100, 'Expected number of agents.'
  )
else:
  _EXPECTED_AGENTS = flags.FLAGS['expected_agents']

if 'expected_ticks' not in flags.FLAGS:
  _EXPECTED_TICKS = flags.DEFINE_integer(
      'expected_ticks', 100, 'Expected number of ticks.'
  )
else:
  _EXPECTED_TICKS = flags.FLAGS['expected_ticks']

if 'start_time' not in flags.FLAGS:
  _START_TIME = flags.DEFINE_string(
      'start_time',
      'Thursday, January 1st, 7:00 AM',
      'Simulation time at tick 1. ISO 8601 also accepted.',
  )
else:
  _START_TIME = flags.FLAGS['start_time']

if 'tick_interval' not in flags.FLAGS:
  _TICK_INTERVAL = flags.DEFINE_integer(
      'tick_interval', 120, 'Minutes of simulated time per tick.'
  )
else:
  _TICK_INTERVAL = flags.FLAGS['tick_interval']

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

_source: map_data.LocalFileSource | None = None
_geography: str = map_data.DEFAULT_RUN_GEOGRAPHY


def pick_geography(
    geography: str = '',
    run_dir: str = '',
    run_id: str = '',
    setting: str = '',
    personas_dir: str = '',
) -> str:
  """Returns the atlas id to draw a run on when --geography is not given."""
  if geography:
    return geography
  if setting or personas_dir:
    return atlas_loader.resolve_atlas_id(setting or None, personas_dir)
  path = (run_dir or run_id).lower()
  if not path or 'paper_runs' in path:
    # The bundled paper runs (the default run) use concordia_island places.
    return map_data.DEFAULT_RUN_GEOGRAPHY
  if 'kerala' in path:
    return atlas_loader.resolve_atlas_id(None, path)
  # run.py's default --personas_date is the Brecksville population.
  return 'brecksville'


def init_source(
    run_dir: str = '',
    run_id: str = '',
    geography: str = '',
    setting: str = '',
    personas_dir: str = '',
    expected_agents: int = 100,
    expected_ticks: int = 100,
) -> map_data.LocalFileSource:
  """Initializes the module-level data source and geography for API handlers."""
  global _source, _geography
  _geography = pick_geography(
      geography, run_dir, run_id, setting, personas_dir
  )
  _source = map_data.LocalFileSource(
      run_id=run_id,
      run_dir=run_dir,
      expected_agents=expected_agents,
      expected_ticks=expected_ticks,
  )
  return _source


def get_source() -> map_data.LocalFileSource:
  """Returns the active data source, initializing with defaults if needed."""
  if _source is None:
    init_source(
        run_dir=_RUN_DIR.value,
        run_id=_RUN_ID.value,
        geography=_GEOGRAPHY.value,
        setting=_SETTING.value,
        personas_dir=_PERSONAS_DIR.value,
        expected_agents=_EXPECTED_AGENTS.value,
        expected_ticks=_EXPECTED_TICKS.value,
    )
  return _source


def api_config() -> dict[str, Any]:
  """Static configuration the client needs once at boot."""
  start = map_data.parse_start_time(_START_TIME.value)
  geos = []
  for gid in atlas_loader.available_atlases():
    try:
      geos.append({
          'id': gid,
          'display_name': atlas_loader.load_atlas(gid).meta['display_name'],
      })
    except atlas_loader.AtlasError as e:
      logging.error('Skipping atlas %s: %s', gid, e)
      geos.append({'id': gid, 'display_name': f'{gid} (failed to load)'})
  src = get_source()
  return {
      'run_id': getattr(src, 'run_id', _RUN_ID.value),
      'geography': _geography,
      'geographies': geos,
      'expected_agents': getattr(
          src, 'expected_agents', _EXPECTED_AGENTS.value
      ),
      'expected_ticks': getattr(src, 'expected_ticks', _EXPECTED_TICKS.value),
      'sim_start_iso': start.isoformat(),
      'tick_interval_min': _TICK_INTERVAL.value,
      'run_dir': getattr(src, 'run_dir', _RUN_DIR.value),
  }


def handle_api(
    path: str, query: dict[str, list[str]]
) -> tuple[int, dict[str, Any]]:
  """Routes one API path. Returns (http_status, json_body)."""
  src = get_source()
  if path == '/api/config':
    return 200, api_config()

  if path == '/api/atlas':
    gid = (query.get('g') or [_geography])[0]
    try:
      return 200, atlas_loader.load_atlas(gid).to_json_dict()
    except atlas_loader.AtlasError as e:
      return 400, {'error': str(e)}

  if path == '/api/data':
    return 200, src.data()

  if path == '/api/location_history':
    return 200, src.location_history()

  return 404, {'error': f'Unknown API path {path!r}'}


def _resolve_static(url_path: str) -> str | None:
  """Maps a /static/... URL onto a file, refusing any path escape."""
  rel = posixpath.normpath(url_path[len('/static/'):]).lstrip('/')
  if not rel or rel.startswith('..'):
    return None
  full = os.path.normpath(os.path.join(STATIC_DIR, rel))
  if not full.startswith(os.path.normpath(STATIC_DIR) + os.sep):
    return None
  return full if os.path.isfile(full) else None


class _Handler(http.server.BaseHTTPRequestHandler):
  """Serves the single-page map app and its JSON API."""

  protocol_version = 'HTTP/1.1'

  def log_message(self, fmt, *args):
    logging.info('%s - %s', self.address_string(), fmt % args)

  def _send(self, status, body, ctype, extra_headers=None):
    self.send_response(status)
    self.send_header('Content-Type', ctype)
    self.send_header('Content-Length', str(len(body)))
    for k, v in (extra_headers or {}).items():
      self.send_header(k, v)
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):  # pylint: disable=invalid-name
    """Serves static map assets and JSON API endpoints."""
    parsed = urllib.parse.urlparse(self.path)
    path = parsed.path
    query = urllib.parse.parse_qs(parsed.query)

    try:
      if path in ('/', '/index.html', '/map', '/map/index.html'):
        index = os.path.join(STATIC_DIR, 'index.html')
        if not os.path.isfile(index):
          self._send(
              500,
              b'static/index.html is missing.',
              'text/plain; charset=utf-8',
          )
          return
        with open(index, 'rb') as f:
          self._send(
              200,
              f.read(),
              'text/html; charset=utf-8',
              {'Cache-Control': 'no-cache'},
          )
        return

      if path.startswith('/static/'):
        full = _resolve_static(path)
        if not full:
          self._send(404, b'Not found', 'text/plain; charset=utf-8')
          return
        ctype, _ = mimetypes.guess_type(full)
        if full.endswith('.js'):
          ctype = 'text/javascript; charset=utf-8'
        elif full.endswith('.css'):
          ctype = 'text/css; charset=utf-8'
        with open(full, 'rb') as f:
          self._send(
              200,
              f.read(),
              ctype or 'application/octet-stream',
              {'Cache-Control': 'no-cache'},
          )
        return

      if path.startswith('/api/'):
        status, body = handle_api(path, query)
        payload = json.dumps(body, default=str).encode('utf-8')
        self._send(
            status,
            payload,
            'application/json; charset=utf-8',
            {'Cache-Control': 'no-store'},
        )
        return

      self._send(404, b'Not found', 'text/plain; charset=utf-8')
    except BrokenPipeError:
      pass
    except Exception:  # pylint: disable=broad-except
      import traceback  # pylint: disable=g-import-not-at-top
      detail = traceback.format_exc()
      logging.error('Unhandled error serving %s:\n%s', self.path, detail)
      try:
        self._send(
            500,
            json.dumps({'error': detail}).encode('utf-8'),
            'application/json; charset=utf-8',
        )
      except Exception:  # pylint: disable=broad-except
        pass


class _ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
  daemon_threads = True
  allow_reuse_address = True


def main(argv: Sequence[str]) -> None:
  del argv
  src = init_source(
      run_dir=_RUN_DIR.value,
      run_id=_RUN_ID.value,
      geography=_GEOGRAPHY.value,
      setting=_SETTING.value,
      personas_dir=_PERSONAS_DIR.value,
      expected_agents=_EXPECTED_AGENTS.value,
      expected_ticks=_EXPECTED_TICKS.value,
  )
  threading.Thread(target=src.data, daemon=True).start()
  threading.Thread(target=src.location_history, daemon=True).start()

  server = _ThreadingServer(('', _PORT.value), _Handler)
  logging.info('Map dashboard running on http://localhost:%d', _PORT.value)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    logging.info('Shutting down.')
    server.shutdown()


if __name__ == '__main__':
  app.run(main)
