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

"""Tests for the Deep Time phone server."""

import gzip
import json
import os
import threading
import time
import typing
from typing import Any
import urllib.error
import urllib.request

from absl.testing import absltest
from examples.deep_time.web import server as server_lib
from examples.deep_time.world import data


class ServerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.demo_dir = self.create_tempdir().full_path
    run = os.path.join(self.demo_dir, 'daisy run')
    os.makedirs(run)
    with open(os.path.join(run, 'summary.json'), 'w', encoding='utf-8') as f:
      json.dump({'outcome': 'victory', 'outcome_reason': 'held <35 C'}, f)
    with open(os.path.join(run, 'report.html'), 'w', encoding='utf-8') as f:
      f.write('<p>report</p>')
    with open(
        os.path.join(os.path.dirname(self.demo_dir), 'secret.txt'),
        'w',
        encoding='utf-8',
    ) as f:
      f.write('secret')
    self.games = server_lib.Games()
    self.server = server_lib.serve(0, self.demo_dir, games=self.games)
    self.base = f'http://127.0.0.1:{self.server.server_address[1]}'
    thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    thread.start()
    self.addCleanup(thread.join, 10)
    self.addCleanup(self.server.server_close)
    self.addCleanup(self.server.shutdown)
    self.addCleanup(self.games.stop)

  def _get(self, path: str) -> tuple[int, bytes]:
    try:
      with urllib.request.urlopen(self.base + path, timeout=30) as response:
        return response.status, response.read()
    except urllib.error.HTTPError as error:
      return error.code, error.read()

  def _post(self, path: str, body: object) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        self.base + path,
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    try:
      with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
      return error.code, json.loads(error.read())

  def test_serves_the_page_and_scenarios(self):
    status, page = self._get('/')
    self.assertEqual(status, 200)
    self.assertIn(b'Deep Time', page)
    status, body = self._get('/api/scenarios')
    self.assertEqual(status, 200)
    self.assertIn('daisyworld', [s['key'] for s in json.loads(body)])
    self.assertEqual(
        json.loads(self._get('/api/state')[1]),
        {'idle': True, 'build': server_lib.BUILD['commit']},
    )
    self.assertEqual(self._get('/nowhere')[0], 404)

  def test_the_page_says_which_build_it_is_and_what_is_new(self):
    _, page = self._get('/')
    self.assertNotIn(server_lib.BUILD_MARK.encode(), page)
    self.assertIn(json.dumps(server_lib.BUILD).encode(), page)
    campaign = json.loads(self._get('/api/campaign')[1])
    self.assertEqual(campaign['build'], server_lib.BUILD)
    self.assertLen(campaign['changes'], 2)  # The newest, for the card.
    self.assertEqual(campaign['changes'][0]['world'], 'great_acceleration')
    newest = data.load('changes.yaml')['changes'][0]
    self.assertEqual(campaign['changes'][0]['when'], newest['when'])

  def test_each_world_has_a_thumbnail_for_its_card(self):
    status, body = self._get('/api/thumb?scenario=archipelago')
    self.assertEqual(status, 200)
    thumb = json.loads(body)
    self.assertLen(thumb['rows'], 30)  # North first, a code a tile.
    self.assertTrue(all(len(row) == 40 for row in thumb['rows']))
    codes = {b['code'] for b in thumb['legend']}
    self.assertContainsSubset(set(''.join(thumb['rows'])), codes)
    self.assertEqual(self._get('/api/thumb?scenario=nowhere')[0], 404)

  def test_the_page_is_self_contained_and_light(self):
    _, page = self._get('/')
    self.assertLess(len(page), 500_000)  # No page over 500 KB.
    self.assertNotRegex(page, rb'<script[^>]+src=')  # No external scripts.
    self.assertNotRegex(page, rb'<link[^>]+href=["\']?https?:')
    self.assertNotIn(b'fetch("/', page)  # Relative URLs only (tailnet path).
    self.assertNotIn(b"fetch('/", page)

  def test_time_controls_only_for_a_watched_game(self):
    self.assertEqual(self._post('/api/speed', {'speed': 'pause'})[0], 400)
    self._post('/api/new', {'scenario': 'daisyworld', 'mode': 'watch'})
    self.assertEqual(
        self._post('/api/speed', {'speed': 'pause'}),
        (200, {'ok': True, 'speed': 'pause'}),
    )
    self.assertEqual(self._post('/api/speed', {'speed': 'warp'})[0], 400)
    self._post('/api/new', {'scenario': 'daisyworld', 'mode': 'play'})
    self.assertEqual(self._post('/api/speed', {'speed': 'fast'})[0], 400)

  def test_compresses_for_clients_that_accept_it(self):
    request = urllib.request.Request(
        self.base + '/', headers={'Accept-Encoding': 'gzip'}
    )
    with urllib.request.urlopen(request) as response:
      self.assertEqual(response.headers['Content-Encoding'], 'gzip')
      self.assertIn(b'Deep Time', gzip.decompress(response.read()))
    _, plain = self._get('/')  # Clients that don't ask get it plain.
    self.assertIn(b'Deep Time', plain)

  def test_starts_a_game_and_takes_a_choice(self):
    status, _ = self._post('/api/new', {'scenario': 'atlantis'})
    self.assertEqual(status, 400)
    status, _ = self._post(
        '/api/new', {'scenario': 'daisyworld', 'role': 'Amber folk'}
    )
    self.assertEqual(status, 400)
    status, body = self._post('/api/new', {'scenario': 'daisyworld'})
    self.assertEqual((status, body), (200, {'ok': True}))
    game = self.games.current
    assert game is not None
    state = {}
    for _ in range(600):
      state = json.loads(self._get('/api/state')[1])
      if state['question']:
        break
      threading.Event().wait(0.05)
    self.assertTrue(state['question'])
    self.assertEqual(state['mode'], 'play')
    request_id = state['question']['request_id']
    response = state['question']['options'][0]
    answer = {'request_id': request_id, 'response': response}
    # What the player looks at before choosing is kept with the turn.
    self.assertEqual(
        self._post('/api/look', {'what': 'sheet'}),
        (200, {'ok': True, 'error': ''}),
    )
    self.assertEqual(
        self._post('/api/choose', answer), (200, {'ok': True, 'error': ''})
    )
    self.assertEqual(game.seats[game.entity].played[0]['looked'], ['sheet'])
    status, body = self._post('/api/choose', answer)  # Already answered.
    self.assertEqual(status, 200)
    self.assertFalse(body['ok'])
    self.assertIn('already been answered', str(body['error']))
    # A new game replaces (and cancels) the old one.
    self._post('/api/new', {'scenario': 'daisyworld', 'mode': 'watch'})
    game.join(30)
    self.assertTrue(game.finished)
    self.assertIsNot(self.games.current, game)
    self.assertFalse(self._post('/api/look', {'what': 'sheet'})[1]['ok'])

  def test_serves_demo_reports_but_nothing_outside_them(self):
    status, index = self._get('/demo/')
    self.assertEqual(status, 200)
    self.assertIn(b'daisy run', index)
    self.assertIn(b'held &lt;35 C', index)
    self.assertIn(b'daisy%20run/report.html', index)
    self.assertEqual(
        self._get('/demo/daisy%20run/report.html'), (200, b'<p>report</p>')
    )
    self.assertEqual(self._get('/demo/daisy%20run/log.html')[0], 404)
    self.assertEqual(self._get('/demo/../secret.txt')[0], 404)
    self.assertEqual(self._get('/demo/%2E%2E/secret.txt')[0], 404)
    self.assertEqual(self._get('/demo/a%00b')[0], 404)

  def test_a_game_for_two_and_each_seat_its_own_view(self):
    players = ['Amber folk', 'Basalt clans']
    status, _ = self._post(
        '/api/new', {'scenario': 'great_acceleration', 'players': players}
    )
    self.assertEqual(status, 200)
    for seat in players:
      state = json.loads(
          self._get('/api/state?seat=' + seat.replace(' ', '%20'))[1]
      )
      self.assertEqual(state['seat'], seat)
    status, _ = self._post(
        '/api/new',
        {'scenario': 'great_acceleration', 'players': ['Amber folk'] * 2},
    )
    self.assertEqual(status, 400)

  def test_works_under_a_published_path_with_or_without_the_prefix(self):
    server = server_lib.serve(0, self.demo_dir, base_path='/deep-time')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    self.addCleanup(thread.join, 10)
    self.addCleanup(server.server_close)
    self.addCleanup(server.shutdown)
    base = f'http://127.0.0.1:{server.server_address[1]}'
    for path in (
        '/deep-time/',
        '/',
        '/deep-time/api/scenarios',
        '/api/scenarios',
    ):
      with urllib.request.urlopen(base + path, timeout=30) as response:
        self.assertEqual(response.status, 200, path)
    # The page asks relative to itself, so it works under any prefix.
    _, page = self._get('/')
    self.assertNotIn(b"api('/api", page)
    self.assertIn(b"api('api/state'", page)
    self.assertIn(b'href="demo/"', page)

  def test_the_map_and_the_phones_viewport(self):
    self.assertEqual(json.loads(self._get('/api/map')[1]), {'idle': True})
    self._post('/api/new', {'scenario': 'daisyworld', 'mode': 'watch'})
    world = json.loads(self._get('/api/map')[1])
    self.assertLen(world['regions'], 48)
    self.assertEqual(json.loads(self._get('/api/viewport')[1]), {})
    report = {'w': 412, 'h': 915, 'dpr': 2.625, 'ua': 'Pixel', 'x': [1]}
    self.assertEqual(self._post('/api/viewport', report), (200, {'ok': True}))
    self.assertEqual(
        json.loads(self._get('/api/viewport')[1]),
        {'w': 412, 'h': 915, 'dpr': 2.625, 'ua': 'Pixel'},
    )
    timed = dict(report, first_draw_ms=640, draw_ms=31, pan_ms=7.5)
    self._post('/api/viewport', timed)  # The phone also times its map.
    latest = json.loads(self._get('/api/viewport')[1])
    self.assertEqual(
        (latest['first_draw_ms'], latest['draw_ms'], latest['pan_ms']),
        (640, 31, 7.5),
    )

  def test_malformed_requests_get_an_answer(self):
    self.assertEqual(self._post('/api/new', [1, 2])[0], 400)
    status, body = self._post(
        '/api/new', {'scenario': 'daisyworld', 'seed': 'many'}
    )
    self.assertEqual(status, 400)
    self.assertIn('seed', str(body['error']))


class LiveMindsTest(absltest.TestCase):

  def test_live_minds_are_offered_only_while_the_budget_lasts(self):
    ledger = os.path.join(self.create_tempdir().full_path, 'spend.json')
    games = server_lib.Games(
        live_minds=('together', 'ollama', 'bogus'),
        budget_usd=1.0,
        ledger=ledger,
    )
    offer = games.minds()
    self.assertEqual(offer['options'], ['mock', 'together', 'ollama'])
    self.assertEqual(offer['left_usd'], 1.0)
    with open(ledger, 'w') as f:
      json.dump([{'usd': 0.6}, {'usd': 0.4}], f)
    self.assertEqual(games.minds()['options'], ['mock', 'ollama'])  # Spent.
    refused = games.start('daisyworld', 'watch', 0, minds='together')
    assert refused is not None
    self.assertIn('not available', refused)
    self.assertIsNone(games.current)

  def test_a_live_games_spend_goes_to_the_ledger(self):
    ledger = os.path.join(self.create_tempdir().full_path, 'spend.json')
    games = server_lib.Games(
        live_minds=('together',), budget_usd=5.0, ledger=ledger
    )

    class _Game:  # What the ledger reads of a finished live game.
      game_id, scenario = 'g1', 'daisyworld'

      class context:  # pylint: disable=invalid-name

        class planet:  # pylint: disable=invalid-name
          tick, outcome = 12, 'victory'

      @staticmethod
      def spend():
        return {
            'minds': 'together',
            'label': 'LIVE',
            'calls': 40,
            'tokens': 50_000,
            'usd': 0.05,
        }

    games._log_spend(typing.cast(Any, _Game()))  # pylint: disable=protected-access
    with open(ledger) as f:
      (row,) = json.load(f)
    self.assertEqual(
        (row['scenario'], row['usd'], row['ticks']), ('daisyworld', 0.05, 12)
    )
    self.assertAlmostEqual(games.minds()['left_usd'], 4.95)

  def test_a_mock_game_reports_free_minds(self):
    games = server_lib.Games()
    self.assertEqual(games.minds()['options'], ['mock'])
    self.assertIsNone(games.start('daisyworld', 'watch', 0))
    self.addCleanup(games.stop)
    spend = games.state()['minds']
    self.assertEqual((spend['minds'], spend['usd']), ('mock', 0.0))


class CampaignProgressTest(absltest.TestCase):

  def test_a_played_games_stars_open_the_next_chapter(self):
    folder = self.create_tempdir().full_path
    games = server_lib.Games(progress=os.path.join(folder, 'progress.json'))

    class _Game:  # What a finished played game tells the server.
      game_id, scenario, seed, mode = 'g', 'daisyworld', 0, 'play'

      class context:  # pylint: disable=invalid-name

        class planet:  # pylint: disable=invalid-name
          tick, outcome = 40, 'victory'

          @staticmethod
          def star_count():
            return 2

      @staticmethod
      def spend():
        return {
            'minds': 'mock',
            'label': 'MOCK',
            'calls': 9,
            'tokens': 0,
            'usd': 0.0,
        }

    played = _Game()
    games._ended(typing.cast(Any, played))  # pylint: disable=protected-access
    # Its recap's chapter: the first, a new best, and the next now open.
    chapter = getattr(played, 'chapter')
    self.assertEqual((chapter['number'], chapter['best']), (1, 2))
    self.assertTrue(chapter['next']['opened'])
    chapters = games.campaign()['chapters']
    self.assertEqual(chapters[0]['scenario'], 'daisyworld')
    self.assertEqual((chapters[0]['stars'], chapters[1]['open']), (2, True))
    watched = _Game()
    watched.mode = 'watch'  # A watched game earns nothing.
    watched.scenario = 'herds'
    games._ended(typing.cast(Any, watched))  # pylint: disable=protected-access
    self.assertEqual(games.campaign()['chapters'][1]['stars'], 0)


class TelemetryFolderTest(absltest.TestCase):

  def test_played_games_are_kept_outside_the_published_folder(self):
    folder = server_lib.telemetry_folder('/runs/web_demos/', None)
    self.assertEqual(folder, '/runs/web_demos-telemetry')
    self.assertFalse(folder.startswith('/runs/web_demos/'))
    self.assertEqual(server_lib.telemetry_folder('/runs/d', '/t'), '/t')
    self.assertIsNone(server_lib.telemetry_folder('/runs/d', ''))  # None.
    self.assertIsNone(server_lib.telemetry_folder(None, None))


class SaveResumeTest(absltest.TestCase):

  def test_a_restarted_server_offers_the_save_and_plays_it_on(self):
    folder = self.create_tempdir().full_path
    games = server_lib.Games(telemetry_dir=folder)
    self.assertIsNone(games.start('daisyworld', 'play', 3, ['Gaia']))
    game = games.current
    assert game is not None
    self.addCleanup(game.join, 30)
    self.addCleanup(game.cancel)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
      question = games.state()['question']
      if question and games.state()['tick'] >= 3:
        break
      if question:
        games.choose(question['request_id'], 'wait and save omega')
      time.sleep(0.02)
    self.assertIsNone(games.saved())  # It is running: nothing to resume.
    # A restart: a new server over the same folder.
    again = server_lib.Games(telemetry_dir=folder)
    saved = again.campaign()['saved']
    assert saved is not None
    self.assertEqual(
        (saved['game_id'], saved['scenario'], saved['players']),
        (game.game_id, 'daisyworld', ['Gaia']),
    )
    self.assertEqual(again.resume('../escape'), 'No such save.')
    self.assertIsNone(again.resume(saved['game_id']))
    resumed = again.current
    assert resumed is not None
    self.addCleanup(resumed.join, 30)
    self.addCleanup(resumed.cancel)
    self.assertEqual(resumed.resumed_at, saved['tick'])
    self.assertIsNone(again.saved())  # Being played on.


def _wait_to(games: server_lib.Games, tick: int) -> None:
  """Answers the game's turns with 'wait and save omega' until `tick`."""
  deadline = time.monotonic() + 120
  while time.monotonic() < deadline:
    state = games.state()
    question = state.get('question')
    if question and state['tick'] >= tick:
      return
    if question:
      games.choose(question['request_id'], 'wait and save omega')
    time.sleep(0.02)
  raise AssertionError(f'did not reach tick {tick}')


class RewindTest(absltest.TestCase):

  def _games(self) -> server_lib.Games:
    games = server_lib.Games(telemetry_dir=self.create_tempdir().full_path)
    self.assertIsNone(games.start('daisyworld', 'play', 3, ['Gaia']))

    def stop():
      game = games.current
      if game is not None:
        game.cancel()
        game.join(30)

    self.addCleanup(stop)
    return games

  def test_a_rewound_turn_replays_exactly_and_rewinds_run_out(self):
    straight = self._games()
    _wait_to(straight, 8)
    rewound = self._games()
    _wait_to(rewound, 6)
    self.assertEqual(rewound.state()['rewind']['left'], 3)
    self.assertTrue(rewound.state()['rewind']['can'])
    first = rewound.current
    self.assertIsNone(rewound.rewind())
    assert first is not None
    first.join(30)
    self.assertTrue(first.superseded)
    _wait_to(rewound, 5)
    state = rewound.state()
    self.assertEqual(state['tick'], 5)  # Turn 5 asked again.
    self.assertEqual(state['rewind']['left'], 2)
    # The moves before keep what they did.
    self.assertTrue(all(m['outcome'] for m in state['moves']))
    _wait_to(rewound, 8)

    def history(games):
      game = games.current
      assert game is not None
      rows = [r for r in game.context.planet.history if r['tick'] <= 7]
      return json.dumps(rows, sort_keys=True, default=str)

    self.assertEqual(history(rewound), history(straight))
    # Back twice more in a row, then no more.
    self.assertIsNone(rewound.rewind())
    _wait_to(rewound, 7)
    self.assertIsNone(rewound.rewind())
    _wait_to(rewound, 6)
    self.assertEqual(rewound.state()['rewind']['left'], 0)
    self.assertEqual(rewound.rewind(), 'All 3 rewinds are used.')
    game = rewound.current
    assert game is not None
    self.assertEqual(game.rewinds, 3)


class MeasuredMindsTest(absltest.TestCase):

  def test_minds_are_measured_from_the_ledger_or_the_data(self):
    folder = self.create_tempdir().full_path
    ledger = os.path.join(folder, 'spend.json')
    games = server_lib.Games(ledger=ledger, live_minds=('together',))
    measured = games.minds()['measured']
    self.assertEqual(measured['together']['games'], 11)  # The data's.
    rows = [
        {'minds': 'together', 'usd': u, 'ticks': 40, 'seconds': s, 'outcome': o}
        for u, s, o in (
            (0.1, 80, 'victory'),
            (0.3, 160, 'defeat'),
            (0.2, 120, 'victory'),
        )
    ]
    with open(ledger, 'w', encoding='utf-8') as f:
      json.dump(rows, f)
    mine = games.minds()['measured']['together']
    self.assertEqual(mine['games'], 3)
    self.assertAlmostEqual(mine['usd_per_game'], 0.2)
    self.assertAlmostEqual(mine['seconds_per_turn'], 3.0)
    self.assertAlmostEqual(mine['wins'], 0.67)
    self.assertEqual(games.minds()['measured']['mock']['games'], 176)

  def test_minds_are_measured_in_each_world(self):
    folder = self.create_tempdir().full_path
    ledger = os.path.join(folder, 'spend.json')
    games = server_lib.Games(ledger=ledger, live_minds=('together',))
    by_scenario = games.minds()['by_scenario']
    self.assertEqual(by_scenario['herds']['mock']['games'], 16)  # The data's.
    with open(ledger, 'w', encoding='utf-8') as f:
      json.dump(
          [
              {
                  'minds': 'ollama',
                  'scenario': 'herds',
                  'usd': 0.0,
                  'ticks': 40,
                  'seconds': seconds,
                  'outcome': outcome,
              }
              for seconds, outcome in ((400, 'victory'), (480, 'defeat'))
          ],
          f,
      )
    here = games.minds()['by_scenario']['herds']['ollama']
    self.assertEqual((here['games'], here['wins']), (2, 0.5))
    self.assertAlmostEqual(here['seconds_per_turn'], 11.0)
    self.assertEqual(here['note'], 'games on this server')
    # Two games are not enough to measure the kind overall.
    self.assertEqual(games.minds()['measured']['ollama']['games'], 3)


if __name__ == '__main__':
  absltest.main()
