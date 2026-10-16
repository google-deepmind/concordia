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

"""Tests for the phone page's server: a whole relay game over HTTP."""

import datetime
import json
import os
import tempfile
import time
import threading
import urllib.error
import urllib.request
from unittest import mock

from absl.testing import absltest
from concordia.language_model import no_language_model
from examples.crucible import campaign
from examples.crucible import toys
from examples.crucible.web import app as server_lib


class ServerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.dir = tempfile.mkdtemp()
    self.games = server_lib.Games('mock', None, 100, self.dir)
    self.server = server_lib.serve(self.games, 0, '/crucible')
    self.base = f'http://127.0.0.1:{self.server.server_address[1]}/crucible'

  def tearDown(self):
    self.games.quit()
    self.server.shutdown()
    super().tearDown()

  def call(self, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        self.base + path,
        data=data,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
      return json.loads(response.read())

  def wait_for_request(self):
    for _ in range(200):
      state = self.call('/api/state')
      game = state['game']
      if game and (game['request'] or game['finished']):
        return game
      time.sleep(0.02)
    self.fail('no request')

  def test_page_visits_are_noted(self):
    urllib.request.urlopen(self.base + '/', timeout=10).read()
    with open(os.path.join(self.dir, 'visits.jsonl')) as f:
      self.assertEqual(len(f.readlines()), 1)

  def test_page_is_served_under_the_base_path(self):
    with urllib.request.urlopen(self.base + '/', timeout=10) as response:
      self.assertIn(b'Twelve Words', response.read())

  def test_play_a_relay_game_with_preview_and_telemetry(self):
    self.assertTrue(self.call('/api/start', {'toy': 'relay'})['ok'])
    game = self.wait_for_request()
    self.assertIn('Deliver this message', game['request']['question'])
    preview = self.call('/api/preview', {'text': 'pass'})
    self.assertIn('reads', preview['summary'])
    while not game['finished']:
      self.assertTrue(
          self.call(
              '/api/answer', {'id': game['request']['id'], 'text': 'pass'}
          )['ok']
      )
      time.sleep(0.05)
      game = self.wait_for_request()
    self.assertEqual(game['outcome']['ending'], 'deal')
    files = os.listdir(self.dir)
    self.assertLen(files, 1)
    with open(os.path.join(self.dir, files[0])) as f:
      kinds = [json.loads(line)['kind'] for line in f]
    self.assertEqual(kinds[0], 'start')
    self.assertIn('move', kinds)
    self.assertEqual(kinds[-1], 'end')

  def test_free_play_and_forecast(self):
    self.call('/api/start', {'toy': 'orders', 'scenario': 'one_hand'})
    self.wait_for_request()
    preview = self.call('/api/preview', {'text': 'Build the project.'})
    self.assertIn('If held to the end', preview['forecast'])
    self.call('/api/start', {'toy': 'orders', 'scenario': 'free'})
    game = self.wait_for_request()
    self.assertEqual(game['scenario'], 'free')
    self.assertIn('Turn 1 of 40', game['request']['question'])

  def test_preview_shows_who_takes_up_someone(self):
    self.call('/api/start', {'toy': 'orders', 'scenario': 'free'})
    self.wait_for_request()
    preview = self.call(
        '/api/preview', {'text': 'Someone keep watch; everyone else build.'}
    )
    predicted = preview['predicted']
    self.assertEqual(predicted['actions']['Ash'], 'watch')  # first on a tie
    self.assertEqual(predicted['why']['Ash'], 'Someone keep watch')
    self.assertEqual(predicted['actions']['Bea'], 'build')
    self.assertEqual(predicted['why']['Bea'], 'everyone else build')

  def test_campaign_gates_chapters_and_records_stars(self):
    state = self.call('/api/state')
    self.assertEqual(state['campaign']['toy'], 'passiton')  # leads the page
    chapters = state['campaigns']['orders']['chapters']
    self.assertEqual([c['open'] for c in chapters[:2]], [True, False])
    with self.assertRaises(urllib.error.HTTPError):
      self.call('/api/start', {'toy': 'orders', 'scenario': 'two_hands'})
    self.assertTrue(
        self.call('/api/start', {'toy': 'orders', 'scenario': 'one_hand'})['ok']
    )
    game = self.wait_for_request()
    self.call('/api/answer', {'id': game['request']['id'], 'text': 'Build.'})
    for _ in range(3):
      game = self.wait_for_request()
      if game['finished']:
        break
      self.call('/api/answer', {'id': game['request']['id'], 'text': 'hold'})
      time.sleep(0.05)
    game = self.wait_for_request()
    self.assertEqual(game['outcome']['stars'], 3)
    for _ in range(100):  # The end is recorded just after the last move.
      chapters = self.call('/api/state')['campaigns']['orders']['chapters']
      if chapters[0]['stars']:
        break
      time.sleep(0.02)
    self.assertEqual(chapters[0]['stars'], 3)
    self.assertTrue(chapters[1]['open'])

  def test_the_daily_opens_with_its_chapter(self):
    games = server_lib.Games('mock', None, 100, tempfile.mkdtemp())
    with self.assertRaisesRegex(ValueError, 'opens with its chapter'):
      games.start('orders', 'mock', None, daily=True)

  def test_daily_challenge(self):
    for chapter in toys.load_data('orders')['campaign']:
      self.games.progress.record(chapter, 1)  # every chapter open
    self.assertTrue(
        self.call('/api/start', {'toy': 'orders', 'daily': True})['ok']
    )
    game = self.wait_for_request()
    self.assertTrue(game['daily'])
    today = campaign.daily(toys.load_data('orders'), datetime.date.today())
    self.assertEqual(game['scenario'], today['scenario'])

  def test_a_bosun_chapter_shows_what_the_crew_heard(self):
    tele = tempfile.mkdtemp()
    chapters = toys.load_data('orders')['campaign']
    with open(os.path.join(tele, 'campaign.json'), 'w') as f:
      json.dump({c: 1 for c in chapters[: chapters.index('the_bosun')]}, f)
    games = server_lib.Games('mock', None, 100, tele)
    server = server_lib.serve(games, 0, '/crucible')
    self.base = f'http://127.0.0.1:{server.server_address[1]}/crucible'
    try:
      self.call('/api/start', {'toy': 'orders', 'scenario': 'the_bosun'})
      game = self.wait_for_request()
      self.call(
          '/api/answer',
          {'id': game['request']['id'], 'text': 'Build; Cy, keep watch.'},
      )
      for _ in range(200):
        game = self.call('/api/state')['game']
        if game['receipts']:
          break
        time.sleep(0.02)
      self.assertIn('Ash passes it on: "Build."', game['receipts'][0])
      cards = {m['name']: m for m in game['view']['crew']}
      self.assertEqual(game['view']['bosun'], 'Ash')
      self.assertEqual(cards['Cy']['heard'], 'Build.')
      self.assertTrue(cards['Cy']['via_bosun'])
    finally:
      games.quit()
      server.shutdown()

  def test_paid_minds_only_within_budget(self):
    games = server_lib.Games(
        'mock', None, 10, tempfile.mkdtemp(), ['ollama', 'together'], 0.5, 1.0
    )
    self.assertIn('together', games.modes)
    games._add_spend({'game': 'g', 'mode': 'together', 'usd': 0.6})  # pylint: disable=protected-access
    self.assertNotIn('together', games.modes)
    self.assertIn('ollama', games.modes)

  def _records(self, games):
    del games
    out = []
    for name in sorted(os.listdir(self.dir)):
      if name.endswith('.jsonl') and name != 'visits.jsonl':
        with open(os.path.join(self.dir, name)) as f:
          out.append([json.loads(line) for line in f if line.strip()])
    return out

  def test_two_quick_starts_leave_one_game_and_one_abandoned_record(self):
    threads = [
        threading.Thread(
            target=self.games.start, args=('orders', 'mock', 'one_hand')
        )
        for _ in range(2)
    ]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    game = self.wait_for_request()
    self.assertFalse(game['finished'])
    for _ in range(100):  # the displaced game writes its end record
      records = self._records(self.games)
      ended = [r for r in records if r[-1]['kind'] == 'end']
      if ended:
        break
      time.sleep(0.02)
    self.assertLen(records, 2)
    self.assertLen(ended, 1)
    self.assertTrue(ended[0][-1]['abandoned'])

  def test_a_provider_error_is_answered_by_the_mock_not_the_end(self):
    class Down(no_language_model.NoLanguageModel):

      def sample_text(self, prompt, **kwargs):
        raise ConnectionError('503 Service Unavailable')

      def sample_choice(self, prompt, responses, **kwargs):
        raise ConnectionError('503 Service Unavailable')

    games = server_lib.Games('mock', None, 100, self.dir, ['ollama'])
    with mock.patch.object(
        server_lib.model_setup, 'build_model', return_value=Down()
    ):
      session = games.start('orders', 'ollama', 'one_hand')
    for move in ('Build.', 'hold', 'hold'):
      for _ in range(200):
        if session.reader.pending() is not None or session.finished:
          break
        time.sleep(0.02)
      if session.finished:
        break
      session.reader.submit(session.reader.pending().request_id, move)
    session.thread.join(timeout=10)
    self.assertEqual(session.error, '')
    self.assertTrue(session.game.outcome()['winner'])  # the mock read "Build."
    end = self._records(games)[-1][-1]
    self.assertGreater(end['usage']['fallbacks'], 0)
    self.assertIn('503', end['usage']['fallback_error'])
    self.assertFalse(end['abandoned'])

  def test_while_a_turn_resolves_the_board_is_as_it_was_asked(self):
    session = self.games.start('orders', 'mock', 'one_hand')
    asked = self.wait_for_request()['view']
    session.game.toy.world.points = 0  # the ledger mid-turn, half applied
    session.game.toy.world.order = 'Half applied.'
    with mock.patch.object(session.reader, 'pending', return_value=None):
      game = self.games.state()['game'] or {}
    view = game['view']
    self.assertEqual(view, asked)

  def test_a_failed_preview_says_so(self):
    self.call('/api/start', {'toy': 'orders', 'scenario': 'one_hand'})
    self.wait_for_request()
    with mock.patch.object(
        self.games.session.game.toy, 'simulate', side_effect=RuntimeError
    ):
      reply = self.call('/api/preview', {'text': 'Build.'})
    self.assertFalse(reply['ok'])
    self.assertIn('preview failed', reply['summary'])

  def test_stale_answer_is_refused(self):
    self.call('/api/start', {'toy': 'orders'})
    self.wait_for_request()
    self.assertFalse(
        self.call('/api/answer', {'id': 'stale', 'text': 'x'})['ok']
    )


if __name__ == '__main__':
  absltest.main()
