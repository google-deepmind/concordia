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

import json
import os
import tempfile
import time
import urllib.error
import urllib.request

from absl.testing import absltest
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

  def test_page_is_served_under_the_base_path(self):
    with urllib.request.urlopen(self.base + '/', timeout=10) as response:
      self.assertIn(b'Crucible', response.read())

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

  def test_campaign_gates_chapters_and_records_stars(self):
    state = self.call('/api/state')
    chapters = state['campaign']['chapters']
    self.assertEqual([c['open'] for c in chapters], [True, False, False, False])
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
      chapters = self.call('/api/state')['campaign']['chapters']
      if chapters[0]['stars']:
        break
      time.sleep(0.02)
    self.assertEqual(chapters[0]['stars'], 3)
    self.assertTrue(chapters[1]['open'])

  def test_daily_challenge(self):
    self.assertTrue(
        self.call('/api/start', {'toy': 'orders', 'daily': True})['ok']
    )
    game = self.wait_for_request()
    self.assertTrue(game['daily'])
    self.assertEqual(game['scenario'], 'voyage')

  def test_stale_answer_is_refused(self):
    self.call('/api/start', {'toy': 'orders'})
    self.wait_for_request()
    self.assertFalse(
        self.call('/api/answer', {'id': 'stale', 'text': 'x'})['ok']
    )


if __name__ == '__main__':
  absltest.main()
