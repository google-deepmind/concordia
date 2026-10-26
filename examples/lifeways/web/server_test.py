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

"""The page's session and server, on a running MOCK world."""

import json
import threading
import time
import urllib.error
import urllib.request

from absl.testing import absltest
from examples.lifeways.web import server as server_lib
from examples.lifeways.web import session as session_lib
from concordia.typing import entity as entity_lib
from concordia.typing import human_input


def _wait(condition, seconds: float = 30.0) -> bool:
  deadline = time.monotonic() + seconds
  while time.monotonic() < deadline:
    if condition():
      return True
    time.sleep(0.05)
  return False


class PageInputTest(absltest.TestCase):

  def _request(self) -> human_input.HumanInputRequest:
    return human_input.HumanInputRequest(
        request_id='r1',
        entity_name='Experimenter',
        action_spec=entity_lib.free_action_spec(call_to_action='What now?'),
        contexts={},
    )

  def test_it_answers_with_the_queued_move_then_the_standing_one(self):
    page = session_lib.PageInput(years_per_second=50)
    page.queue('release belief sacred as a song at 1,1: Hm.')

    self.assertEqual(
        page(self._request()), 'release belief sacred as a song at 1,1: Hm.'
    )
    self.assertEqual(page(self._request()), 'watch')

  def test_closing_raises_rather_than_answering(self):
    page = session_lib.PageInput()
    page.pace(paused=True)
    threading.Timer(0.2, page.close).start()

    with self.assertRaises(session_lib.Closed):
      page(self._request())


class FreeModeTest(absltest.TestCase):

  def test_in_free_mode_years_pass_while_the_experimenter_waits(self):
    session = session_lib.Session(free=0.3)
    session.input.pace(paused=True)  # The Experimenter never answers.
    session.start()
    try:
      self.assertTrue(_wait(lambda: session.state()['year'] >= 3, 20))
    finally:
      session.close()

  def test_in_tick_mode_the_year_waits_for_everyone(self):
    session = session_lib.Session()
    session.input.pace(paused=True)
    session.start()
    try:
      time.sleep(1.0)
      self.assertEqual(session.state()['year'], 0)
    finally:
      session.close()


class ServerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.session = session_lib.Session(years_per_second=50)
    self.server = server_lib.serve(self.session, 0, '/lifeways')
    threading.Thread(target=self.server.serve_forever, daemon=True).start()
    self.base = f'http://127.0.0.1:{self.server.server_address[1]}/lifeways'
    self.session.start()

  def tearDown(self):
    self.session.close()
    self.server.shutdown()
    self.server.server_close()
    super().tearDown()

  def _get(self, path: str):
    with urllib.request.urlopen(self.base + path, timeout=10) as response:
      body = response.read()
      if response.headers['Content-Type'].startswith('application/json'):
        return json.loads(body)
      return body.decode()

  def _post(self, path: str, data):
    request = urllib.request.Request(
        self.base + path,
        data=json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=10) as response:
      return json.loads(response.read())

  def test_the_page_reads_and_plays_the_world(self):
    self.assertIn('<canvas id="map"', self._get('/'))
    self.assertIn('function releaseMove', self._get('/static/app.js'))
    self.assertTrue(_wait(lambda: self._get('/api/state')['year'] >= 3))

    state = self._get('/api/state')
    self.assertLen(state['terrain'], state['rows'])
    self.assertNotEmpty(state['people'])
    self.assertIn('belief', state['custom_kinds'])
    name = state['people'][0]['name']
    self.assertEqual(self._get(f'/api/person?name={name}')['name'], name)
    self.assertNotEmpty(self._get('/api/census?since=-1'))

    self._post(
        '/api/move', {'move': 'release belief sacred as a song at 8,8: Hm.'}
    )
    self.assertTrue(
        _wait(lambda: any(r['released'] for r in self._get('/api/census')))
    )

    paused = self._post('/api/pace', {'paused': True})
    self.assertTrue(paused['paused'])
    year = self._get('/api/state')['year']
    time.sleep(0.5)
    self.assertLessEqual(self._get('/api/state')['year'], year + 1)

    self._post('/api/stop', {})
    self.assertTrue(_wait(lambda: not self._get('/api/state')['running']))
    self.assertEqual(self._get('/api/state')['error'], '')

  def test_unknown_paths_and_people_are_not_found(self):
    for path in ('/api/nothing', '/static/../server.py', '/api/person?name=Zz'):
      with self.assertRaises(urllib.error.HTTPError) as caught:
        self._get(path)
      self.assertEqual(caught.exception.code, 404)


if __name__ == '__main__':
  absltest.main()
