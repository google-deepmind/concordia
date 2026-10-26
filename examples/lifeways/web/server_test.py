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
import os
import threading
import time
from typing import Any
from unittest import mock
import urllib.error
import urllib.request

from absl.testing import absltest
from examples.lifeways import scenarios
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


def _forked_in(twin: session_lib.Session) -> int:
  assert twin.twin_of is not None, 'not a twin'
  return int(twin.twin_of['year'])


def _known(found: dict[str, Any] | None) -> dict[str, Any]:
  assert found is not None, 'not known'
  return found


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

  def test_free_mode_runs_on_through_deaths(self):
    session = session_lib.Session(free=0.02, years_per_second=50)
    session.start()
    try:
      self.assertTrue(_wait(lambda: session.state()['year'] >= 80, 120))
      self.assertEqual(session.state()['error'], '')
    finally:
      session.close()

  def test_a_history_of_none_breaks_nothing(self):
    session = session_lib.Session(years_per_second=50, history_kept=0)
    session.start()
    try:
      self.assertTrue(_wait(lambda: session.state()['year'] >= 3))
      self.assertEqual(session.state()['error'], '')
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


class SaveTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.path = os.path.join(self.create_tempdir().full_path, 'world.json')

  def test_a_saved_world_plays_on_from_its_save(self):
    first = session_lib.Session(
        years_per_second=40, save_path=self.path, save_seconds=0
    )
    self.assertIsNone(first.restored_year)
    first.start()
    try:
      self.assertTrue(_wait(lambda: (first.saved_year or 0) >= 4))
    finally:
      first.close()
    with open(self.path) as f:
      saved = json.load(f)

    second = session_lib.Session(save_path=self.path)

    state = second.state()
    self.assertEqual(state['year'], saved['year'])
    self.assertEqual(state['restored_year'], saved['year'])
    self.assertEqual(state['pace']['years_per_second'], 40)
    self.assertLen(state['people'], len(saved['checkpoint']['entities']) - 1)
    self.assertEqual(second.census(), saved['rows'])
    second.start()
    try:
      self.assertTrue(_wait(lambda: second.state()['year'] > saved['year'] + 1))
      self.assertEqual(second.state()['error'], '')
    finally:
      second.close()

  def test_a_stopping_server_saves_at_the_next_year(self):
    session = session_lib.Session(
        years_per_second=20, save_path=self.path, save_seconds=1e9
    )
    session.start()
    try:
      self.assertTrue(_wait(lambda: session.state()['year'] >= 2))
      self.assertIsNone(session.saved_year)
      self.assertTrue(session.save_soon(timeout=5))
      self.assertGreaterEqual(session.saved_year or 0, 2)
      session.pace(paused=True)
      self.assertFalse(session.save_soon(timeout=1))
    finally:
      session.close()

  def test_a_save_of_another_world_is_set_aside(self):
    first = session_lib.Session(
        years_per_second=40, save_path=self.path, save_seconds=0
    )
    first.start()
    try:
      self.assertTrue(_wait(lambda: first.saved_year is not None))
    finally:
      first.close()

    other = session_lib.Session(seed=1, save_path=self.path)

    self.assertIsNone(other.restored_year)
    self.assertEqual(other.state()['year'], 0)
    self.assertFalse(os.path.exists(self.path))
    self.assertTrue(os.path.exists(self.path + '.set-aside'))

  def test_a_save_that_no_longer_restores_begins_a_new_world(self):
    fresh = session_lib.Session()
    with open(self.path, 'w') as f:
      json.dump(
          {
              'version': session_lib.SAVE_VERSION,
              'world': {'scenario': 'sandbox', 'seed': 0, 'overrides': {}},
              'year': 7,
              'checkpoint': {'entities': {'Ada': 'not a state'}},
          },
          f,
      )

    session = session_lib.Session(save_path=self.path)

    self.assertIsNone(session.restored_year)
    self.assertEqual(session.state()['year'], 0)
    self.assertEqual(
        [p['name'] for p in session.state()['people']],
        [p['name'] for p in fresh.state()['people']],
    )
    self.assertTrue(os.path.exists(self.path + '.set-aside'))


class TwinTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.session = session_lib.Session(years_per_second=50)
    self.session.start()
    self.assertTrue(_wait(lambda: self.session.state()['year'] >= 3))

  def tearDown(self):
    self.session.close()
    super().tearDown()

  def _rows(self, session, first: int, last: int):
    return [r for r in session.census() if first <= r['year'] <= last]

  def test_a_twin_with_nothing_changed_plays_on_as_its_world_does(self):
    twin = self.session.fork('richness', 0.2)  # The Sandbox's own.
    year = _forked_in(twin)

    # The rows before the year forked are the world's; the twin plays on.
    self.assertEqual(
        self._rows(twin, -1, year - 1), self._rows(self.session, -1, year - 1)
    )
    self.assertTrue(
        _wait(
            lambda: min(self.session.state()['year'], twin.state()['year'])
            > year + 10,
            60,
        )
    )
    self.assertLen(self._rows(twin, year, year + 10), 11)
    self.assertEqual(
        self._rows(twin, year, year + 10),
        self._rows(self.session, year, year + 10),
    )

  def test_a_richer_twin_feeds_more(self):
    twin = self.session.fork('richness', 0.5)
    year = _forked_in(twin)

    self.assertTrue(
        _wait(
            lambda: min(self.session.state()['year'], twin.state()['year'])
            > year + 20,
            120,
        )
    )
    living = lambda s: sum(
        r['population'] for r in self._rows(s, year + 11, year + 20)
    )
    self.assertGreater(living(twin), living(self.session))
    self.assertEqual(self.session.state()['twin']['lever'], 'richness')

  def test_releases_reach_the_twin_and_questions_do_not(self):
    twin = self.session.fork(scenarios.WITHOUT, 'story')
    self.session.pace(paused=True)
    self.assertTrue(twin.input.settings()['paused'])
    name = twin.state()['people'][0]['name']

    self.session.move('release belief sacred as a song at 8,8: Hm.')
    self.session.move(f'ask {name}: Why?')

    self.assertEqual(
        twin.input.settings()['queued'],
        ['release belief sacred as a song at 8,8: Hm.'],
    )
    self.assertLen(self.session.input.settings()['queued'], 2)

  def test_a_lever_must_be_one_and_the_world_playing(self):
    for lever, value in (
        ('speed', 1.0),
        ('richness', 5.0),
        ('richness', 'rich'),
        (scenarios.WITHOUT, 'telepathy'),
    ):
      with self.assertRaises(ValueError):
        self.session.fork(lever, value)
    self.session.pace(paused=True)
    with self.assertRaises(RuntimeError):
      self.session.fork('richness', 0.3)
    self.assertIsNone(self.session.state()['twin'])

  def test_a_twin_may_go_without_any_channel(self):
    engine = self.session.run.engine
    for channel in self.session.state()['channels']:
      twin = self.session.fork(scenarios.WITHOUT, channel)
      engine.pause()  # The world waits while its twin plays (a test's pace).
      year = _forked_in(twin)
      self.assertTrue(
          _wait(lambda: twin.state()['year'] >= year + 4, 60), channel
      )
      self.assertEqual(twin.error, '', channel)
      self.assertNotIn(channel, twin.state()['channels'])
      engine.play()


class TreesTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.session = session_lib.Session(years_per_second=50)
    cls.session.start()
    assert _wait(lambda: cls.session.state()['year'] >= 40, 60)
    cls.session.pace(paused=True)
    time.sleep(0.3)

  @classmethod
  def tearDownClass(cls):
    cls.session.close()
    super().tearDownClass()

  def test_a_lineage_runs_from_the_oldest_variant_down(self):
    customs = self.session.run.world_state()['culture']['customs']
    child = next(c for c in customs if c['parent'])

    lineage = _known(self.session.lineage(child['id']))

    variants = lineage['variants']
    self.assertEqual(variants[0]['id'], lineage['root'])
    self.assertEqual(variants[0]['depth'], 0)
    self.assertIn(child['id'], [v['id'] for v in variants])
    depth = {variants[0]['id']: 0}
    for variant in variants[1:]:  # Each under the one it was copied from.
      self.assertEqual(variant['depth'], depth[variant['parent']] + 1)
      depth[variant['id']] = variant['depth']
    self.assertIsNone(self.session.lineage('c999999'))

  def test_holders_say_how_and_from_whom_they_had_a_custom(self):
    living = {p['name'] for p in self.session.state()['people']}
    held = [
        h
        for c in self.session.run.world_state()['culture']['customs']
        for v in (self.session.lineage(c['id']) or {'variants': []})['variants']
        if v['id'] == c['id']
        for h in v['holders']
    ]

    self.assertNotEmpty(held)
    self.assertContainsSubset({h['name'] for h in held}, living)
    self.assertTrue(any(h['via'] == 'observation' and h['from'] for h in held))

  def test_kin_reach_the_remembered_dead(self):
    world = self.session.run.world_state()
    people = {p['name']: p for p in world['people']}
    young = min(people.values(), key=lambda p: -p['born'])

    kin = _known(self.session.kin(young['name']))

    self.assertEqual([p['name'] for p in kin['parents']], young['parents'])
    for parent in kin['parents']:
      self.assertEqual(parent['living'], parent['name'] in people)
    dead = world['dead'][-1]
    remembered = _known(self.session.kin(dead['name']))
    self.assertFalse(remembered['person']['living'])
    self.assertEqual(remembered['person']['died'], dead['died'])
    for child in remembered['children']:
      child_kin = self.session.kin(child['name'])
      if child_kin is not None:  # A child the valley still remembers.
        self.assertIn(dead['name'], [p['name'] for p in child_kin['parents']])
    self.assertIsNone(self.session.kin('Nobody'))


class NotebookTest(absltest.TestCase):

  def test_the_person_writes_and_the_session_notes_moves_and_reports(self):
    session = session_lib.Session(years_per_second=40)
    entry = session.note('hypothesis', ' Songs spread faster than texts. ')
    self.assertEqual(
        (entry['kind'], entry['text'], entry['year']),
        ('hypothesis', 'Songs spread faster than texts.', 0),
    )
    for kind, text in (('rant', 'Hm.'), ('note', ' '), ('note', 'x' * 1001)):
      with self.assertRaises(ValueError):
        session.note(kind, text)

    session.move('release belief sacred as a song at 8,8: The sun sings.')
    session.start()
    try:
      self.assertTrue(
          _wait(
              lambda: any(
                  e['kind'] == session_lib.WORLD
                  and 'is sung at 8,8' in e['text']
                  for e in session.notebook()
              )
          )
      )
      self.assertTrue(
          _wait(
              lambda: any(
                  e['kind'] == session_lib.EVENT
                  and 'most widely held' in e['text']
                  for e in session.notebook()
              )
          )
      )
    finally:
      session.close()
    kinds = [e['kind'] for e in session.notebook()]
    self.assertEqual(kinds[:2], ['hypothesis', session_lib.MOVE])
    ids = [e['id'] for e in session.notebook()]
    self.assertEqual(ids, sorted(set(ids)))
    self.assertEqual(session.notebook(since=ids[-2]), session.notebook()[-1:])
    page = session.notebook_markdown()
    self.assertIn('## Year 0', page)
    self.assertIn('- **hypothesis**: Songs spread faster than texts.', page)

  def test_the_notebook_keeps_all_the_person_wrote_and_bounds_the_rest(self):
    session = session_lib.Session()
    with mock.patch.object(session_lib, 'NOTEBOOK_KEPT', 3):
      session.note('note', 'First.')
      for i in range(5):
        session.move(f'ask Nobody{i}: Why?')
      session.note('observation', 'Last.')

    entries = session.notebook()
    self.assertEqual(
        [e['text'] for e in entries if e['kind'] != session_lib.MOVE],
        ['First.', 'Last.'],
    )
    self.assertLen([e for e in entries if e['kind'] == session_lib.MOVE], 3)

  def test_the_notebook_is_saved_with_the_world(self):
    path = os.path.join(self.create_tempdir().full_path, 'world.json')
    first = session_lib.Session(
        years_per_second=40, save_path=path, save_seconds=0
    )
    first.note('hypothesis', 'Nothing will change.')
    first.start()
    try:
      self.assertTrue(_wait(lambda: (first.saved_year or 0) >= 2))
    finally:
      first.close()

    second = session_lib.Session(save_path=path)
    written = second.note('observation', 'Something changed.')

    texts = [e['text'] for e in second.notebook()]
    self.assertIn('Nothing will change.', texts)
    self.assertEqual(texts[-1], 'Something changed.')
    self.assertGreater(written['id'], second.notebook()[-2]['id'])
    self.assertEqual(written['year'], second.restored_year)


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

  def test_bad_requests_are_refused_and_the_bare_prefix_redirects(self):
    for path, data in (
        ('/api/move', [1]),
        ('/api/pace', {'years_per_second': 'fast'}),
        ('/api/pace', {'paused': 'false'}),
    ):
      with self.assertRaises(urllib.error.HTTPError) as caught:
        self._post(path, data)
      self.assertEqual(caught.exception.code, 400)
    with self.assertRaises(urllib.error.HTTPError) as caught:
      self._get('/api/census?since=abc')
    self.assertEqual(caught.exception.code, 400)
    with urllib.request.urlopen(self.base, timeout=10) as response:
      self.assertEqual(response.url, self.base + '/')

  def test_the_page_forks_a_twin_and_reads_its_census(self):
    self.assertTrue(_wait(lambda: self._get('/api/state')['year'] >= 2))
    self.assertEqual(self._get('/api/census?world=twin'), [])

    forked = self._post('/api/twin', {'lever': 'mutation', 'value': 0.3})

    year = forked['twin']['year']
    self.assertTrue(
        _wait(lambda: self._get('/api/state')['twin']['year_now'] > year + 2)
    )
    rows = self._get(f'/api/census?since={year - 1}&world=twin')
    self.assertNotEmpty(rows)
    self.assertEqual(rows[0]['year'], year)
    for data, code in (
        ({'lever': 'mutation', 'value': 9}, 400),
        ({'lever': 'nonsense', 'value': 1}, 400),
    ):
      with self.assertRaises(urllib.error.HTTPError) as caught:
        self._post('/api/twin', data)
      self.assertEqual(caught.exception.code, code)
    with self.assertRaises(urllib.error.HTTPError) as caught:
      self._get('/api/census?world=other')
    self.assertEqual(caught.exception.code, 400)
    self._post('/api/pace', {'paused': True})
    with self.assertRaises(urllib.error.HTTPError) as caught:
      self._post('/api/twin', {'lever': 'mutation', 'value': 0.1})
    self.assertEqual(caught.exception.code, 409)

    self.assertEqual(self._post('/api/twin', {'end': True}), {'twin': None})
    self.assertIsNone(self._get('/api/state')['twin'])

  def test_the_page_reads_kin_and_lineages(self):
    self.assertTrue(_wait(lambda: self._get('/api/state')['year'] >= 3))
    name = self._get('/api/state')['people'][0]['name']

    kin = self._get(f'/api/kin?name={name}')
    person = self._get(f'/api/person?name={name}')
    lineage = self._get(f'/api/lineage?id={person["customs"][0]["id"]}')

    self.assertEqual(kin['person']['name'], name)
    self.assertIn('from', person['customs'][0])
    self.assertNotEmpty(lineage['variants'])

  def test_the_page_writes_in_the_notebook_and_keeps_a_copy(self):
    entry = self._post('/api/note', {'kind': 'observation', 'text': 'Hm.'})

    self.assertEqual(entry['text'], 'Hm.')
    self.assertIn(entry, self._get('/api/notebook?since=0'))
    self.assertNotIn(entry, self._get(f'/api/notebook?since={entry["id"]}'))
    self.assertIn('- **observation**: Hm.', self._get('/notebook.md'))
    for data in ({'kind': 'rant', 'text': 'Hm.'}, {'kind': 'note'}):
      with self.assertRaises(urllib.error.HTTPError) as caught:
        self._post('/api/note', data)
      self.assertEqual(caught.exception.code, 400)
    with self.assertRaises(urllib.error.HTTPError) as caught:
      self._get('/api/notebook?since=x')
    self.assertEqual(caught.exception.code, 400)

  def test_unknown_paths_and_people_are_not_found(self):
    for path in (
        '/api/nothing',
        '/static/../server.py',
        '/api/person?name=Zz',
        '/api/kin?name=Zz',
        '/api/lineage?id=c999999',
    ):
      with self.assertRaises(urllib.error.HTTPError) as caught:
        self._get(path)
      self.assertEqual(caught.exception.code, 404)


def tearDownModule():
  # Let the closed worlds' loops finish their turn before the process exits.
  session_lib.await_engines()


if __name__ == '__main__':
  absltest.main()
