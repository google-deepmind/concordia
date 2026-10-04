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

"""In-memory origin checks; dispatch is mocked, with no listener or runner."""

from email.message import Message
import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.utils import browser_sessions
from concordia.utils import operation_service
from concordia.utils import simulation_server

_PUBLIC_ORIGIN = 'https://editor.example:10000'


class OperationOriginTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    for name in ('start', 'run_project'):
      guard = mock.patch.object(
          simulation_server.SimulationServer,
          name,
          side_effect=AssertionError('No listener or simulation may start.'),
      )
      guard.start()
      self.addCleanup(guard.stop)
    self.service = mock.create_autospec(
        operation_service.OperationService, instance=True
    )
    self.service.dispatch.return_value = {'result': 'mock dispatch only'}

  def dispatch(
      self,
      origin,
      *,
      host='127.0.0.1:8081',
      public_origin=None,
      session=False,
      path='/api/dispatch',
      forwarded=False,
  ):
    sessions = (
        browser_sessions.BrowserSessions((), secure=public_origin is not None)
        if session
        else None
    )
    server = simulation_server.SimulationServer(
        port=8081,
        operation_service=self.service,
        browser_sessions=sessions,
        public_origin=public_origin,
    )
    handler = object.__new__(server._create_handler())
    handler.path = path
    handler.headers = Message()
    handler.headers['Host'] = host
    if origin is not None:
      handler.headers['Origin'] = origin
    if forwarded:
      handler.headers['Forwarded'] = 'host=editor.example:10000;proto=https'
      handler.headers['X-Forwarded-Host'] = 'editor.example:10000'
      handler.headers['X-Forwarded-Proto'] = 'https'
    body = {'operation': 'project.run', 'arguments': {}}
    handler._operation_body = json.dumps(body).encode('utf-8')
    handler.headers['Content-Length'] = str(len(handler._operation_body))
    handler.headers['Content-Type'] = 'application/json'
    handler._request_audience = 'developer'
    handler._send_json = mock.Mock()
    handler._handle_operation_post()
    self.assertTrue(handler.close_connection)
    return handler, body

  @parameterized.product(
      host=(
          '127.0.0.1:8081',
          'localhost:8081',
          'localhost:8080',
          'localhost',
          '[::1]:8081',
      ),
      session=(False, True),
  )
  def test_local_origin_matches_exact_request_host(self, host, session):
    handler, body = self.dispatch('http://' + host, host=host, session=session)
    self.service.dispatch.assert_called_once_with('developer', body)
    handler._send_json.assert_called_once_with(
        self.service.dispatch.return_value
    )

  @parameterized.parameters(False, True)
  def test_proxy_uses_configured_origin_not_backend_host(self, session):
    handler, body = self.dispatch(
        _PUBLIC_ORIGIN,
        public_origin=_PUBLIC_ORIGIN,
        session=session,
    )
    self.service.dispatch.assert_called_once_with('developer', body)
    handler._send_json.assert_called_once_with(
        self.service.dispatch.return_value
    )

  @parameterized.parameters(
      'http://localhost:8081',
      'http://127.0.0.1:8080',
      'https://127.0.0.1:8081',
      'http://127.0.0.1.example:8081',
      _PUBLIC_ORIGIN,
      'null',
  )
  def test_other_origins_cannot_dispatch_locally(self, origin):
    handler, _ = self.dispatch(origin, forwarded=True)
    self.service.dispatch.assert_not_called()
    handler._send_json.assert_called_once_with(
        {'error': {'code': 'origin', 'message': 'Same-origin requests only.'}},
        403,
    )

  @parameterized.parameters(
      'http://127.0.0.1:8081',
      'http://localhost:8081',
      'https://editor.example',
      'http://editor.example:10000',
      'null',
  )
  def test_proxy_rejection_identifies_configured_editor_address(self, origin):
    handler, _ = self.dispatch(
        origin,
        public_origin=_PUBLIC_ORIGIN,
        forwarded=True,
    )
    self.service.dispatch.assert_not_called()
    result, status = handler._send_json.call_args.args
    self.assertEqual(status, 403)
    self.assertEqual(result['error']['code'], 'origin')
    self.assertIn(
        'Open the editor at ' + _PUBLIC_ORIGIN + '/', result['error']['message']
    )

  @parameterized.product(
      public_origin=(None, _PUBLIC_ORIGIN),
      path=('/api/dispatch', '/api/join'),
  )
  def test_browser_session_still_requires_origin(self, public_origin, path):
    handler, _ = self.dispatch(
        None,
        session=True,
        public_origin=public_origin,
        path=path,
    )
    self.service.dispatch.assert_not_called()
    self.service.publish.assert_not_called()
    result, status = handler._send_json.call_args.args
    self.assertEqual(status, 403)
    self.assertEqual(result['error']['code'], 'origin')

  def test_non_browser_client_keeps_existing_missing_origin_contract(self):
    handler, body = self.dispatch(None)
    self.service.dispatch.assert_called_once_with('developer', body)
    handler._send_json.assert_called_once_with(
        self.service.dispatch.return_value
    )


if __name__ == '__main__':
  absltest.main()
