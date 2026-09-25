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

"""Host-approved browser roles using standard opaque HTTP cookie sessions.

This is a process-lifetime session store, not an account service. Approval is
explicit on a separate trusted host surface. Browser labels are NOT identity
proof: hosts must verify the intended participant before approving a request.
No role, invitation, session credential or private state belongs in a URL/log.
"""

from dataclasses import dataclass
from http import cookies
import secrets
import threading
import uuid

from concordia.utils import operation_service as ops


@dataclass
class Browser:
  principal: str
  label: str = ''
  requested_role: str = ''
  role: str = ''


class BrowserSessions:
  """Bounded sessions: retain reloads, fail closed on process restart."""

  def __init__(self, roles, *, capacity=128, secure=True, cookie_path='/'):
    if not cookie_path.startswith('/') or any(
        c in cookie_path for c in ';\r\n'
    ):
      raise ValueError('Invalid cookie path')
    self.roles = tuple(roles)
    self.secure = secure
    self.path = cookie_path
    self.cookie_name = 'concordia_' + uuid.uuid4().hex
    self._sessions: dict[str, str] = {}
    self._browsers: dict[str, Browser] = {}
    self._capacity = capacity
    self._lock = threading.RLock()

  def identify(self, cookie_header, *, create=False):
    """Return the internal principal and optional Set-Cookie, never a role."""
    jar = cookies.SimpleCookie()
    try:
      jar.load(cookie_header or '')
    except cookies.CookieError:
      pass
    morsel = jar.get(self.cookie_name)
    with self._lock:
      token = morsel.value if morsel else ''
      if token in self._sessions:
        return self._sessions[token], None
      if not create:
        raise ops.OperationError(
            'unauthorized', 'Open the join page in this browser first.'
        )
      if len(self._sessions) >= self._capacity:
        raise ops.OperationError(
            'session_capacity', 'The host must start a new session.'
        )
      token = secrets.token_urlsafe(32)
      principal = 'browser:' + uuid.uuid4().hex
      self._sessions[token] = principal
      self._browsers[principal] = Browser(principal)
      jar = cookies.SimpleCookie()
      jar[self.cookie_name] = token
      jar[self.cookie_name]['path'] = self.path
      jar[self.cookie_name]['httponly'] = True
      jar[self.cookie_name]['samesite'] = 'Strict'
      if self.secure:
        jar[self.cookie_name]['secure'] = True
      return principal, jar[self.cookie_name].OutputString()

  def audience(self, principal):
    with self._lock:
      browser = self._browsers.get(principal)
      if browser is None:
        # Only the trusted local listener supplies fixed developer identity.
        return principal if principal == 'developer' else 'visitor'
      return 'role:' + browser.role if browser.role else 'visitor'

  def request(self, principal, label, role):
    with self._lock:
      browser = self._browsers.get(principal)
      if browser is None:
        raise ops.OperationError('unauthorized', 'Open the join page first.')
      if browser.role:
        raise ops.OperationError(
            'already_joined', 'This browser already holds a role.'
        )
      if role not in self.roles or not label.strip() or len(label) > 80:
        raise ops.OperationError(
            'invalid_join', 'Choose an available role and a short name.'
        )
      if any(b.role == role for b in self._browsers.values()):
        raise ops.OperationError(
            'role_taken', 'That role is already held; ask the host.'
        )
      browser.label, browser.requested_role = label.strip(), role
      return {'requested_role': role, 'label': browser.label}

  def approve(self, principal):
    with self._lock:
      browser = self._browsers.get(principal)
      if browser is None or not browser.requested_role:
        raise ops.OperationError(
            'invalid_join', 'Select a pending join request.'
        )
      if browser.role:
        return {'role': browser.role}
      role = browser.requested_role
      if any(b.role == role for b in self._browsers.values()):
        raise ops.OperationError(
            'role_taken', 'Role already held; no state changed.'
        )
      browser.role = role
      return {'role': role}

  def revoke(self, principal):
    with self._lock:
      browser = self._browsers.get(principal)
      if browser is None:
        raise ops.OperationError('invalid_join', 'Unknown join request.')
      browser.role = browser.requested_role = ''
      return {'revoked': True}

  def own(self, principal):
    with self._lock:
      browser = self._browsers.get(principal)
      return {
          'label': browser.label if browser else '',
          'requested_role': browser.requested_role if browser else '',
          'roles': [
              r
              for r in self.roles
              if not any(b.role == r for b in self._browsers.values())
          ],
      }

  def pending(self):
    with self._lock:
      return [
          {
              'id': b.principal,
              'label': b.label,
              'requested_role': b.requested_role,
              'role': b.role,
          }
          for b in self._browsers.values()
          if b.requested_role
      ]

  def ready(self, roles):
    with self._lock:
      return set(roles) <= {b.role for b in self._browsers.values()}
