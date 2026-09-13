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

"""In-process, reconnectable transport adapter; no simulation logic lives here."""

from collections.abc import Callable
import threading

from concordia.components.agent import human_act_component
from concordia.typing import human_input


class InputClosed(Exception):
  """The controller has stopped waiting for input."""


class StaleRequest(ValueError):
  """A response targets an old or already answered prompt."""


class HumanSession:
  """One controller's inbox, not a global queue or a multiplayer session.

  Closing a browser does not close this inbox. Requests survive reconnects while
  the process runs. A new process always has new request IDs; no action is replayed.
  """

  def __init__(
      self,
      *,
      role: str = 'player',
      on_request: Callable[[], None] | None = None,
      initial_status: str = 'The star-loom is waking…',
  ):
    self.role = role
    self._on_request = on_request
    self._condition = threading.Condition()
    self._pending: human_input.HumanInputRequest | None = None
    self._response: str | None = None
    self._accepted: tuple[str, str] | None = None
    self._closed = False
    self._status = initial_status
    self._entries: list[dict[str, str]] = []
    self._observations_seen = 0
    self._revision = 0
    self._step = 0

  def __call__(self, request: human_input.HumanInputRequest) -> str:
    with self._condition:
      if self._closed:
        raise InputClosed()
      if self._pending is not None:
        raise RuntimeError('This controller already has a pending request.')
      self._pending = request
      self._response = None
      self._status = (
          'Your move' if self.role == 'player' else 'The world awaits you'
      )
      self._revision += 1
      # The standard observation component orders memories chronologically.
      # Only this controlled entity's observations enter the player transcript.
      observations = request.contexts.get('__observation__', '')
      parts = observations.split('[observation] ')[1:]
      for part in parts[self._observations_seen :]:
        self._entries.append({'kind': 'story', 'text': part.strip()})
      self._observations_seen = len(parts)
      self._condition.notify_all()
    if self._on_request is not None:
      self._on_request()
    with self._condition:
      self._condition.wait_for(
          lambda: self._response is not None or self._closed
      )
      if self._closed:
        self._pending = None
        raise InputClosed()
      response = self._response
      assert response is not None  # Guaranteed by wait_for unless closed.
      self._pending = None
      self._response = None
      return response

  def submit(self, request_id: str, response: str) -> bool:
    """Validate before wakeup; retries of the same accepted POST are idempotent."""
    with self._condition:
      if self._accepted == (request_id, response):
        return False
      if (
          self._closed
          or self._pending is None
          or self._pending.request_id != request_id
          or self._response is not None
      ):
        raise StaleRequest(
            'That turn has already moved on. Your draft is kept.'
        )
      human_act_component.validate_response(response, self._pending.action_spec)
      self._accepted = (request_id, response)
      self._response = response
      self._entries.append({'kind': 'action', 'text': response})
      self._status = 'Your action is unfolding…'
      self._revision += 1
      self._condition.notify_all()
      return True

  def add_observation(self, observation: str) -> None:
    """Publish a final observation addressed to this controller's entity."""
    with self._condition:
      if observation.strip():
        self._entries.append({'kind': 'story', 'text': observation})
        self._revision += 1

  def progress(self, step: int, actor: str) -> None:
    with self._condition:
      self._step = step
      self._status = f'{actor} has acted. The story continues…'
      self._revision += 1

  def finish(self, message: str) -> None:
    with self._condition:
      self._status = message
      self._closed = True
      self._pending = None
      self._revision += 1
      self._condition.notify_all()

  def snapshot(self) -> dict:
    """Return only the controller's view, never any other entity's private context."""
    with self._condition:
      pending = None
      if self._pending is not None and self._response is None:
        request = self._pending
        pending = {
            'id': request.request_id,
            'entity': request.entity_name,
            'prompt': request.action_spec.call_to_action,
            'type': request.action_spec.output_type.value,
            'options': list(request.action_spec.options),
            'error': request.error,
            'context': request.context,
        }
      return {
          'revision': self._revision,
          'role': self.role,
          'status': self._status,
          'step': self._step,
          'finished': self._closed,
          'pending': pending,
          'entries': [dict(entry) for entry in self._entries],
      }
