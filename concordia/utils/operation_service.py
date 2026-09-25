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

"""Shared, in-process operation registry for attached editor/CLI transports.

This is not a scheduler or checkpoint manager. Registered handlers own domain
validation and must fail before mutation. A service serializes dispatch and
publishes audience-specific snapshots. Never register an unguarded engine edit.
"""

from collections.abc import Callable, Mapping
import copy
import dataclasses
import json
import queue
import threading
from typing import Any
import uuid


class OperationError(ValueError):
  """A stable, actionable API error."""

  def __init__(self, code: str, message: str):
    super().__init__(message)
    self.code = code


@dataclasses.dataclass(frozen=True)
class Parameter:
  """Required scalar input; schema and validation come from this definition."""

  kind: str
  description: str
  max_length: int = 8192

  def __post_init__(self):
    if self.kind not in ('string', 'integer'):
      raise ValueError('Only scalar string/integer parameters are supported.')

  def schema(self) -> dict[str, Any]:
    result: dict[str, Any] = {
        'type': self.kind,
        'description': self.description,
    }
    if self.kind == 'string':
      result['maxLength'] = self.max_length
    return result

  def validate(self, name: str, value: Any) -> None:
    valid = (
        isinstance(value, str) and len(value) <= self.max_length
        if self.kind == 'string'
        else isinstance(value, int) and not isinstance(value, bool)
    )
    if not valid:
      raise OperationError('invalid_argument', f'{name}: expected {self.kind}.')


@dataclasses.dataclass(frozen=True)
class Operation:
  """One shared query/mutation definition, not a separate CLI implementation."""

  name: str
  description: str
  parameters: Mapping[str, Parameter]
  handler: Callable[[dict[str, Any]], Any]
  audiences: tuple[str, ...] = ('developer',)
  mutation: bool = False
  audience_handlers: Mapping[str, Callable[[dict[str, Any]], Any]] | None = None

  def invoke(self, audience: str, arguments: dict[str, Any]) -> Any:
    """Select a server-bound handler; arguments cannot select another actor."""
    handler = (self.audience_handlers or {}).get(audience, self.handler)
    return handler(arguments)

  def schema(self) -> dict[str, Any]:
    return {
        'name': self.name,
        'description': self.description,
        'mutation': self.mutation,
        'input': {
            'type': 'object',
            'additionalProperties': False,
            'required': list(self.parameters),
            'properties': {k: v.schema() for k, v in self.parameters.items()},
        },
    }


class OperationService:
  """Authoritative scope/revision/retry ledger shared by attached transports.

  Identity references denote this process lifetime, not restorable checkpoints.
  Successful mutation retries return the original result, before stale checks.
  A key cannot be reused for different input, audience or scope. Failure does
  not consume a key or revision. Retry records last for the service lifetime;
  once capacity is reached reject new mutations (never evict replay protection).
  """

  def __init__(
      self,
      *,
      project_id: str,
      branch_id: str = 'initial',
      session_id: str | None = None,
      retry_capacity: int = 4096,
      audience_resolver: Callable[[str], str] | None = None,
  ):
    self.lock = threading.RLock()
    self.audience_resolver = audience_resolver or (lambda principal: principal)
    self.references = {
        'project_id': project_id,
        'branch_id': branch_id,
        'session_id': session_id or str(uuid.uuid4()),
        'run_id': str(uuid.uuid4()),
    }
    self.revision = 0
    self._operations: dict[str, Operation] = {}
    self._views: dict[str, Callable[[], Any]] = {}
    self._clients: dict[queue.Queue, tuple[str, bool]] = {}
    self._retries: dict[str, tuple[str, dict]] = {}
    self._retry_capacity = retry_capacity
    self._events: list[dict] = []
    self._origin: dict = {}

  def register(self, operation: Operation) -> None:
    if operation.name in self._operations:
      raise ValueError(f'Duplicate operation: {operation.name}')
    self._operations[operation.name] = operation

  def set_view(self, audience: str, view: Callable[[], Any]) -> None:
    self._views[audience] = view

  def discover(self, audience: str) -> dict:
    with self.lock:
      audience = self.audience_resolver(audience)
      return self._envelope({
          'operations': [
              op.schema()
              for op in self._operations.values()
              if audience in op.audiences
          ]
      })

  def _envelope(self, result: Any) -> dict:
    return copy.deepcopy({
        'references': self.references,
        'revision': self.revision,
        'result': result,
    })

  def _resolve_view(self, audience: str) -> Callable[[], Any]:
    audience = self.audience_resolver(audience)
    if audience not in self._views:
      raise OperationError('forbidden', 'No view for this audience.')
    return self._views[audience]

  def snapshot(self, audience: str) -> dict:
    with self.lock:
      return self._envelope(self._resolve_view(audience)())

  def events(self) -> list[dict]:
    with self.lock:
      return copy.deepcopy(self._events)

  def publish(self, event: dict) -> None:
    """Record a domain event; player clients get only their explicit view."""
    with self.lock:
      self.revision += 1
      self._events.append(
          {'revision': self.revision, **self._origin, **copy.deepcopy(event)}
      )
      for client, (audience, notifications_only) in list(self._clients.items()):
        if notifications_only:
          # A current-state reader needs only one pending wakeup. Never
          # materialize a potentially large/private view just to discard it.
          try:
            client.put_nowait(None)
          except queue.Full:
            pass
          continue
        # Retained snapshots let slow clients recover the current state.
        try:
          client.put_nowait(self.snapshot(audience))
        except queue.Full:
          try:
            client.get_nowait()
          except queue.Empty:
            pass  # A concurrent consumer freed the queue.
          client.put_nowait(self.snapshot(audience))

  def subscribe(
      self, audience: str, *, notifications_only: bool = False
  ) -> queue.Queue:
    """Subscribe to snapshots, or coalesced wakeups for a current-state reader.

    By default the queue contains owned snapshot envelopes, as before.
    notifications_only queues at most one None token, including an initial
    wakeup. The reader must call snapshot(audience) at delivery time to resolve
    current authorization and state. Wakeups are not an event history; the
    domain event ledger and revision still retain every published event.
    """
    with self.lock:
      if notifications_only:
        # Validate scope without invoking the view.
        self._resolve_view(audience)
      client = queue.Queue(maxsize=1 if notifications_only else 16)
      client.put(None if notifications_only else self.snapshot(audience))
      self._clients[client] = (audience, notifications_only)
      return client

  def unsubscribe(self, client: queue.Queue) -> None:
    with self.lock:
      self._clients.pop(client, None)

  def dispatch(self, audience: str, request: dict) -> dict:
    """Validate before execution; the listener fixes the audience."""
    with self.lock:
      principal = audience
      audience = self.audience_resolver(principal)
      if not isinstance(request, dict) or set(request) - {
          'operation',
          'arguments',
          'references',
          'revision',
          'retry_key',
      }:
        raise OperationError(
            'invalid_request', 'Use the discovered operation envelope.'
        )
      name = request.get('operation')
      if not isinstance(name, str):
        raise OperationError('invalid_request', 'operation must be a string.')
      op = self._operations.get(name)
      if op is None or audience not in op.audiences:
        raise OperationError(
            'unsupported_operation',
            'Operation unavailable for this capability; use discovery.',
        )
      args = request.get('arguments', {})
      if not isinstance(args, dict) or set(args) != set(op.parameters):
        raise OperationError(
            'invalid_argument',
            'Arguments must match the discovered schema exactly.',
        )
      for name, parameter in op.parameters.items():
        parameter.validate(name, args[name])
      if not op.mutation:
        return self._envelope(op.invoke(audience, copy.deepcopy(args)))
      key = request.get('retry_key')
      if not isinstance(key, str) or not 1 <= len(key) <= 128:
        raise OperationError(
            'retry_key_required',
            'Provide a unique retry_key (1–128 characters).',
        )
      fingerprint = json.dumps(
          [principal, audience, request], sort_keys=True, ensure_ascii=False
      )
      if key in self._retries:
        original, result = self._retries[key]
        if fingerprint != original:
          raise OperationError(
              'retry_conflict', 'Retry key already identifies different input.'
          )
        return copy.deepcopy(result)
      if request.get('references') != self.references:
        raise OperationError(
            'wrong_scope', 'Attach to this session/project/run/branch first.'
        )
      if (
          not isinstance(request.get('revision'), int)
          or isinstance(request.get('revision'), bool)
          or request['revision'] != self.revision
      ):
        raise OperationError(
            'stale_revision',
            'State changed; query current state and review your edit.',
        )
      if len(self._retries) >= self._retry_capacity:
        raise OperationError(
            'retry_capacity',
            'Retry ledger full; export evidence and start a new service.',
        )
      self._origin = {'origin': audience, 'operation_id': key}
      try:
        result = self._envelope(op.invoke(audience, copy.deepcopy(args)))
      finally:
        self._origin = {}
      self._retries[key] = (fingerprint, result)
      return copy.deepcopy(result)
