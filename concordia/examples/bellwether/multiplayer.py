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

"""Two controlled roles in the existing Bellwether night, not a new engine.

Only the local host approves browsers. A visitor cannot choose its authorized
role, inspect another inbox or send another actor's response. Other residents
retain standard prefabs and their normal acting policies.
"""

import copy
import json

from concordia.examples.astral_canticle import human_io
from concordia.examples.bellwether import game
from concordia.examples.bellwether import game_service
from concordia.utils import browser_sessions
from concordia.utils import operation_service as ops


class SharedGame(game_service.Game):
  """Human Coordinator and Nell; AI Mara/Ivo/Sam; read-only public spectator."""

  player_audiences = ('role:Coordinator', 'role:Nell')

  def __init__(self, output, *, secure=True, cookie_path='/', **kwargs):
    self.sessions = browser_sessions.BrowserSessions(
        (game.PLAYER, 'Nell', 'spectator'),
        secure=secure,
        cookie_path=cookie_path,
    )
    self.nell_inbox = human_io.HumanSession(
        on_request=self._requested, initial_status='Waiting for your turn'
    )
    self.response_handlers = {
        'role:' + role: lambda args, role=role: self.respond_for(role, args)
        for role in (game.PLAYER, 'Nell')
    }
    self.view_handlers = {
        'role:' + role: lambda _, role=role: self.view_for(role)
        for role in (game.PLAYER, 'Nell')
    }
    super().__init__(output, human_readers={'Nell': self.nell_inbox}, **kwargs)
    self.operations.audience_resolver = self.sessions.audience
    self.operations.set_view('visitor', lambda: {'lobby': True})
    for role in (game.PLAYER, 'Nell', 'spectator'):
      self.operations.set_view(
          'role:' + role, lambda role=role: self.view_for(role)
      )

  def _register(self):
    super()._register()
    for name, description, handler in (
        (
            'session.approve',
            (
                'Approve a browser only after confirming the participant with'
                ' the host.'
            ),
            self._approve,
        ),
        (
            'session.revoke',
            (
                'Revoke a browser role; retain the pending human turn for'
                ' rejoining.'
            ),
            self._revoke,
        ),
    ):
      self.operations.register(
          ops.Operation(
              name,
              description,
              {
                  'request_id': ops.Parameter(
                      'string',
                      'Pending browser ID from host inspection (not a'
                      ' credential)',
                  )
              },
              handler,
              mutation=True,
          )
      )

  def _approve(self, args):
    result = self.sessions.approve(args['request_id'])
    self.operations.publish({'kind': 'browser.approved', **result})
    return result

  def _revoke(self, args):
    result = self.sessions.revoke(args['request_id'])
    self.operations.publish({'kind': 'browser.revoked'})
    return result

  def developer_view(self):
    return {
        **super().developer_view(),
        'join_requests': self.sessions.pending(),
    }

  def _start(self, args):
    if not self.sessions.ready((game.PLAYER, 'Nell')):
      raise ops.OperationError(
          'players_not_ready',
          'The host must approve both players before the night begins.',
      )
    return super()._start(args)

  def view_for(self, role):
    state = super().player_view()
    state['night'] = self.world.view(role)
    state['role'] = role
    state['multiplayer'] = True
    state['players_ready'] = self.sessions.ready((game.PLAYER, 'Nell'))
    if role == 'spectator':
      state['human'] = {
          'pending': None,
          'entries': [],
          'finished': self.phase == 'completed',
          'status': 'Watching public events',
      }
    elif role == 'Nell':
      state['human'] = self.nell_inbox.snapshot()
      if state['human']['pending']:
        task = self.world.task()
        state['task'] = copy.deepcopy(task)
        state['human']['pending']['prompt'] = task.get(
            'purpose', 'Your response'
        )
        purpose = task.get('purpose', '')
        state['decisions'] = ['speak', 'counter', 'revoke']
        if purpose.startswith('request:'):
          state['decisions'] = ['accept', 'decline', 'counter']
        if purpose.startswith('dawn'):
          state['decisions'] = ['speak']
    # No other actor's pending ID, context, task, log or private journal enters
    # this view. The UI receives a projection, not hidden developer data.
    return state

  def respond_for(self, role, args):
    if self.world.next_actor != role:
      raise ops.OperationError(
          'wrong_turn', 'Wait for your own turn; nothing changed.'
      )
    if role == game.PLAYER:
      return super()._respond(args)
    try:
      decision, _ = game.parse_resident_response(args['response'])
      json.loads(
          args['response']
      )  # Human replies must be one exact JSON object.
      purpose = self.world.task().get('purpose', '')
      allowed = {'speak', 'counter', 'revoke'}
      if purpose.startswith('request:'):
        allowed = {'accept', 'decline', 'counter'}
      if purpose.startswith('dawn'):
        allowed = {'speak'}
      if decision not in allowed:
        raise ValueError('Choose an available decision for this turn.')
      changed = self.nell_inbox.submit(args['request_id'], args['response'])
    except (ValueError, TypeError, AttributeError) as error:
      raise ops.OperationError('invalid_action', str(error)) from error
    if changed:
      self.operations.publish({'kind': 'human.accepted', 'actor': role})
    return {'accepted': changed}

  def _finish(self):
    self.nell_inbox.finish(
        'Dawn has come.'
        if not self._failure
        else 'The night stopped; your journal is retained.'
    )
    super()._finish()

  def close(self):
    self.nell_inbox.finish('Service closed.')
    super().close()
