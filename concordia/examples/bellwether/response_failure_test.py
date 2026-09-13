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

"""Unavailable model output is not resident speech, refusal or consent."""

import copy
import json

from concordia.examples.bellwether import game
from concordia.examples.bellwether import game_test
from concordia.examples.bellwether import public_account
import pytest


@pytest.mark.parametrize(
    'response',
    [
        '',
        'PRIVATE_BAD_OUTPUT: I accept everything',
        'null',
        '[]',
        '{"decision":"accept","speech":null}',
        '{"decision":"invented","speech":"Do it."}',
    ],
)
@pytest.mark.parametrize(
    'action,audience',
    [
        ('ask Nell for fuel and part', game.NAMES),
        (
            'message Nell: Are you willing to discuss the request?',
            [game.PLAYER, 'Nell'],
        ),
    ],
)
def test_unreadable_reply_inherits_audience_without_invented_words(
    response, action, audience
):
  world = game_test.world()
  world.resolve(game.PLAYER, action)
  inventory = copy.deepcopy(world.inventory_state())
  world.resolve('Nell', response)
  assert not world.data['commitments']
  assert not world.data['dialogue']
  assert world.inventory_state() == inventory
  notices = [
      e for e in world.data['events'] if e['kind'] == 'response_unavailable'
  ]
  assert len(notices) == 1
  assert set(notices[0]['recipients']) == set(audience)
  assert 'no new decision' in notices[0]['text'].lower()
  assert 'I cannot give a clear commitment' not in json.dumps(world.view())
  for viewer in [*game.NAMES, 'spectator']:
    visible = (
        set(game.NAMES) <= set(audience)
        if viewer == 'spectator'
        else viewer in audience
    )
    assert (
        any(
            e['kind'] == 'response_unavailable'
            for e in world.view(viewer)['journal']
        )
        == visible
    )
    assert 'PRIVATE_BAD_OUTPUT' not in json.dumps(world.view(viewer))
  assert world.data['agenda'] == []


def test_invalid_work_reply_does_not_revoke_previous_consent_or_do_work():
  world = game_test.world()
  game_test.attempt(world, 'ask Ivo to repair')
  before = copy.deepcopy(world.data['commitments'])
  world.resolve(game.PLAYER, 'order repair')
  world.resolve('Ivo', 'INVALID PRIVATE WORDS')
  assert world.data['commitments'] == before
  assert not world.data['repair']
  assert world.inventory_state()['Used']['part'] == 0
  assert len(world.data['dialogue']) == 1


def test_valid_refusal_remains_actual_resident_speech():
  world = game_test.world()
  world.resolve(game.PLAYER, 'ask Nell for fuel and part')
  world.resolve(
      'Nell', '{"decision":"decline","speech":"I decline your request."}'
  )
  assert world.data['dialogue'][-1]['decision'] == 'decline'
  assert world.data['dialogue'][-1]['speech'] == 'I decline your request.'
  assert not any(
      e['kind'] == 'response_unavailable' for e in world.data['events']
  )


def finish_with_missing_dawn_response(world):
  for action in game_test.PRIORITIZE[:-1]:
    game_test.attempt(world, action)
  world.resolve(game.PLAYER, game_test.PRIORITIZE[-1])
  while world.data['agenda']:
    actor = world.next_actor
    world.resolve(
        actor,
        'PRIVATE_DAWN_BAD_OUTPUT'
        if actor == 'Nell'
        else '{"decision":"speak","speech":"Fixture dawn words."}',
    )


def test_dawn_absence_is_explicit_and_public_account_does_not_invent_voice():
  world = game_test.world()
  finish_with_missing_dawn_response(world)
  assert world.finished
  assert world.view()['dawn_responses']['Nell'] is None
  assert world.view()['dawn_responses']['Mara'] == 'Fixture dawn words.'
  assert world.data['epilogue']['services_maintained'] == 6
  assert not any(
      row['actor'] == 'Nell' and row['watch'] == 3
      for row in world.data['dialogue']
  )
  account = public_account.document(world, fixture=True, phase='completed')
  assert any(e['kind'] == 'response_unavailable' for e in account['events'])
  assert 'PRIVATE_DAWN_BAD_OUTPUT' not in json.dumps(account)
  assert 'I cannot give a clear commitment' not in public_account.render_html(
      account
  )
