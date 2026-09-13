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

"""Public setup provenance, not private configuration or replay identity."""

import copy
import json
from unittest import mock

from concordia.examples.bellwether import game_service
from concordia.examples.bellwether import public_account
from concordia.examples.bellwether import researcher
import pytest


@pytest.mark.parametrize('recipe', researcher.RECIPES)
def test_service_export_keeps_declared_setup_not_private_configuration(
    tmp_path, recipe
):
  with mock.patch(
      'concordia.prefabs.simulation.generic.Simulation.play',
      side_effect=AssertionError('No simulation'),
  ):
    value = game_service.Game(tmp_path, recipe=recipe, actor_logic='basic')
    try:
      value.case.manifest['private_note'] = 'PRIVATE_SETUP_SENTINEL'
      value.world.emit('private_message', 'PRIVATE_SETUP_SENTINEL', ['Nell'])
      before = value.operations.snapshot('developer')
      request = {
          'operation': 'game.public_account',
          'arguments': {'format': 'json'},
      }
      exported = value.operations.dispatch('player', request)['result']
      data = json.loads(exported['content'])
      assert 'not a complete run configuration' in data['setup_notice']
      setup = data['declared_setup']
      assert setup['recipe'] == recipe
      assert setup['actor_logic'] == 'basic'
      assert setup['human_roles'] == ['Coordinator']
      assert setup['engine'].endswith('.sequential.Sequential')
      assert setup['fuel_consumed_before_play'] == (
          2 if recipe == 'mutual-aid' else 0
      )
      assert setup['available_fuel_at_start'] == (
          6 if recipe == 'mutual-aid' else 8
      )
      assert 'PRIVATE_SETUP_SENTINEL' not in exported['content']
      assert 'private_note' not in setup
      assert (
          value.operations.dispatch('developer', request)['result'] == exported
      )
      assert (
          value.operations.dispatch('role:spectator', request)['result']
          == exported
      )
      assert value.operations.snapshot('developer') == before
      saved = copy.deepcopy(setup)
      value.world.transfer('Generator', 'Used', 'fuel', 1)
      assert (
          json.loads(
              value.operations.dispatch('player', request)['result']['content']
          )['declared_setup']
          == saved
      )
    finally:
      value.close()


def test_world_only_document_does_not_infer_setup_from_events():
  case = researcher.prepare_case('mutual-aid', lambda request: 'wait')
  data = public_account.document(case.world, fixture=True, phase='paused')
  assert data['declared_setup'] is None
  text = public_account.render_html(data)
  assert 'Setup provenance was not supplied' in text
  assert 'not a complete run configuration' in text


def test_projection_owns_public_lists_and_escapes_setup_text():
  case = researcher.prepare_case('bellwether', lambda request: 'wait')
  case.manifest['human_roles'] = ['Coordinator <img src=x> & public label']
  data = public_account.document(
      case.world, fixture=True, phase='paused', manifest=case.manifest
  )
  data['declared_setup']['human_roles'].append('Nell')
  assert len(case.manifest['human_roles']) == 1
  rendered = public_account.render_html(data)
  assert '<img' not in rendered
  assert '&lt;img src=x&gt;' in rendered
  assert 'model settings are not exported' in rendered
