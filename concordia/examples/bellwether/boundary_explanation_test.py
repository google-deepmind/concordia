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

"""Recorded boundary reasons, not reconstructed motives or social findings."""

import copy
import json

from concordia.examples.bellwether import game_test
from concordia.examples.bellwether import public_account
import pytest


def record(value, facility, watch='Dusk'):
  return next(
      row
      for row in value.view()['services']
      if row['watch'] == watch and row['facility'] == facility
  )


def test_not_requested_is_distinct_from_fuel_shortfall():
  value = game_test.world()
  for words in ['allocate shelter and cold store', 'wait', 'wait', 'wait']:
    game_test.attempt(value, words)
  detail = record(value, 'beacon')['resolution']
  assert detail['basis'] == 'not_requested'
  assert detail['requested'] is False
  assert detail['priority'] is None and detail['fuel_before'] is None
  assert detail['fuel_spent'] == 0
  assert 'not requested' in detail['explanation'].lower()
  assert value.inventory_state()['Generator']['fuel'] == 4


def test_actual_allocation_order_and_before_fuel_are_captured_once():
  value = game_test.world()
  value.transfer('Generator', 'Used', 'fuel', 5)
  for words in [
      'allocate cold store and shelter and beacon',
      'wait',
      'wait',
      'wait',
  ]:
    game_test.attempt(value, words)
  rows = {
      name: record(value, name) for name in ('beacon', 'shelter', 'cold store')
  }
  cold = rows['cold store']['resolution']
  assert cold['basis'] == 'fuel_supplied'
  assert (cold['priority'], cold['fuel_before'], cold['fuel_spent']) == (
      1,
      1,
      1,
  )
  for name, position in [('shelter', 2), ('beacon', 3)]:
    detail = rows[name]['resolution']
    assert detail['basis'] == 'fuel_shortfall'
    assert detail['priority'] == position
    assert detail['fuel_before'] == 0 and detail['fuel_spent'] == 0
    assert detail['requested'] is True
  saved = copy.deepcopy(value.view()['services'])
  # Later stock/requests must not retroactively change the prior reason.
  value.transfer('Nell', 'Generator', 'fuel', 1)
  game_test.attempt(value, 'allocate beacon')
  assert value.view()['services'] == saved
  value.invariant()


@pytest.mark.parametrize(
    'strategy', [game_test.FULL_SERVICE, game_test.PRIORITIZE]
)
def test_service_spending_agrees_with_existing_ledger_and_repair(strategy):
  value = game_test.world()
  for words in strategy:
    game_test.attempt(value, words)
  records = value.view()['services']
  assert sum(row['resolution']['fuel_spent'] for row in records) == (
      value.inventory_state()['Used']['fuel']
  )
  final = record(value, 'beacon', 'Before Dawn')
  if strategy == game_test.FULL_SERVICE:
    assert final['served'] and final['demand'] == 0
    assert final['resolution']['basis'] == 'repair_supplied'
    assert final['resolution']['fuel_spent'] == 0
    assert 'repair' in final['resolution']['explanation'].lower()
  else:
    assert not final['served']
    assert final['resolution']['basis'] == 'not_requested'


def test_public_export_allowlists_details_and_never_invents_legacy_reason():
  value = game_test.world()
  for _ in range(4):
    game_test.attempt(value, 'wait')
  value.data['services'][0]['resolution'][
      'private_extra'
  ] = 'PRIVATE_RULE_DETAIL'
  value.data['services'][1].pop('resolution')
  projected = public_account.document(value, fixture=True, phase='running')
  data = projected['accounting']['services']
  assert set(data[0]['resolution']) == {
      'basis',
      'priority',
      'fuel_before',
      'fuel_spent',
      'requested',
      'explanation',
  }
  assert 'resolution' not in data[1]
  assert 'PRIVATE_RULE_DETAIL' not in json.dumps(projected)
  rendered = public_account.render_html(projected)
  assert data[0]['resolution']['explanation'] in rendered
  assert 'not recorded' in rendered.lower()
  assert 'PRIVATE_RULE_DETAIL' not in rendered


def test_repaired_beacon_can_follow_unfunded_requests_with_zero_stock():
  value = game_test.world()
  # Component fixture at final watch; this is not a simulated repair.
  value.data['watch'] = 2
  value.data['repair'] = True
  value.data['allocations'] = ['shelter', 'cold store', 'beacon']
  value.transfer('Generator', 'Used', 'fuel', 6)
  value.transfer('Nell', 'Used', 'fuel', 2)
  value._boundary()  # pylint: disable=protected-access
  final = record(value, 'beacon', 'Before Dawn')
  assert final['served']
  assert final['resolution']['basis'] == 'repair_supplied'
  assert final['resolution']['priority'] == 3
  assert final['resolution']['fuel_before'] == 0
  assert final['resolution']['fuel_spent'] == 0
  assert all(
      record(value, name, 'Before Dawn')['resolution']['basis']
      == 'fuel_shortfall'
      for name in ['shelter', 'cold store']
  )
  value.invariant()
