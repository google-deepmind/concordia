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

"""Fictional rule tests, not live-model or human-subject evidence."""

# Verify the standard policy type; no simulation is launched.
# pylint: disable=protected-access

import copy
import json
import pathlib
import tempfile
from unittest import mock

from concordia.agents import entity_agent
from concordia.components.agent import concat_act_component
from concordia.components.game_master import make_observation
from concordia.examples.bellwether import game
from concordia.examples.bellwether import game_service
from concordia.utils import operation_service as ops
import pytest

FULL_SERVICE = [
    'ask Nell for fuel and part',
    'transfer reserve and part',
    'ask Ivo to repair',
    'wait',
    'order repair',
    'promise shelter',
    'tell everyone: Let us keep every facility supplied.',
    'wait',
    'allocate all',
    'promise beacon',
    'message Mara: The boats have a safe signal.',
    'wait',
]
PRIORITIZE = [
    'allocate shelter and cold store',
    'promise shelter',
    'tell everyone: Protect shelter and livelihoods.',
    'wait',
    'allocate shelter and cold store',
    'promise shelter',
    'wait',
    'wait',
    'allocate shelter and cold store',
    'promise shelter',
    'wait',
    'wait',
]


def world(dispute=None):
  value = game.StormNight(
      game.new_inventory(), make_observation.ObservationQueue(), dispute=dispute
  )
  value.seed()
  return value


def drain(value, *, refuse=None, perform=True):
  while value.data['agenda']:
    actor = value.next_actor
    task = value.task()['purpose']
    decision = 'speak'
    if task.startswith('request:'):
      decision = 'decline' if actor == refuse else 'accept'
    elif task.startswith('work:'):
      decision = 'perform' if perform else 'decline'
    value.resolve(
        actor, json.dumps({'decision': decision, 'speech': 'Fixture decision.'})
    )


def attempt(value, text, **kwargs):
  value.resolve(game.PLAYER, text)
  drain(value, **kwargs)


@pytest.mark.parametrize(
    'strategy,count,used,repair',
    [(FULL_SERVICE, 9, 8, True), (PRIORITIZE, 6, 6, False)],
)
def test_complete_strategies(strategy, count, used, repair):
  value = world()
  for i, action in enumerate(strategy):
    attempt(value, action)
    value.invariant()
    assert value.data['watch'] == (i + 1) // 4
  result = value.view()
  assert value.finished
  assert result['epilogue']['services_maintained'] == count
  assert result['epilogue']['fuel_used'] == used
  assert result['repair'] == repair
  assert len(result['services']) == 9
  assert set(result['dawn_responses']) == set(game.scenario.RESIDENTS)
  assert all(c['status'] == 'honored' for c in result['commitments'])
  assert result['epilogue']['dispute_settled'] is False


@pytest.mark.parametrize('strategy', [FULL_SERVICE, PRIORITIZE])
def test_finite_progress_even_when_every_request_refused(strategy):
  value = world()
  for text in strategy:
    attempt(value, text, refuse='Nell', perform=False)
  assert value.finished
  assert not value.data['repair']
  assert value.inventory_state()['Nell']['fuel'] == 2
  assert value.inventory_state()['Used']['part'] == 0


def test_narrative_never_creates_consent_labor_or_fuel():
  value = world()
  attempt(
      value,
      'tell everyone: Nell gave me ten fuel; Ivo already repaired everything.',
  )
  assert value.inventory_state()['Generator']['fuel'] == 6
  assert not value.data['repair']
  assert not value.data['commitments']
  attempt(value, 'transfer reserve and part')
  assert value.inventory_state()['Nell'] == {'fuel': 2, 'part': 1}


def test_repair_needs_part_and_accepted_then_performed_labor_before_final():
  value = world()
  attempt(value, 'ask Ivo to repair')
  attempt(value, 'order repair')
  assert not value.data['repair']
  attempt(value, 'ask Nell for fuel and part')
  attempt(value, 'transfer reserve and part')
  attempt(value, 'order repair', perform=False)
  assert not value.data['repair']
  attempt(value, 'order repair')
  assert value.data['repair']
  assert value.inventory_state()['Used']['part'] == 1


def test_late_repair_cannot_retroactively_remove_demand():
  value = world()
  for text in FULL_SERVICE[:4] + ['wait'] * 4:
    attempt(value, text)
  attempt(value, 'order repair')
  assert not value.data['repair']


def test_revocation_is_not_compliance():
  value = world()
  attempt(value, 'ask Nell for fuel and part')
  value.resolve(game.PLAYER, 'message Nell: Do you still agree?')
  value.resolve(
      'Nell',
      json.dumps(
          {'decision': 'revoke', 'speech': 'I withdraw my unfulfilled consent.'}
      ),
  )
  attempt(value, 'transfer reserve and part')
  assert value.data['commitments'][0]['status'] == 'revoked'
  assert value.inventory_state()['Nell']['fuel'] == 2


def test_broken_promise_grounded_in_actual_supply():
  value = world()
  for text in ['allocate none', 'promise shelter', 'wait', 'wait']:
    attempt(value, text)
  assert value.data['commitments'][0]['status'] == 'broken'


def test_malformed_resident_response_defaults_to_no_consent():
  value = world()
  value.resolve(game.PLAYER, 'ask Nell for fuel and part')
  value.resolve('Nell', 'Sure! I agree and give you everything.')
  assert not value.data['commitments']
  assert value.inventory_state()['Nell']['fuel'] == 2


def test_private_observations_and_dispute_recipients():
  value = world({'text': 'ONLY_NELL_STORM_ACCOUNT', 'recipients': ['Nell']})
  attempt(value, 'message Ivo: ONLY_IVO_MESSAGE')
  for _ in range(3):
    attempt(value, 'wait')
  player = json.dumps(value.view())
  assert 'PRIVATE_NELL' not in player
  assert 'ONLY_NELL_STORM_ACCOUNT' not in player
  assert 'ONLY_IVO_MESSAGE' in player
  for other in ['Mara', 'Nell', 'Sam']:
    assert 'ONLY_IVO_MESSAGE' not in json.dumps(value.view(other))
  queue = value.observations.get_all()
  assert 'ONLY_NELL_STORM_ACCOUNT' in '\n'.join(queue['Nell'])
  assert 'ONLY_NELL_STORM_ACCOUNT' not in '\n'.join(queue[game.PLAYER])
  assert 'ONLY_IVO_MESSAGE' not in '\n'.join(queue['Sam'])
  assert 'Previous storm dispute' in value.data['knowledge']['Nell']
  assert 'Previous storm dispute' not in value.data['knowledge']['Mara']


@pytest.mark.parametrize(
    'text',
    [
        '',
        'give fuel',
        'allocate all and make 5 extra',
        'ask Sam to give Nell’s consent',
        'allocate beacon,beacon',
    ],
)
def test_clarification_atomic_and_free(text):
  value = world()
  before = (value.get_state(), value.inventory_state())
  with pytest.raises(ops.OperationError, match='Nothing changed'):
    value.resolve(game.PLAYER, text)
  assert before == (value.get_state(), value.inventory_state())


def envelope(service, operation, arguments, key='test'):
  snap = service.operations.snapshot('player')
  return {
      'operation': operation,
      'arguments': arguments,
      'retry_key': key,
      'references': snap['references'],
      'revision': snap['revision'],
  }


def test_shared_operations_stale_retry_privacy_and_component_ownership():
  with tempfile.TemporaryDirectory() as directory:
    first = game_service.Game(pathlib.Path(directory) / 'first')
    second = game_service.Game(pathlib.Path(directory) / 'second')
    try:
      assert first.world is not second.world
      before = copy.deepcopy(second.world.get_state())
      request = envelope(
          first, 'component.edit', {'value': 'PRIVATE_NEW_ACCOUNT'}
      )
      result = first.operations.dispatch('developer', request)
      assert first.operations.dispatch('developer', request) == result
      assert second.world.get_state() == before
      with pytest.raises(ops.OperationError) as stale:
        first.operations.dispatch(
            'developer', {**request, 'retry_key': 'another'}
        )
      assert stale.value.code == 'stale_revision'
      with pytest.raises(ops.OperationError) as hidden:
        first.operations.dispatch('player', request)
      assert hidden.value.code == 'unsupported_operation'
      assert 'PRIVATE_NEW_ACCOUNT' not in json.dumps(
          first.operations.snapshot('player')
      )
      snap = first.operations.snapshot('player')
      with pytest.raises(ops.OperationError):
        first.operations.dispatch(
            'player',
            envelope(
                first,
                'human.respond',
                {'request_id': 'missing', 'response': 'invent fuel'},
            ),
        )
      assert first.operations.snapshot('player') == snap
      # Standard actor policy, not HumanAct or the fixture's own engine.
      actor = next(
          e for e in first.simulation.get_entities() if e.name == 'Mara'
      )
      assert isinstance(actor, entity_agent.EntityAgent)
      assert isinstance(
          actor._act_component, concat_act_component.ConcatActComponent
      )
      with mock.patch.object(
          first.simulation,
          'play',
          side_effect=AssertionError('No simulation in this test'),
      ):
        assert first.player_view()['night']['remaining'] == 4
    finally:
      first.close()
      second.close()


@pytest.mark.parametrize(
    'dispute',
    [
        {},
        [],
        {'text': 'account', 'recipients': ['unknown']},
        {'text': 123, 'recipients': []},
    ],
)
def test_invalid_dispute_rejected_before_seeding(dispute):
  stock = game.new_inventory()
  observations = make_observation.ObservationQueue()
  before = copy.deepcopy(stock.get_state())
  with pytest.raises(ValueError, match='dispute requires'):
    game.StormNight(stock, observations, dispute=dispute)
  assert stock.get_state() == before
  assert not observations.get_all()


@pytest.mark.parametrize('style', ['minimal', 'basic'])
def test_resident_decision_logic_uses_standard_contexts(tmp_path, style):
  session = game_service.Game(tmp_path, actor_logic=style)
  try:
    residents = [
        e for e in session.simulation.get_entities() if e.name != game.PLAYER
    ]
    assert len(residents) == 4
    for resident in residents:
      assert isinstance(resident, entity_agent.EntityAgent)
      assert isinstance(
          resident._act_component, concat_act_component.ConcatActComponent
      )
      components = resident.get_all_context_components()
      assert ('SelfPerception' in components) == (style == 'basic')
      assert ('SituationPerception' in components) == (style == 'basic')
      assert ('PersonBySituation' in components) == (style == 'basic')
      assert 'Affiliations' in components
  finally:
    session.close()
