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

"""Recipe construction/accounting tests, not social validation or live runs."""

import copy
from unittest import mock

from concordia.agents import entity_agent
from concordia.environment.engines import sequential
from concordia.examples.bellwether import game
from concordia.examples.bellwether import game_prefab
from concordia.examples.bellwether import researcher
from concordia.prefabs.simulation import generic
import numpy as np
import pytest


@pytest.mark.parametrize('name', researcher.RECIPES)
def test_build_each_recipe_with_standard_policies_and_private_accounts(name):
  case = researcher.prepare_case(name, lambda request: 'wait')
  with mock.patch.object(
      generic.Simulation, 'play', side_effect=AssertionError('No run')
  ):
    simulation = generic.Simulation(
        case.config,
        game_prefab.FixtureModel(),
        lambda text: np.zeros(8),
        engine=sequential.Sequential(),
    )
  actors: dict[str, entity_agent.EntityAgent] = {}
  for actor in simulation.get_entities():
    assert isinstance(actor, entity_agent.EntityAgent)
    actors[actor.name] = actor
  assert (
      type(actors['Coordinator'].get_act_component()).__name__
      == 'HumanActComponent'
  )
  assert all(
      type(actors[n].get_act_component()).__name__ == 'ConcatActComponent'
      for n in game.scenario.RESIDENTS
  )
  for name in game.scenario.RESIDENTS:
    params = next(
        i.params for i in case.config.instances if i.params['name'] == name
    )
    assert (
        actors[name].get_component('PreviousStormAccount').get_state()['state']
        == params['account']
    )
    entries = case.world.view(name)['journal']
    assert any(
        x['kind'] == 'private_memory' and x['text'] == params['account']
        for x in entries
    )
  assert all(
      x['kind'] != 'private_memory'
      for x in case.world.view('Coordinator')['journal']
  )
  assert all(
      x['kind'] != 'private_memory'
      for x in case.world.view('spectator')['journal']
  )
  case.world.invariant()


def test_scarcity_is_declared_preconsumption_not_deleted_material():
  case = researcher.prepare_case('mutual-aid', lambda request: 'wait')
  assert case.world.inventory_state()['Generator']['fuel'] == 4
  assert case.world.inventory_state()['Nell']['fuel'] == 2
  assert case.world.inventory_state()['Used']['fuel'] == 2
  assert case.manifest['available_fuel_at_start'] == 6
  assert case.manifest['fuel_consumed_before_play'] == 2
  opening = case.world.view()['journal'][0]['text']
  assert 'Four fuel' in opening and 'BEFORE play' in opening
  assert game.OPENING not in opening
  case.world.invariant()


def test_institution_prose_is_not_consent_and_views_match_actor_context():
  case = researcher.prepare_case('resource-governance', lambda request: 'wait')
  simulation = generic.Simulation(
      case.config,
      game_prefab.FixtureModel(),
      lambda text: np.zeros(8),
      engine=sequential.Sequential(),
  )
  nell = next(e for e in simulation.get_entities() if e.name == 'Nell')
  assert isinstance(nell, entity_agent.EntityAgent)
  affiliation = nell.get_component('Affiliations').get_state()['state']
  assert isinstance(affiliation, str)
  assert 'Proposed charter' in affiliation
  assert 'Proposed charter' in case.world.view()['institutions'][1]['rule']
  before = case.world.inventory_state()
  case.world.resolve('Coordinator', 'transfer reserve and part')
  assert case.world.inventory_state() == before
  assert not case.world.data['commitments']


def test_private_dispute_recipient_recipe_is_delivered_at_boundary():
  case = researcher.prepare_case(
      'institutional-dispute', lambda request: 'wait'
  )
  for _ in range(4):
    case.world.resolve('Coordinator', 'wait')
  for name in ['Mara', 'Nell']:
    assert any(
        x['kind'] == 'disputed_account'
        for x in case.world.view(name)['journal']
    )
  for name in ['Coordinator', 'Ivo', 'Sam', 'spectator']:
    assert not any(
        x['kind'] == 'disputed_account'
        for x in case.world.view(name)['journal']
    )


@pytest.mark.parametrize('logic', ['minimal', 'basic'])
def test_fresh_ownership_and_human_actor_injection(logic):
  one = researcher.prepare_case(
      'resource-governance',
      lambda r: 'wait',
      actor_logic=logic,
      human_readers={'Nell': lambda r: '{}'},
  )
  two = researcher.prepare_case(
      'resource-governance', lambda r: 'wait', actor_logic=logic
  )
  models = []
  for case in [one, two]:
    models.append(
        generic.Simulation(
            case.config,
            game_prefab.FixtureModel(),
            lambda t: np.zeros(8),
            engine=sequential.Sequential(),
        )
    )
  first, second = [
      {e.name: e for e in model.get_entities()} for model in models
  ]
  assert type(first['Nell'].get_act_component()).__name__ == 'HumanActComponent'
  assert (
      type(second['Nell'].get_act_component()).__name__ == 'ConcatActComponent'
  )
  assert first['Nell'].get_component('__memory__') is not second[
      'Nell'
  ].get_component('__memory__')
  one.world.institutions[1]['rule'] = 'Only this run'
  assert 'Only this run' not in str(two.world.get_state())
  assert 'Only this run' not in str(game.INSTITUTIONS)
  assert one.world.observations is not two.world.observations


@pytest.mark.parametrize(
    'bad',
    [
        None,
        {'Mara': 'partial'},
        {**{n: '' for n in game.scenario.RESIDENTS}, 'intruder': 'text'},
    ],
)
def test_invalid_seed_atomic(bad):
  world = game.StormNight(
      game.new_inventory(), game_prefab.make_observation.ObservationQueue()
  )
  before = copy.deepcopy(world.get_state())
  # None is the default; a malformed opening must still reject atomically.
  with pytest.raises(ValueError):
    world.seed(opening=None if bad is None else 'intro', accounts=bad)
  assert world.get_state() == before


def test_invalid_institution_restore_atomic_and_roundtrip():
  one = researcher.prepare_case('resource-governance', lambda r: 'wait').world
  two = researcher.prepare_case('bellwether', lambda r: 'wait').world
  two.set_state(one.get_state())
  assert two.get_state() == one.get_state()
  before = two.get_state()
  invalid = copy.deepcopy(before)
  invalid['institutions'][0]['members'] = ['unknown']
  with pytest.raises(ValueError):
    two.set_state(invalid)
  assert two.get_state() == before


@pytest.mark.parametrize(
    'mapping', [{'Unknown': lambda r: 'wait'}, {'Nell': None}, ['Nell']]
)
def test_unknown_or_noncallable_human_mapping_rejected(mapping):
  with pytest.raises(ValueError):
    researcher.prepare_case(
        'bellwether', lambda r: 'wait', human_readers=mapping
    )
