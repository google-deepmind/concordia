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

"""Unit checks of standard batch contracts, not live simulations."""

import copy
from itertools import permutations
from unittest import mock

from concordia.agents import entity_agent
from concordia.components.game_master import event_resolution
from concordia.environment.engines import simultaneous
from concordia.examples.bellwether import council
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.typing import entity as entity_lib
import numpy as np
import pytest


def balances(ledger):
  return {
      name: ledger.stock.get_player_inventory(name)
      for name in (*council.MEMBERS, 'Community')
  }


@pytest.mark.parametrize('order', list(permutations(council.MEMBERS)))
def test_complete_batch_order_independence_and_retry(order):
  _, ledger = council.configuration()
  choices = {'Nell': 'contribute', 'Ivo': 'keep', 'Sam': 'contribute'}
  raw = '\n'.join(f'{name}: {choices[name]}' for name in order)
  result = ledger.resolve_batch(raw)
  assert balances(ledger) == {
      'Nell': {'fuel': 0},
      'Ivo': {'fuel': 1},
      'Sam': {'fuel': 0},
      'Community': {'fuel': 2},
  }
  assert ledger.resolve_batch(raw) == result
  assert sum(x['fuel'] for x in balances(ledger).values()) == 3
  before = copy.deepcopy(balances(ledger))
  ledger.pre_observe(event_resolution.PUTATIVE_EVENT_TAG + ' Nell: keep')
  with pytest.raises(ValueError):
    ledger.pre_act(
        entity_lib.ActionSpec(
            call_to_action='Resolve', output_type=entity_lib.OutputType.RESOLVE
        )
    )
  assert balances(ledger) == before


@pytest.mark.parametrize(
    'raw',
    [
        '',
        'Nell: contribute\nIvo: keep',
        'Nell: contribute\nIvo: keep\nNell: contribute',
        'Nell: contribute\nIvo: keep\nUnknown: contribute',
        'Nell: speak for everyone\nIvo: keep\nSam: contribute',
    ],
)
def test_incomplete_or_invalid_batch_never_transfers_partially(raw):
  _, ledger = council.configuration()
  before = (ledger.get_state(), balances(ledger))
  with pytest.raises(ValueError):
    ledger.resolve_batch(raw)
  assert (ledger.get_state(), balances(ledger)) == before


def test_actual_standard_engine_selects_three_specs_before_resolution():
  config, ledger = council.configuration(
      human_readers={'Nell': lambda request: 'keep'}
  )
  engine = simultaneous.Simultaneous()
  with mock.patch.object(
      generic.Simulation, 'play', side_effect=AssertionError('No launch')
  ):
    sim = generic.Simulation(
        config,
        no_language_model.NoLanguageModel(),
        lambda text: np.zeros(8),
        engine=engine,
    )
    actors, specs = engine.next_acting(
        sim.get_game_masters()[0], sim.get_entities()
    )
    assert [a.name for a in actors] == list(council.MEMBERS)
    assert all(
        spec.output_type == entity_lib.OutputType.CHOICE
        and spec.options == council.CHOICES
        for spec in specs
    )
    for actor in actors:
      assert isinstance(actor, entity_agent.EntityAgent)
      assert type(actor.get_act_component()).__name__ == (
          'HumanActComponent' if actor.name == 'Nell' else 'ConcatActComponent'
      )
    assert ledger.done is False
    assert balances(ledger)['Community']['fuel'] == 0


def test_fresh_inventory_components_and_state_roundtrip():
  config, one = council.configuration()
  other_config, two = council.configuration()
  assert config is not other_config and one.stock is not two.stock
  one.resolve_batch('Nell: keep\nIvo: contribute\nSam: keep')
  assert balances(two)['Ivo']['fuel'] == 1
  two.set_state(one.get_state())
  assert two.get_state() == one.get_state()
  # Component serialization alone does not restore inventory or the engine.
  assert balances(two)['Community']['fuel'] == 0


def test_unknown_human_role_rejected():
  with pytest.raises(ValueError):
    council.configuration(human_readers={'Coordinator': lambda r: 'keep'})
