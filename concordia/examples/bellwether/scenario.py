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

"""Bellwether prefab scaffold. All entity state lives in standard components."""

import copy
import json
from typing import Any

from concordia.components.agent import constant
from concordia.components.agent import human_act_component
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting
from concordia.components.game_master import switch_act
from concordia.prefabs.entity import minimal
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib

PLAYER = 'Coordinator'
GM = 'Bellwether fixture'
ACCOUNT = 'PreviousStormAccount'
RESOLUTION = (
    'Fixture receipt: the coordinator’s request was recorded. '
    'No resident has accepted a commitment; no fuel, labor or spare '
    'part has moved. Resident negotiation and watch resolution are '
    'not implemented in this one-turn fixture.'
)
PUBLIC = {
    'title': 'Last Light at Bellwether',
    'watch': 'Dusk',
    'locations': [
        'Harbor beacon',
        'Storm shelter',
        'Cooperative cold store',
        'Generator yard',
    ],
    'cast': [
        {'name': 'Mara', 'role': 'Harbor master', 'location': 'Harbor beacon'},
        {
            'name': 'Nell',
            'role': 'Cooperative custodian',
            'location': 'Cooperative cold store',
        },
        {
            'name': 'Ivo',
            'role': 'Engineer · cooperative member',
            'location': 'Generator yard',
        },
        {
            'name': 'Sam',
            'role': 'Shelter steward · cooperative member',
            'location': 'Storm shelter',
        },
    ],
    'opening': (
        'Rain crosses the harbor. You are temporary emergency coordinator. '
        'The beacon, shelter and cold store each need fuel tonight. '
        'Begin with a proposal or wait; inspection is free.'
    ),
}
INITIAL = {
    'watch': 'Dusk',
    'watches': ['Dusk', 'High Tide', 'Before Dawn'],
    'generator_fuel': 6,
    'reserve_owner': 'Nell',
    'reserve_fuel': 2,
    'total_fuel': 8,
    'demand_per_facility_per_watch': 1,
    'baseline_demand': 9,
    'actions_per_watch': 4,
    'repair': {
        'spare_part_owner': 'Nell',
        'labor_required': 'Ivo',
        'completed': False,
        'final_beacon_demand_reduction': 1,
    },
    'commitments': [],
    'transfers': [],
    'services': [],
    'harbor_charter': (
        'Harbor Office represents boat safety; it cannot compel cooperative'
        ' consent.'
    ),
    'cooperative_charter': (
        'Members decide reserve use; authority, notice and acceptance are'
        ' separate.'
    ),
}
RESIDENTS = {
    'Mara': (
        'Protect boats and harbor safety.',
        'The cooperative failed to help us last storm.',
    ),
    'Nell': (
        'Protect members’ livelihoods.',
        (
            'PRIVATE_NELL: two reserve fuel and a spare part; I dispute Mara’s'
            ' account.'
        ),
    ),
    'Ivo': (
        'Repair safely, with an accepted labor commitment.',
        'PRIVATE_IVO: no labor commitment yet.',
    ),
    'Sam': (
        'Represent displaced residents.',
        'PRIVATE_SAM: shelter residents need representation.',
    ),
}


class Resident(prefab_lib.Prefab):
  """Reuse the minimal actor build; each build owns its own memory component."""

  description = 'Bellwether resident using the standard minimal prefab.'

  def build(self, model, memory_bank, **kwargs):
    params: dict[str, Any] = dict(self.params)
    account = params.pop('account', '')
    params['extra_components'] = {
        ACCOUNT: constant.Constant(
            account, pre_act_label='Previous storm account'
        )
    }
    return minimal.Entity(params=params).build(model, memory_bank, **kwargs)


class FixtureGameMaster(prefab_lib.Prefab):
  """Standard SwitchAct + fixed context components; no alternative engine."""

  description = 'Fixed-response fixture, not a live negotiation model.'

  def build(self, model, memory_bank):
    world = copy.deepcopy(self.params['world'])
    extra = {
        switch_act.DEFAULT_NEXT_ACTING_COMPONENT_KEY: (
            next_acting.NextActingInFixedOrder([PLAYER])
        ),
        switch_act.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
            next_acting.FixedActionSpec(
                entity_lib.free_action_spec(
                    call_to_action='What do you propose, {name}?'
                )
            )
        ),
        switch_act.DEFAULT_TERMINATE_COMPONENT_KEY: constant.Constant(
            'No', pre_act_label=''
        ),
        switch_act.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY: constant.Constant(
            GM, pre_act_label=''
        ),
        switch_act.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
            make_observation.MakeObservation(
                model,
                player_names=[PLAYER, *RESIDENTS],
                allow_llm_fallback=False,
            )
        ),
        switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: constant.Constant(
            RESOLUTION, pre_act_label=''
        ),
        'InitialMechanics': constant.Constant(
            json.dumps(world), pre_act_label='Fixture mechanics'
        ),
    }
    params: dict[str, Any] = {
        'name': GM,
        'custom_instructions': 'Fixture only; no social prediction.',
        'extra_components': extra,
    }
    return minimal.Entity(params=params).build(
        model,
        memory_bank,
        act_component=switch_act.SwitchAct(model, entity_names=[PLAYER]),
    )


def configuration(reader):
  """A fresh Config of trusted prefab objects; input is runtime-only wiring."""

  class HumanCoordinator(prefab_lib.Prefab):
    description = 'Human coordinator with standard minimal context.'

    def build(self, model, memory_bank):
      def acting_policy(order):
        return human_act_component.HumanActComponent(
            reader, component_order=order
        )

      return minimal.Entity(params=self.params).build(
          model, memory_bank, act_component_factory=acting_policy
      )

  instances = [
      prefab_lib.InstanceConfig(
          prefab='coordinator',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': PLAYER,
              'custom_instructions': (
                  'You coordinate emergency response; you cannot supply others’'
                  ' consent.'
              ),
          },
      )
  ]
  for name, (goal, account) in RESIDENTS.items():
    instances.append(
        prefab_lib.InstanceConfig(
            prefab='resident',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': name,
                'goal': goal,
                'account': account,
                'custom_instructions': f'You are {name}. {goal}',
            },
        )
    )
  gm_params: dict[str, Any] = {'name': GM, 'world': copy.deepcopy(INITIAL)}
  instances.append(
      prefab_lib.InstanceConfig(
          prefab='fixture',
          role=prefab_lib.Role.GAME_MASTER,
          params=gm_params,
      )
  )
  return prefab_lib.Config(
      default_premise=PUBLIC['opening'],
      default_max_steps=1,
      prefabs={
          'coordinator': HumanCoordinator(),
          'resident': Resident(),
          'fixture': FixtureGameMaster(),
      },
      instances=instances,
  )
