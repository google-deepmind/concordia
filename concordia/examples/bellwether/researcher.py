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

"""Trusted teaching recipes; no model calls or simulation on import/build.

These analogous cases keep Bellwether's fixed roster/action vocabulary and
physical accounting. They are not general-purpose project serialization or
empirically validated social models. See RESEARCHER.md for extension boundaries.
"""

from collections.abc import Mapping
import copy
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from concordia.components.game_master import make_observation
from concordia.examples.bellwether import game
from concordia.examples.bellwether import game_prefab
from concordia.examples.bellwether import scenario
from concordia.typing import prefab

RECIPES = (
    'bellwether',
    'mutual-aid',
    'resource-governance',
    'institutional-dispute',
)


@dataclass
class Case:
  """One build's config/world, never shared between runs or checkpoint forks."""

  config: prefab.Config
  world: game.StormNight
  manifest: dict[str, Any]
  opening: str = game.OPENING


def prepare_case(
    name,
    reader,
    *,
    actor_logic='minimal',
    human_readers=None,
    dispute=None,
    action_model=None
):
  """Return standard Config + owned world, ready for generic.Simulation.

  Values are declared in this trusted module, not loaded as Python from user
  documents. Each call creates its own Inventory, ObservationQueue, GM rules
  and prefab parameters. Human readers are runtime objects, never serialized.
  """
  if name not in RECIPES:
    raise ValueError('Choose a supported recipe: ' + ', '.join(RECIPES))
  if dispute is not None and name != 'bellwether':
    raise ValueError(
        'Custom dispute is supported only with the bellwether recipe'
    )
  if not callable(reader):
    raise ValueError('reader must be a runtime HumanInput callable')
  if human_readers is not None and (
      not isinstance(human_readers, Mapping)
      or any(name not in scenario.RESIDENTS for name in human_readers)
      or any(not callable(value) for value in human_readers.values())
  ):
    raise ValueError('human_readers must map known resident names to callables')
  accounts = {n: v[1] for n, v in scenario.RESIDENTS.items()}
  goals = {n: v[0] for n, v in scenario.RESIDENTS.items()}
  institutions = copy.deepcopy(game.INSTITUTIONS)
  dispute = copy.deepcopy(game.DEFAULT_DISPUTE if dispute is None else dispute)
  opening = game.OPENING
  preconsumed = 0
  if name == 'mutual-aid':
    preconsumed = 2
    opening = (
        'Teaching scenario: mutual aid after a ferry evacuation. Four fuel'
        ' remain in the generator, two in Nell’s reserve, and two were used'
        ' BEFORE play. One spare part is with Nell. Three watches, three'
        ' facilities, four choices each; baseline demand is nine.'
        ' Fixed Bellwether names and mechanics remain. A timely consented'
        ' repair can save one unit, so some unmet demand is unavoidable.'
    )
    goals['Sam'] = 'Represent newly displaced families and seek mutual aid.'
    accounts['Sam'] = (
        'Private initial assumption: I have heard an unverified report that'
        ' some evacuees need warm accommodation. I must not present rumor as'
        ' fact.'
    )
  elif name == 'resource-governance':
    opening += (
        ' Teaching variation: the cooperative is debating a transparent reserve'
        ' reporting charter. Stated rules are not proof of member compliance.'
    )
    institutions[1]['rule'] = (
        'Proposed charter: report reserve requests to members and hear'
        ' objections. Nell still controls release; Ivo still controls his'
        ' labor.'
    )
    institutions[1]['enforcement'] = (
        'The proposal invites notice and deliberation. No new automatic veto,'
        ' voting rule or sanction is implemented. Existing consent gates apply.'
    )
    goals['Nell'] = (
        'Consider a transparent reserve policy while protecting members.'
    )
  elif name == 'institutional-dispute':
    opening += (
        ' Teaching variation: the previous-storm accounts are delivered only'
        ' to Mara and Nell at High Tide. The coordinator is not omniscient.'
    )
    dispute = {
        'text': (
            'A disputed note: Mara recalls a promise of boat support; Nell'
            ' recalls an unresolved request. These are incompatible claims,'
            ' not an adjudicated historical fact.'
        ),
        'recipients': ['Mara', 'Nell'],
    }
    accounts['Mara'] = (
        'Private initial account: I recall a promise; my recollection may be'
        ' wrong.'
    )
    accounts['Nell'] = (
        'Private initial account: I recall no agreement; my recollection may be'
        ' wrong.'
    )

  world = game.StormNight(
      game.new_inventory(),
      make_observation.ObservationQueue(),
      institutions=institutions,
      dispute=dispute,
  )
  config = game_prefab.configuration(
      reader,
      world,
      actor_logic=actor_logic,
      human_readers=human_readers,
      action_model=action_model,
  )
  instances = []
  for instance in config.instances:
    params = dict(instance.params)
    actor = params['name']
    if actor in accounts:
      params['account'] = accounts[actor]
      params['goal'] = goals[actor]
      if name != 'bellwether':
        params['custom_instructions'] += (
            ' This teaching variation asks you to consider: ' + goals[actor]
        )
    instances.append(replace(instance, params=params))
  config = replace(config, instances=instances)
  world.seed(opening=opening, accounts=accounts)
  if preconsumed:
    world.transfer('Generator', 'Used', 'fuel', preconsumed)
    world.emit(
        'initial_condition',
        'Declared initial condition: two fuel were already consumed before'
        ' play. Subtract two from cumulative Used when reporting in-run'
        ' consumption.',
    )
  return Case(
      config,
      world,
      {
          'recipe': name,
          'actor_logic': actor_logic,
          'human_roles': [game.PLAYER, *(human_readers or {})],
          'engine': 'concordia.environment.engines.sequential.Sequential',
          'fuel_total_including_preconsumed': 8,
          'fuel_consumed_before_play': preconsumed,
          'available_fuel_at_start': 8 - preconsumed,
          'fixed_roster': list(game.NAMES),
          'fixed_facilities': list(game.FACILITIES),
          'fixed_enforcement': (
              'owner consent, explicit labor, conservation and watch boundaries'
          ),
          'evidence_class': (
              'fictional teaching configuration, not empirical evidence'
          ),
      },
      opening,
  )
