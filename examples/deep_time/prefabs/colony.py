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

"""The colonists: a C5 entity on the contrib async-deliberation prefab.

They read their own colony's status fresh (it is published where they are)
and their own sheet, deliberate in the background on the slow model, and
act each generation on the fast model.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any, override

from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.prefabs import mission as mission_prefab
from examples.deep_time.prefabs import playable
from examples.deep_time.prefabs import polity as polity_prefab
from examples.deep_time.world import publishing
from concordia.typing import prefab as prefab_lib

SUBSCRIPTIONS = {'Colony status': 'colony.status'}
DEFAULT_GOAL = (
    'Keep every colonist fed and breathing, and warm this world until it is'
    ' habitable.'
)


COLONY_PREFAB = 'deep_time_colony__Entity'


@dataclasses.dataclass
class Colony(playable.RolePolity):
  """The colonists: a people of the async-deliberation make, far away."""

  @override
  def assemble(
      self, context: game_lib.GameContext, params: Mapping[str, Any]
  ) -> Mapping[str, Any]:
    return colony_params(context, str(params['name']))

  @override
  def rules(
      self, context: game_lib.GameContext, params: Mapping[str, Any]
  ) -> None:
    del params  # One colony a game.
    colony_rules(context)


def colony_config(context: game_lib.GameContext) -> prefab_lib.InstanceConfig:
  """The colonists' instance config (plain params; `Colony` builds it)."""
  colony = context.planet.colony
  if colony is None:
    raise ValueError('Colonists need a colony in the scenario.')
  return prefab_lib.InstanceConfig(
      prefab=COLONY_PREFAB,
      role=prefab_lib.Role.ENTITY,
      params={'name': f'Colonists of {colony.name}'},
  )


# Home's transmissions, as they reach the colonists: (label, topic, initial).
MESSAGES = (
    'Messages from home',
    'mission.uplink',
    'No message from home has reached us yet.',
)


def _messages(
    context: game_lib.GameContext, name: str
) -> perception.Instrument:
  """Home's transmissions, heard years after they were sent."""
  label, topic, _ = MESSAGES
  return perception.instrument(
      context,
      topic,
      owner=name,
      label=label,
      transported=True,
      channel=lambda: mission_prefab.light_delay(context),
  )


def colony_rules(context: game_lib.GameContext) -> None:
  """The World GM installs the colonists' rules: actor, sheet, messages."""
  colony = context.planet.colony
  if colony is None:
    raise ValueError('Colonists need a colony in the scenario.')
  rules = context.scenario['colony']
  name = f'Colonists of {colony.name}'
  context.add_actor(
      actors_lib.ColonyActor(
          name,
          context.planet,
          share=float(rules.get('focus_share', 0.15)),
          plan_share=float(rules.get('plan_share', 0.4)),
      ),
      # The colony's news and their own decrees, not its status payload
      # (that is their 'Colony status' reading, rendered).
      hears=('world.tick', 'colony.events', 'colony.decrees'),
  )
  perception.publish(context, name, 'colony', colony.sheet)
  perception.deliver(context, _messages(context, name))


def colony_params(context: game_lib.GameContext, name: str) -> dict[str, Any]:
  """The colonists' entity params, their components included."""
  rules = context.scenario['colony']
  label, topic, initial = MESSAGES
  return {
      'name': name,
      'goal': rules.get('goal', DEFAULT_GOAL),
      'bus': context.bus,
      'subscriptions': dict(SUBSCRIPTIONS),
      'renderers': {
          topic: publishing.renderer_for(topic)
          for topic in SUBSCRIPTIONS.values()
      },
      'clock_topic': 'world.tick',
      'plan_topic': 'colony.plan',
      'sim_scheduler': context.sim,
      'deliberation_latency_ticks': int(rules.get('plan_latency_ticks', 2)),
      'deliberation_model': context.slow(),
      'randomize_choices': False,
      **context.act_params(name),
      'extra_components': {
          'Our colony': perception.read(
              context, name, 'colony', label='\nOur colony'
          ),
          label: perception.subscribe(
              context, perception.reading(name, label, initial, topic=topic)
          ),
      },
  }
