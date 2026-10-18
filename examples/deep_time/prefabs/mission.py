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

"""Mission Control: home's link to the colony, over light-delay streams.

An upstream `minimal__Entity` whose instruments are `pubsub.Subscriber`s on
the colony's topics carried over a contrib `operators.transport` with the
light delay as latency: every reading it sees is years old, and says so.
What it transmits goes out on `mission.uplink`; the colonists hear it over
the same delay, and the colony applies designs and directives on arrival.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any, override

from concordia.contrib.pubsub import operators
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.prefabs import playable
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib

MISSION_PREFAB = 'deep_time_mission__Entity'
MISSION_INSTRUCTIONS = (
    'You are {name}, the agency at home that founded a colony four'
    ' light-years away. Light is the fastest thing there is: what you hear'
    ' from the colony is four years old, and whatever you transmit, a design'
    ' or a directive, arrives four years from now. Send the colonists the'
    ' designs they lack and directives that will still make sense when they'
    ' arrive; do not flood them.'
)


def light_delay(context: game_lib.GameContext) -> operators.Transport:
  """The link between home and the colony: latency of the light delay."""
  colony = context.planet.colony
  assert colony is not None
  return operators.Transport(latency=float(colony.light_delay))


def in_flight(context: game_lib.GameContext) -> str:
  """What home has sent that has not arrived, and what the colony lacks."""
  colony = context.planet.colony
  assert colony is not None
  sent = {what for _, kind, what in colony.incoming if kind == 'design'}
  lacking = [
      label
      for key, label in colony.design_labels.items()
      if key not in colony.designs and key not in sent
  ]
  if colony.incoming:
    parts = [
        (
            f'a survey of where to mine the ice (arrives in year {arrival})'
            if kind == 'site'
            else f'{kind} "{what}" (arrives in year {arrival})'
        )
        for arrival, kind, what in colony.incoming
    ]
    text = 'On its way: ' + '; '.join(parts) + '.'
  else:
    text = 'Nothing is on its way to the colony.'
  if lacking:
    text += (
        ' The colony was founded without the designs for '
        + ', '.join(lacking)
        + '; only home can send them. Its solar panels wear out a little'
        ' every year.'
    )
  return text


@dataclasses.dataclass
class Mission(playable.Role):
  """Mission Control: its light-delayed instruments and its actor."""

  @override
  def assemble(
      self, context: game_lib.GameContext, params: Mapping[str, Any]
  ) -> Mapping[str, Any]:
    return mission_params(context, str(params['name']))


def mission_config(
    context: game_lib.GameContext, name: str = 'Mission Control'
) -> prefab_lib.InstanceConfig:
  """Mission Control's instance config (plain params; `Mission` builds it)."""
  del context  # One Mission Control a game.
  return prefab_lib.InstanceConfig(
      prefab=MISSION_PREFAB,
      role=prefab_lib.Role.ENTITY,
      params={'name': name},
  )


def mission_params(context: game_lib.GameContext, name: str) -> dict[str, Any]:
  """Registers Mission Control's actor and returns its entity params."""
  # Mission Control thinks in its act; no slow mind yet.
  context.add_actor(actors_lib.MissionActor(name, context.planet))
  components: dict[str, entity_component.ContextComponent] = {
      'Colony relay': perception.instrument(
          context,
          'colony.status',
          owner=name,
          label='Colony relay',
          initial='No word from the colony has reached home yet.',
          transported=True,
          channel=lambda: light_delay(context),
      ),
      'Colony news': perception.instrument(
          context,
          'colony.events',
          owner=name,
          label='Colony news',
          initial='No news from the colony has reached home yet.',
          transported=True,
          channel=lambda: light_delay(context),
      ),
      'Transmissions in flight': perception.sheet(
          context,
          name,
          'in_flight',
          lambda: in_flight(context),
          label='\nTransmissions in flight',
      ),
  }
  components = context.with_bus(components)  # The bus is restored first.
  return {
      'name': name,
      'goal': context.scenario.get('steward_goal', ''),
      'custom_instructions': MISSION_INSTRUCTIONS.format(name=name),
      'extra_components': components,
      'extra_components_index': {
          key: 2 + i for i, key in enumerate(components)
      },
      'measurements': context.measurements,
      'randomize_choices': False,
      **context.act_params(name),
  }
