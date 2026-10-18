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

"""Grazers: a herbivore clade on `minimal__Entity`.

The same three tiers as the daisies: ageing instruments (climate, census,
biological news), a genome sheet (`perception.sheet`) and a slow adaptation mind
(`pubsub.Publisher(QuestionOfRecentMemories)`, four-tick latency) whose
answers go to `bio.species.grazers.adaptation`. The grazers' choice of diet
decides whether they crop white or black petals, which warms or cools the
planet. They also hear the Steward's omens.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any, override

from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.prefabs import playable
from examples.deep_time.prefabs import species as species_lib
from concordia.typing import prefab as prefab_lib

GRAZER_PREFAB = 'deep_time_grazers__Entity'
ADAPTATION_TOPIC = species_lib.adaptation_topic('grazers')

INSTRUCTIONS = (
    'You are {name}, a clade of small grazing animals on a young planet. {lore}'
    ' Your choices are how your herds feed and move, and over generations,'
    ' which traits you evolve. You sense the world only through the readings'
    ' below, which may be several ticks old.'
)


@dataclasses.dataclass
class Grazers(playable.Role):
  """The grazers: their senses, their adaptation mind and their actor."""

  @override
  def assemble(
      self, context: game_lib.GameContext, params: Mapping[str, Any]
  ) -> Mapping[str, Any]:
    return grazers_params(context, params)

  @override
  def rules(
      self, context: game_lib.GameContext, params: Mapping[str, Any]
  ) -> None:
    del params  # One clade a game.
    grazers_rules(context)


def grazers_config(context: game_lib.GameContext) -> prefab_lib.InstanceConfig:
  """The grazers' instance config (plain params; `Grazers` builds it).

  Their static settings are read once from the game: who they are, and what
  their adaptation mind asks and how long it takes.
  """
  grazers = context.planet.grazers
  if grazers is None:
    raise ValueError('This scenario has no grazers.')
  return prefab_lib.InstanceConfig(
      prefab=GRAZER_PREFAB,
      role=prefab_lib.Role.ENTITY,
      params={
          'name': grazers.name,
          'lore': grazers.lore,
          'question': grazers.question,
          'latency_ticks': grazers.latency_ticks,
      },
  )


def grazers_rules(context: game_lib.GameContext) -> None:
  """The World GM installs the grazers' rules: actor, sheet, instruments."""
  grazers = context.planet.grazers
  if grazers is None:
    raise ValueError('This scenario has no grazers.')
  context.add_actor(
      actors_lib.GrazerActor(grazers.name, context.planet),
      hears=('world.tick', 'bio.events', 'gaia.omens'),
  )
  species_lib.senses_rules(
      context,
      grazers.name,
      lambda: grazers.genome_text()
      + species_lib.last_shift(context, grazers.name),
  )
  perception.deliver(
      context,
      species_lib.evolving_toward(context, grazers.name, ADAPTATION_TOPIC),
  )


def grazers_params(
    context: game_lib.GameContext, params: Mapping[str, Any]
) -> dict[str, Any]:
  """The grazers' `minimal__Entity` params, their components included."""
  name = str(params['name'])
  components = species_lib.senses(context, name)
  components['Adaptation mind'] = species_lib.adaptation_mind(
      context,
      name,
      ADAPTATION_TOPIC,
      context.slow(),
      str(params['question']),
      int(params['latency_ticks']),
  )
  components['Evolving toward'] = perception.subscribe(
      context, species_lib.evolving_reading(name, ADAPTATION_TOPIC)
  )
  return species_lib.entity_params(
      context,
      name,
      'Keep your herds fed and growing without eating out the meadows'
      ' that feed them.',
      INSTRUCTIONS.format(name=name, lore=params['lore']),
      components,
  )
