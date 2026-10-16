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
biological news), a genome sheet (`ReportFunction`) and a slow adaptation mind
(`pubsub.Publisher(QuestionOfRecentMemories)`, four-tick latency) whose
answers go to `bio.species.grazers.adaptation`. The grazers' choice of diet
decides whether they crop white or black petals, which warms or cools the
planet. They also hear the Steward's omens.
"""

from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time.prefabs import playable
from examples.deep_time.prefabs import species as species_lib
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

GRAZER_PREFAB = playable.PLAYABLE_PREFAB
ADAPTATION_TOPIC = species_lib.adaptation_topic('grazers')

INSTRUCTIONS = (
    'You are {name}, a clade of small grazing animals on a young planet. {lore}'
    ' Your choices are how your herds feed and move, and over generations,'
    ' which traits you evolve. You sense the world only through the readings'
    ' below, which may be several ticks old.'
)


def grazers_config(
    context: game_lib.GameContext,
    slow_model: language_model.LanguageModel,
) -> prefab_lib.InstanceConfig:
  """Registers the grazer actor and returns its entity instance config."""
  grazers = context.planet.grazers
  if grazers is None:
    raise ValueError('This scenario has no grazers.')
  context.add_actor(
      actors_lib.GrazerActor(grazers.name, context.planet),
      hears=('world.tick', 'bio.events', 'gaia.omens'),
  )
  components = species_lib.senses(context, grazers.name, grazers.genome_text)
  components['Adaptation mind'] = species_lib.adaptation_mind(
      context,
      grazers.name,
      ADAPTATION_TOPIC,
      slow_model,
      grazers.question,
      grazers.latency_ticks,
  )
  components['Evolving toward'] = species_lib.evolving_toward(
      context, grazers.name, ADAPTATION_TOPIC
  )
  return prefab_lib.InstanceConfig(
      prefab=GRAZER_PREFAB,
      role=prefab_lib.Role.ENTITY,
      params=species_lib.entity_params(
          context,
          grazers.name,
          'Keep your herds fed and growing without eating out the meadows'
          ' that feed them.',
          INSTRUCTIONS.format(name=grazers.name, lore=grazers.lore),
          components,
      ),
  )
