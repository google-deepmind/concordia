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

"""Species: `minimal__Entity` with instruments and a slow adaptation mind.

Each species is three cognition tiers on stock components:

* **Instruments** (beliefs): TopicBus-bound `pubsub.Subscriber`s for the
  climate, the census and biological news.
* **Slow mind**: `pubsub.Publisher(component=QuestionOfRecentMemories(...),
  skip_if_busy=True)` asks which heritable shift would help. In sim-time mode
  its scheduler is a `LatencyScheduler`, so the answer lands a fixed number of
  ticks later ("evolution takes as long as it takes"); a busy mind ignores
  new triggers. Answers go to `bio.species.<key>.adaptation`, where the World
  GM turns them into bounded trait changes.
* **Fast act**: the minimal entity's `ConcatActComponent` picks a behaviour.

The genome sheet is a stock `ReportFunction` over the planet's traits.
"""

from collections.abc import Callable, Mapping
from typing import Any

from concordia.components.agent import pubsub
from concordia.components.agent import question_of_recent_memories
from concordia.components.agent import report_function
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.components import pact_mind
from examples.deep_time.prefabs import playable
from examples.deep_time.world import data
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib

SPECIES_PREFAB = playable.PLAYABLE_PREFAB

SPECIES_INSTRUCTIONS = (
    'You are {name}, a species on a young planet. {lore} You cannot'
    ' think in words the way animals do; your choices are how your'
    ' populations grow, and over generations, which traits you evolve. You'
    ' sense the world only through the readings below, which may be several'
    ' ticks old.'
)

INSTRUMENTS = (
    ('Climate sense', 'climate.global', 'No sense of the climate yet.'),
    ('Census', 'bio.census', 'No census yet.'),
)


def adaptation_topic(species: str) -> str:
  return f'bio.species.{species}.adaptation'


def senses(
    context: game_lib.GameContext,
    owner: str,
    genome: Callable[[], str],
) -> dict[str, entity_component.ContextComponent]:
  """Genome sheet, climate and census instruments, and biological news."""
  components: dict[str, entity_component.ContextComponent] = {
      'Genome': report_function.ReportFunction(
          function=genome, pre_act_label='Genome'
      )
  }
  for key, topic, initial in INSTRUMENTS:
    components[key] = perception.instrument(
        context, topic, owner=owner, label=key, initial=initial
    )
  components['Biological news'] = perception.news_digest(
      context,
      ['bio.events'],
      owner=owner,
      label='Biological news (bio.events)',
      initial='Nothing has happened to life yet.',
      size=3,
  )
  return components


def adaptation_mind(
    context: game_lib.GameContext,
    owner: str,
    topic: str,
    slow_model: language_model.LanguageModel,
    question: str,
    latency_ticks: int,
) -> pubsub.Publisher:
  """The slow mind that decides which heritable shift to make."""
  mind = question_of_recent_memories.QuestionOfRecentMemories(
      model=slow_model,
      pre_act_label='Adaptation mind',
      question=question,
      answer_prefix='',
      add_to_memory=False,
      components=['Genome', 'Climate sense', 'Census'],
      terminators=(),
      num_memories_to_retrieve=10,
  )
  return context.make_mind(owner, 'Adaptation', topic, mind, latency_ticks)


def evolving_toward(
    context: game_lib.GameContext, owner: str, topic: str
) -> pubsub.Subscriber:
  """The belief showing what the adaptation mind last decided."""
  return perception.instrument(
      context,
      topic,
      owner=owner,
      label='Evolving toward (slow mind)',
      initial='Instinct only: no adaptation has been decided yet.',
  )


def entity_params(
    context: game_lib.GameContext,
    name: str,
    goal: str,
    instructions: str,
    components: Mapping[str, entity_component.ContextComponent],
) -> dict[str, Any]:
  """`minimal__Entity` params with the components after the instructions."""
  components = context.with_bus(components)
  return {
      'name': name,
      'goal': goal,
      'custom_instructions': instructions,
      'extra_components': dict(components),
      'extra_components_index': {
          key: 2 + i for i, key in enumerate(components)
      },
      'measurements': context.measurements,
      'randomize_choices': False,
      **context.act_params(name),
  }


def species_config(
    context: game_lib.GameContext,
    species: str,
    slow_model: language_model.LanguageModel,
) -> prefab_lib.InstanceConfig:
  """Registers a species actor and returns its entity instance config.

  Args:
    context: Shared game state.
    species: Species key in `data/species.yaml`.
    slow_model: Model for the adaptation mind.

  Returns:
    The instance config for a `minimal__Entity`.
  """
  evolution = context.evolution
  if evolution is None:
    raise ValueError('Species need evolution rules in the game context.')
  info = evolution.info(species)
  context.add_actor(
      actors_lib.SpeciesActor(info.name, species, context.planet),
      hears=(
          'world.tick',
          'bio.events',
          'gaia.omens',
          f'diplomacy.{species}.',
          f'bio.alerts.{species}',
      ),
  )
  if context.wakeups is not None and context.scenario.get('wakeups'):
    context.wakeups.add(
        info.name, species, data.load('species.yaml')['wakeups']
    )

  components = senses(
      context, info.name, lambda: context.planet.genome_text(species)
  )
  components['Adaptation mind'] = adaptation_mind(
      context,
      info.name,
      adaptation_topic(species),
      slow_model,
      evolution.question,
      evolution.latency_ticks,
  )
  if context.planet.symbiosis is not None:
    components['Pacts'] = perception.instrument(
        context,
        'bio.pacts',
        owner=info.name,
        label='Pacts',
        initial='No pacts yet.',
    )
    components['Pact mind'] = context.make_mind(
        info.name,
        'Pact',
        f'bio.species.{species}.pact',
        pact_mind.PactMind(
            slow_model, context, species, components=['Genome', 'Census']
        ),
        context.planet.symbiosis.answer_latency,
        trigger_on_observe=True,
    )
  components['Evolving toward'] = evolving_toward(
      context, info.name, adaptation_topic(species)
  )
  return prefab_lib.InstanceConfig(
      prefab=SPECIES_PREFAB,
      role=prefab_lib.Role.ENTITY,
      params=entity_params(
          context,
          info.name,
          'Survive and spread across the bare ground without letting your'
          ' species die out.',
          SPECIES_INSTRUCTIONS.format(name=info.name, lore=info.lore),
          components,
      ),
  )
