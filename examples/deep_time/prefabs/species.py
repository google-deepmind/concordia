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
  new triggers. Answers go to `bio.species.<key>.adaptation`; the species
  enacts one by its own action, which the World GM resolves into bounded
  trait changes.
* **Fast act**: the minimal entity's `ConcatActComponent` picks a behaviour.

The genome sheet is computed by the World GM from the planet's traits and
published each step on the species' own topic (`perception.sheet`); the
species reads it through a stock `Subscriber`.
"""

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any, override

from concordia.components.agent import pubsub
from concordia.components.agent import question_of_recent_memories
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.components import pact_mind
from examples.deep_time.prefabs import playable
from examples.deep_time.world import data
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib

SPECIES_PREFAB = 'deep_time_species__Entity'

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


def _senses(
    context: game_lib.GameContext, owner: str
) -> dict[str, perception.Instrument]:
  """A species' (or the grazers') climate and census instruments."""
  return {
      key: perception.instrument(
          context, topic, owner=owner, label=key, initial=initial
      )
      for key, topic, initial in INSTRUMENTS
  }


def senses_rules(
    context: game_lib.GameContext, owner: str, genome: Callable[[], str]
) -> None:
  """The World GM publishes the genome sheet and runs the instruments."""
  perception.publish(context, owner, 'genome', genome)
  for spec in _senses(context, owner).values():
    perception.deliver(context, spec)


def senses(
    context: game_lib.GameContext, owner: str
) -> dict[str, entity_component.ContextComponent]:
  """Genome sheet, climate and census instruments, and biological news."""
  components: dict[str, entity_component.ContextComponent] = {
      'Genome': perception.read(context, owner, 'genome', label='Genome')
  }
  for key, spec in _senses(context, owner).items():
    components[key] = perception.subscribe(context, spec)
  components['Biological news'] = perception.news_digest(
      context,
      ['bio.events'],
      owner=owner,
      label='Biological news (bio.events)',
      initial='Nothing has happened to life yet.',
      size=3,
  )
  return components


def last_shift(context: game_lib.GameContext, name: str) -> str:
  """When the species (or the grazers) last made a heritable shift."""
  tick = actors_lib.made_at(context.planet, f'{name}:adaptation')
  return f' Our last shift was made at tick {tick}.' if tick is not None else ''


def pact_offer(context: game_lib.GameContext, species: str) -> str:
  """The pact awaiting the species' answer, as its pact mind weighs it."""
  symbiosis = context.planet.symbiosis
  proposal = symbiosis.pending_for(species) if symbiosis else None
  if proposal is None or symbiosis is None:
    return ''
  return pact_mind.offer_sheet(
      proposal.text,
      context.planet.species_name(proposal.proposer),
      symbiosis.question,
  )


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
) -> perception.Instrument:
  """The instrument showing what the adaptation mind last decided."""
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
      'randomize_choices': False,
      **context.act_params(name),
  }


@dataclasses.dataclass
class Species(playable.Role):
  """A species: its senses, its slow minds and its actor.

  Its params say all it needs (`name`, `species`, `lore`), so a species
  founded during play is rebuilt from a checkpoint before the world is.
  """

  @override
  def assemble(
      self, context: game_lib.GameContext, params: Mapping[str, Any]
  ) -> Mapping[str, Any]:
    return species_params(
        context,
        str(params['species']),
        str(params['name']),
        str(params['lore']),
    )

  @override
  def rules(
      self, context: game_lib.GameContext, params: Mapping[str, Any]
  ) -> None:
    species_rules(context, str(params['species']), str(params['name']))


def species_config(
    context: game_lib.GameContext, species: str
) -> prefab_lib.InstanceConfig:
  """A species' instance config (plain params; `Species` builds it).

  Args:
    context: Shared game state.
    species: Species key (in `data/species.yaml`, or founded during play).

  Returns:
    The instance config.
  """
  evolution = context.evolution
  if evolution is None:
    raise ValueError('Species need evolution rules in the game context.')
  info = evolution.info(species)
  return prefab_lib.InstanceConfig(
      prefab=SPECIES_PREFAB,
      role=prefab_lib.Role.ENTITY,
      params={'name': info.name, 'species': species, 'lore': info.lore},
  )


def _pact_instruments(
    context: game_lib.GameContext, name: str, species: str
) -> dict[str, perception.Instrument]:
  """The pacts' instruments, where symbiosis is played."""
  if context.planet.symbiosis is None:
    return {}
  return {
      'Pacts': perception.instrument(
          context,
          'bio.pacts',
          owner=name,
          label='Pacts',
          initial='No pacts yet.',
      ),
      # Its pact mind's counsel on an offer; the species answers on its menu.
      'Counsel on the pact': perception.instrument(
          context,
          f'bio.species.{species}.pact',
          owner=name,
          label='Counsel on the pact',
          initial='No pact awaits our answer.',
          render=lambda message: str(message.payload),
      ),
  }


def species_rules(
    context: game_lib.GameContext, species: str, name: str
) -> None:
  """The World GM installs a species' rules.

  Its actor and wake-ups, its genome sheet and instruments, and the offer
  its pact mind weighs.

  Args:
    context: Shared game state.
    species: Species key.
    name: The species' name.
  """
  context.add_actor(
      actors_lib.SpeciesActor(name, species, context.planet),
      hears=(
          'world.tick',
          'bio.events',
          'gaia.omens',
          f'diplomacy.{species}.',
          f'bio.alerts.{species}',
      ),
  )
  if context.wakeups is not None and context.scenario.get('wakeups'):
    context.wakeups.add(name, species, data.load('species.yaml')['wakeups'])
  senses_rules(
      context,
      name,
      lambda: context.planet.genome_text(species) + last_shift(context, name),
  )
  for spec in _pact_instruments(context, name, species).values():
    perception.deliver(context, spec)
  if context.planet.symbiosis is not None:
    perception.publish(
        context, name, 'offer', lambda: pact_offer(context, species)
    )
  perception.deliver(
      context, evolving_toward(context, name, adaptation_topic(species))
  )


def species_params(
    context: game_lib.GameContext, species: str, name: str, lore: str
) -> dict[str, Any]:
  """A species' `minimal__Entity` params, its components included.

  Args:
    context: Shared game state.
    species: Species key.
    name: The species' name.
    lore: What it is, for its instructions.

  Returns:
    The base prefab's params, its components included.
  """
  evolution = context.evolution
  if evolution is None:
    raise ValueError('Species need evolution rules in the game context.')
  slow_model = context.slow()
  components = senses(context, name)
  components['Adaptation mind'] = adaptation_mind(
      context,
      name,
      adaptation_topic(species),
      slow_model,
      evolution.question,
      evolution.latency_ticks,
  )
  for key, spec in _pact_instruments(context, name, species).items():
    components[key] = perception.subscribe(context, spec)
  if context.planet.symbiosis is not None:
    components['Pact mind'] = context.make_mind(
        name,
        'Pact',
        f'bio.species.{species}.pact',
        pact_mind.PactMind(
            slow_model,
            perception.reader(context, name, 'offer'),
            components=['Genome', 'Census'],
        ),
        context.planet.symbiosis.answer_latency,
        trigger_on_observe=True,
    )
  components['Evolving toward'] = perception.subscribe(
      context, evolving_toward(context, name, adaptation_topic(species))
  )
  return entity_params(
      context,
      name,
      'Survive and spread across the bare ground without letting your'
      ' species die out.',
      SPECIES_INSTRUCTIONS.format(name=name, lore=lore),
      components,
  )
