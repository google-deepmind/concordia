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

"""The Chronicler: a narrator that never blocks the world.

Built from stock components only:

* memory, `ObservationToMemory` and `LastNObservations`: it remembers every
  tick headline and world event it observes;
* a `pubsub.Publisher(component=QuestionOfRecentMemories(...),
  skip_if_busy=True)` that writes the next chapter to the `chronicle` topic,
  landing a few ticks after it starts ("history is written in hindsight");
* the previous chapter as a topic `Subscriber`, for continuity;
* `puppet_act.PuppetActComponent` with a fixed response, so acting each tick
  (which is what triggers the slow mind) costs no language model call.

The upstream `puppet__Entity` prefab cannot take extra components, hence this
small game prefab.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import puppet_act
from concordia.components.agent import question_of_recent_memories
from concordia.contrib.pubsub import topics
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.world import data
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib

CHRONICLE_TOPIC = 'chronicle'
CHRONICLER_PREFAB = 'deep_time_chronicler__Entity'

INSTRUCTIONS = (
    'You are {name}, the Chronicler of a young planet. You write its history'
    ' in short chapters for the people who will one day live there. You'
    ' record what happened, not what should have happened.'
)


def _instruments(
    context: game_lib.GameContext, name: str
) -> dict[str, perception.Instrument]:
  """What the Chronicler reads: the thermometer and its previous chapter."""
  return {
      'Thermometer': perception.instrument(
          context,
          'climate.global',
          owner=name,
          label='Thermometer',
          initial='No reading yet.',
      ),
      'Previous chapter': perception.instrument(
          context,
          CHRONICLE_TOPIC,
          owner=name,
          label='Previous chapter',
          initial='This is the first chapter.',
      ),
  }


@dataclasses.dataclass
class Chronicler(prefab_lib.Prefab):
  """An entity whose only output is a slow chronicle on a topic."""

  description: str = (  # pyrefly: ignore[bad-override]
      'Deep Time Chronicler: remembers world events and writes chapters to'
      ' the chronicle topic through a background pubsub.Publisher.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {'name': 'Chronicler'}
  )
  # The game (a shared service: copying the prefab keeps it).
  context: game_lib.GameContext | None = None

  def rules(
      self, context: game_lib.GameContext, params: Mapping[str, Any]
  ) -> None:
    """The World GM installs the Chronicler's rules: its actor, its readings."""
    name = str(params.get('name', 'Chronicler'))
    context.add_actor(
        actors_lib.ChroniclerActor(name),
        hears=(
            'world.tick',
            'geo.events',
            'bio.events',
            'ocean.alerts',
            'gaia.omens',
            'world.resolutions',
        ),
    )
    for spec in _instruments(context, name).values():
      perception.deliver(context, spec)

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the Chronicler.

    Args:
      model: Model for the puppet act (never used: the response is fixed).
      memory_bank: The Chronicler's memory bank.

    Returns:
      The Chronicler entity.
    """
    name = self.params.get('name', 'Chronicler')
    context = self.context
    if context is None:
      raise ValueError('The Chronicler needs the game context.')
    slow_model = context.slow()
    config = data.load('chronicler.yaml')

    components: dict[str, entity_component.ContextComponent] = {
        # The bus first, so a checkpoint restores it before any belief.
        topics.DEFAULT_TOPIC_BUS_COMPONENT_KEY: topics.TopicBusComponent(
            context.bus
        ),
        agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: (
            agent_components.memory.AssociativeMemory(memory_bank=memory_bank)
        ),
        'Instructions': agent_components.constant.Constant(
            state=INSTRUCTIONS.format(name=name),
            pre_act_label='Role',
        ),
        'ObservationToMemory': (
            agent_components.observation.ObservationToMemory()
        ),
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: (
            agent_components.observation.LastNObservations(history_length=20)
        ),
        **{
            key: perception.subscribe(context, spec)
            for key, spec in _instruments(context, name).items()
        },
    }
    chapter_writer = question_of_recent_memories.QuestionOfRecentMemories(
        model=slow_model,
        pre_act_label='Chapter',
        question=str(config['question']).strip(),
        answer_prefix='',
        add_to_memory=False,
        components=['Thermometer', 'Previous chapter'],
        terminators=(),
        num_memories_to_retrieve=12,
    )
    components['Chronicle mind'] = context.make_mind(
        name,
        'Chronicle',
        CHRONICLE_TOPIC,
        chapter_writer,
        int(config['latency_ticks']),
    )
    act_component = puppet_act.PuppetActComponent(
        model=model,
        fixed_responses={
            actors_lib.CHRONICLER_CALL_TO_ACTION: actors_lib.CHRONICLER_RESPONSE
        },
    )
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components,
    )


def chronicler_config(
    context: game_lib.GameContext, name: str = 'Chronicler'
) -> prefab_lib.InstanceConfig:
  """The Chronicler's instance config (plain params; `Chronicler` builds it)."""
  del context  # One Chronicler a game.
  return prefab_lib.InstanceConfig(
      prefab=CHRONICLER_PREFAB,
      role=prefab_lib.Role.ENTITY,
      params={'name': name},
  )
