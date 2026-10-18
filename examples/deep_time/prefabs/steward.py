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

"""The Steward: the upstream `minimal__Entity` wired to topics (`Steward`).

The Steward perceives the planet only through topic-bound
`pubsub.Subscriber` components ("instruments"), which its prefab (a
`playable.Role`) builds and adds to the minimal entity's; each shows the age
of its reading.

Its **Advisor** is a slow mind: `pubsub.Publisher(component=
QuestionOfRecentMemories(...), skip_if_busy=True)` that reads the same
instruments and writes a briefing to `gaia.briefings`. In sim-time mode the
briefing lands two ticks after it was requested, so the Steward always weighs
advice written about a planet that has since moved on.
"""

# pylint: disable=g-long-ternary

from collections.abc import Mapping
import dataclasses
from typing import Any, override

from examples.deep_time import actors as actors_lib
from examples.deep_time import attention as attention_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.components import advisor_mind
from examples.deep_time.prefabs import playable
from examples.deep_time.world import data
from examples.deep_time.world import publishing
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib

STEWARD_PREFAB = 'deep_time_steward__Entity'
BRIEFINGS_TOPIC = 'gaia.briefings'

STEWARD_INSTRUCTIONS = (
    'You are {name}, the Steward of a living planet: a quiet intelligence that'
    ' can spend omega, a slowly accruing energy, to nudge the world. You see'
    ' the planet only through your instruments, and every reading shows the'
    ' tick it was taken and how old it is. Your advisor thinks slowly: its'
    ' briefings are written about the planet as it was a few ticks ago. You'
    ' may only be able to watch some instruments at once; the others stop'
    ' updating until you watch them again. Spend'
    ' omega when it matters and save it when life can regulate the climate'
    ' by itself.'
)

# (component key, topic, initial belief, carried by instruments) in prompt
# order. The Omega meter is the Steward's own sense, so it is always exact.
INSTRUMENTS = (
    ('Thermometer', 'climate.global', 'No temperature reading yet.', True),
    ('Census', 'bio.census', 'No census yet.', True),
    ('Omega meter', 'gaia.indices', 'No reading yet.', False),
    ('Air gauge', 'cycles.carbon', 'No air reading yet.', True),
    ('Tide gauge', 'ocean.global', 'No ocean reading yet.', True),
)


# The peoples, where there are any: their numbers and how well they eat.
PEOPLES = ('Peoples', 'civ.census', 'No word of the peoples yet.', False)


def _news_topics(context: game_lib.GameContext) -> tuple[str, ...]:
  """The news the Steward hears: the peoples' too where there are any."""
  base = ('geo.events', 'bio.events', 'ocean.alerts')
  return base + (('civ.events',) if context.planet.civ is not None else ())


def advisor_question(
    config: Mapping[str, Any], context: game_lib.GameContext
) -> str:
  """The Advisor's question, listing this scenario's interventions."""
  options = ', '.join(
      i.label.split(' (')[0] for i in context.planet.interventions()
  )
  return str(config['question']).strip().replace('{options}', options)


@dataclasses.dataclass
class Steward(playable.Role):
  """The Steward: its instruments, its advisor and its actor."""

  @override
  def assemble(
      self, context: game_lib.GameContext, params: Mapping[str, Any]
  ) -> Mapping[str, Any]:
    return steward_params(context, str(params['name']))

  @override
  def rules(
      self, context: game_lib.GameContext, params: Mapping[str, Any]
  ) -> None:
    steward_rules(context, str(params['name']))


def steward_config(
    context: game_lib.GameContext, name: str = 'Gaia'
) -> prefab_lib.InstanceConfig:
  """The Steward's instance config (plain params; `Steward` builds it)."""
  del context  # Every game has a Steward of the same make.
  return prefab_lib.InstanceConfig(
      prefab=STEWARD_PREFAB,
      role=prefab_lib.Role.ENTITY,
      params={'name': name},
  )


def _instruments(
    context: game_lib.GameContext,
    name: str,
    attention: attention_lib.Attention | None = None,
) -> dict[str, perception.Instrument]:
  """The Steward's instruments, by component key (the advisor's briefing last).

  Args:
    context: Shared game state.
    name: Entity name.
    attention: Its attention (for the World GM, which runs them).

  Returns:
    Their specs.
  """
  specs = {
      key: perception.instrument(
          context,
          topic,
          owner=name,
          label=key,
          initial=initial,
          render=(
              perception.with_earlier_readings(
                  context.planet, publishing.renderer_for(topic)
              )
              if topic == 'climate.global'
              else publishing.render_peoples
              if key == PEOPLES[0]
              else None
          ),
          transported=transported,
          attention=attention if transported else None,
      )
      for key, topic, initial, transported in _readings(context)
  }
  specs['Advisor briefing'] = perception.instrument(
      context,
      BRIEFINGS_TOPIC,
      owner=name,
      label='Advisor briefing',
      initial='The advisor has not reported yet.',
  )
  return specs


def _readings(
    context: game_lib.GameContext,
) -> tuple[tuple[str, str, str, bool], ...]:
  """The Steward's readings: its instruments, and the peoples where any."""
  return INSTRUMENTS + (
      # Where there are peoples, their plight is seen from afar (like their
      # towns): at once, and without taking an instrument's attention.
      (PEOPLES,)
      if context.planet.civ is not None
      else ()
  )


def steward_rules(context: game_lib.GameContext, name: str) -> None:
  """The World GM installs the Steward's rules.

  Its actor and attention, its instruments (late, lossy, noisy) and the
  almanac its Advisor weighs.

  Args:
    context: Shared game state.
    name: Entity name.
  """
  watchable = [
      key for key, _, _, transported in _readings(context) if transported
  ]
  attention = attention_lib.Attention(watchable, context.attention_slots)
  context.attention[name] = attention
  context.add_actor(
      actors_lib.StewardActor(name, context.planet, attention),
      hears=('world.tick', *_news_topics(context)),
  )
  instruments = _instruments(context, name, attention)
  for key, spec in instruments.items():
    if key != 'Advisor briefing':
      perception.deliver(context, spec)
  perception.publish(context, name, 'almanac', context.planet.almanac)
  perception.deliver(context, instruments['Advisor briefing'])


def steward_params(context: game_lib.GameContext, name: str) -> dict[str, Any]:
  """The Steward's `minimal__Entity` params, its components included.

  Args:
    context: Shared game state.
    name: Entity name.

  Returns:
    The base prefab's params.
  """
  instruments = _instruments(context, name)
  components: dict[str, entity_component.ContextComponent] = {
      key: perception.subscribe(context, spec)
      for key, spec in instruments.items()
      if key != 'Advisor briefing'
  }
  components['World news'] = perception.news_digest(
      context,
      ['geo.events', 'bio.events', 'ocean.alerts'],
      owner=name,
      label='World news (geo.events, bio.events, ocean.alerts)',
      initial='Nothing unusual has happened.',
  )
  if context.planet.civ is not None:  # Their own digest: no crowding out.
    components['Peoples news'] = perception.news_digest(
        context,
        ['civ.events'],
        owner=name,
        label='News of the peoples (civ.events)',
        initial='Nothing has happened among the peoples.',
    )
  advisor_config = data.load('advisor.yaml')
  advisor = advisor_mind.AdvisorMind(
      almanac=perception.reader(context, name, 'almanac'),
      model=context.slow(),
      pre_act_label='Advisor',
      question=advisor_question(advisor_config, context),
      answer_prefix='',
      add_to_memory=False,
      components=list(components),
      terminators=(),
      num_memories_to_retrieve=5,
  )
  components['Advisor'] = context.make_mind(
      name,
      'Advisor',
      BRIEFINGS_TOPIC,
      advisor,
      int(advisor_config['latency_ticks']),
  )
  components['Advisor briefing'] = perception.subscribe(
      context, instruments['Advisor briefing']
  )
  components = context.with_bus(components)  # The bus is restored first.
  return {
      'name': name,
      'goal': (
          context.scenario.get('steward_goal', '')
          or (
              'Keep the global temperature inside the habitable band until'
              ' the end of the scenario without letting the daisies die'
              ' out.'
          )
      ),
      'custom_instructions': STEWARD_INSTRUCTIONS.format(name=name),
      'extra_components': components,
      'extra_components_index': {
          key: 2 + i for i, key in enumerate(components)
      },
      'randomize_choices': False,
      **context.act_params(name),
  }
