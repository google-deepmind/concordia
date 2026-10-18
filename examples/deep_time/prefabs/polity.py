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

"""Polities: peoples built on the contrib async-deliberation prefab.

A polity is the contrib `basic_with_async_deliberation__Entity` (through
`playable.PlayablePolity`, which only lets a person play it) configured with:

* beliefs on the TopicBus: the census of the peoples, the weather, the
  chronicle of civilization and the market, each shown with its age;
* a grand strategy: the stock `Plan` in a background `Publisher` on the slow
  model, landing a few ticks after it starts and published on
  `civ.<key>.plan`, where rivals could one day subscribe to it;
* a storehouse sheet (a stock `ReportFunction` over the ledger and market);
* a fast act that each tick puts more labor into one recipe, learns a tech,
  or places a standing order on the market (`actors.PolityActor`).
"""

# pylint: disable=g-long-ternary

from collections.abc import Mapping
from typing import Any

from concordia.components.agent import constant
from concordia.components.agent import report_function
from concordia.contrib.pubsub import operators
from examples.deep_time import actors as actors_lib
from examples.deep_time import crises as crises_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.components import pact_mind
from examples.deep_time.components import policy_compiler
from examples.deep_time.components import press_mind
from examples.deep_time.components import relay_mind
from examples.deep_time.components import take_mind
from examples.deep_time.prefabs import playable
from examples.deep_time.world import data
from examples.deep_time.world import publishing
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

POLITY_PREFAB = playable.PLAYABLE_POLITY_PREFAB
PREFAB = playable.PlayablePolity

SUBSCRIPTIONS = {
    'Census of the peoples': 'civ.census',
    'Weather': 'climate.global',
    'Chronicle of civilization': 'civ.events',
}
# A people's character sheet, as its mind reads it.
CHARACTER_LINES = (
    ('who', 'Who they are'),
    ('wants', 'What they want'),
    ('fears', 'What they fear'),
    ('grievances', 'Old grievances'),
    ('believes', 'What they believe'),
    ('voice', 'How they speak'),
)
DEFAULT_GOAL = (
    'Keep your people fed through every generation and forge iron into tools'
    ' so that your children work less and live better.'
)


def storehouse(context: game_lib.GameContext, polity: str) -> str:
  """The polity's own sheet: people, stocks, labor, techs, silver, orders."""
  civ = context.planet.civ
  assert civ is not None
  text = civ.sheet(polity)
  if context.planet.trade is not None:
    text += ' ' + context.planet.trade.standing(polity)
  if context.automation is not None:
    text += ' ' + context.automation.describe(polity)
  treaties = context.planet.treaties
  if treaties is not None:
    offer = treaties.pending_for(polity)
    if offer is not None and civ.polities[polity].name in context.humans:
      text += f' An offer awaits our answer: {offer.text}'
    for kind, spec in treaties.kinds.items():
      bound = [civ.polities[k].name for k in treaties.bound(polity, kind)]
      if bound:
        text += (
            f' We are bound by {spec.plural} with the {", ".join(bound)}: at'
            f' most {spec.cap:g} {spec.unit} a generation.'
        )
  gazette = context.planet.gazette
  if gazette is not None:
    text += (
        f' Our credibility in the Gazette: {gazette.credibility(polity):g};'
        f" our miners' consent for coal: {civ.polities[polity].consent:.0%}."
    )
  for aim in context.scenario['civilization'].get('aims', ()):
    text += ' ' + civ.aim(polity, str(aim))
  # What the age asks of the peoples, in this people's numbers (a playbook
  # line: only for coached minds).
  if context.coached:
    text += crises_lib.shared_words(
        crises_lib.shared_goal(context.planet, polity)
    )
  # Trouble the people can see coming, as its page shows it.
  actor = context.actors.get(civ.polities[polity].name)
  options = list(actor.action_spec().options) if actor is not None else []
  name = civ.polities[polity].name
  text += crises_lib.words(
      crises_lib.polity_crises(
          context.planet, polity, options, context.pushes(options, name)
      )
  )
  return text


def character_words(character: Mapping[str, Any]) -> str:
  """A character sheet in words: who a people is, wants, fears, believes."""
  return ' '.join(
      f'{label}: {" ".join(str(character[key]).split()).rstrip(".")}.'
      for key, label in CHARACTER_LINES
      if character.get(key)
  )


def polity_config(
    context: game_lib.GameContext,
    polity: str,
    slow_model: language_model.LanguageModel,
) -> prefab_lib.InstanceConfig:
  """Registers a polity actor and returns its entity instance config.

  Args:
    context: Shared game state (its planet must have a civilization).
    polity: The polity's key in the scenario.
    slow_model: Model for the grand strategy.

  Returns:
    The instance config for the contrib async-deliberation prefab.
  """
  civ = context.planet.civ
  if civ is None:
    raise ValueError('Polities need a civilization in the scenario.')
  people = civ.polities[polity]
  rules = context.scenario['civilization']
  subscriptions = dict(SUBSCRIPTIONS)
  if context.planet.treaties is not None:
    subscriptions['Treaties'] = 'civ.treaties'
  if context.planet.gazette is not None:
    subscriptions['Gazette'] = 'media.gazette'
  # Where the presses retell the Steward's omens, every people reads them.
  owner = rules.get('press_owner') if context.planet.gazette else None
  if owner:
    subscriptions['The Gazette on the omens'] = 'media.omens'
  # The people written as people, and the world's conventions, if given.
  character = (rules.get('characters') or {}).get(polity)
  conventions = str(rules.get('conventions') or '').strip()
  characters = {
      **(
          {
              'Who we are': constant.Constant(
                  state=character_words(character),
                  pre_act_label=f'\nWho the {people.name} are',
              )
          }
          if character
          else {}
      ),
      **(
          {
              'How the world works': constant.Constant(
                  state=conventions, pre_act_label='\nHow the world works'
              )
          }
          if conventions
          else {}
      ),
  }
  # A people a person plays answers treaty offers itself, on its own turn.
  human = people.name in context.humans
  context.add_actor(
      actors_lib.PolityActor(
          people.name,
          polity,
          context.planet,
          share=float(rules.get('focus_share', 0.15)),
          automation=context.automation,
          person=human,
          age=str(rules.get('age', 'the early iron age')),
          coached=context.coached,
      ),
      hears=(
          'world.tick',
          'geo.events',
          f'civ.{polity}.',
          f'civ.alerts.{polity}',
          f'diplomacy.{polity}.',
          *(('media.omens',) if owner else ()),
      ),
  )
  if context.wakeups is not None and rules.get('wakeups'):
    context.wakeups.add(people.name, polity, data.load('civ.yaml')['wakeups'])
  return prefab_lib.InstanceConfig(
      prefab=POLITY_PREFAB,
      role=prefab_lib.Role.ENTITY,
      params={
          'name': people.name,
          'goal': rules.get('polity_goal', DEFAULT_GOAL),
          'bus': context.bus,
          'subscriptions': subscriptions,
          'renderers': {
              topic: publishing.renderer_for(topic)
              for topic in subscriptions.values()
          },
          'clock_topic': 'world.tick',
          'plan_topic': f'civ.{polity}.plan',
          # The market news is a transported belief, not a plain subscription;
          # a people plans as who it is.
          'plan_components': ['Market', *characters],
          'sim_scheduler': context.sim,
          'deliberation_latency_ticks': int(rules.get('plan_latency_ticks', 3)),
          'deliberation_model': slow_model,
          'randomize_choices': False,
          'measurements': context.measurements,
          **context.act_params(people.name),
          'extra_components': {
              **characters,
              # News of nearby lands, later the farther away.
              'Local news': perception.local_news(
                  context, owner=people.name, polity=polity
              ),
              # Prices arrive by courier until the telegraph: news as old as
              # the people's comms (a contrib transport with that latency).
              'Market': perception.instrument(
                  context,
                  'econ.prices',
                  owner=people.name,
                  label='Market',
                  initial='No market news has reached us yet.',
                  transported=True,
                  channel=lambda: operators.Transport(
                      latency=float(
                          context.planet.trade.news_lag(polity)
                          if context.planet.trade
                          else 0
                      )
                  ),
              ),
              'Storehouse': report_function.ReportFunction(
                  function=lambda: storehouse(context, polity),
                  pre_act_label='\nOur storehouse and labor',
              ),
              # Deliberates over a treaty offer the moment it is observed;
              # its counsel is in the people's context, and the people
              # answers on its menu (a person plays without the mind).
              **(
                  {}
                  if human
                  else {
                      'Counsel on the offer': perception.instrument(
                          context,
                          f'civ.{polity}.treaty',
                          owner=people.name,
                          label='Counsel on the offer',
                          initial='No offer awaits our answer.',
                          render=lambda message: str(message.payload),
                      ),
                      'Treaty mind': context.make_mind(
                          people.name,
                          'Treaty',
                          f'civ.{polity}.treaty',
                          pact_mind.TreatyMind(
                              slow_model,
                              context,
                              polity,
                              components=[
                                  'Storehouse',
                                  'Census of the peoples',
                              ],
                          ),
                          context.planet.treaties.answer_latency
                          if context.planet.treaties
                          else 1,
                          trigger_on_observe=True,
                      ),
                  }
              ),
              # Writes the broadsheets the people pays paper for.
              'Press mind': context.make_mind(
                  people.name,
                  'Press',
                  f'civ.{polity}.press',
                  press_mind.PressMind(
                      slow_model,
                      context,
                      polity,
                      components=['Storehouse', 'Census of the peoples'],
                  ),
                  context.planet.gazette.latency
                  if context.planet.gazette
                  else 1,
              ),
              # The presses' owners retell the Steward's omens, a turn later.
              **(
                  {
                      'Relay mind': context.make_mind(
                          people.name,
                          'Relay',
                          f'civ.{polity}.retelling',
                          relay_mind.RelayMind(
                              slow_model,
                              context,
                              components=[
                                  *characters,
                                  'Census of the peoples',
                              ],
                          ),
                          int(
                              context.planet.gazette.relay.get(
                                  'latency_ticks', 1
                              )
                          ),
                      )
                  }
                  if owner == polity and context.planet.gazette is not None
                  else {}
              ),
              # One sentence, in its own voice, on each omen it reads.
              **(
                  {
                      'Take mind': context.make_mind(
                          people.name,
                          'Take',
                          f'civ.{polity}.takes',
                          take_mind.TakeMind(
                              slow_model,
                              context,
                              civ.polities[owner].name,
                              components=[
                                  *characters,
                                  'The Gazette on the omens',
                                  'Storehouse',
                              ],
                          ),
                          1,
                      )
                  }
                  if owner and character
                  else {}
              ),
              # The scribes' draft, as counsel: the people adopts it (or not)
              # by an action on its menu.
              **(
                  {
                      "Our scribes' draft": perception.instrument(
                          context,
                          f'civ.{polity}.policy',
                          owner=people.name,
                          label="Our scribes' draft",
                          initial=(
                              'Our scribes have drafted no standing orders.'
                          ),
                          render=lambda message: (
                              'Standing orders our scribes drafted (adopt them'
                              ' to put them to work): '
                              + context.automation.summary(
                                  polity, str(message.payload)
                              )
                          )
                          if context.automation is not None
                          else str(message.payload),
                      )
                  }
                  if context.automation is not None
                  else {}
              ),
              # Writes standing orders (reflex JSON) once they are unlocked.
              'Policy compiler': context.make_mind(
                  people.name,
                  'Policy',
                  f'civ.{polity}.policy',
                  policy_compiler.PolicyCompiler(
                      slow_model,
                      context,
                      polity,
                      components=['Storehouse', 'Market'],
                  ),
                  int(rules.get('policy_latency_ticks', 4)),
              ),
          },
      },
  )
