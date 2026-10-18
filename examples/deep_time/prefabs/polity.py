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
* a storehouse sheet, which the World GM computes from the ledger and market
  and publishes each step, read through a stock `Subscriber`;
* a fast act that each tick puts more labor into one recipe, learns a tech,
  or places a standing order on the market (`actors.PolityActor`).
"""

# pylint: disable=g-long-ternary

from collections.abc import Mapping
import dataclasses
from typing import Any, override

from concordia.components.agent import constant
from concordia.contrib.pubsub import operators
from concordia.contrib.pubsub import topics
from examples.deep_time import actors as actors_lib
from examples.deep_time import crises as crises_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.components import pact_mind
from examples.deep_time.components import policy_compiler
from examples.deep_time.components import press_mind
from examples.deep_time.components import relay_mind
from examples.deep_time.prefabs import playable
from examples.deep_time.world import data
from examples.deep_time.world import publishing
from concordia.typing import prefab as prefab_lib

POLITY_PREFAB = 'deep_time_polity__Entity'

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
    if offer is not None:
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
  text += our_acts(context, polity)
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


def treaty_offer(context: game_lib.GameContext, polity: str) -> str:
  """The offer awaiting the polity's answer, as its treaty mind weighs it."""
  treaties, civ = context.planet.treaties, context.planet.civ
  proposal = treaties.pending_for(polity) if treaties else None
  if proposal is None or treaties is None or civ is None:
    return ''
  # The reply names the offer's kind, so it settles only that offer.
  what = treaties.kinds[proposal.kind].name
  return pact_mind.offer_sheet(
      proposal.text,
      f'{civ.polities[proposal.proposer].name} ({what})',
      treaties.question,
  )


def commission(context: game_lib.GameContext, polity: str) -> str:
  """The question the polity's printers answer, if it paid for a sheet."""
  gazette = context.planet.gazette
  if gazette is None or polity not in gazette.pending:
    return ''
  return gazette.question_for(polity)


def newest_omen(context: game_lib.GameContext) -> str:
  """The Steward's newest omen, as the printers' relay mind weighs it."""
  planet = context.planet
  gazette = planet.gazette
  if gazette is None or not planet.omens:
    return ''
  _, sign, words = planet.omens[-1]
  return relay_mind.omen_sheet(
      len(planet.omens),
      str(gazette.relay['question']),
      printers=' '.join(
          str(
              context.scenario['civilization'].get('printers')
              or gazette.relay['printers']
          ).split()
      ),
      sign=sign,
      words=words or '(nothing anyone could make out)',
  )


def scribes(context: game_lib.GameContext, polity: str) -> str:
  """What the polity's scribes may write standing orders with."""
  automation = context.automation
  if automation is None:
    return ''
  return policy_compiler.scribes_sheet(
      sorted(automation.allowed_ops(polity)), sorted(automation.levels)
  )


def scribes_draft(
    context: game_lib.GameContext, polity: str, message: topics.Message
) -> str:
  """The scribes' draft as it reaches the people: its orders, as written."""
  if context.automation is None:
    return str(message.payload)
  return (
      'Standing orders our scribes drafted (adopt them, as written, to put'
      ' them to work): '
      + context.automation.summary(polity, str(message.payload))
      + '. As written: '
      + ' '.join(str(message.payload).split())
  )


def our_acts(context: game_lib.GameContext, polity: str) -> str:
  """When the people last enacted its printers' and scribes' drafts."""
  civ = context.planet.civ
  assert civ is not None
  name = civ.polities[polity].name
  said = {
      'commission': 'paid its printers for a sheet',
      'press': 'published a broadsheet',
      'retelling': "printed its printers' retelling of an omen",
      'orders': "adopted its scribes' standing orders",
  }
  done = [
      f'{what} at tick {tick}'
      for key, what in said.items()
      if (tick := actors_lib.made_at(context.planet, f'{name}:{key}'))
      is not None
  ]
  return f' Our people last {"; ".join(done)}.' if done else ''


def character_words(character: Mapping[str, Any]) -> str:
  """A character sheet in words: who a people is, wants, fears, believes."""
  return ' '.join(
      f'{label}: {" ".join(str(character[key]).split()).rstrip(".")}.'
      for key, label in CHARACTER_LINES
      if character.get(key)
  )


@dataclasses.dataclass
class Polity(playable.RolePolity):
  """A people: its beliefs, its slow minds and its actor."""

  @override
  def assemble(
      self, context: game_lib.GameContext, params: Mapping[str, Any]
  ) -> Mapping[str, Any]:
    return polity_params(context, params)

  @override
  def rules(
      self, context: game_lib.GameContext, params: Mapping[str, Any]
  ) -> None:
    polity_rules(context, str(params['polity']))


def polity_config(
    context: game_lib.GameContext, polity: str
) -> prefab_lib.InstanceConfig:
  """A people's instance config (plain params; `Polity` builds it).

  Args:
    context: Shared game state (its planet must have a civilization).
    polity: The polity's key in the scenario.

  Returns:
    The instance config.
  """
  civ = context.planet.civ
  if civ is None:
    raise ValueError('Polities need a civilization in the scenario.')
  # Its static settings, read once from the new game: who owns the presses,
  # whether it keeps scribes, and how long its slow minds take.
  treaties, gazette = context.planet.treaties, context.planet.gazette
  return prefab_lib.InstanceConfig(
      prefab=POLITY_PREFAB,
      role=prefab_lib.Role.ENTITY,
      params={
          'name': civ.polities[polity].name,
          'polity': polity,
          'press_owner': _press_owner(context),
          'scribes': context.automation is not None,
          'treaty_latency': treaties.answer_latency if treaties else None,
          'press_latency': gazette.latency if gazette else None,
          'relay_latency': (
              int(gazette.relay.get('latency_ticks', 1)) if gazette else None
          ),
      },
  )


def _press_owner(context: game_lib.GameContext) -> str | None:
  """The people whose presses retell the Steward's omens, if any."""
  if context.planet.gazette is None:
    return None
  return context.scenario['civilization'].get('press_owner')


def _readings(
    polity: str, *, presses: bool, scribes: bool
) -> dict[str, tuple[str | None, str]]:
  """What a people reads, by component key: (topic, initial belief).

  The topic is named in the reading's label (none for the couriers' digest
  of nearby lands, which reads many).

  Args:
    polity: The polity's key.
    presses: Whether it owns the presses that retell the omens.
    scribes: Whether its scribes draft standing orders.
  """
  readings: dict[str, tuple[str | None, str]] = {
      'Local news': (None, 'No news of nearby lands yet.'),
      'Market': ('econ.prices', 'No market news has reached us yet.'),
      'Counsel on the offer': (
          f'civ.{polity}.treaty',
          'No offer awaits our answer.',
      ),
      "Our printers' sheet": (
          f'civ.{polity}.press',
          'Our printers have written nothing yet.',
      ),
  }
  if presses:
    readings["Our printers' retelling"] = (
        f'civ.{polity}.retelling',
        'Our printers have retold no omen yet.',
    )
  if scribes:
    readings["Our scribes' draft"] = (
        f'civ.{polity}.policy',
        'Our scribes have drafted no standing orders.',
    )
  return readings


def _instruments(
    context: game_lib.GameContext, polity: str
) -> dict[str, perception.Instrument]:
  """A people's instruments as the World GM runs them, by component key."""
  civ = context.planet.civ
  assert civ is not None
  name = civ.polities[polity].name
  owner = _press_owner(context)
  specs = {
      # News of nearby lands, later the farther away.
      'Local news': perception.local_news(context, owner=name, polity=polity),
      # Prices arrive by courier until the telegraph: news as old as the
      # people's comms (a contrib transport with that latency).
      'Market': perception.instrument(
          context,
          'econ.prices',
          owner=name,
          label='Market',
          transported=True,
          channel=lambda: operators.Transport(
              latency=float(
                  context.planet.trade.news_lag(polity)
                  if context.planet.trade
                  else 0
              )
          ),
      ),
      # Its treaty mind's counsel on an offer; the people answers on its menu.
      'Counsel on the offer': perception.instrument(
          context,
          f'civ.{polity}.treaty',
          owner=name,
          label='Counsel on the offer',
          render=lambda message: str(message.payload),
      ),
      # What its printers wrote, for the people to publish (its action
      # carries the words).
      "Our printers' sheet": perception.instrument(
          context,
          f'civ.{polity}.press',
          owner=name,
          label="Our printers' sheet",
          render=lambda message: str(message.payload),
      ),
  }
  if owner == polity:
    # What its printers made of the newest omen, for the people that owns
    # the presses to print (its action carries the words).
    specs["Our printers' retelling"] = perception.instrument(
        context,
        f'civ.{polity}.retelling',
        owner=name,
        label="Our printers' retelling",
        render=lambda message: str(message.payload),
    )
  if context.automation is not None:
    # The scribes' draft, as counsel: the people adopts it (or not) by an
    # action carrying it.
    specs["Our scribes' draft"] = perception.instrument(
        context,
        f'civ.{polity}.policy',
        owner=name,
        label="Our scribes' draft",
        render=lambda message: scribes_draft(context, polity, message),
    )
  return specs


def polity_rules(context: game_lib.GameContext, polity: str) -> None:
  """The World GM installs a people's rules.

  Its actor and wake-ups, its storehouse sheet, what its slow minds weigh,
  and the instruments it reads.

  Args:
    context: Shared game state (its planet must have a civilization).
    polity: The polity's key in the scenario.
  """
  civ = context.planet.civ
  if civ is None:
    raise ValueError('Polities need a civilization in the scenario.')
  name = civ.polities[polity].name
  rules = context.scenario['civilization']
  owner = _press_owner(context)
  character = (rules.get('characters') or {}).get(polity)
  context.add_actor(
      actors_lib.PolityActor(
          name,
          polity,
          context.planet,
          share=float(rules.get('focus_share', 0.15)),
          automation=context.automation,
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
    context.wakeups.add(name, polity, data.load('civ.yaml')['wakeups'])
  specs = _instruments(context, polity)
  for key in ('Local news', 'Market'):
    perception.deliver(context, specs[key])
  perception.publish(
      context, name, 'storehouse', lambda: storehouse(context, polity)
  )
  perception.deliver(context, specs['Counsel on the offer'])
  perception.publish(
      context, name, 'offer', lambda: treaty_offer(context, polity)
  )
  perception.deliver(context, specs["Our printers' sheet"])
  perception.publish(
      context, name, 'commission', lambda: commission(context, polity)
  )
  if owner == polity:
    perception.publish(context, name, 'omen', lambda: newest_omen(context))
    perception.deliver(context, specs["Our printers' retelling"])
  if context.automation is not None:
    perception.deliver(context, specs["Our scribes' draft"])
  perception.publish(context, name, 'scribes', lambda: scribes(context, polity))


def _ticks(params: Mapping[str, Any], key: str) -> int:
  """A slow mind's latency from a people's params (1 where it has none)."""
  value = params.get(key)
  return 1 if value is None else int(value)


def polity_params(
    context: game_lib.GameContext, params: Mapping[str, Any]
) -> dict[str, Any]:
  """A people's async-deliberation params, its components included.

  Args:
    context: Shared game state (its services, and the scenario's words).
    params: Its instance params (`polity_config`).

  Returns:
    The base prefab's params, its components included.
  """
  polity, name = str(params['polity']), str(params['name'])
  owner = params.get('press_owner')
  gazette = params.get('press_latency') is not None
  slow_model = context.slow()
  rules = context.scenario['civilization']
  subscriptions = dict(SUBSCRIPTIONS)
  if params.get('treaty_latency') is not None:
    subscriptions['Treaties'] = 'civ.treaties'
  if gazette:
    subscriptions['Gazette'] = 'media.gazette'
  # Where the presses retell the Steward's omens, every people reads them.
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
                  pre_act_label=f'\nWho the {name} are',
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
  readings = {
      key: perception.reading(name, key, initial, topic=topic)
      for key, (topic, initial) in _readings(
          polity, presses=owner == polity, scribes=bool(params['scribes'])
      ).items()
  }
  return {
      'name': name,
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
      **context.act_params(name),
      'extra_components': {
          **characters,
          # News of nearby lands, later the farther away.
          'Local news': perception.subscribe(context, readings['Local news']),
          # Prices arrive by courier until the telegraph: news as old as
          # the people's comms (a contrib transport with that latency).
          'Market': perception.subscribe(context, readings['Market']),
          'Storehouse': perception.read(
              context,
              name,
              'storehouse',
              label='\nOur storehouse and labor',
          ),
          # Deliberates over a treaty offer the moment it is observed; its
          # counsel is in the people's context, whoever plays it, and the
          # people answers on its menu.
          'Counsel on the offer': perception.subscribe(
              context, readings['Counsel on the offer']
          ),
          'Treaty mind': context.make_mind(
              name,
              'Treaty',
              f'civ.{polity}.treaty',
              pact_mind.PactMind(
                  slow_model,
                  perception.reader(context, name, 'offer'),
                  components=[
                      'Storehouse',
                      'Census of the peoples',
                  ],
              ),
              _ticks(params, 'treaty_latency'),
              trigger_on_observe=True,
          ),
          # What its printers wrote, for the people to publish (its action
          # carries the words).
          "Our printers' sheet": perception.subscribe(
              context, readings["Our printers' sheet"]
          ),
          # Writes the broadsheets the people pays paper for.
          'Press mind': context.make_mind(
              name,
              'Press',
              f'civ.{polity}.press',
              press_mind.PressMind(
                  slow_model,
                  perception.reader(context, name, 'commission'),
                  components=['Storehouse', 'Census of the peoples'],
              ),
              _ticks(params, 'press_latency'),
          ),
          # The presses' owners retell the Steward's omens, a turn later.
          **(
              {
                  'Relay mind': context.make_mind(
                      name,
                      'Relay',
                      f'civ.{polity}.retelling',
                      relay_mind.RelayMind(
                          slow_model,
                          perception.reader(context, name, 'omen'),
                          components=[
                              *characters,
                              'Census of the peoples',
                          ],
                      ),
                      _ticks(params, 'relay_latency'),
                  )
              }
              if owner == polity and gazette
              else {}
          ),
          # What its printers made of the newest omen, for the people that
          # owns the presses to print (its action carries the words).
          **(
              {
                  "Our printers' retelling": perception.subscribe(
                      context, readings["Our printers' retelling"]
                  )
              }
              if owner == polity and gazette
              else {}
          ),
          # The scribes' draft, as counsel: the people adopts it (or not)
          # by an action on its menu.
          **(
              {
                  "Our scribes' draft": perception.subscribe(
                      context, readings["Our scribes' draft"]
                  )
              }
              if params['scribes']
              else {}
          ),
          # Writes standing orders (reflex JSON) once they are unlocked.
          'Policy compiler': context.make_mind(
              name,
              'Policy',
              f'civ.{polity}.policy',
              policy_compiler.PolicyCompiler(
                  slow_model,
                  perception.reader(context, name, 'scribes'),
                  polity,
                  components=['Storehouse', 'Market'],
              ),
              int(rules.get('policy_latency_ticks', 4)),
          ),
      },
  }
