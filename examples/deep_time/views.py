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

"""What the World GM publishes for the people who play: each role's view.

A person plays a role through the stock `HumanActComponent`; the request it
hands their seat carries what the role reads. Their page and seat also need
the role's menu (its moves, those it is saving for and why, what each does),
the trouble it can see coming (its crises, with countdowns) and why it
should look up from a held course. The World GM works these out here, from
what the role knows and the world's rules, and publishes them as the role's
own sheets (`perception.sheet_topic`): the menu as it asks for the role's
action (`ActorActionSpecs`), the rest each step. The seat and the page read
those sheets, never the world.
"""

# pylint: disable=g-long-ternary

import json
import math
from typing import Any

from examples.deep_time import actors as actors_lib
from examples.deep_time import crises as crises_lib
from examples.deep_time import forecast as forecast_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception

MENU = 'menu'
NEEDS = 'needs'
CRISES = 'crises'
LOGBOOK = 'logbook'
IN_FLIGHT = 'in_flight'
TRAITS = 'traits'
UNWATCHED = 'unwatched'
# What a Steward's logbook keeps of each turn: what it read, never the truth.
READ = ('tick', 'believed_temp_c', 'read_tick', 'read_sigma')


def latest(
    context: game_lib.GameContext, entity: str, kind: str, default: Any
) -> Any:
  """A role's latest published sheet of `kind`, decoded (JSON) or `default`."""
  message = context.bus.latest(perception.sheet_topic(entity, kind))
  if message is None:
    return default
  try:
    return json.loads(message.payload)
  except (TypeError, ValueError):
    return default


def menu(context: game_lib.GameContext, entity: str) -> dict[str, Any]:
  """The menu the World GM last offered a role (`ActorActionSpecs`)."""
  return latest(context, entity, MENU, {})


def register(context: game_lib.GameContext, entity: str, mode: str) -> None:
  """Has the World GM publish a role's crises and needs each step.

  Args:
    context: Shared game state.
    entity: The role a person plays (or the page follows).
    mode: 'play' or 'watch' (watching, the Steward's crises are shown).
  """
  topics = {
      perception.sheet_topic(entity, CRISES): lambda: json.dumps(
          crises(context, entity, mode)
      ),
      perception.sheet_topic(entity, NEEDS): lambda: json.dumps(
          needs_you(context, entity)
      ),
  }
  # What is on its way to the role, and when it lands.
  topics[perception.sheet_topic(entity, IN_FLIGHT)] = lambda: json.dumps(
      in_flight(context, entity)
  )
  actor = context.actors.get(entity)
  if isinstance(actor, (actors_lib.SpeciesActor, actors_lib.GrazerActor)):
    topics[perception.sheet_topic(entity, TRAITS)] = lambda: json.dumps(
        traits(context, entity)
    )
  if isinstance(actor, actors_lib.StewardActor):
    topics[perception.sheet_topic(entity, LOGBOOK)] = lambda: json.dumps(
        logbook(context, entity)
    )
    # Which of its instruments it does not watch (their readings freeze).
    topics[perception.sheet_topic(entity, UNWATCHED)] = lambda: json.dumps(
        unwatched(context, entity)
    )
  for topic, function in topics.items():
    context.add_sheet(topic, function)
  # As the world's own state is published when a game is made: from tick 0.
  context.publish_sheets(list(topics))


def logbook(context: game_lib.GameContext, entity: str) -> dict[str, Any]:
  """What a Steward knows to reckon its forecast by (its page reckons it).

  Its readings as they reached it (never the truth), the moves it made, what
  each move does to the temperature turn by turn on this world, the habitable
  band and the moves' keys: with a person's own plans, the page reckons the
  Steward's forecast from these (`forecast`).

  Args:
    context: Shared game state.
    entity: The Steward.

  Returns:
    JSON-safe.
  """
  planet = context.planet
  low, high = planet.band()
  return {
      'tick': planet.tick,
      'rows': [{k: r[k] for k in READ if k in r} for r in planet.history],
      'made': [[t, k] for t, k in planet.made],
      'profiles': planet.move_profiles,
      # A band too wide ever to bind (a thaw's) has no edge to leave by.
      'band': [low, high] if high - low <= 60 else None,
      'acts_in': acts_in(context, entity),
      'keys': {i.option: i.key for i in planet.interventions()},
  }


def forecast(log: Any, planned: Any = ()) -> forecast_lib.Forecast | None:
  """A Steward's forecast from its logbook, with a person's plans, if any."""
  if not log:
    return None
  return forecast_lib.forecast(
      log['rows'],
      [tuple(m) for m in log['made']],
      log['profiles'],
      tick=int(log['tick']),
      acts_in=int(log['acts_in']),
      band=tuple(log['band']) if log['band'] else None,
      planned=planned,
  )


def acts_in(context: game_lib.GameContext, entity: str) -> int:
  """Turns until a role's move acts (Mission Control's cross the void)."""
  actor = context.actors.get(entity)
  colony = context.planet.colony
  if isinstance(actor, actors_lib.MissionActor) and colony is not None:
    return colony.light_delay
  return 1


def crises(
    context: game_lib.GameContext, entity: str | None, mode: str
) -> list[dict[str, Any]]:
  """A role's crises: a Steward's or a people's.

  Watching, the Steward's are shown (whoever the page follows); a person
  playing another role, or a spectator of a played game, sees none here.
  """
  if entity is None:
    return []
  actor = context.actors.get(entity)
  colony = context.planet.colony
  if (
      isinstance(actor, (actors_lib.ColonyActor, actors_lib.MissionActor))
      and colony is not None
      and mode == 'play'
      and not context.planet.outcome
  ):
    options = list(actor.action_spec().options)
    planet = context.planet
    if isinstance(actor, actors_lib.ColonyActor):
      report, age = colony.payload(planet.tick), 0
    else:  # Mission Control: the relay, as late as the light delay.
      message = context.received.get((entity, 'Colony relay'))
      if message is None:
        return []
      report = json.loads(message.payload)
      age = max(0, planet.tick - int(message.time or planet.tick))
    return crises_lib.colony_crises(
        report,
        colony.needs,
        [(r.label, r.outputs) for r in colony.recipes],
        options,
        context.pushes(options, entity),
        age,
    )
  if (
      isinstance(actor, (actors_lib.SpeciesActor, actors_lib.GrazerActor))
      and mode == 'play'
      and not context.planet.outcome
  ):
    options = list(actor.action_spec().options)
    received = {
        label: message
        for (owner, label), message in context.received.items()
        if owner == entity
    }
    pushes = context.pushes(options, entity)
    if isinstance(actor, actors_lib.GrazerActor):
      return crises_lib.grazer_crises(context.planet, received, options, pushes)
    return crises_lib.species_crises(
        context.planet, actor.species, received, options, pushes
    )
  if isinstance(actor, actors_lib.PolityActor) and mode == 'play':
    if context.planet.outcome:
      return []
    options = list(actor.action_spec().options)
    return crises_lib.polity_crises(
        context.planet,
        actor.polity,
        options,
        context.pushes(options, entity),
    )
  if not isinstance(actor, actors_lib.StewardActor):
    if mode == 'play':
      return []
    entity = next(
        (
            n
            for n, a in context.actors.items()
            if isinstance(a, actors_lib.StewardActor)
        ),
        None,
    )
    actor = context.actors.get(entity) if entity else None
    if not isinstance(actor, actors_lib.StewardActor) or entity is None:
      return []
  planet = context.planet
  if planet.outcome:
    return []
  received = {
      label: message
      for (owner, label), message in context.received.items()
      if owner == entity
  }
  options = [*actor.options(), *actor.out_of_reach()]
  found = forecast_lib.of_planet(planet, acts_in=acts_in(context, entity))
  return crises_lib.steward_crises(
      planet, received, found, options, context.pushes(options, entity)
  )


def needs_you(context: game_lib.GameContext, entity: str) -> str:
  """Why a player holding course should look now ('' if nothing does)."""
  planet = context.planet
  if planet.outcome:
    return 'the game is decided'
  actor = context.actors.get(entity)
  civ = planet.civ
  if isinstance(actor, actors_lib.PolityActor) and civ is not None:
    polity = civ.polities.get(actor.polity)
    if polity is not None and polity.food_ratio < 0.95:
      return 'your people go hungry'
    treaties = planet.treaties
    if treaties is not None and treaties.pending_for(actor.polity):
      return 'an offer awaits your answer'
  if isinstance(actor, actors_lib.StewardActor):
    # What the Steward's own thermometer says (late, noisy, perhaps not
    # watched at all): holding must not be an oracle for the truth.
    believed = context.believed(entity, 'climate.global', 'temp_c')
    low, high = planet.band()
    if believed is not None and not low <= believed <= high:
      return 'your thermometer reads outside the band'
    cap = planet.goal().get('max_co2')
    air = context.believed(entity, 'climate.global', 'co2')
    if cap is not None and air is not None and air >= float(cap):
      return f'your thermometer reads the air past ×{float(cap):g} carbon'
    if civ is not None and any(
        p.food_ratio < 0.9 for p in civ.polities.values()
    ):
      return 'a people goes hungry'
  return ''


def unwatched(context: game_lib.GameContext, entity: str) -> list[str]:
  """The instruments a Steward does not watch now (its attention)."""
  attention = context.attention.get(entity)
  return [] if attention is None else attention.unwatched()


def traits(context: game_lib.GameContext, name: str) -> dict[str, Any] | None:
  """A species' own traits and what it is evolving toward (for its page)."""
  planet = context.planet
  actor = context.actors.get(name)
  if isinstance(actor, actors_lib.SpeciesActor):
    traits = planet.daisies.traits(actor.species)
    evolution = planet.evolution
    return {
        'optimum_c': round(float(traits.optimum_c), 1),
        'albedo': round(float(traits.albedo), 2),
        'evolving': [
            evolution.option(p.option).label
            for p in planet.pending_adaptations
            if p.species == actor.species and evolution is not None
        ],
    }
  grazers = planet.grazers
  if isinstance(actor, actors_lib.GrazerActor) and grazers is not None:
    return {
        'optimum_c': round(float(grazers.optimum_c), 1),
        'white_diet': round(float(grazers.diet_white), 2),
        'evolving': [label for label, _ in grazers.evolving()],
    }
  return None


def in_flight(context: game_lib.GameContext, name: str) -> list[dict[str, Any]]:
  """What the world has on its way to a role, and the turn it lands.

  The other half of lag: an adaptation phasing in, a transmission crossing
  the light delay, news a people's couriers are still carrying. (What the
  role's own slow minds are working on, it knows from them.)
  """
  planet, now = context.planet, context.planet.tick
  actor = context.actors.get(name)
  out: list[dict[str, Any]] = []

  def add(label: str, lands: int, kind: str, about: str = '') -> None:
    if lands > now:
      out.append({
          'label': label,
          'lands': int(lands),
          'kind': kind,
          'about': about or label,
      })

  steward = isinstance(actor, actors_lib.StewardActor)
  own = actor.species if isinstance(actor, actors_lib.SpeciesActor) else None
  evolution = planet.evolution
  for pending in planet.pending_adaptations:
    if evolution is not None and (steward or pending.species == own):
      who = planet.species_name(pending.species)
      what = evolution.option(pending.option).label
      add(who, now + pending.ticks_left, 'evolve', f'{what}, phasing in')
  grazers = planet.grazers
  if grazers is not None and (
      steward or isinstance(actor, actors_lib.GrazerActor)
  ):
    who = planet.species_name(grazers.key)
    for what, left in grazers.evolving():
      add(who, now + left, 'evolve', f'{what}, phasing in')
  colony = planet.colony
  if colony is not None and isinstance(actor, actors_lib.MissionActor):
    for arrival, kind, what in colony.incoming:
      label, about = {
          'design': ('Design', colony.design_labels.get(what, what)),
          'site': ('Survey', 'where to mine the ice'),
      }.get(kind, ('Directive', what))
      add(
          label,
          now + arrival - colony.generation,
          'wire',
          f'{label} for {colony.name}: {about}',
      )
  civ = planet.civ
  if civ is not None and isinstance(actor, actors_lib.PolityActor):
    out.extend(news_in_flight(context, actor.polity))
  return sorted(out, key=lambda f: f['lands'])


def news_in_flight(
    context: game_lib.GameContext, polity: str
) -> list[dict[str, Any]]:
  """News still on the couriers' road to a people (as `local_news`)."""
  planet = context.planet
  civ, config, now = planet.civ, planet.news_config, planet.tick
  assert civ is not None
  home = civ.polities[polity].home
  per = float(
      config[
          'wired_ticks_per_region'
          if civ.unlocked(polity, 'wire_news')
          else 'ticks_per_region'
      ]
  )
  # Where the couriers ride to: the people's capital (map rows north first).
  north = planet.tiles.rows - 1
  places = (
      planet.settlements.places.get(polity, []) if planet.settlements else []
  )
  capital = next((p for p in places if p.capital), None)
  out = []
  for ripple in planet.ripples:
    if not ripple.get('local', True):
      continue
    region = int(planet.tiles.region[ripple['row'], ripple['col']])
    steps = planet.region_steps(home, region)
    lands = ripple['tick'] + math.ceil(steps * per - 1e-9)
    if steps <= int(config['reach_regions']) and lands > now:
      away = f'{steps} region{"s" if steps != 1 else ""} off'
      out.append({
          'label': 'News',
          'lands': lands,
          'kind': 'news',
          'about': f"News from {away}, on the couriers' road",
          'sent': int(ripple['tick']),
          'from': [north - int(ripple['row']), int(ripple['col'])],
          'to': (
              [north - capital.row, capital.col]
              if capital is not None
              else None
          ),
          'topic': str(ripple['topic']).split('.', maxsplit=1)[0],
      })
  return out
