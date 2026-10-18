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
  if isinstance(context.actors.get(entity), actors_lib.StewardActor):
    topics[perception.sheet_topic(entity, LOGBOOK)] = lambda: json.dumps(
        logbook(context, entity)
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
