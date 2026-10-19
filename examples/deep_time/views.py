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

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-long-ternary

import json
import math
from typing import Any

from examples.deep_time import actors as actors_lib
from examples.deep_time import crises as crises_lib
from examples.deep_time import forecast as forecast_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.world import data
import numpy as np

MENU = 'menu'
NEEDS = 'needs'
CRISES = 'crises'
LOGBOOK = 'logbook'
IN_FLIGHT = 'in_flight'
TRAITS = 'traits'
UNWATCHED = 'unwatched'
ROLE = 'role'
READINGS = 'readings'
MOVES = 'moves'
STRIP = 'strip'
PEOPLE = 'people'
SEEN = 'seen'
ARARAT = 'ararat'
# What anyone sees of the world from afar, each step (the map's surface).
SURFACE_TOPIC = 'world.surface'
# A Steward's aimed sowings its census shows, the last few ticks of them.
PLANTINGS_SHOWN = 6
# One character per share, for the map's meadows.
_BASE36 = '0123456789abcdefghijklmnopqrstuvwxyz'
# Each kind of role, by its actor's rules (the most specific first).
_KINDS = (
    (actors_lib.StewardActor, 'steward'),
    (actors_lib.PolityActor, 'people'),
    (actors_lib.GrazerActor, 'herds'),
    (actors_lib.SpeciesActor, 'species'),
    (actors_lib.ColonyActor, 'colony'),
    (actors_lib.MissionActor, 'mission'),
    (actors_lib.ChroniclerActor, 'chronicler'),
)
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
      # What kind of role it is (its page draws it so).
      perception.sheet_topic(entity, ROLE): lambda: json.dumps(
          role(context, entity)
      ),
      # What reached it through its instruments, and what its moves did.
      perception.sheet_topic(entity, READINGS): lambda: json.dumps(
          readings(context, entity)
      ),
      perception.sheet_topic(entity, MOVES): lambda: json.dumps(
          moves(context, entity)
      ),
      # The numbers its strip shows: what it read, never the truth.
      perception.sheet_topic(entity, STRIP): lambda: json.dumps(
          strip(context, entity)
      ),
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
  if (entity, 'Census') in context.tapped:
    # The life its census last saw, tile by tile, as late as it arrived.
    topics[perception.sheet_topic(entity, SEEN)] = lambda: json.dumps(
        seen(context, entity)
    )
  if context.planet.colony_world is not None:
    # The colony's world, as its light shows it from where the role is.
    topics[perception.sheet_topic(entity, ARARAT)] = lambda: json.dumps(
        ararat(context, entity)
    )
  if isinstance(actor, actors_lib.PolityActor):
    # A people's own numbers, food, labor and stores (its sheet says them).
    topics[perception.sheet_topic(entity, PEOPLE)] = lambda: json.dumps(
        own_people(context, entity)
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


def register_world(context: game_lib.GameContext) -> None:
  """Has the World GM publish the map's surface each step (`surface`)."""
  context.add_sheet(SURFACE_TOPIC, lambda: json.dumps(surface(context)))
  context.publish_sheets([SURFACE_TOPIC])


def surface(context: game_lib.GameContext) -> dict[str, Any]:
  """What anyone sees of the world from afar: the map's surface.

  The land of each region, its biomes, the herds, news spreading from where
  it happened, smog and blooms, the peoples' towns and roads (and their
  hunger), clouds, rain-soaked ground, burn scars and protected woods. The
  climate, the ice and life tile by tile are not here: a role sees those
  through its own instruments (`seen`).
  """
  planet = context.planet
  tiles = planet.tiles
  land = planet.daisies.land_fraction()
  return {
      'tick': planet.tick,
      'land': [round(float(v), 3) for v in land],
      'civ': planet.civ is not None,
      # Fire risk can be shown only where something can burn.
      'flammable': any(
          planet.fire.flammability(k) > 0 for k in planet.daisies.species()
      ),
      'biomes': planet.biomes.rows(),
      'herds': planet.herds.payload(),  # Animals are seen by all.
      # News of places, spreading at the couriers' pace: [row (north
      # first), col, tick, kind ('shift' for biomes shifting), topic, text].
      'ripples': [
          [
              tiles.rows - 1 - r['row'],
              r['col'],
              int(r['tick']),
              r['topic'].split('.')[0] if r.get('local', True) else 'shift',
              r['topic'].split('.')[0],
              str(r['text']),
          ]
          for r in planet.ripples
      ],
      'news_speed': tiles.per / float(planet.news_config['ticks_per_region']),
      'pollution': (
          planet.pollution.payload() if planet.pollution is not None else None
      ),
      'peoples': _peoples(context),
      'clouds': _clouds(planet),
      'wet': (
          [
              ''.join(str(min(9, int(v * 10))) for v in row)
              for row in planet.fire.wet[::-1]
          ]
          if planet.fire.wet.max() > 0.05
          else None
      ),
      # Fires and their scars ('3' burned this tick, fading to '0').
      'burns': [''.join(str(v) for v in row) for row in planet.fire.scar[::-1]],
      'reserves': _reserves(planet),
      'species': species_looks(planet),
  }


def species_looks(planet: Any) -> dict[str, dict[str, Any]]:
  """Each species as anyone sees it: its name and how bright it looks."""
  return {
      k: {
          'name': planet.species_name(k),
          'albedo': round(float(planet.daisies.traits(k).albedo), 3),
      }
      for k in planet.daisies.species()
  }


def _peoples(context: game_lib.GameContext) -> dict[str, Any] | None:
  """Towns, roads and routes, and how each people eats (seen from afar)."""
  planet = context.planet
  if planet.settlements is None:
    return None
  found = planet.settlements.payload(data.load('settlements.yaml')['colours'])
  if planet.civ is not None:  # How each people eats: hunger shows from afar.
    for p in found['polities']:
      polity = planet.civ.polities.get(p['key'])
      if polity is not None:
        p['fed'] = round(float(polity.food_ratio), 2)
        p['famine'] = bool(polity.famine)
  return found


def _clouds(planet: Any) -> list[list[float]] | None:
  """Each region's cloud albedo (+ bright cloud or ash, − dark haze)."""
  field = planet.daisies.cloud_field()
  if float(np.abs(field).max()) < 0.005:
    return None
  grid = planet.grid
  rows = field.reshape(grid.bands, grid.sectors)[::-1]  # North first.
  return [[round(float(v), 3) for v in row] for row in rows]


def _reserves(planet: Any) -> list[list[Any]]:
  """Protected woods: [row (north first), col, radius in tiles, turns left].

  One per protected region, at its tiles' centre. Everyone sees them: the
  woodcutters find the woods closed.
  """
  tiles = planet.tiles
  found = []
  for region, until in sorted(planet.reserves.items()):
    left = until - planet.tick
    if left <= 0:
      continue
    rows, cols = np.nonzero(tiles.region == region)
    if rows.size == 0:
      continue
    found.append([
        int(tiles.rows - 1 - round(float(rows.mean()))),
        int(round(float(cols.mean()))),
        round(float(np.sqrt(len(rows) / np.pi)), 1),
        int(left),
    ])
  return found


def life_tiles(planet: Any, covers: Any = None) -> dict[str, Any]:
  """Sessile life per tile, north first (now, or as a census saw it).

  `dominant` is the commonest species (one character each, indexing
  `species`), `cover` the total cover in tenths (0-9) and `covers` each
  species' own cover in 36ths (one base-36 character per tile), which the
  page draws as a meadow: each tile a patch of dots, one per share.
  """
  if covers is None:
    covers = dict(planet.tile_life.covers)  # One snapshot (game thread).
  keys, dominant, total = planet.tile_life.dominant(covers)
  land = planet.tiles.land
  return {
      'species': [{'key': k, 'name': planet.species_name(k)} for k in keys],
      'dominant': [
          ''.join(_BASE36[i] if l else '.' for i, l in zip(row, lrow))
          for row, lrow in zip(dominant[::-1], land[::-1])
      ],
      'cover': [
          ''.join(str(min(9, int(c * 10))) for c in row) for row in total[::-1]
      ],
      'covers': [
          [
              ''.join(_BASE36[min(35, int(round(c * 35)))] for c in row)
              for row in covers[key][::-1]
          ]
          for key in keys
      ],
  }


def seen(context: game_lib.GameContext, entity: str) -> dict[str, Any] | None:
  """The life a role's census last saw, tile by tile, and the tick it was taken.

  As late as the census reached it. A sowing shows where it was made once a
  census taken after it is read. None before its first census.
  """
  message = context.received.get((entity, 'Census'))
  if message is None or message.time is None:
    return None
  planet, taken = context.planet, int(message.time)
  covers = planet.life_seen.get(taken)
  if covers is None and taken == planet.tick:
    covers = dict(planet.tile_life.covers)
  if covers is None:
    # A restored game keeps no snapshots: the life map the role last had
    # stands (the bus keeps it) until a census with one reaches it.
    return latest(context, entity, SEEN, None)
  risk = planet.risk_seen.get(taken)
  steward = isinstance(context.actors.get(entity), actors_lib.StewardActor)
  return {
      'tick': taken,
      'life': life_tiles(planet, covers),
      'fire_risk': (
          [''.join(str(min(9, int(v * 10))) for v in row) for row in risk[::-1]]
          if risk is not None
          else None
      ),
      # A Steward's own aimed sowings that census has seen, lately.
      'plantings': [
          {
              'sown': int(p['tick']) - 1,
              'species': p['species'],
              'name': planet.species_name(p['species']),
              'tiles': [
                  ''.join('#' if v else '.' for v in row)
                  for row in p['tiles'][::-1]
              ],
          }
          for p in planet.plantings
          if steward and taken - PLANTINGS_SHOWN < int(p['tick']) <= taken
      ],
  }


def ararat(context: game_lib.GameContext, entity: str) -> dict[str, Any] | None:
  """The colony's world as a role sees it.

  As it is for the colonists, or as its light shows it, years late, for
  everyone at home.
  """
  world = context.planet.colony_world
  if world is None or not world.frames:
    return None
  on_site = isinstance(context.actors.get(entity), actors_lib.ColonyActor)
  frame = world.frames[-1] if on_site else world.frames[0]
  return {
      'frame': frame,
      'late': world.frames[-1]['generation'] - frame['generation'],
  }


def role(context: game_lib.GameContext, entity: str) -> dict[str, Any]:
  """What kind of role an entity is, and its own key and pace."""
  actor = context.actors.get(entity)
  kind = next((k for cls, k in _KINDS if isinstance(actor, cls)), '')
  return {
      'kind': kind,
      'polity': getattr(actor, 'polity', None) if kind == 'people' else None,
      'species': getattr(actor, 'species', None) if kind == 'species' else None,
      'acts_in': acts_in(context, entity),
      # The topics it hears as observations (its access list).
      'hears': list(context.access.get(entity, ())),
  }


def readings(context: game_lib.GameContext, entity: str) -> dict[str, Any]:
  """What reached a role through its instruments, as data (`Deliveries`).

  `received`: each instrument's last arrival, by label (as the role's
  subscriber holds it, rendered); `believed`: the last reading each
  transported instrument carried, by topic (what the role acts on);
  `tapped`: the instruments whose arrivals are kept (the others read a
  topic's latest at once).
  """
  return {
      'tapped': sorted(
          label for owner, label in context.tapped if owner == entity
      ),
      'received': {
          label: message.to_dict()
          for (owner, label), message in sorted(context.received.items())
          if owner == entity
      },
      'believed': {
          topic: message.to_dict()
          for (owner, topic), message in sorted(context.beliefs.items())
          if owner == entity
      },
  }


def moves(context: game_lib.GameContext, entity: str) -> list[list[Any]]:
  """What a role's moves did: [tick, move, outcome], oldest first."""
  actor = context.actors.get(entity)
  return [list(entry) for entry in actor.log] if actor is not None else []


def _believed(
    context: game_lib.GameContext, entity: str, topic: str, path: str
) -> tuple[str, str]:
  """A role's latest reading of a field, and how old it is."""
  message = context.beliefs.get((entity, topic))
  value = context.believed(entity, topic, path)
  if message is None or value is None:
    return '?', 'no reading'
  if message.time is None:
    return f'{value:.1f}', 'read'
  age = max(0, context.planet.tick - int(message.time))
  return f'{value:.1f}', 'now' if age == 0 else f'{age} ticks old'


def strip(
    context: game_lib.GameContext, entity: str, reveal: bool = False
) -> list[dict[str, Any]]:
  """The strip of numbers: the truth if revealed, else what the role read.

  The Steward reads late, noisy instruments but its Omega meter is exact;
  Mission Control hears the colony years late; the colonists see their own
  sky; species and peoples see what their own beliefs last delivered.
  """
  planet = context.planet
  colony = planet.colony
  kind = role(context, entity)['kind']
  stats: list[dict[str, Any]] = [
      # 'turn', as the dock says it.
      {'label': 'turn', 'value': f'{planet.tick}/{planet.max_ticks}'}
  ]
  omega: dict[str, Any] = {
      'label': 'Ω',
      'value': f'{planet.omega:.1f}',
      'exact': round(float(planet.omega), 4),  # For the moves' costs.
  }
  if colony is not None:
    target = f'{colony.target_temp_c:g}'
    if reveal or kind == 'colony':
      sky = f'{colony.temp_c:.1f}', f'sky °C (→{target})'
      people = f'{colony.population:.1f}', 'colonists'
    else:
      value, age = _believed(context, entity, 'colony.status', 'temp_c')
      sky = value, f'sky °C (→{target}), {age}'
      value, age = _believed(context, entity, 'colony.status', 'colonists')
      people = value, f'colonists, {age}'
    # (No Omega: there is no Steward to spend it in Light Delay.)
    return stats + [
        {'label': sky[1], 'value': sky[0]},
        {'label': people[1], 'value': people[0]},
    ]
  if reveal:
    temp = {'label': '°C', 'value': f'{planet.temperature_c():.1f}'}
  else:
    value, age = _believed(context, entity, 'climate.global', 'temp_c')
    temp = {'label': f'°C read, {age}', 'value': value}
  # The Omega meter is exact, and it reports biodiversity.
  bio = {'label': 'biodiversity', 'value': f'{planet.biodiversity():.2f}'}
  if reveal or kind == 'steward':
    return stats + [
        temp,
        omega,
        *(_goal_stats(context, entity, reveal) or [bio]),
    ]
  if kind == 'people' and planet.civ is not None:
    # A people knows its own storehouse exactly (its sheet says so).
    polity = role(context, entity)['polity']
    people = planet.civ.polities[polity]
    grain = planet.civ.ledger.stock(polity, 'grain')
    fed = [
        {'label': 'fed', 'value': f'{min(1.0, people.food_ratio):.0%}'},
        {'label': 'grain', 'value': f'{grain:.0f}'},
    ]
    # The goal all the peoples share, in place of the head count.
    goal = _goal_stats(context, entity, reveal)
    if goal:
      return stats + fed + goal
    return stats + [
        {'label': 'people', 'value': f'{people.population:.1f}'},
        *fed,
    ]
  if kind == 'species':
    # A species senses the census exactly and knows its own seed bank.
    species = role(context, entity)['species']
    stats.append({
        'label': 'cover',
        'value': f'{planet.daisies.cover(species):.0%}',
    })
    if planet.seed_banks is not None:
      bank = planet.seed_banks.level.get(species, 0.0)
      stats.append({'label': 'seed bank', 'value': f'{bank:.0%}'})
    return stats
  if (entity, 'climate.global') in context.beliefs:
    return stats + [temp]
  return stats


def _goal_stats(
    context: game_lib.GameContext, entity: str, reveal: bool
) -> list[dict[str, Any]]:
  """The numbers a peoples' goal is judged on, as a role reads them.

  The Steward reads the air through its thermometer (late and noisy) and
  what the peoples made through their census; a people reads both as they
  are published; the truth once revealed. A reading past its limit is
  marked. Empty where the goal has no such numbers.
  """
  planet = context.planet
  goal = planet.goal()
  if planet.civ is None or goal['kind'] not in ('industrialize', 'iron_age'):
    return []

  # A people reads the weather and the census as they are published.
  public = role(context, entity)['kind'] == 'people'

  def read(key: tuple[str, str], path: str, truth: float):
    """The truth if revealed, else the reading (or None) and its age."""
    if reveal:
      return truth, ''
    topic = {'Peoples': 'civ.census'}.get(key[1], key[1])
    message = (
        context.bus.latest(topic)
        if public
        else context.beliefs.get(key) or context.received.get(key)
    )
    if message is None:
      return None, ''
    value: Any = json.loads(message.payload)
    for part in path.split('.'):
      value = value.get(part) if isinstance(value, dict) else None
    if not isinstance(value, (int, float)):
      return None, ''
    sent = message.time
    age = 0 if sent is None else max(0, planet.tick - int(sent))
    return value, f', {age} tick{"" if age == 1 else "s"} old' if age else ''

  found = []
  if goal['kind'] == 'industrialize':
    cap = float(goal['max_co2'])
    truth = planet.carbon.relative_co2() if planet.carbon else 1.0
    co2, age = read((entity, 'climate.global'), 'co2', truth)
    found.append({
        'label': f'CO₂ / cap{age}',
        'value': f'{"?" if co2 is None else f"{co2:.2f}"}/{cap:g}',
        'warn': co2 is not None and co2 >= cap,
    })
  good, need = (
      ('energy', goal['min_energy'])
      if goal['kind'] == 'industrialize'
      else ('iron', goal['min_iron'])
  )
  # The peoples' census reaches the Steward at once (its 'Peoples').
  made, age = read((entity, 'Peoples'), f'made.{good}', planet.civ.made(good))
  found.append({
      'label': f'{good} made{age}',
      'value': f'{"?" if made is None else f"{made:.0f}"}/{float(need):g}',
  })
  return found


def own_people(
    context: game_lib.GameContext, entity: str
) -> dict[str, Any] | None:
  """A people's own numbers, food, where its hands go, and stores."""
  civ = context.planet.civ
  key = role(context, entity)['polity']
  if civ is None or key is None:
    return None
  polity = civ.polities.get(key)
  if polity is None:
    return None
  stocks = civ.payload()['polities'][key]['stocks']
  population = float(polity.population)
  eats = population * float(civ.food_per_pop)
  # The goal the peoples share, as their census already tells them, with
  # the pace it asks and the people's numbers against its floor.
  shared = crises_lib.shared_goal(context.planet, key)
  if shared and 'good' in shared:
    shared['label'] = (
        f'{shared["good"]} {"forged" if shared["good"] == "iron" else "made"}'
        ' by all peoples'
    )
  return {
      'name': polity.name,
      'population': round(population, 1),
      'fed': round(float(polity.food_ratio), 2),
      # How many seasons the granary would feed the people.
      'seasons': round(float(stocks.get('grain', 0.0)) / max(eats, 1e-9), 1),
      'shared': shared,
      # The road to what the goal asks for, as the people's sheet says it.
      'aims': [
          civ.aim(key, str(aim))
          for aim in context.scenario['civilization'].get('aims', ())
      ],
      'labor': [
          {
              'key': k,
              'label': civ.recipe(k).label,
              'share': round(float(v), 3),
          }
          for k, v in sorted(polity.allocation.items(), key=lambda kv: -kv[1])
          if v > 0
      ],
      # Every trade's label -> key, so new trades in the moves get colours.
      'trades': {r.label: r.key for r in civ.recipes},
      'stocks': {
          k: round(float(v), 1) for k, v in stocks.items() if float(v) > 0.05
      },
  }


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
    species_traits = planet.daisies.traits(actor.species)
    evolution = planet.evolution
    return {
        'optimum_c': round(float(species_traits.optimum_c), 1),
        'albedo': round(float(species_traits.albedo), 2),
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
