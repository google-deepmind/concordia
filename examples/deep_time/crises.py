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

"""Crises with countdowns: trouble a role can see coming.

A crisis is read from the role's own instruments (late and noisy as they
are) and its forecast, never from the truth: the climate leaving the band or
out of it with strain building, biodiversity under a goal's floor, a life
form dwindling, a people going hungry, lands near revolt, a deadline near.
Each says how many turns remain, where that can be told, and which of the
moves on offer (or being saved for) answer it, with their measured pushes.
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-long-ternary,g-short-docstring-punctuation

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any

from examples.deep_time import forecast as forecast_lib
from examples.deep_time.world import planet as planet_lib

# Crises shown at once, the most pressing first.
MOST = 3
# A forecast leaving the band within this many turns is a crisis.
SOON = 4
# A life form under this share of the planet, and shrinking, is in danger.
DWINDLING = 0.02
# Lands this close to their breaking point are a crisis.
RESTLESS = 0.7
# A push of this many degrees or more wrecks rather than answers a crisis.
STRONG_C = 15.0
# A deadline this near is a crisis, for goals that must be reached (not
# held) by the last turn.
DEADLINE = 8
REACH = {
    'seed_ship': 'A seed ship must leave',
    'industrialize': 'The energy must be made',
    'iron_age': 'The iron must be forged',
    'thaw': 'The ice must break',
    'green_islands': 'The islands must be green',
    'terraform': 'The colony must make its world',
}


def _payload(received: Mapping[str, Any], label: str) -> dict[str, Any]:
  """A role's last reading on an instrument, as data ({} if none or text)."""
  message = received.get(label)
  if message is None:
    return {}
  try:
    data = json.loads(message.payload)
  except (TypeError, ValueError):
    return {}
  return data if isinstance(data, dict) else {}


def _strong(push: str) -> bool:
  """A push that wrecks rather than steers (as the page draws it in red)."""
  if re.search(r'freezes|boils|all life|swings', push):
    return True
  return any(
      float(d) >= STRONG_C for d in re.findall(r'(\d+(?:\.\d+)?) °C', push)
  )


def _answers(
    options: Sequence[str], pushes: Mapping[str, str], pattern: str
) -> list[str]:
  """The options whose label or push matches `pattern`, at most three.

  A move that wrecks the world (a strong push) is never an answer.
  """
  found = re.compile(pattern, re.I)
  return [
      o
      for o in options
      if (found.search(o) or found.search(pushes.get(o, '')))
      and not _strong(pushes.get(o, ''))
  ][:3]


def _crisis(
    kind: str,
    title: str,
    detail: str,
    turns: int | None,
    answers: Sequence[str] = (),
    urgency: float = 0.0,
    where: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  return {
      'kind': kind,
      'title': title,
      'detail': detail,
      'turns': turns,
      'answers': list(answers),
      'urgency': urgency,
      # Where on the map it shows, as the role can know it: the
      # whole world ('world': 'hot' or 'cold'), a latitude ('lat', degrees,
      # where its census last counted most), a people ('people': name), a
      # herd ('herd': kind) or the colony ('site'); None if nowhere.
      'where': dict(where) if where else None,
  }


def lat_of(census: Mapping[str, Any], species: str) -> float | None:
  """The latitude (degrees) of the census band holding most of a species."""
  best, lat = 0.0, None
  for band in census.get('bands') or ():
    found = re.match(r'(\d+(?:\.\d+)?)°\s*([NS]?)', str(band.get('band', '')))
    share = float((band.get('cover') or {}).get(species, 0.0))
    if found and share > best:
      best = share
      lat = float(found.group(1)) * (-1.0 if found.group(2) == 'S' else 1.0)
  return lat


def _name(planet: planet_lib.Planet, species: str) -> str:
  """A life form by its public name ('Frost mats'), as the map shows it."""
  name = planet.species_name(species)
  return name[:1].upper() + name[1:]


def _at_lat(census: Mapping[str, Any], species: str) -> dict[str, Any] | None:
  lat = lat_of(census, species)
  return None if lat is None else {'lat': lat}


def steward_crises(
    planet: planet_lib.Planet,
    received: Mapping[str, Any],
    found: forecast_lib.Forecast | None,
    options: Sequence[str],
    pushes: Mapping[str, str],
) -> list[dict[str, Any]]:
  """The Steward's crises, from its instruments and forecast.

  Args:
    planet: The planet (for its public rules: the band, the strain a world
      bears, the goal's floor and deadline, the tick).
    received: The Steward's last reading on each instrument, by label.
    found: Its forecast.
    options: The moves on offer, and those being saved for.
    pushes: Each move's measured push.

  Returns:
    Crises, the most pressing first, at most `MOST`.
  """
  out = []
  low, high = planet.band()
  most = int(planet.scenario.get('max_ticks_out_of_band', 0))
  thermometer = _payload(received, 'Thermometer') or _payload(
      received, 'Climate sense'
  )
  strain = float(thermometer.get('out_of_band_ticks') or 0.0)
  cools = r'\bcools\b'
  warms = r'\bwarms\b'
  if found is not None and high - low <= 60:
    hot = found.now > high
    if hot or found.now < low:
      left = max(0, round(most - strain)) if most else None
      out.append(
          _crisis(
              'band',
              f'{"Too hot" if hot else "Too cold"}: about'
              f' {found.now:.0f} ± {found.now_spread:.0f} °C',
              (
                  f'Out of the band {low:g}–{high:g} °C; strain'
                  f' {strain:.0f} of {most} breaks the world.'
                  if most
                  else f'Out of the band {low:g}–{high:g} °C.'
              ),
              left,
              _answers(options, pushes, cools if hot else warms),
              urgency=3.0 + (1.0 / (1 + left) if left is not None else 0),
              where={'world': 'hot' if hot else 'cold'},
          )
      )
    elif found.leaves_in is not None and found.leaves_in <= SOON:
      top = found.edge == 'top'
      out.append(
          _crisis(
              'band',
              f'{"Heat" if top else "Cold"} coming',
              'Your forecast leaves the band by the'
              f' {found.edge} as your moves fade.',
              found.leaves_in,
              _answers(options, pushes, cools if top else warms),
              urgency=2.0 + 1.0 / found.leaves_in,
              where={'world': 'hot' if top else 'cold'},
          )
      )
  meter = _payload(received, 'Omega meter')
  floor = planet.goal().get('min_biodiversity')
  life = meter.get('biodiversity')
  if floor is not None and life is not None and float(life) < float(floor):
    out.append(
        _crisis(
            'life',
            f'Biodiversity {float(life):.2f}',
            f'The goal needs {float(floor):.2f} or more when it is met.',
            None,
            _answers(options, pushes, r'protect the wild|plant forests|sow'),
            urgency=1.5,
        )
    )
  census = _payload(received, 'Census')
  cover, growth = census.get('cover') or {}, census.get('growth') or {}
  for species, share in sorted(cover.items(), key=lambda kv: kv[1]):
    if 0 < float(share) < DWINDLING and float(growth.get(species, 1.0)) < 1.0:
      out.append(
          _crisis(
              'species',
              f'{_name(planet, species)} dwindling',
              f'{float(share):.1%} of the planet, and shrinking.',
              None,
              _answers(options, pushes, rf'seed {species}|sow {species}'),
              urgency=1.2,
              where=_at_lat(census, species),
          )
      )
      break
  peoples = (_payload(received, 'Peoples').get('polities') or {}).values()
  hungry = sorted(
      (float(p.get('food_ratio', 1.0)), str(p.get('name', '')))
      for p in peoples
      if float(p.get('food_ratio', 1.0)) < 0.95
  )
  if hungry:
    fed, name = hungry[0]
    out.append(
        _crisis(
            'famine',
            f'{name} go hungry',
            f'Fed {fed:.0%}: a people that starves long enough breaks.',
            None,
            urgency=1.4 + 2.0 * (1.0 - fed),
            where={'people': name},
        )
    )
  restless = meter.get('restless') or {}
  if restless:
    name, pressure = max(restless.items(), key=lambda kv: kv[1])
    if float(pressure) >= RESTLESS:
      out.append(
          _crisis(
              'unrest',
              f'Unrest among the {name}',
              f'Their lands are {float(pressure):.0%} of the way to striking'
              ' back.',
              None,
              urgency=1.0 + float(pressure),
              where={'people': name},
          )
      )
  vault = meter.get('vault') or {}
  if vault and float(vault.get('held', 0)) < float(vault.get('need', 0)):
    out.append(
        _crisis(
            'life',
            f'The seed vault holds {float(vault["held"]):.2f}',
            'A ship carries no more of the wild than the vault: it needs'
            f' {float(vault["need"]):g} when the arks leave.',
            None,
            _answers(options, pushes, r'gather the seeds'),
            urgency=1.3,
        )
    )
  # Side objectives near their deadline, as last read: optional,
  # so less pressing than the world's own trouble.
  rules = {str(r['key']): r for r in planet.side_rules()}
  for side in meter.get('side') or ():
    rule = rules.get(str(side.get('key')))
    left = int(side['until']) - planet.tick
    if (
        rule is None
        or side.get('state') != 'open'
        or planet.tick < int(side['from']) - 1
        or not 0 <= left <= SOON
    ):
      continue
    out.append(
        _crisis(
            'side',
            f'Side objective: +{float(side["reward"]):g} Ω',
            f'{side["text"]}.',
            left,
            _answers(options, pushes, rule['answers'])
            if rule.get('answers')
            else [],
            urgency=0.6 + 1.0 / (1 + left),
            # A life form's cover: where the census last counted most of it.
            where=_at_lat(census, str(rule['field'])[len('cover_') :])
            if str(rule['field']).startswith('cover_')
            else None,
        )
    )
  goal = planet.goal()
  left = planet.max_ticks - planet.tick
  if goal.get('kind') in REACH and 0 <= left <= DEADLINE:
    out.append(
        _crisis(
            'deadline',
            f'{left} turn{"s" if left != 1 else ""} left',
            f'{REACH[goal["kind"]]} by turn {planet.max_ticks}.',
            left,
            urgency=0.8 + 1.0 / (1 + left),
        )
    )
  out.sort(key=lambda c: -c['urgency'])
  return out[:MOST]


# A granary that feeds the people fewer seasons than this is a crisis.
GRANARY = 2.0


def polity_crises(
    planet: planet_lib.Planet,
    polity: str,
    options: Sequence[str],
    pushes: Mapping[str, str],
) -> list[dict[str, Any]]:
  """A people's crises, from its own storehouse.

  Hunger and an emptying granary (answered by more hands in the trades that
  make grain), its lands near revolt, and a shared goal's last turns. A
  people's own stores are its to see.

  Args:
    planet: The planet (its civilization holds the people).
    polity: The people's key.
    options: The people's moves.
    pushes: Each move's push.

  Returns:
    Crises, the most pressing first, at most `MOST`.
  """
  civ = planet.civ
  if civ is None or polity not in civ.polities:
    return []
  people = civ.polities[polity]
  out = []
  grain = [r.label for r in civ.recipes if r.outputs.get('grain')]
  feeds = (
      _answers(options, pushes, '|'.join(re.escape(g) for g in grain))
      if grain
      else []
  )
  feeds = [o for o in feeds if o.startswith('put more labor into')] or feeds
  eats = float(people.population) * float(civ.food_per_pop)
  stocks = civ.payload()['polities'][polity]['stocks']
  seasons = float(stocks.get('grain', 0.0)) / max(eats, 1e-9)
  fed = float(people.food_ratio)
  if fed < 0.95:
    out.append(
        _crisis(
            'famine',
            f'We go hungry: fed {fed:.0%}',
            'A people that starves long enough breaks. More hands in the'
            ' fields feed it.',
            None,
            feeds,
            urgency=2.0 + 2.0 * (1.0 - fed),
            where={'people': people.name},
        )
    )
  elif seasons < GRANARY:
    out.append(
        _crisis(
            'famine',
            'The granary is nearly empty',
            f'It holds {seasons:.1f} seasons of bread; then the harvest'
            ' alone must feed us.',
            max(0, int(seasons)),
            feeds,
            urgency=1.6 + (GRANARY - seasons) / GRANARY,
            where={'people': people.name},
        )
    )
  backlash = planet.backlash
  if backlash is not None:
    pressure = backlash.restless(planet.homes()).get(polity, 0.0)
    if float(pressure) >= RESTLESS:
      out.append(
          _crisis(
              'unrest',
              'Our lands are restless',
              f'{float(pressure):.0%} of the way to striking back at us.',
              None,
              urgency=1.0 + float(pressure),
              where={'people': people.name},
          )
      )
  goal = planet.goal()
  left = planet.max_ticks - planet.tick
  if goal.get('kind') in REACH and 0 <= left <= DEADLINE:
    out.append(
        _crisis(
            'deadline',
            f'{left} turn{"s" if left != 1 else ""} left',
            f'{REACH[goal["kind"]]} by turn {planet.max_ticks}.',
            left,
            urgency=0.8 + 1.0 / (1 + left),
        )
    )
  out.sort(key=lambda c: -c['urgency'])
  return out[:MOST]


# Seasons of bread past which a granary is plenty (as the page says it).
PLENTY = 6.0

# A shared goal the peoples make together: the good, its
# field in the planet's history, the verb for making it, and the goal's key
# for how much is needed.
SHARED = {
    'iron_age': ('iron', 'iron_made', 'forge', 'min_iron'),
    'industrialize': ('energy', 'energy_made', 'make', 'min_energy'),
}


def goal_good(planet: planet_lib.Planet) -> str | None:
  """The good the peoples' shared goal asks them to make, if any."""
  kind = planet.goal().get('kind')
  if kind in SHARED:
    return SHARED[str(kind)][0]
  return 'seed_ships' if kind == 'seed_ship' else None


def shared_goal(
    planet: planet_lib.Planet, polity: str
) -> dict[str, Any] | None:
  """What the age asks of a people, in its own numbers.

  The good the peoples must make together (made so far, needed, the pace
  the turns left ask for, and what all made last turn), or the
  seed ship's terms; and the floor under which a people breaks and ends the
  age for all, with this people's numbers against it. The census tells
  every people what all have made, and its own numbers are its own.

  Args:
    planet: The planet (its civilization holds the people).
    polity: The people's key.

  Returns:
    The goal's terms, or None where the peoples share no goal.
  """
  civ = planet.civ
  goal = planet.goal()
  kind = goal.get('kind')
  if civ is None or polity not in civ.polities:
    return None
  if kind not in SHARED and kind != 'seed_ship':
    return None
  start = float(civ.start_population.get(polity, 0.0))
  left = max(0, planet.max_ticks - planet.tick)
  people = civ.polities[polity]
  eats = float(people.population) * float(civ.food_per_pop)
  found: dict[str, Any] = {
      'kind': kind,
      # Seasons of bread in the granary: plenty frees hands.
      'seasons': round(civ.ledger.stock(polity, 'grain') / max(eats, 1e-9), 1),
      'by': planet.max_ticks,
      'left': left,
      'floor': float(goal.get('collapse_fraction', 0.0)),
      'numbers': (
          float(civ.polities[polity].population) / start if start else 1.0
      ),
  }
  if kind == 'seed_ship':
    found['min_biodiversity'] = float(goal['min_biodiversity'])
    found['held'] = _held(civ, polity, 'seed_ships')
    found['next_tech'] = _next_tech(civ, polity, 'seed_ships')
    return found
  good, field, verb, key = SHARED[str(kind)]
  found['held'] = _held(civ, polity, good)
  found['next_tech'] = _next_tech(civ, polity, good)
  made = float(civ.made(good))
  rows = planet.history
  last = (
      max(0.0, float(rows[-1][field]) - float(rows[-2][field]))
      if len(rows) >= 2
      else None
  )
  need = float(goal[key])
  found.update({
      'good': good,
      'verb': verb,
      'made': round(made, 1),
      'need': need,
      # What each turn left must bring, among all the peoples.
      'pace': round(max(0.0, need - made) / left, 1) if left else None,
      # What all made last turn (output grows with tools and techs, so no
      # straight line is drawn from it).
      'last': None if last is None else round(last, 1),
  })
  if kind == 'industrialize':
    found['max_co2'] = float(goal['max_co2'])
  return found


def _next_tech(civ: Any, polity: str, good: str) -> dict[str, Any] | None:
  """The tech that would most help make the good, while research is short

  of it; None when one is affordable now (it is on the menu).
  """
  if civ.best_tech(polity, good) is not None:
    return None
  tech = civ.best_tech(polity, good, affordable=False)
  if tech is None:
    return None
  return {
      'label': tech.label,
      'does': civ.describe_tech(tech.key),
      'cost': float(tech.cost),
      'research': round(civ.stock(polity, 'research'), 1),
  }


def _held(civ: Any, polity: str, good: str) -> dict[str, Any] | None:
  """What held back the people's making of the good (civ.held_back)."""
  short, recipe = civ.held_back(polity, good)
  if not short and not recipe:
    return None
  return {
      'good': good.replace('_', ' '),
      'short': [g.replace('_', ' ') for g in short],
      # The link to strengthen; None when no known recipe makes the last.
      'recipe': civ.recipe(recipe).label if recipe else None,
  }


def held_words(held: Mapping[str, Any] | None) -> str:
  """What held back the good, e.g.

  'our iron was held back by charcoal, and the charcoal by wood: more hands to
  cut wood would free it', or that no one works the trade that makes it.
  """
  if not held:
    return ''
  short = held['short']
  if not short:  # No hands in the trade that makes the good at all.
    return (
        f'hardly anyone works to {held["recipe"]}, so we make little'
        f' {held["good"]}: more hands there would make it'
    )
  verb = 'were' if held['good'].endswith('s') else 'was'  # Seed ships.
  text = f'our {held["good"]} {verb} held back by {short[0]}' + ''.join(
      f', and the {a} by {b}' for a, b in zip(short, short[1:])
  )
  if held['recipe']:
    return text + f': more hands to {held["recipe"]} would free it'
  return text + ', which we do not make: buy it, or learn to make it'


def pressing(
    planet: planet_lib.Planet,
    polity: str,
    options: Sequence[str],
    pushes: Mapping[str, str],
) -> list[dict[str, Any]]:
  """What matters most for a people this turn, as moves.

  The most urgent of its crises that a move answers, the tech that most
  helps its goal's chain if it can learn it now, and the move that
  strengthens the weakest link of its chain last generation, each once.

  Args:
    planet: The planet (its civilization holds the people).
    polity: The people's key.
    options: The people's moves.
    pushes: Each move's push.

  Returns:
    At most three: {'kind': the crisis kind, 'learn' or 'link', 'move',
    'title'}.
  """
  found = []
  for crisis in polity_crises(planet, polity, options, pushes):
    if crisis['answers']:
      found.append({
          'kind': crisis['kind'],
          'move': crisis['answers'][0],
          'title': crisis['title'],
      })
      break
  goal = shared_goal(planet, polity)
  held = goal.get('held') if goal else None
  civ = planet.civ
  good = goal_good(planet)
  tech = civ.best_tech(polity, good) if civ is not None and good else None
  if civ is not None and tech is not None:  # Worth learning now.
    move = f'learn {tech.label} ({tech.cost:g} research)'
    if move in options:
      found.append({
          'kind': 'learn',
          'move': move,
          'title': f'{tech.label} ({civ.describe_tech(tech.key)})',
      })
  if held and held['recipe']:
    move = f'put more labor into: {held["recipe"]}'
    if move in options and all(p['move'] != move for p in found):
      found.append({
          'kind': 'link',
          'move': move,
          'title': held_words(held),
      })
  return found


def pressing_words(found: Sequence[Mapping[str, Any]]) -> str:
  """What matters most, as the end of a people's question (models read the

  question last).
  """
  parts = []
  for item in found:
    if item['kind'] == 'link':
      parts.append(f'Our weakest link: {item["title"]}.')
    elif item['kind'] == 'learn':
      parts.append(f'Worth learning now: {item["title"]}.')
    else:
      parts.append(
          f'Most pressing now: {item["title"]}; {item["move"]} answers it.'
      )
  return ''.join(' ' + p for p in parts)


def shared_words(found: Mapping[str, Any] | None) -> str:
  """What the age asks of a people, as a sentence for its prompt."""
  if not found:
    return ''
  if found['kind'] == 'seed_ship':
    text = (
        ' What the age asks of us: a seed ship must leave for the stars from'
        ' a world still living (biodiversity'
        f' {found["min_biodiversity"]:g} or more) by turn {found["by"]}'
        f' ({found["left"]} turns left).'
    )
  else:
    made, need, good = found['made'], found['need'], found['good']
    text = (
        ' What the age asks of us: the peoples together must'
        f' {found["verb"]} {need:g} {good} by turn {found["by"]}'
    )
    if found['kind'] == 'industrialize':
      text += (
          f', and keep the air under {found["max_co2"]:g} times its old carbon'
      )
    if made >= need:
      text += f'. {made:g} so far: enough; now keep every people whole.'
    else:
      text += f'. {made:g} so far: {need - made:g} to go'
      if found['pace'] is not None:
        text += (
            f' in {found["left"]} turns, about {found["pace"]:g} a turn among'
            ' us all'
        )
      if found['last'] is not None:
        text += f'; last turn all peoples made {found["last"]:g}'
      text += '.'
  if found.get('held'):
    text += f' Last generation {held_words(found["held"])}.'
  if found.get('next_tech'):
    nxt = found['next_tech']
    text += (
        f' The next tech for it: {nxt["label"]} ({nxt["does"]}) at'
        f' {nxt["cost"]:g} research; we have {nxt["research"]:g}: hands to'
        ' study and teach bring it nearer.'
    )
  if found.get('seasons', 0) >= PLENTY:
    text += (
        f' Our granary holds {found["seasons"]:g} seasons of bread: plenty;'
        ' hands could go elsewhere.'
    )
  if found['floor']:
    text += (
        f' A people that falls under {found["floor"]:.0%} of its numbers'
        ' breaks, and the age ends for all; we stand at'
        f' {found["numbers"]:.0%} of ours.'
    )
  return text


def words(found: Sequence[Mapping[str, Any]]) -> str:
  """Crises as a sentence for a prompt (models read what the page shows)."""
  if not found:
    return ''
  parts = []
  for crisis in found:
    text = f'{crisis["title"]}'
    if crisis['turns'] is not None:
      turns = int(crisis['turns'])
      text += f' ({turns} turn{"" if turns == 1 else "s"})'
    text += f': {crisis["detail"]}'
    if crisis['answers']:
      text += ' Answer with: ' + '; '.join(crisis['answers']) + '.'
    parts.append(text)
  return ' Trouble now: ' + ' '.join(parts)


# A store running out within this many generations is a crisis; a sky
# behind its schedule is one in the goal's last this-many generations.
STORE_SOON = 5
SKY_SOON = 12


def colony_crises(
    report: Mapping[str, Any],
    needs: Mapping[str, float],
    recipes: Sequence[tuple[str, Mapping[str, float]]],
    options: Sequence[str],
    pushes: Mapping[str, str],
    age: int = 0,
) -> list[dict[str, Any]]:
  """A colony's crises, from its status report.

  The colonists read their colony as it is; Mission Control reads the report
  the light delay brought, `age` turns old, and its countdowns are as of
  then. A store the colony uses faster than it makes, a store already
  short, and a sky warming slower than the goal needs; each answered by the
  recipes that make what is short (as labor for the colonists, as a
  directive from home).

  Args:
    report: The colony's status (`Colony.payload`).
    needs: Each store's need per colonist a generation.
    recipes: Each recipe's label and outputs.
    options: The role's moves.
    pushes: Each move's push.
    age: Turns since the report left the colony.

  Returns:
    Crises, the most pressing first, at most `MOST`.
  """

  def makers(resource: str) -> list[str]:
    labels = [label for label, outs in recipes if outs.get(resource)]
    if not labels:
      return []
    return _answers(
        options, pushes, '|'.join(re.escape(label) for label in labels)
    )

  late = f' (reported {age} turns ago)' if age > 0 else ''
  stocks, made = report.get('stocks') or {}, report.get('made') or {}
  colonists = float(report.get('colonists') or 0.0)
  out = []
  for resource, short in sorted((report.get('shortfall') or {}).items()):
    if float(short) > 0:
      out.append(
          _crisis(
              'famine',
              f'Short of {resource}',
              f'The colony ran short of {resource} last generation{late}.',
              0,
              makers(resource),
              urgency=3.0,
              where={'site': True},
          )
      )
  for resource, per in sorted(needs.items()):
    need = float(per) * colonists
    gain = float(made.get(resource, 0.0))
    if gain >= need or any(c['title'].endswith(resource) for c in out):
      continue
    lasts = float(stocks.get(resource, 0.0)) / max(need - gain, 1e-9)
    if lasts <= STORE_SOON:
      out.append(
          _crisis(
              'famine',
              f'{resource.capitalize()} runs out',
              f'{gain:.1f} made for {need:.1f} needed a generation{late}.',
              max(0, int(lasts) - age),
              makers(resource),
              urgency=2.0 + 1.0 / (1 + lasts),
              where={'site': True},
          )
      )
  left = int(report.get('left') or 0)
  temp, target = report.get('temp_c'), report.get('target_temp_c')
  if temp is not None and target is not None and 0 < left <= SKY_SOON:
    wanted = (float(target) - float(temp)) / left
    if wanted > float(report.get('warmed') or 0.0) + 1e-9:
      out.append(
          _crisis(
              'band',
              f'The sky lags: {float(temp):.0f} °C of {float(target):g}',
              f'It must warm {wanted:.1f} °C a generation; it warmed'
              f' {float(report.get("warmed") or 0.0):+.1f}{late}.',
              max(0, left - age),
              makers('warmth'),
              urgency=1.5 + 1.0 / (1 + left),
              where={'site': True},
          )
      )
  out.sort(key=lambda c: -c['urgency'])
  return out[:MOST]


# A species under this share of the planet, and shrinking, is dwindling;
# one this many tolerances from its optimum is out of its climate.
SPECIES_LOW = 0.05
OFF_OPTIMUM = 2.0


def species_crises(
    planet: planet_lib.Planet,
    species: str,
    received: Mapping[str, Any],
    options: Sequence[str],
    pushes: Mapping[str, str],
) -> list[dict[str, Any]]:
  """A species' crises, from its census and its own genome.

  Dwindling (answered by spreading while it can, or going dormant to save
  the seed bank), its patches too cold or too hot for its optimum (going
  dormant waits it out), and a rival outgrowing it (spread, or offer a
  pact).

  Args:
    planet: The planet (its history holds the species' own traits).
    species: The species' key.
    received: The species' last reading on each instrument, by label.
    options: The species' moves.
    pushes: Each move's push.

  Returns:
    Crises, the most pressing first, at most `MOST`.
  """
  census = _payload(received, 'Census')
  cover = {k: float(v) for k, v in (census.get('cover') or {}).items()}
  growth = {k: float(v) for k, v in (census.get('growth') or {}).items()}
  local = census.get('local_temp_c') or {}
  if species not in cover:
    return []
  out = []
  own, grows = cover[species], growth.get(species, 1.0)
  name = planet.species_name(species)
  if own < SPECIES_LOW and grows < 1.0:
    out.append(
        _crisis(
            'species',
            f'We are dwindling: {own:.1%} of the planet',
            'Shrinking each turn; dormant seed waits out bad times.',
            None,
            _answers(options, pushes, r'go dormant|spread'),
            urgency=2.5 + (SPECIES_LOW - own) * 10,
            where=_at_lat(census, species),
        )
    )
  row = planet.history[-1] if planet.history else {}
  optimum, width = row.get(f'optimum_{species}'), row.get(
      f'tolerance_{species}'
  )
  temp = local.get(species)
  if optimum is not None and width and temp is not None:
    off = float(temp) - float(optimum)
    if abs(off) > OFF_OPTIMUM * float(width) and grows < 1.0:
      out.append(
          _crisis(
              'band',
              f'Too {"hot" if off > 0 else "cold"} for us:'
              f' {float(temp):.0f} °C',
              f'{name} thrive at {float(optimum):.0f} °C; our slow mind'
              ' evolves toward it, dormancy waits it out.',
              None,
              _answers(options, pushes, r'go dormant'),
              urgency=1.6 + abs(off) / (10 * float(width)),
              where=_at_lat(census, species),
          )
      )
  rivals = [
      (v, k)
      for k, v in cover.items()
      if k != species and v > 1.5 * own and growth.get(k, 1.0) > grows
  ]
  if rivals:
    share, rival = max(rivals)
    rival = planet.species_name(rival)
    out.append(
        _crisis(
            'species',
            f'The {rival} outgrow us',
            f'They hold {share:.0%} of the planet to our {own:.0%}.',
            None,
            _answers(
                options, pushes, rf'spread aggressively|pact to the {rival}'
            ),
            urgency=1.2 + share - own,
            where=_at_lat(census, species),
        )
    )
  out.sort(key=lambda c: -c['urgency'])
  return out[:MOST]


# Herds this much smaller than three turns ago are shrinking; meadows of
# their chosen petals under this share are thin.
HERDS_FALL = 0.15
MEADOWS_THIN = 0.12


def grazer_crises(
    planet: planet_lib.Planet,
    received: Mapping[str, Any],
    options: Sequence[str],
    pushes: Mapping[str, str],
) -> list[dict[str, Any]]:
  """The grazers' crises: shrinking herds, thin meadows.

  The herds know their own numbers and diet; the meadows they read from
  their census. Sparing the meadows or wandering to fresh ones answers both.
  """
  history = planet.history
  if not history or 'grazers' not in history[-1]:
    return []
  out = []
  now = float(history[-1]['grazers'])
  then = float(history[-4]['grazers']) if len(history) >= 4 else now
  if then > 0 and now < (1.0 - HERDS_FALL) * then:
    out.append(
        _crisis(
            'herds',
            f'Our herds shrink: {now / then - 1:+.0%} in 3 turns',
            'Hungry herds thin; spared or fresh meadows feed them.',
            None,
            _answers(options, pushes, r'browse lightly|wander'),
            urgency=2.0 + (1.0 - now / then),
            where={'herd': 'flock'},
        )
    )
  cover = _payload(received, 'Census').get('cover') or {}
  diet = float(history[-1].get('grazer_diet_white', 0.5))
  chosen = 'white' if diet >= 0.5 else 'black'
  meadow = float(cover.get(chosen, 1.0))
  if meadow < MEADOWS_THIN:
    out.append(
        _crisis(
            'life',
            f'The {chosen} meadows are thin: {meadow:.0%}',
            f'We prefer {chosen} petals'
            f' ({diet if chosen == "white" else 1 - diet:.0%} of what we eat);'
            ' wander, or spare them.',
            None,
            _answers(options, pushes, r'wander|browse lightly'),
            urgency=1.5 + (MEADOWS_THIN - meadow) * 5,
            where=_at_lat(_payload(received, 'Census'), chosen),
        )
    )
  out.sort(key=lambda c: -c['urgency'])
  return out[:MOST]
