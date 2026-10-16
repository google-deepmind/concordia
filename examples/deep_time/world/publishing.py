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

"""Publishes world state on the TopicBus and renders it for entities.

Payloads are JSON so that reflexes and reports can parse them; renderers turn
messages into the short sentences that entities read in their prompts.
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-long-ternary,missing-function-docstring

from collections.abc import Callable, Mapping, Sequence
import json
from typing import Any

from concordia.contrib.pubsub import topics
from examples.deep_time.world import planet as planet_lib

WORLD_SOURCE = 'world'
CLIMATE_SOURCE = 'climate model'
BIOSPHERE_SOURCE = 'biosphere'
GAIA_SOURCE = 'gaia'
RESOLUTIONS_TOPIC = 'world.resolutions'
CARBON_SOURCE = 'carbon cycle'
OCEAN_SOURCE = 'ocean'
CIV_SOURCE = 'civilization'
MARKET_SOURCE = 'market'
COLONY_SOURCE = 'colony'


def declare_topics(
    bus: topics.TopicBus, config: Mapping[str, Any]
) -> list[topics.TopicSpec]:
  """Declares the topics listed in `data/topics.yaml`."""
  specs = []
  for item in config['topics']:
    specs.append(
        bus.declare(
            topics.TopicSpec(
                name=item['name'],
                retention=topics.Retention(item.get('retention', 'latest')),
                replay_size=int(item.get('replay_size', 1)),
                description=item.get('description', ''),
            )
        )
    )
  return specs


def publish_state(
    bus: topics.TopicBus,
    planet: planet_lib.Planet,
    time_text: str,
    events: Sequence[planet_lib.WorldEvent] = (),
) -> None:
  """Publishes the planet's current state and this tick's events."""
  bus.publish(
      'world.tick',
      json.dumps({
          'tick': planet.tick,
          'max_ticks': planet.max_ticks,
          'time': time_text,
          'era': planet.era,
      }),
      source=WORLD_SOURCE,
  )
  bus.publish(
      'climate.global',
      json.dumps(planet.climate_payload()),
      source=CLIMATE_SOURCE,
  )
  bus.publish(
      'bio.census', json.dumps(planet.census_payload()), source=BIOSPHERE_SOURCE
  )
  bus.publish(
      'gaia.indices', json.dumps(planet.indices_payload()), source=GAIA_SOURCE
  )
  bus.publish(
      'ocean.global', json.dumps(planet.ocean_payload()), source=OCEAN_SOURCE
  )
  if planet.symbiosis is not None:
    bus.publish(
        'bio.pacts',
        json.dumps(
            planet.symbiosis.payload(planet.species_names(), planet.tick)
        ),
        source=BIOSPHERE_SOURCE,
    )
  if planet.civ is not None:
    bus.publish(
        'civ.census', json.dumps(planet.civ.payload()), source=CIV_SOURCE
    )
  if planet.treaties is not None and planet.civ is not None:
    civ = planet.civ
    names = {k: p.name for k, p in civ.polities.items()}
    kinds = planet.treaties.relevant(
        lambda need: any(
            civ.unlocked(k, need) or need in p.techs
            for k, p in civ.polities.items()
        )
    )
    bus.publish(
        'civ.treaties',
        json.dumps(planet.treaties.payload(names, planet.tick, kinds)),
        source=CIV_SOURCE,
    )
  if planet.gazette is not None and planet.civ is not None:
    bus.publish(
        'media.gazette',
        json.dumps(
            planet.gazette.payload(planet.civ, planet.broadsheet_evidence())
        ),
        source=CIV_SOURCE,
    )
  if planet.colony is not None:
    bus.publish(
        'colony.status',
        json.dumps(planet.colony.payload(planet.tick)),
        source=COLONY_SOURCE,
    )
  if planet.trade is not None:
    bus.publish(
        'econ.prices', json.dumps(planet.trade.payload()), source=MARKET_SOURCE
    )
  if planet.carbon is not None:
    bus.publish(
        'cycles.carbon',
        json.dumps(planet.cycles_payload()),
        source=CARBON_SOURCE,
    )
  for event in events:
    bus.publish(event.topic, event.text, source=WORLD_SOURCE)
    if event.where is not None and event.local:
      # Also on its region's local topic, for those who hear news of a place.
      region = int(planet.tiles.region[event.where])
      bus.publish(
          f'local.r{region}',
          json.dumps({
              'text': event.text,
              'row': event.where[0],
              'col': event.where[1],
              'region': region,
          }),
          source=WORLD_SOURCE,
      )


def _loads(message: topics.Message) -> dict[str, Any]:
  try:
    data = json.loads(message.payload)
  except json.JSONDecodeError:
    return {}
  return data if isinstance(data, dict) else {}


def render_climate(message: topics.Message) -> str:
  data = _loads(message)
  if not data:
    return message.payload
  low, high = data['band_c']
  text = (
      f'Global mean temperature {data["temp_c"]:.1f} C (habitable band'
      f' {low:g} to {high:g} C); sunlight {data["luminosity"]:.2f}; planetary'
      f' albedo {data["albedo"]:.2f}.'
  )
  if data.get('ice'):
    text += f' Ice covers {data["ice"]:.0%} of the planet.'
  if data.get('out_of_band_ticks'):
    text += f' Out of band for {data["out_of_band_ticks"]} ticks!'
  if data.get('bands'):
    text += (
        ' By latitude: '
        + ', '.join(f'{b["band"]} {b["temp_c"]:.0f} C' for b in data['bands'])
        + '.'
    )
  return text


def render_carbon(message: topics.Message) -> str:
  data = _loads(message)
  if not data:
    return message.payload
  stocks = data['reservoirs']
  fluxes = data.get('fluxes') or {}
  text = (
      f'CO2 x{data["relative_co2"]:.2f} of the start (greenhouse'
      f' {data.get("greenhouse_w_m2", data["forcing_w_m2"]):+.1f} W/m2).'
  )
  air = data.get('air')
  if air:
    text += (
        f' Air: oxygen {air["oxygen_pal"]:.4f} of today\'s level, methane'
        f' {air["methane"]:.2f}; iron left to soak up oxygen'
        f' {air["reductant"]:.2f}.'
    )
    if air.get('ozone') is not None and air['ozone'] < 1.0:
      text += f' Ozone shield {air["ozone"]:.0%}.'
  text += (
      f' Air holds {stocks["atmosphere"]:.0f} Gt C'
      f' (x{data["relative_co2"]:.2f}), ocean {stocks["ocean"]:.0f}, life'
      f' {stocks["biomass"]:.0f}, soil {stocks["soil"]:.0f}.'
  )
  if fluxes:
    net_ocean = fluxes.get('ocean_uptake', 0.0)
    net_life = fluxes.get('photosynthesis', 0.0) - fluxes.get('decay', 0.0)
    text += (
        ' Change in the air last tick: volcanoes'
        f' +{fluxes.get("outgassing", 0.0):.1f},'
        f' weathering -{fluxes.get("weathering", 0.0):.1f}, ocean'
        f' {-net_ocean:+.1f}, life {-net_life:+.1f}.'
    )
  if data.get('cap_co2') is not None:
    text += f' The goal caps the air at x{data["cap_co2"]:.2f} of the start.'
  tip = data.get('tipping_co2')
  if tip:
    text += (
        f' A dark haze may tip the ice from x{tip[0]:.1f} of the start, and'
        f' almost surely does above x{tip[1]:.1f} under a clear sky (volcanic'
        f' ash holds the ice); below x{tip[0]:.1f} it only warms a turn.'
    )
  return text


def render_ocean(message: topics.Message) -> str:
  data = _loads(message)
  if not data:
    return message.payload
  if data['sea_area'] <= 0:
    return 'This world has no seas.'
  text = (
      f'Seas cover {data["sea_area"]:.0%} of the planet at'
      f' {data["mean_sst_c"]:.0f} C; sea ice {data["sea_ice"]:.0%}; sea level'
      f' {data["sea_level_m"]:+.0f} m; pH {data["ph"]:.2f}.'
  )
  if data.get('dead_zone', 0.0) >= 0.01:
    text += (
        f' Dead zones choke {data["dead_zone"]:.0%} of the coastal seas'
        ' (fertiliser runoff).'
    )
  return text


def render_civ(message: topics.Message) -> str:
  data = _loads(message)
  if not data:
    return message.payload
  parts = []
  for p in data['polities'].values():
    stocks = p['stocks']
    work = ', '.join(f'{k} {v:.0%}' for k, v in sorted(p['allocation'].items()))
    parts.append(
        f'{p["name"]} ({p["home"]}): {p["population"]:.1f} people, fed'
        f' {min(1.0, p["food_ratio"]):.0%}; grain {stocks["grain"]:.1f}, wood'
        f' {stocks["wood"]:.1f}, charcoal {stocks["charcoal"]:.1f}, ore'
        f' {stocks["ore"]:.1f}, iron {stocks["iron"]:.1f}, tools'
        f' {stocks["tools"]:.1f}; labor: {work}; knows'
        f' {len(p.get("techs", []))} techs.'
    )
  made = data.get('made', {})
  return ' '.join(parts) + f' Iron forged so far: {made.get("iron", 0):.0f}.'


def render_prices(message: topics.Message) -> str:
  data = _loads(message)
  if not data:
    return message.payload
  prices = ', '.join(f'{k} {v:.2f}' for k, v in data['prices'].items())
  transit = ', '.join(
      f'{k} {v:g}' for k, v in data.get('in_transit', {}).items() if v > 0
  )
  text = f'Market prices in silver: {prices}.'
  if transit:
    text += f' On the roads: {transit}.'
  return text


def render_pacts(message: topics.Message) -> str:
  data = _loads(message)
  if not data:
    return message.payload
  pacts = [
      f'{" and ".join(p["partners"])} ({p["ticks_left"]} ticks left)'
      for p in data['pacts']
  ]
  pending = [f'{p["from"]} to {p["to"]}' for p in data['pending']]
  text = 'Pacts: ' + ('; '.join(pacts) if pacts else 'none') + '.'
  if pending:
    text += ' Offers awaiting an answer: ' + '; '.join(pending) + '.'
  return text


def render_treaties(message: topics.Message) -> str:
  data = _loads(message)
  if not data:
    return message.payload
  caps = data.get('caps', {'coal': data['coal_cap']})
  kinds = data.get('kinds') or {
      'coal': {'plural': 'coal treaties', 'unit': 'coal'},
      'fish': {'plural': 'fishing quotas', 'unit': 'fish'},
  }
  parts = []
  for kind, words in kinds.items():
    if kind not in caps:
      continue
    title = words['plural'][:1].upper() + words['plural'][1:]
    unit = words['unit']
    found = [
        f'{" and ".join(t["parties"])} ({t["ticks_left"]} generations left)'
        for t in data['treaties']
        if t.get('kind', 'coal') == kind
    ]
    parts.append(
        f'{title} (at most {caps[kind]:g} {unit} a generation each): '
        + ('; '.join(found) if found else 'none')
        + '.'
    )
  text = ' '.join(parts)
  if data['embargoes']:
    text += ' Embargoes: ' + '; '.join(
        ' and '.join(e) for e in data['embargoes']
    )
    text += '.'
  if data['breaches']:
    text += ' Treaties broken: ' + ', '.join(
        f'{k} {v}' for k, v in data['breaches'].items()
    )
    text += '.'
  return text


def render_gazette(message: topics.Message) -> str:
  data = _loads(message)
  if not data:
    return message.payload
  stances = data['stances']
  sheets = [
      f'{b["author"]}, for {stances.get(b["stance"], b["stance"])}:'
      f' "{b["title"]}" ({b["votes"]:+d} votes)'
      for b in data['broadsheets']
  ]
  consent = ', '.join(f'{k} {v:.0%}' for k, v in data['consent'].items())
  credibility = ', '.join(f'{k} {v:g}' for k, v in data['credibility'].items())
  sky = data['sky']
  return (
      f'The Gazette. Sky watch: the air holds {sky["relative_co2"]:.2f} times'
      ' its old carbon; the most restless land is at'
      f' {sky["restless"]:.0%} of breaking. Broadsheets: '
      + ('; '.join(sheets) if sheets else 'none yet')
      + f'. Consent for coal: {consent}. Credibility: {credibility}.'
  )


def render_colony(message: topics.Message) -> str:
  data = _loads(message)
  if not data:
    return message.payload
  stocks = ', '.join(f'{k} {v:.1f}' for k, v in data['stocks'].items())
  text = (
      f'{data["name"]} (generation {data["tick"]}): {data["colonists"]:.1f}'
      f' colonists; the sky reads {data["temp_c"]:.1f} C (habitable at'
      f' {data["target_temp_c"]:g} C); stores: {stocks}.'
  )
  if data.get('short_of_panels'):
    text += ' The solar farm is short of panels.'
  if data.get('storm'):
    text += ' A dust storm darkens the sky.'
  if data.get('shortfall'):
    text += (
        ' Short of: '
        + ', '.join(f'{k} ({v:.0%})' for k, v in data['shortfall'].items())
        + '.'
    )
  return text


def _herd_trend(herd: Mapping[str, Any]) -> str:
  """', up 40% in 3 ticks' (or down, or steady); '' without a past."""
  if 'change' not in herd:
    return ''
  change, over = float(herd['change']), int(herd['over'])
  ticks = f'{over} tick' + ('s' if over != 1 else '')
  if abs(change) < 0.05:
    return f', steady over {ticks}'
  return f', {"up" if change > 0 else "down"} {abs(change):.0%} in {ticks}'


def render_peoples(message: topics.Message) -> str:
  """The peoples as seen from afar: how many, and how well they eat.

  For the Steward, whose goal may hinge on no people starving: a people fed
  below what it needs is marked, a famine called one.
  """
  data = _loads(message)
  polities = data.get('polities')
  if not polities:
    return message.payload
  parts = []
  for _, p in polities.items() if isinstance(polities, dict) else polities:
    fed = float(p.get('food_ratio', 1.0))
    mark = ' FAMINE' if fed < 0.7 else (' hungry' if fed < 0.95 else '')
    parts.append(
        f'{p["name"]} {float(p["population"]):.1f} people, fed {fed:.0%}{mark}'
    )
  return 'The peoples: ' + '; '.join(parts) + '.'


def render_census(message: topics.Message) -> str:
  data = _loads(message)
  if not data:
    return message.payload
  noun = data.get('noun', 'daisies')
  parts = [
      f'{name} {noun} cover {cover:.0%} (local'
      f' {data["local_temp_c"][name]:.0f} C, growth {data["growth"][name]:.2f})'
      for name, cover in data['cover'].items()
  ]
  text = f'{"; ".join(parts)}; bare ground {data["bare"]:.0%}.'
  for name, herd in (data.get('fauna') or {}).items():
    densest = max(herd['bands'], key=herd['bands'].get)
    text += (
        f' {name.capitalize()} {herd["density"]:.3f} (prefer white petals'
        f' {herd["diet_white"]:.0%}, optimum {herd["optimum_c"]:.0f} C,'
        f' densest at {densest}).'
    )
  herds = data.get('herds') or {}
  if herds:
    text += (
        ' Herds: '
        + '; '.join(
            f'{name}: {h["herds"]} on the map, total size {h["size"]:g} ('
            + (
                'mostly near'
                f' {abs(h["lat"]):.0f}°{"N" if h["lat"] >= 0 else "S"}'
                if h['lat'] is not None
                else 'none left'
            )
            + _herd_trend(h)
            + ')'
            for name, h in herds.items()
        )
        + '.'
    )
  if data.get('bands'):
    names = list(data['cover'])
    text += (
        f' Where ({"/".join(names)}): '
        + ', '.join(
            f'{b["band"]} '
            + '/'.join(f'{b["cover"].get(n, 0):.0%}' for n in names)
            for b in data['bands']
        )
        + '.'
    )
  return text


def _side_state(objective: Mapping[str, Any], tick: int) -> str:
  """A side objective's state in words, as of `tick`."""
  if objective['state'] != 'open':
    return objective['state']
  if tick < objective['from'] - 1:
    return f'from turn {objective["from"]}'
  return f'open, {max(0, objective["until"] - tick)} turns left'


def render_indices(message: topics.Message) -> str:
  data = _loads(message)
  if not data:
    return message.payload
  return (
      f'Omega {data["omega"]:.1f}; biodiversity {data["biodiversity"]:.2f}'
      + (
          f' (the seas at {data["seas"]:.0%} of their life)'
          if data.get('seas') is not None
          else ''
      )
      + f'; stability {data["stability"]:.2f}.'
      + (
          f' Seed vault for the arks: {data["vault"]["held"]:.2f} of'
          f' {data["vault"]["need"]:g} (a ship carries no more of the wild'
          ' than the vault holds).'
          if data.get('vault')
          else ''
      )
      + ''.join(
          f' Side objective: {o["text"]} (+{o["reward"]:g} omega)'
          f' {_side_state(o, data["tick"])}.'
          for o in data.get('side') or ()
      )
      + (
          f' Instruments: {data["instruments"]}.'
          if data.get('instruments')
          else ''
      )
      + (
          ' Restless lands (pressure, share of the breaking point): '
          + ', '.join(f'{k} {v:.0%}' for k, v in data['restless'].items())
          + '.'
          if data.get('restless')
          else ''
      )
  )


RENDERERS: Mapping[str, Callable[[topics.Message], str]] = {
    'climate.global': render_climate,
    'bio.census': render_census,
    'gaia.indices': render_indices,
    'cycles.carbon': render_carbon,
    'ocean.global': render_ocean,
    'bio.pacts': render_pacts,
    'civ.census': render_civ,
    'econ.prices': render_prices,
    'civ.treaties': render_treaties,
    'media.gazette': render_gazette,
    'colony.status': render_colony,
}


def render_tick(message: topics.Message) -> str:
  data = _loads(message)
  if not data:
    return message.payload
  return (
      f'Tick {data["tick"]} of {data.get("max_ticks", "?")}'
      f' ({data["time"]}, {data["era"]} era).'
  )


# How topics read when they arrive as observations (TopicObservations).
OBSERVATION_RENDERERS: Mapping[str, Callable[[topics.Message], str]] = {
    'world.tick': render_tick,
}


def renderer_for(topic: str) -> Callable[[topics.Message], str]:
  return RENDERERS.get(topic, topics.payload)
