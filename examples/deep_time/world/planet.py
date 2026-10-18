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

"""The planet: deterministic world state advanced once per tick.

`Planet` owns the physics (Daisyworld for now), the scenario schedule
(luminosity ramp and shocks), the Steward's energy Ω, the derived indices and
the win/lose judgement. It knows nothing about entities, language models or
topics, so it can be tested and checkpointed on its own.
"""

# pylint: disable=g-complex-comprehension,g-doc-args,g-doc-return-or-yield,g-long-ternary,g-short-docstring-punctuation,missing-function-docstring

from collections.abc import Mapping, Sequence
import dataclasses
import math
import re
from typing import Any

from examples.deep_time.world import ararat as ararat_lib
from examples.deep_time.world import backlash as backlash_lib
from examples.deep_time.world import biomes as biomes_lib
from examples.deep_time.world import carbon as carbon_lib
from examples.deep_time.world import civ as civ_lib
from examples.deep_time.world import climate as climate_lib
from examples.deep_time.world import colony as colony_lib
from examples.deep_time.world import daisyworld
from examples.deep_time.world import data
from examples.deep_time.world import evolution as evolution_lib
from examples.deep_time.world import fauna as fauna_lib
from examples.deep_time.world import fire as fire_lib
from examples.deep_time.world import geography
from examples.deep_time.world import geology as geology_lib
from examples.deep_time.world import grid as grid_lib
from examples.deep_time.world import herds as herds_lib
from examples.deep_time.world import media as media_lib
from examples.deep_time.world import news as news_lib
from examples.deep_time.world import ocean as ocean_lib
from examples.deep_time.world import oxygen as oxygen_lib
from examples.deep_time.world import pollution as pollution_lib
from examples.deep_time.world import seedbank as seedbank_lib
from examples.deep_time.world import settlements as settlements_lib
from examples.deep_time.world import speciation as speciation_lib
from examples.deep_time.world import symbiosis as symbiosis_lib
from examples.deep_time.world import tilelife as tilelife_lib
from examples.deep_time.world import tiles as tiles_lib
from examples.deep_time.world import trade as trade_lib
from examples.deep_time.world import treaties as treaties_lib
import numpy as np

VICTORY = 'victory'
DEFEAT = 'defeat'


@dataclasses.dataclass(frozen=True)
class Intervention:
  """A Steward intervention from `data/interventions.yaml`."""

  key: str
  label: str
  cost: float
  cooldown: int
  effect: Mapping[str, Any]
  # Which way it pushes the world, and about how far (measured; shown to a
  # player beside the move).
  push: str = ''
  # What a measurement of the climate cannot show (e.g. 'greens bare land');
  # kept beside the push measured on each world (push_audit.py).
  note: str = ''

  @property
  def option(self) -> str:
    """The option text shown to the Steward."""
    if self.cost:
      return f'{self.label} [{self.cost:g} omega]'
    return self.label


def interventions_from_config(
    config: Mapping[str, Any],
) -> list[Intervention]:
  return [Intervention(**item) for item in config['interventions']]


def omen_words(text: str, effect: Mapping[str, Any]) -> tuple[str, str]:
  """An omen's sign and words from "<sign>: <words>" (or words alone).

  The sign is one the world can show (the first, if none is named); the
  words are cut to the effect's `max_words`.

  Args:
    text: What the Steward said.
    effect: The omen intervention's effect (its `signs` and `max_words`).

  Returns:
    (sign, words).
  """
  signs = [str(s) for s in effect['signs']]

  def bare(sign: str) -> str:
    return re.sub(r'^(a|an|the) ', '', sign.strip().lower())

  head, colon, rest = text.partition(':')
  sign = next((s for s in signs if colon and bare(head) == bare(s)), None)
  words = (rest if sign else text).strip().strip('"“”').split()
  return sign or signs[0], ' '.join(words[: int(effect['max_words'])])


GEO_EVENTS = 'geo.events'
CIV_EVENTS = 'civ.events'
COLONY_EVENTS = 'colony.events'
BIO_EVENTS = 'bio.events'
# Ticks of the map's life kept, for roles whose census is that late.
LIFE_SEEN_TICKS = 16
# A side objective's states.
SIDE_OPEN, SIDE_MET, SIDE_MISSED = 'open', 'met', 'missed'
# Intervention effects a person may aim at a place on the map.
AIMABLE = frozenset({
    'seed',
    'albedo_pulse',
    'seas',
    'cull',
    'rain',
    'cull_packs',
    'reserve',
    'transplant',
})
OCEAN_ALERTS = 'ocean.alerts'


def _herd_config(overrides: Mapping[str, Any]) -> dict[str, Any]:
  """The herds' config, with a scenario's per-species overrides merged in."""
  config = dict(data.load('herds.yaml'))
  config['species'] = [
      {**spec, **overrides.get(spec['key'], {})} for spec in config['species']
  ]
  return config


@dataclasses.dataclass(frozen=True)
class WorldEvent:
  """Something noteworthy, announced on a topic (and, if it has a place,

  on that region's local topic too).
  """

  topic: str
  text: str
  where: tuple[int, int] | None = None  # Tile (row from the south, column).
  # Whether a placed event is also local news (a planet-wide tally that only
  # ripples from where it shows most is not).
  local: bool = True


@dataclasses.dataclass
class TickReport:
  """What happened during one call to `Planet.advance`."""

  tick: int
  events: list[WorldEvent]

  def texts(self) -> list[str]:
    return [e.text for e in self.events]


class Planet:
  """World state for one scenario."""

  def __init__(
      self,
      scenario: Mapping[str, Any],
      physics: Mapping[str, Any],
      interventions: Sequence[Intervention],
      evolution: evolution_lib.Evolution | None = None,
      lithosphere: geology_lib.Lithosphere | None = None,
      climate_config: Mapping[str, Any] | None = None,
      carbon_config: Mapping[str, Any] | None = None,
      seed: int = 0,
      fauna_config: Mapping[str, Any] | None = None,
      symbiosis_config: Mapping[str, Any] | None = None,
      chemistry_config: Mapping[str, Any] | None = None,
      metabolisms: Mapping[str, str] | None = None,
      instrument_tiers: Sequence[Mapping[str, Any]] = (),
      civ_config: Mapping[str, Any] | None = None,
      colony_config: Mapping[str, Any] | None = None,
  ):
    """Initializes the planet at tick 0.

    Args:
      scenario: One entry of `data/scenarios.yaml`.
      physics: Contents of `data/daisyworld.yaml`.
      interventions: Interventions available to the Steward.
      evolution: Behaviour and adaptation rules (`data/species.yaml`).
      lithosphere: Seeded eruption process (`data/geology.yaml`), if any.
      climate_config: Regional climate (defaults to `data/climate.yaml`).
      carbon_config: Carbon cycle (`data/carbon.yaml`); None disables it.
      seed: Seed for the scenario's geography generator.
      fauna_config: Grazers (`data/fauna.yaml`); None for a world without.
      symbiosis_config: Pacts (`data/symbiosis.yaml`); None disables them.
      chemistry_config: Oxygen and methane (`data/oxygen.yaml`), if any.
      metabolisms: Species key -> 'anaerobic', 'oxygenic' or 'aerobic'.
      instrument_tiers: Instrument tiers (`data/instruments.yaml`), worst first.
        The scenario's `instruments` key names the starting tier; without it the
        Steward's readings are exact and immediate.
      civ_config: Civilization rules (`data/civ.yaml`); the scenario's
        `civilization.polities` lists the peoples. None: no civilization.
    """
    self._scenario = dict(scenario)
    self._interventions = {i.key: i for i in interventions}
    climate_config = climate_config or data.load('climate.yaml')
    ocean_config = data.load('ocean.yaml')
    self.grid = grid_lib.RegionGrid(**climate_config['grid'])
    land = geography.land_fraction(
        scenario.get('geography'), self.grid, seed, ocean_config
    )
    self.climate = climate_lib.EnergyBalanceClimate(
        climate_config, self.grid, land
    )
    self.daisies = daisyworld.from_config(
        physics,
        scenario['initial_cover'],
        self.climate,
        land,
        luminosity=self.luminosity_at(0),
        initial_temp_c=float(scenario.get('initial_temp_c', 0.0)),
        genome=evolution.genome_config if evolution else None,
    )
    # The tile map refines the regions; its land aggregates
    # back to exactly the geography above.
    self.tiles = tiles_lib.TileMap(
        self.grid, land, data.load('tiles.yaml'), seed
    )
    self.tiles.update_ice(
        self.climate.ice_fraction(self.daisies.temperatures_c())
    )
    # Climate layers and biomes per tile.
    self.biomes = biomes_lib.Biomes(self.tiles, data.load('biomes.yaml'))
    self.biomes.update(self.daisies.temperatures_c())
    # Sessile life grows on the tiles; regions aggregate them.
    self.tile_life = tilelife_lib.TileLife(
        self.tiles, self.biomes, self.daisies, data.load('tilelife.yaml')
    )
    self.daisies.tile_growth = self.tile_life.grow
    # Each species' cover at the start (some goals are measured against it).
    self.start_cover = {
        k: self.daisies.cover(k) for k in self.daisies.species()
    }
    # Islands: the land's connected patches (8 around, wrapping), largest
    # first; a scenario may start life on the largest alone.
    goal = dict(scenario.get('goal') or {})
    self.islands = self._find_islands(int(goal.get('min_tiles', 6)))
    self._landmasses: list[np.ndarray] | None = None  # Found when needed.
    if scenario.get('start_on_one_island') and self.islands:
      self._keep_life_on(self.islands[0])
    else:  # Seed banks wait on all land (a barren world's too).
      self.tile_life.reached = self.tiles.land.copy()
    # Wildfire runs through dry, thickly grown tiles.
    self.fire = fire_lib.Wildfire(
        self.tiles,
        self.biomes,
        {**data.load('fire.yaml'), **scenario.get('fire_overrides', {})},
        seed,
    )
    self.burned_tiles = 0
    # Where people chose their next moves should land (actor -> tile).
    self.targets: dict[str, tuple[int, int]] = {}
    # Wild woods the Steward protects: region -> the tick protection ends.
    # The peoples may not fell them.
    self.reserves: dict[int, int] = {}
    self._last_cut = np.zeros(self.grid.size)  # Forest felled last tick.
    # News with a place, kept a few ticks for the map's ripples.
    self.news_config = data.load('news.yaml')
    self.ripples: list[dict[str, Any]] = []
    # Herds and packs with positions, wherever there is food for them.
    self.herds = herds_lib.Herds(
        self.tiles,
        self.biomes,
        self.tile_life.covers if scenario.get('wildlife', True) else {},
        _herd_config(scenario.get('herd_overrides', {})),
        seed,
        fisheries=bool(
            civ_config is not None
            and scenario.get('civilization')
            and scenario.get('wildlife', True)
        ),
        flocks=fauna_config is not None,
    )
    self.species_noun = str(scenario.get('species_noun', 'daisies'))
    self.chemistry: oxygen_lib.Chemistry | None = None
    if chemistry_config is not None:
      self.chemistry = oxygen_lib.Chemistry(chemistry_config, metabolisms or {})
      # Methane warms the starting climate before anything else is measured.
      self.daisies.set_greenhouse_forcing(self.chemistry.forcing())
      self.daisies.solve(self.luminosity_at(0))
      # So the tiles show the warmed start, not the bare one.
      self.tiles.update_ice(
          self.climate.ice_fraction(self.daisies.temperatures_c())
      )
      self.biomes.update(self.daisies.temperatures_c())
    self.civ: civ_lib.Civilization | None = None
    if civ_config is not None and scenario.get('civilization'):
      self.civ = civ_lib.Civilization(
          civ_config,
          scenario['civilization']['polities'],
          self.grid,
          self.daisies.land_fraction(),
          seed=seed,
          techs=civ_config.get('techs', ()),
      )
    self.soot_albedo = float((civ_config or {}).get('soot_albedo', 0.0))
    self.treaties: treaties_lib.Treaties | None = None
    if (
        self.civ is not None
        and civ_config
        and scenario['civilization'].get('treaties')
    ):
      self.treaties = treaties_lib.Treaties(civ_config['treaties'])
    self.colony = (
        colony_lib.Colony(
            {'deadline': int(scenario['max_ticks']), **colony_config}, seed=seed
        )
        if colony_config is not None
        else None
    )
    # The colony's own world on tiles.
    self.colony_world: ararat_lib.ColonyWorld | None = None
    if self.colony is not None:
      self.colony_world = ararat_lib.ColonyWorld(
          self.colony,
          data.load('ararat.yaml'),
          climate_config,
          ocean_config,
          data.load('tiles.yaml'),
          data.load('biomes.yaml'),
          seed,
      )
      # The mine's yield follows its site, once any survey has arrived.
      self.colony.site_factor = self.colony_world.mine_factor
    self.gazette: media_lib.Gazette | None = None
    self.attack_memory = 0
    if self.civ is not None and civ_config and civ_config.get('media'):
      self.gazette = media_lib.Gazette(
          civ_config['media'],
          {k: p.name for k, p in self.civ.polities.items()},
      )
      self.attack_memory = int(civ_config['media']['attack_memory_ticks'])
    # Polity -> the last tick the land struck it.
    self.struck: dict[str, int] = {}
    # Seed ships: (tick, polity, biodiversity at launch), and whole ships
    # counted per people.
    self.launches: list[tuple[int, str, float]] = []
    # The arks' seed vault: the seeds of the wild the
    # Steward has gathered for them; a ship carries no more of the world's
    # life than the vault holds.
    self.vault = 0.0
    # Side objectives: the scenario's optional tasks, each
    # judged on the true world within its window and paid in omega when met.
    self.side = {
        str(rule['key']): SIDE_OPEN for rule in self._scenario.get('side') or ()
    }
    self.ships_launched: dict[str, int] = {}
    self.backlash: backlash_lib.Backlash | None = None
    if self.civ is not None and civ_config and civ_config.get('backlash'):
      self.backlash = backlash_lib.Backlash(
          civ_config['backlash'], self.grid, seed=seed
      )
    self.trade: trade_lib.Trade | None = None
    if self.civ is not None and civ_config and civ_config.get('trade'):
      self.trade = trade_lib.Trade(civ_config['trade'], self.civ)
      if self.treaties is not None:
        self.trade.blocked = self.treaties.embargoed
    # The peoples' settlements on the map, the land they work and their
    # roads.
    self.settlements: settlements_lib.Settlements | None = None
    if self.civ is not None:
      self.settlements = settlements_lib.Settlements(
          self.tiles,
          self.biomes,
          {k: p.regions for k, p in self.civ.polities.items()},
          {k: p.home for k, p in self.civ.polities.items()},
          {k: p.name for k, p in self.civ.polities.items()},
          data.load('settlements.yaml'),
          trade=self.trade is not None,
          conquest=bool(scenario['civilization'].get('conquest')),
          arms=(civ_config or {}).get('arms'),
          war=(civ_config or {}).get('war'),
      )
      self.settlements.update(
          {k: p.population for k, p in self.civ.polities.items()}
      )
    # Where their soot settles and their runoff reaches the sea.
    self.pollution: pollution_lib.Pollution | None = None
    if self.settlements is not None:
      self.pollution = pollution_lib.Pollution(
          self.tiles, self.biomes, data.load('pollution.yaml')
      )
    self.carbon: carbon_lib.CarbonCycle | None = None
    if carbon_config is not None:
      self.carbon = carbon_lib.CarbonCycle(
          carbon_config,
          reference_temp_c=float(
              scenario.get(
                  'carbon_reference_temp_c',
                  self.daisies.global_temperature_c(),
              )
          ),
          initial_cover=sum(
              self.daisies.cover(s) for s in self.daisies.species()
          ),
          land=self.daisies.land(),
      )
    self.speciation: speciation_lib.Speciation | None = None
    if (
        evolution is not None
        and evolution.speciation_config
        and scenario.get('speciation', True)
    ):
      self.speciation = speciation_lib.Speciation(evolution.speciation_config)
    self.new_species: list[speciation_lib.Split] = []
    self.symbiosis: symbiosis_lib.Symbiosis | None = None
    if symbiosis_config is not None:
      self.symbiosis = symbiosis_lib.Symbiosis(symbiosis_config)
    self.grazers: fauna_lib.Grazers | None = None
    if fauna_config is not None:
      self.grazers = fauna_lib.Grazers(fauna_config, self.grid, land)
      self.herds.place_flocks(0, self.grazers, self.tile_life.covers, False)
    self.ocean = ocean_lib.Ocean(
        ocean_config,
        reference_ocean_carbon=(
            self.carbon.amount(carbon_lib.OCEAN) if self.carbon else 1.0
        ),
    )
    self._update_ocean()
    self.tick = 0
    self.era = str(scenario['era'])
    self.luminosity = self.luminosity_at(0)
    self.omega = float(scenario['omega']['start'])
    self.out_of_band = 0.0
    self.instrument_tiers = [dict(t) for t in instrument_tiers]
    start = scenario.get('instruments')
    keys = [t['key'] for t in self.instrument_tiers]
    self.instrument_tier: int | None = (
        keys.index(start) if start in keys else None
    )
    self.outcome: str | None = None
    self.outcome_reason = ''
    self.history: list[dict[str, Any]] = []
    # The life on each tile at the last few ticks: a role
    # sees the map's life as its own census saw it, as late as that.
    self.life_seen: dict[int, dict[str, np.ndarray]] = {}
    # And how dry and fuelled each tile was: its fire danger.
    self.risk_seen: dict[int, np.ndarray] = {}
    # Aimed sowings, where they took root: {'tick' (the first tick
    # whose life holds them), 'species', 'tiles' (a mask, south first)}.
    self.plantings: list[dict[str, Any]] = []
    self._recent_temps: list[float] = []
    self._ready_at: dict[str, int] = {}
    # The paid moves of the game, (tick, key), and what each climate
    # move does turn by turn on this world (push_audit's profiles): the
    # Steward's dead reckoning from its late readings.
    self.made: list[tuple[int, str]] = []
    # The Steward's omens, as (tick, sign, words), oldest first.
    self.omens: list[tuple[int, str, str]] = []
    self.move_profiles: dict[str, list[float]] = {}
    self.evolution = evolution
    self.lithosphere = lithosphere
    self.pending_adaptations: list[evolution_lib.PendingAdaptation] = []
    self.adaptation_counts = {s: 0 for s in self.daisies.species()}
    # The last draft (a mind's decision published on a topic, by its bus
    # seq) each entity enacted by an action, by '<entity>:<kind>'.
    self.enacted: dict[str, int] = {}
    self.behaviours = {s: 'hold' for s in self.daisies.species()}
    self.seed_banks: seedbank_lib.SeedBanks | None = None
    if evolution is not None and evolution.seed_bank_config:
      config = evolution.seed_bank_config
      land = max(1e-9, self.daisies.land())
      # Only species established above the regrowth threshold have seeds;
      # those at the cover floor exist only on paper.
      established = [
          s
          for s in self.daisies.species()
          if self.daisies.cover(s) / land
          >= float(config['reseed']['below_cover'])
      ]
      self.seed_banks = seedbank_lib.SeedBanks(
          config, self.daisies.species(), present=established
      )
    self._record()

  @property
  def scenario(self) -> Mapping[str, Any]:
    return self._scenario

  @property
  def max_ticks(self) -> int:
    return int(self._scenario['max_ticks'])

  def luminosity_at(self, tick: int) -> float:
    """Luminosity ramps as start + (end - start) * frac ** curve, then holds."""
    schedule = self._scenario['luminosity']
    ramp = int(schedule.get('ramp_ticks', self.max_ticks))
    frac = min(1.0, max(0.0, tick / max(1, ramp)))
    span = schedule['end'] - schedule['start']
    return schedule['start'] + span * frac ** schedule.get('curve', 1.0)

  def almanac(self, ticks: int = 5) -> str:
    """What the Advisor alone knows: the sun and the eruptions ahead."""
    now = self.luminosity_at(self.tick)
    later = self.luminosity_at(self.tick + ticks)
    change = later / now - 1.0 if now > 0 else 0.0
    text = (
        f'Almanac (known only to the advisor): the sun gives {now:.3f} now'
        f' and will give {later:.3f} in {ticks} ticks ({change:+.1%}).'
    )
    if self.lithosphere is not None:
      ahead = self.lithosphere.forecast(ticks)
      if ahead:
        text += (
            ' The geologists expect '
            + '; '.join(
                f'a magnitude-{e.magnitude} eruption at {e.volcano} in {k}'
                f' tick{"s" if k > 1 else ""}'
                for k, e in ahead
            )
            + '.'
        )
      else:
        text += f' The geologists expect no eruption in the next {ticks} ticks.'
    return text

  def temperature_c(self) -> float:
    return self.daisies.global_temperature_c()

  def band(self) -> tuple[float, float]:
    low, high = self._scenario['band_c']
    return float(low), float(high)

  def in_band(self) -> bool:
    low, high = self.band()
    return low <= self.temperature_c() <= high

  def living_cover(self) -> float:
    return sum(self.daisies.cover(s) for s in self.daisies.species())

  def cover_of_metabolism(self, metabolism: str) -> float:
    if self.chemistry is None:
      return 0.0
    return self.chemistry.cover_by_metabolism(self.daisies)[metabolism]

  def sea_health(self) -> float | None:
    """How alive the seas are: the fish stock, less its dead zones.

    Returns:
      A share in [0, 1], or None on worlds without fisheries.
    """
    if (
        self.civ is None
        or self.civ.fish_regrowth <= 0  # No fisheries on this world.
        or self.ocean.marine_weight <= 0
    ):
      return None
    return self.civ.fish_stock * (1.0 - self.ocean.dead_zone)

  def biodiversity(self) -> float:
    """Shannon evenness of species covers, scaled by living cover, in [0, 1].

    On worlds with fisheries, emptied or choking seas weigh it down:
    x (1 - marine weight x (1 - sea health)).
    """
    covers = [self.daisies.cover(s) for s in self.daisies.species()]
    total = sum(covers)
    if total <= 0 or len(covers) < 2:
      return 0.0
    entropy = -sum(c / total * math.log(c / total) for c in covers if c > 0)
    land = entropy / math.log(len(covers)) * min(1.0, total)
    health = self.sea_health()
    if health is None:
      return land
    return land * (1.0 - self.ocean.marine_weight * (1.0 - health))

  def stability(self) -> float:
    """1 / (1 + variance of recent temperatures), in (0, 1]."""
    temps = self._recent_temps
    if len(temps) < 2:
      return 1.0
    mean = sum(temps) / len(temps)
    variance = sum((t - mean) ** 2 for t in temps) / len(temps)
    return 1.0 / (1.0 + variance)

  def _extinct_floor(self) -> float:
    """Cover (a share of the planet) at which only the seed bank is left.

    Seed banks wait on the land plants have reached, `min_cover` of each
    tile's land: half as much again is the floor.
    """
    banked = (self.tiles.land & self.tile_life.reached).astype(float)
    share = self.climate.global_mean(self.tile_life.region_mean(banked))
    return self.daisies.params.min_cover * 1.5 * float(share)

  def is_extinct(self, species: str) -> bool:
    """True when only the seed bank (minimum cover) is left."""
    return self.daisies.cover(species) <= self._extinct_floor()

  def is_lifeless(self) -> bool:
    floor = self._extinct_floor()
    return all(self.daisies.cover(s) <= floor for s in self.daisies.species())

  def advance(self) -> TickReport:
    """Advances the world by one tick and judges the scenario."""
    self.tick += 1
    self.luminosity = self.luminosity_at(self.tick)
    events = self._phase_in_adaptations()
    for shock in self._scenario.get('shocks', ()):
      if shock['tick'] == self.tick:
        events.append(self._apply_shock(shock))
    magnitude = 0.0
    if self.lithosphere is not None:
      for eruption in self.lithosphere.step():
        magnitude += eruption.magnitude
        if self.civ is not None:
          self.civ.add_ore(eruption.magnitude)
        self.daisies.add_albedo_pulse(eruption.ash_albedo)
        self.daisies.add_fertility(
            eruption.fertility, self.lithosphere.fertility_decay
        )
        events.append(WorldEvent(GEO_EVENTS, eruption.describe()))
    if self.symbiosis is not None:
      for key in self.symbiosis.expire(self.tick):
        names = ' and '.join(self.species_name(s) for s in key.split('+'))
        events.append(
            WorldEvent(BIO_EVENTS, f'The pact between {names} lapses.')
        )
      self.daisies.set_mutualism(self.symbiosis.multipliers(self.daisies))
    if self.carbon is not None:
      self.carbon.step(
          temp_c=self.temperature_c(),
          cover=self.living_cover(),
          land=self.ice_free_land(),
          eruption_magnitude=magnitude,
          sea_ice=self.ocean.sea_ice,
      )
      self.daisies.set_co2_factor(self.carbon.co2_limitation())
    if self.chemistry is not None:
      for text in self.chemistry.step(self.daisies, self.tick):
        events.append(WorldEvent(BIO_EVENTS, text))
      self.daisies.set_growth_modifiers(
          'chemistry',
          self.chemistry.multipliers(self.daisies, self.climate.insolation),
      )
    self.daisies.set_greenhouse_forcing(self.greenhouse_forcing())
    self._spend_seed_banks()
    self.daisies.step(self.luminosity)
    self.tiles.update_ice(
        self.climate.ice_fraction(self.daisies.temperatures_c())
    )
    for text in self.biomes.update(self.daisies.temperatures_c()):
      # Placed so the map ripples from where it shows most, but a tally of
      # the whole planet is not local news.
      events.append(
          WorldEvent(
              BIO_EVENTS, text, where=news_lib.place_of(text), local=False
          )
      )
    events.extend(self._burn())
    events.extend(self._roam())
    events.extend(self._reseed())
    for alert in self._update_ocean():
      events.append(WorldEvent(OCEAN_ALERTS, alert))
    if self.grazers is not None:
      for text in self.grazers.step(self.daisies):
        events.append(WorldEvent(BIO_EVENTS, text))
      self.herds.place_flocks(self.tick, self.grazers, self.tile_life.covers)
    events.extend(self._speciate())
    if self.civ is not None:
      if self.settlements is not None and self.pollution is not None:
        # What each people's towns can work this tick scales its harvests,
        # and its own coast's dead zones its catch.
        crops = self.pollution.crop_loss()
        local = self.settlements.local(self.tile_life.covers, crops)
        for key, factors in local.items():
          factors['fish'] = self.pollution.fishing(
              self.settlements.tiles_of(key)
          )
          self.civ.polities[key].local = factors
      for text in self._step_civ():
        events.append(WorldEvent(CIV_EVENTS, text))
      for text in self._muster():  # Arms join their garrisons and eat.
        events.append(WorldEvent(CIV_EVENTS, text))
      events.extend(self._wars())  # Armies march, fight, towns rise.
      if self.settlements is not None and self.pollution is not None:
        for text in self.settlements.update(
            {k: p.population for k, p in self.civ.polities.items()},
            # Where conquest is played, only a people's move founds a town.
            found=not self.settlements.conquest,
        ):
          events.append(
              WorldEvent(CIV_EVENTS, text, where=news_lib.place_of(text))
          )
        self.pollution.step(
            {
                k: (self.settlements.tiles_of(k), p.emissions)
                for k, p in self.civ.polities.items()
            },
            self.ocean.dead_zone,
        )
    if self.civ is not None:
      events.extend(self._launch_seed_ships())
      text = self._step_orbit()
      if text:
        events.append(WorldEvent(CIV_EVENTS, text))
    if self.trade is not None:
      for text in self.trade.step():
        events.append(WorldEvent(CIV_EVENTS, text))
    if self.treaties is not None and self.civ is not None:
      usage = {
          kind: {
              k: p.emissions.get(spec.metric, 0.0)
              for k, p in self.civ.polities.items()
          }
          for kind, spec in self.treaties.kinds.items()
      }
      for breaker, wronged in self.treaties.check(usage, self.tick):
        events.append(
            WorldEvent(
                CIV_EVENTS,
                f'The {self.civ.polities[breaker].name} break their'
                f' treaty with the {self.civ.polities[wronged].name}: trade'
                ' between them stops for'
                f' {self.treaties.embargo_ticks} generations.',
            )
        )
    if self.backlash is not None:
      for text in self._step_backlash():
        events.append(WorldEvent(CIV_EVENTS, text))
    if self.gazette is not None and self.civ is not None:
      self.gazette.step(self.civ)
    if self.colony is not None:
      for text in self.colony.step():
        events.append(WorldEvent(COLONY_EVENTS, text))
      if self.colony_world is not None:  # The thaw, on the colony's tiles.
        for text in self.colony_world.update():
          events.append(WorldEvent(COLONY_EVENTS, text))

    temp = self.temperature_c()
    window = int(self._scenario.get('stability_window', 5))
    self._recent_temps = (self._recent_temps + [temp])[-window:]
    omega_cfg = self._scenario['omega']
    self.omega += (
        omega_cfg['per_tick']
        + omega_cfg.get('biodiversity_bonus', 0.0) * self.biodiversity()
    )
    self.out_of_band = self._next_out_of_band()
    self._judge()
    self._record()
    self._judge_side()
    # News with a place spreads over the map from where it happened.
    keep = int(self.news_config['ripple_ticks'])
    self.ripples = [r for r in self.ripples if self.tick - r['tick'] < keep]
    for event in events:
      if event.where is not None:
        self.ripples.append({
            'tick': self.tick,
            'row': event.where[0],
            'col': event.where[1],
            'topic': event.topic,
            'text': event.text,
            'local': event.local,  # A planet-wide tally is not local news.
        })
    return TickReport(tick=self.tick, events=events)

  def region_steps(self, one: int, two: int) -> int:
    """Steps between two regions over the grid (wrapping in longitude)."""
    bands, sectors = self.grid.bands, self.grid.sectors
    r1, c1 = divmod(one, sectors)
    r2, c2 = divmod(two, sectors)
    dc = min((c1 - c2) % sectors, (c2 - c1) % sectors)
    return max(abs(r1 - r2), dc) if bands else 0

  def _evolution(self) -> evolution_lib.Evolution:
    if self.evolution is None:
      raise ValueError('This planet has no evolution rules.')
    return self.evolution

  def cycles_payload(self) -> dict[str, Any]:
    """The carbon gauge's reading: carbon reservoirs, greenhouse and air."""
    payload = self.carbon.payload() if self.carbon else {}
    payload['greenhouse_w_m2'] = round(self.greenhouse_forcing(), 2)
    if self.chemistry is not None:
      payload['air'] = self.chemistry.payload()
    goal = self.goal()
    tip = goal.get('tip_co2')
    if tip:  # Where a dark haze may tip a frozen world, and surely does.
      payload['tipping_co2'] = [float(x) for x in tip]
    if goal.get('max_co2') is not None:  # The goal's cap on the air.
      payload['cap_co2'] = float(goal['max_co2'])
    return payload

  def greenhouse_forcing(self) -> float:
    """Total greenhouse forcing: CO2 plus methane, in W m^-2."""
    forcing = 0.0
    if self.carbon is not None:
      forcing += self.carbon.forcing()
    if self.chemistry is not None:
      forcing += self.chemistry.forcing()
    return forcing

  def species_name(self, species: str) -> str:
    if self.grazers is not None and species == self.grazers.key:
      return self.grazers.name
    if self.evolution is None:
      return f'{species} {self.species_noun}'
    return self.evolution.info(species).name

  def _spend_seed_banks(self) -> None:
    """Behaviours spend or refill seed banks; empty banks exhaust spreaders."""
    banks = self.seed_banks
    if banks is None or self.evolution is None:
      return
    land = max(1e-9, self.daisies.land())
    for species in self.daisies.species():
      key = self.behaviours.get(species, 'hold')
      behaviour = self.evolution.behaviour(key)
      if banks.spend(species, key, self.daisies.cover(species) / land):
        self.daisies.set_behaviour(
            species, banks.exhausted_growth, banks.exhausted_death
        )
      else:
        self.daisies.set_behaviour(species, behaviour.growth, behaviour.death)

  def _reseed(self) -> list[WorldEvent]:
    """Nearly vanished species sprout again from their seed banks."""
    banks, events = self.seed_banks, []
    # Seeds wait out a world outside the band life can grow in.
    if banks is None or not self.in_band():
      return events
    land = max(1e-9, self.daisies.land())
    for species in self.daisies.species():
      amount = banks.reseed(species, self.daisies.cover(species) / land)
      planted = self._sow(species, amount) if amount > 0 else 0.0
      if planted > 0:
        banks.charge(species, planted)
        events.append(
            WorldEvent(
                BIO_EVENTS,
                f'The {self.species_name(species)} sprout again from their'
                ' seed bank.',
            )
        )
    return events

  def set_behaviour(
      self, species: str, behaviour_key: str, actor: str | None = None
  ) -> str:
    """Sets a species' behaviour for the coming ticks.

    A person playing it may have aimed it (`set_target`): spreading then
    pushes growth around the tapped tile this tick (and less elsewhere), and
    wandering herds drift there.

    Args:
      species: The species (or the grazers' key).
      behaviour_key: The chosen behaviour.
      actor: The acting entity, whose target (if any) is used.

    Returns:
      What happened.
    """
    target = self.targets.pop(actor, None) if actor else None
    where = self._place_words(target) if target is not None else ''
    if self.grazers is not None and species == self.grazers.key:
      text = self.grazers.set_behaviour(behaviour_key)
      if target is not None and behaviour_key == 'wander':
        self.grazers.drift(self._target_regions(target))
        text = text.rstrip('.') + where + '.'
      return text
    behaviour = self._evolution().behaviour(behaviour_key)
    self.daisies.set_behaviour(species, behaviour.growth, behaviour.death)
    self.behaviours[species] = behaviour.key
    if target is not None and behaviour.key == 'spread':
      self.daisies.set_focus(
          species, 0.85 + 0.75 * self._target_regions(target)
      )
    return f'{self.species_name(species)} {behaviour.label}{where}.'

  def aim_felling(self, polity: str, actor: str) -> str:
    """Points a people's woodcutters at the woods a person tapped."""
    target = self.targets.pop(actor, None)
    if target is None or self.civ is None:
      return ''
    self.civ.polities[polity].felling = self._target_regions(target)
    return self._place_words(target)

  def aim_mine(self, actor: str, by_wire: bool = False) -> str:
    """Points the colony's ice mine at the tile a person tapped.

    Args:
      actor: The acting entity, whose target (on New Ararat) is used.
      by_wire: Sent from home: the site arrives with the light delay.

    Returns:
      Words on where (empty when nothing was aimed).
    """
    target = self.targets.pop(actor, None)
    if target is None or self.colony is None or self.colony_world is None:
      return ''
    lat = float(self.colony_world.biomes.lat[target[0]])
    where = f' around {abs(lat):.0f}°{"N" if lat >= 0 else "S"} of New Ararat'
    if by_wire:
      self.colony.transmit('site', f'{target[0]},{target[1]}')
    else:
      self.colony.site = target
    return where

  def founding_cost(self) -> dict[str, float]:
    """What founding a town costs a people (where conquest is played)."""
    if self.settlements is None:
      return {}
    return {
        k: float(v)
        for k, v in (self.settlements.config.get('founding') or {}).items()
    }

  def can_found(self, polity: str) -> str:
    """Why a people cannot found a town now ('' if it can)."""
    towns, civ = self.settlements, self.civ
    if towns is None or civ is None or not towns.conquest:
      return 'they found no towns by choice here'
    if towns.founded(polity) >= int(towns.config['max_settlements']):
      return (
          f'they have founded {towns.founded(polity)} towns, the most they can'
      )
    people = civ.polities[polity].population
    need = float(towns.config['town_population']) * (
        len(towns.places[polity]) + 1
    )
    if people < need:
      return (
          f'they are {people:.1f} people in {len(towns.places[polity])} towns;'
          f' settlers for one more need {need:g}'
      )
    if not towns.has_site(polity):
      return 'there is no free site left in their lands'

    short = [
        f'{amount:g} {good}'
        for good, amount in self.founding_cost().items()
        if civ.stock(polity, good) < amount
    ]
    if short:
      return f'it takes {" and ".join(short)}, which they lack'
    return ''

  def found_town(self, polity: str, actor: str) -> str:
    """A people founds a town: where it was aimed, or its best free site.

    Args:
      polity: The people.
      actor: Its entity's name (whose aim, if any, says where).

    Returns:
      What happened, or why not.
    """
    target = self.targets.pop(actor, None)
    assert self.civ is not None
    name = self.civ.polities[polity].name
    why = self.can_found(polity)
    if why:
      return f'{name} found no town: {why}.'
    assert self.settlements is not None
    place, why = self.settlements.found_at(polity, target)
    if place is None:
      return f'{name} found no town: {why}.'
    for good, amount in self.founding_cost().items():
      self.civ.ledger.withdraw(polity, good, amount, 'a new town')
    return news_lib.located(
        f'{name} found a town {self.settlements.where(place)}.',
        (place.row, place.col),
    )

  def change_hands(
      self,
      town: int,
      taker: str,
      *,
      stay: float,
      loyalty: float,
      how: str = 'take',
  ) -> WorldEvent:
    """A town passes to another people, with its share of the people.

    The town's people are its holder's population split across the
    holder's towns by the farmland each works (`Settlements.shares`); the
    holder loses that share, and the taker gains the part that stays.

    Args:
      town: The town's id.
      taker: The people that takes or wins it.
      stay: The share of the town's people that stays under the taker.
      loyalty: Its loyalty to the taker.
      how: 'take' (by arms), 'win' (won over) or 'rise' (back to its
        founder).

    Returns:
      The news, for every people (where it happened).
    """
    if self.settlements is None or self.civ is None:
      raise ValueError('No towns to change hands.')
    found = self.settlements.find(town)
    if found is None:
      raise KeyError(town)
    holder = found[1]
    moved = (
        self.civ.polities[holder].population
        * self.settlements.shares(holder)[town]
    )
    self.civ.polities[holder].population -= moved
    self.civ.polities[taker].population += moved * stay
    text = self.settlements.change_hands(town, taker, loyalty, how)
    return WorldEvent(CIV_EVENTS, text, where=news_lib.place_of(text))

  def queue_adaptation(self, species: str, answer: str) -> str:
    """Queues the adaptation named in a slow mind's answer.

    Args:
      species: Species key.
      answer: Free-text answer of the species' adaptation mind.

    Returns:
      A sentence describing what will happen.
    """
    if self.grazers is not None and species == self.grazers.key:
      return self.grazers.queue_adaptation(answer)
    evolution = self._evolution()
    option = evolution.parse_adaptation(answer)
    name = self.species_name(species)
    if option is None or option.key == 'none':
      return f'{name} stay as they are.'
    self.pending_adaptations.append(
        evolution_lib.PendingAdaptation(
            species=species,
            option=option.key,
            ticks_left=evolution.phase_in_ticks,
        )
    )
    self.adaptation_counts[species] += 1
    return f'{name} begin evolving a {option.label}.'

  def _phase_in_adaptations(self) -> list[WorldEvent]:
    if not self.pending_adaptations:
      return []
    evolution = self._evolution()
    events = []
    still_pending = []
    for pending in self.pending_adaptations:
      option = evolution.option(pending.option)
      traits = evolution.apply_step(
          self.daisies,
          pending.species,
          option,
          1.0 / evolution.phase_in_ticks,
      )
      pending.ticks_left -= 1
      if pending.ticks_left > 0:
        still_pending.append(pending)
      else:
        events.append(
            WorldEvent(
                BIO_EVENTS,
                f'{self.species_name(pending.species)} have evolved a'
                f' {option.label} (optimum {traits.optimum_c:.1f} C, petal'
                f' albedo {traits.albedo:.2f}).',
            )
        )
    self.pending_adaptations = still_pending
    return events

  def war_topic(self, polity: str) -> str:
    """Where a people hears of its own wars at once (`civ.<key>.war`)."""
    return f'civ.{polity}.war'

  def march(self, polity: str, actor: str, intent: str) -> tuple[str, str]:
    """A people marches its capital's garrison on another people's town.

    The town nearest where the move was aimed (by a person on the map), or
    nearest the people's capital (a model's move).

    Args:
      polity: The people.
      actor: Its entity's name (whose aim, if any, says where).
      intent: 'take' or 'raze'.

    Returns:
      Where the army goes and when it arrives, or why not; and the people
      whose town it marches on ('' if none).
    """
    target = self.targets.pop(actor, None)
    towns, civ = self.settlements, self.civ
    assert towns is not None and civ is not None
    name = civ.polities[polity].name
    found = towns.nearest_town(polity, target)
    if found is None:
      return f'{name} march nowhere: there is no town of another people.', ''
    town, holder = found
    army, why = towns.march(polity, town.id, intent, self.tick)
    if army is None:
      return f'{name} march nowhere: {why}.', ''
    kinds = ' and '.join(
        f'{n:.1f} {towns.arms[k]["label"]}'
        for k, n in sorted(army.units.items())
    )
    return (
        news_lib.located(
            f'{name} march {kinds} on a town of the'
            f' {civ.polities[holder].name} {towns.where(town)}, to {intent}'
            f' it; they arrive in turn {army.arrives}.',
            (town.row, town.col),
        ),
        holder,
    )

  def send_envoys(
      self, polity: str, actor: str
  ) -> tuple[str, dict[str, str], str]:
    """A people sends a cart of grain and envoys to another people's town.

    To the town nearest where the move was aimed (by a person on the map),
    or the town nearest to joining it (a model's move). The townsfolk eat
    the grain; the town's regard for the giver grows (twice as much while
    its holder's people go hungry), and it joins the giver when its regard
    reaches what it needs (`Settlements.need`).

    Args:
      polity: The people.
      actor: Its entity's name (whose aim, if any, says where).

    Returns:
      What happened, or why not; the words each side hears at once (by
      people); and the news for every people if the town joined ('' if
      not).
    """
    target = self.targets.pop(actor, None)
    towns, civ = self.settlements, self.civ
    assert towns is not None and civ is not None
    name = civ.polities[polity].name
    cost = float(towns.war.get('gift_grain', 3.0))
    if civ.stock(polity, 'grain') < cost:
      return (
          (
              f'{name} send no envoys: a cart takes {cost:g} grain, which they'
              ' lack.'
          ),
          {},
          '',
      )
    found = towns.best_court(polity, target)
    if found is None:
      return f'{name} send no envoys: there is no town to court.', {}, ''
    town, holder = found
    held_by, where = civ.polities[holder].name, towns.where(town)
    if town.capital:
      return (
          (
              f'{name} send no envoys: the town {where} is the capital of the'
              f' {held_by}, which holds to its own people.'
          ),
          {},
          '',
      )
    civ.ledger.withdraw(polity, 'grain', cost, 'envoys')
    hungry = civ.polities[holder].food_ratio < 1.0 - 1e-9
    gain = float(towns.war.get('gift_regard', 0.15)) * (2.0 if hungry else 1.0)
    regard = towns.court(town.id, polity, gain)
    need = towns.need(town)
    why = f' (twice as warm: the {held_by} go hungry)' if hungry else ''
    if regard < need - 1e-9:
      to_go = towns.to_go(town, polity)
      return (
          news_lib.located(
              f'{name} send {cost:g} grain and envoys to a town of the'
              f' {held_by} {where}: its regard for them is {regard:.2f} of'
              f' the {need:.2f} it needs ({to_go:.0%} to go){why}.',
              (town.row, town.col),
          ),
          {
              holder: (
                  f'The {name} sent grain and envoys to our town {where}: its'
                  f' regard for them is {regard:.2f} of the {need:.2f} it'
                  f' needs to leave us ({to_go:.0%} to go).'
              )
          },
          '',
      )
    units = dict(town.garrison)
    joined = self.change_hands(
        town.id,
        polity,
        stay=1.0,
        loyalty=float(towns.war.get('win_loyalty', 0.6)),
        how='win',
    )
    towns.come_home(holder, units)
    home = (
        ' Its garrison of '
        + ' and '.join(
            f'{n:.1f} {towns.arms[k]["label"]}'
            for k, n in sorted(units.items())
        )
        + ' walked home.'
        if units
        else ''
    )
    return (
        news_lib.located(
            f'{name} send {cost:g} grain and envoys to a town of the'
            f' {held_by} {where}: its regard for them reaches {regard:.2f}'
            f'{why}, and it joins them.',
            (town.row, town.col),
        ),
        {
            holder: (
                f'Our town {where} was won over by the grain and envoys of'
                f' the {name} and left us.{home}'
            ),
            polity: (
                f'A town of the {held_by} {where} was won over by our grain'
                ' and envoys and joined us.'
            ),
        },
        joined.text,
    )

  def feast(self, polity: str, actor: str) -> tuple[str, dict[str, str]]:
    """A people feasts one of its own towns with a cart of grain.

    The town nearest where the move was aimed, or the people's least
    settled town (`Settlements.unsettled`). Its loyalty rises, and every
    other people's regard there falls by as much.

    Args:
      polity: The people.
      actor: Its entity's name (whose aim, if any, says where).

    Returns:
      What happened, or why not; and the words each people that courted the
      town hears at once.
    """
    target = self.targets.pop(actor, None)
    towns, civ = self.settlements, self.civ
    assert towns is not None and civ is not None
    name = civ.polities[polity].name
    cost = float(towns.war.get('gift_grain', 3.0))
    if civ.stock(polity, 'grain') < cost:
      return (
          f'{name} hold no feast: it takes {cost:g} grain, which they lack.',
          {},
      )
    town = towns.unsettled(polity, target)
    if town is None:
      return f'{name} hold no feast: no town of theirs needs one.', {}
    civ.ledger.withdraw(polity, 'grain', cost, 'a feast')
    courted = dict(town.regard)
    towns.feast(town.id, float(towns.war.get('feast_loyalty', 0.15)))
    where = towns.where(town)
    told = {
        other: (
            f'The {name} feasted their town {where}: its regard for us falls'
            f' to {town.regard.get(other, 0.0):.2f}.'
        )
        for other in courted
    }
    return (
        news_lib.located(
            f'{name} feast their town {where} with {cost:g} grain: it holds'
            f' to them at {town.loyalty:.2f}'
            + (", and others' regard there falls." if courted else '.'),
            (town.row, town.col),
        ),
        told,
    )

  def _wars(self) -> list[WorldEvent]:
    """Armies march; those arriving fight; won towns settle or rise.

    Where conquest is played (`data/arms.yaml`, `war`). Every outcome is news
    for every people, where it happened.

    Returns:
      The news.
    """
    towns, civ = self.settlements, self.civ
    if towns is None or civ is None or not towns.war:
      return []
    events = []

    def told(text: str, **own: str) -> None:
      """News for every people; and, at once, each side's own words."""
      events.append(WorldEvent(CIV_EVENTS, text, where=news_lib.place_of(text)))
      for key, words in own.items():
        events.append(WorldEvent(self.war_topic(key), words))

    for army in towns.march_on(self.tick):
      draw = float(
          np.random.default_rng([self.tiles.seed, army.id, self.tick]).random()
      )
      result = towns.fight(army, draw)
      name = civ.polities[army.polity].name
      kept = (
          ' and '.join(
              f'{n:.1f} {towns.arms[k]["label"]}'
              for k, n in sorted(result['survivors'].items())
          )
          or 'no one'
      )
      if result['outcome'] == 'gone':
        towns.come_home(army.polity, result['survivors'])
        told(
            f'The army of the {name} finds no town to fight for and turns'
            ' home.',
            **{
                army.polity: (
                    f'Our army found no town to fight for; {kept} came home.'
                )
            },
        )
        continue
      town, holder = result['town'], result['holder']
      held_by = civ.polities[holder].name
      where = towns.where(town)
      if result['outcome'] == 'repelled':
        towns.come_home(army.polity, result['survivors'])
        told(
            news_lib.located(
                f'The army of the {name} is thrown back from a town of the'
                f' {held_by} {where}.',
                (town.row, town.col),
            ),
            **{
                army.polity: (
                    f'Our army was thrown back from a town of the {held_by}'
                    f' {where}; {kept} came home.'
                ),
                holder: f'Our town {where} threw back the army of the {name}.',
            },
        )
      elif army.intent == 'raze':
        _, share = towns.raze(town.id)
        civ.polities[holder].population *= 1.0 - share
        towns.come_home(army.polity, result['survivors'])
        told(
            news_lib.located(
                f'The {name} raze a town of the {held_by} {where}; its people'
                ' are scattered.',
                (town.row, town.col),
            ),
            **{
                army.polity: (
                    f'Our army razed a town of the {held_by} {where}; {kept}'
                    ' came home.'
                ),
                holder: (
                    f'The {name} razed our town {where}; its people are gone.'
                ),
            },
        )
      else:
        events.append(
            self.change_hands(
                town.id,
                army.polity,
                stay=0.8,
                loyalty=float(towns.war.get('taken_loyalty', 0.35)),
                how='take',
            )
        )
        town.garrison = dict(result['survivors'])  # It holds what it took.
        events.append(
            WorldEvent(
                self.war_topic(army.polity),
                f'Our army took a town of the {held_by} {where}; {kept} hold'
                ' it. It is restless: it holds to us while our arms stay.',
            )
        )
        events.append(
            WorldEvent(
                self.war_topic(holder), f'The {name} took our town {where}.'
            )
        )
    towns.fade_regard()
    for town, holder, kind in towns.unrest():
      held_by = civ.polities[holder].name
      if kind == 'warn':
        need = float(towns.war.get('hold_need', 2.0))
        told(
            news_lib.located(
                f'A town {towns.where(town)} held by the {held_by} is restless:'
                ' it will rise unless held by arms.',
                (town.row, town.col),
            ),
            **{
                holder: (
                    f'Our town {towns.where(town)} is restless: it rises next'
                    f' turn unless arms of strength {need:g} hold it.'
                )
            },
        )
      else:
        events.append(
            self.change_hands(
                town.id, town.polity, stay=1.0, loyalty=0.6, how='rise'
            )
        )
        events.append(
            WorldEvent(
                self.war_topic(holder),
                f'Our town {towns.where(town)} rose against us and returned'
                f' to the {civ.polities[town.polity].name}.',
            )
        )
        events.append(
            WorldEvent(
                self.war_topic(town.polity),
                f'Our old town {towns.where(town)} rose against the'
                f' {held_by} and is ours again.',
            )
        )
    return events

  def _muster(self) -> list[str]:
    """New arms join their people's capital; every unit eats grain.

    Where conquest is played (`data/arms.yaml`). A people that cannot feed
    its arms sees half of the unfed share desert, and every people hears it.

    Returns:
      The news.
    """
    towns, civ = self.settlements, self.civ
    if towns is None or civ is None or not towns.arms:
      return []
    news = []
    for key, people in civ.polities.items():
      for kind in towns.arms:
        made = civ.stock(key, kind)
        if made > 1e-9:
          civ.ledger.withdraw(key, kind, made, 'mustered')
          towns.muster(key, kind, made)
      need = towns.upkeep(key)
      if need <= 1e-9:
        continue
      eaten = min(need, civ.stock(key, 'grain'))
      if eaten > 1e-9:
        civ.ledger.withdraw(key, 'grain', eaten, 'arms')
      if eaten < need - 1e-9:
        lost = towns.desert(key, 0.5 * (1.0 - eaten / need))
        gone = ', '.join(
            f'{n:.1f} {towns.arms[k]["label"]}'
            for k, n in sorted(lost.items())
            if n >= 0.05
        )
        if gone:
          news.append(f'{people.name} cannot feed their arms: {gone} desert.')
    return news

  def _step_civ(self) -> list[str]:
    """Runs the polities' work, food and population for one tick."""
    civ = self.civ
    assert civ is not None
    species = civ.forest_species
    has_forest = species in self.daisies.species()
    forest = (
        self.daisies.cover_map(species)
        if has_forest
        else np.zeros(self.grid.size)
    )

    # Protected woods are out of the woodcutters' reach.
    open_land = 1.0 - self.protected()
    forest = forest * open_land

    def clear(cut: np.ndarray) -> None:
      cut = np.asarray(cut) * open_land
      self._last_cut = cut
      if has_forest:
        self.daisies.graze(species, cut)

    def emit(amount: float, reason: str) -> None:
      # Charcoal and bloomery carbon comes from felled wood (biomass), not
      # the mantle; burning it moves it to the air at once.
      if self.carbon is not None:
        self.carbon.burn(amount, reason)

    def darken(amount: float) -> None:
      # Soot settles on snow and fields: a decaying darkening of the planet.
      self.daisies.add_albedo_pulse(-self.soot_albedo * amount)

    civ.biodiversity = self.biodiversity()
    civ.ocean_ph = self.ocean.ph
    civ.sea_ice = self.ocean.sea_ice
    civ.dead_zone = self.ocean.dead_zone
    return civ.step(
        self.daisies.temperatures_c(),
        self.daisies.fertility(),
        forest,
        clear_forest=clear,
        emit=emit,
        soot=darken,
        fossil=lambda amount, reason: (
            self.carbon.unearth(amount, reason) if self.carbon else None
        ),
        runoff=self.ocean.feed,
    )

  def protected(self) -> np.ndarray:
    """1 for each region whose woods are protected now, else 0."""
    shield = np.zeros(self.grid.size)
    for region, until in self.reserves.items():
      if until > self.tick:
        shield[region] = 1.0
    return shield

  def _reserve_regions(self, target: tuple[int, int] | None) -> list[int]:
    """The regions a reserve covers: the tapped tile's region and its

    neighbours, or where the peoples felled most wood last tick.
    """
    if target is not None:
      return [
          int(r) for r in np.nonzero(self._target_regions(target) >= 0.6)[0]
      ]
    felled = np.nonzero(self._last_cut > 1e-9)[0]
    ranked = sorted(felled, key=lambda r: -self._last_cut[r])
    return [int(r) for r in ranked[:3]]

  def propose_treaty(
      self, proposer: str, partner: str, kind: str = 'coal'
  ) -> str | None:
    """Records a treaty proposal; returns its text, or None."""
    if self.treaties is None or self.civ is None:
      return None
    names = {k: p.name for k, p in self.civ.polities.items()}
    proposal = self.treaties.propose(proposer, partner, names, self.tick, kind)
    return proposal.text if proposal else None

  def answer_treaty(
      self, partner: str, answer: str, accepted: bool | None = None
  ) -> tuple[str, str] | None:
    """Settles a treaty proposal from the partner's answer.

    Args:
      partner: Who answers.
      answer: The reply, "Reply to <proposer> (<kind>): ..." (free text from a
        treaty mind, or a person's menu choice).
      accepted: The decision, when known (a person's choice); otherwise it is
        read from the text.

    Returns:
      (proposer, event text), or None if nothing was settled.
    """
    if self.treaties is None or self.civ is None:
      return None
    if accepted is None:
      accepted = bool(symbiosis_lib.parse_answer(answer))
    # The treaty mind answers "Reply to <proposer>: ..."; an answer written
    # to an earlier offer must not settle a newer one.
    keys = {p.name: k for k, p in self.civ.polities.items()}
    kinds = {spec.name: kind for kind, spec in self.treaties.kinds.items()}
    to = re.match(
        r'\s*Reply to (.+?)(?: \(('
        + '|'.join(re.escape(name) for name in kinds)
        + r')\))?:',
        answer,
    )
    proposer = keys.get(to.group(1)) if to else None
    kind = kinds.get(to.group(2)) if to and to.group(2) else None
    proposal = self.treaties.answer(
        partner, bool(accepted), self.tick, proposer=proposer, kind=kind
    )
    if proposal is None:
      return None
    a = self.civ.polities[proposal.proposer].name
    b = self.civ.polities[partner].name
    what = self.treaties.kinds[proposal.kind].name
    terms = self.treaties.terms(proposal.kind)
    if accepted:
      text = (
          f'The {b} sign the {what} offered by the {a}: for'
          f' {self.treaties.duration} generations each {terms} a'
          ' generation.'
      )
    else:
      text = f'The {b} decline the {what} offered by the {a}.'
    return proposal.proposer, text

  def _launch_seed_ships(self) -> list[WorldEvent]:
    """Records each whole seed ship a people completes, and the world it left."""
    civ, events = self.civ, []
    if civ is None:
      return events
    for key, people in sorted(civ.polities.items()):
      whole = int(math.floor(civ.stock(key, 'seed_ships') + 1e-9))
      for _ in range(whole - self.ships_launched.get(key, 0)):
        world = self.biodiversity()
        diversity = min(world, self.vault) if self.has_vault() else world
        self.launches.append((self.tick, key, diversity))
        events.append(
            WorldEvent(
                CIV_EVENTS,
                f'The {people.name} launch a seed ship toward the stars,'
                f' carrying a world of biodiversity {diversity:.2f}'
                + (
                    f' (the world held {world:.2f}; the seed vault'
                    f' {self.vault:.2f})'
                    if self.has_vault()
                    else ''
                )
                + '.',
            )
        )
      self.ships_launched[key] = max(whole, self.ships_launched.get(key, 0))
    return events

  def _step_orbit(self) -> str | None:
    """Satellites in orbit feed Gaia's instruments (the best tier there is)."""
    civ = self.civ
    if civ is None or self.instrument_tier is None or civ.in_orbit() <= 0:
      return None
    best = len(self.instrument_tiers) - 1
    if self.instrument_tier >= best:
      return None
    self.instrument_tier = best
    return (
        'Weather satellites circle the planet: Gaia now reads it through'
        f' {self.instrument_text()}.'
    )

  def commission_broadsheet(self, polity: str) -> str | None:
    """Spends paper on a broadsheet the people's press will write."""
    gazette, civ = self.gazette, self.civ
    if gazette is None or civ is None:
      return None
    if not gazette.can_print(polity, self.tick, civ.stock(polity, 'paper')):
      return None
    civ.ledger.withdraw(polity, 'paper', gazette.paper_cost, 'broadsheet')
    gazette.commission(polity, self.tick)
    return f'The {civ.polities[polity].name} set their presses running.'

  def commission_pamphlet(self, polity: str) -> str | None:
    """Pays the coal guilds for pamphlets the people's press will write."""
    gazette, civ = self.gazette, self.civ
    if gazette is None or civ is None:
      return None
    if not gazette.can_pamphlet(polity, self.tick, civ.stock(polity, 'silver')):
      return None
    civ.ledger.withdraw(
        polity, 'silver', gazette.pamphlet_silver, 'the coal guilds'
    )
    gazette.commission(polity, self.tick, kind='pamphlet')
    return (
        f'The {civ.polities[polity].name} pay the coal guilds'
        f' {gazette.pamphlet_silver:g} silver for pamphlets.'
    )

  def broadsheet_evidence(self) -> media_lib.Evidence:
    """What the public instruments show, to judge broadsheets by."""
    civ = self.civ
    restless = (
        self.backlash.restless(self.homes()).values() if self.backlash else ()
    )
    return media_lib.Evidence(
        relative_co2=self.carbon.relative_co2() if self.carbon else 1.0,
        restless=max(restless, default=0.0),
        attacked=frozenset(
            k
            for k, t in self.struck.items()
            if self.tick - t <= self.attack_memory
        ),
        fed=({k: p.food_ratio for k, p in civ.polities.items()} if civ else {}),
    )

  def publish_broadsheet(self, polity: str, text: str) -> str | None:
    """Posts the press's broadsheet on the Gazette; returns the news."""
    gazette, civ = self.gazette, self.civ
    if gazette is None or civ is None:
      return None
    sheet = gazette.publish(
        polity,
        text,
        f'generation {self.tick}',
        self.broadsheet_evidence(),
        civ,
    )
    if sheet is None:
      return None
    verdict = 'believe it' if sheet.votes > 0 else 'call it false'
    return (
        f'The {sheet.author} print a {sheet.kind} in the Gazette for'
        f' {gazette.stances[sheet.stance]}: "{sheet.title}". The other'
        f' peoples {verdict}.'
    )

  def publish_retelling(self, polity: str, text: str) -> str | None:
    """Posts the printers' retelling of an omen on the Gazette; the news."""
    gazette, civ = self.gazette, self.civ
    if gazette is None or civ is None:
      return None
    sheet = gazette.publish(
        polity,
        text,
        f'generation {self.tick}',
        self.broadsheet_evidence(),
        civ,
        kind='omen',
    )
    if sheet is None:
      return None
    verdict = 'believe it' if sheet.votes > 0 else 'call it false'
    return (
        f"The {sheet.author}'s printers tell the omen in the Gazette for"
        f' {gazette.stances[sheet.stance]}: "{sheet.title}". The other'
        f' peoples {verdict}.'
    )

  def homes(self) -> dict[str, list[int]]:
    civ = self.civ
    return {k: list(p.regions) for k, p in civ.polities.items()} if civ else {}

  def _step_backlash(self) -> list[str]:
    """Pollution drifts and settles; restless lands strike their peoples."""
    civ, backlash = self.civ, self.backlash
    assert civ is not None and backlash is not None
    for _, people in sorted(civ.polities.items()):
      backlash.emit(people.regions, people.emissions)
    forest_key = civ.forest_species
    forest = np.zeros(self.grid.size)
    grass = np.zeros(self.grid.size)
    for species in self.daisies.species():
      if species == forest_key:
        forest = forest + self.daisies.cover_map(species)
      else:
        grass = grass + self.daisies.cover_map(species)
    names = {k: p.name for k, p in civ.polities.items()}
    texts = []
    for attack in backlash.step(forest, grass, self.homes(), names):
      civ.strike(
          attack.polity, attack.harvest_loss, attack.ticks, attack.deaths
      )
      self.struck[attack.polity] = self.tick
      if self.gazette is not None:
        self.gazette.blame_smoke(attack.polity, civ)
      texts.append(attack.text)
    return texts

  def _speciate(self) -> list[WorldEvent]:
    if self.speciation is None or self.evolution is None:
      return []
    split = self.speciation.check(
        self.daisies, self.grid, self._next_key, self.tick
    )
    if split is None:
      return []
    parent = self.species_name(split.parent)
    self.evolution.add_species(
        evolution_lib.SpeciesInfo(split.child, f'{parent} of {split.where}', '')
    )
    self.adaptation_counts[split.child] = 0
    self.behaviours[split.child] = 'hold'
    if self.seed_banks is not None:
      self.seed_banks.add(split.child)
    if (
        self.chemistry is not None
        and split.parent in self.chemistry.metabolisms
    ):
      # Daughters keep their parent's metabolism (and its oxygen chemistry).
      self.chemistry.metabolisms[split.child] = self.chemistry.metabolisms[
          split.parent
      ]
    self.new_species.append(split)
    climate = 'colder' if split.colder else 'warmer'
    return [
        WorldEvent(
            BIO_EVENTS,
            f'A new species splits from the {parent} in the {climate}'
            f' {split.where} (optimum {split.optimum_c:.0f} C, petal albedo'
            f' {split.albedo:.2f}).',
        )
    ]

  def _next_key(self, parent: str) -> str:
    root = parent.rstrip('0123456789')
    return f'{root}{len(self.daisies.species()) + 1}'

  def species_names(self) -> dict[str, str]:
    keys = list(self.daisies.species())
    if self.grazers is not None:
      keys.append(self.grazers.key)
    return {k: self.species_name(k) for k in keys}

  def propose_pact(self, proposer: str, partner: str) -> str | None:
    """Records a pact proposal; returns its text, or None if not possible."""
    if self.symbiosis is None:
      return None
    proposal = self.symbiosis.propose(
        proposer, partner, self.tick, self.species_names()
    )
    return proposal.text if proposal else None

  def answer_pact(self, partner: str, answer: str) -> tuple[str, str] | None:
    """Settles a proposal from a partner's free-text answer.

    Returns:
      (proposer key, announcement) or None if nothing was pending.
    """
    if self.symbiosis is None:
      return None
    accepted = symbiosis_lib.parse_answer(answer)
    proposal = self.symbiosis.answer(partner, bool(accepted), self.tick)
    if proposal is None:
      return None
    a, b = self.species_name(proposal.proposer), self.species_name(partner)
    if accepted:
      text = (
          f'{b} accept the pact with {a}: mixed meadows for'
          f' {self.symbiosis.duration} ticks.'
      )
    else:
      text = f'{b} decline the pact offered by {a}.'
    return proposal.proposer, text

  def take_new_species(self) -> list[speciation_lib.Split]:
    """Returns and clears the species founded since the last call."""
    founded, self.new_species = self.new_species, []
    return founded

  def genome_text(self, species: str) -> str:
    if self.grazers is not None and species == self.grazers.key:
      return self.grazers.genome_text()
    text = self._evolution().genome_text(
        self.daisies, species, self.species_noun
    )
    pending = [
        p.option for p in self.pending_adaptations if p.species == species
    ]
    if pending:
      text += f' Currently evolving: {", ".join(pending)}.'
    if self.seed_banks is not None:
      text += ' ' + self.seed_banks.text(species)
    return text

  def _apply_shock(self, shock: Mapping[str, Any]) -> WorldEvent:
    kind = shock['kind']
    if kind == 'blight':
      self.daisies.kill(shock['species'], float(shock['fraction']))
    elif kind == 'albedo_pulse':
      self.daisies.add_albedo_pulse(float(shock['delta']))
    else:
      raise ValueError(f'Unknown shock kind {kind!r}')
    topic = BIO_EVENTS if kind == 'blight' else GEO_EVENTS
    return WorldEvent(
        str(shock.get('topic', topic)), str(shock.get('text', kind))
    )

  def progress(self) -> list[dict[str, Any]]:
    """How far the scenario's goal has come, for the page's HUD.

    Each item has a label, a few words of value, a fraction (0-1, how far
    along) and whether it is on track now (`ok`).
    """

    def item(label, value, fraction, ok):
      return {
          'label': label,
          'value': value,
          'fraction': round(float(min(1.0, max(0.0, fraction))), 3),
          'ok': bool(ok),
      }

    goal = self.goal()
    kind = goal['kind']
    items = [
        item(
            'turn',
            f'{self.tick} of {self.max_ticks}',
            self.tick / max(1, self.max_ticks),
            True,
        )
    ]
    low, high = self.band()
    temp = self.temperature_c()
    band = item(
        'climate',
        f'{temp:.1f} °C (band {low:g} to {high:g})',
        1.0 if self.in_band() else 0.0,
        self.in_band(),
    )
    most = int(self._scenario.get('max_ticks_out_of_band', 0))
    strain = item(
        'strain',
        f'{self.out_of_band:.0f} of {most}'
        + ('' if self.in_band() else ' · out of band now'),
        1.0 - self.out_of_band / max(1, most),
        self.out_of_band < most,
    )
    if kind == 'thaw':
      start = float(self._scenario.get('initial_temp_c', temp))
      want = float(goal['temp_c'])
      ice, cap = self.ice_cover(), float(goal['max_ice'])
      items += [
          item(
              'warmth',
              f'{temp:.1f} °C of {want:g} °C needed',
              (temp - start) / max(1e-9, want - start),
              temp >= want,
          ),
          item(
              'ice',
              f'{ice:.0%} of the planet (at most {cap:.0%})',
              1.0 - max(0.0, ice - cap) / max(1e-9, 1.0 - cap),
              ice <= cap,
          ),
      ]
    elif kind == 'oxygenate':
      o2 = self.chemistry.oxygen() if self.chemistry else 0.0
      aerobes = self.cover_of_metabolism(oxygen_lib.AEROBIC)
      need_o2, need_aer = float(goal['min_o2']), float(goal['min_aerobes'])
      items += [
          item(
              'oxygen',
              f'{o2:.3f} of {need_o2:g} needed',
              o2 / need_o2,
              o2 >= need_o2,
          ),
          item(
              'aerobes',
              f'{aerobes:.1%} of {need_aer:.0%} needed',
              aerobes / need_aer,
              aerobes >= need_aer,
          ),
          band,
      ]
    elif kind in ('iron_age', 'industrialize', 'seed_ship') and self.civ:
      civ = self.civ
      if kind == 'iron_age':
        iron, need = civ.made('iron'), float(goal['min_iron'])
        items.append(
            item(
                'iron',
                f'{iron:.0f} of {need:g} forged',
                iron / need,
                iron >= need,
            )
        )
      elif kind == 'industrialize':
        energy, need = civ.made('energy'), float(goal['min_energy'])
        co2 = self.carbon.relative_co2() if self.carbon else 1.0
        cap = float(goal['max_co2'])
        items += [
            item(
                'energy',
                f'{energy:.0f} of {need:g} made',
                energy / need,
                energy >= need,
            ),
            item(
                'CO₂',
                f'{co2:.2f}× the start (under {cap:g}×)',
                1.0 - max(0.0, co2 - 1.0) / max(1e-9, cap - 1.0),
                co2 < cap,
            ),
        ]
      else:
        best = max((b for _, _, b in self.launches), default=None)
        bio, need = self.biodiversity(), float(goal['min_biodiversity'])
        items += [
            item(
                'seed ship',
                'launched' if best is not None else 'not yet',
                1.0 if best is not None else 0.0,
                best is not None,
            ),
            item(
                'biodiversity',
                f'{bio:.2f} (a ship needs {need:g})',
                bio / need,
                bio >= need,
            ),
        ]
      worst = min(
          p.population / max(1e-9, civ.start_population[k])
          for k, p in civ.polities.items()
      )
      floor = float(goal['collapse_fraction'])
      items.append(
          item(
              'peoples',
              f'the hungriest at {worst:.0%} of its start',
              (worst - floor) / max(1e-9, 1.0 - floor),
              worst > floor,
          )
      )
      items.append(band)
    elif kind == 'terraform' and self.colony is not None:
      colony = self.colony
      start, want = colony.start_temp_c, colony.target_temp_c
      items += [
          item(
              'sky',
              f'{colony.temp_c:.1f} °C of {want:g} °C',
              (colony.temp_c - start) / max(1e-9, want - start),
              colony.temp_c >= want,
          ),
          item(
              'colonists',
              f'{colony.population:.1f} of {colony.start_population:.0f}'
              ' landed',
              colony.population / max(1e-9, colony.start_population),
              colony.population >= colony.start_population,
          ),
      ]
    elif kind == 'keep_herds':
      for key, need in goal['min'].items():
        have = self.herds.total(key)
        items.append(
            item(
                self._herd_name(key),
                f'total size {have:.1f} of {float(need):g} needed',
                have / float(need),
                have >= float(need),
            )
        )
      items.append(band)
    elif kind == 'keep_woods':
      species = str(goal['species'])
      share = self.daisies.cover(species) / max(
          1e-9, self.start_cover.get(species, 0.0)
      )
      need = float(goal['min_share'])
      items += [
          item(
              self.species_name(species).lower(),
              f'{share:.0%} of their land at the start ({need:.0%} needed)',
              share / need if need else 1.0,
              share >= need,
          ),
          band,
      ]
    elif kind == 'green_islands':
      green, total = self.green_islands()
      need = int(np.ceil(float(goal['min_share']) * total))
      items += [
          item(
              'islands',
              f'{green} of {total} green ({need} needed)',
              green / max(1, need),
              green >= need,
          ),
          band,
      ]
    else:  # Hold the band: survive to the end.
      items += [band, strain]
    if kind in ('keep_woods', 'green_islands', 'keep_herds'):
      items.append(strain)  # These are lost to band strain too.
    return items

  def _find_islands(self, min_tiles: int) -> list[np.ndarray]:
    """The land's connected patches of at least `min_tiles`, largest first."""
    land = self.tiles.land
    rows, cols = land.shape
    seen = np.zeros_like(land)
    islands = []
    for start in zip(*np.nonzero(land)):
      if seen[start]:
        continue
      patch = np.zeros_like(land)
      stack = [start]
      seen[start] = True
      while stack:
        r, c = stack.pop()
        patch[r, c] = True
        for dr in (-1, 0, 1):
          for dc in (-1, 0, 1):
            nr, nc = r + dr, (c + dc) % cols
            if 0 <= nr < rows and land[nr, nc] and not seen[nr, nc]:
              seen[nr, nc] = True
              stack.append((nr, nc))
      if patch.sum() >= min_tiles:
        islands.append(patch)
    return sorted(islands, key=lambda p: -int(p.sum()))

  def _keep_life_on(self, island: np.ndarray) -> None:
    """Clears every plant off the land outside one island."""
    life = self.tile_life
    for species in list(life.covers):
      life.covers[species] = np.where(island, life.covers[species], 0.0)
      self.daisies.set_cover(species, life.region_mean(life.covers[species]))
    life.reached = island & self.tiles.land

  def green_islands(self) -> tuple[int, int]:
    """(islands whose land is a third or more covered, islands)."""
    need = float(self.goal().get('min_cover', 0.33))
    total = np.sum(list(self.tile_life.covers.values()), axis=0)
    green = sum(
        1 for island in self.islands if float(total[island].mean()) >= need
    )
    return green, len(self.islands)

  def goal(self) -> dict[str, Any]:
    return dict(self._scenario.get('goal') or {'kind': 'hold_band'})

  def stars(self) -> list[dict[str, Any]]:
    """The scenario's stars and which are earned.

    The first star is the goal; each further one (the scenario's `stars`)
    is a harder way to reach it, earned only with the goal. They are judged
    on the true world, so `met` is None until the game is decided.

    Returns:
      [{'text', 'met'}], the goal first.
    """
    decided = self.outcome is not None
    won = self.outcome == VICTORY
    found = [{'text': 'Reach the goal', 'met': won if decided else None}]
    for rule in self._scenario.get('stars') or ():
      met = won and self._star_met(rule) if decided else None
      found.append({'text': str(rule['text']), 'met': met})
    return found

  def star_count(self) -> int:
    """Stars earned (0 until the game is won)."""
    return sum(1 for star in self.stars() if star['met'])

  def side_rules(self) -> list[dict[str, Any]]:
    """The scenario's side objectives, as data."""
    return [dict(rule) for rule in self._scenario.get('side') or ()]

  def _side_ok(self, rule: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    """Whether this tick's true world meets a side objective's test."""
    value = row.get(str(rule['field']))
    if value is None:
      return False
    if 'at_least' in rule:
      return float(value) >= float(rule['at_least'])
    return float(value) <= float(rule['at_most'])

  def _judge_side(self) -> None:
    """Judges the open side objectives on this tick; pays those met.

    A `hold` is met when its test held on every tick of its window, and
    missed the first tick it fails; a `reach` is met the first tick its
    test holds in the window, and missed when the window closes.
    """
    row = self.history[-1] if self.history else {}
    for rule in self.side_rules():
      key = str(rule['key'])
      start, until = int(rule.get('from', 1)), int(rule['until'])
      if self.side.get(key) != SIDE_OPEN or self.tick < start:
        continue
      ok = self._side_ok(rule, row)
      if rule['kind'] == 'hold':
        state = (
            SIDE_MISSED if not ok else SIDE_MET if self.tick >= until else None
        )
      elif rule['kind'] == 'reach':
        state = SIDE_MET if ok else SIDE_MISSED if self.tick >= until else None
      else:
        raise ValueError(f'Unknown side objective kind {rule["kind"]!r}')
      if state is None:
        continue
      self.side[key] = state
      if state == SIDE_MET:
        self.omega += float(rule.get('reward', 0.0))

  def side_payload(self) -> list[dict[str, Any]]:
    """Each side objective's rule, reward and state, for the page.

    Its state is a verdict on the true world, as the goal's is; how close
    the world is, the role reads from its own instruments.
    """
    return [
        {
            'key': str(rule['key']),
            'text': str(rule['text']),
            'kind': str(rule['kind']),
            'from': int(rule.get('from', 1)),
            'until': int(rule['until']),
            'reward': float(rule.get('reward', 0.0)),
            'state': self.side.get(str(rule['key']), SIDE_OPEN),
            'turns': max(0, int(rule['until']) - self.tick),
        }
        for rule in self.side_rules()
    ]

  def _star_met(self, rule: Mapping[str, Any]) -> bool:
    """Whether a decided game meets one star's rule."""
    kind, last = rule['kind'], self.history[-1] if self.history else {}
    if kind == 'by_tick':  # Reached the goal by this tick.
      return self.tick <= int(rule['tick'])
    if kind == 'max_strain':  # Never this far out of the band.
      worst = max(
          (row.get('out_of_band', 0.0) for row in self.history), default=0.0
      )
      return worst <= float(rule['value'])
    if kind == 'at_least':  # A final reading at least this.
      return float(last.get(rule['field'], 0.0)) >= float(rule['value'])
    if kind == 'at_most':  # A final reading at most this.
      return float(last.get(rule['field'], math.inf)) <= float(rule['value'])
    if kind == 'omega_left':  # Ω saved at the end.
      return self.omega >= float(rule['value'])
    if kind == 'colonists':  # Colonists alive at the end.
      return self.colony is not None and self.colony.population >= float(
          rule['value']
      )
    raise ValueError(f'Unknown star rule {kind!r}')

  def ice_free_land(self) -> float:
    land = self.daisies.land_fraction()
    ice = self.climate.ice_fraction(self.daisies.temperatures_c())
    return self.climate.global_mean(land * (1.0 - ice))

  def ice_cover(self) -> float:
    return self.climate.global_mean(self.daisies.ice_map())

  def _update_ocean(self) -> list[str]:
    return self.ocean.update(
        weights=self.climate.weights,
        land=self.daisies.land_fraction(),
        ice=self.climate.ice_fraction(self.daisies.temperatures_c()),
        temps_c=self.daisies.temperatures_c(),
        ocean_carbon=(
            self.carbon.amount(carbon_lib.OCEAN) if self.carbon else None
        ),
    )

  def ocean_payload(self) -> dict[str, Any]:
    payload = self.ocean.payload(1.0 - self.daisies.land())
    payload['tick'] = self.tick
    return payload

  def _next_out_of_band(self) -> float:
    """Ticks of band stress: +1 out of band; in band it heals or resets.

    With `band_recovery_per_tick` in the scenario, stress heals that much per
    tick in band, so a planet that keeps overheating briefly still dies of
    the accumulated damage. Without it, any tick in band resets the count.

    Returns:
      The new stress count.
    """
    if not self.in_band():
      # Far outside the band (more than `severe_out_of_band_c`), strain
      # mounts twice as fast: a wrecked world ends in a few turns, not ten.
      low, high = self.band()
      temp = self.temperature_c()
      far = max(low - temp, temp - high)
      severe = float(self._scenario.get('severe_out_of_band_c', 10.0))
      return self.out_of_band + (2.0 if far > severe else 1.0)
    recovery = self._scenario.get('band_recovery_per_tick')
    if recovery is None:
      return 0.0
    return max(0.0, self.out_of_band - float(recovery))

  def _band_defeat(self) -> str:
    """Why a planet strained out of its band is lost."""
    low, high = self.band()
    temp = self.temperature_c()
    severe = float(self._scenario.get('severe_out_of_band_c', 10.0))
    if max(low - temp, temp - high) > severe:  # Wrecked, not merely strained.
      return (
          f'the climate broke: {temp:.0f} C, far outside the band'
          f' ({low:g} to {high:g} C)'
      )
    most = int(self._scenario.get('max_ticks_out_of_band', 0))
    return (
        f'temperature out of band too long (strain {self.out_of_band:.0f} of'
        f' {most})'
    )

  def _judge(self) -> None:
    if self.outcome is not None:
      return
    goal = self.goal()
    if goal['kind'] == 'iron_age' and self.civ is not None:
      fallen = self.civ.collapsed(float(goal['collapse_fraction']))
      if fallen:
        names = ', '.join(self.civ.polities[k].name for k in fallen)
        self.outcome = DEFEAT
        self.outcome_reason = f'famine broke the {names}'
        return
      if self.tick >= self.max_ticks:
        iron = self.civ.made('iron')
        if not self.in_band():
          self.outcome = DEFEAT
          self.outcome_reason = 'the planet ended outside the habitable band'
        elif iron >= float(goal['min_iron']):
          self.outcome = VICTORY
          self.outcome_reason = (
              f'the peoples forged {iron:.0f} iron and endured'
          )
        else:
          self.outcome = DEFEAT
          self.outcome_reason = (
              f'only {math.floor(iron)} of the'
              f' {float(goal["min_iron"]):.0f} iron was'
              ' forged'
          )
        return
    if goal['kind'] == 'industrialize' and self.civ is not None:
      fallen = self.civ.collapsed(float(goal['collapse_fraction']))
      if fallen:
        names = ', '.join(self.civ.polities[k].name for k in fallen)
        self.outcome = DEFEAT
        self.outcome_reason = f'famine broke the {names}'
        return
      if self.tick >= self.max_ticks:
        energy = self.civ.made('energy')
        co2 = self.carbon.relative_co2() if self.carbon else 1.0
        if not self.in_band():
          self.outcome = DEFEAT
          self.outcome_reason = 'the planet ended outside the habitable band'
        elif co2 > float(goal['max_co2']):
          self.outcome = DEFEAT
          self.outcome_reason = (
              f'the air holds {co2:.2f} times the starting CO2'
          )
        elif energy >= float(goal['min_energy']):
          self.outcome = VICTORY
          self.outcome_reason = (
              f'the peoples made {energy:.0f} energy and kept the air'
          )
        else:
          self.outcome = DEFEAT
          self.outcome_reason = (
              f'only {math.floor(energy)} of the'
              f' {float(goal["min_energy"]):.0f}'
              ' energy was made'
          )
        return
    if goal['kind'] == 'oxygenate' and self.tick >= self.max_ticks:
      aerobes = self.cover_of_metabolism(oxygen_lib.AEROBIC)
      o2 = self.chemistry.oxygen() if self.chemistry else 0.0
      if self.is_lifeless():
        pass
      elif not self.in_band():
        self.outcome = DEFEAT
        self.outcome_reason = 'the planet ended outside the habitable band'
        return
      elif o2 >= float(goal['min_o2']) and aerobes >= float(
          goal['min_aerobes']
      ):
        self.outcome = VICTORY
        self.outcome_reason = 'aerobic life inherited an oxygen world'
        return
      else:
        short = []
        if o2 < float(goal['min_o2']):
          short.append(f"oxygen ({o2:.3f} of today's level)")
        if aerobes < float(goal['min_aerobes']):
          short.append(f'aerobe cover ({aerobes:.0%})')
        self.outcome = DEFEAT
        self.outcome_reason = f'{" and ".join(short)} fell short'
        return
    if goal['kind'] == 'seed_ship' and self.civ is not None:
      fallen = self.civ.collapsed(float(goal['collapse_fraction']))
      # A launch counts when a people completes a whole ship, and is judged
      # by the world it leaves at that moment.
      best = max((b for _, _, b in self.launches), default=None)
      if fallen:
        names = ', '.join(self.civ.polities[k].name for k in fallen)
        self.outcome = DEFEAT
        self.outcome_reason = f'famine broke the {names}'
      elif best is not None and best >= float(goal['min_biodiversity']):
        self.outcome = VICTORY
        self.outcome_reason = (
            'a seed ship left for the stars from a living world'
            f' (biodiversity {best:.2f})'
        )
      elif self.out_of_band >= int(self._scenario['max_ticks_out_of_band']):
        self.outcome = DEFEAT
        self.outcome_reason = self._band_defeat()
      elif self.tick >= self.max_ticks:
        self.outcome = DEFEAT
        need = float(goal['min_biodiversity'])
        if best is not None and self.has_vault() and self.vault < need:
          self.outcome_reason = (
              'seed ships left with too few seeds of the wild (the vault'
              f' held {math.floor(self.vault * 100) / 100:.2f} of {need:g})'
          )
        else:
          self.outcome_reason = (
              'seed ships left, but from a world too poor (biodiversity'
              f' {math.floor(best * 100) / 100:.2f} at best)'
              if best is not None
              else 'no seed ship left for the stars'
          )
      return
    if goal['kind'] == 'terraform' and self.colony is not None:
      colony = self.colony
      if colony.population < float(goal['lost_fraction']) * (
          colony.start_population
      ):
        self.outcome = DEFEAT
        self.outcome_reason = f'the colony at {colony.name} was lost'
      elif colony.terraformed() and (
          colony.population >= colony.start_population
      ):
        self.outcome = VICTORY
        self.outcome_reason = (
            f'{colony.name} is habitable at {colony.temp_c:.0f} C with'
            f' {colony.population:.0f} colonists'
        )
      elif self.tick >= self.max_ticks:
        self.outcome = DEFEAT
        if colony.terraformed():
          self.outcome_reason = (
              f'{colony.name} is habitable at {colony.temp_c:.0f} C but only'
              f' {colony.population:.0f} of {colony.start_population:.0f}'
              ' colonists remain'
          )
        else:
          self.outcome_reason = (
              f'{colony.name} reached only {colony.temp_c:.0f} C with'
              f' {colony.population:.0f} colonists'
          )
      return
    if goal['kind'] == 'keep_herds':
      gone = [k for k in goal['min'] if not self.herds.herds.get(k)]
      if gone:
        names = ' and '.join(self._herd_name(k) for k in gone)
        self.outcome = DEFEAT
        self.outcome_reason = f'the last of the {names} are gone'
      elif self.out_of_band >= int(self._scenario['max_ticks_out_of_band']):
        self.outcome = DEFEAT
        self.outcome_reason = self._band_defeat()
      elif self.tick >= self.max_ticks:
        short = [
            f'{self._herd_name(k)} ({self.herds.total(k):.1f} of {float(v):g})'
            for k, v in goal['min'].items()
            if self.herds.total(k) < float(v)
        ]
        if short:
          self.outcome = DEFEAT
          self.outcome_reason = f'too few {" and ".join(short)} were left'
        else:
          self.outcome = VICTORY
          self.outcome_reason = 'the herds and the packs came through'
      return
    if goal['kind'] == 'keep_woods':
      species = str(goal['species'])
      if self.is_lifeless():
        self.outcome = DEFEAT
        self.outcome_reason = f'all {self.species_noun} died'
      elif self.out_of_band >= int(self._scenario['max_ticks_out_of_band']):
        self.outcome = DEFEAT
        self.outcome_reason = self._band_defeat()
      elif self.tick >= self.max_ticks:
        share = self.daisies.cover(species) / max(
            1e-9, self.start_cover.get(species, 0.0)
        )
        name = self.species_name(species).lower()
        if share >= float(goal['min_share']):
          self.outcome = VICTORY
          self.outcome_reason = f'the {name} kept {share:.0%} of their land'
        else:
          self.outcome = DEFEAT
          self.outcome_reason = (
              f'the {name} kept only {share:.0%} of their land'
          )
      return
    if goal['kind'] == 'green_islands':
      if self.is_lifeless():
        self.outcome = DEFEAT
        self.outcome_reason = f'all {self.species_noun} died'
      elif self.out_of_band >= int(self._scenario['max_ticks_out_of_band']):
        self.outcome = DEFEAT
        self.outcome_reason = self._band_defeat()
      elif self.tick >= self.max_ticks:
        green, total = self.green_islands()
        need = int(np.ceil(float(goal['min_share']) * total))
        if green >= need:
          self.outcome = VICTORY
          self.outcome_reason = f'{green} of the {total} islands are green'
        else:
          self.outcome = DEFEAT
          self.outcome_reason = (
              f'only {green} of the {total} islands are green ({need} needed)'
          )
      return
    if goal['kind'] == 'thaw':
      if self.temperature_c() >= float(
          goal['temp_c']
      ) and self.ice_cover() <= float(goal['max_ice']):
        self.outcome, self.outcome_reason = VICTORY, 'the ice broke'
      elif self.tick >= self.max_ticks:
        self.outcome, self.outcome_reason = DEFEAT, 'the planet stayed frozen'
      return
    if self.is_lifeless():
      self.outcome = DEFEAT
      self.outcome_reason = f'all {self.species_noun} died'
    elif self.out_of_band >= int(self._scenario['max_ticks_out_of_band']):
      self.outcome = DEFEAT
      self.outcome_reason = self._band_defeat()
    elif self.tick >= self.max_ticks:
      if self.in_band():
        self.outcome, self.outcome_reason = VICTORY, 'the planet stayed alive'
      else:
        self.outcome = DEFEAT
        self.outcome_reason = 'the planet ended outside the habitable band'

  def interventions(self) -> list[Intervention]:
    return list(self._interventions.values())

  def ready(self, intervention: Intervention) -> bool:
    """Whether an intervention is off its cooldown now."""
    return self.tick >= self._ready_at.get(intervention.key, 0)

  def intervention_for_option(self, option: str) -> Intervention | None:
    """Maps a Steward's chosen option text back to an intervention."""
    text = option.strip().lower()
    for intervention in self._interventions.values():
      if text == intervention.option.lower() or text.startswith(
          intervention.label.lower()
      ):
        return intervention
    return None

  def set_target(self, actor: str, row: int, col: int) -> None:
    """Where an actor's next move should land (a tile, south first)."""
    self.targets[actor] = (
        int(np.clip(row, 0, self.tiles.rows - 1)),
        int(col) % self.tiles.cols,
    )

  def _near_target(self, target: tuple[int, int]) -> np.ndarray:
    """Steps (Chebyshev, wrapping in longitude) from a tile to every tile."""
    rows, cols = self.tiles.rows, self.tiles.cols
    r, c = target
    dc = np.abs((np.arange(cols) - c + cols // 2) % cols - cols // 2)
    return np.maximum(np.abs(np.arange(rows) - r)[:, None], dc[None, :])

  def _target_regions(self, target: tuple[int, int]) -> np.ndarray:
    """Weights per region: the target's 1, its neighbours' 0.6, theirs 0.3."""
    region = int(self.tiles.region[target])
    weights = np.zeros(self.grid.size)
    ring = set(self.grid.neighbours(region))
    for n in {m for r in ring for m in self.grid.neighbours(r)}:
      weights[n] = 0.3
    for n in ring:
      weights[n] = 0.6
    weights[region] = 1.0
    return weights

  def _place_words(self, target: tuple[int, int]) -> str:
    lat = float(self.biomes.lat[target[0]])
    return f' around {abs(lat):.0f}°{"N" if lat >= 0 else "S"}'

  def _tile_area(self) -> np.ndarray:
    """Each tile's share of the planet's area (tiles shrink to the poles)."""
    weights = self.grid.area_weights()
    return (weights / weights.sum())[self.tiles.region] / self.tile_life.cells

  def _bare_tiles(self) -> np.ndarray:
    life = self.tile_life
    return np.clip(
        self.tiles.land - np.sum(list(life.covers.values()), axis=0), 0.0, None
    )

  def _plant_tiles(self, species: str, added: np.ndarray) -> float:
    """Adds cover per tile (and to its regions); returns the planet share."""
    life = self.tile_life
    life.covers[species] = life.covers[species] + added
    planted = self.daisies.plant(species, life.region_mean(added))
    return float(self.climate.global_mean(planted))

  def _fill(
      self, amount: float, room: np.ndarray, area: np.ndarray
  ) -> tuple[np.ndarray, float]:
    """Up to `amount` of the planet spread over `room` pro rata."""
    room_area = float((room * area).sum())
    if room_area <= 0 or amount <= 0:
      return np.zeros_like(room), 0.0
    take = min(room_area, amount)
    return room * take / room_area, take

  def _sow(self, species: str, amount: float, carry: bool = False) -> float:
    """Plants a share of the planet's area where seeds can take hold.

    Where plants have reached all the land, it is spread over the bare land
    pro rata, as always. On a world whose life has not reached every shore
    (an archipelago), seeds sprout only on land life has reached; carried
    seed (`carry`, the Steward's unaimed sowing) first fills the bare island
    nearest that land, which then counts as reached.

    Returns:
      The share of the planet planted.
    """
    life = self.tile_life
    if species not in life.covers or amount <= 0:
      return 0.0
    land = self.tiles.land
    if bool(life.reached[land].all()):
      return self.daisies.seed(species, amount)
    bare, area = self._bare_tiles(), self._tile_area()
    added = np.zeros_like(bare)
    left = amount
    if carry:
      island = self._nearest_bare_island()
      if island is not None:
        added, took = self._fill(left, np.where(island, bare * 0.8, 0.0), area)
        left -= took
        life.reached = life.reached | island
    more, _ = self._fill(left, np.where(life.reached, bare, 0.0) - added, area)
    added = added + np.maximum(0.0, more)
    return self._plant_tiles(species, added) if added.any() else 0.0

  def _transplant(
      self,
      actor: str,
      intervention: Intervention,
      target: tuple[int, int] | None,
      where: str,
  ) -> str:
    """Carries seed from a species' stands to bare land.

    It costs no omega but cover: the species is thinned everywhere in
    proportion to its cover, and only a share of what is carried takes root
    on bare land (where the person tapped, or the nearest bare island).
    """
    effect = intervention.effect
    species = str(effect['species'])
    if species not in self.daisies.species():
      return f'{actor} looked for {species} to carry, but there is none.'
    cover = self.daisies.cover_map(species)
    total = float(self.climate.global_mean(cover))
    amount = min(float(effect['amount']), 0.5 * total)
    if amount <= 1e-4:
      return f'{actor} found too little {species} to carry.'
    taken = self.daisies.graze(species, cover * (amount / total))
    took = float(self.climate.global_mean(taken))
    sown = took * float(effect['take_root'])
    planted = (
        self._seed_at(species, sown, target)
        if target is not None
        else self._sow(species, sown, carry=True)
    )
    return (
        f'{actor} chose to {intervention.label}{where}: {took:.3f} of the'
        f' planet thinned, {planted:.3f} took root.'
    )

  def _nearest_bare_island(self) -> np.ndarray | None:
    """The island life has not reached nearest the land it has."""
    reached = self.tile_life.reached & self.tiles.land
    if not reached.any():
      return None
    # Steps from reached land (8 around, wrapping in longitude).
    rows = self.tiles.rows
    dist = np.where(reached, 0, -1)
    front = reached
    for step in range(1, rows + self.tiles.cols):
      grown = front.copy()
      for dy in (-1, 0, 1):
        moved = np.zeros_like(front)
        if dy > 0:
          moved[dy:] = front[: rows - dy]
        elif dy < 0:
          moved[: rows + dy] = front[-dy:]
        else:
          moved = front
        for dx in (-1, 0, 1):
          grown |= np.roll(moved, dx, axis=1)
      new = grown & (dist < 0)
      if not new.any():
        break
      dist[new] = step
      front = grown
    best, best_steps = None, None
    for island in self.islands:
      if float(reached[island].mean()) >= 0.5:
        continue
      steps = int(dist[island].min())
      if steps >= 0 and (best_steps is None or steps < best_steps):
        best, best_steps = island, steps
    return best

  def _seed_at(
      self, species: str, amount: float, target: tuple[int, int]
  ) -> float:
    """Plants a share of the planet's area on the bare land nearest a tile.

    Returns:
      The share of the planet planted.
    """
    life = self.tile_life
    if species not in life.covers:
      return 0.0
    bare, area = self._bare_tiles(), self._tile_area()
    steps = self._near_target(target)
    planted_area = 0.0
    added = np.zeros_like(bare)
    shore = None  # The landmass the seed came down on; it stays there.
    for ring in range(int(steps.max()) + 1):
      if planted_area >= amount - 1e-12:
        break
      room = np.where(steps == ring, bare * 0.8, 0.0)
      if shore is not None:
        room = np.where(shore, room, 0.0)
      more, took = self._fill(amount - planted_area, room, area)
      if took <= 0:  # The sea, or no bare ground: try further out.
        continue
      if shore is None:
        shore = self._landmass(more > 0)
        more = np.where(shore, more, 0.0)
        took = float((more * area).sum())
      added += more
      planted_area += took
    if not added.any():
      return 0.0
    life.reached = life.reached | (added > 0)
    self.plantings = [
        *self.plantings[-7:],
        {'tick': self.tick + 1, 'species': species, 'tiles': added > 0},
    ]
    return self._plant_tiles(species, added)

  def _landmass(self, where: np.ndarray) -> np.ndarray:
    """The connected land holding the first of these tiles."""
    if self._landmasses is None:
      self._landmasses = self._find_islands(1)
    first = tuple(int(v) for v in np.argwhere(where)[0])
    for mass in self._landmasses:
      if mass[first]:
        return mass
    return where

  def apply_intervention(
      self, actor: str, intervention: Intervention, words: str = ''
  ) -> str:
    """Applies an intervention if affordable and off cooldown.

    A person may have chosen where (`set_target`): seeding then plants on
    the bare land nearest the tile, clouds and haze gather over its region
    and neighbours, restored marshes clear the runoff off the nearest coast
    first, and a cull thins the herds there. Otherwise, and for models, the
    planet spreads the effect as it always has.

    Args:
      actor: Name of the acting entity, for messages.
      intervention: The intervention to apply.
      words: For an omen, its sign and what it is taken to say, as
        "<sign>: <words>" (a sign alone, or nothing, is allowed).

    Returns:
      A sentence describing what happened.
    """
    target = self.targets.pop(actor, None)
    effect = intervention.effect
    where = self._place_words(target) if target is not None else ''
    if effect['kind'] == 'none':
      return f'{actor} waits. Omega is {self.omega:.1f}.'
    if intervention.cost > self.omega:
      return (
          f'{actor} tried to {intervention.label} but has only'
          f' {self.omega:.1f} of the {intervention.cost:g} omega needed.'
      )
    ready_at = self._ready_at.get(intervention.key, 0)
    if self.tick < ready_at:
      return (
          f'{actor} tried to {intervention.label} but it recharges until'
          f' tick {ready_at}.'
      )
    if effect['kind'] == 'instruments' and not self.can_upgrade_instruments():
      return f'{actor} already has the best instruments there are.'
    if effect['kind'] == 'cull_packs':
      packs = self.herds.herds.get(str(effect['species']), [])
      if not packs:
        return f'{actor} looked for packs to thin, but there are none.'
      reach = int(effect['reach'])
      if target is not None:
        near = self._near_target(target)
        if all(near[p.row, p.col] > reach for p in packs):
          return f'{actor} found no packs within {reach} tiles of there.'
    self.omega -= intervention.cost
    self._ready_at[intervention.key] = self.tick + intervention.cooldown
    self.made.append((self.tick, intervention.key))
    if effect['kind'] == 'omen':
      sign, said = omen_words(words, effect)
      self.omens.append((self.tick, sign, said))
      end = '' if said[-1:] in '.!?' else '.'
      return f'{actor} sent an omen: {sign} over every land' + (
          f', and it was taken to say: "{said}{end}"'
          if said
          else '; no one could say what it meant.'
      )
    if effect['kind'] == 'instruments':
      assert self.instrument_tier is not None
      self.instrument_tier += 1
      return f'{actor} built {self.instrument_text()}.'
    if effect['kind'] == 'seed':
      if target is not None:
        planted = self._seed_at(
            effect['species'], float(effect['amount']), target
        )
      else:
        planted = self._sow(
            effect['species'], float(effect['amount']), carry=True
        )
      return (
          f'{actor} chose to {intervention.label}{where}: {planted:.2f} of the'
          ' planet was planted.'
      )
    if effect['kind'] == 'vault':
      gain = float(self.goal()['vault']['rate']) * self.biodiversity()
      self.vault = min(1.0, self.vault + gain)
      return (
          f'{actor} chose to {intervention.label}: the seed vault holds'
          f' {self.vault:.2f} (+{gain:.2f}).'
      )
    if effect['kind'] == 'transplant':
      return self._transplant(actor, intervention, target, where)
    if effect['kind'] == 'albedo_pulse':
      self.daisies.add_albedo_pulse(
          float(effect['delta']),
          self._target_regions(target) if target is not None else None,
      )
      return f'{actor} chose to {intervention.label}{where}.'
    if effect['kind'] == 'cull':
      if self.grazers is None:
        return f'{actor} looked for herds to thin, but there are none.'
      lost = self.grazers.cull(
          float(effect['fraction']),
          self._target_regions(target) if target is not None else None,
      )
      return (
          f'{actor} chose to {intervention.label}{where}: the herds shrink by'
          f' {lost:.3f}.'
      )
    if effect['kind'] == 'cull_packs':
      species = str(effect['species'])
      if target is not None:
        near = self._near_target(target) <= int(effect['reach'])
        lost = self.herds.thin(species, float(effect['fraction']), near)
      else:  # Unaimed: a lighter hand everywhere.
        lost = self.herds.thin(species, float(effect['fraction']) / 2.0)
      return (
          f'{actor} chose to {intervention.label}{where}: the packs lose'
          f' {lost:.2f} of their number.'
      )
    if effect['kind'] == 'rain':
      # Soaks the ground against fire: around the tap, or lightly everywhere.
      if target is not None:
        reach = float(effect['reach'])
        near = np.clip(
            1.0 - self._near_target(target) / (reach + 1.0), 0.0, 1.0
        )
        self.fire.soak(float(effect['amount']), near)
      else:
        self.fire.soak(float(effect['amount']) * float(effect['spread_share']))
      return f'{actor} chose to {intervention.label}{where}.'
    if effect['kind'] == 'reserve':
      regions = self._reserve_regions(target)
      if not regions:
        return f'{actor} found no woods being felled to protect.'
      ticks = int(effect['ticks'])
      for region in regions:
        self.reserves[region] = max(
            self.reserves.get(region, 0), self.tick + 1 + ticks
        )
      return (
          f'{actor} chose to {intervention.label}{where}: the peoples may not'
          f' fell the woods of {len(regions)} lands for {ticks} turns.'
      )
    if effect['kind'] == 'cleanse':
      if self.backlash is not None:
        self.backlash.cleanse(float(effect['fraction']))
      return f'{actor} chose to {intervention.label}.'
    if effect['kind'] == 'seas':
      if self.civ is None:
        return f'{actor} tended the coasts, but no one fishes these seas.'
      self.ocean.nutrients *= 1.0 - float(effect['flush'])
      if target is not None and self.pollution is not None:
        # The nearest coast's runoff is cleared first.
        near = self._near_target(target) <= 4
        self.pollution.plume[near] *= max(
            0.0, 1.0 - 2.0 * float(effect['flush'])
        )
      self.civ.fish_stock = min(
          1.0, self.civ.fish_stock + float(effect['restock'])
      )
      return (
          f'{actor} chose to {intervention.label}{where}: the fish stocks'
          ' stand at'
          f' {self.civ.fish_stock:.0%}.'
      )
    if effect['kind'] == 'reductant':
      if self.chemistry is None:
        return f'{actor} washed iron into the seas, to no effect.'
      self.chemistry.add_reductant(float(effect['amount']), intervention.key)
      return f'{actor} chose to {intervention.label}.'
    if effect['kind'] == 'outgassing':
      if self.carbon is None:
        return f'{actor} stirred the volcanoes, but this world has no carbon.'
      released = self.carbon.outgas(float(effect['amount']), intervention.key)
      return (
          f'{actor} chose to {intervention.label}: {released:.0f} Gt of carbon'
          ' rose into the air.'
      )
    if effect['kind'] == 'veil':  # The strong moves.
      delta = float(effect['delta'])
      self.daisies.add_albedo_pulse(delta, lasting=True)
      words = (
          f'{actor} chose to {intervention.label}: the sky'
          f' {"pales" if delta > 0 else "darkens"}, the planet now reflecting'
          f' {abs(delta):.0%} {"more" if delta > 0 else "less"} sunlight,'
          ' fading over years'
      )
      carbon = float(effect.get('carbon', 0.0))
      if carbon > 0 and self.carbon is not None:
        released = self.carbon.outgas(carbon, intervention.key)
        words += f'; {released:.0f} Gt of carbon rose into the air'
      return words + '.'
    if effect['kind'] == 'carbon_drawdown':
      if self.carbon is None:
        return (
            f'{actor} spread basalt dust, but this world has no carbon cycle.'
        )
      drawn = self.carbon.drawdown(float(effect['amount']), intervention.key)
      return (
          f'{actor} chose to {intervention.label}: {drawn:.0f} Gt of carbon'
          ' left the air.'
      )
    raise ValueError(f'Unknown effect kind {effect["kind"]!r}')

  def climate_payload(self) -> dict[str, Any]:
    low, high = self.band()
    return {
        'tick': self.tick,
        'temp_c': round(self.temperature_c(), 2),
        'luminosity': round(self.luminosity, 4),
        'albedo': round(self.daisies.planetary_albedo(), 4),
        'band_c': [low, high],
        'out_of_band_ticks': self.out_of_band,
        'co2': round(self.carbon.relative_co2(), 3) if self.carbon else None,
        'greenhouse_w_m2': round(self.greenhouse_forcing(), 2),
        'oxygen_pal': (
            round(self.chemistry.oxygen(), 4) if self.chemistry else None
        ),
        'methane': (
            round(self.chemistry.methane(), 2) if self.chemistry else None
        ),
        'iron': (
            round(self.chemistry.reductant(), 2) if self.chemistry else None
        ),
        'ice': round(self.ice_cover(), 3),
        'bands': [
            {'band': b['band'], 'temp_c': b['temp_c'], 'ice': b['ice']}
            for b in self.daisies.band_summary()
        ],
    }

  def _herd_name(self, key: str) -> str:
    return str(self.herds.specs.get(key, {'name': key})['name']).lower()

  def _herd_census(self) -> dict[str, dict[str, Any]]:
    """The herds' census, with each kind's change over the last 3 ticks.

    A snapshot alone hides a boom: packs that double in three ticks look as
    healthy as packs that hold steady, until they have eaten out the herds.
    """
    out = self.herds.census()
    back = min(3, len(self.history) - 1)
    if back <= 0:
      return out
    past = self.history[-1 - back]
    for key, spec in self.herds.specs.items():
      before = past.get(f'herds_{key}')
      if spec['name'] in out and before is not None and before > 0.05:
        out[spec['name']]['change'] = round(
            self.herds.total(key) / float(before) - 1.0, 2
        )
        out[spec['name']]['over'] = back
    return out

  def census_payload(self) -> dict[str, Any]:
    snapshot = self.daisies.snapshot(self.luminosity)
    return {
        'noun': self.species_noun,
        'tick': self.tick,
        'cover': snapshot['cover'],
        'bare': snapshot['bare'],
        'local_temp_c': snapshot['local_temp_c'],
        'growth': snapshot['growth'],
        'bands': snapshot['bands'],
        'fauna': (
            {self.grazers.key: self.grazers.payload()} if self.grazers else {}
        ),
        'herds': self._herd_census(),
    }

  def has_vault(self) -> bool:
    """Whether the goal's ships carry only what the seed vault holds."""
    return bool(self.goal().get('vault'))

  def indices_payload(self) -> dict[str, Any]:
    seas = self.sea_health()
    vault = (
        {'held': round(self.vault, 3), 'need': self.goal()['min_biodiversity']}
        if self.has_vault()
        else None
    )
    found: dict[str, Any] = {
        'vault': vault,
        'tick': self.tick,
        'omega': round(self.omega, 2),
        'biodiversity': round(self.biodiversity(), 4),
        'seas': None if seas is None else round(seas, 3),
        'stability': round(self.stability(), 4),
        'instruments': self.instrument_text(),
        'restless': (
            {
                self.civ.polities[k].name: v
                for k, v in self.backlash.restless(self.homes()).items()
            }
            if self.backlash and self.civ
            else {}
        ),
    }
    if self.side:  # Side objectives, where the scenario sets any.
      found['side'] = self.side_payload()
    return found

  def instrument(self) -> dict[str, Any] | None:
    """The Steward's instrument tier, or None for exact, instant readings."""
    if self.instrument_tier is None:
      return None
    return self.instrument_tiers[self.instrument_tier]

  def can_upgrade_instruments(self) -> bool:
    return (
        self.instrument_tier is not None
        and self.instrument_tier < len(self.instrument_tiers) - 1
    )

  def instrument_text(self) -> str:
    """How good the Steward's instruments are, for its prompt."""
    tier = self.instrument()
    if tier is None:
      return ''
    latency = int(tier['latency'])
    late = {0: 'on time', 1: '1 tick late'}.get(
        latency, f'{latency} ticks late'
    )
    return (
        f'{tier["label"]} (temperatures ±{float(tier["noise_sigma"]):g} C,'
        f' {late}, {float(tier["drop_rate"]):.0%} of readings lost)'
    )

  def _burn(self) -> list[WorldEvent]:
    """This tick's wildfires: cover lost, carbon to the air, news."""
    life = self.tile_life
    oxygen = self.chemistry.oxygen() if self.chemistry is not None else None
    burned = self.fire.step(self.tick, life.covers, oxygen)
    life.growth_boost = 1.0 + self.fire.ash
    self.burned_tiles = int(burned.sum())
    if not self.burned_tiles:
      return []
    floor = self.daisies.params.min_cover * self.tiles.land
    area = 0.0
    for species, loss in self.fire.losses(life.covers, burned, floor).items():
      life.covers[species] = life.covers[species] - loss
      before = self.daisies.cover(species)
      self.daisies.graze(species, life.region_mean(loss))
      area += before - self.daisies.cover(species)
    if self.carbon is not None and area > 0:
      self.carbon.burn(self.carbon.biomass_per_cover * area, 'wildfire')
    news = int(self.fire.config['news_tiles'])
    events = []
    for patch in self.fire.fires(burned)[:2]:
      if patch.sum() >= news:
        rows, cols = np.nonzero(patch)
        # Longitude wraps: average the columns round the circle.
        turn = 2 * np.pi * cols / self.tiles.cols
        angle = np.arctan2(np.sin(turn).mean(), np.cos(turn).mean())
        col = (
            int(round(angle / (2 * np.pi) * self.tiles.cols)) % self.tiles.cols
        )
        events.append(
            WorldEvent(
                BIO_EVENTS,
                self.fire.describe(patch),
                where=(int(round(rows.mean())), col),
            )
        )
    return events

  def _roam(self) -> list[WorldEvent]:
    """This tick's herds: grazing, hunting, disease and movement."""
    life = self.tile_life
    floor = self.daisies.params.min_cover * self.tiles.land
    news, eaten, sown = self.herds.step(
        self.tick,
        life.covers,
        self.fire.burned_last,
        floor,
        fish_stock=self.civ.fish_stock if self.civ is not None else None,
        dead=self.pollution.dead if self.pollution is not None else None,
    )
    for species, crop in eaten.items():
      if species in life.covers and crop.any():
        life.covers[species] = life.covers[species] - crop
        self.daisies.graze(species, life.region_mean(crop))
    for species, seeds in sown.items():
      if species in life.covers and seeds.any():
        life.covers[species] = life.covers[species] + seeds
        self.daisies.plant(species, life.region_mean(seeds))
    return [
        WorldEvent(BIO_EVENTS, text, where=news_lib.place_of(text))
        for text in news
    ]

  def _record(self) -> None:
    snapshot = self.daisies.snapshot(self.luminosity)
    row: dict[str, Any] = {
        'tick': self.tick,
        'luminosity': snapshot['luminosity'],
        'temp_c': snapshot['temp_c'],
        'albedo': snapshot['albedo'],
        'cloud_albedo': snapshot['cloud_albedo'],
        'bare': snapshot['bare'],
        'omega': round(self.omega, 2),
        'biodiversity': round(self.biodiversity(), 4),
        'stability': round(self.stability(), 4),
        'out_of_band': self.out_of_band,
        'ice': snapshot['ice'],
        'co2': round(self.carbon.relative_co2(), 4) if self.carbon else 1.0,
        'oxygen': round(self.chemistry.oxygen(), 5) if self.chemistry else 0.0,
        'methane': (
            round(self.chemistry.methane(), 3) if self.chemistry else 0.0
        ),
        'population': round(self.civ.population(), 2) if self.civ else 0.0,
        'iron_made': round(self.civ.made('iron'), 2) if self.civ else 0.0,
        'attacks': self.backlash.count if self.backlash else 0,
        'energy_made': round(self.civ.made('energy'), 2) if self.civ else 0.0,
        'sea_level_m': round(self.ocean.sea_level_m, 1),
        'sea_ice': round(self.ocean.sea_ice, 3),
        'ocean_ph': round(self.ocean.ph, 3),
        'greenhouse': round(self.carbon.forcing(), 3) if self.carbon else 0.0,
        'fertility': round(self.daisies.fertility(), 4),
        'eruptions': self.lithosphere.count if self.lithosphere else 0,
        'burned_tiles': self.burned_tiles,
    }
    for species in self.herds.specs:
      row[f'herds_{species}'] = round(self.herds.total(species), 3)
    for species, cover in snapshot['cover'].items():
      row[f'cover_{species}'] = cover
    for i, band in enumerate(snapshot['bands']):
      row[f'temp_band{i}'] = band['temp_c']
    if self.grazers is not None:
      row['grazers'] = round(self.grazers.total(), 5)
      row['grazer_diet_white'] = round(self.grazers.diet_white, 3)
      row['grazer_optimum'] = round(self.grazers.optimum_c, 2)
    for species in self.daisies.species():
      traits = self.daisies.traits(species)
      row[f'optimum_{species}'] = round(traits.optimum_c, 3)
      row[f'albedo_{species}'] = round(traits.albedo, 4)
      row[f'tolerance_{species}'] = round(traits.tolerance, 2)
      row[f'vigor_{species}'] = round(traits.vigor, 2)
      row[f'hardiness_{species}'] = round(traits.hardiness, 2)
      row[f'behaviour_{species}'] = self.behaviours[species]
      row[f'adaptations_{species}'] = self.adaptation_counts[species]
    self.history.append(row)
    self.life_seen[self.tick] = {
        k: np.array(v, copy=True) for k, v in self.tile_life.covers.items()
    }
    self.risk_seen[self.tick] = self.fire.risk(self.life_seen[self.tick])
    for old in [t for t in self.life_seen if t < self.tick - LIFE_SEEN_TICKS]:
      del self.life_seen[old]
      self.risk_seen.pop(old, None)

  def get_state(self) -> dict[str, Any]:
    return {
        'tick': self.tick,
        'era': self.era,
        'luminosity': self.luminosity,
        'omega': self.omega,
        'out_of_band': self.out_of_band,
        'instrument_tier': self.instrument_tier,
        'outcome': self.outcome,
        'outcome_reason': self.outcome_reason,
        'recent_temps': list(self._recent_temps),
        'ready_at': dict(self._ready_at),
        'made': [[t, k] for t, k in self.made],
        'omens': [[t, sign, said] for t, sign, said in self.omens],
        'reserves': {str(k): v for k, v in self.reserves.items()},
        'last_cut': [float(x) for x in self._last_cut],
        'daisies': self.daisies.get_state(),
        'history': [dict(row) for row in self.history],
        'pending_adaptations': [p.to_dict() for p in self.pending_adaptations],
        'adaptation_counts': dict(self.adaptation_counts),
        'enacted': dict(self.enacted),
        'behaviours': dict(self.behaviours),
        'seed_banks': self.seed_banks.get_state() if self.seed_banks else None,
        'lithosphere': (
            self.lithosphere.get_state() if self.lithosphere else None
        ),
        'carbon': self.carbon.get_state() if self.carbon else None,
        'chemistry': self.chemistry.get_state() if self.chemistry else None,
        'civ': self.civ.get_state() if self.civ else None,
        'backlash': self.backlash.get_state() if self.backlash else None,
        'treaties': self.treaties.get_state() if self.treaties else None,
        'gazette': self.gazette.get_state() if self.gazette else None,
        'colony': self.colony.get_state() if self.colony else None,
        'struck': dict(self.struck),
        'launches': [list(x) for x in self.launches],
        'vault': self.vault,
        'side': dict(self.side),
        'ships_launched': dict(self.ships_launched),
        'trade': self.trade.get_state() if self.trade else None,
        'ocean': self.ocean.get_state(),
        'tiles': self.tiles.get_state(),
        'biomes': self.biomes.get_state(),
        'tile_life': self.tile_life.get_state(),
        'fire': self.fire.get_state(),
        'herds': self.herds.get_state(),
        'settlements': (
            self.settlements.get_state() if self.settlements else None
        ),
        'pollution': self.pollution.get_state() if self.pollution else None,
        'colony_world': (
            self.colony_world.get_state() if self.colony_world else None
        ),
        'grazers': self.grazers.get_state() if self.grazers else None,
        'speciation': self.speciation.get_state() if self.speciation else None,
        'symbiosis': self.symbiosis.get_state() if self.symbiosis else None,
        'species_info': (
            [dataclasses.asdict(s) for s in self.evolution.species]
            if self.evolution
            else []
        ),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.life_seen = {}  # Display only: refilled as the game goes on.
    self.risk_seen = {}
    self.plantings = []
    self.tick = int(state['tick'])
    self.era = str(state['era'])
    self.luminosity = float(state['luminosity'])
    self.omega = float(state['omega'])
    self.out_of_band = float(state['out_of_band'])
    self.instrument_tier = state.get('instrument_tier', self.instrument_tier)
    self.outcome = state['outcome']
    self.outcome_reason = str(state['outcome_reason'])
    self._recent_temps = [float(t) for t in state['recent_temps']]
    self._ready_at = {k: int(v) for k, v in state['ready_at'].items()}
    self.made = [(int(t), str(k)) for t, k in state.get('made', ())]
    self.omens = [
        (int(t), str(sign), str(said))
        for t, sign, said in state.get('omens', ())
    ]
    self.reserves = {
        int(k): int(v) for k, v in (state.get('reserves') or {}).items()
    }
    if state.get('last_cut') is not None:
      self._last_cut = np.array(state['last_cut'], dtype=float)
    self.daisies.set_state(state['daisies'])
    self.history = [dict(row) for row in state['history']]
    self.pending_adaptations = [
        evolution_lib.PendingAdaptation(**p)
        for p in state.get('pending_adaptations', [])
    ]
    self.adaptation_counts = dict(state.get('adaptation_counts', {}))
    self.enacted = {k: int(v) for k, v in state.get('enacted', {}).items()}
    self.behaviours = dict(state.get('behaviours', {}))
    if self.seed_banks is not None and state.get('seed_banks'):
      self.seed_banks.set_state(state['seed_banks'])
    if self.lithosphere is not None and state.get('lithosphere'):
      self.lithosphere.set_state(state['lithosphere'])
    if self.carbon is not None and state.get('carbon'):
      self.carbon.set_state(state['carbon'])
    if self.chemistry is not None and state.get('chemistry'):
      self.chemistry.set_state(state['chemistry'])
    if self.civ is not None and state.get('civ'):
      self.civ.set_state(state['civ'])
    if self.backlash is not None and state.get('backlash'):
      self.backlash.set_state(state['backlash'])
    if self.treaties is not None and state.get('treaties'):
      self.treaties.set_state(state['treaties'])
    if self.gazette is not None and state.get('gazette'):
      self.gazette.set_state(state['gazette'])
    if self.colony is not None and state.get('colony'):
      self.colony.set_state(state['colony'])
    self.struck = {k: int(v) for k, v in state.get('struck', {}).items()}
    self.vault = float(state.get('vault', 0.0))
    self.side.update(
        {str(k): str(v) for k, v in (state.get('side') or {}).items()}
    )
    self.launches = [
        (int(t), str(k), float(b)) for t, k, b in state.get('launches', [])
    ]
    self.ships_launched = {
        k: int(v) for k, v in state.get('ships_launched', {}).items()
    }
    if self.trade is not None and state.get('trade'):
      self.trade.set_state(state['trade'])
    if state.get('ocean'):
      self.ocean.set_state(state['ocean'])
    if state.get('tiles'):
      self.tiles.set_state(state['tiles'])
    if state.get('biomes'):
      self.biomes.set_state(state['biomes'])
    self.biomes.update(self.daisies.temperatures_c())
    if state.get('tile_life'):
      # The saved tiles are exact; region changes since the save reach them
      # at the next substep, as they would have in the live game.
      self.tile_life.set_state(state['tile_life'])
    else:  # An older checkpoint: lay the regions onto the tiles.
      self.tile_life.reconcile(self.daisies)
    if state.get('herds'):
      self.herds.set_state(state['herds'])
    if state.get('settlements') and self.settlements is not None:
      self.settlements.set_state(state['settlements'])
    if state.get('pollution') and self.pollution is not None:
      self.pollution.set_state(state['pollution'])
    if state.get('colony_world') and self.colony_world is not None:
      self.colony_world.set_state(state['colony_world'])
    if state.get('fire'):
      self.fire.set_state(state['fire'])
      self.tile_life.growth_boost = 1.0 + self.fire.ash
    if self.civ is not None:
      # The seas' state the peoples last saw (pushed before each step).
      self.civ.ocean_ph = self.ocean.ph
      self.civ.sea_ice = self.ocean.sea_ice
      self.civ.dead_zone = self.ocean.dead_zone
    if self.grazers is not None and state.get('grazers'):
      self.grazers.set_state(state['grazers'])
    if self.speciation is not None and state.get('speciation'):
      self.speciation.set_state(state['speciation'])
    if self.symbiosis is not None and state.get('symbiosis'):
      self.symbiosis.set_state(state['symbiosis'])
    if self.evolution is not None and state.get('species_info'):
      self.evolution.species = [
          evolution_lib.SpeciesInfo(**s) for s in state['species_info']
      ]
