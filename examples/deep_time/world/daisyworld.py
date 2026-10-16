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

"""Regional Daisyworld (Watson & Lovelock, 1983) on a latitude × sector grid.

Daisies grow on the land of each region when their local temperature suits
them. Their albedo changes the region's albedo, which changes the regional
energy balance (`world/climate.py`), which diffuses heat to the neighbours:
the daisy thermostat, now with poles, an equator and ice. Coefficients live in
`data/daisyworld.yaml` and `data/climate.yaml`; traits can change at run time
(adaptation), so they are part of the checkpointed state.
"""

# pylint: disable=g-doc-args,missing-function-docstring

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
from typing import Any

from examples.deep_time.world import climate as climate_lib
import numpy as np


@dataclasses.dataclass
class SpeciesTraits:
  """Heritable traits of a daisy species.

  Genome levels (0-10) shape the niche and the life history; every level
  costs growth (see `GenomeEffects`).
  """

  albedo: float
  optimum_c: float
  growth_width: float
  tolerance: float = 0.0
  vigor: float = 0.0
  hardiness: float = 0.0

  def to_dict(self) -> dict[str, float]:
    return dataclasses.asdict(self)

  def levels(self) -> float:
    return self.tolerance + self.vigor + self.hardiness


@dataclasses.dataclass(frozen=True)
class GenomeEffects:
  """How genome levels change growth and death (`data/species.yaml`)."""

  tolerance_per_level: float = 0.0
  vigor_per_level: float = 0.0
  hardiness_per_level: float = 0.0
  upkeep_per_level: float = 0.0

  @classmethod
  def from_config(cls, config: Mapping[str, Any] | None) -> 'GenomeEffects':
    if not config:
      return cls()
    return cls(
        tolerance_per_level=float(config['tolerance_per_level']),
        vigor_per_level=float(config['vigor_per_level']),
        hardiness_per_level=float(config['hardiness_per_level']),
        upkeep_per_level=float(config['upkeep_per_level']),
    )

  def niche_width(self, traits: SpeciesTraits) -> float:
    return traits.growth_width / (
        1.0 + self.tolerance_per_level * traits.tolerance
    )

  def growth_factor(self, traits: SpeciesTraits) -> float:
    vigor = 1.0 + self.vigor_per_level * traits.vigor
    upkeep = max(0.0, 1.0 - self.upkeep_per_level * traits.levels())
    return vigor * upkeep

  def death_factor(self, traits: SpeciesTraits) -> float:
    return max(0.1, 1.0 - self.hardiness_per_level * traits.hardiness)


@dataclasses.dataclass(frozen=True)
class DaisyParams:
  """Fixed biological coefficients."""

  bare_albedo: float
  death_rate: float
  min_cover: float
  substeps: int
  dt: float
  cloud_decay: float
  # A lasting veil (ash winters, soot and methane) decays more slowly.
  veil_decay: float = 0.85

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> 'DaisyParams':
    values = {}
    for field in dataclasses.fields(cls):
      cast = int if field.type in (int, 'int') else float
      if field.name in data or field.default is dataclasses.MISSING:
        values[field.name] = cast(data[field.name])
    return cls(**values)


class Daisyworld:
  """Daisy species on the land of a regional energy-balance planet."""

  def __init__(
      self,
      params: DaisyParams,
      traits: Mapping[str, SpeciesTraits],
      cover: Mapping[str, float],
      climate: climate_lib.EnergyBalanceClimate,
      land_fraction: Sequence[float] | np.ndarray,
      luminosity: float = 1.0,
      initial_temp_c: float = 0.0,
      genome: GenomeEffects | None = None,
  ):
    """Initializes the model and solves the initial climate.

    Args:
      params: Biological coefficients.
      traits: Traits per species name.
      cover: Initial fraction of each region's *land* covered by each species.
      climate: The regional climate solver.
      land_fraction: Fraction of each region that is land.
      luminosity: Luminosity for the initial climate solve.
      initial_temp_c: Starting guess for the climate solve; with ice the climate
        can be bistable, so a cold guess finds the frozen state.
      genome: Effects of genome levels (none by default).

    Raises:
      ValueError: If covers are negative or exceed the land.
    """
    self._params = params
    self._genome = genome or GenomeEffects()
    self._climate = climate
    self._grid = climate.grid
    self._land = np.clip(np.asarray(land_fraction, dtype=float), 0.0, 1.0)
    if self._land.shape != (self._grid.size,):
      raise ValueError('land_fraction must have one entry per region')
    self._traits = {k: copy.copy(v) for k, v in traits.items()}
    self._cover = {
        k: float(cover.get(k, 0.0)) * self._land for k in self._traits
    }
    if any(float(cover.get(k, 0.0)) < 0 for k in self._traits) or np.any(
        self.bare_map() < -1e-9
    ):
      raise ValueError(f'Invalid initial cover: {dict(cover)}')
    self._behaviour = {k: (1.0, 1.0) for k in self._traits}
    self._cloud = 0.0
    # Clouds or haze over chosen regions (a targeted pulse), decaying alike.
    self._local_cloud = np.zeros(self._grid.size)
    self._veil = 0.0  # A lasting, planet-wide albedo change.
    self._fertility = 1.0
    self._fertility_decay = 1.0
    self._co2_factor = 1.0
    # Growth multipliers by source (e.g. 'mutualism', 'chemistry'), then by
    # species, per region.
    self._modifiers: dict[str, dict[str, np.ndarray]] = {}
    # A species' growth pushed toward some regions for one tick (a person
    # aiming "spread aggressively"), by species.
    self._focus: dict[str, np.ndarray] = {}
    self._forcing = 0.0
    self._temps = np.full(self._grid.size, float(initial_temp_c))
    self._albedo = np.zeros(self._grid.size)
    # Grows life on finer tiles instead: called once per
    # substep with (this model, dt), it returns the new cover per region.
    self.tile_growth: (
        Callable[['Daisyworld', float], dict[str, np.ndarray]] | None
    ) = None
    self._solve(luminosity)

  @property
  def params(self) -> DaisyParams:
    return self._params

  @property
  def climate(self) -> climate_lib.EnergyBalanceClimate:
    return self._climate

  def species(self) -> list[str]:
    return list(self._traits)

  def traits(self, species: str) -> SpeciesTraits:
    return self._traits[species]

  def set_traits(self, species: str, traits: SpeciesTraits) -> None:
    self._traits[species] = copy.copy(traits)

  def add_species(
      self, species: str, traits: SpeciesTraits, cover: np.ndarray
  ) -> None:
    """Founds a new species with the given cover taken from its parent.

    The caller moves the founders: `cover` must already have been removed
    from the parent (see `world/speciation.py`), so land is never exceeded.

    Args:
      species: New species key.
      traits: Its traits.
      cover: Founding cover per region (fraction of region area).
    """
    if species in self._traits:
      raise ValueError(f'Species {species!r} already exists')
    self._traits[species] = copy.copy(traits)
    self._cover[species] = np.maximum(
        self._params.min_cover * self._land, np.asarray(cover, dtype=float)
    )
    self._behaviour[species] = (1.0, 1.0)

  def move_cover(self, source: str, target: str, amount: np.ndarray) -> None:
    """Moves cover from one species to another, region by region."""
    current = self._cover[source]
    floor = self._params.min_cover * self._land
    moved = np.clip(amount, 0.0, np.maximum(0.0, current - floor))
    self._cover[source] = current - moved
    self._cover[target] = self._cover[target] + moved

  def set_cover(self, species: str, cover: np.ndarray) -> None:
    """Sets a species' cover per region outright (no seed-bank floor)."""
    self._cover[species] = np.clip(np.asarray(cover, dtype=float), 0.0, None)

  def plant(self, species: str, amount: np.ndarray) -> np.ndarray:
    """Adds cover per region on bare land; returns what was planted."""
    planted = np.clip(amount, 0.0, np.maximum(0.0, self.bare_map()))
    self._cover[species] = self._cover[species] + planted
    return planted

  def set_behaviour(
      self, species: str, growth_multiplier: float, death_multiplier: float
  ) -> None:
    """Sets how hard a species pushes growth and how much it risks dying."""
    self._behaviour[species] = (growth_multiplier, death_multiplier)

  def behaviour(self, species: str) -> tuple[float, float]:
    return self._behaviour[species]

  def land_fraction(self) -> np.ndarray:
    return self._land.copy()

  # Area-weighted global summaries (fractions of the whole planet).

  def _global(self, values: np.ndarray) -> float:
    return self._climate.global_mean(values)

  def cover(self, species: str) -> float:
    return self._global(self._cover[species])

  def cover_map(self, species: str) -> np.ndarray:
    return self._cover[species].copy()

  def bare_map(self) -> np.ndarray:
    bare = self._land.copy()
    for cover in self._cover.values():
      bare = bare - cover
    return bare

  def bare(self) -> float:
    return self._global(self.bare_map())

  def land(self) -> float:
    return self._global(self._land)

  def cloud_albedo(self) -> float:
    return self._cloud + self._veil + self._global(self._local_cloud)

  def fertility(self) -> float:
    """Growth multiplier from mineral ash; decays back toward 1."""
    return self._fertility

  def add_fertility(self, amount: float, decay: float) -> None:
    """Fertilises bare ground; the excess decays by `decay` per tick."""
    self._fertility += amount
    self._fertility_decay = decay

  def set_co2_factor(self, factor: float) -> None:
    """Scales daisy growth by CO2 availability (1 at the reference)."""
    self._co2_factor = max(0.0, float(factor))

  def set_growth_modifiers(
      self, source: str, multipliers: Mapping[str, np.ndarray]
  ) -> None:
    """Per-species, per-region growth multipliers from one source.

    Args:
      source: Name of the source, e.g. 'mutualism' or 'chemistry'; setting a
        source replaces its previous multipliers.
      multipliers: Species key -> multiplier per region.
    """
    self._modifiers[source] = {
        k: np.asarray(v, dtype=float) for k, v in multipliers.items()
    }

  def set_mutualism(self, multipliers: Mapping[str, np.ndarray]) -> None:
    """Growth multipliers from symbiosis pacts."""
    self.set_growth_modifiers('mutualism', multipliers)

  def growth_modifier(self, species: str) -> np.ndarray | float:
    result: np.ndarray | float = 1.0
    for multipliers in self._modifiers.values():
      if species in multipliers:
        result = result * multipliers[species]
    return result

  def greenhouse_forcing(self) -> float:
    return self._forcing

  def set_greenhouse_forcing(self, watts_per_m2: float) -> None:
    """Sets the greenhouse forcing used by the next climate solves."""
    self._forcing = float(watts_per_m2)

  def temperatures_c(self) -> np.ndarray:
    return self._temps.copy()

  def global_temperature_c(self) -> float:
    return self._global(self._temps)

  def planetary_albedo(self) -> float:
    return self._global(self._albedo)

  def ice_map(self) -> np.ndarray:
    open_fraction = self.bare_map() + (1.0 - self._land)
    return open_fraction * self._climate.ice_fraction(self._temps)

  # Physics.

  def _albedo_fn(self):
    p = self._params
    flower = np.zeros(self._grid.size)
    for k, t in self._traits.items():
      flower = flower + self._cover[k] * t.albedo
    bare = np.maximum(0.0, self.bare_map())
    ocean = 1.0 - self._land
    open_fraction = bare + ocean
    with np.errstate(invalid='ignore', divide='ignore'):
      open_albedo = np.where(
          open_fraction > 0,
          (bare * p.bare_albedo + ocean * self._climate.ocean_albedo)
          / open_fraction,
          0.0,
      )

    def albedo(temps: np.ndarray) -> np.ndarray:
      surface = self._climate.surface_albedo(
          open_albedo, flower, open_fraction, temps
      )
      return np.clip(
          surface + self._cloud + self._veil + self._local_cloud, 0.0, 1.0
      )

    return albedo

  def solve(self, luminosity: float) -> None:
    """Re-solves the climate (e.g. after the forcing changed)."""
    self._solve(luminosity)

  def _solve(self, luminosity: float) -> None:
    self._temps, self._albedo = self._climate.equilibrium(
        luminosity, self._albedo_fn(), self._forcing, self._temps
    )

  def local_temperature_c(self, species: str) -> np.ndarray:
    """Flower temperature per region: darker flowers run warmer."""
    contrast = self._albedo - self._traits[species].albedo
    return self._temps + self._climate.local_heating_c * contrast

  @property
  def genome(self) -> GenomeEffects:
    return self._genome

  def local_offset_c(self, species: str) -> np.ndarray:
    """How much warmer than its region a species' flowers run, per region."""
    contrast = self._albedo - self._traits[species].albedo
    return self._climate.local_heating_c * contrast

  def niche(self, species: str) -> tuple[float, float]:
    """(optimum temperature, niche width coefficient) of a species."""
    traits = self._traits[species]
    return traits.optimum_c, self._genome.niche_width(traits)

  def growth_factors(self, species: str) -> np.ndarray:
    """Growth multipliers per region: behaviour, soil, air, genome, sources."""
    growth_mult, _ = self._behaviour[species]
    traits = self._traits[species]
    factor = (
        growth_mult
        * self._fertility
        * self._co2_factor
        * self._genome.growth_factor(traits)
        * self.growth_modifier(species)
        * self._focus.get(species, 1.0)
    )
    return np.broadcast_to(np.asarray(factor, dtype=float), self._grid.size)

  def set_focus(self, species: str, factors: np.ndarray) -> None:
    """Multiplies a species' growth by region for the next tick only."""
    self._focus[species] = np.asarray(factors, dtype=float)

  def death_rate_of(self, species: str) -> float:
    _, death_mult = self._behaviour[species]
    traits = self._traits[species]
    return (
        self._params.death_rate * death_mult * self._genome.death_factor(traits)
    )

  def growth_rate(self, species: str) -> np.ndarray:
    traits = self._traits[species]
    local = self.local_temperature_c(species)
    width = self._genome.niche_width(traits)
    return np.maximum(0.0, 1.0 - width * (traits.optimum_c - local) ** 2)

  def step(self, luminosity: float) -> None:
    """Integrates one tick: climate equilibrium and daisy growth."""
    p = self._params
    floor = p.min_cover * self._land
    for _ in range(p.substeps):
      self._solve(luminosity)
      if self.tile_growth is not None:
        self._cover = self.tile_growth(self, p.dt)
        continue
      bare = np.maximum(0.0, self.bare_map())
      new_cover = {}
      for name, cover in self._cover.items():
        growth_mult, death_mult = self._behaviour[name]
        traits = self._traits[name]
        rate = (
            self.growth_rate(name)
            * growth_mult
            * self._fertility
            * self._co2_factor
            * self._genome.growth_factor(traits)
            * self.growth_modifier(name)
        )
        death = p.death_rate * death_mult * self._genome.death_factor(traits)
        change = cover * (bare * rate - death)
        new_cover[name] = np.maximum(floor, cover + p.dt * change)
      total = sum(new_cover.values())
      over = total > self._land
      if np.any(over):
        scale = np.where(over, self._land / np.maximum(total, 1e-12), 1.0)
        new_cover = {k: v * scale for k, v in new_cover.items()}
      self._cover = new_cover
    self._solve(luminosity)
    self._focus = {}
    self._cloud *= p.cloud_decay
    if abs(self._cloud) < 1e-4:
      self._cloud = 0.0
    self._local_cloud *= p.cloud_decay
    self._local_cloud[np.abs(self._local_cloud) < 1e-4] = 0.0
    self._veil *= p.veil_decay
    if abs(self._veil) < 1e-4:
      self._veil = 0.0
    self._fertility = 1.0 + (self._fertility - 1.0) * self._fertility_decay
    if self._fertility - 1.0 < 1e-4:
      self._fertility = 1.0

  def seed(self, species: str, amount: float) -> float:
    """Plants `amount` of the planet's area onto bare land, pro rata.

    Returns:
      The area (fraction of the planet) actually planted.
    """
    bare = np.maximum(0.0, self.bare_map())
    available = self._global(bare)
    planted = max(0.0, min(amount, available))
    if planted > 0:
      self._cover[species] = self._cover[species] + bare * planted / available
    return planted

  def kill(self, species: str, fraction: float) -> float:
    """Kills a fraction of a species everywhere; returns the area lost."""
    current = self._cover[species]
    floor = self._params.min_cover * self._land
    lost = np.clip(current * fraction, 0.0, np.maximum(0.0, current - floor))
    self._cover[species] = current - lost
    return self._global(lost)

  def graze(self, species: str, eaten: np.ndarray) -> np.ndarray:
    """Removes grazed cover per region (never below the seed bank).

    Args:
      species: Daisy species.
      eaten: Cover to remove per region (fraction of region area).

    Returns:
      The cover actually removed per region.
    """
    current = self._cover[species]
    floor = self._params.min_cover * self._land
    removed = np.clip(eaten, 0.0, np.maximum(0.0, current - floor))
    self._cover[species] = current - removed
    return removed

  def add_albedo_pulse(
      self,
      delta: float,
      regions: np.ndarray | None = None,
      lasting: bool = False,
  ) -> None:
    """Adds a decaying change to planetary albedo (clouds, ash, haze).

    Args:
      delta: The change to the planet's mean albedo.
      regions: Weights per region to gather it over (e.g. a target and its
        neighbours): the same mean change, concentrated there (at most 0.5 a
        region). None spreads it evenly.
      lasting: A veil over the whole planet that fades slowly (`veil_decay` a
        tick, not `cloud_decay`): an ash winter, a pall of soot.
    """
    if lasting:
      self._veil += delta
      return
    if regions is None:
      self._cloud += delta
      return
    weights = np.clip(np.asarray(regions, dtype=float), 0.0, None)
    mean = self._global(weights)
    if mean <= 0:
      self._cloud += delta
      return
    self._local_cloud = np.clip(
        self._local_cloud + delta * weights / mean, -0.5, 0.5
    )

  def local_cloud(self) -> np.ndarray:
    return self._local_cloud.copy()

  def cloud_field(self) -> np.ndarray:
    """Albedo pulses over each region: the planet-wide veil plus local cloud."""
    return self._cloud + self._veil + self._local_cloud

  def _cover_weighted(self, species: str, values: np.ndarray) -> float:
    weights = self._cover[species] * self._climate.weights
    total = weights.sum()
    if total <= 0:
      return float(np.dot(self._climate.weights, values))
    return float(np.dot(weights, values) / total)

  def band_summary(self) -> list[dict[str, Any]]:
    """Per latitude band: temperature, ice and cover of each species."""
    grid = self._grid
    temps = grid.band_means(self._temps)
    ice = grid.band_means(self.ice_map())
    covers = {k: grid.band_means(v) for k, v in self._cover.items()}
    return [
        {
            'band': grid.band_name(b),
            'temp_c': round(float(temps[b]), 1),
            'ice': round(float(ice[b]), 3),
            'cover': {k: round(float(v[b]), 3) for k, v in covers.items()},
        }
        for b in range(grid.bands)
    ]

  def snapshot(self, luminosity: float) -> dict[str, Any]:
    """Returns a JSON-friendly summary of the current state."""
    return {
        'luminosity': round(luminosity, 4),
        'temp_c': round(self.global_temperature_c(), 2),
        'albedo': round(self.planetary_albedo(), 4),
        'cloud_albedo': round(self.cloud_albedo(), 4),
        'ice': round(self._global(self.ice_map()), 4),
        'cover': {k: round(self.cover(k), 4) for k in self._traits},
        'bare': round(self.bare(), 4),
        'local_temp_c': {
            k: round(self._cover_weighted(k, self.local_temperature_c(k)), 2)
            for k in self._traits
        },
        'growth': {
            k: round(self._cover_weighted(k, self.growth_rate(k)), 4)
            for k in self._traits
        },
        'bands': self.band_summary(),
    }

  def get_state(self) -> dict[str, Any]:
    return {
        'cover': {k: v.tolist() for k, v in self._cover.items()},
        'cloud': self._cloud,
        'veil': self._veil,
        'local_cloud': self._local_cloud.tolist(),
        'focus': {k: v.tolist() for k, v in self._focus.items()},
        'traits': {k: v.to_dict() for k, v in self._traits.items()},
        'behaviour': {k: list(v) for k, v in self._behaviour.items()},
        'fertility': self._fertility,
        'fertility_decay': self._fertility_decay,
        'forcing': self._forcing,
        'co2_factor': self._co2_factor,
        'modifiers': {
            source: {k: v.tolist() for k, v in m.items()}
            for source, m in self._modifiers.items()
        },
        'temps': self._temps.tolist(),
        'albedo': self._albedo.tolist(),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self._traits = {k: SpeciesTraits(**v) for k, v in state['traits'].items()}
    self._cover = {
        k: np.array(v, dtype=float) for k, v in state['cover'].items()
    }
    self._cloud = float(state['cloud'])
    self._veil = float(state.get('veil', 0.0))
    self._local_cloud = np.array(
        state.get('local_cloud', np.zeros(self._grid.size)), dtype=float
    )
    self._focus = {
        k: np.array(v, dtype=float) for k, v in state.get('focus', {}).items()
    }
    self._behaviour = {
        k: (float(v[0]), float(v[1]))
        for k, v in state.get('behaviour', {}).items()
    }
    for k in self._traits:
      self._behaviour.setdefault(k, (1.0, 1.0))
    self._fertility = float(state.get('fertility', 1.0))
    self._fertility_decay = float(state.get('fertility_decay', 1.0))
    self._forcing = float(state.get('forcing', 0.0))
    self._co2_factor = float(state.get('co2_factor', 1.0))
    self._modifiers = {
        source: {k: np.array(v, dtype=float) for k, v in m.items()}
        for source, m in state.get('modifiers', {}).items()
    }
    self._temps = np.array(state['temps'], dtype=float)
    self._albedo = np.array(state['albedo'], dtype=float)


def from_config(
    config: Mapping[str, Any],
    cover: Mapping[str, float],
    climate: climate_lib.EnergyBalanceClimate,
    land_fraction: Sequence[float] | np.ndarray,
    luminosity: float = 1.0,
    initial_temp_c: float = 0.0,
    genome: Mapping[str, Any] | None = None,
) -> Daisyworld:
  """Builds a Daisyworld from `data/daisyworld.yaml` and a climate.

  Args:
    config: Contents of `data/daisyworld.yaml`.
    cover: Initial cover of each species' land.
    climate: The climate solver.
    land_fraction: Land per region.
    luminosity: Initial luminosity.
    initial_temp_c: Starting guess for the climate solve.
    genome: The `genome` block of `data/species.yaml` (initial levels and
      effects), or None for classic Daisyworld.

  Returns:
    The model.
  """
  initial = dict((genome or {}).get('initial', {}))
  traits = {
      name: SpeciesTraits(
          **{k: float(v) for k, v in values.items()},
          **{k: float(v) for k, v in initial.items()},
      )
      for name, values in config['species'].items()
  }
  return Daisyworld(
      DaisyParams.from_dict(config),
      traits,
      cover,
      climate,
      land_fraction,
      luminosity,
      initial_temp_c,
      GenomeEffects.from_config(genome),
  )
