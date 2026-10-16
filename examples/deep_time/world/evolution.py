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

"""Evolution rules: behaviours (fast acts) and adaptations (slow minds).

A species' slow mind answers in words ("warmer optimum, because ..."). The
rules here turn that answer into a bounded trait change, phase it in over a
few ticks and charge its cost, so language models choose *what* to evolve but
never *how much*. Coefficients come from `data/species.yaml`.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

from examples.deep_time.world import choices
from examples.deep_time.world import daisyworld


@dataclasses.dataclass(frozen=True)
class SpeciesInfo:
  key: str
  name: str
  lore: str


@dataclasses.dataclass(frozen=True)
class Behaviour:
  key: str
  label: str
  growth: float
  death: float


@dataclasses.dataclass(frozen=True)
class AdaptationOption:
  """A heritable shift: steps of optimum, petal albedo and genome levels."""

  key: str
  label: str
  optimum: int = 0
  albedo: int = 0
  tolerance: int = 0
  vigor: int = 0
  hardiness: int = 0


GENOME_TRAITS = ('tolerance', 'vigor', 'hardiness')


@dataclasses.dataclass
class PendingAdaptation:
  """An adaptation being phased in over several ticks."""

  species: str
  option: str
  ticks_left: int

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


class Evolution:
  """Parses and applies species behaviours and adaptations."""

  def __init__(self, config: Mapping[str, Any]):
    """Initializes the rules from `data/species.yaml`."""
    self.species = [SpeciesInfo(**s) for s in config['species']]
    self.seed_bank_config = config.get('seed_bank')
    self.behaviours = [
        Behaviour(
            key=b['key'],
            label=b['label'],
            growth=float(b['growth']),
            death=float(b['death']),
        )
        for b in config['behaviours']
    ]
    adaptation = config['adaptation']
    self.options = [AdaptationOption(**o) for o in adaptation['options']]
    self.question = str(adaptation['question']).strip()
    self.latency_ticks = int(adaptation['latency_ticks'])
    self.phase_in_ticks = max(1, int(adaptation['phase_in_ticks']))
    self.optimum_step_c = float(adaptation['optimum_step_c'])
    self.albedo_step = float(adaptation['albedo_step'])
    self.optimum_range_c = tuple(
        float(x) for x in adaptation['optimum_range_c']
    )
    self.albedo_range = tuple(float(x) for x in adaptation['albedo_range'])
    self.tolerance_cost = float(adaptation['tolerance_cost'])
    self.genome_config = config.get('genome')
    self.genome_budget = float((self.genome_config or {}).get('budget', 30))
    self.speciation_config = config.get('speciation')

  def add_species(self, info: SpeciesInfo) -> None:
    self.species.append(info)

  def rename(self, key: str, name: str, lore: str) -> None:
    self.species = [
        SpeciesInfo(key, name, lore) if s.key == key else s
        for s in self.species
    ]

  def info(self, key: str) -> SpeciesInfo:
    return next(s for s in self.species if s.key == key)

  def behaviour(self, key: str) -> Behaviour:
    return choices.by_key(self.behaviours, key)

  def behaviour_for_option(self, text: str) -> Behaviour | None:
    return choices.by_label_prefix(self.behaviours, text)

  def option(self, key: str) -> AdaptationOption:
    return choices.by_key(self.options, key)

  def parse_adaptation(self, text: str) -> AdaptationOption | None:
    """Returns the option named earliest in a free-text answer, if any."""
    return choices.earliest_named(self.options, text)

  def apply_step(
      self,
      world: daisyworld.Daisyworld,
      species: str,
      option: AdaptationOption,
      fraction: float,
  ) -> daisyworld.SpeciesTraits:
    """Applies `fraction` of one adaptation step to a species' traits."""
    traits = world.traits(species)
    low_c, high_c = self.optimum_range_c
    low_a, high_a = self.albedo_range
    optimum = traits.optimum_c + option.optimum * self.optimum_step_c * fraction
    albedo = traits.albedo + option.albedo * self.albedo_step * fraction
    width = traits.growth_width
    if option.optimum:
      width *= (1.0 + self.tolerance_cost) ** fraction
    levels = {
        name: min(
            10.0,
            max(0.0, getattr(traits, name) + getattr(option, name) * fraction),
        )
        for name in GENOME_TRAITS
    }
    self._enforce_budget(levels, option)
    new_traits = daisyworld.SpeciesTraits(
        albedo=min(high_a, max(low_a, albedo)),
        optimum_c=min(high_c, max(low_c, optimum)),
        growth_width=width,
        **levels,
    )
    world.set_traits(species, new_traits)
    return new_traits

  def _enforce_budget(
      self, levels: dict[str, float], option: AdaptationOption
  ) -> None:
    """Over budget, the largest trait not being raised pays the excess."""
    excess = sum(levels.values()) - self.genome_budget
    while excess > 1e-9:
      payers = [
          name
          for name in GENOME_TRAITS
          if getattr(option, name) <= 0 and levels[name] > 0
      ] or [name for name in GENOME_TRAITS if levels[name] > 0]
      if not payers:
        return
      payer = max(payers, key=lambda name: levels[name])
      paid = min(levels[payer], excess)
      levels[payer] -= paid
      excess -= paid

  def genome_text(
      self, world: daisyworld.Daisyworld, species: str, noun: str = 'daisies'
  ) -> str:
    traits = world.traits(species)
    growth = world.genome.growth_factor(traits)
    return (
        f'Genome of the {species} {noun}: optimum {traits.optimum_c:.1f} C;'
        f' petal albedo {traits.albedo:.2f}; niche width'
        f' {world.genome.niche_width(traits):.4f} (larger is pickier);'
        f' tolerance {traits.tolerance:.1f}, vigor {traits.vigor:.1f},'
        f' hardiness {traits.hardiness:.1f}'
        f' ({traits.levels():.0f} of {self.genome_budget:.0f} levels; net'
        f' growth x{growth:.2f} after upkeep).'
    )
