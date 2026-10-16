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

"""Speciation: a species that spans very different climates splits.

Each tick, every established daisy species is split into the colder and the
warmer half of its range (weighted by cover and area). If the two halves have
lived more than `split_threshold_c` apart for `split_ticks` ticks in a row, the
half farther from the parent's optimum founds a new species: it takes a share
of the outlying populations, an optimum matched to where it lives and a petal
colour shifted toward what that climate needs. Naming and the new entity are
the game layer's job; this module only decides and moves the populations.
"""

# pylint: disable=g-doc-args

from collections.abc import Callable, Mapping
import dataclasses
import re
from typing import Any

from examples.deep_time.world import daisyworld
from examples.deep_time.world import grid as grid_lib
import numpy as np


@dataclasses.dataclass(frozen=True)
class Split:
  """A speciation event decided this tick."""

  parent: str
  child: str
  colder: bool
  where: str
  optimum_c: float
  albedo: float
  founded_cover: float


class Speciation:
  """Detects divergent ranges and founds daughter species."""

  def __init__(self, config: Mapping[str, Any]):
    self.split_threshold_c = float(config['split_threshold_c'])
    self.split_ticks = int(config['split_ticks'])
    self.earliest_tick = int(config.get('earliest_tick', 0))
    self.max_species = int(config['max_species'])
    self.min_cover = float(config['min_cover'])
    self.albedo_shift = float(config['albedo_shift'])
    self.founder_share = float(config['founder_share'])
    self.naming_prompt = str(config['naming_prompt']).strip()
    self.streaks: dict[str, int] = {}
    self.founded = 0

  def spread(
      self, daisies: daisyworld.Daisyworld, species: str
  ) -> tuple[float, np.ndarray, np.ndarray]:
    """Temperature gap between the warm and cold halves of a range.

    Returns:
      (gap in C, mask of cold-half regions, local flower temperatures).
    """
    weights = daisies.cover_map(species) * daisies.climate.weights
    local = daisies.local_temperature_c(species)
    occupied = daisies.cover_map(species) > 3 * daisies.params.min_cover * (
        daisies.land_fraction() + 1e-12
    )
    if occupied.sum() < 2 or weights[occupied].sum() <= 0:
      return 0.0, np.zeros_like(local, dtype=bool), local
    order = np.argsort(local)
    cumulative = np.cumsum(np.where(occupied, weights, 0.0)[order])
    half = cumulative[-1] / 2
    cold = np.zeros_like(local, dtype=bool)
    cold[order[cumulative <= half]] = True
    cold &= occupied
    warm = occupied & ~cold
    if not cold.any() or not warm.any():
      return 0.0, cold, local
    mean = lambda mask: float(
        np.dot(weights[mask], local[mask]) / weights[mask].sum()
    )
    return mean(warm) - mean(cold), cold, local

  def check(
      self,
      daisies: daisyworld.Daisyworld,
      grid: grid_lib.RegionGrid,
      child_key: Callable[[str], str],
      tick: int = 0,
  ) -> Split | None:
    """Updates streaks and founds at most one new species this tick.

    Args:
      daisies: The daisy world (covers change if a species splits).
      grid: The planet grid, for naming where the founders live.
      child_key: Gives the key of a new species, from its parent's key.
      tick: Current tick; nothing splits before `earliest_tick`.

    Returns:
      The split, or None.
    """
    living = [
        s for s in daisies.species() if daisies.cover(s) >= self.min_cover
    ]
    plan = None
    for species in daisies.species():
      gap, cold, local = (
          self.spread(daisies, species)
          if species in living
          else (0.0, None, None)
      )
      if gap > self.split_threshold_c:
        self.streaks[species] = self.streaks.get(species, 0) + 1
      else:
        self.streaks[species] = 0
      if (
          plan is None
          and cold is not None
          and local is not None
          and self.streaks[species] >= self.split_ticks
          and tick >= self.earliest_tick
          and len(daisies.species()) < self.max_species
      ):
        plan = (species, cold, local)
    if plan is None:
      return None
    parent, cold, local = plan
    child = child_key(parent)
    traits = daisies.traits(parent)
    weights = daisies.cover_map(parent) * daisies.climate.weights
    warm = (weights > 0) & ~cold
    mean = lambda mask: float(
        np.dot(weights[mask], local[mask]) / max(1e-12, weights[mask].sum())
    )
    cold_t, warm_t = mean(cold), mean(warm)
    colder = abs(cold_t - traits.optimum_c) >= abs(warm_t - traits.optimum_c)
    mask = cold if colder else warm
    shift = -self.albedo_shift if colder else self.albedo_shift
    # Founders inherit the parent's genome levels and niche width.
    child_traits = dataclasses.replace(
        traits,
        albedo=float(np.clip(traits.albedo + shift, 0.1, 0.9)),
        optimum_c=cold_t if colder else warm_t,
    )
    founders = np.where(mask, daisies.cover_map(parent) * self.founder_share, 0)
    daisies.add_species(child, child_traits, np.zeros_like(founders))
    daisies.move_cover(parent, child, founders)
    self.streaks[parent] = 0
    self.streaks[child] = 0
    self.founded += 1
    bands = grid.band_means(mask.astype(float))
    where = grid.band_name(int(np.argmax(bands)))
    return Split(
        parent=parent,
        child=child,
        colder=colder,
        where=where,
        optimum_c=round(child_traits.optimum_c, 2),
        albedo=round(child_traits.albedo, 3),
        founded_cover=round(daisies.cover(child), 4),
    )

  def get_state(self) -> dict[str, Any]:
    return {'streaks': dict(self.streaks), 'founded': self.founded}

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.streaks = {k: int(v) for k, v in state.get('streaks', {}).items()}
    self.founded = int(state.get('founded', 0))


_NAME_RE = re.compile(
    r'Name:\s*(.+?)\.\s*Lore:\s*(.+)', re.IGNORECASE | re.DOTALL
)


def parse_name(text: str, noun: str = 'daisies') -> tuple[str, str] | None:
  """Parses `Name: <name>. Lore: <sentence>` from a model's answer.

  Args:
    text: The model's answer.
    noun: What the scenario calls its species ("daisies", "mats"); names are
      made to end with it.

  Returns:
    (name, lore), or None if the answer does not parse.
  """
  match = _NAME_RE.search(text)
  if not match:
    return None
  name = match.group(1).strip().strip('"*').strip()
  lore = match.group(2).strip().splitlines()[0].strip().strip('"*').strip()
  if not name or len(name) > 40 or not lore:
    return None
  if not name.lower().endswith(noun.lower()):
    name = f'{name} {noun}'
  return name[0].upper() + name[1:], lore
