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

"""Grazers: the first consumers of the food web.

A grazer clade lives on the land of every region and eats daisies with a
Holling type II functional response. Its diet preference decides *which*
petals it crops, so grazing changes the planet's albedo: herds that prefer
white petals darken and warm the world, herds that prefer black petals cool
it. Grazers grow where food is plentiful and the climate suits them, die back
where it does not, and wander between neighbouring land regions. Like the
daisies, they choose a behaviour each tick and evolve through a slow
adaptation mind whose answer this module turns into bounded trait changes.
"""

# pylint: disable=g-complex-comprehension,g-doc-return-or-yield,missing-function-docstring

from collections.abc import Mapping
import dataclasses
from typing import Any

from examples.deep_time.world import choices
from examples.deep_time.world import daisyworld
from examples.deep_time.world import grid as grid_lib
import numpy as np


@dataclasses.dataclass(frozen=True)
class GrazerBehaviour:
  key: str
  label: str
  attack: float
  mortality: float
  migration: float


@dataclasses.dataclass(frozen=True)
class GrazerOption:
  key: str
  label: str
  optimum: int
  diet: int


@dataclasses.dataclass
class _Pending:
  option: str
  ticks_left: int


class Grazers:
  """A grazer population per region with traits, behaviour and adaptation."""

  def __init__(
      self,
      config: Mapping[str, Any],
      grid: grid_lib.RegionGrid,
      land: np.ndarray,
  ):
    """Initializes the herds on every land region.

    Args:
      config: Contents of `data/fauna.yaml`.
      grid: The planet grid.
      land: Land fraction per region.
    """
    g = config['grazers']
    self.key = str(g['key'])
    self.name = str(g['name'])
    self.lore = str(g['lore'])
    self.attack_rate = float(g['attack_rate'])
    self.handling_time = float(g['handling_time'])
    self.efficiency = float(g['efficiency'])
    self.mortality = float(g['mortality'])
    self.crowding = float(g.get('crowding', 0.0))
    self.migration = float(g['migration'])
    self.substeps = max(1, int(g['substeps']))
    self.boom_bust = float(g['boom_bust_threshold'])
    self.optimum_c = float(g['optimum_c'])
    self.growth_width = float(g['growth_width'])
    self.diet_white = float(g['diet_white'])
    self.behaviours = [
        GrazerBehaviour(
            key=b['key'],
            label=b['label'],
            attack=float(b['attack']),
            mortality=float(b['mortality']),
            migration=float(b['migration']),
        )
        for b in config['behaviours']
    ]
    adaptation = config['adaptation']
    self.options = [GrazerOption(**o) for o in adaptation['options']]
    self.question = str(adaptation['question']).strip()
    self.latency_ticks = int(adaptation['latency_ticks'])
    self.phase_in_ticks = max(1, int(adaptation['phase_in_ticks']))
    self.optimum_step_c = float(adaptation['optimum_step_c'])
    self.diet_step = float(adaptation['diet_step'])
    low, high = adaptation.get('diet_range', (0.0, 1.0))
    self.diet_range = (float(low), float(high))
    self._grid = grid
    self._weights = grid.area_weights()
    self._land = np.asarray(land, dtype=float)
    self._floor = float(g['min_density']) * self._land
    self.density = float(g['initial_density']) * self._land
    self.behaviour = 'graze'
    self.adaptations = 0
    self._pending: list[_Pending] = []
    self._links = [
        (r, n, min(self._land[r], self._land[n]))
        for r in range(grid.size)
        for n in grid.neighbours(r)
        if r < n and min(self._land[r], self._land[n]) > 0
    ]

  def total(self) -> float:
    return float(np.dot(self._weights, self.density))

  def is_extinct(self) -> bool:
    return self.total() <= float(np.dot(self._weights, self._floor)) * 1.5

  def behaviour_spec(self) -> GrazerBehaviour:
    return choices.by_key(self.behaviours, self.behaviour)

  def behaviour_for_option(self, text: str) -> GrazerBehaviour | None:
    return choices.by_label_prefix(self.behaviours, text)

  def set_behaviour(self, key: str) -> str:
    self.behaviour = self.behaviour_spec_by_key(key).key
    return f'{self.name} {self.behaviour_spec().label}.'

  def behaviour_spec_by_key(self, key: str) -> GrazerBehaviour:
    return choices.by_key(self.behaviours, key)

  def suitability(self, temps_c: np.ndarray) -> np.ndarray:
    return np.maximum(
        0.0, 1.0 - self.growth_width * (self.optimum_c - temps_c) ** 2
    )

  def step(self, daisies: daisyworld.Daisyworld) -> list[str]:
    """Grazes, grows, dies back and migrates for one tick.

    Args:
      daisies: The daisy world to graze (covers are reduced in place).

    Returns:
      Noteworthy event texts (booms, crashes, extinction, adaptations).
    """
    events = self._phase_in()
    before = self.total()
    behaviour = self.behaviour_spec()
    attack = self.attack_rate * behaviour.attack
    suit = self.suitability(daisies.temperatures_c())
    # Pale petals (albedo above bare ground) count as white, dark as black.
    prefs = {
        s: (
            self.diet_white
            if daisies.traits(s).albedo >= daisies.params.bare_albedo
            else 1.0 - self.diet_white
        )
        for s in daisies.species()
    }
    dt = 1.0 / self.substeps
    for _ in range(self.substeps):
      covers = {s: daisies.cover_map(s) for s in prefs}
      food = sum(prefs[s] * covers[s] for s in prefs)
      intake = attack * food / (1.0 + attack * self.handling_time * food)
      eaten_total = self.density * intake * dt
      with np.errstate(invalid='ignore', divide='ignore'):
        shares = {
            s: np.where(food > 0, prefs[s] * covers[s] / food, 0.0)
            for s in prefs
        }
      eaten = sum(daisies.graze(s, eaten_total * shares[s]) for s in prefs)
      with np.errstate(invalid='ignore', divide='ignore'):
        realised = np.where(self.density > 0, eaten / (self.density * dt), 0.0)
      growth = self.efficiency * realised * suit
      death = (
          self.mortality * behaviour.mortality + self.crowding * self.density
      )
      self.density = np.maximum(
          self._floor, self.density + dt * self.density * (growth - death)
      )
      self._migrate(self.migration * behaviour.migration * dt)
    after = self.total()
    if before > 0 and not self.is_extinct():
      change = after / before - 1.0
      if change >= self.boom_bust:
        events.append(f'{self.name} herds boom ({change:+.0%}).')
      elif change <= -self.boom_bust:
        events.append(f'{self.name} herds crash ({change:+.0%}).')
    return events

  def _migrate(self, rate: float) -> None:
    """Moves herds between neighbouring land regions, conserving biomass.

    The flux between two regions uses the same area-weighted conductance as
    heat diffusion, then is divided by each region's area, so the planet's
    total grazer biomass is unchanged.

    Args:
      rate: Fraction of the density difference exchanged per link.
    """
    if rate <= 0:
      return
    areas = self._weights * self._grid.size
    change = np.zeros_like(self.density)
    for r, n, shared in self._links:
      conductance = 0.5 * (areas[r] + areas[n])
      flux = rate * shared * conductance * (self.density[n] - self.density[r])
      change[r] += flux / areas[r]
      change[n] -= flux / areas[n]
    self.density = np.maximum(self._floor, self.density + change)

  def cull(self, fraction: float, regions: np.ndarray | None = None) -> float:
    """Removes a fraction of the herds; returns the density lost.

    Args:
      fraction: Share removed (everywhere, or at the heaviest weight).
      regions: Weights per region (0-1) to cull where they are; None culls every
        herd alike.
    """
    before = self.total()
    share = min(1.0, max(0.0, fraction))
    if regions is not None:
      share = share * np.clip(np.asarray(regions, dtype=float), 0.0, 1.0)
    self.density = np.maximum(self._floor, self.density * (1.0 - share))
    return before - self.total()

  def drift(self, weights: np.ndarray, share: float = 0.3) -> None:
    """Moves a share of every herd toward the weighted regions (biomass kept).

    Args:
      weights: Pull per region (only land regions receive herds).
      share: Share of the grazers that moves.
    """
    pull = np.clip(np.asarray(weights, dtype=float), 0.0, None) * (
        self._land > 0
    )
    areas = self._weights
    room = float((pull * areas).sum())
    if room <= 0:
      return
    moved = share * float((self.density * areas).sum())
    self.density = self.density * (1.0 - share) + moved * pull / room
    self.density = np.maximum(self._floor, self.density)

  def parse_adaptation(self, text: str) -> GrazerOption | None:
    return choices.earliest_named(self.options, text)

  def queue_adaptation(self, answer: str) -> str:
    option = self.parse_adaptation(answer)
    if option is None or option.key == 'none':
      return f'{self.name} stay as they are.'
    self._pending.append(_Pending(option.key, self.phase_in_ticks))
    self.adaptations += 1
    return f'{self.name} begin evolving to {option.label}.'

  def _phase_in(self) -> list[str]:
    events = []
    still = []
    for pending in self._pending:
      option = choices.by_key(self.options, pending.option)
      fraction = 1.0 / self.phase_in_ticks
      self.optimum_c += option.optimum * self.optimum_step_c * fraction
      self.diet_white = min(
          self.diet_range[1],
          max(
              self.diet_range[0],
              self.diet_white + option.diet * self.diet_step * fraction,
          ),
      )
      pending.ticks_left -= 1
      if pending.ticks_left > 0:
        still.append(pending)
      else:
        events.append(
            f'{self.name} have evolved to {option.label} (optimum'
            f' {self.optimum_c:.1f} C, white-petal preference'
            f' {self.diet_white:.0%}).'
        )
    self._pending = still
    return events

  def evolving(self) -> list[tuple[str, int]]:
    """Adaptations still phasing in: (label, ticks left), oldest first."""
    return [
        (choices.by_key(self.options, p.option).label, p.ticks_left)
        for p in self._pending
    ]

  def genome_text(self) -> str:
    text = (
        f'Genome of the {self.key}: optimum {self.optimum_c:.1f} C; prefers'
        f' white petals {self.diet_white:.0%} of the time.'
    )
    pending = [p.option for p in self._pending]
    if pending:
      text += f' Currently evolving: {", ".join(pending)}.'
    return text

  def payload(self) -> dict[str, Any]:
    bands = self._grid.band_means(self.density)
    return {
        'density': round(self.total(), 4),
        'diet_white': round(self.diet_white, 3),
        'optimum_c': round(self.optimum_c, 2),
        'behaviour': self.behaviour,
        'bands': {
            self._grid.band_name(b): round(float(bands[b]), 4)
            for b in range(self._grid.bands)
        },
    }

  def get_state(self) -> dict[str, Any]:
    return {
        'density': self.density.tolist(),
        'optimum_c': self.optimum_c,
        'diet_white': self.diet_white,
        'behaviour': self.behaviour,
        'adaptations': self.adaptations,
        'pending': [dataclasses.asdict(p) for p in self._pending],
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.density = np.array(state['density'], dtype=float)
    self.optimum_c = float(state['optimum_c'])
    self.diet_white = float(state['diet_white'])
    self.behaviour = str(state['behaviour'])
    self.adaptations = int(state['adaptations'])
    self._pending = [_Pending(**p) for p in state['pending']]
