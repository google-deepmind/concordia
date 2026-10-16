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

"""The carbon cycle on the contrib stock-flow ledger.

Reservoirs are ledger holders and every flux is a ledger transfer with a
reason, so carbon is conserved and each tick's fluxes are auditable:

* volcanic outgassing (mantle → atmosphere), boosted by eruptions;
* silicate weathering (atmosphere → sediments), faster when warm, when CO₂ is
  high and where plants cover the land: the planet's slow thermostat;
* ocean exchange toward a temperature-dependent equilibrium (warm seas hold
  less carbon);
* photosynthesis and decay (atmosphere ↔ biomass → soil → atmosphere) and
  burial (soil → sediments), then subduction (sediments → mantle).

Atmospheric carbon sets the greenhouse forcing used by the climate.
"""

from collections.abc import Mapping
import math
from typing import Any

from concordia.contrib.components.game_master import stock_flow_ledger

CARBON = 'carbon'
ATMOSPHERE = 'atmosphere'
OCEAN = 'ocean'
BIOMASS = 'biomass'
SOIL = 'soil'
SEDIMENTS = 'sediments'
MANTLE = 'mantle'


class CarbonCycle:
  """Carbon reservoirs and fluxes, one step per tick."""

  def __init__(
      self,
      config: Mapping[str, Any],
      *,
      reference_temp_c: float,
      initial_cover: float,
      land: float,
      ledger: stock_flow_ledger.Ledger | None = None,
  ):
    """Initializes reservoirs in a steady state.

    Args:
      config: Contents of `data/carbon.yaml`.
      reference_temp_c: Global temperature at which weathering plus burial
        balance baseline outgassing (usually the starting climate; a frozen
        scenario passes a warm reference so weathering stalls under ice).
      initial_cover: Living cover (fraction of the planet) at the start.
      land: Ice-free land fraction of the planet in the reference climate.
      ledger: Ledger to keep the books in (a new one by default).
    """
    self.ledger = ledger if ledger is not None else stock_flow_ledger.Ledger()
    self.reference = float(config['atmosphere_reference'])
    self.forcing_per_e_fold = float(config['forcing_per_e_fold'])
    self.baseline_outgassing = float(config['outgassing']['baseline'])
    self.outgassing_per_magnitude = float(
        config['outgassing']['per_eruption_magnitude']
    )
    weathering = config['weathering']
    self.temperature_sensitivity = float(weathering['temperature_sensitivity'])
    self.co2_exponent = float(weathering['co2_exponent'])
    self.biotic_enhancement = float(weathering['biotic_enhancement'])
    self.reference_temp_c = float(reference_temp_c)
    ocean = config['ocean']
    self.exchange_rate = float(ocean['exchange_rate'])
    self.ratio_at_reference = float(ocean['ratio_at_reference'])
    self.ratio_per_c = float(ocean['ratio_per_c'])
    self.biomass_per_cover = float(config['biomass']['per_cover'])
    self.co2_half_saturation = float(config['biomass']['co2_half_saturation'])
    self.biomass_growth = float(config['biomass']['growth_rate'])
    self.turnover = float(config['biomass']['turnover'])
    self.decay_rate = float(config['soil']['decay_rate'])
    self.burial_rate = float(config['soil']['burial_rate'])
    self.subduction_rate = float(config['subduction_rate'])

    # Start in steady state: biomass matches the living cover, soil inflow
    # (litterfall) matches decay plus burial, and weathering plus burial
    # balance baseline outgassing.
    biomass = self.biomass_per_cover * initial_cover
    soil = self.turnover * biomass / (self.decay_rate + self.burial_rate)
    initial = dict(config['reservoirs'], biomass=biomass, soil=soil)
    for reservoir, amount in initial.items():
      self.ledger.deposit(reservoir, CARBON, float(amount), reason='initial')
    burial = self.burial_rate * soil
    self.weathering_scale = max(0.0, self.baseline_outgassing - burial) / max(
        1e-9, land * (1.0 + self.biotic_enhancement * initial_cover)
    )
    self.last_fluxes: dict[str, float] = {}

  def amount(self, reservoir: str) -> float:
    return self.ledger.stock(reservoir, CARBON)

  def reservoirs(self) -> dict[str, float]:
    return {
        r: round(self.amount(r), 3)
        for r in (ATMOSPHERE, OCEAN, BIOMASS, SOIL, SEDIMENTS, MANTLE)
    }

  def relative_co2(self) -> float:
    return self.amount(ATMOSPHERE) / self.reference

  def co2_limitation(self) -> float:
    """Growth factor from CO2: 1 at the reference, falling as CO2 runs out."""
    c = max(0.0, self.relative_co2())
    k = self.co2_half_saturation
    return c / (c + k) * (1.0 + k)

  def forcing(self) -> float:
    """Greenhouse forcing in W m^-2 relative to the reference atmosphere."""
    return self.forcing_per_e_fold * math.log(
        max(1e-3 * self.reference, self.amount(ATMOSPHERE)) / self.reference
    )

  def ocean_ratio(self, temp_c: float) -> float:
    return max(
        0.5,
        self.ratio_at_reference
        + self.ratio_per_c * (temp_c - self.reference_temp_c),
    )

  def weathering_rate(self, temp_c: float, cover: float, land: float) -> float:
    return (
        self.weathering_scale
        * land
        * (1.0 + self.biotic_enhancement * cover)
        * math.exp(
            self.temperature_sensitivity * (temp_c - self.reference_temp_c)
        )
        * max(0.0, self.relative_co2()) ** self.co2_exponent
    )

  def _move(
      self, source: str, target: str, amount: float, reason: str
  ) -> float:
    if amount < 0:
      source, target, amount = target, source, -amount
    return self.ledger.transfer(
        source, target, CARBON, amount, reason=reason, partial=True
    )

  def outgas(self, amount: float, reason: str) -> float:
    """Moves carbon from the mantle into the air (e.g. roused volcanoes)."""
    return self._move(MANTLE, ATMOSPHERE, amount, reason)

  def burn(self, amount: float, reason: str) -> float:
    """Moves carbon from living biomass into the air (e.g. charcoal kilns)."""
    return self._move(BIOMASS, ATMOSPHERE, amount, reason)

  def unearth(self, amount: float, reason: str) -> float:
    """Moves fossil carbon from the sediments into the air (burning coal)."""
    return self._move(SEDIMENTS, ATMOSPHERE, amount, reason)

  def drawdown(self, amount: float, reason: str) -> float:
    """Moves carbon from the air into sediments (e.g. basalt dust)."""
    return self._move(ATMOSPHERE, SEDIMENTS, amount, reason)

  def step(
      self,
      *,
      temp_c: float,
      cover: float,
      land: float,
      eruption_magnitude: float = 0.0,
      sea_ice: float = 0.0,
  ) -> dict[str, float]:
    """Runs one tick of fluxes.

    Args:
      temp_c: Global mean temperature (last tick).
      cover: Living cover, fraction of the planet.
      land: Ice-free land fraction of the planet (frozen land weathers no rock).
      eruption_magnitude: Sum of this tick's eruption magnitudes.
      sea_ice: Fraction of the sea frozen over (chokes ocean gas exchange).

    Returns:
      The signed fluxes of this tick, keyed by process.
    """
    # Every flux is computed from the stocks at the start of the tick and
    # only then applied, so the order of processes does not matter.
    air = self.amount(ATMOSPHERE)
    biomass = self.amount(BIOMASS)
    soil = self.amount(SOIL)
    ratio = self.ocean_ratio(temp_c)
    capacity = self.biomass_per_cover * cover * self.co2_limitation()
    gap = self.biomass_growth * (capacity - biomass)
    litter = self.turnover * biomass
    planned = [
        (
            'outgassing',
            MANTLE,
            ATMOSPHERE,
            self.baseline_outgassing
            + self.outgassing_per_magnitude * eruption_magnitude,
        ),
        (
            'weathering',
            ATMOSPHERE,
            SEDIMENTS,
            self.weathering_rate(temp_c, cover, land),
        ),
        (
            'ocean_uptake',
            ATMOSPHERE,
            OCEAN,
            self.exchange_rate
            * (1.0 - min(1.0, max(0.0, sea_ice)))
            * (ratio * air - self.amount(OCEAN))
            / (1.0 + ratio),
        ),
        ('photosynthesis', ATMOSPHERE, BIOMASS, max(0.0, litter + gap)),
        ('litterfall', BIOMASS, SOIL, litter + max(0.0, -gap)),
        ('decay', SOIL, ATMOSPHERE, self.decay_rate * soil),
        ('burial', SOIL, SEDIMENTS, self.burial_rate * soil),
        (
            'subduction',
            SEDIMENTS,
            MANTLE,
            self.subduction_rate * self.amount(SEDIMENTS),
        ),
    ]
    fluxes = {}
    for name, source, target, amount in planned:
      moved = self._move(source, target, amount, name.replace('_', ' '))
      fluxes[name] = math.copysign(moved, amount)
    self.ledger.advance()
    self.last_fluxes = {k: round(v, 4) for k, v in fluxes.items()}
    return dict(self.last_fluxes)

  def payload(self) -> dict[str, Any]:
    return {
        'reservoirs': self.reservoirs(),
        'relative_co2': round(self.relative_co2(), 4),
        'forcing_w_m2': round(self.forcing(), 3),
        'fluxes': dict(self.last_fluxes),
    }

  def get_state(self) -> dict[str, Any]:
    return {
        'ledger': self.ledger.get_state(),
        'weathering_scale': self.weathering_scale,
        'reference_temp_c': self.reference_temp_c,
        'last_fluxes': dict(self.last_fluxes),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.ledger.set_state(state['ledger'])
    self.weathering_scale = float(state['weathering_scale'])
    self.reference_temp_c = float(state['reference_temp_c'])
    self.last_fluxes = dict(state.get('last_fluxes', {}))
