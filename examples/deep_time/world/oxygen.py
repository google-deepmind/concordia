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

"""Oxygen and methane: the Great Oxidation.

Oxygenic photosynthesis makes oxygen, but at first reduced iron and volcanic
gases mop it up. Methanogens fill the air with methane, a strong greenhouse
gas that keeps a faint young sun's planet warm. When the reductant sink runs
out, oxygen accumulates: it poisons anaerobes, feeds aerobes and burns the
methane away, so the planet cools. Oxygen also builds an ozone shield; until
it forms, ultraviolet sears life where the sun is strongest, so early life
clings to the high latitudes and spreads toward the tropics as the shield
thickens. All three gases are ledger stocks on the contrib `StockFlowLedger`,
so every flux is audited.
"""

from collections.abc import Mapping
import math
from typing import Any

from concordia.contrib.components.game_master import stock_flow_ledger
from examples.deep_time.world import daisyworld
import numpy as np

AIR = 'air'
CRUST = 'crust'
OXYGEN = 'oxygen'
METHANE = 'methane'
REDUCTANT = 'reductant'

ANAEROBIC = 'anaerobic'
OXYGENIC = 'oxygenic'
AEROBIC = 'aerobic'


class Chemistry:
  """Atmospheric oxygen and methane, with crustal reductants."""

  def __init__(
      self,
      config: Mapping[str, Any],
      metabolisms: Mapping[str, str],
      ledger: stock_flow_ledger.Ledger | None = None,
  ):
    """Initializes the gases.

    Args:
      config: Contents of `data/oxygen.yaml`.
      metabolisms: Species key -> 'anaerobic', 'oxygenic' or 'aerobic'. Unlisted
        species are unaffected by oxygen.
      ledger: Ledger to keep the books in (a new one by default).
    """
    self.ledger = ledger if ledger is not None else stock_flow_ledger.Ledger()
    initial = config['initial']
    self.ledger.deposit(AIR, OXYGEN, float(initial['oxygen']), 'initial')
    self.ledger.deposit(AIR, METHANE, float(initial['methane']), 'initial')
    self.ledger.deposit(
        CRUST, REDUCTANT, float(initial['reductant']), 'initial'
    )
    self.oxygen_per_cover = float(config['oxygen_per_cover'])
    self.methane_per_cover = float(config['methane_per_cover'])
    self.sink_rate = float(config['reductant_sink_rate'])
    self.reductant_supply = float(config['reductant_supply'])
    self.oxidation_rate = float(config['oxidation_rate'])
    self.methane_loss = float(config['methane_lifetime_loss'])
    self.oxygen_loss = float(config['oxygen_loss'])
    self.respiration = float(config['respiration_per_cover'])
    self.methane_forcing_k = float(config['methane_forcing'])
    self.toxic_oxygen = float(config['toxic_oxygen'])
    self.aerobic_half = float(config['aerobic_half'])
    self.metabolisms = dict(metabolisms)
    ozone = config.get('ozone', {})
    self.ozone_half = float(ozone.get('half_oxygen', 0.0))
    self.uv_penalty = float(ozone.get('uv_penalty', 0.0))
    self.shield_alert = float(ozone.get('alert_shield', 1.0))
    self.last_fluxes: dict[str, float] = {}
    self.oxidized_at: int | None = None
    self.shielded_at: int | None = None  # When the ozone shield formed.

  def oxygen(self) -> float:
    return self.ledger.stock(AIR, OXYGEN)

  def methane(self) -> float:
    return self.ledger.stock(AIR, METHANE)

  def reductant(self) -> float:
    return self.ledger.stock(CRUST, REDUCTANT)

  def forcing(self) -> float:
    """Methane greenhouse forcing in W m^-2."""
    return self.methane_forcing_k * math.sqrt(max(0.0, self.methane()))

  def ozone(self) -> float:
    """The ozone shield in [0, 1] (1 without UV rules)."""
    if self.ozone_half <= 0:
      return 1.0
    o2 = max(0.0, self.oxygen())
    return o2 / (o2 + self.ozone_half)

  def uv(self, insolation: np.ndarray) -> np.ndarray:
    """Ultraviolet per region: the unshielded share of its sunlight."""
    sun = np.asarray(insolation, dtype=float)
    return (1.0 - self.ozone()) * sun / max(1e-9, float(sun.max()))

  def growth_factor(self, metabolism: str) -> float:
    o2 = max(0.0, self.oxygen())
    if metabolism == ANAEROBIC:
      return max(0.0, 1.0 - o2 / self.toxic_oxygen)
    if metabolism == AEROBIC:
      return o2 / (o2 + self.aerobic_half)
    return 1.0

  def multipliers(
      self,
      daisies: daisyworld.Daisyworld,
      insolation: np.ndarray | None = None,
  ) -> dict[str, np.ndarray]:
    """Growth per species and region: oxygen's effect, then ultraviolet's.

    Args:
      daisies: The living cover.
      insolation: Sunlight per region; with it (and UV rules), unshielded
        ultraviolet slows growth most where the sun is strongest.

    Returns:
      Species key -> multiplier per region.
    """
    size = daisies.land_fraction().shape
    sear = np.ones(size)
    if insolation is not None and self.uv_penalty > 0:
      sear = 1.0 - self.uv_penalty * self.uv(insolation)
    return {
        s: self.growth_factor(m) * sear
        for s, m in self.metabolisms.items()
        if s in daisies.species()
    }

  def cover_by_metabolism(
      self, daisies: daisyworld.Daisyworld
  ) -> dict[str, float]:
    totals = {ANAEROBIC: 0.0, OXYGENIC: 0.0, AEROBIC: 0.0}
    for species in daisies.species():
      metabolism = self.metabolisms.get(species)
      if metabolism in totals:
        totals[metabolism] += daisies.cover(species)
    return totals

  def step(self, daisies: daisyworld.Daisyworld, tick: int) -> list[str]:
    """Runs one tick of gas fluxes; returns noteworthy event texts."""
    covers = self.cover_by_metabolism(daisies)
    o2, ch4, red = self.oxygen(), self.methane(), self.reductant()
    produced = self.oxygen_per_cover * covers[OXYGENIC]
    methane = self.methane_per_cover * covers[ANAEROBIC]
    breathed = self.respiration * covers[AEROBIC] * self.growth_factor(AEROBIC)
    available = o2 + produced
    sunk = min(available, self.sink_rate * available, red)
    # Methane burns in proportion to methane and free oxygen, never using
    # more oxygen (two per methane) than the air holds.
    free = max(0.0, available - sunk)
    oxidised = min(
        ch4 + methane,
        free / 2.0,
        self.oxidation_rate * (ch4 + methane) * free,
    )
    ledger = self.ledger
    ledger.deposit(AIR, OXYGEN, produced, 'oxygenic photosynthesis')
    ledger.deposit(AIR, METHANE, methane, 'methanogenesis')
    ledger.deposit(CRUST, REDUCTANT, self.reductant_supply, 'volcanic gases')
    ledger.withdraw(AIR, OXYGEN, sunk, 'reduced by iron', partial=True)
    ledger.withdraw(CRUST, REDUCTANT, sunk, 'oxidised iron', partial=True)
    ledger.withdraw(AIR, METHANE, oxidised, 'methane oxidation', partial=True)
    ledger.withdraw(
        AIR, OXYGEN, 2 * oxidised, 'methane oxidation', partial=True
    )
    ledger.withdraw(
        AIR,
        METHANE,
        self.methane_loss * self.methane(),
        'photolysis',
        partial=True,
    )
    ledger.withdraw(AIR, OXYGEN, breathed, 'aerobic respiration', partial=True)
    ledger.withdraw(
        AIR,
        OXYGEN,
        self.oxygen_loss * self.oxygen(),
        'rock weathering',
        partial=True,
    )
    ledger.advance()
    self.last_fluxes = {
        'oxygen_produced': round(produced, 5),
        'oxygen_sunk': round(sunk, 5),
        'methane_produced': round(methane, 4),
        'methane_oxidised': round(oxidised, 4),
        'oxygen_breathed': round(breathed, 5),
    }
    events = []
    if self.shielded_at is None and self.ozone() >= self.shield_alert:
      self.shielded_at = tick
      events.append(
          'An ozone shield forms high in the air: the sunlit tropics open to'
          ' life.'
      )
    if self.oxidized_at is None and self.oxygen() >= self.toxic_oxygen:
      self.oxidized_at = tick
      events.append(
          'The iron sinks are full: free oxygen floods the air, poisoning'
          ' the anaerobes and burning away the methane.'
      )
    return events

  def add_reductant(self, amount: float, reason: str) -> None:
    self.ledger.deposit(CRUST, REDUCTANT, amount, reason)

  def payload(self) -> dict[str, Any]:
    return {
        'oxygen_pal': round(self.oxygen(), 5),
        'methane': round(self.methane(), 3),
        'reductant': round(self.reductant(), 3),
        'methane_forcing_w_m2': round(self.forcing(), 2),
        'oxidized_at': self.oxidized_at,
        'ozone': round(self.ozone(), 3),
        'fluxes': dict(self.last_fluxes),
    }

  def get_state(self) -> dict[str, Any]:
    return {
        'ledger': self.ledger.get_state(),
        'last_fluxes': dict(self.last_fluxes),
        'oxidized_at': self.oxidized_at,
        'shielded_at': self.shielded_at,
        'metabolisms': dict(self.metabolisms),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.ledger.set_state(state['ledger'])
    self.last_fluxes = dict(state.get('last_fluxes', {}))
    self.oxidized_at = state.get('oxidized_at')
    self.shielded_at = state.get('shielded_at')
    self.metabolisms = dict(state.get('metabolisms', self.metabolisms))
