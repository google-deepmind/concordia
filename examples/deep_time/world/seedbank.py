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

"""Seed banks: the cost of spreading and the source of comebacks.

Each daisy species keeps a seed bank (0 to 1). Spreading aggressively spends
it; holding steady refills it slowly and going dormant refills it fast. A
species that keeps spreading on an empty bank is exhausted: it loses the
spread's growth bonus and dies faster. A species that has nearly vanished
sprouts again from a full enough bank onto bare ground, so a species that
husbanded its seeds can come back from a bad spell.
"""

from collections.abc import Mapping, Sequence
from typing import Any


class SeedBanks:
  """One seed bank per species, and what spending it does."""

  def __init__(
      self,
      config: Mapping[str, Any],
      species: Sequence[str],
      present: Sequence[str] | None = None,
  ):
    """Initializes the banks.

    Args:
      config: The `seed_bank` section of `data/species.yaml`.
      species: Every daisy species.
      present: Species alive at the start; the others (not yet seeded) start
        with an empty bank. Defaults to all.
    """
    self.capacity = float(config['capacity'])
    self.change = {k: float(v) for k, v in config['change'].items()}
    self.exhausted = float(config['exhausted'])
    self.exhausted_growth = float(config['exhausted_growth'])
    self.exhausted_death = float(config['exhausted_death'])
    reseed = config['reseed']
    self.reseed_below = float(reseed['below_cover'])
    self.reseed_amount = float(reseed['amount'])
    self.reseed_cost = float(reseed['cost'])
    start = float(config['start'])
    alive = set(species if present is None else present)
    self.level = {s: start if s in alive else 0.0 for s in species}

  def add(self, species: str) -> None:
    self.level.setdefault(species, float(self.capacity) / 2)

  def spend(
      self, species: str, behaviour: str, share_of_land: float = 1.0
  ) -> bool:
    """One tick of a behaviour; returns whether the species is exhausted.

    Args:
      species: The species.
      behaviour: Its behaviour this tick ('spread', 'hold', 'dormant',
        'smother').
      share_of_land: Its cover as a share of land; a species below the regrowth
        threshold makes no seeds (its bank does not refill).

    Returns:
      Whether it spreads on an exhausted bank.
    """
    change = self.change.get(behaviour, 0.0)
    if change > 0 and share_of_land < self.reseed_below:
      change = 0.0
    level = self.level.get(species, 0.0) + change
    self.level[species] = min(self.capacity, max(0.0, level))
    growing = behaviour in ('spread', 'smother')  # Smothering spends faster.
    return growing and self.level[species] < self.exhausted

  def reseed(self, species: str, share_of_land: float) -> float:
    """Area a nearly vanished species may sprout again (0 if none)."""
    if share_of_land >= self.reseed_below or self.level.get(species, 0.0) < (
        self.reseed_cost
    ):
      return 0.0
    return self.reseed_amount

  def charge(self, species: str, planted: float) -> None:
    """Spends the bank for what actually sprouted (pro rata)."""
    share = min(1.0, planted / max(1e-9, self.reseed_amount))
    self.level[species] = max(
        0.0, self.level.get(species, 0.0) - self.reseed_cost * share
    )

  def text(self, species: str) -> str:
    level = self.level.get(species, 0.0)
    note = ''
    if level < self.exhausted:
      note = ' (nearly spent: spreading now exhausts us)'
    return f'Seed bank: {level:.0%}{note}.'

  def get_state(self) -> dict[str, Any]:
    return {'level': dict(self.level)}

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.level = {k: float(v) for k, v in state['level'].items()}
