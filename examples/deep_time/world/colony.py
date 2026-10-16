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

"""A colony on a world light-years from home (Light Delay).

The colonists landed with a few designs (how to mine ice, grow food under
glass, split water for air and tend a solar farm), some stores and solar
panels. Each generation they choose where to put their labor; recipes run on
a contrib `stock_flow_ledger.Ledger` of their own, and the contrib ratio
planner (`balance_chain`) can balance a whole chain at once. Colonists eat
food and breathe air; shortfalls kill, plenty grows the colony. Terraforming
plants turn power into warmth (which the sky slowly radiates away again),
and a warmer world makes the greenhouses yield more. New designs (such as
fabricating solar panels) come only from home, at the speed of light: the colony
and home see each other years late.
"""

# pylint: disable=missing-function-docstring

from collections.abc import Callable, Mapping
from typing import Any

from concordia.contrib.components.game_master import stock_flow_ledger
from examples.deep_time.world import civ as civ_lib
import numpy as np

HOLDER = 'colony'


class Colony:
  """The colonists, their stores, their labor and their world's climate."""

  def __init__(self, config: Mapping[str, Any], seed: int = 0):
    self.name = str(config['name'])
    self.light_delay = int(config['light_delay_ticks'])
    self.needs = {k: float(v) for k, v in config['needs'].items()}
    self.growth = float(config['growth'])
    self.deaths = float(config['deaths'])
    self.labor_per_colonist = float(config['labor_per_colonist'])
    # Each solar panel supports this many solar-farm batches a generation:
    # power is capped by panels, however much labor tends them.
    self.batches_per_panel = float(config['batches_per_panel'])
    self.wear = {k: float(v) for k, v in config.get('wear', {}).items()}
    env = config['environment']
    self.start_temp_c = float(env['temp_c'])
    self.temp_c = self.start_temp_c
    self.warmth_c = float(env['warmth_c_per_unit'])
    self.cooling = float(env['cooling'])
    self.target_temp_c = float(env['target_temp_c'])
    self.greenhouse_per_c = float(env['greenhouse_per_c'])
    self.recipes = civ_lib.parse_recipes(config['recipes'])
    # Where the colonists mine ice (a tile of `world/ararat.py`), if chosen,
    # and how that site's ice compares with the landing's (set each
    # generation from the map).
    self.site: tuple[int, int] | None = None
    self.mine_factor = 1.0
    # How good a site's ice is (set by the planet from New Ararat's map).
    self.site_factor: Callable[[tuple[int, int] | None], float] | None = None
    self.design_labels = {k: str(v) for k, v in config['designs'].items()}
    # What a labor shift to each recipe does, and each design (measured;
    # shown beside the moves).
    self.pushes = {k: str(v) for k, v in config.get('pushes', {}).items()}
    start = config['start']
    self.population = float(start['colonists'])
    self.start_population = self.population
    self.designs = list(start['designs'])
    self.allocation = self._normalised(start['allocation'])
    self.ledger = stock_flow_ledger.Ledger()
    for resource, amount in sorted(start['stocks'].items()):
      self.ledger.deposit(HOLDER, resource, float(amount), 'landing')
    self.shortfall: dict[str, float] = {}
    self.made: dict[str, float] = {}
    self.last_made: dict[str, float] = {}  # Made in the last generation.
    self.generation = 0
    self.deadline = int(config.get('deadline', 40))  # Generations to warm.
    self.warmed = 0.0  # How much the sky warmed last generation (°C).
    storms = config.get('storms', {})
    self.storm_chance = float(storms.get('chance', 0.0))
    self.storm_solar = float(storms.get('solar_factor', 1.0))
    self.storm = False  # A dust storm dims the solar farm this generation.
    self.panel_limited = False  # Had the solar farm more hands than panels?
    self._rng = np.random.default_rng(seed + 4242)
    # Transmissions from home on their way: (arrival generation, kind, what).
    self.incoming: list[tuple[int, str, str]] = []
    self.directive = ''  # The latest directive from home, as received.

  def _normalised(self, allocation: Mapping[str, float]) -> dict[str, float]:
    known = {r.key for r in self.recipes}
    shares = {
        k: max(0.0, float(v)) for k, v in allocation.items() if k in known
    }
    total = sum(shares.values())
    if total <= 0:
      return {self.recipes[0].key: 1.0}
    return {k: v / total for k, v in shares.items() if v > 0}

  def stock(self, resource: str) -> float:
    return self.ledger.stock(HOLDER, resource)

  def known(self) -> list[civ_lib.RecipeSpec]:
    """Recipes the colonists have the designs for."""
    return [
        r
        for r in self.recipes
        if r.requires is None or r.requires in self.designs
    ]

  def solar_yield(self) -> float:
    """The solar farm's output per batch: full, or dimmed by a dust storm."""
    return self.storm_solar if self.storm else 1.0

  def solar_batches(self) -> float:
    """The most solar-farm batches the panels support a generation."""
    return self.stock('panels') * self.batches_per_panel

  def _factor(self, spec: civ_lib.RecipeSpec) -> float:
    if spec.key == 'ice_mine':
      return self.mine_factor
    if spec.yield_kind == 'panels':
      return self.solar_yield()
    if spec.yield_kind == 'warmth':
      warmer = max(0.0, self.temp_c - self.start_temp_c)
      return 1.0 + self.greenhouse_per_c * warmer
    return 1.0

  def _effective(
      self, spec: civ_lib.RecipeSpec, name: str, calm: bool = False
  ) -> stock_flow_ledger.Recipe:
    """The recipe as run now (or, with `calm`, as run without a storm)."""
    factor = 1.0 if calm and spec.yield_kind == 'panels' else self._factor(spec)
    return stock_flow_ledger.Recipe(
        name=name,
        inputs=dict(spec.inputs),
        outputs={r: a * factor for r, a in spec.outputs.items()},
    )

  def shift_labor(self, recipe: str, share: float) -> None:
    share = min(1.0, max(0.0, share))
    scaled = {k: v * (1.0 - share) for k, v in self.allocation.items()}
    scaled[recipe] = scaled.get(recipe, 0.0) + share
    self.allocation = self._normalised(scaled)

  def plan(self, target: str, share: float) -> dict[str, float]:
    """Labor shares (summing to `share`) for `target`'s balanced chain."""
    known = {r.key: r for r in self.known()}
    chain = stock_flow_ledger.balance_chain(
        # Plans assume a calm sky: a storm passes, the allocation stays.
        [self._effective(spec, key, calm=True) for key, spec in known.items()],
        target,
    )
    weights = {
        k: rate * known[k].labor
        for k, rate in chain.batches.items()
        if rate > 0
    }
    total = sum(weights.values())
    return {k: share * w / total for k, w in weights.items()}

  def adopt_plan(self, target: str, share: float) -> dict[str, float]:
    shares = self.plan(target, share)
    scaled = {k: v * (1.0 - share) for k, v in self.allocation.items()}
    for key, value in shares.items():
      scaled[key] = scaled.get(key, 0.0) + value
    self.allocation = self._normalised(scaled)
    return shares

  def receive_design(self, design: str) -> str | None:
    """A design transmitted from home arrives; returns the news, if new."""
    if design in self.designs or design not in self.design_labels:
      return None
    self.designs.append(design)
    return (
        f'A transmission from home reaches {self.name}: the design for'
        f' {self.design_labels[design]}.'
    )

  def transmit(self, kind: str, what: str) -> int:
    """Home sends a design or a directive; returns its arrival generation."""
    arrival = self.generation + self.light_delay
    self.incoming.append((arrival, kind, what))
    return arrival

  def step(self) -> list[str]:
    """One generation of work, food, air and climate; returns the news."""
    events = []
    # This step makes generation `self.generation + 1`: what was sent at g
    # with a delay of d arrives in the step that makes generation g + d.
    now = self.generation + 1
    due = [m for m in self.incoming if m[0] <= now]
    self.incoming = [m for m in self.incoming if m[0] > now]
    for _, kind, what in due:
      if kind == 'design':
        news = self.receive_design(what)
        if news:
          events.append(news)
      elif kind == 'site':  # Home's survey: where to mine the ice.
        row, col = (int(v) for v in what.split(','))
        self.site = (row, col)
        events.append(
            f"Home's survey reaches {self.name}: mine the ice where it marked."
        )
      else:
        self.directive = what
        events.append(f'A directive from home reaches {self.name}: {what}')
    if self.site_factor is not None:  # After any survey has arrived.
      self.mine_factor = self.site_factor(self.site)
    self.storm = bool(self._rng.random() < self.storm_chance)
    if self.storm:
      events.append(
          f'A dust storm darkens the sky over {self.name}: the solar farm'
          f' runs at {self.storm_solar:.0%}.'
      )
    labor = self.population * self.labor_per_colonist
    made: dict[str, float] = {}
    for spec in self.known():
      share = self.allocation.get(spec.key, 0.0)
      if share <= 0:
        continue
      recipe = self._effective(spec, f'{self.name}: {spec.label}')
      wanted = labor * share / spec.labor
      if spec.yield_kind == 'panels':
        cap = self.solar_batches()
        self.panel_limited = wanted > cap + 1e-9
        wanted = min(wanted, cap)
      batches = self.ledger.apply_recipe(HOLDER, recipe, wanted, partial=True)
      for resource, amount in recipe.outputs.items():
        made[resource] = made.get(resource, 0.0) + batches * amount
    before_c = self.temp_c
    warmth = self.stock('warmth')
    if warmth > 0:
      self.ledger.withdraw(HOLDER, 'warmth', warmth, 'the sky warms')
      self.temp_c += self.warmth_c * warmth
    # The sky radiates part of its warming away each generation.
    self.temp_c -= self.cooling * (self.temp_c - self.start_temp_c)
    self.warmed = self.temp_c - before_c
    self.generation += 1
    self.shortfall = {}
    worst = 0.0
    for resource, per in sorted(self.needs.items()):
      need = per * self.population
      got = self.ledger.withdraw(
          HOLDER, resource, need, 'colonists', partial=True
      )
      if got < need - 1e-9:
        self.shortfall[resource] = 1.0 - got / need
        worst = max(worst, self.shortfall[resource])
    before = self.population
    if worst > 0:
      self.population *= 1.0 - self.deaths * worst
      lost = before - self.population
      short = ' and '.join(sorted(self.shortfall))
      events.append(
          f'{self.name} runs short of {short}: {lost:.1f} colonists die.'
      )
    else:
      self.population *= 1.0 + self.growth
    for resource, share in self.wear.items():
      held = self.stock(resource)
      if held > 0:
        self.ledger.withdraw(HOLDER, resource, held * share, 'wear')
    for resource, amount in made.items():
      self.made[resource] = self.made.get(resource, 0.0) + amount
    self.last_made = dict(made)
    self.ledger.advance()
    return events

  def terraformed(self) -> bool:
    return self.temp_c >= self.target_temp_c

  def payload(self, tick: int) -> dict[str, Any]:
    stocks = ('food', 'air', 'water', 'power', 'panels')
    return {
        'name': self.name,
        'tick': tick,
        'colonists': round(self.population, 2),
        'temp_c': round(self.temp_c, 2),
        'target_temp_c': self.target_temp_c,
        'stocks': {r: round(self.stock(r), 2) for r in stocks},
        'labor': {k: round(v, 3) for k, v in sorted(self.allocation.items())},
        'designs': [self.design_labels.get(d, d) for d in self.designs],
        'shortfall': {k: round(v, 3) for k, v in self.shortfall.items()},
        'solar_yield': round(self.solar_yield(), 3),
        'storm': self.storm,
        'short_of_panels': self.panel_limited,
        # What the colony made last generation, how far its sky warmed and
        # how many generations its goal has left.
        'made': {k: round(v, 3) for k, v in sorted(self.last_made.items())},
        'warmed': round(self.warmed, 3),
        'left': max(0, self.deadline - self.generation),
    }

  def sheet(self) -> str:
    """The colonists' own sheet, for their prompt."""
    stocks = ', '.join(
        f'{r} {self.stock(r):.1f}'
        for r in ('food', 'air', 'water', 'power', 'panels')
    )
    labor = ', '.join(
        f'{r.label} {self.allocation[r.key]:.0%}'
        for r in self.known()
        if self.allocation.get(r.key, 0.0) > 0
    )
    needs = ', '.join(
        f'{r} {per * self.population:.1f}'
        for r, per in sorted(self.needs.items())
    )
    designs = ', '.join(self.design_labels.get(d, d) for d in self.designs)
    text = (
        f'{self.population:.1f} colonists. Stores: {stocks}. Each generation'
        f' they need {needs}. Labor: {labor}. Designs: {designs}. The sky'
        f' reads {self.temp_c:.1f} C; the world is habitable at'
        f' {self.target_temp_c:g} C.'
    )
    power = next(
        (
            r.outputs.get('power', 0.0)
            for r in self.recipes
            if r.key == 'solar_farm'
        ),
        0.0,
    )
    text += (
        f' The panels can make at most {self.solar_batches() * power:.1f}'
        ' power a generation'
        + (' (a dust storm cuts that now)' if self.storm else '')
        + '.'
    )
    if self.panel_limited:
      text += ' Last generation the solar farm was short of panels.'
    if self.directive:
      text += f' The latest directive from home: {self.directive}'
    left = self.deadline - self.generation
    if left > 0 and not self.terraformed():
      text += (
          f' Last generation the sky warmed {self.warmed:+.1f} C; to be'
          f' habitable in the {left} generations left it must warm'
          f' {(self.target_temp_c - self.temp_c) / left:.1f} C a generation'
          ' (and the warmer it gets, the more it radiates away).'
      )
    if self.last_made or self.generation:
      # What each need is made at against what is used: a store that falls
      # says how long it lasts (a model playing the colonists cannot tell
      # from the stores alone).
      flows = []
      for resource, per in sorted(self.needs.items()):
        made = self.last_made.get(resource, 0.0)
        need = per * self.population
        flow = f'{resource} {made:.1f} made for {need:.1f} needed'
        if made < need - 1e-9:
          lasts = self.stock(resource) / (need - made)
          flow += f' (the store lasts {lasts:.0f} more generations)'
        flows.append(flow)
      text += ' Last generation: ' + '; '.join(flows) + '.'
    if self.shortfall:
      text += (
          ' Last generation we ran short of '
          + ', '.join(
              f'{r} ({v:.0%})' for r, v in sorted(self.shortfall.items())
          )
          + '.'
      )
    return text

  def get_state(self) -> dict[str, Any]:
    return {
        'ledger': self.ledger.get_state(),
        'population': self.population,
        'temp_c': self.temp_c,
        'designs': list(self.designs),
        'allocation': dict(self.allocation),
        'shortfall': dict(self.shortfall),
        'made': dict(self.made),
        'last_made': dict(self.last_made),
        'generation': self.generation,
        'warmed': self.warmed,
        'storm': self.storm,
        'panel_limited': self.panel_limited,
        'incoming': [list(m) for m in self.incoming],
        'directive': self.directive,
        'rng': self._rng.bit_generator.state,
        'site': list(self.site) if self.site is not None else None,
        'mine_factor': self.mine_factor,
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.ledger.set_state(state['ledger'])
    self.population = float(state['population'])
    self.temp_c = float(state['temp_c'])
    self.designs = list(state['designs'])
    self.allocation = dict(state['allocation'])
    self.shortfall = dict(state.get('shortfall', {}))
    self.made = dict(state.get('made', {}))
    self.last_made = dict(state.get('last_made', {}))
    self.generation = int(state.get('generation', 0))
    self.warmed = float(state.get('warmed', 0.0))
    self.storm = bool(state.get('storm', False))
    self.panel_limited = bool(state.get('panel_limited', False))
    self.incoming = [
        (int(a), str(k), str(w)) for a, k, w in state.get('incoming', [])
    ]
    self.directive = str(state.get('directive', ''))
    site = state.get('site')
    self.site = (int(site[0]), int(site[1])) if site else None
    self.mine_factor = float(state.get('mine_factor', 1.0))
    if 'rng' in state:
      self._rng.bit_generator.state = dict(state['rng'])
