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

"""Civilization: polities, labor, food and the first recipes.

Each polity settles a home region and its neighbours, feeds its people with
grain and spends its labor on recipes: farming (yield set by the local
climate), cutting wood (yield set by the forests, which it clears), burning
charcoal, mining ore (seeded deposits that eruptions refresh), smelting iron
in bloomeries and forging tools (which raise every worker's output). Stocks
are holders on the contrib `StockFlowLedger`, recipes are its `Recipe`s, and
the kilns' and bloomeries' carbon goes to the planet's carbon cycle. How a
polity splits its labor is its decision; this module keeps the books.
"""

# pylint: disable=g-complex-comprehension,missing-function-docstring

from collections.abc import Callable, Collection, Mapping, Sequence
import dataclasses
from typing import Any

from concordia.contrib.components.game_master import stock_flow_ledger
from examples.deep_time.world import grid as grid_lib
import numpy as np

# Techs a people's sheet lists as within reach (cheapest first).
REACH = 10


# A share of a people's labor under which hardly anyone works a trade: it is
# then the link to strengthen, whatever it lacks.
FEW_HANDS = 0.02


@dataclasses.dataclass(frozen=True)
class RecipeSpec:
  """A recipe as configured: labor per batch and an optional yield driver."""

  key: str
  label: str
  labor: float
  inputs: Mapping[str, float]
  outputs: Mapping[str, float]
  yield_kind: str | None = None
  requires: str | None = None  # Tech that unlocks the recipe.


@dataclasses.dataclass(frozen=True)
class Tech:
  """A technology: its cost in research, prerequisites and effects."""

  key: str
  label: str
  tier: int
  cost: float
  requires: tuple[str, ...]
  effects: Mapping[str, Any]


@dataclasses.dataclass
class Polity:
  """One people: where they live, how many they are, how they work."""

  key: str
  name: str
  home: int
  regions: list[int]
  population: float
  allocation: dict[str, float]
  food_ratio: float = 1.0
  famine: bool = False
  firsts: list[str] = dataclasses.field(default_factory=list)
  techs: list[str] = dataclasses.field(default_factory=list)
  # Yield multipliers by kind ('farming', 'forest', 'ore'): what the land
  # of this people is good for.
  endowment: dict[str, float] = dataclasses.field(default_factory=dict)
  # Last tick's bottlenecks: recipe key -> the input it ran short of.
  shortages: dict[str, str] = dataclasses.field(default_factory=dict)
  # Last tick's pollution (soot, fossil_co2, slag) and any blight on its
  # harvests: (share of the harvest lost, ticks left).
  emissions: dict[str, float] = dataclasses.field(default_factory=dict)
  blight: tuple[float, int] = (0.0, 0)
  # The people's consent for coal (the Gazette sways it): coal mines yield
  # in proportion to it.
  consent: float = 1.0
  # Plan target -> the tick it was last adopted.
  planned: dict[str, int] = dataclasses.field(default_factory=dict)
  # Yield factors by kind from the land its settlements work, now against
  # when they were laid out (set each tick from world/settlements.py).
  local: dict[str, float] = dataclasses.field(default_factory=dict)
  # Where a person sent its woodcutters this tick (weights per region), or
  # None to fell wherever the woods are thickest.
  felling: np.ndarray | None = None


def parse_recipes(items: Sequence[Mapping[str, Any]]) -> list[RecipeSpec]:
  """Recipe specs from their YAML entries."""
  return [
      RecipeSpec(
          key=r['key'],
          label=r['label'],
          labor=float(r['labor']),
          inputs={k: float(v) for k, v in r.get('inputs', {}).items()},
          outputs={k: float(v) for k, v in r['outputs'].items()},
          yield_kind=r.get('yield'),
          requires=r.get('requires'),
      )
      for r in items
  ]


def describe_change(
    change: Mapping[str, float],
    pollutants: Mapping[str, str] | None = None,
    made: Collection[str] | None = None,
    per: str = 'a generation',
) -> str:
  """A labor move's push in words: the two biggest gains and costs.

  Args:
    change: Resource -> change (`Civilization.labor_shift`).
    pollutants: Resource -> word, for what is not a good (smoke, slag), told
      apart.
    made: The goods the move's recipe makes: only these count as its gains (not
      what other trades, given fewer hands, stop using). None: any.
    per: The period of the change.

  Returns:
    E.g. 'steel +3.2 a generation; coal −1.6, ore −0.8; coal smoke +1.6'.
  """
  pollutants = pollutants or {}
  goods = {r: v for r, v in change.items() if r not in pollutants}
  gains = sorted(
      (
          (v, r)
          for r, v in goods.items()
          if v >= 0.05 and (made is None or r in made)
      ),
      reverse=True,
  )
  costs = sorted((v, r) for r, v in goods.items() if v <= -0.05)
  text = ', '.join(f'{r} +{v:.1f}' for v, r in gains[:2])
  text = f'{text} {per}' if text else f'little more made {per}'
  if costs:
    text += '; ' + ', '.join(f'{r} −{abs(v):.1f}' for v, r in costs[:2])
  dirt = [
      f'{word} {"+" if change[r] > 0 else "−"}{abs(change[r]):.1f}'
      for r, word in pollutants.items()
      if abs(change.get(r, 0.0)) >= 0.05
  ]
  return text + ('; ' + ', '.join(dirt) if dirt else '')


class Civilization:
  """Polities, their stocks and recipes on a stock-flow ledger."""

  def __init__(
      self,
      config: Mapping[str, Any],
      polities: Sequence[Mapping[str, Any]],
      grid: grid_lib.RegionGrid,
      land: np.ndarray,
      seed: int = 0,
      ledger: stock_flow_ledger.Ledger | None = None,
      techs: Sequence[Mapping[str, Any]] = (),
  ):
    """Initializes the polities.

    Args:
      config: Contents of `data/civ.yaml`.
      polities: Scenario polities: `key`, `name`, `home` ([band, sector], or
        'auto' for the best free temperate land), `population`, `allocation`
        (recipe key -> labor share) and optionally `endowment` (yield kind ->
        multiplier), `techs` known at the start and `stocks` (resource ->
        amount) to start with.
      grid: The planet grid.
      land: Land fraction per region.
      seed: Seed of the ore deposits.
      ledger: Ledger to keep the books in (a new one by default).
      techs: Technologies (`data/techs.yaml`); none by default.
    """
    self.ledger = ledger if ledger is not None else stock_flow_ledger.Ledger()
    pop = config['population']
    self.growth = float(pop['growth'])
    self.food_per_pop = float(pop['food_per_pop'])
    self.starvation = float(pop['starvation'])
    self.famine_ratio = float(pop['famine_ratio'])
    self.capacity_per_land = float(pop['capacity_per_land'])
    self.labor_per_pop = float(pop['labor_per_pop'])
    self.tool_bonus = float(config['tools']['bonus'])
    self.wear = {
        k: float(v)
        for k, v in config.get(
            'wear', {'tools': config['tools']['wear']}
        ).items()
    }
    self.panels_per_pop = float(config.get('panels', {}).get('per_pop', 1.0))
    self.seed_reference = float(config.get('seeds', {}).get('reference', 1.0))
    fish = config.get('fisheries', {})
    self.fish_stock = float(fish.get('stock', 0.0))
    self.fish_regrowth = float(fish.get('regrowth', 0.0))
    self.fish_capacity = float(fish.get('capacity', 1.0))
    self.ph_reference = float(fish.get('ph_reference', 8.2))
    self.fish_remnant = float(fish.get('remnant', 0.0))
    self.ph_penalty = float(fish.get('ph_penalty', 0.0))
    # The seas' state, set by the planet before each step.
    self.ocean_ph = self.ph_reference
    self.sea_ice = 0.0
    self.dead_zone = 0.0  # Share of the coastal seas starved of oxygen.
    # The planet's biodiversity, set by the planet before each step.
    self.biodiversity = 0.0
    satellites = config.get('satellites', {})
    self.satellites_per_pop = float(satellites.get('per_pop', 1.0))
    self.forecast_bonus = float(satellites.get('farming_bonus', 0.0))
    self.spoilage = float(config['grain_spoilage'])
    self.co2_gt = float(config['co2_gt_per_unit'])
    self.fossil_gt = float(config['fossil_gt_per_unit'])
    self.consent_kinds = tuple(config.get('consent_yields', ()))
    self.research_rate = float(config.get('research_rate', 1.0))
    # Milestones announced the first time a people makes them.
    self.firsts = {
        str(k): str(v)
        for k, v in config.get(
            'firsts',
            {
                'iron': 'make their first iron',
                'tools': 'make their first tools',
            },
        ).items()
    }
    plans = config.get('plans', {})
    self.plan_targets = tuple(plans.get('targets', ()))
    self.plan_share = float(plans.get('share', 0.4))
    # Last tick's conditions (temperatures, fertility, forest), for plans.
    self._env: tuple[list[float], float, list[float]] | None = None
    self.farm_optimum = float(config['farming']['optimum_c'])
    self.unlock_words = {
        str(k): str(v) for k, v in (config.get('unlock_words') or {}).items()
    }
    self.pollutant_words = {
        str(k): str(v) for k, v in (config.get('pollutant_words') or {}).items()
    }
    self.farm_width = float(config['farming']['width_c'])
    forest = config['forest']
    self.forest_species = str(forest['species'])
    self.full_cover = float(forest['full_cover'])
    self.cover_per_wood = float(forest['cover_per_wood'])
    self.recipes = parse_recipes(config['recipes'])
    self._recipe_index: dict[str, RecipeSpec] = {}
    self._tech_index: dict[str, Tech] = {}
    # Tech multipliers by (the techs a people knows, kind, path).
    self._effects: dict[tuple[Any, ...], float] = {}
    self.techs = [
        Tech(
            key=t['key'],
            label=t['label'],
            tier=int(t['tier']),
            cost=float(t['cost']),
            requires=tuple(t.get('requires', ())),
            effects=dict(t.get('effects', {})),
        )
        for t in techs
    ]
    self._grid = grid
    self._land = np.asarray(land, dtype=float)
    rng = np.random.default_rng(seed + 7919)
    # Seeded deposits per region (ore first, from the original stream, so
    # adding deposits does not move the ore).
    self.deposits: dict[str, np.ndarray] = {}
    for index, (kind, spec) in enumerate(config['deposits'].items()):
      stream = (
          rng
          if index == 0
          else np.random.default_rng(seed + 7919 * (index + 1))
      )
      self.deposits[kind] = (
          float(spec['mean_richness'])
          * stream.lognormal(0.0, float(spec['spread']), grid.size)
          * (self._land > 0)
      )
    self.ore_gain = float(config['deposits']['ore']['eruption_gain'])
    energy = config.get('energy', {})
    self.energy_per_pop = float(energy.get('per_pop', 0.0))
    self.energy_bonus = float(energy.get('bonus', 0.0))
    radius = int(config['settle_radius'])
    self.polities: dict[str, Polity] = {}
    claimed: set[int] = set()
    for p in polities:
      # Keep the next home far enough away that territories cannot overlap.
      taken = claimed | {n for r in claimed for n in grid.neighbours(r)}
      home = (
          self._place(taken) if p['home'] == 'auto' else grid.region(*p['home'])
      )
      regions = [r for r in self._settle(home, radius) if r not in claimed]
      claimed |= set(regions)
      self.polities[p['key']] = Polity(
          key=p['key'],
          name=p['name'],
          home=home,
          regions=regions,
          population=float(p['population']),
          allocation=self._normalised(p['allocation']),
          endowment={k: float(v) for k, v in p.get('endowment', {}).items()},
          techs=list(p.get('techs', ())),
      )
      for resource, amount in p.get('stocks', {}).items():
        self.ledger.deposit(p['key'], resource, float(amount), 'start')
    self.start_population = {k: p.population for k, p in self.polities.items()}
    # Share of each people's regions that touch the sea (for fishing).
    self.coastal = {
        k: (
            sum(
                any(self._land[n] < 0.5 for n in grid.neighbours(r))
                for r in p.regions
            )
            / max(1, len(p.regions))
        )
        for k, p in self.polities.items()
    }
    # Everything ever produced, by resource (the ledger's statistics are
    # per tick), and what was used up each tick.
    self.totals: dict[str, float] = {}
    self.consumption: list[dict[str, float]] = []

  def _place(self, taken: set[int]) -> int:
    """The landiest temperate region not next to another polity."""
    bands = range(1, self._grid.bands - 1)
    candidates = [
        r
        for r in range(self._grid.size)
        if self._grid.band_of(r) in bands
        and r not in taken
        and self._land[r] > 0
    ]
    if not candidates:
      candidates = [r for r in range(self._grid.size) if self._land[r] > 0]
    return max(candidates, key=lambda r: (self._land[r], -r))

  def _settle(self, home: int, radius: int) -> list[int]:
    regions = {home}
    frontier = {home}
    for _ in range(radius):
      frontier = {
          n for r in frontier for n in self._grid.neighbours(r)
      } - regions
      regions |= frontier
    return sorted(r for r in regions if self._land[r] > 0)

  def _normalised(self, allocation: Mapping[str, float]) -> dict[str, float]:
    known = {r.key for r in self.recipes}
    shares = {
        k: max(0.0, float(v)) for k, v in allocation.items() if k in known
    }
    total = sum(shares.values())
    if total <= 0:
      return {self.recipes[0].key: 1.0}
    return {k: v / total for k, v in shares.items() if v > 0}

  def effect(self, polity: Polity, kind: str, *path: str) -> float:
    """The product of every known tech's multiplier at `kind` / `path`."""
    key = (tuple(polity.techs), kind, path)
    if key in self._effects:
      return self._effects[key]
    product = 1.0
    for tech in self.techs:
      if tech.key not in polity.techs:
        continue
      value: Any = tech.effects.get(kind)
      for part in path:
        value = value.get(part) if isinstance(value, Mapping) else None
      if isinstance(value, (int, float)):
        product *= float(value)
    self._effects[key] = product
    return product

  def aim(self, polity: str, resource: str) -> str:
    """How a people can make a resource its goal asks for, in words.

    The recipes that make it, whether the people knows them or what it must
    learn first (and the research it has), and what each recipe uses: so a
    people (or the model playing it) sees the road to its goal.

    Args:
      polity: The people.
      resource: The good its goal asks for (e.g. 'energy').

    Returns:
      E.g. 'Aim: energy, made by burn coal in steam engines (learn steam
      power first: 30 research; we have 21.7), which uses coal 2.'
    """
    people = self.polities[polity]
    research = self.stock(polity, 'research')
    ways = []
    for spec in self.recipes:
      if resource not in spec.outputs:
        continue
      uses = ', '.join(
          f'{a:g} {r.replace("_", " ")}' for r, a in spec.inputs.items()
      )
      if spec.requires is None or spec.requires in people.techs:
        how = 'we know it'
      else:
        tech = self.tech(spec.requires)
        missing = [r for r in tech.requires if r not in people.techs]
        how = f'learn {tech.label} first: {tech.cost:g} research'
        if missing:
          labels = ', '.join(self.tech(r).label for r in missing)
          how += f', after {labels}'
        how += f'; we have {research:.1f}'
      ways.append(
          f'{spec.label} ({how})' + (f', which uses {uses}' if uses else '')
      )
    name = resource.replace('_', ' ')
    if not ways:
      return f'Aim: {name}, which no recipe makes.'
    return f'Aim: {name}, made by ' + '; or '.join(ways) + '.'

  def held_back(
      self, polity: str, resource: str
  ) -> tuple[list[str], str | None]:
    """What held back a people's making of a resource last generation.

    From the known recipe that makes the resource (the one with the most
    hands), follows the input it stood idle for want of to the known recipe
    that makes that input, and so on, until a recipe that was not short of
    anything: its hands are the link to strengthen.

    Args:
      polity: The people.
      resource: The good its goal asks for (e.g. 'iron').

    Returns:
      The goods found short, in order, and the recipe at the end of the
      chain: ([], None) when nothing stood idle; ([], recipe) when hardly
      any hands (under `FEW_HANDS`) work the recipe that makes the good;
      (goods, None) when the
      last good short is one the people knows no recipe for (it must buy
      it).
    """
    people = self.polities[polity]

    def maker(good: str) -> str | None:
      known = [
          spec
          for spec in self.recipes
          if good in spec.outputs
          and (spec.requires is None or spec.requires in people.techs)
      ]
      if not known:
        return None
      return max(known, key=lambda s: people.allocation.get(s.key, 0.0)).key

    few = lambda key: people.allocation.get(key, 0.0) < FEW_HANDS

    def lacking(key: str) -> str | None:
      """The input a recipe lacks for even one batch, if any."""
      spec = self.recipe(key)
      recipe = stock_flow_ledger.Recipe(key, spec.inputs, spec.outputs)
      if self.ledger.max_batches(polity, recipe) >= 1:
        return None
      return self.ledger.limiting_input(polity, recipe)

    recipe = maker(resource)
    if recipe is None:
      return [], None
    if recipe not in people.shortages and not few(recipe):
      return [], None  # Its makers had hands and all they needed.
    short: list[str] = []
    while recipe is not None:
      if recipe in people.shortages:
        good = people.shortages[recipe]
      elif few(recipe):  # Few hands: the link, unless it lacks an input.
        good = lacking(recipe)
        if good is None:
          break
      else:
        break  # Hands and inputs: more hands here would free the rest.
      if good in short:  # A loop: stop at the first good seen again.
        break
      short.append(good)
      recipe = maker(good)
    return short, recipe

  def goal_chain(self, polity: str, resource: str) -> list[str]:
    """The known recipes that make a good and, in turn, its inputs.

    For each good, the known recipe with the most hands that makes it; the
    good's own maker first.

    Args:
      polity: The people.
      resource: The good its goal asks for.

    Returns:
      Recipe keys, the good's maker first.
    """
    people = self.polities[polity]
    chain: list[str] = []
    goods = [resource]
    while goods:
      good = goods.pop(0)
      known = [
          spec
          for spec in self.recipes
          if good in spec.outputs
          and (spec.requires is None or spec.requires in people.techs)
      ]
      if not known:
        continue
      best = max(known, key=lambda s: people.allocation.get(s.key, 0.0))
      if best.key not in chain:
        chain.append(best.key)
        goods.extend(best.inputs)
    return chain

  def best_tech(
      self, polity: str, resource: str, affordable: bool = True
  ) -> Tech | None:
    """The tech that most helps a people make the good its goal asks for.

    Scored on the good's chain (`goal_chain`): a recipe's output raised by
    a factor f counts f - 1, an input cut to g counts 1 - g, and a recipe
    unlocked that makes the good or a chain input counts 0.5.

    Args:
      polity: The people.
      resource: The good its goal asks for.
      affordable: Only techs its research can pay for now.

    Returns:
      The best such tech (the cheaper of equals), or None if none helps.
    """
    chain = self.goal_chain(polity, resource)
    needed = {resource} | {
        good for key in chain for good in self.recipe(key).inputs
    }
    research = self.stock(polity, 'research')
    best, score = None, 0.0
    for tech in sorted(self.available_techs(polity), key=lambda t: t.cost):
      if affordable and tech.cost > research + 1e-9:
        continue
      effects = tech.effects
      gain = sum(
          float(f) - 1.0
          for key, f in (effects.get('outputs') or {}).items()
          if key in chain
      )
      gain += sum(
          max(0.0, 1.0 - float(g))
          for key, cuts in (effects.get('inputs') or {}).items()
          if key in chain
          for g in cuts.values()
      )
      if any(
          spec.requires == tech.key and needed & set(spec.outputs)
          for spec in self.recipes
      ):
        gain += 0.5
      if gain > score + 1e-9:
        best, score = tech, gain
    return best

  def describe_tech(self, key: str) -> str:
    """What learning a tech does, in words, from its effects (the data's).

    Args:
      key: The tech.

    Returns:
      E.g. 'farm the land +25%; unlocks steam engines'.
    """
    effects = self.tech(key).effects
    labels = {r.key: r.label for r in self.recipes}
    parts = [
        f'unlocks {r.label}' for r in self.recipes if r.requires == key
    ] + [
        self.unlock_words[tag]
        for tag in effects.get('unlocks') or ()
        if tag in self.unlock_words
    ]
    for recipe, factor in (effects.get('outputs') or {}).items():
      parts.append(f'{labels.get(recipe, recipe)} {float(factor) - 1:+.0%}')
    for recipe, inputs in (effects.get('inputs') or {}).items():
      for resource, factor in inputs.items():
        more = 'more' if float(factor) > 1 else 'less'
        parts.append(
            f'{labels.get(recipe, recipe)} uses {abs(float(factor) - 1):.0%}'
            f' {more} {resource}'
        )
    words = {
        'clearing': 'forest cleared per wood',
        'spoilage': 'grain spoiling in store',
        'research': 'research',
    }
    for kind, what in words.items():
      if effects.get(kind) is not None:
        parts.append(f'{what} {float(effects[kind]) - 1:+.0%}')
    for resource, factor in (effects.get('emissions') or {}).items():
      word = self.pollutant_words.get(resource, resource)
      parts.append(f'{word} {float(factor) - 1:+.0%}')
    return '; '.join(parts[:3]) or 'new knowledge'

  def tech(self, key: str) -> Tech:
    if len(self._tech_index) != len(self.techs):
      self._tech_index = {t.key: t for t in self.techs}
    return self._tech_index[key]

  def available_techs(self, polity: str) -> list[Tech]:
    """Techs the polity does not know whose prerequisites it does."""
    known = set(self.polities[polity].techs)
    return [
        t
        for t in self.techs
        if t.key not in known and all(r in known for r in t.requires)
    ]

  def learn(self, polity: str, tech: str) -> str:
    """Spends research on a tech; returns what happened."""
    people = self.polities[polity]
    spec = self.tech(tech)
    if spec not in self.available_techs(polity):
      return f'{people.name} cannot learn {spec.label} yet.'
    research = self.ledger.stock(polity, 'research')
    if research < spec.cost - 1e-9:
      return (
          f'{people.name} have {research:.1f} of the {spec.cost:g} research'
          f' that {spec.label} needs.'
      )
    self.ledger.withdraw(polity, 'research', spec.cost, f'learn {tech}')
    people.techs.append(tech)
    return f'{people.name} learn {spec.label}.'

  def unlocked(self, polity: str, tag: str) -> bool:
    """Whether a tech the polity knows unlocks `tag`."""
    return any(
        tag in self.tech(t).effects.get('unlocks', ())
        for t in self.polities[polity].techs
    )

  def recipe(self, key: str) -> RecipeSpec:
    if len(self._recipe_index) != len(self.recipes):
      self._recipe_index = {r.key: r for r in self.recipes}
    return self._recipe_index[key]

  def set_allocation(self, polity: str, allocation: Mapping[str, float]):
    self.polities[polity].allocation = self._normalised(allocation)

  def shift_labor(self, polity: str, recipe: str, share: float) -> None:
    """Moves `share` of a polity's labor onto one recipe."""
    current = dict(self.polities[polity].allocation)
    share = min(1.0, max(0.0, share))
    scaled = {k: v * (1.0 - share) for k, v in current.items()}
    scaled[recipe] = scaled.get(recipe, 0.0) + share
    self.set_allocation(polity, scaled)

  def stock(self, polity: str, resource: str) -> float:
    return self.ledger.stock(polity, resource)

  def labor_shift(
      self, polity: str, recipe: str, share: float
  ) -> dict[str, float]:
    """What moving `share` of a people's labor to `recipe` makes and costs.

    Worked out as a generation's work is (`step`): the people's labor with
    its tools and energy, each recipe as this people runs it on its land,
    in the latest weather. Inputs permitting: a recipe short of an input
    makes less.

    Args:
      polity: The people.
      recipe: The recipe that gets the labor.
      share: The share of all labor moved (taken from every recipe pro rata).

    Returns:
      Resource -> change a generation (+ made, − used or no longer made).
    """
    return self.labor_shifts(polity, [recipe], share)[recipe]

  def labor_shifts(
      self, polity: str, recipes: Sequence[str], share: float
  ) -> dict[str, dict[str, float]]:
    """`labor_shift` for several recipes, each recipe's run worked out once.

    Args:
      polity: The people.
      recipes: The recipes that might get the labor, one move each.
      share: The share of all labor each move shifts.

    Returns:
      Recipe -> resource -> change a generation.
    """
    people = self.polities[polity]
    size = len(self._land)
    temps, fertility, forest = self._env or (
        [self.farm_optimum] * size,
        1.0,
        [0.0] * size,
    )
    temps, forest = np.asarray(temps), np.asarray(forest)
    productivity = 1.0 + self.tool_bonus * min(
        1.0, self.stock(polity, 'tools') / max(people.population, 1e-9)
    )
    demand = self.energy_per_pop * people.population
    if demand > 0:
      productivity *= 1.0 + self.energy_bonus * min(
          1.0, self.stock(polity, 'energy') / demand
      )
    labor = people.population * self.labor_per_pop * productivity
    share = min(1.0, max(0.0, share))
    before = dict(people.allocation)
    specs = self.recipes_for(polity)
    runs: dict[str, stock_flow_ledger.Recipe] = {}
    found = {}
    for recipe in recipes:
      after = {k: v * (1.0 - share) for k, v in before.items()}
      after[recipe] = after.get(recipe, 0.0) + share
      after = self._normalised(after)
      change: dict[str, float] = {}
      for spec in specs:
        moved = after.get(spec.key, 0.0) - before.get(spec.key, 0.0)
        if abs(moved) < 1e-9:
          continue
        if spec.key not in runs:
          runs[spec.key] = self._effective(
              people, spec, temps, fertility, forest, spec.label
          )
        run = runs[spec.key]
        batches = labor * moved / spec.labor
        for resource, amount in run.outputs.items():
          change[resource] = change.get(resource, 0.0) + batches * amount
        for resource, amount in run.inputs.items():
          change[resource] = change.get(resource, 0.0) - batches * amount
      found[recipe] = change
    return found

  def made(self, resource: str) -> float:
    """Total ever produced of a resource, by every polity."""
    return self.totals.get(resource, 0.0)

  def add_ore(self, magnitude: int) -> None:
    """An eruption exposes fresh ore on every land region."""
    self.deposits['ore'] = self.deposits['ore'] + (
        self.ore_gain * magnitude * (self._land > 0)
    )

  def _mean(self, values: np.ndarray, regions: Sequence[int]) -> float:
    weights = self._land[regions]
    if weights.sum() <= 0:
      return 0.0
    return float(np.dot(weights, values[regions]) / weights.sum())

  def _yield(
      self,
      kind: str | None,
      polity: Polity,
      temps_c: np.ndarray,
      fertility: float,
      forest: np.ndarray,
  ) -> float:
    consent = polity.consent if kind in self.consent_kinds else 1.0
    return (
        consent
        * polity.endowment.get(kind or '', 1.0)
        * polity.local.get(kind or '', 1.0)
        * self._base_yield(kind, polity, temps_c, fertility, forest)
    )

  def _base_yield(
      self,
      kind: str | None,
      polity: Polity,
      temps_c: np.ndarray,
      fertility: float,
      forest: np.ndarray,
  ) -> float:
    if kind == 'farming':
      suit = np.maximum(
          0.0, 1.0 - ((temps_c - self.farm_optimum) / self.farm_width) ** 2
      )
      loss = polity.blight[0] if polity.blight[1] > 0 else 0.0
      return (
          self._mean(suit, polity.regions)
          * fertility
          * (1.0 - loss)
          * (1.0 + self.forecast_bonus * self.coverage(polity))
      )
    if kind == 'forest':
      cover = self._mean(forest, polity.regions)
      return min(1.0, cover / self.full_cover)
    if kind in self.deposits:
      return self._mean(self.deposits[kind], polity.regions)
    if kind == 'fish':
      acid = max(
          0.0,
          1.0 - self.ph_penalty * max(0.0, self.ph_reference - self.ocean_ph),
      )
      return (
          self.fish_stock
          * self.coastal.get(polity.key, 0.0)
          * (1.0 - self.sea_ice)
          * (1.0 - self.dead_zone)
          * acid
      )
    if kind == 'biodiversity':
      return min(1.0, self.biodiversity / max(1e-9, self.seed_reference))
    if kind == 'panels':
      panels = self.ledger.stock(polity.key, 'panels')
      return min(
          1.0, panels / max(1e-9, self.panels_per_pop * polity.population)
      )
    return 1.0

  @property
  def ore(self) -> np.ndarray:
    return self.deposits['ore']

  def unlocked_recipe(self, polity: str, recipe: str) -> bool:
    spec = self.recipe(recipe)
    return spec.requires is None or spec.requires in self.polities[polity].techs

  def recipes_for(self, polity: str) -> list[RecipeSpec]:
    """The recipes a polity knows how to run."""
    return [r for r in self.recipes if self.unlocked_recipe(polity, r.key)]

  def known_goods(self, polity: str) -> set[str]:
    """Goods a polity makes or uses with the recipes it knows (and food)."""
    goods = {'grain'}
    for recipe in self.recipes_for(polity):
      goods |= set(recipe.inputs) | set(recipe.outputs)
    return goods

  def step(
      self,
      temps_c: np.ndarray,
      fertility: float,
      forest: np.ndarray,
      *,
      clear_forest: Callable[[np.ndarray], Any],
      emit: Callable[[float, str], Any],
      soot: Callable[[float], Any] = lambda amount: None,
      fossil: Callable[[float, str], Any] = lambda amount, reason: None,
      runoff: Callable[[float], Any] = lambda amount: None,
  ) -> list[str]:
    """Runs one tick of work, food and population for every polity.

    Args:
      temps_c: Temperature per region.
      fertility: Soil fertility multiplier (1 normally; volcanic ash raises it).
      forest: Forest cover per region (share of the region).
      clear_forest: Removes forest cover per region (woodcutting).
      emit: Sends carbon (Gt C) to the air, with a reason.
      soot: Sends soot (from coal) to the air, darkening the planet.
      fossil: Sends fossil carbon (Gt C, from coal) to the air.
      runoff: Sends fertiliser runoff to the coastal seas.

    Returns:
      Noteworthy event texts.
    """
    self._env = (
        [float(t) for t in temps_c],
        float(fertility),
        [float(f) for f in forest],
    )
    catch = 0.0  # Grain fished from the shared seas this tick.
    events = []
    ledger = self.ledger
    for key in sorted(self.polities):
      polity = self.polities[key]
      tools = ledger.stock(key, 'tools')
      productivity = 1.0 + self.tool_bonus * min(
          1.0, tools / max(polity.population, 1e-9)
      )
      demand = self.energy_per_pop * polity.population
      if demand > 0 and ledger.stock(key, 'energy') > 0:
        used = ledger.withdraw(key, 'energy', demand, 'machines', partial=True)
        productivity *= 1.0 + self.energy_bonus * used / demand
      labor = polity.population * self.labor_per_pop * productivity
      polity.shortages = {}
      polity.emissions = {}
      for spec in self.recipes:
        if spec.requires is not None and spec.requires not in polity.techs:
          continue
        share = polity.allocation.get(spec.key, 0.0)
        if share <= 0:
          continue
        recipe = self._effective(
            polity,
            spec,
            temps_c,
            fertility,
            forest,
            f'{polity.name}: {spec.label}',
        )
        wanted = labor * share / spec.labor
        short = ledger.limiting_input(key, recipe)
        batches = ledger.apply_recipe(key, recipe, wanted, partial=True)
        if spec.yield_kind == 'fish':
          caught = batches * recipe.outputs.get('grain', 0.0)
          catch += caught
          polity.emissions['catch'] = (
              polity.emissions.get('catch', 0.0) + caught
          )
        if 'coal' in recipe.inputs:
          polity.emissions['coal'] = polity.emissions.get('coal', 0.0) + (
              batches * recipe.inputs['coal']
          )
        if 'slag' in recipe.outputs:
          polity.emissions['slag'] = polity.emissions.get('slag', 0.0) + (
              batches * recipe.outputs['slag']
          )
        if short is not None and batches < 0.8 * wanted:
          polity.shortages[spec.key] = short
        if spec.yield_kind == 'forest' and batches > 0:
          wood = batches * recipe.outputs.get('wood', 0.0)
          wood *= self.effect(polity, 'clearing')
          self._clear(polity, forest, wood, clear_forest)
        for resource in recipe.outputs:
          first = f'first {resource}'
          if (
              batches > 0
              and resource in self.firsts
              and (first not in polity.firsts)
          ):
            polity.firsts.append(first)
            events.append(f'{polity.name} {self.firsts[resource]}.')
      coal_smoke = ledger.stock(key, 'fossil_co2')
      if coal_smoke > 0:
        ledger.withdraw(key, 'fossil_co2', coal_smoke, 'smoke', partial=True)
        polity.emissions['fossil_co2'] = coal_smoke
        fossil(coal_smoke * self.fossil_gt, f'{polity.name} steam engines')
      smoke = ledger.stock(key, 'soot')
      if smoke > 0:
        ledger.withdraw(key, 'soot', smoke, 'chimneys', partial=True)
        polity.emissions['soot'] = smoke
        soot(smoke)
      washed = ledger.stock(key, 'runoff')
      if washed > 0:
        ledger.withdraw(key, 'runoff', washed, 'rivers', partial=True)
        polity.emissions['runoff'] = washed
        runoff(washed)
      co2 = ledger.stock(key, 'co2')
      if co2 > 0:
        ledger.withdraw(key, 'co2', co2, 'smoke', partial=True)
        emit(co2 * self.co2_gt, f'{polity.name} kilns and bloomeries')
      events.extend(self._feed(polity))
      if polity.blight[1] > 0:
        polity.blight = (polity.blight[0], polity.blight[1] - 1)
      ledger.withdraw(
          key,
          'grain',
          self.spoilage
          * self.effect(polity, 'spoilage')
          * ledger.stock(key, 'grain'),
          'spoilage',
          partial=True,
      )
      for good, share in self.wear.items():
        ledger.withdraw(
            key, good, share * ledger.stock(key, good), 'wear', partial=True
        )
    consumed = {}
    for resource, stats in ledger.production().items():
      self.totals[resource] = self.totals.get(resource, 0.0) + stats['produced']
      consumed[resource] = stats['consumed']
    self.consumption.append(consumed)
    ledger.advance()
    # The shared seas: logistic regrowth (none in the dead zones), less this
    # tick's catch.
    stock = self.fish_stock
    stock += self.fish_regrowth * stock * (1.0 - stock) * (1.0 - self.dead_zone)
    stock -= catch / max(1e-9, self.fish_capacity)
    self.fish_stock = min(1.0, max(self.fish_remnant, stock))
    return events

  def _clear(
      self,
      polity: Polity,
      forest: np.ndarray,
      wood: float,
      clear_forest: Callable[[np.ndarray], Any],
  ) -> None:
    regions = polity.regions
    weights = forest[regions] * self._land[regions]
    if polity.felling is not None:  # Aimed: mostly the chosen woods.
      aimed = weights * polity.felling[regions]
      if aimed.sum() > 0:
        weights = 0.15 * weights / weights.sum() + 0.85 * aimed / aimed.sum()
      polity.felling = None
    if weights.sum() <= 0:
      return
    cut = np.zeros_like(forest)
    cut[regions] = wood * self.cover_per_wood * weights / weights.sum()
    clear_forest(cut)

  def _feed(self, polity: Polity) -> list[str]:
    need = polity.population * self.food_per_pop
    eaten = self.ledger.withdraw(
        polity.key, 'grain', need, 'eaten', partial=True
    )
    polity.food_ratio = eaten / need if need > 0 else 1.0
    capacity = self.capacity_per_land * float(self._land[polity.regions].sum())
    events = []
    if polity.food_ratio >= 1.0 - 1e-9:
      crowding = max(0.0, 1.0 - polity.population / max(capacity, 1e-9))
      polity.population *= 1.0 + self.growth * crowding
    else:
      polity.population *= 1.0 - self.starvation * (1.0 - polity.food_ratio)
    hungry = polity.food_ratio < self.famine_ratio
    if hungry and not polity.famine:
      events.append(f'Famine among the {polity.name}.')
    polity.famine = hungry
    return events

  def sheet(self, polity: str) -> str:
    """One polity's own storehouse, people and labor, for its prompt."""
    p = self.polities[polity]
    shown = ['grain', 'wood', 'charcoal', 'ore', 'iron', 'tools']
    if any(self.tech(t).tier >= 3 for t in p.techs):
      shown += ['coal', 'steel', 'energy']
    if any(self.tech(t).tier >= 4 for t in p.techs):
      shown += ['glass', 'wire', 'fertiliser', 'panels']
    if self.unlocked(polity, 'broadsheets'):
      shown += ['paper']
    if self.unlocked(polity, 'orbit'):
      shown += ['fuel', 'rockets', 'satellites']
    if 'arks' in p.techs:
      shown += ['seeds']
    stocks = ', '.join(f'{r} {self.ledger.stock(polity, r):.1f}' for r in shown)
    labor = ', '.join(
        f'{self.recipe(k).label} {v:.0%}'
        for k, v in sorted(p.allocation.items(), key=lambda kv: -kv[1])
    )
    known = ', '.join(self.tech(t).label for t in p.techs) or 'none yet'
    reach = ', '.join(
        f'{t.label} ({t.cost:g})'
        for t in sorted(self.available_techs(polity), key=lambda t: t.cost)[
            :REACH
        ]
    )
    words = {'farming': 'farmland', 'forest': 'forest'}
    rich = [words.get(k, k) for k, v in sorted(p.endowment.items()) if v >= 1.5]
    poor = [words.get(k, k) for k, v in sorted(p.endowment.items()) if v <= 0.6]
    land = ''
    if rich or poor:
      parts = []
      if rich:
        parts.append(f'rich in {" and ".join(rich)}')
      if poor:
        parts.append(f'poor in {" and ".join(poor)}')
      land = f' Our land is {", ".join(parts)}.'
    idle = '; '.join(
        f'{self.recipe(k).label} stood partly idle for want of {r}'
        for k, r in sorted(p.shortages.items())
    )
    if idle:
      land += f' Last generation {idle}.'
    burned = p.emissions.get('coal', 0.0)
    if burned > 0:
      land += f' Last generation we burned {burned:.1f} coal.'
    caught = p.emissions.get('catch', 0.0)
    if caught > 0:
      land += f' Last generation we caught {caught:.1f} fish.'
    if 'fishing_boats' in p.techs:
      land += (
          f" The seas' fish stocks stand at {self.fish_stock:.0%} of their"
          f' plenty; {self.coastal.get(polity, 0.0):.0%} of our land is'
          ' coast.'
      )
      if self.dead_zone >= 0.01:
        land += (
            f' Dead zones choke {self.dead_zone:.0%} of the coastal seas,'
            ' and fish neither live nor breed there.'
        )
    washed = p.emissions.get('runoff', 0.0)
    if washed > 0:
      land += (
          f' Last generation {washed:.2f} of our fertiliser washed down the'
          ' rivers to the sea.'
      )
    return (
        f'{p.population:.1f} people, fed {min(1.0, p.food_ratio):.0%} last'
        f' generation. Storehouse: {stocks}. Labor: {labor}. Research'
        f' {self.ledger.stock(polity, "research"):.1f}; we know: {known}.'
        + (f' Within reach: {reach}.' if reach else '')
        + land
    )

  def _effective(
      self,
      polity: Polity,
      spec: RecipeSpec,
      temps_c: np.ndarray,
      fertility: float,
      forest: np.ndarray,
      name: str,
  ) -> stock_flow_ledger.Recipe:
    """A recipe as this people runs it: its land, techs and emissions."""
    factor = self._yield(
        spec.yield_kind, polity, temps_c, fertility, forest
    ) * self.effect(polity, 'outputs', spec.key)
    if spec.key == 'study':
      factor *= self.effect(polity, 'research')
    return stock_flow_ledger.Recipe(
        name=name,
        inputs={
            r: a * self.effect(polity, 'inputs', spec.key, r)
            for r, a in spec.inputs.items()
        },
        outputs={
            r: (
                a
                * factor
                * self.effect(polity, 'emissions', r)
                * (self.research_rate if r == 'research' else 1.0)
            )
            for r, a in spec.outputs.items()
        },
    )

  def coverage(self, polity: Polity) -> float:
    """Share of full weather-satellite coverage this people has (0 to 1)."""
    satellites = self.ledger.stock(polity.key, 'satellites')
    return min(
        1.0, satellites / max(1e-9, self.satellites_per_pop * polity.population)
    )

  def in_orbit(self) -> float:
    """Satellites in orbit, every people's together."""
    return sum(self.ledger.stock(k, 'satellites') for k in self.polities)

  def plannable(self, polity: str) -> list[RecipeSpec]:
    """The plan targets this people knows and may plan for."""
    if not self.unlocked(polity, 'plans'):
      return []
    return [
        self.recipe(k)
        for k in self.plan_targets
        if self.unlocked_recipe(polity, k)
    ]

  def plan(
      self, polity: str, target: str
  ) -> tuple[dict[str, float], dict[str, float]]:
    """A balanced chain for `target`, as this people's land runs it.

    The contrib ratio planner (`stock_flow_ledger.balance_chain`) works back
    from the target through the producer of every input; batches become
    labor shares that sum to `plan_share`. Energy comes from solar panels
    if the people has any, else from coal.

    Args:
      polity: The people's key.
      target: Recipe key to plan for.

    Returns:
      (recipe key -> labor share, resource -> amount a tick to buy in).
    """
    people = self.polities[polity]
    size = len(self._land)
    temps, fertility, forest = self._env or (
        [self.farm_optimum] * size,
        1.0,
        [0.0] * size,
    )
    known = {r.key: r for r in self.recipes_for(polity)}
    recipes = [
        self._effective(
            people, spec, np.array(temps), fertility, np.array(forest), key
        )
        for key, spec in known.items()
    ]
    solar = next((r for r in recipes if r.name == 'solar_power'), None)
    producers = (
        {'energy': 'solar_power'}
        if solar is not None and solar.outputs.get('energy', 0.0) > 0
        else {}
    )
    chain = stock_flow_ledger.balance_chain(recipes, target, 1.0, producers)
    weights = {
        k: rate * known[k].labor
        for k, rate in chain.batches.items()
        if rate > 0
    }
    total = sum(weights.values())
    shares = {k: self.plan_share * w / total for k, w in weights.items()}
    return shares, dict(chain.external)

  def adopt_plan(self, polity: str, target: str) -> dict[str, float]:
    """Puts `plan_share` of the labor into `target`'s balanced chain."""
    shares, _ = self.plan(polity, target)
    current = self.polities[polity].allocation
    scaled = {k: v * (1.0 - self.plan_share) for k, v in current.items()}
    for key, share in shares.items():
      scaled[key] = scaled.get(key, 0.0) + share
    self.set_allocation(polity, scaled)
    return shares

  def strike(
      self, polity: str, harvest_loss: float, ticks: int, deaths: float
  ) -> None:
    """The land strikes a people: blighted harvests, or deaths."""
    people = self.polities[polity]
    if ticks > 0:
      people.blight = (harvest_loss, ticks)
    people.population *= 1.0 - deaths

  def collapsed(self, fraction: float) -> list[str]:
    """Polities whose population fell below `fraction` of the start."""
    return [
        k
        for k, p in sorted(self.polities.items())
        if p.population < fraction * self.start_population[k]
    ]

  def population(self) -> float:
    return sum(p.population for p in self.polities.values())

  def incoming(self, polity: str) -> dict[str, float]:
    """Goods on their way to a polity (bought, not yet delivered)."""
    totals: dict[str, float] = {}
    for shipment in self.ledger.shipments():
      if shipment['target'] == polity:
        resource = shipment['resource']
        totals[resource] = totals.get(resource, 0.0) + shipment['amount']
    return totals

  def payload(self) -> dict[str, Any]:
    resources = (
        'grain',
        'wood',
        'charcoal',
        'ore',
        'iron',
        'tools',
        'slag',
        'coal',
        'steel',
        'energy',
    )
    return {
        'polities': {
            key: {
                'name': p.name,
                'population': round(p.population, 2),
                'food_ratio': round(p.food_ratio, 3),
                'allocation': {k: round(v, 3) for k, v in p.allocation.items()},
                'stocks': {
                    r: round(self.ledger.stock(key, r), 2) for r in resources
                },
                # On hand plus on the road: what a careful buyer counts.
                'position': {
                    r: round(
                        self.ledger.stock(key, r)
                        + self.incoming(key).get(r, 0.0),
                        2,
                    )
                    for r in resources
                },
                'home': self._grid.region_name(p.home),
                'techs': list(p.techs),
            }
            for key, p in sorted(self.polities.items())
        },
        'made': {
            r: round(self.made(r), 2)
            for r in ('grain', 'iron', 'tools', 'energy')
        },
    }

  def get_state(self) -> dict[str, Any]:
    return {
        'ledger': self.ledger.get_state(),
        'deposits': {k: v.tolist() for k, v in self.deposits.items()},
        'totals': dict(self.totals),
        'consumption': [dict(c) for c in self.consumption],
        'env': list(self._env) if self._env else None,
        'biodiversity': self.biodiversity,
        'fish_stock': self.fish_stock,
        'polities': {
            k: {
                'population': p.population,
                'allocation': dict(p.allocation),
                'food_ratio': p.food_ratio,
                'famine': p.famine,
                'firsts': list(p.firsts),
                'techs': list(p.techs),
                'blight': list(p.blight),
                'consent': p.consent,
                'planned': dict(p.planned),
                'emissions': dict(p.emissions),
                'local': dict(p.local),  # Plans read it between ticks.
                'felling': (
                    p.felling.tolist() if p.felling is not None else None
                ),
            }
            for k, p in self.polities.items()
        },
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.ledger.set_state(state['ledger'])
    for kind, values in state['deposits'].items():
      self.deposits[kind] = np.array(values, dtype=float)
    self.totals = {k: float(v) for k, v in state.get('totals', {}).items()}
    self.consumption = [dict(c) for c in state.get('consumption', [])]
    self.biodiversity = float(state.get('biodiversity', 0.0))
    self.fish_stock = float(state.get('fish_stock', self.fish_stock))
    env = state.get('env')
    self._env = (list(env[0]), float(env[1]), list(env[2])) if env else None
    for key, saved in state['polities'].items():
      polity = self.polities[key]
      polity.population = float(saved['population'])
      polity.allocation = dict(saved['allocation'])
      polity.food_ratio = float(saved['food_ratio'])
      polity.famine = bool(saved['famine'])
      polity.firsts = list(saved['firsts'])
      polity.techs = list(saved.get('techs', []))
      loss, ticks = saved.get('blight', (0.0, 0))
      polity.blight = (float(loss), int(ticks))
      polity.consent = float(saved.get('consent', 1.0))
      polity.planned = {k: int(v) for k, v in saved.get('planned', {}).items()}
      polity.emissions = {
          k: float(v) for k, v in saved.get('emissions', {}).items()
      }
      polity.local = {k: float(v) for k, v in saved.get('local', {}).items()}
      felling = saved.get('felling')
      polity.felling = (
          np.array(felling, dtype=float) if felling is not None else None
      )
