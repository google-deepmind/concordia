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

"""A deterministic, conserved stock-and-flow ledger for game masters.

`game_master/inventory.Inventory` asks a language model whether narrated
events changed what players hold. Economic and ecological simulations also
need the opposite: exact bookkeeping that language models cannot bend. This
module provides it:

* `Ledger`: stocks of resources held by named holders (players, regions,
  reservoirs). Every change is an explicit flow with a reason: `deposit` and
  `withdraw` cross the system boundary, `transfer` moves stock between
  holders, `apply_recipe` converts inputs into outputs (`max_batches` says
  how many batches stock allows, `limiting_input` which input bounds them),
  and `Link`s ship
  stock with a per-tick capacity and a latency in ticks. Per-tick production
  statistics, an audit trail and a conservation check make every quantity
  traceable.
* `balance_chain`: the ratio planner. Given recipes and a target rate for one
  of them, it works back through the producers of every input and returns how
  many batches a tick each recipe must run so that nothing upstream is short
  or piles up (the "ratio calculator" of factory games), plus the inputs no
  recipe makes.
* `StockFlowLedger`: a game master component that holds a `Ledger` and shows a
  compact summary in the game master's context.

Example:

```python
ledger = stock_flow_ledger.Ledger()
ledger.deposit('Aurelia', 'ore', 10, reason='mine')
bloomery = stock_flow_ledger.Recipe(
    'bloomery', inputs={'ore': 2, 'charcoal': 1}, outputs={'iron': 1, 'slag': 1}
)
ledger.add_link(stock_flow_ledger.Link(
    'river', source='Aurelia', target='Borea', resource='iron',
    capacity=5, latency=2))
ledger.apply_recipe('Aurelia', bloomery, batches=3)
ledger.ship('river', 3)
ledger.advance()  # next tick; the iron arrives in Borea two ticks after it left
```
"""

from collections.abc import Iterable, Mapping, Sequence
import copy
import dataclasses
import math
import threading
from typing import Any, cast

from concordia.components.agent import action_spec_ignored
from concordia.typing import entity_component

EXTERNAL = '__external__'
DEFAULT_LEDGER_COMPONENT_KEY = '__ledger__'
DEFAULT_LEDGER_PRE_ACT_LABEL = 'Ledger'
_EPSILON = 1e-9


class InsufficientStockError(ValueError):
  """Raised when a flow needs more stock than a holder has."""


@dataclasses.dataclass(frozen=True)
class Recipe:
  """Converts inputs into outputs (and by-products) at one holder.

  Attributes:
    name: Recipe name, used in flow reasons.
    inputs: Resource amounts consumed per batch.
    outputs: Resource amounts produced per batch, by-products included.
  """

  name: str
  inputs: Mapping[str, float]
  outputs: Mapping[str, float]


@dataclasses.dataclass(frozen=True)
class Link:
  """A route that moves one resource between two holders.

  Attributes:
    name: Unique link name.
    source: Holder that sends.
    target: Holder that receives.
    resource: Resource carried.
    capacity: Maximum amount shipped per tick.
    latency: Ticks between shipping and delivery; 0 delivers immediately.
  """

  name: str
  source: str
  target: str
  resource: str
  capacity: float = math.inf
  latency: int = 0


@dataclasses.dataclass(frozen=True)
class Flow:
  """One entry of the audit trail."""

  tick: int
  kind: str
  resource: str
  amount: float
  source: str
  target: str
  reason: str = ''

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


@dataclasses.dataclass
class _Shipment:
  link: str
  resource: str
  amount: float
  target: str
  arrival_tick: int
  reason: str

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def _empty_stats() -> dict[str, float]:
  return {'produced': 0.0, 'consumed': 0.0, 'moved': 0.0, 'shipped': 0.0}


class Ledger:
  """Stocks, flows, recipes and links, with conservation checks.

  Thread-safe. Amounts are floats; stocks may not go negative unless
  `allow_negative` is set.
  """

  def __init__(self, *, allow_negative: bool = False, audit_limit: int = 5000):
    """Initializes an empty ledger at tick 0.

    Args:
      allow_negative: Whether holders may go into debt.
      audit_limit: Maximum number of flows kept in the audit trail (oldest are
        dropped first; statistics and conservation are unaffected).
    """
    self._allow_negative = allow_negative
    self._audit_limit = audit_limit
    self._lock = threading.RLock()
    self._tick = 0
    self._stocks: dict[str, dict[str, float]] = {}
    self._links: dict[str, Link] = {}
    self._used_capacity: dict[str, float] = {}
    self._transit: list[_Shipment] = []
    self._audit: list[Flow] = []
    self._stats: dict[int, dict[str, dict[str, float]]] = {}
    # Net amount of each resource that entered the system (deposits and
    # recipe outputs minus withdrawals and recipe inputs).
    self._boundary: dict[str, float] = {}

  # Queries.

  @property
  def tick(self) -> int:
    with self._lock:
      return self._tick

  def stock(self, holder: str, resource: str) -> float:
    with self._lock:
      return self._stocks.get(holder, {}).get(resource, 0.0)

  def holdings(self, holder: str) -> dict[str, float]:
    with self._lock:
      return {k: v for k, v in self._stocks.get(holder, {}).items() if v}

  def holders(self) -> list[str]:
    with self._lock:
      return sorted(self._stocks)

  def resources(self) -> list[str]:
    with self._lock:
      names: set[str] = set().union(*self._stocks.values())
      names.update(s.resource for s in self._transit)
      return sorted(names)

  def in_transit(self, resource: str | None = None) -> float:
    with self._lock:
      return sum(
          s.amount
          for s in self._transit
          if resource is None or s.resource == resource
      )

  def shipments(self) -> list[dict[str, Any]]:
    with self._lock:
      return [s.to_dict() for s in self._transit]

  def total(self, resource: str, include_transit: bool = True) -> float:
    """Sum of a resource over all holders (and in transit)."""
    with self._lock:
      held = sum(stocks.get(resource, 0.0) for stocks in self._stocks.values())
      return held + (self.in_transit(resource) if include_transit else 0.0)

  def audit(self) -> dict[str, float]:
    """Per resource, the total minus the net boundary inflow (should be ~0)."""
    with self._lock:
      names = set(self.resources()) | set(self._boundary)
      return {
          r: self.total(r) - self._boundary.get(r, 0.0) for r in sorted(names)
      }

  def is_conserved(self, tolerance: float = 1e-6) -> bool:
    return all(abs(v) <= tolerance for v in self.audit().values())

  def flows(self, tick: int | None = None) -> list[Flow]:
    """The audit trail, optionally for one tick."""
    with self._lock:
      return [f for f in self._audit if tick is None or f.tick == tick]

  def production(self, tick: int | None = None) -> dict[str, dict[str, float]]:
    """Factorio-style statistics per resource for a tick (default: current).

    Args:
      tick: Tick to inspect, or None for the current tick.

    Returns:
      For each resource: `produced` (deposits and recipe outputs), `consumed`
      (withdrawals and recipe inputs), `moved` (transfers and deliveries) and
      `shipped` (put on links).
    """
    with self._lock:
      stats = self._stats.get(self._tick if tick is None else tick, {})
      return {r: dict(v) for r, v in sorted(stats.items())}

  # Mutations.

  def _record(
      self,
      kind: str,
      resource: str,
      amount: float,
      source: str,
      target: str,
      reason: str,
  ) -> None:
    """Appends a flow entry to the bounded audit trail."""
    self._audit.append(
        Flow(self._tick, kind, resource, amount, source, target, reason)
    )
    if len(self._audit) > self._audit_limit:
      del self._audit[: len(self._audit) - self._audit_limit]

  def _stat(self, resource: str, key: str, amount: float) -> None:
    tick_stats = self._stats.setdefault(self._tick, {})
    tick_stats.setdefault(resource, _empty_stats())[key] += amount

  def _add(self, holder: str, resource: str, amount: float) -> None:
    stocks = self._stocks.setdefault(holder, {})
    stocks[resource] = stocks.get(resource, 0.0) + amount
    if abs(stocks[resource]) < _EPSILON:
      stocks[resource] = 0.0

  def _available(
      self, holder: str, resource: str, amount: float, partial: bool
  ) -> float:
    """Returns the amount that can be drawn from `holder`, or raises."""
    if amount < 0:
      raise ValueError('Amounts must be non-negative.')
    if self._allow_negative:
      return amount
    have = max(0.0, self.stock(holder, resource))
    if amount > have + _EPSILON:
      if not partial:
        raise InsufficientStockError(
            f'{holder} has {have:g} {resource}, needs {amount:g}'
        )
      return have
    return min(amount, have)

  def deposit(
      self, holder: str, resource: str, amount: float, reason: str = ''
  ) -> float:
    """Adds stock from outside the system (a declared source)."""
    if amount < 0:
      raise ValueError('Amounts must be non-negative.')
    with self._lock:
      self._add(holder, resource, amount)
      self._boundary[resource] = self._boundary.get(resource, 0.0) + amount
      self._stat(resource, 'produced', amount)
      self._record('deposit', resource, amount, EXTERNAL, holder, reason)
      return amount

  def withdraw(
      self,
      holder: str,
      resource: str,
      amount: float,
      reason: str = '',
      partial: bool = False,
  ) -> float:
    """Removes stock to outside the system (a declared sink).

    Args:
      holder: Holder to take from.
      resource: Resource.
      amount: Amount requested.
      reason: Audit-trail reason.
      partial: If True, take what is available instead of raising.

    Returns:
      The amount withdrawn.
    """
    with self._lock:
      amount = self._available(holder, resource, amount, partial)
      self._add(holder, resource, -amount)
      self._boundary[resource] = self._boundary.get(resource, 0.0) - amount
      self._stat(resource, 'consumed', amount)
      self._record('withdraw', resource, amount, holder, EXTERNAL, reason)
      return amount

  def transfer(
      self,
      source: str,
      target: str,
      resource: str,
      amount: float,
      reason: str = '',
      partial: bool = False,
  ) -> float:
    """Moves stock between holders (conserved); returns the amount moved."""
    with self._lock:
      amount = self._available(source, resource, amount, partial)
      self._add(source, resource, -amount)
      self._add(target, resource, amount)
      self._stat(resource, 'moved', amount)
      self._record('transfer', resource, amount, source, target, reason)
      return amount

  def max_batches(self, holder: str, recipe: Recipe) -> float:
    """How many batches the holder's stock allows (inf for no inputs)."""
    with self._lock:
      limits = [
          self.stock(holder, r) / amount
          for r, amount in recipe.inputs.items()
          if amount > 0
      ]
      return max(0.0, min(limits)) if limits else math.inf

  def limiting_input(self, holder: str, recipe: Recipe) -> str | None:
    """The input that bounds `max_batches` for the holder: its bottleneck.

    The input of which the holder has the fewest batches' worth in stock
    (the first such in the recipe's order), so a game can say what holds a
    recipe back, and which producer to strengthen.

    Args:
      holder: Where the recipe would run.
      recipe: The recipe.

    Returns:
      The limiting input, or None for a recipe without inputs.
    """
    with self._lock:
      needs = [(r, amount) for r, amount in recipe.inputs.items() if amount > 0]
      if not needs:
        return None
      limiting, _ = min(
          needs, key=lambda need: self.stock(holder, need[0]) / need[1]
      )
      return limiting

  def apply_recipe(
      self,
      holder: str,
      recipe: Recipe,
      batches: float = 1.0,
      reason: str = '',
      partial: bool = False,
  ) -> float:
    """Consumes inputs and produces outputs at a holder.

    Args:
      holder: Where the recipe runs.
      recipe: The recipe.
      batches: Number of batches (may be fractional).
      reason: Audit-trail reason (defaults to the recipe name).
      partial: If True, run as many batches as stock allows.

    Returns:
      The number of batches run.

    Raises:
      InsufficientStockError: If inputs are missing and `partial` is False.
    """
    if batches < 0:
      raise ValueError('batches must be non-negative')
    with self._lock:
      if not self._allow_negative:
        possible = self.max_batches(holder, recipe)
        if batches > possible + _EPSILON:
          if not partial:
            raise InsufficientStockError(
                f'{holder} can run {recipe.name} {possible:g} times, not'
                f' {batches:g}'
            )
          batches = possible
      why = reason or recipe.name
      for resource, amount in recipe.inputs.items():
        quantity = amount * batches
        self._add(holder, resource, -quantity)
        self._boundary[resource] = self._boundary.get(resource, 0.0) - quantity
        self._stat(resource, 'consumed', quantity)
        self._record('recipe_in', resource, quantity, holder, EXTERNAL, why)
      for resource, amount in recipe.outputs.items():
        quantity = amount * batches
        self._add(holder, resource, quantity)
        self._boundary[resource] = self._boundary.get(resource, 0.0) + quantity
        self._stat(resource, 'produced', quantity)
        self._record('recipe_out', resource, quantity, EXTERNAL, holder, why)
      return batches

  def add_link(self, link: Link) -> None:
    with self._lock:
      if link.name in self._links:
        raise ValueError(f'Duplicate link {link.name!r}')
      self._links[link.name] = link

  def links(self) -> list[Link]:
    with self._lock:
      return list(self._links.values())

  def remaining_capacity(self, link_name: str) -> float:
    with self._lock:
      link = self._links[link_name]
      return max(0.0, link.capacity - self._used_capacity.get(link_name, 0.0))

  def ship(self, link_name: str, amount: float, reason: str = '') -> float:
    """Puts stock on a link, limited by this tick's capacity and the stock.

    Shipments arrive `latency` ticks later, during `advance`. A zero-latency
    link delivers immediately.

    Args:
      link_name: Name of the registered link to ship on.
      amount: Requested amount to ship.
      reason: Optional audit-trail reason.

    Returns:
      The amount shipped.
    """
    with self._lock:
      link = self._links[link_name]
      amount = min(amount, self.remaining_capacity(link_name))
      amount = self._available(link.source, link.resource, amount, True)
      if amount <= 0:
        return 0.0
      self._used_capacity[link_name] = (
          self._used_capacity.get(link_name, 0.0) + amount
      )
      self._add(link.source, link.resource, -amount)
      self._stat(link.resource, 'shipped', amount)
      self._record(
          'ship', link.resource, amount, link.source, link.name, reason
      )
      shipment = _Shipment(
          link.name,
          link.resource,
          amount,
          link.target,
          self._tick + link.latency,
          reason,
      )
      if link.latency <= 0:
        self._deliver(shipment)
      else:
        self._transit.append(shipment)
      return amount

  def _deliver(self, shipment: _Shipment) -> None:
    self._add(shipment.target, shipment.resource, shipment.amount)
    self._stat(shipment.resource, 'moved', shipment.amount)
    self._record(
        'deliver',
        shipment.resource,
        shipment.amount,
        shipment.link,
        shipment.target,
        shipment.reason,
    )

  def advance(self, ticks: int = 1) -> list[Flow]:
    """Moves to the next tick: resets link capacity, delivers shipments due.

    Args:
      ticks: Number of ticks to advance.

    Returns:
      The delivery flows recorded during this advance.
    """
    if ticks < 0:
      raise ValueError('ticks must be non-negative')
    with self._lock:
      start = len(self._audit)
      for _ in range(ticks):
        self._tick += 1
        self._used_capacity.clear()
        due = [s for s in self._transit if s.arrival_tick <= self._tick]
        self._transit = [
            s for s in self._transit if s.arrival_tick > self._tick
        ]
        for shipment in due:
          self._deliver(shipment)
      return self._audit[start:]

  def forget_statistics_before(self, tick: int) -> None:
    """Drops per-tick statistics older than `tick` to bound memory."""
    with self._lock:
      self._stats = {t: s for t, s in self._stats.items() if t >= tick}

  # Checkpointing.

  def get_state(self) -> dict[str, Any]:
    """Returns a JSON-serializable snapshot of the ledger state."""
    with self._lock:
      return {
          'tick': self._tick,
          'allow_negative': self._allow_negative,
          'stocks': {h: dict(s) for h, s in self._stocks.items()},
          'links': [dataclasses.asdict(l) for l in self._links.values()],
          'used_capacity': dict(self._used_capacity),
          'transit': [s.to_dict() for s in self._transit],
          'boundary': dict(self._boundary),
          'stats': {
              str(t): {r: dict(v) for r, v in s.items()}
              for t, s in self._stats.items()
          },
          'audit': [f.to_dict() for f in self._audit],
      }

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Restores the ledger state from `get_state`."""
    with self._lock:
      self._tick = int(state.get('tick', 0))
      self._allow_negative = bool(
          state.get('allow_negative', self._allow_negative)
      )
      self._stocks = {
          h: {r: float(v) for r, v in s.items()}
          for h, s in state.get('stocks', {}).items()
      }
      self._links = {l['name']: Link(**l) for l in state.get('links', [])}
      self._used_capacity = {
          k: float(v) for k, v in state.get('used_capacity', {}).items()
      }
      self._transit = [_Shipment(**s) for s in state.get('transit', [])]
      self._boundary = {
          k: float(v) for k, v in state.get('boundary', {}).items()
      }
      self._stats = {
          int(t): {r: dict(v) for r, v in s.items()}
          for t, s in state.get('stats', {}).items()
      }
      self._audit = [Flow(**f) for f in state.get('audit', [])]

  def __deepcopy__(self, memo: dict[int, Any] | None = None) -> 'Ledger':
    copied = Ledger(
        allow_negative=self._allow_negative, audit_limit=self._audit_limit
    )
    copied.set_state(copy.deepcopy(self.get_state(), memo))
    return copied


@dataclasses.dataclass(frozen=True)
class ChainPlan:
  """A balanced production chain.

  Attributes:
    batches: Recipe name -> batches a tick, for the target and everything
      upstream of it.
    external: Resource -> amount a tick that no recipe in the chain makes (raw
      materials to hold, buy or ship in).
  """

  batches: Mapping[str, float]
  external: Mapping[str, float]


def balance_chain(
    recipes: Sequence[Recipe],
    target: str,
    batches: float = 1.0,
    producers: Mapping[str, str] | None = None,
) -> ChainPlan:
  """Balances the chain that feeds `target` at `batches` a tick.

  Each input is made by its producer: the recipe named in `producers`, or else
  the first recipe in `recipes` with a positive net output of it (output less
  what it consumes itself). A producer without a positive net output counts as
  absent, and inputs without a producer are external. Demand is propagated in
  dependency order, so a resource shared by several consumers (say, energy for
  two workshops) is made once, in total; a recipe that makes several needed
  resources (co-products) runs at the rate its most demanded output needs.

  Args:
    recipes: The recipes available; names must be unique.
    target: Name of the recipe to run.
    batches: Batches a tick of the target.
    producers: Optional resource -> recipe name choices.

  Returns:
    The batches a tick of every recipe in the chain, and the external inputs.

  Raises:
    KeyError: If `target` (or a chosen producer) is not among `recipes`.
    ValueError: If recipe names repeat, or the chain is circular.
  """
  by_name = {r.name: r for r in recipes}
  if len(by_name) != len(recipes):
    raise ValueError('recipe names must be unique')
  if target not in by_name:
    raise KeyError(target)
  chosen = dict(producers or {})

  def net(recipe: Recipe, resource: str) -> float:
    return recipe.outputs.get(resource, 0.0) - recipe.inputs.get(resource, 0.0)

  def producer(resource: str) -> Recipe | None:
    if resource in chosen:
      recipe = by_name[chosen[resource]]
      return recipe if net(recipe, resource) > 0 else None
    return next((r for r in recipes if net(r, resource) > 0), None)

  def supplied(recipe: Recipe) -> list[tuple[str, float, Recipe | None]]:
    """Inputs a recipe draws from outside itself, with their producers."""
    return [
        (resource, amount, producer(resource))
        for resource, amount in sorted(recipe.inputs.items())
        if recipe.outputs.get(resource, 0.0) <= 0  # Self-supplied otherwise.
    ]

  # Depth-first topological order of the recipes the target depends on.
  order: list[str] = []
  state: dict[str, int] = {}  # 1 = visiting, 2 = done.

  def visit(name: str) -> None:
    if state.get(name) == 2:
      return
    if state.get(name) == 1:
      raise ValueError(f'circular production chain through {name!r}')
    state[name] = 1
    for _, _, upstream in supplied(by_name[name]):
      if upstream is not None:
        visit(upstream.name)
    state[name] = 2
    order.append(name)

  visit(target)
  rates: dict[str, float] = {}
  demand: dict[str, float] = {}  # Resource -> amount a tick its producer owes.
  external: dict[str, float] = {}
  for name in reversed(order):  # Every consumer before its producers.
    recipe = by_name[name]
    rate = float(batches) if name == target else 0.0
    for resource, need in demand.items():
      upstream = producer(resource)
      if upstream is not None and upstream.name == name:
        rate = max(rate, need / net(recipe, resource))
    rates[name] = rate
    for resource, amount, upstream in supplied(recipe):
      need = rate * amount
      if upstream is None:
        external[resource] = external.get(resource, 0.0) + need
      else:
        demand[resource] = demand.get(resource, 0.0) + need
  return ChainPlan(batches=rates, external=external)


def format_holdings(
    ledger: Ledger,
    holders: Sequence[str] | None = None,
    resources: Iterable[str] | None = None,
    precision: int = 3,
) -> str:
  """One line per holder: `holder: resource=amount, ...`."""
  wanted = set(resources) if resources is not None else None
  lines = []
  for holder in holders if holders is not None else ledger.holders():
    items = [
        f'{r}={v:.{precision}g}'
        for r, v in sorted(ledger.holdings(holder).items())
        if wanted is None or r in wanted
    ]
    if items:
      lines.append(f'{holder}: {", ".join(items)}')
  return '\n'.join(lines)


class StockFlowLedger(action_spec_ignored.ActionSpecIgnored):
  """A game master component holding a `Ledger`.

  Its pre-act value is a compact summary of holdings, so the game master's
  context always reflects the exact books. Game logic calls the ledger's
  methods directly (through `ledger`); nothing here calls a language model.
  """

  def __init__(
      self,
      ledger: Ledger | None = None,
      *,
      holders: Sequence[str] | None = None,
      resources: Iterable[str] | None = None,
      pre_act_label: str = DEFAULT_LEDGER_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      ledger: The ledger to hold (a new empty one by default).
      holders: Holders to show in the summary (all by default).
      resources: Resources to show in the summary (all by default).
      pre_act_label: Label shown in the game master's context.
    """
    super().__init__(pre_act_label)
    self._ledger = ledger if ledger is not None else Ledger()
    self._holders = list(holders) if holders is not None else None
    self._resources = list(resources) if resources is not None else None

  @property
  def ledger(self) -> Ledger:
    return self._ledger

  def _make_pre_act_value(self) -> str:
    return format_holdings(self._ledger, self._holders, self._resources)

  def get_state(self) -> entity_component.ComponentState:
    return {'ledger': self._ledger.get_state()}

  def set_state(self, state: entity_component.ComponentState) -> None:
    ledger_state = state.get('ledger')
    if isinstance(ledger_state, Mapping):
      self._ledger.set_state(cast(Mapping[str, Any], ledger_state))

  def __deepcopy__(
      self, memo: dict[int, Any] | None = None
  ) -> 'StockFlowLedger':
    return StockFlowLedger(
        ledger=copy.deepcopy(self._ledger, memo),
        holders=copy.deepcopy(self._holders, memo),
        resources=copy.deepcopy(self._resources, memo),
        pre_act_label=self.get_pre_act_label(),
    )
