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

"""Trade and logistics between polities.

A polity may spend its turn putting a good up for sale or asking to buy one.
Each tick the orders clear on the contrib `MarketPlace` (a clearing house:
buyers and sellers meet at the midpoint of their limit prices). Sold goods
do not teleport: they leave the seller's storehouse on a contrib
`StockFlowLedger` link and arrive ticks later, limited by what porters (or,
once a people knows roads, carts) can carry. Silver changes hands at once.
"""

# pylint: disable=missing-function-docstring

from collections.abc import Callable, Mapping
import dataclasses
import math
import re
import statistics
from typing import Any

from concordia.contrib.components.game_master import marketplace
from concordia.contrib.components.game_master import stock_flow_ledger
from examples.deep_time.world import civ as civ_lib

SILVER = 'silver'
_SALE_RE = re.compile(
    r'Seller (\S+) sold (\d+) units at (\S+) of (\S+) to Buyer (\S+)'
)


@dataclasses.dataclass(frozen=True)
class GoodSpec:
  key: str
  category: str
  price: float
  surplus: float
  requires: str | None = None  # Unlock tag both trading peoples need.


class Trade:
  """A clearing-house market and porter routes between every pair of peoples."""

  def __init__(self, config: Mapping[str, Any], civ: civ_lib.Civilization):
    """Initializes the market.

    Args:
      config: Contents of `data/trade.yaml`.
      civ: The civilization whose polities trade (and whose ledger holds their
        goods and silver).
    """
    self._civ = civ
    self.ledger = civ.ledger
    self.margin = float(config['margin'])
    self.price_response = float(config['price_response'])
    self.sell_share = float(config['sell_share'])
    self.order_ticks = int(config['order_ticks'])
    self.goods = [GoodSpec(**g) for g in config['goods']]
    self.refill_to = {
        good: float(levels['refill_to'])
        for good, levels in config['automation']['levels'].items()
    }
    self.prices = {g.key: g.price for g in self.goods}
    news = config.get('news', {})
    self.news_latency = int(news.get('latency', 0))
    self.wire_latency = int(news.get('wire_latency', 0))
    # Prices after each clearing, newest last (as many as news can lag).
    self.history: list[dict[str, float]] = [dict(self.prices)]
    self.stale_orders = 0  # Orders priced more than 10% off the market.
    keys = sorted(civ.polities)
    for key in keys:
      self.ledger.deposit(key, SILVER, float(config['silver_start']), 'start')
    self.market = marketplace.MarketPlace(
        acting_player_names=keys,
        agents=[
            marketplace.MarketplaceAgent(key, 'polity', 0.0, {}, [])
            for key in keys
        ],
        goods=[
            marketplace.Good(g.category, 'plain', g.key, price=g.price)
            for g in self.goods
        ],
        market_type='clearing_house',
    )
    routes = config['routes']
    grid = routes.get('grid', {})
    self.grid_loss = float(grid.get('loss', 0.0))
    for kind in ('porters', 'roads', 'rail', 'grid'):
      if kind not in routes:
        continue
      route = routes[kind]
      for source in keys:
        for target in keys:
          if source == target:
            continue
          for good in self.goods:
            if (kind == 'grid') != (good.requires == 'grid'):
              continue  # Energy goes only by wire; goods never do.
            self.ledger.add_link(
                stock_flow_ledger.Link(
                    name=self._link(kind, source, target, good.key),
                    source=source,
                    target=target,
                    resource=good.key,
                    capacity=float(route['capacity']),
                    latency=int(route['latency']),
                )
            )
    # Standing orders: polity -> [(good, side, clearings left)].
    self.intents: dict[str, list[tuple[str, str, int]]] = {}
    self.trades = 0
    self.voided = 0  # Matched sales an embargo voided.
    # Pairs that may not trade (e.g. an embargo after a broken treaty).
    self.blocked: Callable[[str, str], bool] = lambda a, b: False
    # Units bid for, per good, at each clearing (for the bullwhip ratio),
    # and bids placed while earlier purchases were still on the road.
    self.bids: list[dict[str, int]] = []
    self.blind_reorders = 0

  @staticmethod
  def _link(kind: str, source: str, target: str, good: str) -> str:
    return f'{kind}:{source}->{target}:{good}'

  def _wanted(self, polity: str, good: str) -> int:
    """Units a buyer orders: enough to refill its store.

    Without tallies (writing) a people counts only what is in its store, so
    while earlier purchases are on the road it keeps ordering more: the
    bullwhip.

    Args:
      polity: The buyer.
      good: The good.

    Returns:
      Units to bid for.
    """
    have = self.ledger.stock(polity, good)
    if self._civ.unlocked(polity, 'tally'):
      have += self._civ.incoming(polity).get(good, 0.0)
    target = self.refill_to.get(good, self.good(good).surplus)
    return max(0, int(math.ceil(target - have)))

  def _reserve(self, polity: str, good: str) -> float:
    """What a polity keeps back when selling: half a surplus, plus food."""
    reserve = self.good(good).surplus / 2.0
    if good == 'grain':
      people = self._civ.polities[polity]
      reserve += people.population * self._civ.food_per_pop
    return reserve

  def good(self, key: str) -> GoodSpec:
    return next(g for g in self.goods if g.key == key)

  def sellable(self, polity: str) -> list[str]:
    """Goods the polity holds a surplus of."""
    known = self._civ.known_goods(polity)
    return [
        g.key
        for g in self.goods
        if g.key in known
        and self._allowed(polity, g)
        and self.ledger.stock(polity, g.key) >= g.surplus
    ]

  def _allowed(self, polity: str, good: GoodSpec) -> bool:
    return good.requires is None or self._civ.unlocked(polity, good.requires)

  def buyable(self, polity: str) -> list[str]:
    """Goods the polity could afford at least one unit of."""
    silver = self.ledger.stock(polity, SILVER)
    known = self._civ.known_goods(polity)
    return [
        g.key
        for g in self.goods
        if g.key in known
        and self._allowed(polity, g)
        and silver >= self.prices[g.key] * (1.0 + self.margin)
    ]

  def intend(self, polity: str, good: str, side: str) -> None:
    """Records a standing order for the next `order_ticks` clearings.

    A new order on a good replaces any order the polity has on it, so a
    people never bids for what it is also offering.

    Args:
      polity: The ordering polity.
      good: The good.
      side: 'ask' (sell) or 'bid' (buy).
    """
    orders = [o for o in self.intents.get(polity, []) if o[0] != good]
    self.intents[polity] = orders + [(good, side, self.order_ticks)]

  def standing(self, polity: str) -> str:
    """The polity's silver and standing orders, for its prompt."""
    orders = ', '.join(
        f'{"offer" if side == "ask" else "buy"} {good} ({left} more)'
        for good, side, left in self.intents.get(polity, [])
    )
    silver = self.ledger.stock(polity, SILVER)
    return f'Silver {silver:.1f}. Standing orders: {orders or "none"}.'

  def step(self) -> list[str]:
    """Places this tick's orders, clears the market and ships what sold."""
    market = self.market
    for key in sorted(self._civ.polities):
      agent = market.agent(key)
      agent.cash = self.ledger.stock(key, SILVER)
      agent.inventory = {
          g.key: int(math.floor(self.ledger.stock(key, g.key)))
          for g in self.goods
      }
    volume = {g.key: {'ask': 0, 'bid': 0} for g in self.goods}
    for key, intents in sorted(self.intents.items()):
      for good, side, _ in intents:
        price = self.believed_price(key, good)
        stale = abs(price - self.prices[good]) > 0.1 * self.prices[good]
        if side == 'ask':
          spare = self.ledger.stock(key, good) - self._reserve(key, good)
          qty = int(max(0.0, spare) * self.sell_share)
          limit = price * (1.0 - self.margin)
        else:
          limit = price * (1.0 + self.margin)
          qty = int(self.ledger.stock(key, SILVER) / limit) if limit else 0
          qty = min(qty, self._wanted(key, good))
        if qty > 0:
          self.stale_orders += int(stale)
          market.place_order(key, good, side, limit, qty)
          volume[good][side] += qty
          if side == 'bid' and self._civ.incoming(key).get(good, 0.0) > 0:
            self.blind_reorders += 1
    self.bids.append({good: placed['bid'] for good, placed in volume.items()})
    self.intents = {
        key: [(g, s, left - 1) for g, s, left in orders if left > 1]
        for key, orders in self.intents.items()
    }
    # The clearing house fills at the midpoint of the limits, which is the
    # posted price; prices move with the balance of orders instead: unmet
    # demand raises them, unsold supply lowers them.
    _, completed, _ = market.clear()
    for good, placed in volume.items():
      total = placed['ask'] + placed['bid']
      if total > 0:
        imbalance = (placed['bid'] - placed['ask']) / total
        self.prices[good] = max(
            0.05, self.prices[good] * math.exp(self.price_response * imbalance)
        )
    self.history = (self.history + [dict(self.prices)])[
        -(self.news_lag_max() + 1) :
    ]
    for key in self._civ.polities:
      market.drain_outcomes(key)
    events = []
    for text in completed:
      match = _SALE_RE.search(text)
      if match is None:
        continue
      seller, qty, price, good, buyer = match.groups()
      events.extend(self._ship(seller, buyer, good, int(qty), float(price)))
    return events

  def _ship(
      self,
      seller: str,
      buyer: str,
      good: str,
      qty: int,
      price: float,
  ) -> list[str]:
    if seller == buyer:
      return []
    if self.blocked(seller, buyer):
      # The clearing house matched an embargoed pair: the deal falls
      # through, and both orders are spent (the cost of an embargo).
      self.voided += 1
      names = self._civ.polities
      return [
          f'The embargo voids a sale of {qty} {good} from the'
          f' {names[seller].name} to the {names[buyer].name}.'
      ]
    kind = 'porters'
    if self.good(good).requires == 'grid':
      kind = 'grid'
    elif self._civ.unlocked(seller, 'rail'):
      kind = 'rail'
    elif self._civ.unlocked(seller, 'logistics'):
      kind = 'roads'
    link = self._link(kind, seller, buyer, good)
    shipped = self.ledger.ship(link, qty, f'{seller} sells {good}')
    if shipped <= 0:
      return []
    self.ledger.transfer(
        buyer,
        seller,
        SILVER,
        shipped * price,
        f'{buyer} buys {good}',
        partial=True,
    )
    latency = next(l.latency for l in self.ledger.links() if l.name == link)
    self.trades += 1
    names = self._civ.polities
    if kind == 'grid':
      # Delivered at once; the lines lose a share on the way.
      self.ledger.withdraw(
          buyer,
          good,
          shipped * self.grid_loss,
          'transmission losses',
          partial=True,
      )
      return [
          f'{names[seller].name} sell {shipped:g} {good} to'
          f' {names[buyer].name} at {price:.2f} silver over the power grid'
          f' ({self.grid_loss:.0%} lost on the lines).'
      ]
    return [
        f'{names[seller].name} sell {shipped:g} {good} to'
        f' {names[buyer].name} at {price:.2f} silver; it arrives in'
        f' {latency} ticks by {kind}.'
    ]

  def news_lag_max(self) -> int:
    return max(self.news_latency, self.wire_latency)

  def news_lag(self, polity: str) -> int:
    """How many clearings old a people's price news is."""
    if self._civ.unlocked(polity, 'wire_news'):
      return self.wire_latency
    return self.news_latency

  def believed_price(self, polity: str, good: str) -> float:
    """The price a people believes, from news `news_lag` clearings old."""
    lag = min(self.news_lag(polity), len(self.history) - 1)
    return self.history[-1 - lag].get(good, self.prices[good])

  def bullwhip(self, skip: int = 5) -> dict[str, float]:
    """Order amplification per good: var(units bid) / var(units used).

    Above 1, buyers' orders swing more than their real needs: the bullwhip
    effect of ordering on stale or partial information.

    Args:
      skip: Ticks to ignore at the start (the settling-in).

    Returns:
      Good -> ratio, for goods with orders and varying use.
    """
    used = self._civ.consumption[skip:]
    bids = self.bids[skip:]
    ratios = {}
    for good in (g.key for g in self.goods):
      orders = [b.get(good, 0) for b in bids]
      use = [c.get(good, 0.0) for c in used]
      if len(orders) < 3 or not any(orders):
        continue
      var_use = statistics.pvariance(use) if len(use) > 1 else 0.0
      if var_use > 1e-9:
        ratios[good] = round(statistics.pvariance(orders) / var_use, 2)
    return ratios

  def payload(self) -> dict[str, Any]:
    return {
        'prices': {k: round(v, 2) for k, v in self.prices.items()},
        'silver': {
            k: round(self.ledger.stock(k, SILVER), 2)
            for k in sorted(self._civ.polities)
        },
        'in_transit': {
            g.key: round(self.ledger.in_transit(g.key), 2) for g in self.goods
        },
        'trades': self.trades,
    }

  def get_state(self) -> dict[str, Any]:
    return {
        'prices': dict(self.prices),
        'intents': {k: [list(i) for i in v] for k, v in self.intents.items()},
        'trades': self.trades,
        'bids': [dict(b) for b in self.bids],
        'blind_reorders': self.blind_reorders,
        'market_history': list(self.market.history),
        'history': [dict(h) for h in self.history],
        'stale_orders': self.stale_orders,
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.prices = {k: float(v) for k, v in state['prices'].items()}
    self.history = [
        {k: float(v) for k, v in h.items()}
        for h in state.get('history', [self.prices])
    ]
    self.stale_orders = int(state.get('stale_orders', 0))
    self.intents = {
        k: [(g, side, int(left)) for g, side, left in v]
        for k, v in state['intents'].items()
    }
    self.trades = int(state['trades'])
    self.bids = [dict(b) for b in state.get('bids', [])]
    self.blind_reorders = int(state.get('blind_reorders', 0))
    self.market.history = list(state.get('market_history', []))
