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

"""Automation: peoples' standing orders as reflexes.

A people spends one turn to adopt a standing order such as "sell ore
whenever we hold more than 6". It becomes a contrib `reflexes.ReflexSpec`
on the census topic, installed on the TopicBus: from then on the order is
placed whenever the condition holds, with no language model call. Which
reflex operators a people may use depends on its techs (granaries unlock
`threshold`, mathematics `moving_average`). The automation ratio is the
share of market orders placed by reflexes rather than by turns.
"""

# pylint: disable=missing-function-docstring

from collections.abc import Mapping
import dataclasses
import re
import threading
from typing import Any

from concordia.contrib.pubsub import reflexes as reflexes_lib
from concordia.contrib.pubsub import topics
from examples.deep_time.world import civ as civ_lib
from examples.deep_time.world import trade as trade_lib

BASE_OPS = frozenset({'json_field'})
PUBLIC_INPUTS = frozenset({'civ.census', 'econ.prices'})
_ORDER_TOPIC_RE = re.compile(r'^civ\.([^.]+)\.orders$')


class Automation:
  """Standing orders as reflexes on the bus, feeding the market."""

  def __init__(
      self,
      bus: topics.TopicBus,
      civ: civ_lib.Civilization,
      trade: trade_lib.Trade,
      config: Mapping[str, Any],
  ):
    """Initializes automation.

    Args:
      bus: The topic bus (census in, orders out).
      civ: The civilization (techs gate the operators).
      trade: The market that receives the orders.
      config: `automation` section of `data/trade.yaml`: `cooldown` and `levels`
        (good -> {sell_above, buy_below}).
    """
    self._bus = bus
    self._civ = civ
    self._trade = trade
    self.cooldown = float(config['cooldown'])
    self.max_reflexes = int(config['max_reflexes'])
    self.levels = {
        good: {k: float(v) for k, v in levels.items()}
        for good, levels in config['levels'].items()
    }
    self.engine = reflexes_lib.Reflexes(bus)
    self._lock = threading.Lock()
    self.reflex_orders = 0
    self.turn_orders = 0
    # The last draft of standing orders each people adopted (bus seq): a
    # draft is only a draft until the people adopts it by an action.
    self.adopted: dict[str, int] = {}
    bus.matching('civ.').subscribe(self._on_order)

  def allowed_ops(self, polity: str) -> set[str]:
    """Reflex operators the polity's techs unlock."""
    ops = set(BASE_OPS)
    for tech in self._civ.polities[polity].techs:
      for tag in self._civ.tech(tech).effects.get('unlocks', ()):
        if tag.startswith('reflex:'):
          ops.add(tag.split(':', 1)[1])
    return ops

  def counts_transit(self, polity: str) -> bool:
    """Whether the polity's scribes count goods on the road (writing)."""
    return self._civ.unlocked(polity, 'tally')

  @staticmethod
  def name(polity: str, side: str, good: str) -> str:
    return f'{polity}:{side}:{good}'

  def _spec(self, polity: str, side: str, good: str) -> dict[str, Any]:
    level_key = 'sell_above' if side == 'ask' else 'buy_below'
    bound = 'above' if side == 'ask' else 'below'
    # With writing, scribes tally goods on the road too, so a buy order does
    # not keep firing while earlier purchases are still in transit.
    field = 'position' if self.counts_transit(polity) else 'stocks'
    return {
        'name': self.name(polity, side, good),
        'input': 'civ.census',
        'pipeline': [
            {'op': 'json_field', 'path': f'polities.{polity}.{field}.{good}'},
            {'op': 'threshold', bound: self.levels[good][level_key]},
        ],
        'emit': {'topic': f'civ.{polity}.orders', 'template': f'{side} {good}'},
        'cooldown': self.cooldown,
    }

  def offers(self, polity: str) -> dict[str, tuple[str, str]]:
    """Option text -> (side, good) for orders the polity could adopt."""
    if 'threshold' not in self.allowed_ops(polity):
      return {}
    installed = set(self.engine.names(polity))
    held = set(self._orders(polity).values())
    known = self._civ.known_goods(polity)
    options = {}
    for good, levels in sorted(self.levels.items()):
      if good not in known:
        continue
      stock = self._civ.stock(polity, good)
      for side, verb, word, level, relevant in (
          # Offer only the orders the storehouse calls for now, to keep the
          # list of choices short.
          (
              'ask',
              'sell',
              'more',
              levels['sell_above'],
              stock >= levels['sell_above'] / 2,
          ),
          (
              'bid',
              'buy',
              'less',
              levels['buy_below'],
              stock < levels['buy_below'],
          ),
      ):
        if (
            not relevant
            or self.name(polity, side, good) in installed
            or (side, good) in held
        ):
          continue
        text = (
            f'set a standing order: {verb} {good} whenever we hold {word}'
            f' than {level:g}'
        )
        options[text] = (side, good)
    return options

  def _orders(self, polity: str) -> dict[str, tuple[str, str]]:
    """Reflex name -> (side, good) for the polity's installed orders."""
    state = self.engine.get_state()
    orders = {}
    for name in self.engine.names(polity):
      side, _, good = state[name]['spec']['emit']['template'].partition(' ')
      orders[name] = (side, good)
    return orders

  def cancels(self, polity: str) -> dict[str, str]:
    """Option text -> reflex name for orders the polity could cancel."""
    options = {}
    for name, (side, good) in self._orders(polity).items():
      verb = 'selling' if side == 'ask' else 'buying'
      options[f'cancel the standing order for {verb} {good}'] = name
    return options

  def adopt(self, polity: str, side: str, good: str) -> str:
    """Installs a standing order as a reflex; returns what happened."""
    spec = reflexes_lib.ReflexSpec.from_dict(
        self._spec(polity, side, good), self.allowed_ops(polity)
    )
    self.engine.install(spec, owner=polity)
    verb = 'sell' if side == 'ask' else 'buy'
    return (
        f'{self._civ.polities[polity].name} set a standing order to {verb}'
        f' {good}.'
    )

  def cancel(self, polity: str, name: str) -> str:
    self.engine.remove(name)
    return f'{self._civ.polities[polity].name} cancel a standing order.'

  def draft(self, polity: str) -> str | None:
    """The scribes' latest draft of standing orders not yet adopted, if it
    would add any (read from `civ.<key>.policy`, where the mind wrote it)."""
    latest = self._bus.latest(f'civ.{polity}.policy')
    if latest is None or latest.seq <= self.adopted.get(polity, -1):
      return None
    if not self.compile(polity, str(latest.payload), dry_run=True):
      return None
    return str(latest.payload)

  def summary(self, polity: str, text: str) -> str:
    """A draft in words: the orders it would place (sell/buy a good)."""
    names = self.compile(polity, text, dry_run=True, describe=True)
    return '; '.join(names) if names else 'nothing that can be kept'

  def adopt_draft(self, polity: str) -> str:
    """Installs the scribes' latest draft (a people's action); what it did."""
    latest = self._bus.latest(f'civ.{polity}.policy')
    people = self._civ.polities[polity].name
    if latest is None or latest.seq <= self.adopted.get(polity, -1):
      return f'The scribes of the {people} have no new standing orders.'
    self.adopted[polity] = latest.seq
    installed = self.compile(polity, str(latest.payload))
    if not installed:
      return (
          f'The scribes of the {people} wrote no standing order that can be'
          ' kept.'
      )
    return (
        f"The {people} adopt their scribes' {len(installed)} standing"
        f' order{"s" if len(installed) > 1 else ""}:'
        f' {self.summary(polity, str(latest.payload))}.'
    )

  def compile(
      self,
      polity: str,
      text: str,
      dry_run: bool = False,
      describe: bool = False,
  ) -> list[str]:
    """Installs the standing orders a people's scribes wrote as JSON.

    Each reflex is validated by the contrib `reflexes.parse_specs` against
    the people's unlocked operators, and must read a public topic and place
    only this people's own orders (`ask <good>` or `bid <good>` on
    `civ.<key>.orders`). At most `max_reflexes` are kept per people.

    Args:
      polity: The people's key.
      text: The scribes' answer.
      dry_run: Only say what would be installed.
      describe: Name each order in words ("sell ore") instead of its reflex.

    Returns:
      Names of the reflexes installed (or that would be).
    """
    specs, _ = reflexes_lib.parse_specs(text, self.allowed_ops(polity))
    installed = []
    held = set(self._orders(polity).values())
    for spec in specs:
      side, _, good = spec.template.partition(' ')
      if (
          spec.input not in PUBLIC_INPUTS
          or spec.emit_topic != f'civ.{polity}.orders'
          or side not in ('ask', 'bid')
          or good not in self.levels
          or (side, good) in held
      ):
        continue
      held.add((side, good))
      if len(self.engine.names(polity)) + len(installed) * dry_run >= (
          self.max_reflexes
      ):
        break
      if dry_run:
        installed.append(
            f'{"sell" if side == "ask" else "buy"} {good}'
            if describe
            else spec.name
        )
        continue
      spec = dataclasses.replace(spec, name=f'{polity}:policy:{spec.name}'[:60])
      self.engine.install(spec, owner=polity)
      installed.append(spec.name)
    return installed

  def _on_order(self, message: topics.Message) -> None:
    if self._bus.is_restoring:  # A checkpoint's replay: no new orders.
      return
    # A draft of standing orders (`civ.<key>.policy`) is installed only when
    # its people adopts it by an action (`adopt_draft`), never here.
    match = _ORDER_TOPIC_RE.match(message.topic)
    if match is None:
      return
    side, _, good = message.payload.partition(' ')
    if side not in ('ask', 'bid') or good not in self.levels:
      return
    self._trade.intend(match.group(1), good, side)
    with self._lock:
      self.reflex_orders += 1

  def note_turn_order(self) -> None:
    with self._lock:
      self.turn_orders += 1

  def ratio(self) -> float:
    """Share of market orders placed by reflexes."""
    with self._lock:
      total = self.reflex_orders + self.turn_orders
      return self.reflex_orders / total if total else 0.0

  def describe(self, polity: str) -> str:
    orders = self._orders(polity)
    if not orders:
      return 'No standing orders by reflex.'
    parts = []
    for name, (side, good) in orders.items():
      fires = self.engine.stats(name)['fires']
      verb = 'sell' if side == 'ask' else 'buy'
      parts.append(f'{verb} {good} (placed {fires} times)')
    return 'Reflex orders: ' + '; '.join(parts) + '.'

  def get_state(self) -> dict[str, Any]:
    with self._lock:
      return {
          'engine': self.engine.get_state(),
          'reflex_orders': self.reflex_orders,
          'turn_orders': self.turn_orders,
          'adopted': dict(self.adopted),
      }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.engine.set_state(state['engine'])
    with self._lock:
      self.reflex_orders = int(state['reflex_orders'])
      self.turn_orders = int(state['turn_orders'])
      self.adopted = {k: int(v) for k, v in state.get('adopted', {}).items()}
