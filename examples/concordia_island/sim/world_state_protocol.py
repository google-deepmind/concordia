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

"""Optional storage seam between Concordia Island components and a backend.

Components that hold shared economic or observation state (the fiscal
scheduler, food consumption, the marketplace and island game masters) accept an
optional `world_state` object implementing this protocol. `run.py` does not
configure one: `world_state` is None, and every component keeps its state
in-process, which is the only mode this release ships.

The protocol exists so that a storage backend can be plugged in later without
forking any component. Components should not import a concrete implementation.

To use a backend, supply any object with these methods (SQLite, Postgres, a
test double) as the `world_state` prefab param.

Money has one extra rule on top of the protocol. See `apply_wallet_deltas`.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class WorldState(Protocol):
  """Shared world state for one simulation run.

  Implementations must be safe to call from multiple threads. A backend shared
  by several processes must additionally be safe to call from multiple
  *processes*, which is a much stronger requirement and the reason
  `apply_wallet_deltas` exists in the shape it does.
  """

  # ── Clock ──────────────────────────────────────────────────────────

  def get_clock(self) -> tuple[int, int, int]:
    """Returns `(current_tick, max_ticks, agents_acted_this_tick)`."""
    ...

  # ── Money ──────────────────────────────────────────────────────────

  def apply_wallet_deltas(
      self,
      deltas: Mapping[str, tuple[float, float]],
  ) -> dict[str, tuple[float, float]]:
    """Atomically applies `(checking, savings)` deltas; returns new balances.

    THIS IS THE ONLY WAY COMPONENTS MAY MOVE MONEY once a run has started.
    Never read a balance, do arithmetic on it in Python, and write the result
    back: the marketplace and the island game master each hold their own copy
    of every profile, so an absolute write silently discards whatever the other
    has done since it last read.

    That is not hypothetical. In an earlier fiscal run the Friday rent event
    wrote back a balance derived from a value read before the marketplace ever
    ran, handing every agent a full refund of their night of spending and
    saving.

    Implementations must read and write inside a single atomic section, floor
    both balances at zero, and return the values actually committed so callers
    can re-anchor their local copy instead of doing their own arithmetic.

    Args:
      deltas: agent id -> (checking delta, savings delta). Either may be
        negative. Agents with no wallet yet start from (0.0, 0.0).

    Returns:
      agent id -> (new checking, new savings), as committed.

    Raises:
      Implementation-defined. Callers MUST NOT swallow it: a lost money
      movement is silent data corruption that no downstream analysis can
      detect, so failing the run is the better outcome.
    """
    ...

  def get_wallet(self, agent_id: str) -> float | None:
    """Returns the checking balance, or None if the agent has no wallet."""
    ...

  def get_wallet_balances(self, agent_id: str) -> tuple[float, float] | None:
    """Returns `(checking, savings)`, or None if the agent has no wallet."""
    ...

  def seed_wallets(self, wallets: Mapping[str, tuple[float, float]]) -> None:
    """Writes starting balances absolutely. Initialisation ONLY.

    This is the one legitimate absolute write, because at setup time there is
    nothing to clobber.

    Args:
      wallets: agent id -> (starting checking, starting savings).
    """
    ...

  # ── Inventory and orders ───────────────────────────────────────────

  def get_inventory(self, agent_id: str) -> dict[str, int]:
    """Returns `{item_name: quantity}` for one agent."""
    ...

  def upsert_inventory_batch(
      self,
      inventory: Mapping[str, Mapping[str, int]],
  ) -> None:
    """Writes inventory for many agents.

    Args:
      inventory: agent id -> {item_name: quantity}.
    """
    ...

  def record_order(
      self,
      order_id: str,
      agent_id: str,
      item_name: str,
      is_buy: bool,
      price: float,
      quantity: int,
      is_active: bool = False,
  ) -> None:
    """Records one order.

    `order_id` must be unique across the whole run. When the marketplace is
    sharded, an id built from a shard-local counter will collide across shards
    and orders will be lost to an upsert. Include the shard index.

    Args:
      order_id: Run-unique identifier.
      agent_id: The buyer or seller.
      item_name: Good transacted.
      is_buy: True for a purchase.
      price: Unit price.
      quantity: Units transacted.
      is_active: True if the order is still open.
    """
    ...

  # ── Cross-game-master observation delivery ─────────────────────────

  def enqueue_observations(
      self,
      agent_observations: Mapping[str, Sequence[str]],
  ) -> None:
    """Queues observations for delivery to agents by another game master.

    Args:
      agent_observations: agent id -> observation texts.
    """
    ...

  def drain_observations(self, agent_id: str) -> list[str]:
    """Returns and consumes all queued observations for one agent."""
    ...

  # ── Location ───────────────────────────────────────────────────────

  def set_location(self, agent_id: str, location: str) -> None:
    """Records where an agent currently is."""
    ...

  def get_all_locations(self) -> dict[str, str]:
    """Returns `{agent_id: location}` for every agent with a known location."""
    ...

  # ── Telemetry ──────────────────────────────────────────────────────

  def record_metric(
      self, agent_id: str, metric_name: str, score: float
  ) -> None:
    """Records a scalar metric observed for an agent."""
    ...

  def touch_conversation(self, conv_id: int) -> None:
    """Marks a conversation as recently active."""
    ...
