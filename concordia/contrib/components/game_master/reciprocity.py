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

"""Gifts and debts: who owes whom a return, and by when.

`Reciprocity` is a ledger of obligations between named participants. A gift
or a loan (`give`) leaves the receiver owing the giver a return (by default
`return_share` of what it was given) within `term` steps. A return
(`repay`) settles the debtor's oldest debts to that creditor first; a
creditor may `forgive`. Once a debt is past due, `overdue` reports it as a
default, once, for the game to judge (a bad name, a grudge, a court). When
someone leaves, `settle_estate` passes what was owed to them to an heir (or
cancels it) and cancels, or passes on, what they owed.

Each debtor keeps at most `max_open` open debts (the oldest are dropped), so
a ledger that runs indefinitely stays bounded. Amounts are plain numbers of
any good the game names; the ledger moves no goods itself: the game moves
them and records the obligation.

It is state and rules only, for the GM's other components to call by key
(`DEFAULT_RECIPROCITY_COMPONENT_KEY`). It adds nothing to prompts.

  ledger = reciprocity.Reciprocity(term=5)
  ledger.give('Ada', 'Bo', 2.0, step=10, good='food')  # Bo owes Ada 2.
  ledger.repay('Bo', 'Ada', 1.0, step=12)  # Bo owes Ada 1.
  ledger.overdue(step=16)  # Bo's last 1 is past due: a default.
"""

from collections.abc import Sequence
import dataclasses
import threading
from typing import Any, override

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

# The key a game master registers its ledger under, by convention.
DEFAULT_RECIPROCITY_COMPONENT_KEY = '__reciprocity__'

# The steps a return is due within; the share of a gift owed back; the most
# open debts a debtor keeps; amounts below this count as settled.
TERM = 5
RETURN_SHARE = 1.0
MAX_OPEN = 10
_SETTLED = 1e-9


@dataclasses.dataclass(frozen=True)
class Debt:
  """An obligation: `debtor` owes `creditor` a return.

  Attributes:
    id: The debt's id.
    debtor: Who owes.
    creditor: To whom.
    amount: How much is still owed.
    good: Of what (the game's word).
    given: The step the gift was made.
    due: The step by which it is to be returned.
  """

  id: str
  debtor: str
  creditor: str
  amount: float
  good: str
  given: int
  due: int


class Reciprocity(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A ledger of who owes whom a return. Thread-safe."""

  def __init__(
      self,
      term: int = TERM,
      return_share: float = RETURN_SHARE,
      max_open: int = MAX_OPEN,
  ):
    """Initializes the ledger, with nothing owed.

    Args:
      term: The steps within which a return is due.
      return_share: The share of a gift owed back (by default).
      max_open: The most open debts a debtor keeps (the oldest are dropped).

    Raises:
      ValueError: If `term` or `max_open` is below 1, or `return_share` is
        negative.
    """
    super().__init__()
    if term < 1:
      raise ValueError(f'term must be at least 1, not {term}.')
    if max_open < 1:
      raise ValueError(f'max_open must be at least 1, not {max_open}.')
    if return_share < 0.0:
      raise ValueError(f'return_share must not be negative: {return_share}.')
    self._term = int(term)
    self._return_share = float(return_share)
    self._max_open = int(max_open)
    self._lock = threading.RLock()
    self._debts: dict[str, Debt] = {}
    self._defaulted: set[str] = set()
    self._next = 1

  def give(
      self,
      giver: str,
      receiver: str,
      amount: float,
      step: int,
      good: str = '',
      owed: float | None = None,
  ) -> Debt | None:
    """Records a gift: the receiver owes the giver a return.

    Args:
      giver: Who gave.
      receiver: Who received (and now owes).
      amount: How much was given.
      step: When.
      good: Of what.
      owed: What is owed back (`return_share` of `amount` if None).

    Returns:
      The debt, or None if nothing is owed (a gift to oneself, or none).
    """
    owe = self._return_share * float(amount) if owed is None else float(owed)
    if giver == receiver or owe <= _SETTLED:
      return None
    with self._lock:
      debt = Debt(
          id=f'd{self._next}',
          debtor=receiver,
          creditor=giver,
          amount=owe,
          good=str(good),
          given=int(step),
          due=int(step) + self._term,
      )
      self._next += 1
      self._debts[debt.id] = debt
      open_debts = self._open_of(receiver)
      for dropped in open_debts[: max(0, len(open_debts) - self._max_open)]:
        self._close(dropped.id)
      return debt

  def repay(
      self, debtor: str, creditor: str, amount: float, step: int
  ) -> float:
    """A return: settles the debtor's oldest debts to the creditor first.

    Args:
      debtor: Who returns.
      creditor: To whom.
      amount: How much.
      step: When (a debt past due may still be repaid).

    Returns:
      How much of `amount` settled debts (the rest owed nothing).
    """
    del step
    left = max(0.0, float(amount))
    with self._lock:
      for debt in self._open_of(debtor):
        if left <= _SETTLED:
          break
        if debt.creditor != creditor:
          continue
        paid = min(left, debt.amount)
        left -= paid
        if debt.amount - paid <= _SETTLED:
          self._close(debt.id)
        else:
          self._debts[debt.id] = dataclasses.replace(
              debt, amount=debt.amount - paid
          )
      return max(0.0, float(amount)) - left

  def forgive(self, creditor: str, debtor: str) -> float:
    """A creditor forgives all a debtor owes it.

    Args:
      creditor: Who forgives.
      debtor: Whom.

    Returns:
      How much was forgiven.
    """
    with self._lock:
      forgiven = 0.0
      for debt in self._open_of(debtor):
        if debt.creditor == creditor:
          forgiven += debt.amount
          self._close(debt.id)
      return forgiven

  def overdue(self, step: int) -> list[Debt]:
    """The debts past due and not yet reported, reported now (once each).

    A default stays owed (it may still be repaid) but is not reported again.

    Args:
      step: The step now.

    Returns:
      The newly defaulted debts, oldest first.
    """
    with self._lock:
      found = [
          debt
          for debt in self._ordered()
          if debt.due < step and debt.id not in self._defaulted
      ]
      self._defaulted.update(debt.id for debt in found)
      self._logging_channel({
          'Key': 'Debts',
          'Value': {'open': len(self._debts), 'defaulted': len(found)},
      })
      return found

  def settle_estate(
      self, name: str, heir: str | None = None, debts_pass: bool = False
  ) -> None:
    """One who leaves: what it was owed passes to an heir, or is cancelled.

    Args:
      name: Who leaves.
      heir: Who is owed in its place (None: its claims are cancelled).
      debts_pass: Whether what it owed passes to the heir too (else it is
        cancelled).
    """
    with self._lock:
      for debt in self._ordered():
        if debt.creditor == name:
          if heir is None or heir == debt.debtor:
            self._close(debt.id)
          else:
            self._debts[debt.id] = dataclasses.replace(debt, creditor=heir)
        elif debt.debtor == name:
          if debts_pass and heir is not None and heir != debt.creditor:
            self._debts[debt.id] = dataclasses.replace(debt, debtor=heir)
          else:
            self._close(debt.id)

  def owed_by(self, name: str) -> list[Debt]:
    """What someone owes, oldest first."""
    with self._lock:
      return self._open_of(name)

  def owed_to(self, name: str) -> list[Debt]:
    """What is owed to someone, oldest first."""
    with self._lock:
      return [d for d in self._ordered() if d.creditor == name]

  def owes(self, debtor: str, creditor: str) -> float:
    """How much one owes another in all."""
    with self._lock:
      return sum(
          d.amount for d in self._open_of(debtor) if d.creditor == creditor
      )

  def size(self) -> int:
    """How many debts are open."""
    with self._lock:
      return len(self._debts)

  def _ordered(self) -> list[Debt]:
    return sorted(self._debts.values(), key=lambda d: (d.given, _order(d.id)))

  def _open_of(self, debtor: str) -> list[Debt]:
    return [d for d in self._ordered() if d.debtor == debtor]

  def _close(self, debt_id: str) -> None:
    self._debts.pop(debt_id, None)
    self._defaulted.discard(debt_id)

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'next': self._next,
          'debts': [
              [d.id, d.debtor, d.creditor, d.amount, d.good, d.given, d.due]
              for d in self._ordered()
          ],
          'defaulted': sorted(self._defaulted, key=_order),
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    saved = state.get('debts')
    debts: Sequence[Any] = saved if isinstance(saved, (list, tuple)) else ()
    saved_defaulted = state.get('defaulted')
    defaulted: Sequence[Any] = (
        saved_defaulted if isinstance(saved_defaulted, (list, tuple)) else ()
    )
    saved_next = state.get('next')
    with self._lock:
      self._next = int(saved_next) if isinstance(saved_next, int) else 1
      self._debts = {}
      for debt_id, debtor, creditor, amount, good, given, due in debts:
        self._debts[str(debt_id)] = Debt(
            id=str(debt_id),
            debtor=str(debtor),
            creditor=str(creditor),
            amount=float(amount),
            good=str(good),
            given=int(given),
            due=int(due),
        )
      self._defaulted = {str(d) for d in defaulted if str(d) in self._debts}


def _order(debt_id: str) -> int:
  """A debt's number, for ordering ('d12' -> 12)."""
  return int(debt_id.lstrip('d') or 0)
