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

"""Pacts: agreements offered, accepted, kept, broken and ended.

`Pacts` keeps the agreements between participants (or groups the game names
as participants). One offers a pact (`offer`) of a kind, with terms in the
game's words and plain data; the other may `accept` it within
`answer_within` steps, or `decline`; an offer left unanswered lapses. A
pact accepted holds for `lasts` steps (0: until ended). The game decides
what a pact asks and records a breach (`breach`); a pact breached more than
`tolerance` times is broken. A pact also ends when its term runs out
(`expire`), when a party leaves (`forget`), or by the game's word (`end`).

At most `max_pacts` pacts hold and `max_offers` offers wait (the oldest go
first), and each remembers the last `max_history` pacts ended, so an
indefinite world stays bounded.

It is state and rules only, for the GM's other components to call by key
(`DEFAULT_PACTS_COMPONENT_KEY`). It adds nothing to prompts: the game
offers the moves to propose and accept, and tells what came of them.

  pacts = pacts_lib.Pacts()
  offer = pacts.offer('Ada', 'Bo', 'aid', 'We help each other.', step=3)
  pact = pacts.accept('Bo', offer, step=4)
  pacts.breach(pact, by='Bo', step=9, why='kept full stores')  # Broken.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import threading
from typing import Any, override

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

# The key a game master registers its pacts under, by convention.
DEFAULT_PACTS_COMPONENT_KEY = '__pacts__'

# Within how many steps an offer may be accepted; the breaches a pact bears
# before it breaks; the most pacts holding, offers waiting, and pacts ended
# remembered.
ANSWER_WITHIN = 1
TOLERANCE = 0
MAX_PACTS = 50
MAX_OFFERS = 20
MAX_HISTORY = 20


@dataclasses.dataclass(frozen=True)
class Pact:
  """An agreement.

  Attributes:
    id: The pact (or the offer, while it waits).
    kind: The game's kind of pact.
    parties: Who is bound: the one who offered, then the one who accepted.
    terms: What it says, in the game's words.
    data: The game's plain data for it.
    made: The step it was made (or offered, while it waits).
    lasts: Its term in steps (0: until ended).
    until: The step its term runs out (None: until ended, or while it
      waits).
    breaches: The breaches recorded: (step, by, why).
    ended: The step it ended (None while it holds).
    why_ended: 'term', 'broken', 'left', or the game's word ('' while it
      holds).
  """

  id: str
  kind: str
  parties: tuple[str, str]
  terms: str
  data: Mapping[str, Any]
  made: int
  lasts: int = 0
  until: int | None = None
  breaches: tuple[tuple[int, str, str], ...] = ()
  ended: int | None = None
  why_ended: str = ''


class Pacts(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Agreements between participants. Thread-safe."""

  def __init__(
      self,
      answer_within: int = ANSWER_WITHIN,
      tolerance: int = TOLERANCE,
      max_pacts: int = MAX_PACTS,
      max_offers: int = MAX_OFFERS,
      max_history: int = MAX_HISTORY,
  ):
    """Initializes the pacts, with none made or offered.

    Args:
      answer_within: Within how many steps an offer may be accepted.
      tolerance: The breaches a pact bears before it breaks.
      max_pacts: The most pacts holding (the oldest end first).
      max_offers: The most offers waiting (the oldest lapse first).
      max_history: The most pacts ended remembered.

    Raises:
      ValueError: If a count is out of range.
    """
    super().__init__()
    for label, value, least in (
        ('answer_within', answer_within, 1),
        ('tolerance', tolerance, 0),
        ('max_pacts', max_pacts, 1),
        ('max_offers', max_offers, 1),
        ('max_history', max_history, 1),
    ):
      if value < least:
        raise ValueError(f'{label} must be at least {least}, not {value}.')
    self._answer_within = int(answer_within)
    self._tolerance = int(tolerance)
    self._max_pacts = int(max_pacts)
    self._max_offers = int(max_offers)
    self._max_history = int(max_history)
    self._lock = threading.RLock()
    self._offers: dict[str, Pact] = {}
    self._pacts: dict[str, Pact] = {}
    self._history: list[Pact] = []
    self._next = 1

  def offer(
      self,
      proposer: str,
      partner: str,
      kind: str,
      terms: str,
      step: int,
      lasts: int = 0,
      data: Mapping[str, Any] | None = None,
  ) -> str | None:
    """Offers a pact (the same kind between the same two is one offer).

    Args:
      proposer: Who offers.
      partner: To whom.
      kind: The game's kind of pact.
      terms: What it says.
      step: When; it may be accepted from the next step.
      lasts: Its term in steps once made (0: until ended).
      data: The game's plain data for it.

    Returns:
      The offer's id, or None if the two already hold a pact of the kind,
      or one would offer itself.
    """
    if proposer == partner:
      return None
    with self._lock:
      if self.between(proposer, partner, kind):
        return None
      for waiting in self._offers.values():
        if waiting.kind == kind and set(waiting.parties) == {proposer, partner}:
          return waiting.id
      offer_id = f'p{self._next}'
      self._next += 1
      self._offers[offer_id] = Pact(
          id=offer_id,
          kind=str(kind),
          parties=(str(proposer), str(partner)),
          terms=str(terms),
          data=dict(data or {}),
          made=int(step),
          lasts=int(lasts),
      )
      while len(self._offers) > self._max_offers:
        del self._offers[min(self._offers, key=_order)]
      return offer_id

  def offers_to(self, name: str, step: int) -> list[Pact]:
    """The offers someone may accept at a step, oldest first."""
    with self._lock:
      return [
          o
          for o in self._ordered(self._offers)
          if o.parties[1] == name and self._answerable(o, step)
      ]

  def accept(self, partner: str, offer_id: str, step: int) -> str | None:
    """Accepts an offer: the pact holds from now.

    Args:
      partner: Who accepts (the one offered to).
      offer_id: Which offer.
      step: When (an offer not yet answerable, or lapsed, is not accepted).

    Returns:
      The pact's id (the offer's), or None if it could not be accepted.
    """
    with self._lock:
      found = self._offers.get(offer_id)
      if found is None or found.parties[1] != partner:
        return None
      if not self._answerable(found, step):
        return None
      del self._offers[offer_id]
      pact = dataclasses.replace(
          found,
          made=int(step),
          until=step + found.lasts if found.lasts else None,
      )
      self._pacts[pact.id] = pact
      while len(self._pacts) > self._max_pacts:
        self._close(min(self._pacts, key=_order), step, 'crowded out')
      self._logging_channel({
          'Key': 'Pact made',
          'Value': {'kind': pact.kind, 'parties': list(pact.parties)},
      })
      return pact.id

  def decline(self, partner: str, offer_id: str) -> bool:
    """Declines an offer; whether there was one to decline."""
    with self._lock:
      found = self._offers.get(offer_id)
      if found is None or found.parties[1] != partner:
        return False
      del self._offers[offer_id]
      return True

  def breach(
      self, pact_id: str, by: str, step: int, why: str = ''
  ) -> Pact | None:
    """Records a breach of a pact; past `tolerance`, the pact is broken.

    Args:
      pact_id: Which pact.
      by: Who breached it.
      step: When.
      why: What was done (the game's words).

    Returns:
      The pact after the breach (ended if broken), or None if it does not
      hold.
    """
    with self._lock:
      found = self._pacts.get(pact_id)
      if found is None:
        return None
      found = dataclasses.replace(
          found, breaches=found.breaches + ((int(step), str(by), str(why)),)
      )
      self._pacts[pact_id] = found
      if len(found.breaches) > self._tolerance:
        return self._close(pact_id, step, 'broken')
      return found

  def end(self, pact_id: str, step: int, why: str = '') -> Pact | None:
    """Ends a pact by the game's word; the pact ended, or None."""
    with self._lock:
      if pact_id not in self._pacts:
        return None
      return self._close(pact_id, step, why)

  def expire(self, step: int) -> list[Pact]:
    """Pacts whose term ran out end, and unanswered offers lapse.

    Returns:
      The pacts ended.
    """
    with self._lock:
      for offer_id in [
          o.id
          for o in self._offers.values()
          if o.made + self._answer_within < step
      ]:
        del self._offers[offer_id]
      return [
          self._close(p.id, step, 'term')
          for p in self._ordered(self._pacts)
          if p.until is not None and p.until <= step
      ]

  def forget(self, name: str, step: int) -> list[Pact]:
    """One who leaves: its pacts end, and its offers are withdrawn.

    Returns:
      The pacts ended.
    """
    with self._lock:
      for offer_id in [
          o.id for o in self._offers.values() if name in o.parties
      ]:
        del self._offers[offer_id]
      return [
          self._close(p.id, step, 'left')
          for p in self._ordered(self._pacts)
          if name in p.parties
      ]

  def pacts_of(self, name: str, kind: str | None = None) -> list[Pact]:
    """The pacts someone holds (of a kind, if given), oldest first."""
    with self._lock:
      return [
          p
          for p in self._ordered(self._pacts)
          if name in p.parties and (kind is None or p.kind == kind)
      ]

  def between(self, a: str, b: str, kind: str | None = None) -> list[Pact]:
    """The pacts two hold together (of a kind, if given)."""
    with self._lock:
      return [p for p in self.pacts_of(a, kind) if b in p.parties]

  def history(self) -> list[Pact]:
    """The pacts ended lately, the latest last."""
    with self._lock:
      return list(self._history)

  def size(self) -> int:
    """How many pacts hold."""
    with self._lock:
      return len(self._pacts)

  def _answerable(self, offer: Pact, step: int) -> bool:
    return offer.made < step <= offer.made + self._answer_within

  def _close(self, pact_id: str, step: int, why: str) -> Pact:
    ended = dataclasses.replace(
        self._pacts.pop(pact_id), ended=int(step), why_ended=str(why)
    )
    self._history.append(ended)
    del self._history[: max(0, len(self._history) - self._max_history)]
    return ended

  @staticmethod
  def _ordered(records: Mapping[str, Pact]) -> list[Pact]:
    return [records[k] for k in sorted(records, key=_order)]

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'next': self._next,
          'offers': [_record(p) for p in self._ordered(self._offers)],
          'pacts': [_record(p) for p in self._ordered(self._pacts)],
          'history': [_record(p) for p in self._history],
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    def read(key: str) -> list[Pact]:
      saved = state.get(key)
      records: Sequence[Any] = saved if isinstance(saved, (list, tuple)) else ()
      return [_pact(r) for r in records]

    saved_next = state.get('next')
    with self._lock:
      self._next = int(saved_next) if isinstance(saved_next, int) else 1
      self._offers = {p.id: p for p in read('offers')}
      self._pacts = {p.id: p for p in read('pacts')}
      self._history = read('history')


def _record(pact: Pact) -> dict[str, Any]:
  return {
      'id': pact.id,
      'kind': pact.kind,
      'parties': list(pact.parties),
      'terms': pact.terms,
      'data': dict(pact.data),
      'made': pact.made,
      'lasts': pact.lasts,
      'until': pact.until,
      'breaches': [list(b) for b in pact.breaches],
      'ended': pact.ended,
      'why_ended': pact.why_ended,
  }


def _pact(record: Mapping[str, Any]) -> Pact:
  until = record.get('until')
  ended = record.get('ended')
  a, b = record['parties']
  return Pact(
      id=str(record['id']),
      kind=str(record.get('kind', '')),
      parties=(str(a), str(b)),
      terms=str(record.get('terms', '')),
      data=dict(record.get('data') or {}),
      made=int(record.get('made', 0)),
      lasts=int(record.get('lasts', 0)),
      until=None if until is None else int(until),
      breaches=tuple(
          (int(s), str(by), str(why))
          for s, by, why in record.get('breaches') or ()
      ),
      ended=None if ended is None else int(ended),
      why_ended=str(record.get('why_ended', '')),
  )


def _order(record_id: str) -> int:
  """A pact's number, for ordering ('p12' -> 12)."""
  return int(record_id.lstrip('p') or 0)
