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

"""An assembly: motions proposed, voted on, and passed into law.

`Assembly` keeps a body's motions and the laws it has passed. A member
proposes a motion (`propose`); it stays open for `open_for` steps, while
members vote for or against it (`vote`, once each, a later vote replacing
an earlier). When its time is up the game closes it (`close`) with the
members eligible then: it passes if at least `quorum` of them voted and the
votes meet the `rule`:

- `majority`: more for than against;
- `two_thirds`: at least two thirds of the votes for;
- `consensus`: at least one for, none against.

A motion passed is a law in force for `term` steps (0: until repealed), and
the game decides what a law does. At most `max_laws` are in force (the
oldest lapse first) and `max_motions` open (a new one beyond is refused), so
an assembly that runs indefinitely stays bounded.

It is state and rules only, for the GM's other components to call by key
(`DEFAULT_ASSEMBLY_COMPONENT_KEY`). It adds nothing to prompts: the game
offers the moves to propose and vote, and tells what was decided.

  council = assembly.Assembly(rule='majority', quorum=0.5)
  motion = council.propose('Ada', 'We spare the fields of others.', step=3)
  council.vote('Bo', motion, True, step=4)
  decided = council.close(step=4, eligible=['Ada', 'Bo', 'Cy'])
"""

from collections.abc import Collection, Mapping, Sequence
import dataclasses
import threading
from typing import Any, override

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

# The key a game master registers its assembly under, by convention.
DEFAULT_ASSEMBLY_COMPONENT_KEY = '__assembly__'

MAJORITY = 'majority'
TWO_THIRDS = 'two_thirds'
CONSENSUS = 'consensus'
RULES = (MAJORITY, TWO_THIRDS, CONSENSUS)

# For how many steps a motion stays open; the share of the eligible who
# must vote; for how many steps a law stays in force (0: until repealed);
# the most motions open, and laws in force.
OPEN_FOR = 1
QUORUM = 0.0
TERM = 0
MAX_MOTIONS = 5
MAX_LAWS = 10


@dataclasses.dataclass(frozen=True)
class Decision:
  """A motion closed.

  Attributes:
    motion_id: The motion.
    text: What it proposed.
    kind: The game's kind of motion ('' if none).
    proposer: Who proposed it.
    passed: Whether it passed.
    votes_for: Eligible members who voted for it.
    votes_against: Eligible members who voted against it.
    eligible: How many were eligible.
    reason: Why it failed ('' if it passed): 'quorum' or the rule.
  """

  motion_id: str
  text: str
  kind: str
  proposer: str
  passed: bool
  votes_for: Sequence[str]
  votes_against: Sequence[str]
  eligible: int
  reason: str = ''


@dataclasses.dataclass
class _Motion:
  id: str
  text: str
  kind: str
  proposer: str
  proposed: int
  data: dict[str, Any]
  votes: dict[str, bool] = dataclasses.field(default_factory=dict)


class Assembly(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A body's motions, votes and laws. Thread-safe."""

  def __init__(
      self,
      rule: str = MAJORITY,
      quorum: float = QUORUM,
      open_for: int = OPEN_FOR,
      term: int = TERM,
      max_motions: int = MAX_MOTIONS,
      max_laws: int = MAX_LAWS,
  ):
    """Initializes the assembly, with no motions and no laws.

    Args:
      rule: How votes decide: `majority`, `two_thirds` or `consensus`.
      quorum: The share of the eligible who must vote, in [0, 1].
      open_for: For how many steps a motion stays open.
      term: For how many steps a law stays in force (0: until repealed).
      max_motions: The most motions open at once.
      max_laws: The most laws in force (the oldest lapse first).

    Raises:
      ValueError: If the rule is unknown, the quorum outside [0, 1], or a
        count out of range.
    """
    super().__init__()
    if rule not in RULES:
      raise ValueError(f'rule must be one of {RULES}, not {rule!r}.')
    if not 0.0 <= quorum <= 1.0:
      raise ValueError(f'quorum must be in [0, 1], not {quorum}.')
    for label, value, least in (
        ('open_for', open_for, 1),
        ('term', term, 0),
        ('max_motions', max_motions, 1),
        ('max_laws', max_laws, 1),
    ):
      if value < least:
        raise ValueError(f'{label} must be at least {least}, not {value}.')
    self._rule = rule
    self._quorum = float(quorum)
    self._open_for = int(open_for)
    self._term = int(term)
    self._max_motions = int(max_motions)
    self._max_laws = int(max_laws)
    self._lock = threading.RLock()
    self._motions: dict[str, _Motion] = {}
    self._laws: dict[str, dict[str, Any]] = {}
    self._next = 1

  def propose(
      self,
      proposer: str,
      text: str,
      step: int,
      kind: str = '',
      data: Mapping[str, Any] | None = None,
  ) -> str | None:
    """Proposes a motion (the same text open twice is one motion).

    Args:
      proposer: Who proposes it (its vote is not cast for it).
      text: What it proposes.
      step: When; it is open from the next step for `open_for` steps.
      kind: The game's kind of motion.
      data: The game's plain data for it (kept with the law if it passes).

    Returns:
      The motion's id (an open one's, if the same text is open), or None if
      `max_motions` are open already.
    """
    with self._lock:
      for motion in self._motions.values():
        if motion.text == text:
          return motion.id
      if len(self._motions) >= self._max_motions:
        return None
      motion_id = f'm{self._next}'
      self._next += 1
      self._motions[motion_id] = _Motion(
          id=motion_id,
          text=str(text),
          kind=str(kind),
          proposer=str(proposer),
          proposed=int(step),
          data=dict(data or {}),
      )
      return motion_id

  def vote(self, voter: str, motion_id: str, yes: bool, step: int) -> bool:
    """Casts (or changes) a member's vote on an open motion.

    Args:
      voter: Who votes.
      motion_id: On which motion.
      yes: For (True) or against.
      step: When (a motion not yet open, or closed, takes no votes).

    Returns:
      Whether the vote was cast.
    """
    with self._lock:
      motion = self._motions.get(motion_id)
      if motion is None or not self._is_open(motion, step):
        return False
      motion.votes[voter] = bool(yes)
      return True

  def open_motions(self, step: int) -> list[dict[str, Any]]:
    """The motions open for votes at a step, oldest first.

    Args:
      step: The step.

    Returns:
      Each as {id, text, kind, proposer, proposed, data, for, against}.
    """
    with self._lock:
      return [
          {
              'id': m.id,
              'text': m.text,
              'kind': m.kind,
              'proposer': m.proposer,
              'proposed': m.proposed,
              'data': dict(m.data),
              'for': sum(1 for v in m.votes.values() if v),
              'against': sum(1 for v in m.votes.values() if not v),
          }
          for m in self._ordered()
          if self._is_open(m, step)
      ]

  def close(self, step: int, eligible: Collection[str]) -> list[Decision]:
    """Closes the motions whose time is up, and lets lapsed laws go.

    Only the votes of those eligible now count.

    Args:
      step: The step now (motions open through it close).
      eligible: Who may vote now.

    Returns:
      The decisions, oldest motion first.
    """
    members = set(eligible)
    with self._lock:
      decided = []
      for motion in self._ordered():
        if motion.proposed + self._open_for > step:
          continue
        votes_for = sorted(
            v for v, yes in motion.votes.items() if yes and v in members
        )
        against = sorted(
            v for v, yes in motion.votes.items() if not yes and v in members
        )
        reason = self._fails(len(votes_for), len(against), len(members))
        decided.append(
            Decision(
                motion_id=motion.id,
                text=motion.text,
                kind=motion.kind,
                proposer=motion.proposer,
                passed=not reason,
                votes_for=votes_for,
                votes_against=against,
                eligible=len(members),
                reason=reason,
            )
        )
        del self._motions[motion.id]
        if not reason:
          self._enact(motion, step)
      for law_id in [
          law_id
          for law_id, law in self._laws.items()
          if law['expires'] is not None and law['expires'] <= step
      ]:
        del self._laws[law_id]
      self._logging_channel({
          'Key': 'Assembly',
          'Value': {
              'decided': len(decided),
              'passed': sum(d.passed for d in decided),
              'laws': len(self._laws),
          },
      })
      return decided

  def in_force(self) -> list[dict[str, Any]]:
    """The laws in force, oldest first.

    Returns:
      Each as {id, text, kind, data, proposer, passed, expires}.
    """
    with self._lock:
      return [
          dict(self._laws[k], data=dict(self._laws[k]['data']))
          for k in sorted(self._laws, key=_order)
      ]

  def repeal(self, law_id: str) -> bool:
    """Repeals a law in force; whether there was one."""
    with self._lock:
      return self._laws.pop(law_id, None) is not None

  def forget(self, name: str) -> None:
    """Forgets a member's votes (one who has left); its motions stand."""
    with self._lock:
      for motion in self._motions.values():
        motion.votes.pop(name, None)

  def _fails(self, votes_for: int, against: int, eligible: int) -> str:
    voted = votes_for + against
    if not voted or voted < self._quorum * eligible:
      return 'quorum'
    if self._rule == MAJORITY and votes_for > against:
      return ''
    if self._rule == TWO_THIRDS and 3 * votes_for >= 2 * voted:
      return ''
    if self._rule == CONSENSUS and votes_for and not against:
      return ''
    return self._rule

  def _enact(self, motion: _Motion, step: int) -> None:
    law_id = f'law{_order(motion.id)}'
    self._laws[law_id] = {
        'id': law_id,
        'text': motion.text,
        'kind': motion.kind,
        'data': dict(motion.data),
        'proposer': motion.proposer,
        'passed': step,
        'expires': step + self._term if self._term else None,
    }
    while len(self._laws) > self._max_laws:
      del self._laws[min(self._laws, key=_order)]

  def _is_open(self, motion: _Motion, step: int) -> bool:
    return motion.proposed < step <= motion.proposed + self._open_for

  def _ordered(self) -> list[_Motion]:
    return sorted(self._motions.values(), key=lambda m: _order(m.id))

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'next': self._next,
          'motions': [
              {**dataclasses.asdict(m), 'votes': dict(sorted(m.votes.items()))}
              for m in self._ordered()
          ],
          'laws': self.in_force(),
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    saved_motions = state.get('motions')
    motions: Sequence[Any] = (
        saved_motions if isinstance(saved_motions, (list, tuple)) else ()
    )
    saved_laws = state.get('laws')
    laws: Sequence[Any] = (
        saved_laws if isinstance(saved_laws, (list, tuple)) else ()
    )
    saved_next = state.get('next')
    restored_motions: dict[str, _Motion] = {}
    for record in motions:
      try:
        motion = _Motion(
            id=str(record['id']),
            text=str(record.get('text', '')),
            kind=str(record.get('kind', '')),
            proposer=str(record.get('proposer', '')),
            proposed=int(record.get('proposed', 0)),
            data=dict(record.get('data') or {}),
            votes={
                str(k): bool(v) for k, v in (record.get('votes') or {}).items()
            },
        )
      except (AttributeError, KeyError, TypeError, ValueError):
        continue  # A malformed record is skipped.
      restored_motions[motion.id] = motion
    restored_laws: dict[str, dict[str, Any]] = {}
    for law in laws:
      try:
        expires = law.get('expires')
        restored_laws[str(law['id'])] = {
            'id': str(law['id']),
            'text': str(law.get('text', '')),
            'kind': str(law.get('kind', '')),
            'data': dict(law.get('data') or {}),
            'proposer': str(law.get('proposer', '')),
            'passed': int(law.get('passed', 0)),
            'expires': None if expires is None else int(expires),
        }
      except (AttributeError, KeyError, TypeError, ValueError):
        continue
    # Never below the ids restored (a law keeps its motion's number).
    next_id = max(
        int(saved_next) if isinstance(saved_next, int) else 1,
        max(
            (_order(k) + 1 for k in (*restored_motions, *restored_laws)),
            default=1,
        ),
    )
    with self._lock:
      self._motions, self._laws = restored_motions, restored_laws
      self._next = next_id


def _order(record_id: str) -> int:
  """A motion's or law's number, for ordering ('m12', 'law12' -> 12)."""
  digits = ''.join(c for c in record_id if c.isdigit())
  return int(digits or 0)
