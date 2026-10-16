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

"""Treaties between peoples: coal, fishing quotas, runoff.

A people that knows diplomacy may propose a treaty to another: for a number
of generations each keeps one measured quantity under a cap (coal burned,
fish caught, fertiliser runoff washed to the sea). Proposals and answers are
language (the peoples' minds decide); this module keeps the books: pending
proposals, active treaties, who broke which, and the trade embargoes that
follow a breach. Like the symbiosis pacts of the daisies, but between
peoples, and with teeth. Each kind's wording and metric is data
(`civ.yaml`), so a new kind needs no code.
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield

from collections.abc import Callable, Mapping, Sequence
import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class TreatyKind:
  """What one kind of treaty caps, and how the game words it."""

  key: str
  cap: float
  requires: tuple[str, ...]  # Unlock tags or tech keys both peoples need.
  proposal: str  # Template: {proposer}, {partner}, {duration}, {cap}.
  name: str  # 'coal treaty'
  plural: str  # 'coal treaties'
  metric: str  # The `emissions` entry the cap applies to.
  unit: str  # What the cap counts, e.g. 'coal'.
  terms: str  # Template with {cap}: 'burns at most {cap} coal'.


@dataclasses.dataclass(frozen=True)
class Proposal:
  proposer: str
  partner: str
  text: str
  tick: int = 0  # When it was offered (offers lapse unanswered).
  kind: str = 'coal'  # What the treaty caps: 'coal' burned or 'fish' caught.


def treaty_key(a: str, b: str, kind: str = 'coal') -> str:
  """Key of a treaty: the pair, prefixed with its kind unless coal."""
  pair = '+'.join(sorted((a, b)))
  return pair if kind == 'coal' else f'{kind}:{pair}'


def _refusal_key(proposer: str, partner: str, kind: str) -> str:
  """Who offered whom what: a refusal holds back only that offer."""
  return f'{kind}:{proposer}>{partner}'


def parse_key(key: str) -> tuple[str, str, str]:
  """(kind, a, b) of a treaty key."""
  kind, _, pair = key.rpartition(':')
  a, b = pair.split('+')
  return kind or 'coal', a, b


class Treaties:
  """Pending proposals, active treaties, breaches and embargoes."""

  def __init__(self, config: Mapping[str, Any]):
    self.coal_cap = float(config['coal_cap'])
    # Treaty kinds: what each caps, how it is proposed and worded. Coal's cap
    # and proposal sit at the top level; its wording under `kinds.coal`.
    specs: dict[str, Mapping[str, Any]] = {
        'coal': {
            'cap': self.coal_cap,
            'proposal': config['proposal'],
            **config.get('kinds', {}).get('coal', {}),
        },
        **{k: v for k, v in config.get('kinds', {}).items() if k != 'coal'},
    }
    self.kinds: dict[str, TreatyKind] = {}
    for kind, spec in specs.items():
      name = str(spec.get('name', f'{kind} treaty'))
      self.kinds[kind] = TreatyKind(
          key=kind,
          cap=float(spec['cap']),
          requires=tuple(spec.get('requires', ['treaties'])),
          proposal=str(spec['proposal']).strip(),
          name=name,
          plural=str(spec.get('plural', f'{name}s')),
          metric=str(spec.get('metric', kind)),
          unit=str(spec.get('unit', kind)),
          terms=str(spec.get('terms', f'uses at most {{cap}} {kind}')),
      )
    self.caps = {k: v.cap for k, v in self.kinds.items()}
    self.duration = int(config['duration_ticks'])
    self.answer_latency = int(config['answer_latency_ticks'])
    self.embargo_ticks = int(config['embargo_ticks'])
    self.offer_ticks = int(config['offer_ticks'])
    # A refused offer may not be repeated to the same people for a while.
    self.refusal_ticks = int(config.get('refusal_ticks', 0))
    # (kind, proposer, partner) as text -> tick refused: the refused people
    # may still make its own offer the other way.
    self.refused: dict[str, int] = {}
    self.question = str(config['question']).strip()
    self.pending: dict[str, Proposal] = {}
    # Treaty key -> (first tick checked, last tick checked). The generation
    # after signing is grace: its coal was set to burn before the signing.
    self.active: dict[str, tuple[int, int]] = {}
    self.embargoes: dict[str, int] = {}  # Pair key -> last tick of embargo.
    self.breaches: dict[str, int] = {}  # Polity -> treaties broken.
    self.signed = 0

  def terms(self, kind: str) -> str:
    """What each signatory of a kind of treaty promises, e.g.

    'burns at most 4 coal'.
    """
    spec = self.kinds[kind]
    return spec.terms.format(cap=f'{spec.cap:g}')

  def bound(self, polity: str, kind: str = 'coal') -> list[str]:
    """The peoples a polity is bound to by treaties of a kind."""
    found = []
    for key in self.active:
      k, a, b = parse_key(key)
      if k == kind and polity in (a, b):
        found.append(b if a == polity else a)
    return sorted(found)

  def can_propose(
      self,
      proposer: str,
      partner: str,
      kind: str = 'coal',
      tick: int | None = None,
  ) -> bool:
    """Whether `proposer` may offer `partner` a treaty of `kind` now.

    Args:
      proposer: Who would offer.
      partner: To whom.
      kind: The kind of treaty.
      tick: The current tick; with it, an offer refused fewer than
        `refusal_ticks` ago may not be repeated.

    Returns:
      True if the offer is possible.
    """
    key = treaty_key(proposer, partner, kind)
    offered = self.pending.get(proposer)
    refused = self.refused.get(_refusal_key(proposer, partner, kind))
    return (
        proposer != partner
        and (
            tick is None
            or refused is None
            or tick - refused >= self.refusal_ticks
        )
        and kind in self.caps
        and key not in self.active
        and treaty_key(proposer, partner) not in self.embargoes
        and partner not in self.pending
        # Not while the partner's own offer to the proposer awaits an answer.
        and (offered is None or offered.proposer != partner)
    )

  def propose(
      self,
      proposer: str,
      partner: str,
      names: Mapping[str, str],
      tick: int = 0,
      kind: str = 'coal',
  ) -> Proposal | None:
    """Records a proposal; None if one is not possible now."""
    if not self.can_propose(proposer, partner, kind, tick):
      return None
    text = self.kinds[kind].proposal.format(
        proposer=names[proposer],
        partner=names[partner],
        duration=self.duration,
        cap=f'{self.caps[kind]:g}',
    )
    proposal = Proposal(proposer, partner, text, tick, kind)
    self.pending[partner] = proposal
    return proposal

  def pending_for(self, partner: str) -> Proposal | None:
    return self.pending.get(partner)

  def answer(
      self,
      partner: str,
      accepted: bool,
      tick: int,
      proposer: str | None = None,
      kind: str | None = None,
  ) -> Proposal | None:
    """Settles the proposal to `partner`; returns it (or None).

    Args:
      partner: Who answers.
      accepted: Whether they sign.
      tick: The current tick.
      proposer: If given, the answer only settles an offer from this people (an
        answer written to an earlier offer must not settle a newer one).
      kind: If given, it only settles an offer of this kind.

    Returns:
      The settled proposal, or None if there was none to settle.
    """
    proposal = self.pending.get(partner)
    if proposal is None or proposer not in (None, proposal.proposer):
      return None
    if kind not in (None, proposal.kind):
      return None
    del self.pending[partner]
    if not accepted:
      self.refused[_refusal_key(proposal.proposer, partner, proposal.kind)] = (
          tick
      )
    if accepted:
      first = tick + 2
      self.active[treaty_key(proposal.proposer, partner, proposal.kind)] = (
          first,
          first + self.duration - 1,
      )
      self.signed += 1
    return proposal

  def check(
      self,
      usage: Mapping[str, Any],
      tick: int,
  ) -> list[tuple[str, str]]:
    """Ends lapsed treaties and finds breaches.

    A party that used more than a treaty's cap (coal burned, fish caught)
    breaks it: the treaty ends and trade between the parties is embargoed.

    Args:
      usage: Kind -> polity -> amount this tick; a plain polity -> amount
        mapping means coal.
      tick: The current tick.

    Returns:
      (breaker, wronged) pairs.
    """
    if all(not isinstance(v, Mapping) for v in usage.values()):
      usage = {'coal': usage}
    for key, (_, last) in list(self.active.items()):
      if last < tick:
        del self.active[key]
    for key, until in list(self.embargoes.items()):
      if until <= tick:
        del self.embargoes[key]
    for partner, proposal in list(self.pending.items()):
      if tick - proposal.tick > self.offer_ticks:
        del self.pending[partner]
    broken = []
    for key, (first, _) in sorted(self.active.items()):
      if tick < first:
        continue
      kind, a, b = parse_key(key)
      used = usage.get(kind, {})
      over = [
          (breaker, wronged)
          for breaker, wronged in ((a, b), (b, a))
          if used.get(breaker, 0.0) > self.caps[kind] + 1e-9
      ]
      for breaker, wronged in over:
        broken.append((breaker, wronged))
        self.breaches[breaker] = self.breaches.get(breaker, 0) + 1
      if over:
        self.embargoes[treaty_key(a, b)] = tick + self.embargo_ticks
        del self.active[key]
    return broken

  def embargoed(self, a: str, b: str) -> bool:
    return treaty_key(a, b) in self.embargoes

  def relevant(self, knows: Callable[[str], bool]) -> list[str]:
    """Kinds worth listing: someone can do what each caps.

    A kind is listed once some people meets its requirements other than the
    treaty tag itself (coal always; fishing quotas once someone has boats;
    runoff pacts once someone knows chemistry), or once one is in force.

    Args:
      knows: Whether any people has an unlock tag or knows a tech.

    Returns:
      Kind keys, in data order.
    """
    active = {parse_key(key)[0] for key in self.active}
    return [
        kind
        for kind, spec in self.kinds.items()
        if kind in active
        or all(knows(need) for need in spec.requires if need != 'treaties')
    ]

  def payload(
      self,
      names: Mapping[str, str],
      tick: int,
      kinds: Sequence[str] | None = None,
  ) -> dict[str, Any]:
    """The public record; `kinds` limits which kinds are listed."""
    listed = list(self.kinds) if kinds is None else list(kinds)
    return {
        'treaties': [
            {
                'kind': parse_key(key)[0],
                'parties': [names[s] for s in parse_key(key)[1:]],
                'ticks_left': last - tick + 1,
            }
            for key, (_, last) in sorted(self.active.items())
        ],
        'embargoes': [
            [names[s] for s in key.split('+')] for key in sorted(self.embargoes)
        ],
        'breaches': {names[k]: v for k, v in sorted(self.breaches.items())},
        'coal_cap': self.coal_cap,
        'caps': {k: self.caps[k] for k in listed},
        'kinds': {
            k: {'plural': self.kinds[k].plural, 'unit': self.kinds[k].unit}
            for k in listed
        },
    }

  def get_state(self) -> dict[str, Any]:
    return {
        'pending': [dataclasses.asdict(p) for p in self.pending.values()],
        'active': {k: list(v) for k, v in self.active.items()},
        'embargoes': dict(self.embargoes),
        'breaches': dict(self.breaches),
        'signed': self.signed,
        'refused': dict(self.refused),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.pending = {
        p['partner']: Proposal(**p) for p in state.get('pending', [])
    }
    self.active = {
        k: (int(v[0]), int(v[1])) for k, v in state.get('active', {}).items()
    }
    self.embargoes = {k: int(v) for k, v in state.get('embargoes', {}).items()}
    self.breaches = {k: int(v) for k, v in state.get('breaches', {}).items()}
    self.signed = int(state.get('signed', 0))
    self.refused = {k: int(v) for k, v in state.get('refused', {}).items()}
