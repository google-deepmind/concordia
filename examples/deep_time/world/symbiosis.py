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

"""Symbiosis: mutualism pacts negotiated between species.

Proposals and answers are language (the species' minds decide); this module
keeps the books: pending proposals, active pacts and their expiry, and the
growth multiplier a pact gives each partner where both share land.
"""

# pylint: disable=g-complex-comprehension

from collections.abc import Mapping
import dataclasses
from typing import Any

from examples.deep_time.world import daisyworld
import numpy as np


@dataclasses.dataclass(frozen=True)
class Proposal:
  proposer: str
  partner: str
  tick: int
  text: str


def pact_key(a: str, b: str) -> str:
  return '+'.join(sorted((a, b)))


class Symbiosis:
  """Pending proposals and active pacts between species."""

  def __init__(self, config: Mapping[str, Any]):
    self.bonus = float(config['bonus'])
    self.half_saturation = float(config['half_saturation'])
    self.duration = int(config['duration_ticks'])
    self.answer_latency = int(config['answer_latency_ticks'])
    self.proposal_template = str(config['proposal']).strip()
    self.question = str(config['question']).strip()
    self.pending: dict[str, Proposal] = {}
    self.active: dict[str, int] = {}
    self.formed = 0
    self.refused = 0

  def has_pact(self, a: str, b: str) -> bool:
    return pact_key(a, b) in self.active

  def partners(self, species: str) -> list[str]:
    return sorted(
        other
        for key in self.active
        for other in key.split('+')
        if species in key.split('+') and other != species
    )

  def propose(
      self,
      proposer: str,
      partner: str,
      tick: int,
      names: Mapping[str, str],
  ) -> Proposal | None:
    """Records a proposal; None if a pact or a proposal already exists."""
    if self.has_pact(proposer, partner) or partner in self.pending:
      return None
    text = self.proposal_template.format(
        proposer=names.get(proposer, proposer),
        partner=names.get(partner, partner),
        duration=self.duration,
        bonus=self.bonus,
    )
    proposal = Proposal(proposer, partner, tick, text)
    self.pending[partner] = proposal
    return proposal

  def pending_for(self, partner: str) -> Proposal | None:
    return self.pending.get(partner)

  def answer(self, partner: str, accepted: bool, tick: int) -> Proposal | None:
    """Settles the pending proposal to `partner`; returns it (or None)."""
    proposal = self.pending.pop(partner, None)
    if proposal is None:
      return None
    if accepted:
      self.active[pact_key(proposal.proposer, partner)] = tick + self.duration
      self.formed += 1
    else:
      self.refused += 1
    return proposal

  def expire(self, tick: int) -> list[str]:
    """Drops lapsed pacts; returns their keys."""
    lapsed = [key for key, until in self.active.items() if until <= tick]
    for key in lapsed:
      del self.active[key]
    return lapsed

  def multipliers(
      self, daisies: daisyworld.Daisyworld
  ) -> dict[str, np.ndarray]:
    """Growth multiplier per species and region from active pacts."""
    result = {}
    species = set(daisies.species())
    for key in self.active:
      a, b = key.split('+')
      if a not in species or b not in species:
        continue
      for me, other in ((a, b), (b, a)):
        partner = daisies.cover_map(other)
        boost = self.bonus * partner / (partner + self.half_saturation)
        result[me] = result.get(me, np.ones_like(partner)) * (1.0 + boost)
    return result

  def payload(self, names: Mapping[str, str], tick: int) -> dict[str, Any]:
    return {
        'pacts': [
            {
                'partners': [names.get(s, s) for s in key.split('+')],
                'ticks_left': until - tick,
            }
            for key, until in sorted(self.active.items())
        ],
        'pending': [
            {
                'from': names.get(p.proposer, p.proposer),
                'to': names.get(p.partner, p.partner),
            }
            for p in self.pending.values()
        ],
        'formed': self.formed,
        'refused': self.refused,
    }

  def get_state(self) -> dict[str, Any]:
    return {
        'pending': [dataclasses.asdict(p) for p in self.pending.values()],
        'active': dict(self.active),
        'formed': self.formed,
        'refused': self.refused,
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.pending = {
        p['partner']: Proposal(**p) for p in state.get('pending', [])
    }
    self.active = {k: int(v) for k, v in state.get('active', {}).items()}
    self.formed = int(state.get('formed', 0))
    self.refused = int(state.get('refused', 0))


def parse_answer(text: str) -> bool | None:
  """True for accept, False for decline, whichever is said first."""
  lowered = text.lower()
  said = [
      (position, verdict)
      for verdict, word in ((True, 'accept'), (False, 'decline'))
      if (position := lowered.find(word)) >= 0
  ]
  return min(said)[1] if said else None
