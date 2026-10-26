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

"""The public record of customs: every variant that has entered the world.

A custom is a cultural variant: a norm, a value, a belief, a skill, a habit or
a learning bias, with a domain (its locus) and a text (its thick content).
What a person holds is a component of that person's own entity, built from a
custom's spec; this record is the world's, kept by the World GM. Every custom
enters the world through the GM (a founder's, a mutation in transmission, an
innovation), so the GM knows each variant's content by its id without ever
reading anyone's mind: people name the ids they pass on.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

import numpy as np

# The loci of culture: recombination and capacity work per domain.
DOMAINS = (
    'kinship',
    'food',
    'sharing',
    'authority',
    'sacred',
    'craft',
    'learning',
    'identity',
    'language',
)

# The kinds of custom, each built as its own kind of component.
KINDS = ('norm', 'value', 'belief', 'skill', 'habit', 'bias')


@dataclasses.dataclass(frozen=True)
class Custom:
  """One cultural variant, as the world records it."""

  id: str
  kind: str
  domain: str
  text: str
  # Kind-specific settings, JSON-safe (a skill's level, a habit's rule...).
  params: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  # The variant it was copied from with an error, if any.
  parent: str = ''
  # How it entered the world: 'founder', 'mutation', 'innovation', ...
  origin: str = 'founder'
  year: int = 0

  def spec(self) -> dict[str, Any]:
    """The JSON spec a person's component is built from."""
    return {
        'id': self.id,
        'kind': self.kind,
        'domain': self.domain,
        'text': self.text,
        'params': dict(self.params),
    }

  def to_dict(self) -> dict[str, Any]:
    return {
        **self.spec(),
        'parent': self.parent,
        'origin': self.origin,
        'year': self.year,
    }

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> 'Custom':
    return cls(
        id=str(data['id']),
        kind=str(data['kind']),
        domain=str(data['domain']),
        text=str(data['text']),
        params=dict(data.get('params', {})),
        parent=str(data.get('parent', '')),
        origin=str(data.get('origin', 'founder')),
        year=int(data.get('year', 0)),
    )


class Culture:
  """The world's record of customs, by id."""

  def __init__(self):
    self._customs: dict[str, Custom] = {}
    self._next = 1

  def __contains__(self, custom_id: str) -> bool:
    return custom_id in self._customs

  def __len__(self) -> int:
    return len(self._customs)

  def get(self, custom_id: str) -> Custom:
    return self._customs[custom_id]

  def add(
      self,
      kind: str,
      domain: str,
      text: str,
      params: Mapping[str, Any] | None = None,
      *,
      parent: str = '',
      origin: str = 'founder',
      year: int = 0,
  ) -> Custom:
    """Records a new variant and returns it."""
    if kind not in KINDS:
      raise ValueError(f'Unknown kind of custom: {kind}')
    if domain not in DOMAINS:
      raise ValueError(f'Unknown domain: {domain}')
    custom = Custom(
        id=f'c{self._next}',
        kind=kind,
        domain=domain,
        text=' '.join(text.split()),
        params=dict(params or {}),
        parent=parent,
        origin=origin,
        year=year,
    )
    self._next += 1
    self._customs[custom.id] = custom
    return custom

  def copy(
      self, custom_id: str, rng: np.random.Generator, mutation: float, year: int
  ) -> Custom:
    """A copy of a variant, as transmission makes it.

    With probability `mutation` the copy errs: a skill is learned a level
    lower or higher, a text loses or doubles its emphasis. An erring copy is a
    new variant (its parent recorded); a faithful one is the same variant.

    Args:
      custom_id: The variant copied.
      rng: The world's random stream for this transmission.
      mutation: The chance of a copying error.
      year: Now.

    Returns:
      The variant the receiver gets.
    """
    custom = self.get(custom_id)
    if rng.random() >= mutation:
      return custom
    params = dict(custom.params)
    text = custom.text
    if custom.kind == 'skill':
      level = int(params.get('level', 1))
      params['level'] = max(1, level + (1 if rng.random() < 0.3 else -1))
    else:
      words = text.rstrip('.').split()
      if len(words) > 4 and rng.random() < 0.5:
        del words[int(rng.integers(1, len(words)))]  # Something is lost.
        text = ' '.join(words) + '.'
      else:
        text = text.rstrip('.') + ', always.'  # Something is stressed.
    return self.add(
        custom.kind,
        custom.domain,
        text,
        params,
        parent=custom.id,
        origin='mutation',
        year=year,
    )

  def recombine(
      self,
      mother: Sequence[str],
      father: Sequence[str],
      rng: np.random.Generator,
  ) -> list[str]:
    """A child's variants from two parents', domain by domain.

    For each domain either parent holds customs in, the child takes that
    domain's customs from one parent, chosen at random (from the only parent
    who has any, if just one does).

    Args:
      mother: The ids one parent passes on.
      father: The ids the other passes on.
      rng: The world's random stream for this birth.

    Returns:
      The ids the child starts with (before copying errors).
    """
    by_parent = []
    for ids in (mother, father):
      domains: dict[str, list[str]] = {}
      for custom_id in ids:
        if custom_id in self._customs:
          domains.setdefault(self.get(custom_id).domain, []).append(custom_id)
      by_parent.append(domains)
    child = []
    for domain in DOMAINS:
      offers = [d[domain] for d in by_parent if d.get(domain)]
      if offers:
        child.extend(offers[int(rng.integers(len(offers)))])
    return child

  def get_state(self) -> dict[str, Any]:
    return {
        'next': self._next,
        'customs': [c.to_dict() for c in self._customs.values()],
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self._next = int(state['next'])
    self._customs = {
        str(c['id']): Custom.from_dict(c) for c in state.get('customs', [])
    }
