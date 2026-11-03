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

"""Customs an entity holds as components of its own, and a Learner.

A custom is one cultural variant (a norm, a belief, a skill, a habit) that an
entity holds as a context component of its own (`Custom`): its context tells
the entity's decision logic what the custom says. The entity's `Learner`
takes up customs that a game master offers in the entity's observations,
keeps them within capacity, and adds and removes `Custom` components through
`EntityAgent.add_component` and `remove_component`, like an organism taking
up and losing plasmids.

The game master offers a custom with one line of an observation
(`offer_line`):

  [custom id=c7 kind=skill domain=food via=observation from=Ada] Smoking
  fish over green alder. |params={"level": 2}

Kinds, domains and channels (`via`) are the game's own words. A kind of
custom in a domain is a locus: a new custom replaces the one held there. A
game sets each channel's chance of uptake (`uptake`), and may subclass the
`Learner` to change an offer's chance (`chance`) or to refuse a custom
(`accepts`). Draws are seeded by the entity, the offer and the observation,
so uptake does not depend on thread timing.

Give the entity `custom_factory` as its component factory, so that a restored
entity rebuilds the customs it had taken up:

  learner = customs.Learner(uptake={'observation': 0.5})
  agent = entity_agent.EntityAgent(
      'Ada',
      act_component=...,
      context_components={'learner': learner, ...},
      component_factory=customs.custom_factory,
  )
"""

from collections.abc import Mapping, Sequence
import dataclasses
import json
import re
import threading
from typing import Any, override
import zlib

from concordia.agents import entity_agent
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import numpy as np

# The prefix of the entity's key for a custom's component.
COMPONENT_PREFIX = 'custom '
# How a custom held from birth came (the default of `Learner.inherit`).
BIRTH = 'birth'

OFFER = re.compile(
    r'\[custom id=(?P<id>[^ \]]+) kind=(?P<kind>[^ \]]+)'
    r' domain=(?P<domain>[^ \]]+) via=(?P<via>[^ \]]+)'
    r'(?: from=(?P<source>[^ \]]+))?\]'
    r' (?P<text>[^\n]*?)(?: \|params=(?P<params>\{[^\n]*\}))?$',
    re.MULTILINE,
)


@dataclasses.dataclass(frozen=True)
class Offer:
  """A custom offered in an observation, and how it came."""

  spec: Mapping[str, Any]
  via: str
  source: str = ''

  @property
  def id(self) -> str:
    return str(self.spec['id'])


def offer_line(spec: Mapping[str, Any], via: str, source: str = '') -> str:
  """The line of an observation that offers a custom.

  Args:
    spec: The custom: its `id`, `kind`, `domain`, `text` and optional
      `params` (a JSON object).
    via: The channel it comes by (e.g. 'observation').
    source: Optional. Whom it comes from.

  Returns:
    The line (`OFFER` reads it back).
  """
  head = (
      f'[custom id={spec["id"]} kind={spec["kind"]}'
      f' domain={spec["domain"]} via={via}'
      + (f' from={source}' if source else '')
      + ']'
  )
  text = ' '.join(str(spec['text']).split())
  params = spec.get('params')
  tail = f' |params={json.dumps(params, sort_keys=True)}' if params else ''
  return f'{head} {text}{tail}'


def parse_offers(observation: str) -> list[Offer]:
  """The customs offered in an observation, in order.

  An offer whose params are not a JSON object is passed over.

  Args:
    observation: What the entity observes.

  Returns:
    The offers.
  """
  offers = []
  for match in OFFER.finditer(observation):
    params: Any = {}
    if match.group('params'):
      try:
        params = json.loads(match.group('params'))
      except json.JSONDecodeError:
        continue
      if not isinstance(params, dict):
        continue
    offers.append(
        Offer(
            spec={
                'id': match.group('id'),
                'kind': match.group('kind'),
                'domain': match.group('domain'),
                'text': match.group('text').strip(),
                'params': params,
            },
            via=match.group('via'),
            source=match.group('source') or '',
        )
    )
  return offers


def component_name(custom_id: str) -> str:
  """The entity's key for the component of a custom."""
  return f'{COMPONENT_PREFIX}{custom_id}'


class Custom(entity_component.ContextComponent):
  """A custom an entity holds: its context is what the custom says."""

  def __init__(self, spec: Mapping[str, Any]):
    """Initializes the custom.

    Args:
      spec: Its `id`, `kind`, `domain`, `text` and optional `params`.
    """
    super().__init__()
    self.spec = dict(spec)

  @property
  def id(self) -> str:
    return str(self.spec['id'])

  @property
  def kind(self) -> str:
    return str(self.spec['kind'])

  @property
  def domain(self) -> str:
    return str(self.spec['domain'])

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    level = self.spec.get('params', {}).get('level')
    detail = f' (level {level})' if level is not None else ''
    return (
        f'A {self.kind} of ours ({self.domain}) [{self.id}]:'
        f' {self.spec["text"]}{detail}'
    )

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


def custom_factory(
    name: str, spec: entity_component.ComponentState
) -> entity_component.ContextComponent:
  """An entity's component factory: a custom from the spec it was added with.

  Args:
    name: The component's key (unused: the spec names the custom).
    spec: The custom's spec.

  Returns:
    The custom's component.
  """
  del name
  return Custom(spec)


class Learner(entity_component.ContextComponent):
  """Takes up the customs offered to its entity, and keeps them in bounds.

  An offer is taken up with the chance its channel has (`uptake`), changed
  by `chance`. Each further offer of the same custom in one observation is
  another chance, so what many do spreads faster. A custom `accepts` refuses
  is passed over: by default, one with a `level` param no better than the
  level of the custom held at its locus.

  A kind of custom in a domain is a locus, like a plasmid's incompatibility
  group: a new custom replaces the one the entity holds there. Past
  `per_domain` customs of a domain, or `capacity` in all, the oldest goes.
  The Learner remembers how each custom came, and from whom.
  """

  def __init__(
      self,
      uptake: Mapping[str, float],
      seed: int = 0,
      capacity: int = 10,
      per_domain: int | None = None,
  ):
    """Initializes the Learner.

    Args:
      uptake: The chance of taking up an offer, by channel (0 if absent).
      seed: Seeds the draws, with the entity's name, the offer and the
        observation.
      capacity: The most customs held.
      per_domain: The most customs of one domain held (no bound if None).
    """
    super().__init__()
    self._uptake = dict(uptake)
    self._seed = seed
    self._capacity = capacity
    self._per_domain = per_domain
    self._lock = threading.Lock()
    # What the entity holds, oldest first: id -> kind, domain, level, via,
    # from (including customs whose components join when the current call
    # ends).
    self._held: dict[str, dict[str, Any]] = {}

  def held(self) -> list[str]:
    """The ids of the customs held, oldest first."""
    with self._lock:
      return list(self._held)

  def came(self, custom_id: str) -> Mapping[str, str] | None:
    """How a custom held came, and from whom (None if it is not held)."""
    with self._lock:
      held = self._held.get(custom_id)
      return (
          None if held is None else {'via': held['via'], 'from': held['from']}
      )

  def inherit(
      self, spec: Mapping[str, Any], via: str = BIRTH, source: str = ''
  ) -> None:
    """Takes a custom the entity is born or comes with (no draw).

    Args:
      spec: The custom.
      via: How it came.
      source: Optional. From whom.
    """
    with self._lock:
      self._take(Offer(spec=dict(spec), via=via, source=source))

  def chance(self, offer: Offer) -> float:
    """The chance of taking up an offer: its channel's (override to change).

    Args:
      offer: The offer.

    Returns:
      A chance from 0 to 1.
    """
    return float(self._uptake.get(offer.via, 0.0))

  def accepts(
      self, spec: Mapping[str, Any], locus: Sequence[Mapping[str, Any]]
  ) -> bool:
    """Whether a custom is wanted at all (override to change).

    By default, a custom with a `level` param is wanted only if it is better
    than every custom held at its locus.

    Args:
      spec: The custom offered.
      locus: What is held at its locus (kind and domain).

    Returns:
      Whether to take it up, if the draw allows.
    """
    level = spec.get('params', {}).get('level')
    if level is None:
      return True
    return all(int(level) > int(held['level'] or 0) for held in locus)

  @override
  def pre_observe(self, observation: str) -> str:
    offers: dict[str, list[Offer]] = {}
    for offer in parse_offers(observation):
      offers.setdefault(offer.id, []).append(offer)
    name = self.get_entity().name
    with self._lock:
      for custom_id, seen in offers.items():
        if custom_id in self._held:
          continue
        chances = [min(1.0, max(0.0, self.chance(o))) for o in seen]
        missed = float(np.prod([1.0 - c for c in chances]))
        draw = np.random.default_rng([
            self._seed,
            zlib.crc32(f'{name}/{custom_id}/{observation}'.encode()),
        ]).random()
        if draw < 1.0 - missed:
          self._take(seen[int(np.argmax(chances))])
    return ''

  def _take(self, offer: Offer) -> None:
    """Holds a custom, making room for it (in the lock)."""
    entity = self.get_entity()
    if not isinstance(entity, entity_agent.EntityAgent):
      raise TypeError('A Learner belongs to an EntityAgent.')
    spec = offer.spec
    if offer.id in self._held:
      return
    locus = [
        c
        for c, held in self._held.items()
        if (held['kind'], held['domain']) == (spec['kind'], spec['domain'])
    ]
    if not self.accepts(spec, [self._held[c] for c in locus]):
      return
    if locus:
      gone = locus[0]
    elif self._per_domain is not None and (
        sum(h['domain'] == spec['domain'] for h in self._held.values())
        >= self._per_domain
    ):
      gone = next(
          c for c, h in self._held.items() if h['domain'] == spec['domain']
      )
    elif len(self._held) >= self._capacity:
      gone = next(iter(self._held))
    else:
      gone = None
    if gone is not None:
      entity.remove_component(component_name(gone))
      del self._held[gone]
    entity.add_component(component_name(offer.id), Custom(spec), dict(spec))
    self._held[offer.id] = {
        'kind': str(spec['kind']),
        'domain': str(spec['domain']),
        'level': spec.get('params', {}).get('level'),
        'via': offer.via,
        'from': offer.source,
    }

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {'held': [{'id': c, **held} for c, held in self._held.items()]}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    data: Mapping[str, Any] = state
    with self._lock:
      self._held = {
          str(held['id']): {
              'kind': str(held['kind']),
              'domain': str(held['domain']),
              'level': held.get('level'),
              'via': str(held.get('via', '')),
              'from': str(held.get('from', '')),
          }
          for held in data.get('held', ())
      }
