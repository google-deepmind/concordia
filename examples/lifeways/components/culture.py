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

"""A person's culture: customs as components, and the Learner that keeps them.

Each custom a person holds is a context component of the person's own entity,
added and removed while the person lives (`EntityAgent.add_component`). Its
context tells the person's decision logic what the custom says. The Learner is
the person's own machinery for culture: it takes up the customs the GM offers
in observations, by the person's own learning biases, and keeps the person's
customs within capacity.
"""

from collections.abc import Callable, Mapping
import re
import threading
from typing import Any, cast, override
import zlib

from concordia.agents import entity_agent
from examples.lifeways.world import culture as culture_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import numpy as np

# A rule that plays a role: the question and the contexts, to an answer.
Rule = Callable[[entity_lib.ActionSpec, Mapping[str, str]], str]

COMPONENT_PREFIX = 'custom '

_LABELS = {
    culture_lib.NORM: 'A custom of ours',
    'known norm': 'A custom others expect of us',
    culture_lib.VALUE: 'Something we hold dear',
    culture_lib.BELIEF: 'Something we believe',
    culture_lib.SKILL: 'Something we know how to do',
    culture_lib.HABIT: 'A habit of ours',
    culture_lib.BIAS: 'Whom we learn from',
}


def component_name(custom_id: str) -> str:
  """The entity's key for the component of a custom."""
  return f'{COMPONENT_PREFIX}{custom_id}'


class Custom(entity_component.ContextComponent):
  """A custom a person holds: its context is what the custom says."""

  def __init__(self, spec: Mapping[str, Any]):
    super().__init__()
    self.spec = dict(spec)

  @property
  def id(self) -> str:
    return str(self.spec['id'])

  @property
  def domain(self) -> str:
    return str(self.spec['domain'])

  @property
  def kind(self) -> str:
    return str(self.spec['kind'])

  @property
  def form(self) -> str:
    return culture_lib.form_of(self.spec)

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    level = self.spec.get('params', {}).get('level')
    detail = f' (level {level})' if level is not None else ''
    label = _LABELS.get(
        'known norm' if self.form == culture_lib.KNOWN else self.kind,
        'A custom',
    )
    return f'{label} ({self.domain}) [{self.id}]: {self.spec["text"]}{detail}'

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


def offer_chance(
    offer: re.Match[str], uptake: Mapping[str, float], biases: str
) -> float:
  """The chance a person takes up one offer.

  Args:
    offer: The offer (a match of `world.culture.OFFER`).
    uptake: The person's base chance by channel.
    biases: The texts of the person's learning biases, lowercase.

  Returns:
    The chance, from the channel's and changed by the biases.
  """
  via = offer.group('via')
  chance = float(uptake.get(via, 0.0))
  if 'traditional' in biases and via not in (
      culture_lib.BIRTH,
      culture_lib.UPBRINGING,
  ):
    chance *= 0.3
  if 'new ways' in biases:
    chance *= 1.5
  stores = offer.group('stores')
  if 'stores are full' in biases and stores is not None:
    chance *= 0.5 + float(stores) / 8.0
  return min(1.0, chance)


def make_custom(
    name: str, spec: entity_component.ComponentState
) -> entity_component.ContextComponent:
  """The component factory of a person's entity: a custom from its spec.

  Args:
    name: The component's key (unused: the spec names the custom).
    spec: The custom's spec, as the entity's state recorded it.

  Returns:
    The custom's component.
  """
  del name
  return Custom(spec)


class Learner(entity_component.ContextComponent):
  """Takes up offered customs by the person's biases; keeps capacity.

  An offer (a line of an observation the GM delivered) is taken up with a
  chance set by the channel it came by (`uptake`) and the person's learning
  biases: a 'traditional' person rarely takes up what is not its own family's,
  one open to 'new ways' more often, and one who learns from 'those whose
  stores are full' copies the well fed more. Each further person seen doing
  the same is another chance (so what many do spreads faster). A skill is
  taken up only if it is better than the one the person has, and a norm
  merely expected of the person never displaces one it has made its own.

  A kind of custom in a domain is a locus, like a plasmid's incompatibility
  group: a new custom replaces the one the person holds there. A person holds
  at most `per_domain` customs of a domain and `capacity` in all; past either,
  the oldest goes. Every draw is seeded by the world seed, the person and the
  offer, so uptake never depends on thread timing.
  """

  def __init__(
      self,
      seed: int,
      uptake: Mapping[str, float],
      per_domain: int = 2,
      capacity: int = 8,
  ):
    super().__init__()
    self._seed = int(seed)
    self._uptake = dict(uptake)
    self._per_domain = int(per_domain)
    self._capacity = int(capacity)
    self._lock = threading.Lock()
    # What the person holds, oldest first (including customs whose components
    # join when the current call ends): id -> kind, domain, level, channel.
    self._held: dict[str, dict[str, Any]] = {}

  def inherit(
      self, spec: Mapping[str, Any], via: str = culture_lib.BIRTH
  ) -> None:
    """Takes a custom the person is born (or comes) with, within capacity.

    Args:
      spec: The custom's spec.
      via: How it came: 'birth' or 'arrival'.

    Raises:
      TypeError: If the Learner's entity is not an `EntityAgent`.
    """
    entity = self.get_entity()
    if not isinstance(entity, entity_agent.EntityAgent):
      raise TypeError('A Learner belongs to an EntityAgent.')
    with self._lock:
      if str(spec['id']) in self._held or not self._make_room(entity, spec):
        return
      entity.add_component(component_name(spec['id']), Custom(spec), spec)
      self._hold(spec, via)

  def held(self) -> list[str]:
    """The ids of the customs the person holds, oldest first."""
    with self._lock:
      return list(self._held)

  def _hold(self, spec: Mapping[str, Any], via: str) -> None:
    self._held[str(spec['id'])] = {
        'kind': str(spec['kind']),
        'domain': str(spec['domain']),
        'level': int(spec.get('params', {}).get('level', 0)),
        'form': culture_lib.form_of(spec),
        'via': via,
    }

  def _biases(self, entity: entity_agent.EntityAgent) -> str:
    return ' '.join(
        c.spec['text'].lower()
        for c in entity.get_all_context_components().values()
        if isinstance(c, Custom) and c.kind == culture_lib.BIAS
    )

  @override
  def pre_observe(self, observation: str) -> str:
    entity = self.get_entity()
    if not isinstance(entity, entity_agent.EntityAgent):
      return ''
    biases = self._biases(entity)
    offers: dict[str, list[re.Match[str]]] = {}
    for offer in culture_lib.OFFER.finditer(observation):
      offers.setdefault(offer.group('id'), []).append(offer)
    with self._lock:
      for custom_id, seen in offers.items():
        if custom_id in self._held:
          continue
        chances = [offer_chance(o, self._uptake, biases) for o in seen]
        missed = float(np.prod([1.0 - c for c in chances]))
        draw = np.random.default_rng([
            self._seed,
            zlib.crc32(f'{entity.name}/{custom_id}/{observation}'.encode()),
        ]).random()
        if draw >= 1.0 - missed:
          continue
        offer = seen[int(np.argmax(chances))]
        params: dict[str, Any] = {}
        if offer.group('level'):
          params['level'] = int(offer.group('level'))
        if offer.group('form'):
          params['form'] = offer.group('form')
        spec = {
            'id': custom_id,
            'kind': offer.group('kind'),
            'domain': offer.group('domain'),
            'text': offer.group('text').strip(),
            'params': params,
        }
        if self._make_room(entity, spec):
          entity.add_component(component_name(custom_id), Custom(spec), spec)
          self._hold(spec, offer.group('via'))
    return ''

  def _make_room(
      self, entity: entity_agent.EntityAgent, spec: Mapping[str, Any]
  ) -> bool:
    """Drops what a new custom displaces; False if it is not wanted."""
    kind, domain = spec['kind'], spec['domain']
    level = int(spec.get('params', {}).get('level', 0))
    locus = [
        c
        for c, held in self._held.items()
        if held['kind'] == kind and held['domain'] == domain
    ]
    if kind == culture_lib.SKILL and any(
        self._held[c]['level'] >= level for c in locus
    ):
      return False  # Nobody learns a worse way to do what they do.
    if culture_lib.form_of(spec) == culture_lib.KNOWN and any(
        self._held[c]['form'] == culture_lib.INTERNALIZED for c in locus
    ):
      return False  # What is one's own is not displaced by what is expected.
    same = [c for c, held in self._held.items() if held['domain'] == domain]
    if locus:
      victim = locus[0]
    elif len(same) >= self._per_domain:
      victim = same[0]
    elif len(self._held) >= self._capacity:
      victim = next(iter(self._held))
    else:
      return True
    entity.remove_component(component_name(victim))
    del self._held[victim]
    return True

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'held': [{'id': c, **held} for c, held in self._held.items()],
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    data: Mapping[str, Any] = state
    with self._lock:
      self._held = {
          str(held['id']): {
              'kind': str(held['kind']),
              'domain': str(held['domain']),
              'level': int(held['level']),
              'form': str(held.get('form', '')),
              'via': str(held['via']),
          }
          for held in data.get('held', ())
      }


class RuleAct(entity_component.ActingComponent):
  """Acts by a rule of the question and the contexts (a bot, the watcher).

  A rule that keeps state must expose `get_state`/`set_state`; this component
  saves and restores it with its own.
  """

  def __init__(self, rule: Rule):
    """Initializes the component.

    Args:
      rule: The rule that answers.
    """
    super().__init__()
    self._rule = rule

  @override
  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    return self._rule(action_spec, context)

  @override
  def get_state(self) -> entity_component.ComponentState:
    get_state = getattr(self._rule, 'get_state', None)
    if not callable(get_state):
      return {}
    return {'rule': cast(entity_component.ComponentState, get_state())}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    set_state = getattr(self._rule, 'set_state', None)
    if callable(set_state) and 'rule' in state:
      set_state(state['rule'])
