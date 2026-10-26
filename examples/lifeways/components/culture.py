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

from collections.abc import Mapping
import re
from typing import Any, cast, override
import zlib

from concordia.agents import entity_agent
from concordia.typing import entity_component
import numpy as np

# How an offer of a custom appears in an observation.
OFFER = re.compile(
    r'\[custom id=(?P<id>c\d+) kind=(?P<kind>\w+) domain=(?P<domain>\w+)'
    r'(?: level=(?P<level>\d+))? via=(?P<via>\w+)\] (?P<text>[^\n]*)'
)
COMPONENT_PREFIX = 'custom '

_LABELS = {
    'norm': 'A custom of ours',
    'value': 'Something we hold dear',
    'belief': 'Something we believe',
    'skill': 'Something we know how to do',
    'habit': 'A habit of ours',
    'bias': 'Whom we learn from',
}


def component_name(custom_id: str) -> str:
  """The entity's key for the component of a custom."""
  return f'{COMPONENT_PREFIX}{custom_id}'


def offer_line(spec: Mapping[str, Any], via: str) -> str:
  """An observation line offering a custom (what the Learner reads)."""
  level = spec.get('params', {}).get('level')
  return (
      f'[custom id={spec["id"]} kind={spec["kind"]} domain={spec["domain"]}'
      f'{f" level={level}" if level is not None else ""} via={via}]'
      f' {spec["text"]}'
  )


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

  @override
  def pre_act(self, action_spec) -> str:
    del action_spec
    level = self.spec.get('params', {}).get('level')
    detail = f' (level {level})' if level is not None else ''
    return (
        f'{_LABELS.get(self.kind, "A custom")} ({self.domain}) [{self.id}]:'
        f' {self.spec["text"]}{detail}'
    )

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


def make_custom(
    name: str, spec: entity_component.ComponentState
) -> entity_component.ContextComponent:
  """The component factory of a person's entity: a custom from its spec."""
  del name
  return Custom(spec)


class Learner(entity_component.ContextComponent):
  """Takes up offered customs by the person's biases; keeps capacity.

  An offer (a line of an observation the GM delivered) is taken up with a
  chance set by how it came (`uptake`, by channel) and the person's learning
  biases (a 'traditionalist' person rarely takes up what is not its own
  family's). A person holds at most `per_domain` customs of a domain and
  `capacity` in all: a new custom displaces the oldest of its domain, or
  overall. Every draw is seeded by the world seed, the person and the offer,
  so a person's uptake never depends on thread timing.
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
    # Custom ids in the order they were taken up (oldest first, including
    # those whose components join when the current call ends), each one's
    # domain and how it came.
    self._held: list[str] = []
    self._domain: dict[str, str] = {}
    self._via: dict[str, str] = {}

  def note_inherited(self, custom_id: str, domain: str) -> None:
    """Records a custom the person was born with."""
    if custom_id not in self._held:
      self._held.append(custom_id)
      self._domain[custom_id] = domain
      self._via[custom_id] = 'birth'

  def held(self) -> list[str]:
    return list(self._held)

  def _biases(self, entity: entity_agent.EntityAgent) -> str:
    return ' '.join(
        c.spec['text'].lower()
        for c in entity.get_all_context_components().values()
        if isinstance(c, Custom) and c.kind == 'bias'
    )

  @override
  def pre_observe(self, observation: str) -> str:
    entity = self.get_entity()
    if not isinstance(entity, entity_agent.EntityAgent):
      return ''
    biases = self._biases(entity)
    for match in OFFER.finditer(observation):
      custom_id, via = match.group('id'), match.group('via')
      if custom_id in self._held:
        continue
      chance = float(self._uptake.get(via, 0.0))
      if 'traditional' in biases and via not in ('birth', 'upbringing'):
        chance *= 0.3
      if 'new ways' in biases:
        chance = min(1.0, chance * 1.5)
      draw = np.random.default_rng([
          self._seed,
          zlib.crc32(f'{entity.name}/{custom_id}/{observation}'.encode()),
      ]).random()
      if draw >= chance:
        continue
      spec = {
          'id': custom_id,
          'kind': match.group('kind'),
          'domain': match.group('domain'),
          'text': match.group('text').strip(),
          'params': (
              {'level': int(match.group('level'))}
              if match.group('level')
              else {}
          ),
      }
      self._make_room(entity, spec['domain'])
      entity.add_component(component_name(custom_id), Custom(spec), spec)
      self._held.append(custom_id)
      self._domain[custom_id] = spec['domain']
      self._via[custom_id] = via
    return ''

  def _make_room(self, entity: entity_agent.EntityAgent, domain: str) -> None:
    """Drops the oldest custom of a full domain, or overall, for a new one."""
    same = [c for c in self._held if self._domain.get(c) == domain]
    victim = None
    if len(same) >= self._per_domain:
      victim = same[0]
    elif len(self._held) >= self._capacity:
      victim = self._held[0]
    if victim is not None:
      entity.remove_component(component_name(victim))
      self._held.remove(victim)
      self._domain.pop(victim, None)
      self._via.pop(victim, None)

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {
        'held': list(self._held),
        'domain': dict(self._domain),
        'via': dict(self._via),
    }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    data: Mapping[str, Any] = state
    self._held = [str(c) for c in data.get('held', ())]
    self._domain = {str(k): str(v) for k, v in data.get('domain', {}).items()}
    self._via = {str(k): str(v) for k, v in data.get('via', {}).items()}


class RuleAct(entity_component.ActingComponent):
  """Acts by a rule of the question and the contexts (a bot, the watcher).

  A rule that keeps state must expose `get_state`/`set_state`; this component
  saves and restores it with its own.
  """

  def __init__(self, rule):
    super().__init__()
    self._rule = rule

  @override
  def get_action_attempt(self, contexts, action_spec) -> str:
    return self._rule(action_spec, contexts)

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
