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
import json
import re
import threading
from typing import Any, cast, override
import zlib

from concordia.agents import entity_agent
from concordia.components.agent import question_of_recent_memories
from examples.lifeways.world import culture as culture_lib
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import numpy as np

# A rule that plays a role: the question and the contexts, to an answer.
Rule = Callable[[entity_lib.ActionSpec, Mapping[str, str]], str]

COMPONENT_PREFIX = 'custom '
# How many of the customs a Learner let go it remembers (the latest).
LET_GO_KEPT = 5
# Why a custom was let go to make room for another.
SAME_LOCUS, DOMAIN_FULL, CAPACITY_FULL = 'locus', 'domain', 'capacity'

_LABELS = {
    culture_lib.NORM: 'A custom of ours',
    'known norm': 'A custom others expect of us',
    culture_lib.VALUE: 'Something we hold dear',
    culture_lib.BELIEF: 'Something we believe',
    culture_lib.SKILL: 'Something we know how to do',
    culture_lib.HABIT: 'A habit of ours',
    culture_lib.BIAS: 'Whom we learn from',
    culture_lib.DIALECT: 'How we speak',
    culture_lib.MARRIAGE: 'Whom we marry',
    culture_lib.QUESTION: 'A question we ask ourselves',
}
# How many recent memories a question custom reads.
QUESTION_MEMORIES = 10


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


class QuestionCustom(Custom):
  """A custom of reasoning: a question the person asks itself before acting.

  The question is put by the stock `QuestionOfRecentMemories` (an
  InteractiveDocument over the person's recent memories, as Concordia asks
  models), once an act; its answer joins what the person acts on. So a
  person's logic of appropriateness, or of consequence, is a custom it was
  raised in, and may drop.
  """

  def __init__(
      self, spec: Mapping[str, Any], model: language_model.LanguageModel
  ):
    super().__init__(spec)
    self._question = question_of_recent_memories.QuestionOfRecentMemories(
        model=model,
        pre_act_label=_LABELS[culture_lib.QUESTION],
        # Asked as written (the stock component fills `{agent_name}`; a
        # custom's words never hold braces).
        question=str(spec['text']).replace('{', '(').replace('}', ')'),
        answer_prefix='{agent_name} ',
        add_to_memory=False,
        num_memories_to_retrieve=QUESTION_MEMORIES,
    )

  @override
  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    super().set_entity(entity)
    self._question.set_entity(entity)

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    asked = super().pre_act(action_spec)
    return f'{asked}\n{self._question.get_pre_act_value()}'

  @override
  def update(self) -> None:
    self._question.update()  # A new question for the next act.


def build_custom(
    spec: Mapping[str, Any], model: language_model.LanguageModel | None
) -> Custom:
  """A custom's component: a question custom asks with `model`."""
  if spec['kind'] == culture_lib.QUESTION and model is not None:
    return QuestionCustom(spec, model)
  return Custom(spec)


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
  if via == culture_lib.PREACHING:
    # Content bias: what is preached moves beliefs and values most, and
    # nobody learns a skill from a sermon.
    chance *= {
        culture_lib.BELIEF: 1.5,
        culture_lib.VALUE: 1.5,
        culture_lib.SKILL: 0.0,
    }.get(offer.group('kind'), 1.0)
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
  # Deference: one who heeds those who judge at the court takes up a
  # judge's ways the more readily the more it has judged (up to 3 times).
  judged = int(offer.group('judged') or 0)
  if 'who judge' in biases and judged:
    chance *= 1.0 + 0.5 * min(judged, 4)
  return min(1.0, chance)


def make_custom(
    name: str,
    spec: entity_component.ComponentState,
    model: language_model.LanguageModel | None = None,
) -> entity_component.ContextComponent:
  """The component factory of a person's entity: a custom from its spec.

  Args:
    name: The component's key (unused: the spec names the custom).
    spec: The custom's spec, as the entity's state recorded it.
    model: The model a question custom asks with (bound by the prefab).

  Returns:
    The custom's component.
  """
  del name
  return build_custom(spec, model)


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
      capacity: int = 10,
      model: language_model.LanguageModel | None = None,
  ):
    super().__init__()
    # The model the person's question customs ask with.
    self._model = model
    self._seed = int(seed)
    self._uptake = dict(uptake)
    self._per_domain = int(per_domain)
    self._capacity = int(capacity)
    self._lock = threading.Lock()
    # What the person holds, oldest first (including customs whose components
    # join when the current call ends): id -> kind, domain, level, channel.
    self._held: dict[str, dict[str, Any]] = {}
    # The customs it let go to make room, latest last (`LET_GO_KEPT`): what
    # each was, what it made room for, and why (`SAME_LOCUS`, ...).
    self._let_go: list[dict[str, str]] = []

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
      entity.add_component(
          component_name(spec['id']), build_custom(spec, self._model), spec
      )
      self._hold(spec, via)

  def held(self) -> list[str]:
    """The ids of the customs the person holds, oldest first."""
    with self._lock:
      return list(self._held)

  def _hold(self, spec: Mapping[str, Any], via: str, source: str = '') -> None:
    # What the person remembers of a custom: how it came, and from whom.
    self._held[str(spec['id'])] = {
        'kind': str(spec['kind']),
        'domain': str(spec['domain']),
        'level': int(spec.get('params', {}).get('level', 0)),
        'form': culture_lib.form_of(spec),
        'via': via,
        'from': source,
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
        if offer.group('params'):
          try:
            params = json.loads(offer.group('params'))
          except json.JSONDecodeError:
            continue  # Not an offer a person can make sense of.
          if not isinstance(params, dict):
            continue
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
          entity.add_component(
              component_name(custom_id), build_custom(spec, self._model), spec
          )
          self._hold(spec, offer.group('via'), offer.group('source') or '')
    return ''

  def _displaced(self, spec: Mapping[str, Any]) -> tuple[bool, str | None, str]:
    """What taking up a custom would cost, changing nothing.

    Args:
      spec: The custom offered.

    Returns:
      Whether it would be taken up at all, the custom it would push out
      (None if there is room), and why (`SAME_LOCUS`, `DOMAIN_FULL`,
      `CAPACITY_FULL`; '' if nothing).
    """
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
      return False, None, ''  # Nobody learns a worse way to do what they do.
    if culture_lib.form_of(spec) == culture_lib.KNOWN and any(
        self._held[c]['form'] == culture_lib.INTERNALIZED for c in locus
    ):
      return False, None, ''  # One's own is not displaced by what is expected.
    same = [c for c, held in self._held.items() if held['domain'] == domain]
    if locus:
      return True, locus[0], SAME_LOCUS
    if len(same) >= self._per_domain:
      return True, same[0], DOMAIN_FULL
    if len(self._held) >= self._capacity:
      return True, next(iter(self._held)), CAPACITY_FULL
    return True, None, ''

  def would_take(self, spec: Mapping[str, Any]) -> dict[str, Any]:
    """What this person would do with a custom offered (reading only).

    The rule uptake uses to make room, asked without taking anything up:
    for the page's preview of a release.

    Args:
      spec: The custom offered: kind, domain, and params (level, form).

    Returns:
      `wanted` (False if it would not take it at all), and `pushes`, the id
      of the custom it would let go to make room (or None), and `why`.
    """
    with self._lock:
      wanted, victim, why = self._displaced(spec)
    return {'wanted': wanted, 'pushes': victim, 'why': why}

  def _make_room(
      self, entity: entity_agent.EntityAgent, spec: Mapping[str, Any]
  ) -> bool:
    """Drops what a new custom displaces; False if it is not wanted."""
    wanted, victim, why = self._displaced(spec)
    if not wanted:
      return False
    if victim is None:
      return True
    dropped = entity.get_all_context_components().get(component_name(victim))
    self._let_go.append({
        'id': victim,
        'kind': self._held[victim]['kind'],
        'domain': self._held[victim]['domain'],
        'text': dropped.spec['text'] if isinstance(dropped, Custom) else '',
        'for': str(spec.get('id', '')),
        'for_text': str(spec.get('text', '')),
        'why': why,
    })
    del self._let_go[:-LET_GO_KEPT]
    entity.remove_component(component_name(victim))
    del self._held[victim]
    return True

  def let_go(self) -> list[dict[str, str]]:
    """The customs let go to make room, latest last (measurement, display)."""
    with self._lock:
      return [dict(record) for record in self._let_go]

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'held': [{'id': c, **held} for c, held in self._held.items()],
          'let_go': [dict(record) for record in self._let_go],
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
              'from': str(held.get('from', '')),
          }
          for held in data.get('held', ())
      }
      self._let_go = [  # Saves before cycle 36 kept none.
          {str(k): str(v) for k, v in dict(record).items()}
          for record in data.get('let_go', ())
      ][-LET_GO_KEPT:]


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
