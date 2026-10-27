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

"""A person of Lifeways: memory, observation, identity, a Learner, customs.

The person's customs are components of its own entity, added from its params
at birth (those its parents passed on) and later by its Learner. Who plays the
person is its acting component: a model through the stock
`ConcatActComponent` by default, or the factory in `act_component_factory`
(a person's `HumanActComponent`, a rule).
"""

from collections.abc import Mapping
import dataclasses
from typing import Any
import zlib

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import concat_act_component
from concordia.components.agent import constant
from concordia.components.agent import memory
from concordia.components.agent import observation
from examples.lifeways.components import culture
from examples.lifeways.prefabs import measures
from examples.lifeways.world import culture as culture_lib
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib
import numpy as np

PERSON_PREFAB = 'lifeways_person'
# A person someone plays: born after the played person dies, it lives on as
# it (the session's `Person`, with the page's acting component).
PLAYED_PREFAB = 'lifeways_played_person'
EMBEDDING_SIZE = 16


def hash_embedder(text: str) -> np.ndarray:
  """A cheap deterministic sentence embedder (memory needs one).

  Args:
    text: The text.

  Returns:
    Its embedding, the same for the same text in any process.
  """
  rng = np.random.default_rng(zlib.crc32(text.encode()))
  return rng.standard_normal(EMBEDDING_SIZE)


IDENTITY = 'Who I am'
LEARNER = 'learner'

# How often an offered custom is taken up, by the way it came.
DEFAULT_UPTAKE = {
    culture_lib.BIRTH: 1.0,
    culture_lib.UPBRINGING: 0.5,
    culture_lib.APPRENTICESHIP: 0.8,
    culture_lib.OBSERVATION: 0.15,
    culture_lib.STORY: 0.3,
    culture_lib.READING: 0.6,
    culture_lib.SCHOOL: 0.7,
    culture_lib.RITE: 0.4,
    culture_lib.PREACHING: 0.2,
    culture_lib.RELEASE: 0.25,
    culture_lib.SANCTION: 0.3,
    culture_lib.PRACTICE: 0.4,
}


@dataclasses.dataclass
class Person(prefab_lib.Prefab):
  """A person: memory, observation, identity, a Learner and customs.

  Params:
    name: The person's name.
    born: The year of birth.
    sex: 'f', 'm', or '' (asexual mode).
    parents: The parents' names.
    arrived: The year the person came to the valley, if not born in it.
    seed: The world seed (the Learner's draws).
    customs: The specs of the customs the person starts with.
    uptake: The Learner's chance of taking up an offer, by channel.
    per_domain: The most customs held of one domain.
    capacity: The most customs held in all.
    act_component_factory: Optional. Builds the acting component from the
      context components' order (a person's `HumanActComponent`, a rule).
      Not serializable: checkpoints drop it, so a restore passes it again.
  """

  description: str = (  # pyrefly: ignore[bad-override]
      'A Lifeways person whose customs are components of its own entity.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {'name': 'Ada', 'born': 0, 'customs': []}
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the person's entity.

    Args:
      model: The model a person played by a model acts with.
      memory_bank: The person's own memory bank.

    Returns:
      The person's entity, its inherited customs added.
    """
    self.check_extra_components()
    name = str(self.params['name'])
    parents = list(self.params.get('parents', ()))
    born = int(self.params.get('born', 0))
    arrived = self.params.get('arrived')
    if born < 0:
      who = f'{name}, born {-born} years before the valley was settled'
    else:
      who = f'{name}, born in year {born}'
    if parents:
      who += f', child of {" and ".join(parents)}'
    if arrived is not None:
      who += f', who came to the valley in year {arrived}'
    learner = culture.Learner(
        seed=int(self.params.get('seed', 0)),
        uptake=dict(self.params.get('uptake', DEFAULT_UPTAKE)),
        per_domain=int(self.params.get('per_domain', 2)),
        capacity=int(self.params.get('capacity', 10)),
    )
    components = {
        memory.DEFAULT_MEMORY_COMPONENT_KEY: memory.AssociativeMemory(
            memory_bank=memory_bank
        ),
        'observation_to_memory': observation.ObservationToMemory(),
        observation.DEFAULT_OBSERVATION_COMPONENT_KEY: (
            observation.LastNObservations(history_length=6)
        ),
        IDENTITY: constant.Constant(state=who + '.', pre_act_label=IDENTITY),
        LEARNER: learner,
    }
    order = [IDENTITY, observation.DEFAULT_OBSERVATION_COMPONENT_KEY]
    factory = self.params.get('act_component_factory')
    act = (
        factory(order)
        if factory is not None
        else concat_act_component.ConcatActComponent(
            model=model,
            component_order=order,
            prefix_entity_name=False,
            randomize_choices=False,
        )
    )
    entity = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act,
        context_components=components,
        measurements=measures.measurements(),
        component_factory=culture.make_custom,
    )
    for spec in self.params.get('customs', ()):
      learner.inherit(
          spec, culture_lib.BIRTH if arrived is None else culture_lib.ARRIVAL
      )
    return entity
