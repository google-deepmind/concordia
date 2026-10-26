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

"""The Experimenter: the player, who shapes the world and never edits minds.

Its moves are interventions the World GM resolves like any answer. Played by
a person (`act_component_factory`: a `HumanActComponent`) or a rule; by
default a rule that watches. Its name is the World's `EXPERIMENTER`, the one
participant who is not a person of the valley.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import memory
from concordia.components.agent import observation
from examples.lifeways.components import culture
from examples.lifeways.prefabs import measures
from examples.lifeways.world import world as world_lib
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib

EXPERIMENTER_PREFAB = 'lifeways_experimenter'


def watch(
    action_spec: entity_lib.ActionSpec, contexts: Mapping[str, str]
) -> str:
  """The default rule: watch, year after year.

  Args:
    action_spec: The question (unused).
    contexts: What the Experimenter knows (unused).

  Returns:
    The move to watch.
  """
  del action_spec, contexts
  return world_lib.WATCH


@dataclasses.dataclass
class Experimenter(prefab_lib.Prefab):
  """The Experimenter's entity.

  Params:
    act_component_factory: Optional. Builds the acting component from the
      context components' order; a rule that watches by default. Not
      serializable: checkpoints drop it, so a restore passes it again.
  """

  description: str = (  # pyrefly: ignore[bad-override]
      'The Lifeways Experimenter, who intervenes on the world, not on minds.'
  )
  params: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the Experimenter's entity.

    Args:
      model: Unused.
      memory_bank: The Experimenter's memory bank.

    Returns:
      The Experimenter's entity.
    """
    del model
    self.check_extra_components()
    components = {
        memory.DEFAULT_MEMORY_COMPONENT_KEY: memory.AssociativeMemory(
            memory_bank=memory_bank
        ),
        'observation_to_memory': observation.ObservationToMemory(),
        observation.DEFAULT_OBSERVATION_COMPONENT_KEY: (
            observation.LastNObservations(history_length=4)
        ),
    }
    factory = self.params.get('act_component_factory')
    act = (
        factory([observation.DEFAULT_OBSERVATION_COMPONENT_KEY])
        if factory is not None
        else culture.RuleAct(watch)
    )
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=world_lib.EXPERIMENTER,
        act_component=act,
        context_components=components,
        measurements=measures.measurements(),
    )
