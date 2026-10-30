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
import re
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import memory
from concordia.components.agent import observation
from examples.lifeways.components import culture
from examples.lifeways.components import memory as lifeways_memory
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


class Plan:
  """A rule that makes the planned move in its year, and watches otherwise.

  The year is the latest the world's status line told the Experimenter (not
  any "Year N." an interview answer might hold). The plan is fixed, so the
  rule keeps no state.
  """

  def __init__(self, plan: Mapping[int | str, str]):
    """Initializes the plan.

    Args:
      plan: The Experimenter's answer by year (e.g. a release).
    """
    self._plan = {int(year): str(answer) for year, answer in plan.items()}

  def __call__(
      self, action_spec: entity_lib.ActionSpec, contexts: Mapping[str, str]
  ) -> str:
    del action_spec
    years = [
        int(y)
        for text in contexts.values()
        for y in world_lib.EXPERIMENTER_STATUS.findall(text)
    ]
    if not years:
      return world_lib.WATCH
    return self._plan.get(max(years), world_lib.WATCH)


@dataclasses.dataclass
class Experimenter(prefab_lib.Prefab):
  """The Experimenter's entity.

  Params:
    plan: Optional. The Experimenter's answer by year (e.g. a release in the
      World's `RELEASE_FORM`): played by the `Plan` rule.
    act_component_factory: Optional. Builds the acting component from the
      context components' order; a rule (the plan, or watching) by default.
      Not serializable: checkpoints drop it, so a restore passes it again.
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
      memory_bank: Unused: the Experimenter keeps a bounded memory.

    Returns:
      The Experimenter's entity.
    """
    del model, memory_bank
    self.check_extra_components()
    components = {
        # The Experimenter lives as long as the world: it keeps only the
        # latest of what it observed.
        memory.DEFAULT_MEMORY_COMPONENT_KEY: lifeways_memory.RecentMemory(
            kept=32
        ),
        'observation_to_memory': observation.ObservationToMemory(),
        observation.DEFAULT_OBSERVATION_COMPONENT_KEY: (
            observation.LastNObservations(history_length=4)
        ),
    }
    factory = self.params.get('act_component_factory')
    plan = self.params.get('plan')
    if factory is not None:
      act = factory([observation.DEFAULT_OBSERVATION_COMPONENT_KEY])
    elif plan:
      act = culture.RuleAct(Plan(plan))
    else:
      act = culture.RuleAct(watch)
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=world_lib.EXPERIMENTER,
        act_component=act,
        context_components=components,
        measurements=measures.measurements(),
    )
