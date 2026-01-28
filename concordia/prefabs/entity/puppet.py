# Copyright 2024 DeepMind Technologies Limited.
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

"""A prefab for a 'puppet' agent with fixed responses."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import puppet_act
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """A prefab for a puppet agent with fixed responses.

  This prefab creates an agent that responds with pre-configured fixed responses
  for specific calls to action. If no fixed response matches, it falls back to
  generating a response using the language model.
  """

  description: str = (
      "An entity with fixed responses for specific calls to action."
  )
  params: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: {
          "name": "Puppet Agent",
          "fixed_responses": {},
          "goal": "",
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the puppet agent."""
    name = self.params.get("name", "Puppet Agent")
    fixed_responses_map = self.params.get("fixed_responses", {})
    goal = self.params.get("goal", "")

    mem = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)
    instr = agent_components.instructions.Instructions(agent_name=name)
    obs_to_mem = agent_components.observation.ObservationToMemory()
    obs = agent_components.observation.LastNObservations(history_length=100)

    act_comp = puppet_act.PuppetActComponent(
        model=model,
        fixed_responses=fixed_responses_map,
    )

    components = {
        agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: mem,
        "Instructions": instr,
        "ObservationToMemory": obs_to_mem,
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: obs,
    }

    if goal:
      components["Goal"] = agent_components.constant.Constant(
          state=goal, pre_act_label="\nGoal"
      )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_comp,
        context_components=components,
    )
