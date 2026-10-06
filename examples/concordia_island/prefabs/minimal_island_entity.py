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

"""Minimal Island Entity prefab for Concordia simulation.

Implements a true minimal prompted agent (rolling memory + instructions + clock)
with ExperienceReflection psychometric measurement (ESM, BFI-10, MEMS, SWLS,
GHQ-12).
All survey tasks have save_to_memory=False so they are purely observational
and do NOT causally affect the agent's behavior, memory, or decision-making.
"""

from collections.abc import Mapping, Sequence
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib
from concordia.utils import async_measurements as async_measurements_lib

from examples.concordia_island.prefabs import island_entity
from examples.concordia_island.sim import experience_reflection as experience_reflection_lib
from examples.concordia_island.sim import important_memories as important_memories_lib
from examples.concordia_island.sim import schedule_awareness as schedule_awareness_lib


def _noncausal_tasks(
    ticks_per_day: int = 8,
    big_five: str = "bfi10",
) -> list[experience_reflection_lib.MeasurementTask]:
  """Default measurement tasks with ALL save_to_memory=False.

  Unlike default_tasks(), this excludes journal_task (which has
  save_to_memory=True) to guarantee zero causal influence on the agent.

  Args:
    ticks_per_day: Number of ticks per simulated day, used to set the frequency
      of daily measurement tasks.
    big_five: Big Five battery, "bfi10" (default) or "bfi2".

  Returns:
    A list of MeasurementTask instances, all with save_to_memory=False.
  """
  return [
      experience_reflection_lib.esm_monologue_task(frequency=4),
      experience_reflection_lib.esm_affect_task(frequency=4),
      experience_reflection_lib.journal_task(
          frequency=ticks_per_day, save_to_memory=False
      ),
      experience_reflection_lib.big_five_task(
          big_five, frequency=ticks_per_day
      ),
      experience_reflection_lib.mems_nightly_task(frequency=ticks_per_day),
      # No mems_open_ended_tasks — they have save_to_memory=False but are
      # verbose open-ended prompts that slow down the minimal agent
      experience_reflection_lib.swls_task(frequency=ticks_per_day),
      experience_reflection_lib.ghq12_task(frequency=ticks_per_day),
  ]


@dataclasses.dataclass
class MinimalIslandEntity(prefab_lib.Prefab):
  """Prefab for minimal island simulation entities.

  Strips out SituationPerception, SelfPerception, MovementDecision, and
  PersonBySituation. Relies strictly on raw memory, instructions, clock,
  and rolling history.

  The ExperienceReflection component is added purely as measurement
  instrumentation. Every survey task uses save_to_memory=False so the
  measurements do NOT influence the agent's behavior at all.
  """

  description: str = (
      "A true minimal prompted resident with psychometric polling (all"
      " save_to_memory=False)"
  )
  params: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: {
          "name": "Resident",
          "personality": "friendly and curious",
          "backstory": "",
          "home_place": "town_square",
          "work_place": None,
          "available_locations": [],
          "initial_observation": "",
          "memories": [],
          "tick_interval_minutes": 120,
          "experience_sampling": True,
          "big_five": "bfi10",
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    name = self.params.get("name", "Resident")
    personality = self.params.get("personality", "friendly and curious")
    backstory = self.params.get("backstory", "")
    home_place = self.params.get("home_place", "town_square")
    work_place = self.params.get("work_place", None)
    available_locations: Sequence[str] = self.params.get(
        "available_locations", []
    )
    initial_observation = self.params.get("initial_observation", "")
    memories = self.params.get("memories", [])
    formative_memories = self.params.get("formative_memories", [])
    age = self.params.get("age", None)

    # Standard memory setup
    memory_component_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = agent_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    # Seed formative memories
    if age is not None:
      memory_bank.add(f"[self] {name} is {age} years old.")
    if personality:
      memory_bank.add(f"[self] {name} is {personality}.")
    if backstory:
      memory_bank.add(f"[background] {backstory}")
    for memory in memories:
      memory_bank.add(memory)
    for memory in formative_memories:
      if not memory.startswith("["):
        memory_bank.add(f"[formative] {memory}")
      else:
        memory_bank.add(memory)
    if initial_observation:
      memory_bank.add(initial_observation)

    # 1. Instructions (stable personality prompt)
    instructions_key = "Instructions"
    instructions = agent_components.instructions.Instructions(
        agent_name=name,
        pre_act_label=f"\n{name}'s core traits",
    )

    # 2. Clock schedule awareness
    schedule_awareness_key = "ScheduleAwareness"
    schedule_awareness = schedule_awareness_lib.ScheduleAwareness(
        agent_name=name,
        home_place=home_place,
        work_place=work_place,
        available_locations=available_locations,
        pre_act_label=f"\n{name}'s current schedule",
    )

    observation_to_memory_key = "observation_to_memory"
    observation_to_memory = agent_components.observation.ObservationToMemory()

    # 4. Rolling observations (Last 50 history window)
    observation_component_key = (
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = important_memories_lib.ImportantMemories(
        recent_history_length=50,
        pre_act_label="\nRecent events",
    )

    # 5. Dynamic Location tracking
    current_location_key = "CurrentLocation"
    current_location_component = island_entity.DynamicLocation(
        initial_location=home_place,
        pre_act_label=f"\n{name}'s current location",
    )

    # Location constant
    location_info = f"{name} lives at {home_place}."
    if work_place:
      location_info += f" They work at {work_place}."
    if available_locations:
      location_info += (
          f" Places they can visit: {', '.join(available_locations[:15])}."
      )
    location_info_key = "LocationInfo"
    location_info_component = agent_components.constant.Constant(
        state=location_info,
        pre_act_label="\nLocation information",
    )

    components = {
        instructions_key: instructions,
        location_info_key: location_info_component,
        schedule_awareness_key: schedule_awareness,
        observation_to_memory_key: observation_to_memory,
        observation_component_key: observation,
        current_location_key: current_location_component,
        memory_component_key: memory_component,
    }

    # Canonical order for building action prompt context:
    # Instructions + Location + Schedule + Observations + Location
    component_order = [
        instructions_key,
        location_info_key,
        schedule_awareness_key,
        observation_component_key,
        current_location_key,
    ]

    # Experience sampling nightly questionnaires
    tick_interval = self.params.get("tick_interval_minutes", 120)
    ticks_per_day = max(1, (16 * 60) // tick_interval)
    experience_sampling = self.params.get("experience_sampling", True)
    big_five = self.params.get("big_five", "bfi10")

    if experience_sampling:
      experience_reflection_key = "ExperienceReflection"
      experience_reflection = experience_reflection_lib.ExperienceReflection(
          model=model,
          tasks=_noncausal_tasks(
              ticks_per_day=ticks_per_day, big_five=big_five
          ),
          context_components=component_order,
      )
      components[experience_reflection_key] = experience_reflection

    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        component_order=component_order,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components,
        measurements=async_measurements_lib.ReactiveMeasurements(),
    )
