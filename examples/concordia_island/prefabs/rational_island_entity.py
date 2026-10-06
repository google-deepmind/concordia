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

"""Rational Island Entity prefab for Concordia simulation.

A consequence-driven, goal-optimizing agent with location awareness,
perception components, and island-specific behaviors.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import question_of_recent_memories
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib
from concordia.utils import async_measurements as async_measurements_lib

from examples.concordia_island.prefabs import island_entity
from examples.concordia_island.sim import experience_reflection as experience_reflection_lib
from examples.concordia_island.sim import important_memories as important_memories_lib
from examples.concordia_island.sim import schedule_awareness as schedule_awareness_lib


@dataclasses.dataclass
class RationalIslandEntity(prefab_lib.Prefab):
  """Prefab for rational, consequence-driven island simulation entities."""

  description: str = (
      "A rational resident optimizing for their goal on the island"
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          "name": "Resident",
          "personality": "friendly and curious",
          "backstory": "",
          "home_place": "town_square",
          "work_place": None,
          "available_locations": [],
          "initial_observation": "",
          "memories": [],
          "goal": "",
          "randomize_choices": True,
          "prefix_entity_name": True,
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
    traits = self.params.get("traits", {})
    age = self.params.get("age", None)
    remove_traits_from_memory = self.params.get(
        "remove_traits_from_memory", True
    )

    entity_goal = self.params.get("goal", "")
    randomize_choices = self.params.get("randomize_choices", True)
    prefix_entity_name = self.params.get("prefix_entity_name", True)
    tick_interval = self.params.get("tick_interval_minutes", 120)
    ticks_per_day = max(1, (16 * 60) // tick_interval)
    experience_sampling = self.params.get("experience_sampling", True)
    big_five = self.params.get("big_five", "bfi10")

    # Seed associative memory
    if age is not None:
      memory_bank.add(f"[self] {name} is {age} years old.")
    if personality:
      memory_bank.add(f"[self] {name} is {personality}.")
    if backstory:
      memory_bank.add(f"[background] {backstory}")
    if traits and not remove_traits_from_memory:
      trait_descriptions = {
          "openness": "Openness to experience",
          "conscientiousness": "Conscientiousness",
          "extraversion": "Extraversion",
          "agreeableness": "Agreeableness",
          "neuroticism": "Neuroticism",
          "locus_of_control": "Locus of control (0=external, 1=internal)",
          "social_trust": "Social trust (0=suspicious, 1=trusting)",
          "religiosity": "Religiosity (0=secular, 1=devout)",
          "technology_attitude": (
              "Technology attitude (0=skeptical, 1=embracing)"
          ),
          "community_orientation": (
              "Community orientation (0=individualist, 1=collectivist)"
          ),
      }
      trait_lines = []
      for trait_key, value in traits.items():
        label = trait_descriptions.get(trait_key, trait_key)
        trait_lines.append(f"  {label}: {value}")
      trait_text = "\n".join(trait_lines)
      memory_bank.add(f"[self] {name}'s psychological profile:\n{trait_text}")
    for memory in memories:
      memory_bank.add(memory)
    for memory in formative_memories:
      memory_bank.add(memory)
    if initial_observation:
      memory_bank.add(initial_observation)

    instructions_key = "Instructions"
    instructions = agent_components.instructions.Instructions(
        agent_name=name,
        pre_act_label=f"\n{name}'s core traits",
    )

    location_info = f"{name} lives at {home_place}."
    if work_place:
      location_info += f" They work at {work_place}."
      location_info += (
          " Working hours are 9:00 AM to 5:00 PM, Monday through Friday."
          f" {name} should go to {work_place} when it is a weekday morning"
          " and leave work when the workday ends. On Saturday and Sunday,"
          f" {name} does NOT go to work."
      )
    if available_locations:
      available_str = ", ".join(available_locations[:15])
      if "sunset_apartments" in home_place:
        available_str += ", sunset_apartments_common_room"
      location_info += f" Places they can visit: {available_str}."
    location_info += (
        f" {name} moves between locations throughout the day (e.g. home to"
        " work, work to errands, errands to home, third spaces to socialize)."
    )
    if "sunset_apartments" in home_place:
      location_info += (
          f" {name} can visit the sunset_apartments_common_room to socialize"
          " with neighbors before or after work."
      )

    location_info_key = "LocationInfo"
    location_info_component = agent_components.constant.Constant(
        state=location_info,
        pre_act_label="\nLocation information",
    )

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

    observation_component_key = (
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = important_memories_lib.ImportantMemories(
        recent_history_length=50,
        pre_act_label="\nRecent events",
    )

    current_location_key = "CurrentLocation"
    current_location_component = island_entity.DynamicLocation(
        initial_location=home_place,
        pre_act_label=f"\n{name}'s current location",
    )

    life_altering_events_key = "LifeAlteringEvents"
    life_altering_events = island_entity.TaggedMemories(
        tags=("[life altering event]",),
        pre_act_label="\nLife altering events",
    )

    # --- Rational Consequence Decision Components ---
    situation_perception_key = "SituationPerception"
    situation_perception = question_of_recent_memories.SituationPerception(
        model=model,
        num_memories_to_retrieve=10,
        pre_act_label=(
            f"\nQuestion: What situation is {name} in right now?\nAnswer"
        ),
    )

    relevant_memories_key = "RelevantMemories"
    relevant_memories = (
        agent_components.all_similar_memories.AllSimilarMemories(
            model=model,
            components=[situation_perception_key],
            num_memories_to_retrieve=10,
            pre_act_label="\nRecalled memories and observations",
        )
    )

    options_perception_key = "AvailableOptionsPerception"
    options_perception = agent_components.question_of_recent_memories.AvailableOptionsPerception(
        model=model,
        components=[
            observation_component_key,
            relevant_memories_key,
            situation_perception_key,
        ],
        pre_act_label=(
            f"\nQuestion: Which options are available to {name} "
            "right now?\nAnswer"
        ),
    )

    best_option_perception_key = "BestOptionPerception"

    if entity_goal:
      goal_key = "Goal"
      overarching_goal = agent_components.constant.Constant(
          state=entity_goal, pre_act_label="\nOverarching goal"
      )
      best_option_label = (
          f"\nQuestion: Of the options available to {name}, and "
          "given their goal, which choice of action or strategy is "
          f"best for {name} to take right now?\nAnswer"
      )
    else:
      goal_key = None
      overarching_goal = None
      best_option_label = (
          f"\nQuestion: Of the options available to {name}, "
          "which choice of action or strategy is "
          f"best for {name} to take right now?\nAnswer"
      )

    best_option_components = [
        observation_component_key,
        relevant_memories_key,
        situation_perception_key,
        options_perception_key,
    ]
    if overarching_goal:
      best_option_components.append(goal_key)

    best_option_perception = (
        agent_components.question_of_recent_memories.BestOptionPerception(
            model=model,
            components=best_option_components,
            pre_act_label=best_option_label,
        )
    )

    memory_component_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = agent_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    # Construct the components dictionary
    components = {
        instructions_key: instructions,
        location_info_key: location_info_component,
        schedule_awareness_key: schedule_awareness,
        observation_to_memory_key: observation_to_memory,
        observation_component_key: observation,
        current_location_key: current_location_component,
        life_altering_events_key: life_altering_events,
        situation_perception_key: situation_perception,
        relevant_memories_key: relevant_memories,
        options_perception_key: options_perception,
        best_option_perception_key: best_option_perception,
        memory_component_key: memory_component,
    }

    # Define prompt rendering hierarchy
    component_order = [
        instructions_key,
        location_info_key,
        schedule_awareness_key,
        observation_component_key,
        current_location_key,
    ]

    if overarching_goal is not None:
      components[goal_key] = overarching_goal
      component_order.insert(1, goal_key)

    component_order.extend([
        situation_perception_key,
        relevant_memories_key,
        options_perception_key,
        best_option_perception_key,
    ])

    if experience_sampling:
      experience_reflection_key = "ExperienceReflection"
      experience_reflection = experience_reflection_lib.ExperienceReflection(
          model=model,
          tasks=experience_reflection_lib.default_tasks(
              ticks_per_day=ticks_per_day, big_five=big_five
          )
          + experience_reflection_lib.ai_survey_tasks(frequency=ticks_per_day),
          context_components=component_order,
      )
      components[experience_reflection_key] = experience_reflection

    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        component_order=component_order,
        randomize_choices=randomize_choices,
        prefix_entity_name=prefix_entity_name,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components,
        measurements=async_measurements_lib.ReactiveMeasurements(),
    )
