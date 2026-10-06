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

"""Enacted Self Agent (ESA) island entity prefab for Concordia simulation.

Implements the ESA decision logic by adding working memory and emotional
persistence to the standard IslandEntity.
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
from examples.concordia_island.sim import daily_self_perception as daily_self_perception_lib
from examples.concordia_island.sim import emotion_processing_components
from examples.concordia_island.sim import experience_reflection as experience_reflection_lib
from examples.concordia_island.sim import important_memories as important_memories_lib
from examples.concordia_island.sim import periodic_working_memory as periodic_working_memory_lib
from examples.concordia_island.sim import schedule_awareness as schedule_awareness_lib


@dataclasses.dataclass
class ESAIslandEntity(prefab_lib.Prefab):
  """Prefab for Enacted Self Agent (ESA) island simulation entities.

  Adds WorkingMemory and Emotion components to the IslandEntity stack.
  """

  description: str = "A person with working memory and emotional persistence"
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
          "use_working_memory": True,
          "use_emotion": True,
          "tick_interval_minutes": 120,
          "experience_sampling": True,
          "big_five": "bfi10",
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

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

    use_working_memory = self.params.get("use_working_memory", True)
    use_emotion = self.params.get("use_emotion", True)
    tick_interval = self.params.get("tick_interval_minutes", 120)
    ticks_per_day = max(1, (16 * 60) // tick_interval)
    experience_sampling = self.params.get("experience_sampling", True)
    big_five = self.params.get("big_five", "bfi10")

    # Seed memory
    if age is not None:
      memory_bank.add(f"[self] {name} is {age} years old.")
    if personality:
      memory_bank.add(f"[self] {name} is {personality}.")
    if backstory:
      memory_bank.add(f"[background] {backstory}")
    if traits and not remove_traits_from_memory:
      if isinstance(traits, str):
        memory_bank.add(f"{name}'s psychological profile:\n{traits}")
      else:
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
        memory_bank.add(f"{name}'s psychological profile:\n{trait_text}")
    for memory in memories:
      memory_bank.add(memory)
    for memory in formative_memories:
      if not memory.startswith("["):
        memory_bank.add(f"[formative] {memory}")
      else:
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

    components = {
        instructions_key: instructions,
        location_info_key: location_info_component,
        schedule_awareness_key: schedule_awareness,
        observation_to_memory_key: observation_to_memory,
        observation_component_key: observation,
        current_location_key: current_location_component,
    }

    relevant_memories_key = "RelevantMemories"
    relevant_memories = (
        agent_components.all_similar_memories.AllSimilarMemories(
            model=model,
            components=[observation_component_key],
        )
    )
    components[relevant_memories_key] = relevant_memories

    if use_working_memory:
      working_memory = periodic_working_memory_lib.PeriodicWorkingMemory(
          model=model,
          memory_component_key="__memory__",
          components=[observation_component_key, relevant_memories_key],
          update_interval_ticks=4,
      )
      components["WorkingMemory"] = working_memory

    if use_emotion:
      emotion_experience = emotion_processing_components.EmotionalExperience(
          model=model,
          agent_name=name,
          relevant_memories_component_key=relevant_memories_key,
          observation_component_key=observation_component_key,
      )
      components["EmotionalExperience"] = emotion_experience

      emotion_expression = emotion_processing_components.EmotionalExpression(
          model=model,
          agent_name=name,
          emotional_experience_component_key="EmotionalExperience",
      )
      components["EmotionalExpression"] = emotion_expression

    situation_perception_key = "SituationPerception"
    situation_perception = question_of_recent_memories.SituationPerception(
        model=model,
        num_memories_to_retrieve=10,
        pre_act_label=(
            f"\nQuestion: What situation is {name} in right now?\nAnswer"
        ),
    )
    components[situation_perception_key] = situation_perception

    self_perception_key = "SelfPerception"
    self_perception = daily_self_perception_lib.DailySelfPerception(
        model=model,
        num_memories_to_retrieve=10,
        components=[
            situation_perception_key,
        ],
        observation_component_key=observation_component_key,
        pre_act_label=f"\nQuestion: What kind of person is {name}?\nAnswer",
    )
    components[self_perception_key] = self_perception

    available_str = ", ".join(available_locations[:15])
    movement_decision_key = "MovementDecision"
    movement_decision = question_of_recent_memories.QuestionOfRecentMemories(
        model=model,
        question=(
            "Given {agent_name}'s current situation and goals, should"
            " {agent_name} stay at their current location or move to a"
            f" different place? Available locations: {available_str}."
            " Consider what {agent_name} wants to accomplish next and"
            " whether it requires going somewhere else."
        ),
        answer_prefix="{agent_name} should ",
        add_to_memory=False,
        num_memories_to_retrieve=10,
        components=[
            situation_perception_key,
            self_perception_key,
        ],
        pre_act_label=(
            f"\nQuestion: Should {name} stay or move to a different"
            " location?\nAnswer"
        ),
    )
    components[movement_decision_key] = movement_decision

    person_by_situation_key = "PersonBySituation"
    person_by_situation = question_of_recent_memories.PersonBySituation(
        model=model,
        num_memories_to_retrieve=5,
        components=[
            self_perception_key,
            situation_perception_key,
            movement_decision_key,
        ],
        pre_act_label=(
            f"\nQuestion: What would a person like {name} do in "
            "a situation like this?\nAnswer"
        ),
    )
    components[person_by_situation_key] = person_by_situation

    memory_component_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = agent_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )
    components[memory_component_key] = memory_component

    component_order = [
        instructions_key,
        location_info_key,
        schedule_awareness_key,
        observation_component_key,
        current_location_key,
    ]

    if use_working_memory:
      component_order.append("WorkingMemory")

    if use_emotion:
      component_order.append("EmotionalExperience")
      component_order.append("EmotionalExpression")

    component_order.extend([
        situation_perception_key,
        self_perception_key,
        movement_decision_key,
        person_by_situation_key,
    ])

    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        component_order=component_order,
    )

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

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components,
        measurements=async_measurements_lib.ReactiveMeasurements(),
    )
