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

"""Island Entity prefab for Concordia simulation.

A conversational agent with location awareness, perception components,
and island-specific behaviors.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import re

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import question_of_recent_memories
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.utils import async_measurements as async_measurements_lib

from examples.concordia_island.sim import daily_self_perception as daily_self_perception_lib
from examples.concordia_island.sim import experience_reflection as experience_reflection_lib
from examples.concordia_island.sim import important_memories as important_memories_lib
from examples.concordia_island.sim import schedule_awareness as schedule_awareness_lib


class DynamicLocation(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """Tracks agent location by parsing GM-prepended location prefix.

  LocationAwareMakeObservation prepends `// location:` to observations.
  This component extracts the location from that prefix and returns it
  in pre_act, so the agent knows where it is.
  """

  def __init__(
      self,
      initial_location: str,
      pre_act_label: str = "\nCurrent location",
  ):
    super().__init__(pre_act_label)
    self._current_location = initial_location
    self._initial_location = initial_location
    self._last_logged_location = initial_location

  def pre_observe(self, observation: str) -> str:
    import logging as sys_logging  # pylint: disable=g-import-not-at-top

    sys_logging.info(
        "[DynamicLocation Debug] %s: pre_observe called with: %r",
        self.get_entity().name if hasattr(self, "get_entity") else "Unknown",
        observation,
    )
    match = re.match(r"^//\s*([a-z_0-9]+)[\s\[:]", observation)
    if match:
      new_location = match.group(1)
      if new_location == self._last_logged_location:
        summary = f"Location: {new_location}"
      else:
        summary = (
            f"Location changed: {self._last_logged_location} → {new_location}"
        )
      self._logging_channel({
          "Key": "DynamicLocation",
          "Summary": summary,
          "Value": new_location,
          "observation_prefix": observation[:80],
      })
      self._last_logged_location = new_location
      self._current_location = new_location
    return ""

  def _make_pre_act_value(self) -> str:
    return self._current_location

  def get_state(self) -> entity_component.ComponentState:
    return {
        "current_location": self._current_location,
        "initial_location": self._initial_location,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._current_location = state.get(
        "current_location", self._initial_location
    )
    self._initial_location = state.get(
        "initial_location", self._initial_location
    )


class TaggedMemories(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """Component that returns memories containing specific tags."""

  def __init__(
      self,
      tags: tuple[str, ...],
      memory_component_key: str = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
      pre_act_label: str = "\nTagged events",
  ):
    super().__init__(pre_act_label)
    self._tags = tags
    self._memory_component_key = memory_component_key

  def _make_pre_act_value(self) -> str:
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=agent_components.memory.Memory
    )
    tagged_memories = memory.scan(lambda m: any(tag in m for tag in self._tags))

    output = "\n".join(tagged_memories) + "\n"
    self._logging_channel({
        "Key": self.get_pre_act_label(),
        "Value": output.splitlines(),
        "count": len(tagged_memories),
    })
    return output

  def get_state(self) -> entity_component.ComponentState:
    return {
        "tags": self._tags,
        "memory_component_key": self._memory_component_key,
        "pre_act_label": self.get_pre_act_label(),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if "tags" in state:
      self._tags = state["tags"]
    if "memory_component_key" in state:
      self._memory_component_key = state["memory_component_key"]


@dataclasses.dataclass
class IslandEntity(prefab_lib.Prefab):
  """Prefab for island simulation entities.

  Creates a conversational agent with:
  - Memory component for observations
  - Observation history
  - Personality via instructions
  - SelfPerception, SituationPerception, PersonBySituation
  - Location constants and dynamic location tracking
  - Backstory + persona memories seeded into memory bank

  Params:
    name: Agent name
    personality: Personality description
    backstory: Background story (added to initial memory)
    home_place: Home location ID
    work_place: Work location ID (optional)
    available_locations: List of locations agent can visit
    initial_observation: First observation for the agent
    memories: List of persona memory strings to seed
  """

  description: str = "An island resident with perception and location awareness"
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

    memory_component_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = agent_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

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
    current_location_component = DynamicLocation(
        initial_location=home_place,
        pre_act_label=f"\n{name}'s current location",
    )

    situation_perception_key = "SituationPerception"
    situation_perception = question_of_recent_memories.SituationPerception(
        model=model,
        num_memories_to_retrieve=10,
        pre_act_label=(
            f"\nQuestion: What situation is {name} in right now?\nAnswer"
        ),
    )

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

    # Compute reflection period: approximately once per simulated day.
    # With 120-min ticks and ~16 waking hours, that's 8 ticks/day.
    tick_interval = self.params.get("tick_interval_minutes", 120)
    ticks_per_day = max(1, (16 * 60) // tick_interval)
    experience_sampling = self.params.get("experience_sampling", True)
    big_five = self.params.get("big_five", "bfi10")

    components = {
        instructions_key: instructions,
        location_info_key: location_info_component,
        schedule_awareness_key: schedule_awareness,
        observation_to_memory_key: observation_to_memory,
        observation_component_key: observation,
        current_location_key: current_location_component,
        situation_perception_key: situation_perception,
        self_perception_key: self_perception,
        movement_decision_key: movement_decision,
        person_by_situation_key: person_by_situation,
        memory_component_key: memory_component,
    }

    # Component order for building agent context prompts.
    # Used by both ConcatActComponent (for actions) and ExperienceReflection
    # (for questionnaires) so both see the full agent identity.
    component_order = [
        instructions_key,
        location_info_key,
        schedule_awareness_key,
        observation_component_key,
        current_location_key,
        situation_perception_key,
        self_perception_key,
        movement_decision_key,
        person_by_situation_key,
    ]

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
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components,
        measurements=async_measurements_lib.ReactiveMeasurements(),
    )
