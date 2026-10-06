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

"""Island Convo Entity prefab for Concordia simulation.

Extends the standard island entity with conversation-enriching components:
- PinkNoiseStrategy: Decides whether to converge or diverge in conversations
- LastSentence: Identifies what to respond to in the last utterance
- SwitchingActComponent: Routes between basic (movement/choices) and enriched
  (conversation) component orders based on ActionSpec output type

Use with --convo_agent=true flag to enable richer, more dynamic conversations.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import List

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import concat_act_component
from concordia.components.agent import question_of_recent_memories
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging as logging_lib
from concordia.typing import prefab as prefab_lib
from concordia.utils import async_measurements as async_measurements_lib

from examples.concordia_island.prefabs import island_entity
from examples.concordia_island.sim import experience_reflection as experience_reflection_lib
from examples.concordia_island.sim import important_memories as important_memories_lib
from examples.concordia_island.sim import schedule_awareness as schedule_awareness_lib


class PinkNoiseStrategy(question_of_recent_memories.QuestionOfRecentMemories):
  """Decides conversational strategy: converge (stay on topic) or diverge.

  Based on 'pink noise' dynamics — balancing stability with flexibility
  to keep conversations engaging rather than repetitive.
  """

  def __init__(
      self,
      agent_name: str,
      model: language_model.LanguageModel,
      components: List[str],
      **kwargs,
  ):
    question = (
        f"As {agent_name}, your goal is to maintain an engaging conversation."
        " This means balancing stability (staying on topic) with flexibility"
        " (introducing new, related ideas). Review the recent conversation."
        " Has the immediate micro-topic become interesting or repetitive?"
        " Based on this, choose a strategy for what to say next:\nA."
        " **Converge:** Stay on the micro-topic to deepen the conversation for"
        " several turns. Choose this if the topic has more to explore.\nB."
        " **Diverge:** Broaden the topic by connecting it to a more abstract"
        " theme, a related personal anecdote, or a question about them. Choose"
        " this if the current micro-topic is becoming repetitive after several"
        " turns.\n Don't diverge too much, and don't introduce too many new"
        " micro-topics. You should aim to stay on the current micro-topic for"
        " a few turns, and then diverge."
    )
    default_pre_act_label = "\n--- Conversational Strategy ---\n{question}"

    if kwargs.get("pre_act_label") is None:
      kwargs["pre_act_label"] = default_pre_act_label

    super().__init__(
        question=question,
        model=model,
        answer_prefix="",
        add_to_memory=False,
        memory_tag="[pink noise strategy]",
        components=components,
        **kwargs,
    )


class SwitchingActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
  """Routes between basic and conversation-enriched component orders.

  For CHOICE action types (movement, multiple choice), uses the basic
  component order — same as the standard island entity.

  For FREE output (conversation turns), uses the enriched order with
  LastSentence and PinkNoiseStrategy appended.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      basic_component_order: List[str],
      convo_component_order: List[str],
  ):
    super().__init__()
    self._basic_component_order = basic_component_order
    self._convo_component_order = convo_component_order
    self._basic_act = concat_act_component.ConcatActComponent(
        model=model,
        component_order=basic_component_order,
    )
    self._convo_act = concat_act_component.ConcatActComponent(
        model=model,
        component_order=convo_component_order,
        prefix_entity_name=False,
    )

  def set_entity(
      self, entity: entity_agent_with_logging.EntityAgentWithLogging
  ):
    super().set_entity(entity)
    self._basic_act.set_entity(entity)
    self._convo_act.set_entity(entity)

  def set_logging_channel(self, logging_channel: logging_lib.LoggingChannel):
    super().set_logging_channel(logging_channel)
    self._basic_act.set_logging_channel(logging_channel)
    self._convo_act.set_logging_channel(logging_channel)

  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    if (
        action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES
        or getattr(action_spec, "tag", "") != "speech"
    ):
      delegate = self._basic_act
      filtered_contexts = {
          key: contexts[key]
          for key in self._basic_component_order
          if key in contexts
      }
      result = delegate.get_action_attempt(filtered_contexts, action_spec)
    else:
      delegate = self._convo_act
      # Ensure all contexts are strings to prevent [object Object] issues
      stringified_contexts = {k: str(v) for k, v in contexts.items()}
      result = delegate.get_action_attempt(stringified_contexts, action_spec)
    self._logging_channel({"Summary": f"Action: {result}", "Value": result})
    return result

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass


@dataclasses.dataclass
class IslandConvoEntity(prefab_lib.Prefab):
  """Island entity with conversation-enriching components.

  Identical to IslandEntity for movement, perception, and daily routines.
  During conversations (FREE output), adds:
  - LastSentence: Identifies what to respond to in the last utterance
  - PinkNoiseStrategy: Decides whether to converge or diverge

  Params: Same as IslandEntity.
  """

  description: str = (
      "An island resident with enriched conversation capabilities"
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

    # --- Seed memories (identical to island_entity) ---
    hardcoded_formative = []
    if age is not None:
      hardcoded_formative.append(f"[self] {name} is {age} years old.")
    if personality:
      hardcoded_formative.append(f"[self] {name} is {personality}.")
    if backstory:
      hardcoded_formative.append(f"[background] {backstory}")
    if traits and not remove_traits_from_memory:
      if isinstance(traits, str):
        hardcoded_formative.append(f"{name}'s psychological profile:\n{traits}")
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
        hardcoded_formative.append(
            f"{name}'s psychological profile:\n{trait_text}"
        )

    for memory in hardcoded_formative:
      memory_bank.add(memory)

    for memory in memories:
      memory_bank.add(memory)
    for memory in formative_memories:
      if not memory.startswith("["):
        memory_bank.add(f"[formative] {memory}")
      else:
        memory_bank.add(memory)
    if initial_observation:
      memory_bank.add(initial_observation)

    # --- Standard island entity components ---
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
        pre_act_label=f"\n{name}'s current schedule",
    )

    observation_to_memory_key = "observation_to_memory"
    observation_to_memory = agent_components.observation.ObservationToMemory()

    observation_component_key = (
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = important_memories_lib.ImportantMemories(
        recent_history_length=50,
        formative_memories=hardcoded_formative,
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

    situation_perception_key = "SituationPerception"
    situation_perception = question_of_recent_memories.SituationPerception(
        model=model,
        num_memories_to_retrieve=25,
        components=[observation_component_key],
        pre_act_label=(
            f"\nQuestion: What situation is {name} in right now?\nAnswer"
        ),
    )

    self_perception_key = "SelfPerception"
    self_perception = question_of_recent_memories.SelfPerception(
        model=model,
        num_memories_to_retrieve=25,
        components=[
            situation_perception_key,
            life_altering_events_key,
            observation_component_key,
        ],
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
            life_altering_events_key,
        ],
        pre_act_label=(
            f"\nQuestion: What would a person like {name} do in "
            "a situation like this?\nAnswer"
        ),
    )

    tick_interval = self.params.get("tick_interval_minutes", 120)
    ticks_per_day = max(1, (16 * 60) // tick_interval)
    experience_sampling = self.params.get("experience_sampling", True)
    big_five = self.params.get("big_five", "bfi10")

    # --- Conversation-enriching components ---
    last_sentence_key = "LastSentence"
    last_sentence = question_of_recent_memories.QuestionOfRecentMemories(
        model=model,
        pre_act_label=(
            "\n--- Last Sentence ---\nQuestion: Is there something in the"
            f" last sentence in the conversation that {name} could respond"
            " to to move the conversation forward?\nAnswer"
        ),
        num_memories_to_retrieve=1,
        question=(
            "Is there something in the last sentence in the conversation"
            f" that {name} could respond to to move the conversation"
            " forward?"
        ),
        answer_prefix="",
        add_to_memory=False,
    )

    pink_noise_strategy_key = "PinkNoiseStrategy"
    pink_noise_strategy = PinkNoiseStrategy(
        model=model,
        agent_name=name,
        components=[
            situation_perception_key,
            self_perception_key,
            observation_component_key,
            last_sentence_key,
        ],
        num_memories_to_retrieve=1,
    )

    # --- All components ---
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
        life_altering_events_key: life_altering_events,
        last_sentence_key: last_sentence,
        pink_noise_strategy_key: pink_noise_strategy,
        memory_component_key: memory_component,
    }

    if experience_sampling:
      experience_reflection_key = "ExperienceReflection"
      experience_reflection = experience_reflection_lib.ExperienceReflection(
          model=model,
          tasks=experience_reflection_lib.default_tasks(
              ticks_per_day=ticks_per_day, big_five=big_five
          )
          + experience_reflection_lib.ai_survey_tasks(frequency=ticks_per_day),
      )
      components[experience_reflection_key] = experience_reflection

    # Basic order: used for movement decisions and multiple-choice actions.
    basic_component_order = [
        instructions_key,
        location_info_key,
        schedule_awareness_key,
        observation_component_key,
        current_location_key,
        situation_perception_key,
        self_perception_key,
        movement_decision_key,
        person_by_situation_key,
        life_altering_events_key,
    ]

    # Convo order: used for free-form conversation turns.
    # Adds LastSentence and PinkNoiseStrategy to guide dialogue.
    convo_component_order = [
        instructions_key,
        schedule_awareness_key,
        observation_component_key,
        current_location_key,
        self_perception_key,
        situation_perception_key,
        person_by_situation_key,
        life_altering_events_key,
        last_sentence_key,
        pink_noise_strategy_key,
    ]

    act_component = SwitchingActComponent(
        model=model,
        basic_component_order=basic_component_order,
        convo_component_order=convo_component_order,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components,
        measurements=async_measurements_lib.ReactiveMeasurements(),
    )
