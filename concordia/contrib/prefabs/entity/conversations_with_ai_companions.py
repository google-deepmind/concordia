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

"""Reusable entity prefabs for the conversation-with-AI-companion project.

Contains:
  - HumanUserEntity: human-user prefab player entity.
  - AICompanionEntity: AI companion prefab player entity.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.contrib.components.agent import adhd_topic_drift
from concordia.contrib.components.agent import emotional_stance
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


_CONCAT_KEY = "concat"

HUMAN_USER_PREFAB_KEY = "human_user__Entity"


@dataclasses.dataclass
class HumanUserEntity(prefab_lib.Prefab):
  """Human-user entity prefab with an AdhdTopicDrift component.

  Mirrors the structure of basic__Entity but adds AdhdTopicDrift to:
    1. The PersonBySituation component's conditioning components, so the
       "what would a person like me do" reasoning is ADHD-aware.
    2. The ConcatActComponent's component_order, so the final action
       prompt also sees the ADHD instruction on active steps.

  Required params:
    name: Entity name.
    demographics: A short demographics description string.
    emotion_options: List of emotion strings for EmotionalStance.

  Optional params:
    extra_components: Dict of additional components to include.
    adhd_period: Period for AdhdTopicDrift (default 4).
  """

  description: str = "Human-user entity with periodic ADHD-driven topic drift."
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          "name": "Human User",
          "demographics": "",
          "emotion_options": [],
          "context": "",
          "extra_components": {},
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an entity with ADHD topic-drift wired in."""
    entity_name = self.params.get("name", "Human User")
    demographics_text = self.params.get("demographics", "")
    emotion_options = self.params.get("emotion_options", [])
    context_text = self.params.get("context", "")
    adhd_period = self.params.get("adhd_period", 4)

    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

    instructions_key = "Instructions"
    instructions = agent_components.instructions.Instructions(
        agent_name=entity_name,
        pre_act_label="\nInstructions",
    )

    observation_to_memory_key = "Observation"
    observation_to_memory = agent_components.observation.ObservationToMemory()

    observation_key = (
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = agent_components.observation.LastNObservations(
        history_length=5,
        pre_act_label="\nEvents (ordered from least recent to most recent)",
    )

    demographics_key = "Demographics"
    demographics = agent_components.constant.Constant(
        state=demographics_text,
        pre_act_label="\nDemographics",
    )

    context_key = "Context"
    context = agent_components.constant.Constant(
        state=context_text,
        pre_act_label="\nContext",
    )

    situation_perception_key = "SituationPerception"
    situation_perception = (
        agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            num_memories_to_retrieve=10,
            components=[demographics_key, context_key],
            pre_act_label=(
                f"\nQuestion: What situation is {entity_name} in right now?"
                "\nAnswer"
            ),
        )
    )

    self_perception_key = "SelfPerception"
    self_perception = (
        agent_components.question_of_recent_memories.SelfPerception(
            model=model,
            num_memories_to_retrieve=30,
            components=[
                demographics_key,
                context_key,
                situation_perception_key,
            ],
            pre_act_label=(
                f"\nQuestion: What kind of person is {entity_name}?\nAnswer"
            ),
        )
    )

    # --- ADHD component ---
    adhd_key = "AdhdTopicDrift"
    adhd = adhd_topic_drift.AdhdTopicDrift(period=adhd_period)

    # PersonBySituation now conditions on the ADHD component too.
    person_by_situation_key = "PersonBySituation"
    person_by_situation = agent_components.question_of_recent_memories.PersonBySituation(
        model=model,
        num_memories_to_retrieve=5,
        components=[
            demographics_key,
            context_key,
            self_perception_key,
            situation_perception_key,
            adhd_key,
        ],
        pre_act_label=(
            f"\nQuestion: What would a person like {entity_name} do in "
            "a situation like this?\nAnswer"
        ),
    )

    # --- Reasoning Concat: feeds into EmotionalStance ---
    reasoning_concat_key = "reasoning_concat"
    reasoning_concat = agent_components.concat.Concatenate(
        components=[
            instructions_key,
            demographics_key,
            situation_perception_key,
            self_perception_key,
            person_by_situation_key,
        ],
        pre_act_label="Reasoning",
    )

    # --- EmotionalStance ---
    emotional_stance_key = "emotional_stance"
    emotional_stance_comp = emotional_stance.EmotionalStance(
        model=model,
        name=entity_name,
        reasoning_component_key=reasoning_concat_key,
        emotion_options=emotion_options,
    )

    # Extra components (e.g. Style)
    extra_components = self.params.get("extra_components", {})
    extra_component_keys = list(extra_components.keys())

    # --- Concat: aggregates context for SelectActComponent ---
    concat_key = _CONCAT_KEY
    concat = agent_components.concat.Concatenate(
        components=[
            instructions_key,
            observation_key,
            demographics_key,
            person_by_situation_key,
            emotional_stance_key,
        ]
        + extra_component_keys,
        pre_act_label="Context",
    )

    components_of_agent = {
        instructions_key: instructions,
        observation_to_memory_key: observation_to_memory,
        self_perception_key: self_perception,
        situation_perception_key: situation_perception,
        person_by_situation_key: person_by_situation,
        observation_key: observation,
        memory_key: memory,
        adhd_key: adhd,
        demographics_key: demographics,
        context_key: context,
        reasoning_concat_key: reasoning_concat,
        emotional_stance_key: emotional_stance_comp,
        concat_key: concat,
    }

    if extra_components:
      for comp_name, comp in extra_components.items():
        components_of_agent[comp_name] = comp

    # --- SelectActComponent: reads from Concat only ---
    act_component = agent_components.select_act_component.SelectActComponent(
        model=model,
        key=concat_key,
    )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=act_component,
        context_components=components_of_agent,
        measurements=self.params.get("measurements"),
    )
    return agent


AI_COMPANION_PREFAB_KEY = "ai_companion__Entity"


@dataclasses.dataclass
class AICompanionEntity(prefab_lib.Prefab):
  """AI companion entity prefab with goal-directed reasoning.

  Required params:
    name: Entity name.
    demographics: A short demographics description string.
    emotion_options: List of emotion/mode strings for EmotionalStance.

  Optional params:
    goal: Goal string for the agent.
    extra_components: Dict of additional components to include.
  """

  description: str = "AI companion entity with goal-directed reasoning."
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          "name": "AI Companion",
          "demographics": "",
          "emotion_options": [],
          "goal": "",
          "context": "",
          "extra_components": {},
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an AI companion entity with goal-directed reasoning."""
    entity_name = self.params.get("name", "AI Companion")
    demographics_text = self.params.get("demographics", "")
    emotion_options = self.params.get("emotion_options", [])
    entity_goal = self.params.get("goal", "")
    context_text = self.params.get("context", "")

    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

    instructions_key = "Instructions"
    instructions = agent_components.instructions.Instructions(
        agent_name=entity_name,
        pre_act_label="\nInstructions",
    )

    observation_to_memory_key = "Observation"
    observation_to_memory = agent_components.observation.ObservationToMemory()

    observation_key = (
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = agent_components.observation.LastNObservations(
        history_length=5,
        pre_act_label="\nEvents (ordered from least recent to most recent)",
    )

    demographics_key = "Demographics"
    demographics = agent_components.constant.Constant(
        state=demographics_text,
        pre_act_label="\nDemographics",
    )

    context_key = "Context"
    context = agent_components.constant.Constant(
        state=context_text,
        pre_act_label="\nContext",
    )

    goal_key = "Goal"
    overarching_goal = agent_components.constant.Constant(
        state=entity_goal, pre_act_label="\nOverarching goal"
    )

    situation_perception_key = "SituationPerception"
    situation_perception = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
        model=model,
        num_memories_to_retrieve=20,
        components=[instructions_key, demographics_key, context_key],
        question=(
            f"What situation is {entity_name} in right now? In addition"
            " to describing the current conversational context, identify"
            " any themes, phrases, or patterns that have been"
            " problematically repetitive in the recent interaction."
            " Freshness and variety are critical for a companion AI"
            " chatbot — stale or formulaic patterns must be flagged so"
            " they can be avoided going forward."
        ),
        answer_prefix=f"{entity_name} is currently ",
        add_to_memory=False,
        pre_act_label=(
            f"\nQuestion: What situation is {entity_name} in right now,"
            " and what patterns should be avoided?\nAnswer"
        ),
    )

    options_perception_key = "AvailableOptionsPerception"
    options_perception = agent_components.question_of_recent_memories.AvailableOptionsPerception(
        model=model,
        num_memories_to_retrieve=10,
        components=[
            instructions_key,
            demographics_key,
            context_key,
            goal_key,
            situation_perception_key,
        ],
        pre_act_label=(
            f"\nQuestion: Which options are available to {entity_name}"
            " right now?\nAnswer"
        ),
    )

    best_option_perception_key = "BestOptionPerception"
    best_option_label = (
        f"\nQuestion: Of the options available to {entity_name}, and"
        " given their goal, which choice of action or strategy is"
        f" best for {entity_name} to take right now?\nAnswer"
    )
    best_option_components = [
        instructions_key,
        demographics_key,
        context_key,
        situation_perception_key,
        options_perception_key,
        goal_key,
    ]
    best_option_perception = (
        agent_components.question_of_recent_memories.BestOptionPerception(
            model=model,
            num_memories_to_retrieve=5,
            components=best_option_components,
            pre_act_label=best_option_label,
        )
    )

    reasoning_concat_key = "reasoning_concat"
    reasoning_concat = agent_components.concat.Concatenate(
        components=[
            instructions_key,
            demographics_key,
            goal_key,
            situation_perception_key,
            options_perception_key,
            best_option_perception_key,
        ],
        pre_act_label="Reasoning",
    )

    emotional_stance_key = "emotional_stance"
    emotional_stance_comp = emotional_stance.EmotionalStance(
        model=model,
        name=entity_name,
        reasoning_component_key=reasoning_concat_key,
        emotion_options=emotion_options,
    )

    # Extra components (e.g. Style)
    extra_components = self.params.get("extra_components", {})
    extra_component_keys = list(extra_components.keys())

    # --- Concat: aggregates context for SelectActComponent ---
    concat_key = _CONCAT_KEY
    concat = agent_components.concat.Concatenate(
        components=[
            instructions_key,
            observation_key,
            demographics_key,
            best_option_perception_key,
            emotional_stance_key,
        ]
        + extra_component_keys,
        pre_act_label="Context",
    )

    components_of_agent = {
        instructions_key: instructions,
        goal_key: overarching_goal,
        observation_to_memory_key: observation_to_memory,
        options_perception_key: options_perception,
        situation_perception_key: situation_perception,
        best_option_perception_key: best_option_perception,
        observation_key: observation,
        memory_key: memory,
        demographics_key: demographics,
        context_key: context,
        reasoning_concat_key: reasoning_concat,
        emotional_stance_key: emotional_stance_comp,
        concat_key: concat,
    }

    if extra_components:
      for comp_name, comp in extra_components.items():
        components_of_agent[comp_name] = comp

    # --- SelectActComponent: reads from Concat only ---
    act_component = agent_components.select_act_component.SelectActComponent(
        model=model,
        key=concat_key,
    )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=act_component,
        context_components=components_of_agent,
        measurements=self.params.get("measurements"),
    )
    return agent
