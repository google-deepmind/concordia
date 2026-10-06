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

"""Emotion processing components in Concordia."""

import copy
import json
import time
import typing
from typing import Any

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import all_similar_memories
from concordia.components.agent import memory as memory_component
from concordia.components.agent import observation as observation_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity_component


class EmotionalExperience(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A component that models an entity's internal emotional experiences.

  This component generates a first-person description of the agent's current
  emotional state based on recent observations and relevant memories,
  and adds this to the agent's memory.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      agent_name: str,
      *,
      relevant_memories_component_key: str = "RelevantMemories",
      observation_component_key: str = observation_component.DEFAULT_OBSERVATION_COMPONENT_KEY,
      pre_act_label: str = "Current Emotional Experience",
      emotion_history_length: int = 5,
      memory_component_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
      **kwargs,
  ):
    """Initializes the EmotionalExperience component.

    Args:
      model: The language model used to generate emotional experience.
      agent_name: The name of the agent this component belongs to.
      relevant_memories_component_key: Key for accessing relevant memories.
      observation_component_key: Key for accessing recent observations.
      pre_act_label: Label used in the pre-act output.
      emotion_history_length: Number of past emotions to keep in history.
      memory_component_key: Key for accessing the agent's associative memory.
      **kwargs: Additional keyword arguments for the base class.
    """
    super().__init__(pre_act_label=pre_act_label, **kwargs)
    self._model = model
    self._agent_name = agent_name
    self._relevant_memories_component_key = relevant_memories_component_key
    self._observation_component_key = observation_component_key
    self._emotion_history_length = emotion_history_length
    self._memory_component_key = memory_component_key
    self._emotion_history: list[str] | None = None
    self._current_experience: str | None = None

  def _generate_experience(self) -> str:
    """Generates a description of the current emotional experience."""
    relevant_memories_instance = self.get_entity().get_component(
        self._relevant_memories_component_key,
        type_=all_similar_memories.AllSimilarMemories,
    )
    observation_component_instance = self.get_entity().get_component(
        self._observation_component_key,
        type_=observation_component.LastNObservations,
    )

    relevant_memories = relevant_memories_instance.get_pre_act_value()
    if not relevant_memories:
      raise ValueError(
          f"{self._agent_name} [EmotionalExperience Component]: No"
          " relevant memories found."
      )

    recent_observations = observation_component_instance.get_pre_act_value()
    if not recent_observations:
      raise ValueError(
          f"{self._agent_name} [EmotionalExperience Component]: No"
          " recent observations found."
      )

    question = (
        f"Given {self._agent_name}'s current situation, past emotional"
        " experiences, personality, social context, and ongoing interactions,"
        f" what salient emotions is {self._agent_name} feeling right now?"
        " Respond in first person. For example, 'I am feeling somewhat amused"
        " but also a little frustrated.' Be concise unless the situation and"
        " your personality calls for verbosity. Response MUST begin with 'I am"
        " feeling'"
    )

    max_retries = 3
    retry_delay_seconds = 2.0

    for attempt in range(max_retries):
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(
          f"{self._agent_name}'s Context:\n"
          f"Relevant Memories: {relevant_memories}\n"
          f"Recent Observations: {recent_observations}\n"
      )
      if self._emotion_history:
        prompt.statement(
            f"Past emotional experiences: {', '.join(self._emotion_history)}\n"
        )

      response = prompt.open_question(question=question, max_tokens=400)

      if response:
        if not response.startswith("I am feeling"):
          logging.warning(
              "%s [EmotionalExperience Component]: LLM returned experience that"
              " does not start with 'I am feeling': '%s'.",
              self._agent_name,
              response,
          )
        return response

      logging.warning(
          "%s [EmotionalExperience Component]: LLM returned empty/invalid "
          "response (attempt %d/%d): %r. Retrying after %.1f seconds...",
          self._agent_name,
          attempt + 1,
          max_retries,
          response,
          retry_delay_seconds,
      )
      time.sleep(retry_delay_seconds)

    fallback_response = (
        "I am feeling uncertain about my current emotional state."
    )
    logging.warning(
        "%s [EmotionalExperience Component]: All %d retries failed, "
        "using fallback response: '%s'.",
        self._agent_name,
        max_retries,
        fallback_response,
    )
    return fallback_response

  def get_current_experience(self) -> str | None:
    """Returns the agent's current emotional experience."""
    if self._current_experience is None:
      logging.info(
          "%s [EmotionalExperience Component]: Initializing emotional"
          " experience based on premise",
          self._agent_name,
      )
      self._update_experience()
    return self._current_experience

  def _update_experience(self) -> None:
    """Generates and updates the current emotional experience."""
    new_experience = self._generate_experience()
    self._current_experience = new_experience

    # Add experience to history.
    if self._emotion_history:
      self._emotion_history.append(new_experience)
    else:
      self._emotion_history = [self._current_experience]
    if len(self._emotion_history) > self._emotion_history_length:
      self._emotion_history.pop(0)

    # Add experience to memory.
    memory_instance = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.AssociativeMemory
    )
    memory_instance.add(text=f"{self._current_experience}")

  def _make_pre_act_value(self) -> str:
    """Computes the value to be used in the pre-act stage."""
    self._update_experience()
    output = (
        # pyrefly: ignore[unsupported-operation]
        f"Current Emotion: {self._current_experience}\nPrevious Emotions:"
        f' {", ".join(self._emotion_history[:-1])}'
    )
    self._logging_channel({"Key": self.get_pre_act_label(), "Value": output})
    return output

  def reset_history(self) -> None:
    """Resets the emotion history and current experience.

    Call this between simulation episodes to prevent stale history from
    a prior conversation leaking into a new one.
    """
    self._emotion_history = None
    self._current_experience = None

  def get_state(self) -> entity_component.ComponentState:
    """Gets the component's state for serialization."""
    return {
        "_emotion_history": copy.copy(self._emotion_history),
        "_current_experience": self._current_experience,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the component's state from serialization."""
    self._emotion_history = typing.cast(
        list[str] | None, state["_emotion_history"]
    )
    if self._emotion_history is not None:
      self._emotion_history = list(self._emotion_history)
    self._current_experience = typing.cast(
        str | None, state["_current_experience"]
    )


class EmotionalExpression(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A component for modeling an entity's outward emotional expression.

  It generates a narrative description of the expression and parses it
  into structured channel data (e.g., facial expression, prosody).
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      agent_name: str,
      *,
      emotional_experience_component_key: str = "EmotionalExperience",
      pre_act_label: str = "Outward Emotional Expression",
      memory_component_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
      expression_modality_keys: list[str] | None = None,
      **kwargs,
  ):
    """Initializes the EmotionalExpression component.

    Args:
      model: The language model for generating expressions.
      agent_name: The name of the agent.
      emotional_experience_component_key: Key for EmotionalExperience component.
      pre_act_label: Label for the pre-act output.
      memory_component_key: Key for the agent's associative memory.
      expression_modality_keys: Optional list of keys for expression modalities.
      **kwargs: Additional keyword arguments.
    """
    super().__init__(pre_act_label=pre_act_label, **kwargs)
    self._model = model
    self._agent_name = agent_name
    self._emotional_experience_component_key = (
        emotional_experience_component_key
    )
    self._memory_component_key = memory_component_key
    self._current_narrative_expression: str | None = None
    self._expression_modalities: dict[str, str] = {}
    self._expression_modality_keys: list[str]
    if expression_modality_keys is None:
      self._expression_modality_keys = [
          "facial_expression",
          "prosody",
          "posture",
          "gestures",
          "other_cues",
      ]
    else:
      self._expression_modality_keys = expression_modality_keys

  def _generate_narrative_expression(self, current_emotion_text: str) -> str:
    """Generates a narrative description of the outward expression."""
    narrative_expression_example = (
        "Alice's brow furrows and their lips press together tightly. They"
        " tap their fingers impatiently on the table, and their voice has a"
        " sharp, slightly strained quality. Their posture is rigid."
    )
    narrative_question = (
        f"Given that {self._agent_name} is feeling '{current_emotion_text}',"
        " how do they outwardly express this? Describe their likely tone of"
        " voice, facial expressions, posture, gestures, proxemics, and any"
        " other observable behaviors related to emotion. This expression"
        f" should be consistent with {self._agent_name}'s personality, current"
        " social context, surroundings,and ongoing interactions. Provide a"
        " rich, descriptive paragraph. For example,"
        f" '{narrative_expression_example}'. Do not speculate. Do not invent"
        " emotional states not found in how they are are currently feeling. Be"
        " concise unless the situation and your personality calls for"
        " verbosity."
    )

    max_retries = 3
    retry_delay_seconds = 2.0

    for attempt in range(max_retries):
      narrative_expression_prompt = interactive_document.InteractiveDocument(
          self._model
      )
      narrative_expression = narrative_expression_prompt.open_question(
          question=narrative_question, max_tokens=400
      )

      if narrative_expression:
        return narrative_expression

      logging.warning(
          "%s [EmotionExpression Component]: LLM returned empty/invalid "
          "response (attempt %d/%d): %r. Retrying after %.1f seconds...",
          self._agent_name,
          attempt + 1,
          max_retries,
          narrative_expression,
          retry_delay_seconds,
      )
      time.sleep(retry_delay_seconds)

    fallback_expression = (
        f"{self._agent_name}'s expression appears neutral, with no obvious "
        "outward signs of strong emotion."
    )
    logging.warning(
        "%s [EmotionExpression Component]: All %d retries failed, "
        "using fallback expression: '%s'.",
        self._agent_name,
        max_retries,
        fallback_expression,
    )
    return fallback_expression

  def _parse_expression_modalities(
      self, narrative_expression: str
  ) -> dict[str, Any]:
    """Parses the narrative expression into structured expression modalities."""
    modality_keys_str = "; ".join(
        [f'"{i}"' for i in self._expression_modality_keys]
    )
    json_example_str = json.dumps(
        {key: "..." for key in self._expression_modality_keys}
    )
    prompt = f"""Analyze the following emotional expression: '{narrative_expression}'

Extract the key details for the following expression modalities: {modality_keys_str}.
For each modality, provide a description focusing on salient emotions, including the intensity and towards whom it may be directed.
Do not include descriptions of emotions or cues that are absent. If a modality is not explicitly mentioned, try to infer it from the overall description.
Do not use poetic language. Do not invent emotional states not present in the description. Do not narrate. Do not over-interpret.

Your response must be a JSON object where the keys are exactly the modality names provided. The structure of your output must match this example: {json_example_str}
The entire output should be ONLY the JSON object, starting with {{ and ending with }}.
"""
    parsed_expression_response = self._model.sample_text(prompt)
    clean_json_str = parsed_expression_response.strip()

    # Clean up markdown fences if present.
    if clean_json_str.startswith("```json"):
      clean_json_str = clean_json_str[7:].strip()
    if clean_json_str.endswith("```"):
      clean_json_str = clean_json_str[:-3].strip()
    if clean_json_str.startswith("{{") and clean_json_str.endswith("}}"):
      clean_json_str = clean_json_str[1:-1].strip()

    try:
      modality_data = json.loads(clean_json_str)
      # Also add the parsed raw string for debugging purposes.
      modality_data["expression_modalities"] = parsed_expression_response
      # Ensure all expected keys are present
      for key in self._expression_modality_keys:
        if key not in modality_data:
          modality_data[key] = "Not specified"
      return modality_data
    except (json.JSONDecodeError, ValueError) as e:
      logging.warning(
          "%s [Modality Parsing JSON Error]: %s - Raw: %s",
          self._agent_name,
          e,
          parsed_expression_response,
      )
      # Return raw response as is.
      modality_data = {"expression_modalities": parsed_expression_response}
      return modality_data

  def _make_pre_act_value(self) -> str:
    """Computes the value to be used in the pre-act stage."""
    emotion_expression_instance = self.get_entity().get_component(
        self._emotional_experience_component_key, type_=EmotionalExperience
    )
    current_emotion_text = emotion_expression_instance.get_current_experience()

    if not current_emotion_text:
      raise ValueError(
          f"{self._agent_name} [EmotionExpression Component]: No current"
          " emotion. This should not happen."
      )

    # 1. Generate Narrative Emotional Expression
    self._current_narrative_expression = self._generate_narrative_expression(
        current_emotion_text
    )

    # Add expression to memory.
    memory_instance = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.AssociativeMemory
    )
    memory_instance.add(
        text=(
            "Based on what I was feeling, my outward emotional expression was:"
            f" {self._current_narrative_expression}"
        )
    )

    # 2. Parse to expression modalities
    self._expression_modalities = self._parse_expression_modalities(
        self._current_narrative_expression
    )

    self._logging_channel({
        "Key": self.get_pre_act_label(),
        "Expression Narrative": self._current_narrative_expression,
        "Expression Modalities": self._expression_modalities,
    })
    return f"Expressing: {self._current_narrative_expression}"

  def get_expression_modalities(self) -> dict[str, str]:
    """Returns the extracted expression modalities."""
    if not self._expression_modalities:
      raise ValueError(
          f"{self._agent_name} [EmotionExpression Component]: No expression"
          " modalities. This should not happen."
      )
    return self._expression_modalities

  def get_current_narrative_expression(self) -> str | None:
    """Returns the current narrative description of the expression."""
    return self._current_narrative_expression

  def get_state(self) -> entity_component.ComponentState:
    """Gets the component's state for serialization."""
    return {
        "_current_narrative_expression": self._current_narrative_expression,
        "_expression_modalities": self._expression_modalities,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the component's state from serialization."""
    self._current_narrative_expression = typing.cast(
        str | None, state["_current_narrative_expression"]
    )
    self._expression_modalities = typing.cast(
        dict[str, str], state["_expression_modalities"]
    )
