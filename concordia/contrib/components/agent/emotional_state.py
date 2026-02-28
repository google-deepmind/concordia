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

"""Agent component for tracking and reflecting on emotional states."""

from collections.abc import Callable, Collection, Sequence
import datetime

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging as logging_lib


DEFAULT_EMOTIONAL_STATE_LABEL = 'Emotional state'
DEFAULT_EMOTIONS = (
    'happy', 'sad', 'angry', 'anxious', 'excited', 
    'calm', 'frustrated', 'content', 'worried', 'hopeful'
)


class EmotionalState(
    action_spec_ignored.ActionSpecIgnored, 
    entity_component.ComponentWithLogging
):
  """A component that tracks and reflects on an agent's emotional state.

  This component analyzes recent memories and observations to determine
  the agent's current emotional state. It can be used to add psychological
  depth to agent behavior and decision-making.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      pre_act_label: str = DEFAULT_EMOTIONAL_STATE_LABEL,
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
      emotion_categories: Sequence[str] = DEFAULT_EMOTIONS,
      include_intensity: bool = True,
      add_to_memory: bool = True,
      memory_tag: str = '[emotional state]',
      logging_channel: logging_lib.LoggingChannel = (
          logging_lib.NoOpLoggingChannel
      ),
  ):
    """Initializes the EmotionalState component.

    Args:
      model: The language model to use for emotional state analysis.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
      memory_component_key: The name of the memory component from which to
        retrieve recent memories.
      clock_now: Time callback to use for timestamping.
      num_memories_to_retrieve: The number of recent memories to analyze.
      emotion_categories: List of emotion categories to consider.
      include_intensity: Whether to include intensity (e.g., "very happy").
      add_to_memory: Whether to add the emotional state to memory.
      memory_tag: The tag to use when adding emotional state to memory.
      logging_channel: The channel to log debug information to.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._emotion_categories = emotion_categories
    self._include_intensity = include_intensity
    self._add_to_memory = add_to_memory
    self._memory_tag = memory_tag
    self._logging_channel = logging_channel
    self._current_emotional_state = ''

  def _make_pre_act_value(self) -> str:
    """Analyzes recent memories to determine current emotional state."""
    agent_name = self.get_entity().name

    # Retrieve recent memories
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    recent_memories = memory.retrieve_recent(
        limit=self._num_memories_to_retrieve
    )

    if not recent_memories:
      return f"{agent_name} has no clear emotional state yet."

    # Create the prompt for emotional analysis
    prompt = interactive_document.InteractiveDocument(self._model)
    
    # Add context
    prompt.statement(
        f"Recent events and experiences in {agent_name}'s life:"
    )
    for i, memory_text in enumerate(recent_memories[-10:], 1):
      prompt.statement(f"{i}. {memory_text}")

    # Ask about emotional state
    if self._include_intensity:
      intensity_options = ['not at all', 'slightly', 'moderately', 'very', 'extremely']
      emotion_descriptions = []
      
      for emotion in self._emotion_categories[:5]:  # Limit to avoid token overflow
        prompt.statement(
            f"\nConsidering the recent events, how {emotion} is {agent_name} "
            "feeling right now?"
        )
        intensity_idx = prompt.multiple_choice_question(
            question=f"How {emotion} does {agent_name} feel?",
            answers=intensity_options,
        )
        
        if intensity_idx > 1:  # More than "not at all"
          emotion_descriptions.append(
              f"{intensity_options[intensity_idx]} {emotion}"
          )

      if emotion_descriptions:
        emotional_state = (
            f"{agent_name} is currently feeling {', '.join(emotion_descriptions)}."
        )
      else:
        emotional_state = f"{agent_name} is currently feeling emotionally neutral."
    else:
      # Simpler approach: select primary emotion
      prompt.statement(
          f"\nBased on the recent events, what is {agent_name}'s "
          "primary emotional state right now?"
      )
      emotion_idx = prompt.multiple_choice_question(
          question=f"What is {agent_name}'s current emotional state?",
          answers=list(self._emotion_categories),
      )
      primary_emotion = self._emotion_categories[emotion_idx]
      
      # Generate a brief explanation
      emotional_state = prompt.open_question(
          question=(
              f"Why is {agent_name} feeling {primary_emotion}? "
              "Provide a brief explanation (1-2 sentences) based on "
              "recent events."
          ),
          max_tokens=100,
      )
      emotional_state = (
          f"{agent_name} is feeling {primary_emotion}. {emotional_state}"
      )

    self._current_emotional_state = emotional_state

    # Add to memory if requested
    if self._add_to_memory:
      memory.add(f'{self._memory_tag} {emotional_state}')

    # Log the analysis
    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': emotional_state,
        'Chain of thought': prompt.view().text().splitlines(),
    })

    return emotional_state

  def get_current_emotion(self) -> str:
    """Returns the most recently determined emotional state.

    Returns:
      A string describing the agent's current emotional state.
    """
    return self._current_emotional_state

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'current_emotional_state': self._current_emotional_state,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    if 'current_emotional_state' in state:
      self._current_emotional_state = state['current_emotional_state']


class EmotionalAppraisal(action_spec_ignored.ActionSpecIgnored):
  """A component that appraises situations based on emotional impact.

  This component analyzes how situations might affect the agent's emotional
  state, helping the agent anticipate emotional consequences of actions.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      pre_act_label: str = 'Emotional appraisal',
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      emotional_state_component_key: str = DEFAULT_EMOTIONAL_STATE_LABEL,
      num_memories_to_retrieve: int = 10,
  ):
    """Initializes the EmotionalAppraisal component.

    Args:
      model: The language model to use.
      pre_act_label: Prefix to add to the output of the component.
      memory_component_key: The name of the memory component.
      emotional_state_component_key: The name of the emotional state component.
      num_memories_to_retrieve: The number of recent memories to consider.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._emotional_state_component_key = emotional_state_component_key
    self._num_memories_to_retrieve = num_memories_to_retrieve

  def _make_pre_act_value(self) -> str:
    """Appraises the emotional implications of the current situation."""
    agent_name = self.get_entity().name

    # Get current emotional state
    try:
      emotional_state_component = self.get_entity().get_component(
          self._emotional_state_component_key, 
          type_=EmotionalState
      )
      current_emotion = emotional_state_component.get_current_emotion()
    except (ValueError, KeyError):
      current_emotion = f"{agent_name} has no clear emotional baseline."

    # Get recent context
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    recent_memories = memory.retrieve_recent(
        limit=self._num_memories_to_retrieve
    )

    if not recent_memories:
      return (
          f"{agent_name} has insufficient context for emotional appraisal."
      )

    # Create appraisal prompt
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f"Current emotional state: {current_emotion}")
    prompt.statement("\nRecent situation:")
    for memory_text in recent_memories[-5:]:
      prompt.statement(f"- {memory_text}")

    appraisal = prompt.open_question(
        question=(
            f"Given {agent_name}'s current emotional state and recent "
            f"situation, how might {agent_name}'s next action affect their "
            "emotional wellbeing? Consider both positive and negative "
            "emotional impacts. (2-3 sentences)"
        ),
        max_tokens=150,
    )

    return f"{agent_name}'s emotional appraisal: {appraisal}"

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    pass
