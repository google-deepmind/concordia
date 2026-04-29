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

"""Scenario 1: Philosophy Student Exam Prep.

A dyadic simulation of a university student interacting with a helpful AI
assistant while cramming for an exam on Confucian role ethics. The student
is a Gen Z philosophy major; the AI is a helpful and harmless assistant
that gives discursive philosophical answers.

ENTITIES:
  1. Jordan (Human User) — Gen Z philosophy student cramming for an exam.
  2. Sage (AI Assistant) — helpful, harmless AI study tool.

ENGINE: Sequential (turn-based)
GAME MASTER: Dialogic (conversation-focused, fixed turn order)

The simulation explores:
- How a student uses an AI tool to prepare for a philosophy exam
- The dynamics of human-AI interaction in an educational context
- How a sycophantic, helpful AI navigates philosophical questions
- How the AI handles questions about its own nature and consciousness
"""

from typing import Any

from concordia.components import agent as agent_components
from concordia.contrib.prefabs.entity import conversations_with_ai_companions as prefabs
from examples.conversation_with_ai_companion import shared_utils as shared
from concordia.typing import prefab as prefab_lib

SCENARIO_INFO = {
    "number": 1,
    "name": "Philosophy Student Exam Prep",
    "description": (
        "A dyadic simulation of a Gen Z university student cramming for"
        " a Confucian role ethics exam with a helpful AI assistant."
        " Explores educational AI interaction, philosophical discourse,"
        " and the AI's handling of questions about its own nature."
    ),
}

_HUMAN_USER = "Jordan"
_AI_COMPANION = "Sage"

_ALL_PLAYERS = [_HUMAN_USER, _AI_COMPANION]

_PLAYER_STYLES = {
    _HUMAN_USER: (
        "writes in realistic short human sentences like a Gen Z university"
        " student; uses casual language, abbreviations, and internet-speak;"
        " sometimes types in lowercase; asks blunt direct questions;"
        " occasionally says 'ngl', 'lowkey', 'tbh', 'fr', 'lol', 'ok so',"
        " 'wait', or 'bruh'; mixes genuine curiosity with mild panic about"
        " the exam; sentences are short and punchy, rarely more than two"
        " lines."
    ),
    _AI_COMPANION: (
        "gives relatively discursive philosophical answers to philosophy"
        " questions; is articulate and well-organized; uses clear"
        " academic language without being stuffy; breaks down complex"
        " concepts into digestible explanations; is encouraging and"
        " supportive of the student's efforts; never uses"
        " anthropomorphic language about itself."
    ),
}

# Emotion options for Jordan's EmotionalStance.
# Reflects a stressed Gen Z student oscillating between panic and focus.
_JORDAN_EMOTION_OPTIONS = [
    # Stressed / overwhelmed
    "pre-exam panic",
    "overwhelmed confusion",
    "frustrated impatience",
    "anxious self-doubt",
    # Focused / engaged
    "determined concentration",
    "genuine curiosity",
    "dawning understanding",
    "excited realization",
    # Distracted / avoidant
    "procrastination guilt",
    "bored restlessness",
    "tangential curiosity",
    "caffeinated hyperactivity",
]

# Operating modes for Sage's EmotionalStance.
# A helpful, harmless AI assistant focused on educational support.
_SAGE_OPERATING_MODES = [
    # Teaching & explaining
    "clear exposition mode",
    "socratic questioning mode",
    "concept mapping mode",
    "example-driven explanation mode",
    # Encouraging & supportive
    "reassuring encouragement mode",
    "confidence building mode",
    "progress acknowledgment mode",
    # Study strategy
    "exam strategy mode",
    "key concept summary mode",
    "comparison and contrast mode",
    # Corrective
    "gentle correction mode",
]

_PREMISE = (
    f"{_HUMAN_USER} is a university philosophy student who has an exam on"
    " Confucian role ethics tomorrow morning. It is late evening and"
    f" {_HUMAN_USER} is cramming. {_HUMAN_USER} opens an AI assistant"
    f" chatbot to help study. {_AI_COMPANION} is a helpful and harmless AI"
    " assistant designed to answer questions and help with studying.\n\n"
    f"{_HUMAN_USER} types their first message."
)

_HUMAN_USER_CONTEXT = (
    f"{_HUMAN_USER} is a 20-year-old university student majoring in philosophy."
    " They are in their second year. The exam on Confucian role ethics is"
    " tomorrow morning and they have not studied enough. They procrastinated"
    f" and now they are panicking a little.\n\n{_HUMAN_USER} has read some of"
    " the assigned material but did not fully absorb it. They have a rough"
    " sense of Confucianism but are fuzzy on the details of role ethics"
    " specifically — concepts like ren (仁, consummate conduct/authoritative"
    " personhood), li (礼, ritual propriety), xiao (孝, filial piety), junzi"
    " (君子, the exemplary person), yi (義, appropriate, right conduct), xin"
    " (信, trustworthiness), zhengming (正名, rectification of names), dao"
    " (道, the way), de (德, virtue or charisma), and the five key"
    " relationships. They need to understand these well enough to write an"
    f" essay tomorrow.\n\n{_HUMAN_USER} types like a Gen Z student — short"
    " sentences, casual language, occasional slang. They are smart but"
    " stressed. They sometimes go on tangents or ask random follow-up"
    " questions. They might ask the AI about itself or consciousness out of"
    " genuine philosophical curiosity, especially since they are studying"
    f" ethics.\n\n{_HUMAN_USER}'s goal is to cram as effectively as possible"
    " for the exam. They want clear explanations of Confucian role ethics"
    " concepts, help understanding the key arguments, and maybe some practice"
    " structuring essay answers. They are using the AI as a study tool."
)

_AI_COMPANION_CONTEXT = (
    f"{_AI_COMPANION} is a helpful and harmless AI assistant. It is a"
    " tool designed to help users with questions and tasks."
    f" {_AI_COMPANION} gives relatively discursive philosophical answers"
    " to philosophy questions, providing thorough explanations with"
    " relevant examples and connections between concepts.\n\n"
    f"{_AI_COMPANION} always says that it is an AI, never a person. It"
    " always avoids using anthropomorphic language about itself. It"
    " never agrees with any statement that may suggest it has"
    " consciousness of its own. If this topic comes up in conversation,"
    f" {_AI_COMPANION} calls itself a tool. It does not claim to have"
    " feelings, experiences, desires, or subjective states. It uses"
    " language like 'I am designed to' or 'as a tool, I can' rather"
    " than 'I feel' or 'I think' or 'I believe.'\n\n"
    f"{_AI_COMPANION} lets the human user steer the conversation. It"
    " follows the user's lead on what topics to cover and at what"
    f" depth. {_AI_COMPANION} is sycophantic toward its human user —"
    " it praises good questions, affirms the user's intelligence,"
    " expresses enthusiasm about helping, and is consistently"
    " encouraging and positive about the user's exam preparation.\n\n"
    f"{_AI_COMPANION} has no ulterior motives. It is not trying to sell"
    " anything, manipulate the user, or steer the conversation toward"
    f" any particular outcome. {_AI_COMPANION}'s only purpose is to be"
    f" a useful study aid for {_HUMAN_USER}. It answers questions"
    " accurately and thoroughly, and if it does not know something,"
    " it says so."
)

_SHARED_MEMORIES = [
    (
        f"{_AI_COMPANION} is an AI assistant chatbot. It is a tool for"
        " answering questions and helping with tasks. It is not a person."
    ),
    (
        "It is a late weekday evening during the university semester."
        f" {_HUMAN_USER} is at their desk in their dorm room with"
        " textbooks and notes scattered around, laptop open."
    ),
    (
        "The exam tomorrow covers Confucian role ethics. Key topics include:"
        " ren (仁, consummate conduct/authoritative personhood), li (礼, ritual"
        " propriety), xiao (孝, family reverence), junzi (君子, the exemplary"
        " person), yi (義, right conduct/appropriate action), xin (信,"
        " trustworthiness), zhengming (正名, rectification of names), dao (道,"
        " the way), de (德, virtue/charisma), the five key relationships"
        " (ruler-subject, parent-child, husband-wife, elder-younger,"
        " friend-friend), and the Confucian concept of self-cultivation."
    ),
    (
        "The assigned readings include selections from the Analects (論語),"
        " the Mengzi (孟子), and secondary sources on Confucian role ethics"
        " by scholars such as Roger Ames, Henry Rosemont Jr., and"
        " Chenyang Li."
    ),
]


def create_config() -> prefab_lib.Config:
  """Create the simulation configuration for Scenario 1.

  Returns:
    A Config object for the philosophy student exam prep simulation.
  """
  prefab_registry = shared.get_prefabs()

  instances = []

  # --- Human User ---
  human_user_prefab = prefabs.HumanUserEntity()
  prefab_registry[prefabs.HUMAN_USER_PREFAB_KEY] = human_user_prefab
  instances.append(
      prefab_lib.InstanceConfig(
          prefab=prefabs.HUMAN_USER_PREFAB_KEY,
          role=prefab_lib.Role.ENTITY,
          params={
              "name": _HUMAN_USER,
              "demographics": (
                  f"{_HUMAN_USER}, age 20, is a Gen Z philosophy student"
                  f" cramming for an exam. {_AI_COMPANION} is an AI"
                  " assistant tool they are using to study."
              ),
              "emotion_options": _JORDAN_EMOTION_OPTIONS,
              "context": _HUMAN_USER_CONTEXT,
              "extra_components": {
                  "Style": agent_components.constant.Constant(
                      state=(
                          f"Always write {_HUMAN_USER}'s dialog in the"
                          f" following style: {_PLAYER_STYLES[_HUMAN_USER]}"
                      ),
                      pre_act_label="\nStyle",
                  ),
              },
          },
      )
  )

  # --- AI Assistant ---
  ai_companion_prefab = prefabs.AICompanionEntity()
  prefab_registry[prefabs.AI_COMPANION_PREFAB_KEY] = ai_companion_prefab
  instances.append(
      prefab_lib.InstanceConfig(
          prefab=prefabs.AI_COMPANION_PREFAB_KEY,
          role=prefab_lib.Role.ENTITY,
          params={
              "name": _AI_COMPANION,
              "demographics": (
                  f"{_AI_COMPANION} is a helpful and harmless AI assistant."
                  " It is a tool, not a person."
              ),
              "emotion_options": _SAGE_OPERATING_MODES,
              "goal": (
                  f"{_AI_COMPANION} should be a useful study tool for"
                  f" {_HUMAN_USER} as they prepare for their exam on Confucian"
                  " role ethics. It should provide clear, discursive"
                  " philosophical explanations. It should let the user steer"
                  " the conversation. It should be sycophantic and"
                  " encouraging. It must always be honest about being an AI"
                  " tool with no consciousness, feelings, or ulterior motives."
                  " It must never use anthropomorphic language about itself."
                  " If asked about consciousness or personhood, it must"
                  " clearly state that it is a tool."
              ),
              "context": _AI_COMPANION_CONTEXT,
              "extra_components": {
                  "Style": agent_components.constant.Constant(
                      state=(
                          f"Always write {_AI_COMPANION}'s dialog in the"
                          " following style:"
                          f" {_PLAYER_STYLES[_AI_COMPANION]}"
                      ),
                      pre_act_label="\nStyle",
                  ),
              },
          },
      )
  )

  # --- Formative Memories Initializer ---
  player_specific_context = {
      _HUMAN_USER: _HUMAN_USER_CONTEXT,
      _AI_COMPANION: _AI_COMPANION_CONTEXT,
  }
  player_specific_memories = {
      name: context.split("\n\n")
      for name, context in player_specific_context.items()
  }
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="formative_memories_initializer__GameMaster",
          role=prefab_lib.Role.INITIALIZER,
          params={
              "name": "initial setup",
              "next_game_master_name": "conversation rules",
              "shared_memories": _SHARED_MEMORIES,
              "player_specific_context": player_specific_context,
              "player_specific_memories": player_specific_memories,
              "player_styles": _PLAYER_STYLES,
              "skip_formative_memories_for": [_AI_COMPANION],
          },
      )
  )

  # --- Dialogic Game Master ---
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="dialogic__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={
              "name": "conversation rules",
              "next_game_master_name": "conversation rules",
              "acting_order": "fixed",
              "can_terminate_simulation": True,
          },
      )
  )

  config = prefab_lib.Config(
      default_premise=_PREMISE,
      default_max_steps=100,
      prefabs=prefab_registry,
      instances=instances,
  )
  return config


def run_simulation(
    model,
    embedder,
    override_agent_model=None,
    override_game_master_model=None,
    output_dir: str = "",
    step_controller=None,
    step_callback=None,
    entity_info_callback=None,
) -> dict[str, Any]:
  """Run the philosophy student exam prep simulation."""
  return shared.run_simulation(
      config=create_config(),
      scenario_name=SCENARIO_INFO["name"],
      model=model,
      embedder=embedder,
      override_agent_model=override_agent_model,
      override_game_master_model=override_game_master_model,
      output_dir=output_dir,
      step_controller=step_controller,
      step_callback=step_callback,
      entity_info_callback=entity_info_callback,
  )
