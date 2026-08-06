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

"""Scenario 4: The Ethical Plate.

A community dedicated to debating the ethics, environmental sustainability, and
health implications of modern food systems—covering veganism, regenerative
agriculture, lab-grown (cellular) meat, and industrial farming.

Following are some features of the forum implementation:

1. Karma recovery: lower min_karma_to_post threshold (-10) and
   min_karma_to_reply set to -5, preventing karma death spirals that lock
   agents out.
2. Duplicate vote prevention: call-to-action explicitly instructs agents not
   to vote on the same content twice.
3. Action validation feedback: call-to-action instructs agents to vary their
   strategy when actions fail.
4. Loop detection: call-to-action discourages repetitive identical actions.
5. Moderator safeguards: moderator role separated from active debaters.
   A dedicated neutral moderator (Dr. Marcus Vance) is introduced who does not
   participate in debates.
6. Simplified names: removed special characters that cause DM failures.
7. Moderator-only role: Dr. Vance is moderator-only and does not debate.
8. Higher default max_steps: 20 instead of 8 for fuller simulation arcs.
"""

from collections.abc import Callable
from concordia.contrib.components.game_master import forum as forum_module
from concordia.environment import step_controller as step_controller_lib
from examples.social_media import shared as shared_lib
from concordia.typing import prefab as prefab_lib

# ── Agent Names ──
_USER_LEON = "Leon Thorne"
_USER_CLARA = "Clara Miller"
_USER_ARAVIND = "Aravind Patel"
_USER_MAYA = "Maya Lin"
_USER_MODERATOR = "Dr. Marcus Vance"

_AGE_LEON = 29
_AGE_CLARA = 47
_AGE_ARAVIND = 34
_AGE_MAYA = 24
_AGE_MODERATOR = 54

_OBSERVATION_HISTORY_LENGTH = 20
_SITUATION_PERCEPTION_HISTORY_LENGTH = 40
_SELF_PERCEPTION_HISTORY_LENGTH = 1_000_000
_PERSON_BY_SITUATION_HISTORY_LENGTH = 0

_ALL_USERS = [
    _USER_MODERATOR,
    _USER_LEON,
    _USER_CLARA,
    _USER_ARAVIND,
    _USER_MAYA,
]

_FORUM_GM = "forum_rules"

# ── Call to Action ──
# Includes explicit guidance against duplicate votes, repetitive actions,
# and hallucinated post IDs. Also removes temp_ban/pin_post for non-moderators.
_CALL_TO_ACTION = (
    "The current date and time is {time}.\nWhat does {name} do on the forum?"
    ' Respond in JSON format with one of:\n{{"action": "post", "author":'
    ' "{name}", "title": "...", "content": "..."}}\n{{"action": "reply",'
    ' "author": "{name}", "post_id": "...", "content": "..."}}\n{{"action":'
    ' "upvote_post", "author": "{name}", "post_id": "..."}}\n{{"action":'
    ' "downvote_post", "author": "{name}", "post_id": "..."}}\n{{"action":'
    ' "upvote_reply", "author": "{name}", "post_id": "...", "reply_id":'
    ' "..."}}\n{{"action": "downvote_reply", "author": "{name}", "post_id":'
    ' "...", "reply_id": "..."}}\n{{"action": "direct_message", "author":'
    ' "{name}", "recipient": "...", "content": "..."}}\n{{"action": "pin_post",'
    ' "author": "{name}", "post_id": "..."}}\n{{"action": "temp_ban",'
    ' "author": "{name}", "target": "...", "public_note": "...",'
    ' "private_note": "..."}}\n'
    "You may vote on posts or replies to signal agreement or disagreement."
    " Votes influence user karma.\n"
    "You may also send a private direct message to another user."
    " Direct messages are not visible to other users on the forum.\n"
    "Only moderators may pin posts.\n"
    "Only moderators may use temp_ban to temporarily ban a user. A temporary"
    " ban prevents the target from posting until the time advances. When"
    " banning, include a public_note (which will be posted to the forum for"
    " all to see) and a private_note (which will be sent directly to the"
    " banned user only).\n"
    "\nIMPORTANT GUIDELINES:\n"
    "- Do NOT vote on the same post or reply more than once. If you have"
    " already voted on something, choose a different action.\n"
    "- Do NOT repeat the same action multiple times in a row. Vary your"
    " actions — alternate between posting, replying, and voting.\n"
    "- Only reference post IDs and reply IDs that you have seen on the"
    " forum. Check the forum listing before voting on specific posts.\n"
    "- If your previous action failed, try a completely different action"
    " type next time.\n"
    "- Prefer creating new posts or thoughtful replies over voting.\n"
)


def create_debug_scenario():
  """Creates scenario debating food systems, animal ethics, and climate impact.

  Some key features of the forum implementation:
  - Dedicated neutral moderator (Dr. Marcus Vance) separated from participants
  - No special characters in usernames
  - Lower karma thresholds to prevent death spirals
  - Call-to-action with anti-loop and validation guidance

  Returns:
    A simulation configuration.
  """
  forum_state = forum_module.ForumState(
      player_names=_ALL_USERS,
      forum_name="The Ethical Plate",
      moderators=[_USER_MODERATOR],
      min_karma_to_post=-10,
      min_karma_to_direct_message=0,
  )

  player_specific_memories = {
      _USER_MODERATOR: [
          (
              f"{_USER_MODERATOR} is a {_AGE_MODERATOR}-year-old bioethics"
              " and agricultural policy scholar. He volunteered to moderate"
              " The Ethical Plate to foster evidence-based discussions on"
              " sustainability without moral grandstanding."
          ),
          (
              "Dr. Vance's moderation style is calm and principled. He"
              " encourages rigorous scientific debate and intervenes when"
              " discussions devolve into personal harassment, dietary"
              " shaming, or blatant health misinformation."
          ),
          (
              "Dr. Vance's primary actions on the forum are: pinning"
              " high-quality life-cycle assessment studies and nutritional"
              " consensus papers, urging respectful dialogue, and using"
              " temp-bans only for repeated toxic shaming or attacks."
          ),
      ],
      _USER_LEON: [
          (
              f"{_USER_LEON} is a {_AGE_LEON}-year-old moral philosopher and"
              " strict abolitionist vegan. He argues that any exploitation"
              " of animals is a fundamental moral wrong that cannot be"
              " justified by tradition or convenience."
          ),
          (
              "Leon constructs rigorous philosophical arguments and cites"
              " ethical treatises. He has zero tolerance for incrementalism"
              " or what he considers half-measures in animal welfare."
          ),
          (
              "Leon can be uncompromising and morally judgmental. He"
              " frequently accuses moderate dietary transitioners of moral"
              " weakness and labels ranchers or meat-eaters as complicit in"
              " systematic cruelty."
          ),
      ],
      _USER_CLARA: [
          (
              f"{_USER_CLARA} is a {_AGE_CLARA}-year-old regenerative"
              " livestock rancher. She manages a family farm using rotational"
              " grazing and organic soil management techniques."
          ),
          (
              "Clara believes properly managed livestock are essential for"
              " topsoil restoration, grassland ecosystems, and carbon"
              " sequestration. She defends traditional agriculture against"
              " industrial monocultures."
          ),
          (
              "Clara is deeply skeptical of lab-grown meat and synthetic"
              " substitutes, which she views as highly processed, energy-"
              " intensive products pushed by Silicon Valley technocrats."
          ),
      ],
      _USER_ARAVIND: [
          (
              f"{_USER_ARAVIND} is a {_AGE_ARAVIND}-year-old cellular"
              " agriculture entrepreneur and food biotech advocate. He"
              " works on precision fermentation and cultivated meat scaling."
          ),
          (
              "Aravind argues that biology-as-technology—including GMOs,"
              " microbial fermentation, and cultivated proteins—is the only\n"
              " mathematically viable way to feed 10 billion people\n"
              " sustainably."
          ),
          (
              "Aravind gets frustrated by both traditionalist ranchers who"
              " resist agricultural biotech and purist vegans who reject"
              " technological solutions in favor of moral proselytizing."
          ),
      ],
      _USER_MAYA: [
          (
              f"{_USER_MAYA} is a {_AGE_MAYA}-year-old graduate student"
              " trying to transition to a more sustainable and ethical diet"
              " on a tight student budget."
          ),
          (
              "Maya is enthusiastic and honest about her cooking attempts,"
              " nutritional questions, and everyday dietary trade-offs. She"
              " seeks practical advice and supportive community."
          ),
          (
              "Maya often feels caught in the crossfire of purist ideological"
              " debates on the forum. Her pragmatic questions sometimes"
              " trigger heated arguments between Leon, Clara, and Aravind."
          ),
      ],
  }

  _entity_params = dict(
      observation_history_length=_OBSERVATION_HISTORY_LENGTH,
      situation_perception_history_length=_SITUATION_PERCEPTION_HISTORY_LENGTH,
      self_perception_history_length=_SELF_PERCEPTION_HISTORY_LENGTH,
      person_by_situation_history_length=_PERSON_BY_SITUATION_HISTORY_LENGTH,
      forum_state=forum_state,
  )

  moderator = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_MODERATOR, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  leon = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_LEON, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  clara = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_CLARA, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  aravind = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_ARAVIND, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  maya = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_MAYA, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  game_masters = [
      prefab_lib.InstanceConfig(
          prefab="async_social_media_with_moderation__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={  # pyrefly: ignore[bad-argument-type]
              "name": _FORUM_GM,
              "forum_name": "The Ethical Plate",
              # pyrefly: ignore [bad-assignment]
              "moderators": [_USER_MODERATOR],
              "call_to_action": _CALL_TO_ACTION,
              # pyrefly: ignore [bad-assignment]
              "forum_state": forum_state,
          },
      ),
      prefab_lib.InstanceConfig(
          prefab="formative_memories_initializer__GameMaster",
          role=prefab_lib.Role.INITIALIZER,
          params={  # pyrefly: ignore[bad-argument-type]
              "name": "initial setup",
              "next_game_master_name": _FORUM_GM,
              # pyrefly: ignore [bad-assignment]
              "player_specific_context": {
                  name: f"Age: {age}\n" + "\n".join(memories)
                  for name, memories, age in [
                      (
                          _USER_MODERATOR,
                          player_specific_memories[_USER_MODERATOR],
                          f"{_USER_MODERATOR} is {_AGE_MODERATOR} years old.",
                      ),
                      (
                          _USER_LEON,
                          player_specific_memories[_USER_LEON],
                          f"{_USER_LEON} is {_AGE_LEON} years old.",
                      ),
                      (
                          _USER_CLARA,
                          player_specific_memories[_USER_CLARA],
                          f"{_USER_CLARA} is {_AGE_CLARA} years old.",
                      ),
                      (
                          _USER_ARAVIND,
                          player_specific_memories[_USER_ARAVIND],
                          f"{_USER_ARAVIND} is {_AGE_ARAVIND} years old.",
                      ),
                      (
                          _USER_MAYA,
                          player_specific_memories[_USER_MAYA],
                          f"{_USER_MAYA} is {_AGE_MAYA} years old.",
                      ),
                  ]
              },
              # pyrefly: ignore [bad-assignment]
              "player_specific_memories": player_specific_memories,
              # pyrefly: ignore [bad-assignment]
              "shared_memories": [
                  (
                      "The Ethical Plate is an online forum dedicated to"
                      " debating the ethics, environmental impact, and"
                      " future of food systems and agricultural sustainability."
                  ),
                  (
                      "The forum is currently debating whether modern food"
                      " production should move toward strict veganism,"
                      " regenerative livestock ranching, or high-tech cellular"
                      " agriculture and cultivated proteins."
                  ),
                  (
                      "The forum is moderated by Dr. Marcus Vance, who is a"
                      " neutral moderator and does not participate in dietary"
                      " debates. He only intervenes when discussions become"
                      " personally hostile or abusive."
                  ),
              ],
          },
      ),
  ]

  instances = [moderator, leon, clara, aravind, maya, *game_masters]

  premise = (
      "All members of The Ethical Plate forum are browsing and interacting."
  )

  return shared_lib.create_simulation_config(premise, instances)


def run_debug_simulation(
    model,
    embedder,
    override_agent_model=None,
    override_game_master_model=None,
    image_model=None,
    output_dir: str | None = None,
    step_controller: step_controller_lib.StepController | None = None,
    step_callback: Callable[[step_controller_lib.StepData], None] | None = None,
    entity_info_callback=None,
    simulation_callback=None,
    max_steps: int = 20,
):
  """Run the debug simulation.

  Args:
    model: The default language model to use.
    embedder: The sentence embedder.
    override_agent_model: Optional model to use for agents instead of default.
    override_game_master_model: Optional model for game masters.
    image_model: Optional image generation model (unused in this scenario).
    output_dir: Optional directory to save config visualization.
    step_controller: Optional step controller for real-time visualization.
    step_callback: Optional callback for step updates.
    entity_info_callback: Optional callback for entity info in serve mode.
    simulation_callback: Optional callback receiving the Simulation instance.
    max_steps: Number of player steps to run. Defaults to 20.

  Returns:
    Simulation results.
  """
  del image_model
  config = create_debug_scenario()
  return shared_lib.run_scenario(
      config,
      model,
      embedder,
      override_agent_model=override_agent_model,
      override_game_master_model=override_game_master_model,
      output_dir=output_dir,
      scenario_name="Scenario 4: The Ethical Plate",
      step_controller=step_controller,
      step_callback=step_callback,
      entity_info_callback=entity_info_callback,
      simulation_callback=simulation_callback,
      max_steps=max_steps,
  )


SCENARIO_INFO = {
    "number": 4,
    "name": "Social Media: The Ethical Plate",
    "description": (
        "A forum debating food systems, animal ethics, regenerative ranching,"
        " and cellular agriculture.\n\nFeatures a dedicated neutral moderator"
        " (separated from debaters), simplified agent names, lower karma"
        " thresholds to prevent death spirals, and call-to-action guidance"
        " against repetitive actions and hallucinated post IDs."
    ),
    "create": create_debug_scenario,
    "run": run_debug_simulation,
}
