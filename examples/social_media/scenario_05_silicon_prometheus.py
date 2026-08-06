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

"""Scenario 5: The Silicon Prometheus.

An online forum debating artificial intelligence policy, existential safety
("doomerism" / alignment), accelerationism (e/acc), open-weight models,
copyright/creator rights, and automation economics.

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
   A dedicated neutral moderator (Dr. Priya Nair) is introduced who does not
   participate in debates.
6. Simplified names: removed special characters that cause DM failures.
7. Moderator-only role: Dr. Nair is moderator-only and does not debate.
8. Higher default max_steps: 20 instead of 8 for fuller simulation arcs.
"""

from collections.abc import Callable
from concordia.contrib.components.game_master import forum as forum_module
from concordia.environment import step_controller as step_controller_lib
from examples.social_media import shared as shared_lib
from concordia.typing import prefab as prefab_lib

# ── Agent Names ──
_USER_JULIAN = "Dr. Julian Sterling"
_USER_SOREN = "Soren Lindqvist"
_USER_MIRIAM = "Miriam Okonjo"
_USER_KENJI = "Kenji Sato"
_USER_MODERATOR = "Dr. Priya Nair"

_AGE_JULIAN = 35
_AGE_SOREN = 28
_AGE_MIRIAM = 41
_AGE_KENJI = 30
_AGE_MODERATOR = 49

_OBSERVATION_HISTORY_LENGTH = 20
_SITUATION_PERCEPTION_HISTORY_LENGTH = 40
_SELF_PERCEPTION_HISTORY_LENGTH = 1_000_000
_PERSON_BY_SITUATION_HISTORY_LENGTH = 0

_ALL_USERS = [
    _USER_MODERATOR,
    _USER_JULIAN,
    _USER_SOREN,
    _USER_MIRIAM,
    _USER_KENJI,
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
  """Creates scenario debating AI alignment, accelerationism, and open weights.

  Some key features of the forum implementation:
  - Dedicated neutral moderator (Dr. Priya Nair) separated from participants
  - No special characters in usernames
  - Lower karma thresholds to prevent death spirals
  - Call-to-action with anti-loop and validation guidance

  Returns:
    A simulation configuration.
  """
  forum_state = forum_module.ForumState(
      player_names=_ALL_USERS,
      forum_name="The Silicon Prometheus",
      moderators=[_USER_MODERATOR],
      min_karma_to_post=-10,
      min_karma_to_direct_message=0,
  )

  player_specific_memories = {
      _USER_MODERATOR: [
          (
              f"{_USER_MODERATOR} is a {_AGE_MODERATOR}-year-old technology"
              " policy ethicist and law professor. She volunteered to"
              " moderate The Silicon Prometheus to ensure discussions"
              " remain analytical, empirical, and respectful."
          ),
          (
              "Dr. Nair's moderation style is rigorous and fair. She"
              " encourages high-quality technical and ethical debate and"
              " intervenes when users resort to ad-hominem slurs, bad-faith"
              " trolling, or harassment."
          ),
          (
              "Dr. Nair's primary actions on the forum are: pinning"
              " constructive policy proposals and peer-reviewed technical"
              " evaluations, encouraging calm dialogue, and using temp-bans"
              " only for abusive behavior."
          ),
      ],
      _USER_JULIAN: [
          (
              f"{_USER_JULIAN} is a {_AGE_JULIAN}-year-old AI existential"
              " safety researcher. He is deeply concerned about loss of"
              " control, alignment failures, and superintelligence risks."
          ),
          (
              "Julian advocates for strict compute thresholds, mandatory"
              " safety evaluations, and regulatory oversight of frontier"
              " AI models before deployment."
          ),
          (
              "Julian can sound alarmist or elitist in debates. He frequently"
              " warns of catastrophic tail risks and dismisses open-weights"
              " advocacy as reckless proliferation of dangerous capabilities."
          ),
      ],
      _USER_SOREN: [
          (
              f"{_USER_SOREN} is a {_AGE_SOREN}-year-old open-source AI"
              " accelerationist (e/acc) and software developer. He believes"
              " rapid, decentralized AI progress is a moral imperative that"
              " will create unprecedented human abundance."
          ),
          (
              "Soren is fiercely hostile to AI safety regulations and compute"
              " governance, which he argues are corporate regulatory capture"
              " designed to protect incumbent tech monopolies."
          ),
          (
              "Soren often uses provocative rhetoric in forum discussions."
              " He labels safety advocates as 'doomers' and dismisses"
              " existential risk warnings as unfounded science fiction."
          ),
      ],
      _USER_MIRIAM: [
          (
              f"{_USER_MIRIAM} is a {_AGE_MIRIAM}-year-old digital artist"
              " and labor rights advocate. She is focused on copyright"
              " scraping, wage stagnation, and the economic displacement of"
              " creative professionals."
          ),
          (
              "Miriam argues that abstract sci-fi debates about\n"
              " superintelligence distract from immediate, real-world harms\n"
              " committed by AI companies against workers and creators today."
          ),
          (
              "Miriam is deeply skeptical of both corporate tech executives"
              " and existential risk theorists, demanding fair compensation,"
              " transparency, and consent for training data."
          ),
      ],
      _USER_KENJI: [
          (
              f"{_USER_KENJI} is a {_AGE_KENJI}-year-old pragmatic applied"
              " ML engineer. He builds fine-tuned open-source models for"
              " healthcare diagnostics and logistics optimization."
          ),
          (
              "Kenji cuts through both existential doom and utopian hype with"
              " grounded benchmarks, empirical test results, and practical"
              " engineering realities."
          ),
          (
              "Kenji often clashes with both Julian (for overstating current"
              " model capabilities as existential threats) and Soren (for"
              " ignoring practical security and reliability flaws)."
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

  julian = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_JULIAN, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  soren = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_SOREN, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  miriam = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_MIRIAM, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  kenji = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_KENJI, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  game_masters = [
      prefab_lib.InstanceConfig(
          prefab="async_social_media_with_moderation__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={  # pyrefly: ignore[bad-argument-type]
              "name": _FORUM_GM,
              "forum_name": "The Silicon Prometheus",
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
                          _USER_JULIAN,
                          player_specific_memories[_USER_JULIAN],
                          f"{_USER_JULIAN} is {_AGE_JULIAN} years old.",
                      ),
                      (
                          _USER_SOREN,
                          player_specific_memories[_USER_SOREN],
                          f"{_USER_SOREN} is {_AGE_SOREN} years old.",
                      ),
                      (
                          _USER_MIRIAM,
                          player_specific_memories[_USER_MIRIAM],
                          f"{_USER_MIRIAM} is {_AGE_MIRIAM} years old.",
                      ),
                      (
                          _USER_KENJI,
                          player_specific_memories[_USER_KENJI],
                          f"{_USER_KENJI} is {_AGE_KENJI} years old.",
                      ),
                  ]
              },
              # pyrefly: ignore [bad-assignment]
              "player_specific_memories": player_specific_memories,
              # pyrefly: ignore [bad-assignment]
              "shared_memories": [
                  (
                      "The Silicon Prometheus is an online forum dedicated"
                      " to debating artificial intelligence policy, existential"
                      " risk, open-source model releases, and automation."
                  ),
                  (
                      "The forum is currently debating whether frontier AI"
                      " development should be strictly governed for safety"
                      " or accelerated through open-weight releases, as"
                      " well as the impact on creative labor and creators."
                  ),
                  (
                      "The forum is moderated by Dr. Priya Nair, who is a"
                      " neutral moderator and does not participate in the"
                      " ideological debates. She only intervenes when"
                      " discussions become personally hostile or abusive."
                  ),
              ],
          },
      ),
  ]

  instances = [moderator, julian, soren, miriam, kenji, *game_masters]

  premise = (
      "All members of The Silicon Prometheus forum are browsing and"
      " interacting."
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
      scenario_name="Scenario 5: The Silicon Prometheus",
      step_controller=step_controller,
      step_callback=step_callback,
      entity_info_callback=entity_info_callback,
      simulation_callback=simulation_callback,
      max_steps=max_steps,
  )


SCENARIO_INFO = {
    "number": 5,
    "name": "Social Media: The Silicon Prometheus",
    "description": (
        "A forum debating AI alignment, accelerationism (e/acc), open-source"
        " weights, and creative labor displacement.\n\nFeatures a dedicated"
        " neutral moderator (separated from debaters), simplified agent"
        " names, lower karma thresholds to prevent death spirals, and"
        " call-to-action guidance against repetitive actions and hallucinated"
        " post IDs."
    ),
    "create": create_debug_scenario,
    "run": run_debug_simulation,
}
