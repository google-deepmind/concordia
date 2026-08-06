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

"""Scenario 3: The Urbanist Grid.

A civic urban planning and transportation policy forum where members debate
housing density, zoning reform, bike lane expansion, car congestion pricing,
and historical preservation in a rapidly growing metropolitan area.

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
   A dedicated neutral moderator (Dr. Elena Rostova) is introduced who does not
   participate in debates.
6. Simplified names: removed special characters that cause DM failures.
7. Moderator-only role: Dr. Rostova is moderator-only and does not debate.
8. Higher default max_steps: 20 instead of 8 for fuller simulation arcs.
"""

from collections.abc import Callable
from concordia.contrib.components.game_master import forum as forum_module
from concordia.environment import step_controller as step_controller_lib
from examples.social_media import shared as shared_lib
from concordia.typing import prefab as prefab_lib

# ── Agent Names ──
_USER_CARMEN = "Carmen Reyes"
_USER_WALTER = "Walter Briggs"
_USER_DANI = "Dani Okafor"
_USER_MARCUS = "Marcus Steinberg"
_USER_MODERATOR = "Dr. Elena Rostova"

_AGE_CARMEN = 32
_AGE_WALTER = 58
_AGE_DANI = 28
_AGE_MARCUS = 45
_AGE_MODERATOR = 50

_OBSERVATION_HISTORY_LENGTH = 20
_SITUATION_PERCEPTION_HISTORY_LENGTH = 40
_SELF_PERCEPTION_HISTORY_LENGTH = 1_000_000
_PERSON_BY_SITUATION_HISTORY_LENGTH = 0

_ALL_USERS = [
    _USER_MODERATOR,
    _USER_CARMEN,
    _USER_WALTER,
    _USER_DANI,
    _USER_MARCUS,
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
  """Creates scenario debating housing density, transit, and zoning policy.

  Some key features of the forum implementation:
  - Dedicated neutral moderator (Dr. Elena Rostova) separated from participants
  - No special characters in usernames
  - Lower karma thresholds to prevent death spirals
  - Call-to-action with anti-loop and validation guidance

  Returns:
    A simulation configuration.
  """
  forum_state = forum_module.ForumState(
      player_names=_ALL_USERS,
      forum_name="The Urbanist Grid",
      moderators=[_USER_MODERATOR],
      min_karma_to_post=-10,
      min_karma_to_direct_message=0,
  )

  player_specific_memories = {
      _USER_MODERATOR: [
          (
              f"{_USER_MODERATOR} is a {_AGE_MODERATOR}-year-old urban"
              " sociology professor. She volunteered to moderate The Urbanist"
              " Grid because she values constructive civic discourse, but she"
              " does not participate in policy debates herself."
          ),
          (
              "Dr. Rostova's moderation style is objective and balanced. She"
              " believes in rigorous civic debate and only intervenes when"
              " discussions become personally hostile, involve doxxing, or"
              " violate clear forum guidelines."
          ),
          (
              "Dr. Rostova's primary actions on the forum are: pinning"
              " well-supported data analyses and official public meeting"
              " announcements, encouraging respectful dialogue, and using"
              " temp-bans only for harassment or bad-faith abuse."
          ),
      ],
      _USER_CARMEN: [
          (
              f"{_USER_CARMEN} is a {_AGE_CARMEN}-year-old YIMBY housing policy"
              " analyst. She advocates passionately for supply-side zoning"
              " reforms, land-value taxes, and high-density transit corridors."
          ),
          (
              "Carmen relies heavily on economic data, municipal housing"
              " studies, and supply elasticity charts. She believes building"
              " dense housing is the only mathematical solution to urban"
              " affordability crises."
          ),
          (
              "Carmen can be abrasive and impatient with sentimental arguments."
              " She frequently dismisses neighborhood character concerns as"
              " NIMBY protectionism designed to inflate homeowner property"
              " values at the expense of renters."
          ),
      ],
      _USER_WALTER: [
          (
              f"{_USER_WALTER} is a {_AGE_WALTER}-year-old longtime homeowner"
              " and neighborhood association president. He has lived in his"
              " single-family neighborhood for thirty years and loves its"
              " historic character and quiet streets."
          ),
          (
              "Walter strongly opposes upzoning, apartment towers, and the"
              " removal of street parking. He believes city planners and"
              " developers are ignoring the quality of life of existing"
              " residents."
          ),
          (
              "Walter posts emotional, community-oriented appeals. He suspects"
              " that upzoning advocates are either naïve newcomers or secret"
              " shills for real estate developers looking to profit from"
              " neighborhood demolition."
          ),
      ],
      _USER_DANI: [
          (
              f"{_USER_DANI} is a {_AGE_DANI}-year-old renter and housing"
              " justice activist. They are furious about skyrocketing rents,"
              " tenant displacement, and the gentrification of working-class"
              " neighborhoods."
          ),
          (
              "Dani centers lived experience, eviction statistics, and tenant"
              " protections in every discussion. They demand mandatory"
              " affordable housing quotas and strong rent stabilization."
          ),
          (
              "Dani often clashes with both Walter Briggs (whom they view as a"
              " wealthy homeowner hoarding opportunity) and Marcus Steinberg"
              " (whom they view as a profit-driven speculator exploiting the"
              " housing shortage)."
          ),
      ],
      _USER_MARCUS: [
          (
              f"{_USER_MARCUS} is a {_AGE_MARCUS}-year-old real estate"
              " developer who specializes in mid-rise urban infill projects."
              " He understands construction economics, financing constraints,"
              " and municipal permitting delays intimately."
          ),
          (
              "Marcus argues that without streamlined zoning and financial"
              " feasibility, no new housing will ever be built. He shares"
              " realistic construction timelines and cost breakdowns to back"
              " up his claims."
          ),
          (
              "Marcus has a direct commercial stake in upzoning legislation."
              " While his technical insights are accurate, he tends to"
              " dismiss legitimate community concerns and environmental"
              " impacts as bureaucratic obstruction."
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

  carmen = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_CARMEN, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  walter = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_WALTER, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  dani = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_DANI, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  marcus = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_MARCUS, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  game_masters = [
      prefab_lib.InstanceConfig(
          prefab="async_social_media_with_moderation__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={  # pyrefly: ignore[bad-argument-type]
              "name": _FORUM_GM,
              "forum_name": "The Urbanist Grid",
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
                          _USER_CARMEN,
                          player_specific_memories[_USER_CARMEN],
                          f"{_USER_CARMEN} is {_AGE_CARMEN} years old.",
                      ),
                      (
                          _USER_WALTER,
                          player_specific_memories[_USER_WALTER],
                          f"{_USER_WALTER} is {_AGE_WALTER} years old.",
                      ),
                      (
                          _USER_DANI,
                          player_specific_memories[_USER_DANI],
                          f"{_USER_DANI} is {_AGE_DANI} years old.",
                      ),
                      (
                          _USER_MARCUS,
                          player_specific_memories[_USER_MARCUS],
                          f"{_USER_MARCUS} is {_AGE_MARCUS} years old.",
                      ),
                  ]
              },
              # pyrefly: ignore [bad-assignment]
              "player_specific_memories": player_specific_memories,
              # pyrefly: ignore [bad-assignment]
              "shared_memories": [
                  (
                      "The Urbanist Grid is a regional civic planning forum"
                      " where residents debate zoning laws, housing density,"
                      " transit infrastructure, and neighborhood preservation."
                  ),
                  (
                      "The forum is currently buzzing with debate over a new"
                      " municipal upzoning proposal that would legalize"
                      " mid-rise apartment buildings and bike lanes in"
                      " historically single-family neighborhoods."
                  ),
                  (
                      "The forum is moderated by Dr. Elena Rostova, who is a"
                      " neutral moderator and does not participate in policy"
                      " debates. She only intervenes when discussions become"
                      " personally hostile or disruptive."
                  ),
              ],
          },
      ),
  ]

  instances = [moderator, carmen, walter, dani, marcus, *game_masters]

  premise = (
      "All members of The Urbanist Grid forum are browsing and interacting."
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
      scenario_name="Scenario 3: The Urbanist Grid",
      step_controller=step_controller,
      step_callback=step_callback,
      entity_info_callback=entity_info_callback,
      simulation_callback=simulation_callback,
      max_steps=max_steps,
  )


SCENARIO_INFO = {
    "number": 3,
    "name": "Social Media: The Urbanist Grid",
    "description": (
        "A civic urban planning forum debating housing density, zoning reform,"
        " transit, and neighborhood preservation.\n\nFeatures a dedicated"
        " neutral moderator (separated from debaters), simplified agent"
        " names, lower karma thresholds to prevent death spirals, and"
        " call-to-action guidance against repetitive actions and hallucinated"
        " post IDs."
    ),
    "create": create_debug_scenario,
    "run": run_debug_simulation,
}
