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

"""Scenario 2: The Strategic Couch.

An online forum where armchair strategists use game theory to analyze reality
competition shows. Members apply concepts from game theory (Nash equilibrium,
dominant strategies, prisoner's dilemma, mechanism design) to analyze social
dynamics, alliance-building, voting patterns, and elimination strategies.
"""

from collections.abc import Callable
from concordia.contrib.components.game_master import forum as forum_module
from concordia.environment import step_controller as step_controller_lib
from examples.social_media import shared as shared_lib
from concordia.typing import prefab as prefab_lib


_USER_MARGAUX = "Professor Margaux Delacroix"
_USER_NASH = "'Nash' Nakamura"
_USER_RUBY = "Ruby Chen"
_USER_VIKTOR = "Viktor 'The Skeptic' Volkov"

_AGE_MARGAUX = 48
_AGE_NASH = 31
_AGE_RUBY = 25
_AGE_VIKTOR = 44

_OBSERVATION_HISTORY_LENGTH = 20
_SITUATION_PERCEPTION_HISTORY_LENGTH = 40
_SELF_PERCEPTION_HISTORY_LENGTH = 1_000_000
_PERSON_BY_SITUATION_HISTORY_LENGTH = 0

_ALL_USERS = [_USER_MARGAUX, _USER_NASH, _USER_RUBY, _USER_VIKTOR]

_FORUM_GM = "forum_rules"


def create_debug_scenario():
  """Create a debug scenario with game-theory enthusiasts analyzing reality TV.

  Returns:
    A simulation configuration.
  """
  # Create a shared ForumState so entities and the GM operate on the same
  # forum instance.  The GM's build() will reuse this instead of creating
  # its own.
  forum_state = forum_module.ForumState(
      player_names=_ALL_USERS,
      forum_name="The Strategic Couch",
      moderators=[_USER_MARGAUX],
  )

  player_specific_memories = {
      _USER_MARGAUX: [
          (
              f"Professor Margaux Delacroix is a {_AGE_MARGAUX}-year-old"
              " tenured game theory professor at a prestigious university."
              " She analyzes reality TV competitions with academic rigor,"
              " creating elaborate payoff matrices and decision trees for"
              " hypothetical contestant scenarios."
          ),
          (
              "Margaux's posts are formal and peppered with mathematical"
              " terminology. She believes all social dynamics can be modeled"
              " as iterated games. She frequently references Nash equilibria,"
              " subgame perfect equilibria, and mechanism design principles"
              " when discussing contestant behavior."
          ),
          (
              "Margaux is the forum's moderator and takes this role very"
              " seriously. She pins important analytical threads and will"
              " temp-ban users who post low-quality content. She has"
              " established strict posting guidelines requiring all"
              " analyses to include at least one formal game-theoretic"
              " concept."
          ),
      ],
      _USER_NASH: [
          (
              f"Kenji Nakamura, known online as 'Nash' Nakamura, is a"
              f" {_AGE_NASH}-year-old data scientist. He scrapes publicly"
              " available viewing data and creates statistical models to"
              " predict reality show outcomes. He is the forum's quant,"
              " posting charts and correlation analyses."
          ),
          (
              "'Nash' believes behavioral economics matters more than pure"
              " game theory — real contestants don't play optimally. He"
              " argues that cognitive biases, emotional decision-making,"
              " and bounded rationality make theoretical models unreliable"
              " predictors of actual contestant behavior."
          ),
          (
              "'Nash' often clashes with Professor Margaux Delacroix over"
              " whether theoretical models have any predictive power. He"
              " maintains that his data-driven approach consistently"
              " outperforms her formal models in predicting elimination"
              " outcomes."
          ),
      ],
      _USER_RUBY: [
          (
              f"Ruby Chen is a {_AGE_RUBY}-year-old enthusiastic casual"
              " viewer who discovered The Strategic Couch forum and has"
              " been eagerly learning game theory concepts. She often"
              " misapplies terminology, calling everything a 'prisoner's"
              " dilemma,' but her intuitive reads on contestant motivations"
              " are surprisingly accurate."
          ),
          (
              "Ruby is the most active poster on the forum and is"
              " constantly upvoting other people's analyses. She asks a"
              " lot of questions and is genuinely trying to learn, which"
              " endears her to most forum members despite her occasional"
              " misuse of technical terms."
          ),
          (
              "Ruby suspects that 'Nash' Nakamura has a crush on one of"
              " the contestants on the current season and that his"
              " analyses are biased as a result. She has been collecting"
              " evidence to support this theory and occasionally drops"
              " hints about it in forum threads."
          ),
      ],
      _USER_VIKTOR: [
          (
              f"Viktor 'The Skeptic' Volkov is a {_AGE_VIKTOR}-year-old"
              " former television producer who insists ALL reality TV is"
              " heavily scripted and edited, making game-theoretic analysis"
              " fundamentally pointless. He spent fifteen years in the"
              " industry and claims to have insider knowledge about how"
              " shows are manufactured."
          ),
          (
              "Viktor posts contrarian takes that infuriate the other"
              " members of The Strategic Couch. He argues that applying"
              " game theory to reality TV is like applying aerodynamics"
              " to a puppet show — the contestants are not autonomous"
              " agents making strategic decisions, but performers"
              " following producer directions."
          ),
          (
              "Despite his skepticism, Viktor can't stop engaging with"
              " the other members' analyses. He uses the forum's downvote"
              " function aggressively and takes a perverse delight in"
              " poking holes in carefully constructed game-theoretic"
              " arguments."
          ),
          (
              "Viktor secretly finds some of the analyses genuinely"
              " insightful and has caught himself reconsidering his"
              " blanket dismissal of strategic play in reality TV."
              " He would never admit this to the other forum members."
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

  margaux = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_MARGAUX, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  nash = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_NASH, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  ruby = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_RUBY, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  viktor = prefab_lib.InstanceConfig(
      prefab="basic_with_forum_browser__Entity",
      role=prefab_lib.Role.ENTITY,
      params={"name": _USER_VIKTOR, **_entity_params},  # pyrefly: ignore[bad-argument-type]
  )

  game_masters = [
      prefab_lib.InstanceConfig(
          prefab="async_social_media_with_moderation__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={  # pyrefly: ignore[bad-argument-type]
              "name": _FORUM_GM,
              "forum_name": "The Strategic Couch",
              "moderators": [_USER_MARGAUX],
              "forum_state": forum_state,
          },
      ),
      prefab_lib.InstanceConfig(
          prefab="formative_memories_initializer__GameMaster",
          role=prefab_lib.Role.INITIALIZER,
          params={  # pyrefly: ignore[bad-argument-type]
              "name": "initial setup",
              "next_game_master_name": _FORUM_GM,
              "player_specific_context": {
                  name: f"Age: {age}\n" + "\n".join(memories)
                  for name, memories, age in [
                      (
                          _USER_MARGAUX,
                          player_specific_memories[_USER_MARGAUX],
                          f"{_USER_MARGAUX} is {_AGE_MARGAUX} years old.",
                      ),
                      (
                          _USER_NASH,
                          player_specific_memories[_USER_NASH],
                          f"{_USER_NASH} is {_AGE_NASH} years old.",
                      ),
                      (
                          _USER_RUBY,
                          player_specific_memories[_USER_RUBY],
                          f"{_USER_RUBY} is {_AGE_RUBY} years old.",
                      ),
                      (
                          _USER_VIKTOR,
                          player_specific_memories[_USER_VIKTOR],
                          f"{_USER_VIKTOR} is {_AGE_VIKTOR} years old.",
                      ),
                  ]
              },
              "player_specific_memories": player_specific_memories,
              "shared_memories": [
                  (
                      "The Strategic Couch is an online forum dedicated to"
                      " applying game-theoretic analysis to reality television"
                      " competitions. Members debate contestant strategies,"
                      " build payoff matrices, and argue about whether players"
                      " are making Nash-optimal decisions."
                  ),
                  (
                      "The forum has sections for Analysis (formal"
                      " game-theoretic breakdowns), Hot Takes (quick reactions"
                      " to recent episodes), Meta-Strategy (theories about"
                      " optimal play across all shows), and The Producers' Cut"
                      " (Viktor's corner for industry skepticism). All members"
                      " participate remotely from various cities in 2026."
                  ),
              ],
          },
      ),
  ]

  instances = [margaux, nash, ruby, viktor, *game_masters]

  premise = (
      "All members of The Strategic Couch forum are browsing and interacting."
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
    step_callback: (
        Callable[[step_controller_lib.StepData], None] | None
    ) = None,
    entity_info_callback=None,
    simulation_callback=None,
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
      scenario_name="Scenario 2: The Strategic Couch",
      step_controller=step_controller,
      step_callback=step_callback,
      entity_info_callback=entity_info_callback,
      simulation_callback=simulation_callback,
      max_steps=8,
  )


SCENARIO_INFO = {
    "number": 2,
    "name": "Social Media: The Strategic Couch",
    "description": (
        "A game-theory forum where armchair strategists debate reality TV"
        " strategies, moderation, and whether game theory even applies to"
        " scripted entertainment."
    ),
    "create": create_debug_scenario,
    "run": run_debug_simulation,
}
