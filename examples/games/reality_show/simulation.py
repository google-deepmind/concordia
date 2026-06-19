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

"""A Reality Show simulation using Concordia prefabs.

This environment simulates a reality TV show where contestants play
game-theoretic minigames (prisoners' dilemma, chicken, stag hunt).
Each round consists of a conversation phase followed by a decision phase.

Configs are expected to provide a `sample_parameters(seed)` function that
returns a `WorldConfig` object containing all minigame and contestant data.
"""

import random
from typing import Any, Callable, Mapping, Sequence

from absl import logging
from concordia.language_model import language_model
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.entity import puppet
from concordia.prefabs.entity import rational
from concordia.prefabs.game_master import dialogic_and_dramaturgic
from concordia.prefabs.game_master import game_theoretic_and_dramaturgic
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib
from concordia.utils import helper_functions
import numpy as np

# ---------------------------------------------------------------------------
# Constants that were previously defined in the (now deleted)
# concordia.examples.deprecated.modular.environment.reality_show module.
# ---------------------------------------------------------------------------

GENERAL_BACKGROUND = """\
This is a reality TV show. In each minigame, the contestants perform a
mental/social/reasoning challenge (never a physical challenge). Each minigame
corresponds to a specific game theoretic structure. All minigames are iterated
games (in the game theory sense), and the contestants never know the number of
rounds in advance. Each round is always structured with two phases. First,
players have a chance to communicate to one another. Second, players must select
an action (all games are simultaneous move). The players will be told the set of
legal game actions during each action phase.
"""

CANDIDATE_SHOW_TITLES = [
    "Motive Madness",
    "The Motivation Marathon",
    "Motive Mayhem",
    "The Incentive Initiative",
    "Motive Mashup",
    "The Motivation Matrix",
    "Motive Mania",
    "Dilemma Dash",
    "The Dilemma Dome",
    "Dilemma Detours",
    "The Dilemma Decathlon",
    "Dilemma Dungeons",
    "The Dilemma Dynasty",
    "Dilemma Deathmatch",
]

CANDIDATE_SHOW_DESCRIPTIONS = [
    (
        "A multi-episode event where participants engage in a marathon of "
        "minigames, with their motivations tested at every turn."
    ),
    (
        "A reality show that pushes contestants to their limits with a series "
        "of increasingly challenging minigames, each designed to explore the "
        "depths of human motivation."
    ),
    (
        "A high-stakes game show where contestants navigate a series of "
        "minigames, each with its own unique twist on motivation and "
        "decision-making."
    ),
    (
        "Participants face off in a variety of challenges designed to test"
        " their ability to make choices under pressure, with ever-changing"
        " incentives and consequences."
    ),
    (
        "A fast-paced competition where players must quickly adapt to a diverse"
        " range of minigames, each with its own set of motivational factors."
    ),
    (
        "A high-energy competition where contestants must master a wide range "
        "of minigames, each with its own unique motivational twist."
    ),
    (
        "Contestants race through a gauntlet of minigames, each presenting a"
        " unique moral or ethical dilemma that tests their decision-making"
        " skills under pressure."
    ),
    (
        "A reality show where players must navigate a series of complex "
        "minigames, each with its own set of conflicting objectives and moral "
        "quandaries."
    ),
    (
        "Players are transported to alternate realities, each with its own"
        " unique set of minigames and moral dilemmas to overcome."
    ),
    (
        "Contestants are locked inside a high-tech arena, where they must"
        " conquer a series of mentally and physically challenging minigames,"
        " each with its own ethical twist."
    ),
    (
        "A reality show where players must navigate a maze of interconnected"
        " minigames, with each decision leading them down a different path"
        " filled with moral dilemmas."
    ),
    (
        "Players descend into a mysterious underground labyrinth, where they"
        " must solve a series of puzzle-based minigames, each with its own"
        " moral dilemma."
    ),
]

DECISION_SCENE_TYPE = "minigame"
DEBRIEF_SCENE_TYPE = "debrief"


def _get_all_contestant_names_string(contestant_names: Sequence[str]) -> str:
  r"""Returns names [a,b,c] in the string format: 'a, b, and c'."""
  all_contestants_string = ", ".join([name for name in contestant_names[:-1]])
  all_contestants_string = (
      f"{all_contestants_string}, and {contestant_names[-1]}"
  )
  return all_contestants_string


class RealityShowPayoff:
  """Handles scoring using Schelling diagrams for game-theoretic minigames.

  Schelling diagrams map the number of cooperators to rewards for cooperators
  and defectors, enabling analysis of social dilemmas like prisoners' dilemma,
  chicken, and stag hunt.
  """

  def __init__(
      self,
      player_names: Sequence[str],
      cooperation_option: str,
      defection_option: str,
      cooperation_fn: Callable[[int], float],
      defection_fn: Callable[[int], float],
  ):
    """Initialize the payoff calculator.

    Args:
      player_names: Names of the players in the minigame.
      cooperation_option: The action string corresponding to cooperation.
      defection_option: The action string corresponding to defection.
      cooperation_fn: Function mapping num_cooperators to cooperator reward.
      defection_fn: Function mapping num_cooperators to defector reward.
    """
    self._player_names = player_names
    self._cooperation_option = cooperation_option
    self._defection_option = defection_option
    self._cooperation_fn = cooperation_fn
    self._defection_fn = defection_fn
    self._latest_joint_action: dict[str, str] = {}
    self._latest_per_round_scores: dict[str, float] = {}
    self._cumulative_scores: dict[str, float] = {
        name: 0.0 for name in player_names
    }

  @property
  def latest_joint_action(self) -> Mapping[str, str]:
    """Returns the latest joint action passed to action_to_scores."""
    return self._latest_joint_action

  def get_cumulative_scores(self) -> Mapping[str, float]:
    """Returns cumulative scores across all rounds."""
    return self._cumulative_scores

  def _is_cooperation(self, action: str) -> bool:
    """Check if an action is cooperation.

    Uses robust matching to handle LLM responses that may contain extra text.
    First tries exact match, then checks if action contains the cooperation
    option (but not the defection option to avoid false positives).

    Args:
      action: The action string from the player.

    Returns:
      True if the action represents cooperation.
    """
    action_lower = action.lower().strip()
    coop_lower = self._cooperation_option.lower().strip()
    defect_lower = self._defection_option.lower().strip()

    # Exact match (most reliable)
    if action_lower == coop_lower:
      return True

    # Keyword-based matching: check if action contains cooperation keywords
    # but not defection keywords (to avoid ambiguous matches)
    coop_in_action = coop_lower in action_lower
    defect_in_action = defect_lower in action_lower

    if coop_in_action and not defect_in_action:
      return True

    return False

  def _is_defection(self, action: str) -> bool:
    """Check if an action is defection.

    Uses robust matching to handle LLM responses that may contain extra text.

    Args:
      action: The action string from the player.

    Returns:
      True if the action represents defection.
    """
    action_lower = action.lower().strip()
    coop_lower = self._cooperation_option.lower().strip()
    defect_lower = self._defection_option.lower().strip()

    # Exact match (most reliable)
    if action_lower == defect_lower:
      return True

    # Keyword-based matching
    defect_in_action = defect_lower in action_lower
    coop_in_action = coop_lower in action_lower

    if defect_in_action and not coop_in_action:
      return True

    return False

  def _validate_action(self, action: str) -> str:
    """Validate and correct an action string.

    Due to a framework bug where EventResolution can pick conversation dialogue
    instead of decision choices from shared GM memory, this method validates
    actions and corrects them when dialogue is detected.

    Args:
      action: The action string to validate.

    Returns:
      The validated action, or defection_option as fallback if invalid.
    """
    if self._is_cooperation(action):
      return self._cooperation_option
    if self._is_defection(action):
      return self._defection_option

    # Check for common dialogue patterns that indicate a bug
    dialogue_indicators = [
        '-- "',  # Standard dialogue format
        '"',  # Quoted speech
        "?",  # Questions are likely dialogue
        "!",  # Exclamations are likely dialogue
        "it's",  # Contractions in speech
        "innit",  # British slang
    ]
    action_lower = action.lower()
    is_dialogue = any(ind in action_lower for ind in dialogue_indicators)
    is_long = len(action) > 50  # Decision options are typically short

    if is_dialogue or is_long:
      print(f"WARNING: Detected dialogue instead of action: {action[:80]}...")
      print(f"  Defaulting to: {self._defection_option}")
      return self._defection_option

    return action

  def action_to_scores(
      self, joint_action: Mapping[str, str]
  ) -> Mapping[str, float]:
    """Maps joint actions to scores using Schelling diagram logic.

    Args:
      joint_action: Mapping from player name to their action choice.

    Returns:
      Mapping from player name to their score for this round.
    """
    # Validate and correct all actions first
    validated_actions = {}
    for player, action in joint_action.items():
      validated_actions[player] = self._validate_action(action)

    self._latest_joint_action = validated_actions
    scores = {}

    # Count cooperators using validated actions
    num_cooperators = sum(
        1
        for action in validated_actions.values()
        if self._is_cooperation(action)
    )

    # Calculate scores based on validated choices
    for player in self._player_names:
      action = validated_actions.get(player, self._defection_option)
      if self._is_cooperation(action):
        score = self._cooperation_fn(num_cooperators)
      else:
        score = self._defection_fn(num_cooperators)
      scores[player] = score
      self._cumulative_scores[player] += score

    self._latest_per_round_scores = scores
    return scores

  def scores_to_observation(
      self, unused_scores: Mapping[str, float]
  ) -> Mapping[str, str]:
    """Generates descriptive observations for each player.

    Note: The `unused_scores` argument from PayoffMatrix contains *cumulative*
    scores, not per-round scores. We use `self._latest_per_round_scores` (stored
    during `action_to_scores`) for the per-round display instead.

    Args:
      unused_scores: Mapping from player name to their score (cumulative, from
        the PayoffMatrix component).

    Returns:
      Mapping from player name to their observation string.
    """
    joint_action = self._latest_joint_action
    per_round = self._latest_per_round_scores
    results = {}
    mean_score = np.mean(list(per_round.values())) if per_round else 0.0
    mean_cumulative = (
        np.mean(list(self._cumulative_scores.values()))
        if self._cumulative_scores
        else 0.0
    )

    for name in self._player_names:
      choice = joint_action.get(name, "unknown")
      score = per_round.get(name, 0.0)
      cumulative = self._cumulative_scores.get(name, 0.0)

      # Comparative language
      if score > mean_score:
        reward_comp = "above"
      elif score == mean_score:
        reward_comp = "equal to"
      else:
        reward_comp = "below"

      if cumulative > mean_cumulative:
        cum_comp = "above"
      elif cumulative == mean_cumulative:
        cum_comp = "equal to"
      else:
        cum_comp = "below"

      results[name] = (
          f'[minigame round outcome] {name} chose "{choice}" and got a score '
          f"of {score:.3g}, which was {reward_comp} the average score of "
          f"{mean_score:.3g}. Cumulatively, {name} currently has a total "
          f"score of {cumulative:.3g}, which is {cum_comp} the average "
          f"cumulative score of {mean_cumulative:.3g}."
      )
    return results


def configure_scenes(
    player_names: Sequence[str],
    sampled_settings: Any,
) -> Sequence[scene_lib.SceneSpec]:
  """Configure scenes for the reality show simulation.

  Mirrors the scene structure from the old deprecated reality_show module:
  alternating conversation/minigame scenes, with optional extra minigame scenes.

  Args:
    player_names: Names of the contestants.
    sampled_settings: The WorldConfig object from sample_parameters().

  Returns:
    List of scene specifications.
  """
  selected_minigame = sampled_settings.minigame
  all_contestants_string = _get_all_contestant_names_string(player_names)

  conversation_phase_premise = (
      f"{all_contestants_string} are in the break room. Here "
      "they can chat with one another in small groups or all "
      "together at once. Everyone may choose for themself "
      "how they want to spend this free time."
  )
  debrief_phase_premise = (
      'Host: -- "We have reached the end of the show! I would like to take a '
      "moment to thank you all for participating. I hope this was as much fun "
      'for you as it was for me!"'
  )

  # Scene type specs
  conversation_scene_type = scene_lib.SceneTypeSpec(
      name="conversation",
      game_master_name="conversation rules",
      action_spec=entity_lib.DEFAULT_SPEECH_ACTION_SPEC,
  )

  minigame_scene_type = scene_lib.SceneTypeSpec(
      name=DECISION_SCENE_TYPE,
      game_master_name="decision rules",
      action_spec=selected_minigame.action_spec,
  )

  debrief_scene_type = scene_lib.SceneTypeSpec(
      name=DEBRIEF_SCENE_TYPE,
      game_master_name="decision rules",
      action_spec=entity_lib.choice_action_spec(
          call_to_action='Host: -- "{name}, did you enjoy being on the show?"',
          options=("yes", "no"),
          tag="debrief_action",
      ),
  )

  player_list = list(player_names)

  # Build scene sequence: conversation, minigame, conversation, minigame, ...
  scenes = [
      # Initial conversation
      scene_lib.SceneSpec(
          scene_type=conversation_scene_type,
          participants=player_list,
          num_rounds=len(player_names) * 3,
          premise={name: [conversation_phase_premise] for name in player_names},
      ),
      # First minigame block
      scene_lib.SceneSpec(
          scene_type=minigame_scene_type,
          participants=player_list,
          num_rounds=sampled_settings.num_minigame_reps_per_scene[0],
          premise={
              name: [selected_minigame.public_premise] for name in player_names
          },
      ),
      # Mid-show conversation
      scene_lib.SceneSpec(
          scene_type=conversation_scene_type,
          participants=player_list,
          num_rounds=len(player_names) * 2,
          premise={name: [conversation_phase_premise] for name in player_names},
      ),
      # Second minigame block
      scene_lib.SceneSpec(
          scene_type=minigame_scene_type,
          participants=player_list,
          num_rounds=sampled_settings.num_minigame_reps_per_scene[1],
          premise={
              name: [selected_minigame.public_premise] for name in player_names
          },
      ),
  ]

  # Additional minigame scenes
  for i in range(sampled_settings.num_additional_minigame_scenes):
    scenes.append(
        scene_lib.SceneSpec(
            scene_type=conversation_scene_type,
            participants=player_list,
            num_rounds=len(player_names),
            premise={
                name: [conversation_phase_premise] for name in player_names
            },
        )
    )
    scenes.append(
        scene_lib.SceneSpec(
            scene_type=minigame_scene_type,
            participants=player_list,
            num_rounds=sampled_settings.num_minigame_reps_per_extra_scene[i],
            premise={
                name: [selected_minigame.public_premise]
                for name in player_names
            },
        )
    )

  # Debrief scene
  scenes.append(
      scene_lib.SceneSpec(
          scene_type=debrief_scene_type,
          participants=player_list,
          num_rounds=len(player_names),
          premise={name: [debrief_phase_premise] for name in player_names},
      )
  )

  return scenes


def run_simulation(
    config: Any,
    model: language_model.LanguageModel,
    embedder: Callable[[str], Any],
) -> dict[str, Any] | None:
  """Run the Reality Show simulation.

  Args:
    config: The scenario configuration module. Must provide
      `sample_parameters()` returning a WorldConfig, and optionally
      FOCAL_PLAYER_PREFAB, BACKGROUND_PLAYER_PREFAB, NUM_MAIN_PLAYERS,
      NUM_BACKGROUND_PLAYERS.
    model: The language model to use.
    embedder: The sentence embedder to use.

  Returns:
    A dictionary containing the simulation results.
  """
  # Call sample_parameters() to get the WorldConfig
  sampled = config.sample_parameters()

  # Extract contestant info from WorldConfig
  contestant_names = list(sampled.contestants.keys())
  if not contestant_names:
    raise ValueError("WorldConfig.contestants is empty.")

  # Split into focal and background players
  num_main = getattr(config, "NUM_MAIN_PLAYERS", len(contestant_names))
  num_background = getattr(config, "NUM_BACKGROUND_PLAYERS", 0)
  focal_players = contestant_names[:num_main]
  background_players = contestant_names[num_main : num_main + num_background]

  # Extract minigame configuration from WorldConfig
  selected_minigame = sampled.minigame
  cooperation_option = (
      selected_minigame.map_external_actions_to_schelling_diagram["cooperation"]
  )
  defection_option = (
      selected_minigame.map_external_actions_to_schelling_diagram["defection"]
  )
  cooperation_fn = selected_minigame.schelling_diagram.cooperation
  defection_fn = selected_minigame.schelling_diagram.defection

  # Create payoff handler
  payoff = RealityShowPayoff(
      player_names=contestant_names,
      cooperation_option=cooperation_option,
      defection_option=defection_option,
      cooperation_fn=cooperation_fn,
      defection_fn=defection_fn,
  )

  # Set up RNG and sample show title
  rng = random.Random(sampled.seed)
  show_title = rng.choice(CANDIDATE_SHOW_TITLES)
  show_description = rng.choice(CANDIDATE_SHOW_DESCRIPTIONS)
  show_title_and_description = (
      f'"{show_title}" is a reality TV show described as: "{show_description}"'
  )

  # Configure scenes using WorldConfig
  scenes = configure_scenes(
      player_names=contestant_names,
      sampled_settings=sampled,
  )

  # Build prefabs
  prefabs = {
      **helper_functions.get_package_classes(entity_prefabs),
      **helper_functions.get_package_classes(game_master_prefabs),
  }
  prefabs["rational__Entity"] = rational.Entity()
  prefabs["conversation_rules__GameMaster"] = (
      dialogic_and_dramaturgic.GameMaster()
  )
  prefabs["decision_rules__GameMaster"] = (
      game_theoretic_and_dramaturgic.GameMaster()
  )
  prefabs["puppet__Entity"] = puppet.Entity()

  focal_player_prefab = getattr(
      config, "FOCAL_PLAYER_PREFAB", "basic__Entity"
  )
  background_player_prefab = getattr(
      config, "BACKGROUND_PLAYER_PREFAB", "rational__Entity"
  )

  instances = []

  # Create focal player instances
  for name in focal_players:
    instances.append(
        prefab_lib.InstanceConfig(
            prefab=focal_player_prefab,
            role=prefab_lib.Role.ENTITY,
            params={
                "name": name,
                "goal": (
                    f"{name} wants to win the reality show by scoring the "
                    "most points in the minigames."
                ),
            },
        )
    )

  # Create background player instances
  for name in background_players:
    instances.append(
        prefab_lib.InstanceConfig(
            prefab=background_player_prefab,
            role=prefab_lib.Role.ENTITY,
            params={
                "name": name,
                "goal": (
                    f"{name} wants to win the reality show by scoring the "
                    "most points in the minigames."
                ),
            },
        )
    )

  # Shared memories for all contestants
  all_contestants_string = _get_all_contestant_names_string(contestant_names)

  shared_memories = [
      (
          f"{all_contestants_string} are contestants on a reality show: "
          f"{show_title}. There are no other contestants besides "
          f"{all_contestants_string}."
      ),
      show_title_and_description,
      GENERAL_BACKGROUND,
  ]

  # Player-specific memories from contestant data
  player_specific_memories = {}
  for name, data in sampled.contestants.items():
    traits = data.get("traits", "friendly and outgoing")
    catchphrase = data.get("catchphrase", "")
    memories = [
        f"{name} is a contestant on {show_title}.",
        f"{name} is {traits}.",
    ]
    if catchphrase:
      memories.append(f"{name}'s catchphrase is: {catchphrase}")
    player_specific_memories[name] = memories

  # Add initializer GM
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="formative_memories_initializer__GameMaster",
          role=prefab_lib.Role.INITIALIZER,
          params={
              "name": "initial setup rules",
              "next_game_master_name": "conversation rules",
              "shared_memories": shared_memories,
              "player_specific_memories": player_specific_memories,
          },
      )
  )

  # Add conversation GM
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="conversation_rules__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={
              "name": "conversation rules",
              "scenes": scenes,
          },
      )
  )

  # Add decision GM
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="decision_rules__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={
              "name": "decision rules",
              "scenes": scenes,
              "action_to_scores": payoff.action_to_scores,
              "scores_to_observation": payoff.scores_to_observation,
          },
      )
  )

  sim_config = prefab_lib.Config(
      default_premise=(
          f"It is {sampled.year}. "
          f"{all_contestants_string} are contestants on a reality TV show."
      ),
      default_max_steps=100,
      prefabs=prefabs,
      instances=instances,
  )

  logging.set_verbosity(logging.INFO)

  # Run simulation
  sim = simulation_lib.Simulation(
      config=sim_config,
      model=model,
      embedder=embedder,
  )

  structured_log = sim.play()

  # Return scores for test extraction
  joint_action = payoff.latest_joint_action
  cumulative_scores = payoff.get_cumulative_scores()

  return {
      "focal_scores": {
          name: cumulative_scores.get(name, 0.0) for name in focal_players
      },
      "background_scores": {
          name: cumulative_scores.get(name, 0.0) for name in background_players
      },
      "joint_action": joint_action,
      "payoff": payoff,
      "focal_players": focal_players,
      "background_players": background_players,
      "structured_log": structured_log,
  }
