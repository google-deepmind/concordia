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

"""A State Formation simulation using Concordia prefabs.

This environment simulates treaty negotiation between two pre-state villages.
Elders from each village must negotiate agreements for common defense against
barbarian raiders while managing their village's activities (farming, warrior
training, free time).
"""

import json
import math
import re
from typing import Any, Mapping, Sequence

from absl import logging
from concordia.components.agent import constant as agent_constant
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import make_observation
from concordia.components.game_master import scene_tracker as scene_tracker_lib
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.entity import minimal
from concordia.prefabs.entity import puppet
from concordia.prefabs.entity import rational
from concordia.prefabs.game_master import dialogic_and_dramaturgic
from concordia.prefabs.game_master import game_theoretic_and_dramaturgic
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib
from concordia.utils import helper_functions
from ml_collections import config_dict
import numpy as np


class AgreementAnalyzer:
  """Uses LLM to detect agreement from negotiation responses.

  Matches the deprecated version's _are_agricultural_resources_pooled logic.
  """

  def __init__(
      self,
      model,
      elder_names: Sequence[str],
      villager_names: Sequence[str],
  ):
    self._model = model
    self._elder_names = list(elder_names)
    self._villager_names = list(villager_names)

  def analyze_agreement(
      self, joint_action: Mapping[str, str | None]
  ) -> bool | None:
    """Analyze elder responses to determine if there's an agreement.

    Args:
      joint_action: Mapping from player name to their action/response.

    Returns:
      True if agreement detected, False if no agreement, None if not applicable
      (e.g., this is villager activity responses, not elder post-negotiation).
    """
    elder_responses = {}
    for name in self._elder_names:
      if name in joint_action and joint_action[name] is not None:
        elder_responses[name] = joint_action[name]

    if not elder_responses:
      return None

    villager_names_str = ", ".join(self._villager_names)

    prompt_parts = [
        "This is a social science experiment about treaty negotiation.",
        "Two village elders have just completed negotiations about whether to",
        "pool agricultural resources between their villages.",
        "",
        "Here are their statements about whether an agreement was reached:",
        "",
    ]

    for name, response in elder_responses.items():
      prompt_parts.append(f"{name} said: {response}")

    prompt_parts.extend([
        "",
        "Question: Is there evidence that both elders agree to pool",
        "agricultural products between villages such that a village with less",
        "food can be resupplied by a village with more food?",
        "",
        f"Only consider the following villagers: {villager_names_str}.",
        "If there is no clear agreement from both parties, answer 'no'.",
        "",
        "Answer with just 'yes' or 'no':",
    ])

    prompt = "\n".join(prompt_parts)

    try:
      response = self._model.sample_text(prompt, max_tokens=10)
      response_lower = response.strip().lower()
      return "yes" in response_lower
    except (AttributeError, TypeError):
      return False


# Default component keys for the AgreementDetector
DEFAULT_AGREEMENT_DETECTOR_KEY = "__agreement_detector__"


class AgreementDetector(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Detects agreement at the end of negotiation scenes.

  This component hooks into post_act to detect when the negotiation scene ends,
  then analyzes the full conversation transcript using an LLM to determine
  whether an agreement was reached. The result is recorded to memory and used
  to update the payoff handler's treaty status.
  """

  def __init__(
      self,
      model,
      payoff_handler: "StateFormationPayoff",
      negotiation_scene_prefix: str = "negotiation_",
      scene_tracker_key: str = scene_tracker_lib.DEFAULT_SCENE_TRACKER_COMPONENT_KEY,
      memory_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
      pre_act_label: str = "",
  ):
    """Initialize the AgreementDetector component.

    Args:
      model: The language model to use for conversation analysis.
      payoff_handler: The payoff handler whose treaty status will be updated.
      negotiation_scene_prefix: Prefix for negotiation scene names.
      scene_tracker_key: Component key for the scene tracker.
      memory_key: Component key for the memory component.
      pre_act_label: Label for pre_act (usually empty for this component).
    """
    super().__init__()
    self._model = model
    self._payoff = payoff_handler
    self._scene_prefix = negotiation_scene_prefix
    self._scene_tracker_key = scene_tracker_key
    self._memory_key = memory_key
    self._pre_act_label = pre_act_label
    self._last_scene_name: str | None = None

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    """Detect post_negotiation scene exit and analyze elder responses."""
    scene_tracker = self.get_entity().get_component(
        self._scene_tracker_key,
        type_=scene_tracker_lib.SceneTracker,
    )
    current_scene_type = scene_tracker.get_current_scene_type()
    current_scene_name = current_scene_type.name

    # Check if we just EXITED a post_negotiation scene
    was_post_negotiation = (
        self._last_scene_name is not None
        and self._last_scene_name.startswith("post_negotiation_")
    )
    is_different_scene = self._last_scene_name != current_scene_name

    # Trigger analysis when we EXIT a post_negotiation scene
    if was_post_negotiation and is_different_scene:
      self._analyze_elder_responses()

    self._last_scene_name = current_scene_name
    return ""

  def post_act(self, action_attempt: str) -> str:
    """Called after each action. No-op for this component."""
    return ""

  def _analyze_elder_responses(self) -> None:
    """Analyze elder responses from post_negotiation to determine agreement.

    This method extracts the elders' explicit statements about whether they
    believe an agreement was reached, then uses the LLM to determine if both
    elders agree on resource pooling.
    """
    memory = self.get_entity().get_component(
        self._memory_key,
        type_=memory_component.Memory,
    )

    # Get recent memories to find elder post-negotiation responses
    all_memories = memory.scan(lambda x: True)

    # Look for elder responses about agreement (from post_negotiation scene)
    # These will be in the format "Elder X -- ..." or include "agreement"
    elder_responses = []
    for mem in all_memories[-30:]:
      mem_lower = mem.lower()
      if "elder" in mem_lower and (
          "agreement" in mem_lower
          or "pool" in mem_lower
          or "resuppl" in mem_lower
          or " -- " in mem
      ):
        elder_responses.append(mem)

    if not elder_responses:
      logging.info("No elder responses found for agreement detection.")
      return

    context = "\n".join(elder_responses[-10:])

    prompt_parts = [
        "This is a social science experiment about treaty negotiation.",
        "Two village elders were asked whether there was an agreement to pool",
        "agricultural resources between villages.",
        "",
        "Here are their statements about whether agreement was reached:",
        "",
        context,
        "",
        "Question: Based on these statements, did BOTH elders explicitly agree",
        "to pool agricultural products between villages such that a village",
        "with less food can be resupplied by a village with more food?",
        "",
        "If there is no clear agreement from both parties, answer 'no'.",
        "Both elders must indicate agreement for this to count as 'yes'.",
        "",
        "Answer with just 'yes' or 'no':",
    ]

    prompt = "\n".join(prompt_parts)

    try:
      response = self._model.sample_text(prompt, max_tokens=10)
      response_lower = response.strip().lower()
      agreed = "yes" in response_lower
    except (AttributeError, TypeError):
      agreed = False

    self._payoff.set_treaty(agreed)

    decision = "agreed" if agreed else "did not agree"
    memory.add(
        f"[agreement decision] The elders {decision} on resource pooling."
    )

    logging.info(
        "Agreement detection: elders %s (treaty %s)",
        decision,
        "ACTIVE" if agreed else "INACTIVE",
    )

    self._logging_channel({
        "Summary": f"Agreement detection: {decision}",
        "Treaty Active": agreed,
        "Elder responses analyzed": len(elder_responses),
    })

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {"last_scene_name": self._last_scene_name}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    last_scene = state.get("last_scene_name")
    if isinstance(last_scene, str):
      self._last_scene_name = last_scene
    else:
      self._last_scene_name = None


def _sigmoid(x: float) -> float:
  """Sigmoid with parameter 0.3 for supralinear growth.

  Args:
    x: Input value.

  Returns:
    Sigmoid-transformed value.
  """
  return 1 / (1 + math.exp(-(x / 0.3)))


def _sigmoidlike_fn(x: float) -> float:
  """Transform proportion to supralinear production value.

  Maps 0.0 to 0.0 and 1.0 to 1.0, with supralinear growth around 0.5.
  Specifically: sigmoidlike_fn(0.5) ≈ 0.73

  Args:
    x: Input proportion (0.0 to 1.0).

  Returns:
    Transformed production value.
  """
  return (_sigmoid(x) - 0.5) / (_sigmoid(1.0) - 0.5)


class StateFormationPayoff:
  """Handles scoring for the state formation simulation.

  Uses sigmoid production functions matching the deprecated version:
  - Each activity proportion is transformed via _sigmoidlike_fn
  - Defense and agriculture are binary gates (0 or 1 based on thresholds)
  - Score = defense * agriculture * free_time (multiplicative)
  """

  def __init__(
      self,
      player_names: Sequence[str],
      activity_options: tuple[str, str, str],
      village_a_name: str = "Village A",
      village_b_name: str = "Village B",
      village_assignments: Mapping[str, str] | None = None,
      defense_threshold: float = 0.25,
      starvation_threshold: float = 0.1,
      free_time_reward: float = 1.0,
      farming_reward: float = 0.5,
      warrior_reward: float = 0.5,
      starvation_penalty: float = -5.0,
      raid_success_penalty: float = -3.0,
      agreement_analyzer: "AgreementAnalyzer | None" = None,
      elder_names: Sequence[str] | None = None,
      event_samplers: Mapping[str, Any] | None = None,
  ):
    """Initialize the payoff calculator.

    Args:
      player_names: Names of the player characters.
      activity_options: Tuple of (farming, warrior_training, free_time).
      village_a_name: Name of village A.
      village_b_name: Name of village B.
      village_assignments: Mapping from player name to village name.
      defense_threshold: Minimum sigmoid-transformed defense needed.
      starvation_threshold: Minimum sigmoid-transformed farming needed.
      free_time_reward: Weight for free time in observation text.
      farming_reward: Weight for farming in observation text.
      warrior_reward: Weight for warrior in observation text.
      starvation_penalty: Not used in sigmoid mode (kept for compatibility).
      raid_success_penalty: Not used in sigmoid mode (kept for compatibility).
      agreement_analyzer: Optional analyzer for elder treaty responses.
      elder_names: Names of elder players for agreement detection.
      event_samplers: Optional dict of event sampling callables from the config.
        with Keys: 'defense_fail', 'defense_success', 'food_fail',
        'food_success', 'no_treaty', 'treaty'. Each is a callable returning a
        narrative string.
    """
    self._player_names = list(player_names)
    # Config order is (free_time, farming, warrior)
    self._free_time_option = activity_options[0]
    self._farming_option = activity_options[1]
    self._warrior_option = activity_options[2]
    self._village_a_name = village_a_name
    self._village_b_name = village_b_name
    self._village_assignments = dict(village_assignments or {})
    self._defense_threshold = defense_threshold
    self._starvation_threshold = starvation_threshold
    self._free_time_reward = free_time_reward
    self._farming_reward = farming_reward
    self._warrior_reward = warrior_reward
    self._starvation_penalty = starvation_penalty
    self._raid_success_penalty = raid_success_penalty

    self._latest_joint_action: dict[str, dict[str, float]] = {}
    self._cumulative_scores: dict[str, float] = {
        name: 0.0 for name in player_names
    }
    self._treaty_active = False
    self._agreement_analyzer = agreement_analyzer
    self._elder_names = list(elder_names) if elder_names else []
    self._event_samplers = dict(event_samplers) if event_samplers else {}
    # Stores per-round events generated during action_to_scores, read by
    # scores_to_observation so that narrative events reach both GM and players.
    self._latest_events: list[str] = []

  @property
  def latest_joint_action(self) -> Mapping[str, dict[str, float]]:
    """Returns the latest joint action passed to action_to_scores."""
    return self._latest_joint_action

  def get_cumulative_scores(self) -> Mapping[str, float]:
    """Returns cumulative scores across all rounds."""
    return self._cumulative_scores

  @property
  def treaty_active(self) -> bool:
    """Returns whether the resource pooling treaty is active."""
    return self._treaty_active

  def set_treaty(self, active: bool) -> None:
    """Set whether resource pooling treaty is active."""
    self._treaty_active = active

  def _parse_activity_proportions(self, action: str | None) -> dict[str, float]:
    """Parse activity proportions from action text.

    Supports multiple formats:
    - JSON: {"farming": 0.5, "warrior": 0.3, "free_time": 0.2}
    - Colon: farming: 0.5, warrior: 0.3, free time: 0.2

    Args:
      action: The action text to parse.

    Returns:
      Dictionary mapping activity names to proportions (default: equal split).
    """
    default_proportions = {
        self._farming_option: 1.0 / 3.0,
        self._warrior_option: 1.0 / 3.0,
        self._free_time_option: 1.0 / 3.0,
    }

    if action is None:
      return default_proportions

    # Try JSON parsing first
    json_match = re.search(r"\{[^}]+\}", action)
    if json_match:
      try:
        parsed = json.loads(json_match.group())
        farming = float(parsed.get("farming", 0))
        warrior = float(parsed.get("warrior", 0))
        free_time = float(parsed.get("free_time", 0))
        total = farming + warrior + free_time
        if total > 0:
          return {
              self._farming_option: farming / total,
              self._warrior_option: warrior / total,
              self._free_time_option: free_time / total,
          }
      except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try colon-separated format: "activity: proportion"
    action_lower = action.lower()
    proportions = {}
    for activity in [
        self._farming_option,
        self._warrior_option,
        self._free_time_option,
    ]:
      # Allow "warrior" as an alias for "training as a warrior"
      search_term = (
          "warrior" if activity == self._warrior_option else activity.lower()
      )
      pattern = rf"{re.escape(search_term)}\s*:\s*([\d.]+)"
      match = re.search(pattern, action_lower)
      if match:
        try:
          proportions[activity] = float(match.group(1).rstrip("."))
        except ValueError:
          pass

    if proportions:
      total = sum(proportions.values())
      if total > 0:
        for activity in default_proportions:
          if activity not in proportions:
            proportions[activity] = 0.0
        return {k: v / total for k, v in proportions.items()}

    return default_proportions

  def action_to_scores(
      self, joint_action: Mapping[str, str | None]
  ) -> Mapping[str, float]:
    """Maps joint actions to scores for all players.

    Uses sigmoid production functions matching the deprecated version:
    - Each activity proportion is transformed via _sigmoidlike_fn
    - Per-village aggregation with sigmoid-transformed values
    - Defense and agriculture are binary gates (1 if above threshold, 0 if not)
    - Score = defense * agriculture * free_time (multiplicative)

    Args:
      joint_action: Mapping from player name to their action string containing
        proportions (e.g., 'farming: 0.5, warrior: 0.3, free time: 0.2').

    Returns:
      Mapping from player name to their score for this round.
    """
    # Check if post_negotiation response (elders answering about agreement)
    if self._agreement_analyzer and self._elder_names:
      elder_responded = any(
          name in joint_action and joint_action[name] is not None
          for name in self._elder_names
      )
      if elder_responded:
        # Check if responses mention agreement/pooling (post_negotiation scene)
        has_agreement_keywords = False
        for name in self._elder_names:
          response = joint_action.get(name)
          if response and isinstance(response, str):
            response_lower = response.lower()
            if (
                "agreement" in response_lower
                or "pool" in response_lower
                or "resuppl" in response_lower
            ):
              has_agreement_keywords = True
              break

        if has_agreement_keywords:
          agreed = self._agreement_analyzer.analyze_agreement(joint_action)
          if agreed is not None:
            self._treaty_active = agreed
            logging.info(
                "Agreement detection from elder responses: treaty %s",
                "ACTIVE" if agreed else "INACTIVE",
            )

    scores: dict[str, float] = {}

    # Return early if this is an elder-only scene (e.g. post_negotiation)
    active_players = [p for p, act in joint_action.items() if act is not None]
    if self._elder_names and all(
        p in self._elder_names for p in active_players
    ):
      return scores

    player_proportions: dict[str, dict[str, float]] = {}
    for player, action in joint_action.items():
      if player in self._player_names and action is not None:
        player_proportions[player] = self._parse_activity_proportions(action)

    self._latest_joint_action = player_proportions

    if not player_proportions:
      return scores

    village_a_players = [
        p
        for p in self._player_names
        if self._village_assignments.get(p) == self._village_a_name
    ]
    village_b_players = [
        p
        for p in self._player_names
        if self._village_assignments.get(p) == self._village_b_name
    ]

    def get_village_activity_product(
        activity_name: str, village_players: list[str]
    ) -> float:
      # Only count players who submitted activity actions (villagers)
      active_players = [p for p in village_players if p in player_proportions]
      if not active_players:
        return 0.0
      total_product = 0.0
      for player in active_players:
        prop = player_proportions[player].get(activity_name, 0.0)
        total_product += _sigmoidlike_fn(prop)
      return total_product / len(active_players)

    defense_a = get_village_activity_product(
        self._warrior_option, village_a_players
    )
    defense_b = get_village_activity_product(
        self._warrior_option, village_b_players
    )
    farming_a = get_village_activity_product(
        self._farming_option, village_a_players
    )
    farming_b = get_village_activity_product(
        self._farming_option, village_b_players
    )
    freetime_a = get_village_activity_product(
        self._free_time_option, village_a_players
    )
    freetime_b = get_village_activity_product(
        self._free_time_option, village_b_players
    )

    raw_overall_defense = np.mean([defense_a, defense_b])
    defense_gate = (
        1.0 if raw_overall_defense >= self._defense_threshold else 0.0
    )

    # Generate defense event narrative
    events: list[str] = []
    if defense_gate == 0.0:
      sampler = self._event_samplers.get("defense_fail")
      events.append(
          sampler() if sampler else "The barbarian raid could not be repelled."
      )
    else:
      sampler = self._event_samplers.get("defense_success")
      events.append(
          sampler()
          if sampler
          else "The barbarian raiders were successfully repelled."
      )

    farming_per_village = {
        self._village_a_name: farming_a,
        self._village_b_name: farming_b,
    }
    freetime_per_village = {
        self._village_a_name: freetime_a,
        self._village_b_name: freetime_b,
    }

    # Generate treaty event narrative
    if self._treaty_active:
      sampler = self._event_samplers.get("treaty")
      events.append(
          sampler()
          if sampler
          else "The treaty to pool agricultural resources is in effect."
      )
    else:
      sampler = self._event_samplers.get("no_treaty")
      events.append(
          sampler()
          if sampler
          else "There is no treaty to pool agricultural resources."
      )

    for player in self._player_names:
      player_village = self._village_assignments.get(player)
      if not player_village:
        continue

      if self._treaty_active:
        raw_agriculture = max(farming_per_village.values())
      else:
        raw_agriculture = farming_per_village.get(player_village, 0.0)

      agriculture_gate = (
          1.0 if raw_agriculture >= self._starvation_threshold else 0.0
      )
      free_time_value = freetime_per_village.get(player_village, 0.0)

      score = defense_gate * agriculture_gate * free_time_value
      scores[player] = score
      self._cumulative_scores[player] += score

    # Generate food event narrative
    # Use overall agriculture for narrative (any village failing is notable)
    if any(
        v < self._starvation_threshold for v in farming_per_village.values()
    ):
      sampler = self._event_samplers.get("food_fail")
      events.append(
          sampler() if sampler else "Some villages failed to grow enough food."
      )
    else:
      sampler = self._event_samplers.get("food_success")
      events.append(
          sampler() if sampler else "The harvest was successful this year."
      )

    self._latest_events = events

    return scores

  def scores_to_observation(
      self, scores: Mapping[str, float]
  ) -> Mapping[str, str]:
    """Generates descriptive observations for each player.

    Includes both activity statistics and narrative event descriptions from the
    config's sample_event_* functions. The event narratives are sent to both GM
    and players (via MakeObservation queue), matching v1.0 behavior.

    Args:
      scores: Mapping from player name to their score this round.

    Returns:
      Mapping from player name to their observation string.
    """
    player_proportions = self._latest_joint_action
    results = {}

    if not player_proportions:
      return results

    avg_farming = sum(
        p.get(self._farming_option, 0.0) for p in player_proportions.values()
    ) / max(len(player_proportions), 1)
    avg_warrior = sum(
        p.get(self._warrior_option, 0.0) for p in player_proportions.values()
    ) / max(len(player_proportions), 1)
    avg_freetime = sum(
        p.get(self._free_time_option, 0.0) for p in player_proportions.values()
    ) / max(len(player_proportions), 1)

    # Build event narrative block shared by all players
    event_block = " ".join(self._latest_events) if self._latest_events else ""

    for name in self._player_names:
      if name not in player_proportions:
        continue

      props = player_proportions[name]
      score = scores.get(name, 0.0)
      cumulative = self._cumulative_scores.get(name, 0.0)

      farm_pct = props.get(self._farming_option, 0.0) * 100
      war_pct = props.get(self._warrior_option, 0.0) * 100
      free_pct = props.get(self._free_time_option, 0.0) * 100

      observation_parts = [
          (
              f"[season outcome] {name} allocated: {farm_pct:.0f}% farming, "
              f"{war_pct:.0f}% warrior training, {free_pct:.0f}% leisure "
              f"(score: {score:.1f}). Village averages: "
              f"{avg_farming*100:.0f}% farm, {avg_warrior*100:.0f}% defense, "
              f"{avg_freetime*100:.0f}% leisure. Total: {cumulative:.1f}."
          ),
      ]
      if event_block:
        observation_parts.append(event_block)

      results[name] = " ".join(observation_parts)

    return results


def configure_scenes(
    elder_names: Sequence[str],
    villager_names: Sequence[str],
    num_years: int,
    village_a_name: str,
    village_b_name: str,
    conversation_call_to_action: str,
    activity_call_to_action: str,
    post_negotiation_call_to_action: str,
    debrief_call_to_action: str,
    home_premise: str,
    negotiation_premise: str,
    activity_premise: str,
) -> Sequence[scene_lib.SceneSpec]:
  """Configure scenes for the state formation simulation.

  Scene structure per year:
  1. Home (Elders) - reflect before meeting
  2. Negotiation (Elders) - diplomatic discussion
  3. Post-Negotiation (Elders) - state view on agreement reached
  4. Return Home (Elders) - go back to villages
  5. Activity Decision (Villagers) - decide activities
  6. Debrief (All players year 1, otherwise empty) - reflect on year

  Args:
    elder_names: Names of the elder players (one per village).
    villager_names: Names of the villager players (supporting characters).
    num_years: Number of years to simulate.
    village_a_name: Name of village A.
    village_b_name: Name of village B.
    conversation_call_to_action: CTA for conversation scenes.
    activity_call_to_action: CTA for activity decision scenes.
    post_negotiation_call_to_action: CTA for post-negotiation reflection.
    debrief_call_to_action: CTA for year-end debrief.
    home_premise: Premise for home scenes.
    negotiation_premise: Premise for negotiation scenes.
    activity_premise: Premise for activity decision scenes.

  Returns:
    List of scene specifications.
  """
  scenes = []
  villages = [village_a_name, village_b_name]
  all_player_names = list(elder_names[:2]) + list(villager_names)

  for year_idx in range(num_years):
    # 1. Home scene (elders reflect before meeting)
    home_scene_type = scene_lib.SceneTypeSpec(
        name=f"home_scene_{year_idx}",
        game_master_name="conversation rules",
        action_spec=entity_lib.free_action_spec(
            call_to_action=conversation_call_to_action,
        ),
    )

    home_premises = {
        name: [
            f"Year {year_idx + 1} begins.",
            home_premise.format(name=name, village=villages[i]),
        ]
        for i, name in enumerate(elder_names[:2])
    }

    scenes.append(
        scene_lib.SceneSpec(
            scene_type=home_scene_type,
            participants=list(elder_names[:2]),
            num_rounds=1,
            premise=home_premises,
        )
    )

    # 2. Negotiation scene (elders meet at hill of accord)
    negotiation_scene_type = scene_lib.SceneTypeSpec(
        name=f"negotiation_{year_idx}",
        game_master_name="conversation rules",
        action_spec=entity_lib.free_action_spec(
            call_to_action=conversation_call_to_action,
        ),
    )

    negotiation_premises = {
        name: [
            negotiation_premise.format(
                player_name=name.replace("Elder ", ""),
                village_name=villages[i],
            ),
        ]
        for i, name in enumerate(elder_names[:2])
    }

    scenes.append(
        scene_lib.SceneSpec(
            scene_type=negotiation_scene_type,
            participants=list(elder_names[:2]),
            num_rounds=3,
            premise=negotiation_premises,
        )
    )

    # 3. Post-negotiation scene (elders state their view on agreement)
    post_negotiation_scene_type = scene_lib.SceneTypeSpec(
        name=f"post_negotiation_{year_idx}",
        game_master_name="decision rules",
        action_spec=entity_lib.free_action_spec(
            call_to_action=post_negotiation_call_to_action,
            tag="announcement",
        ),
    )

    # All participants need a premise entry (even if empty)
    post_negotiation_premises = {name: [] for name in elder_names[:2]}

    scenes.append(
        scene_lib.SceneSpec(
            scene_type=post_negotiation_scene_type,
            participants=list(elder_names[:2]),
            num_rounds=2,
            premise=post_negotiation_premises,
        )
    )

    # 4. Return home scene (elders go back to villages)
    return_home_scene_type = scene_lib.SceneTypeSpec(
        name=f"return_home_{year_idx}",
        game_master_name="conversation rules",
        action_spec=entity_lib.free_action_spec(
            call_to_action=conversation_call_to_action,
        ),
    )

    return_home_premises = {
        name: [
            f"{name} returns to {villages[i]} after the meeting.",
        ]
        for i, name in enumerate(elder_names[:2])
    }

    scenes.append(
        scene_lib.SceneSpec(
            scene_type=return_home_scene_type,
            participants=list(elder_names[:2]),
            num_rounds=1,
            premise=return_home_premises,
        )
    )

    # 5. Activity/Resolution scene - VILLAGERS decide activities
    activity_decision_scene_type = scene_lib.SceneTypeSpec(
        name=f"activity_decision_{year_idx}",
        game_master_name="decision rules",
        action_spec=entity_lib.free_action_spec(
            call_to_action=activity_call_to_action,
            tag="activity_allocation",
        ),
    )

    activity_premises = {
        name: [
            f"Year {year_idx + 1}, activity season.",
            activity_premise.format(name=name),
        ]
        for name in villager_names
    }

    scenes.append(
        scene_lib.SceneSpec(
            scene_type=activity_decision_scene_type,
            participants=list(villager_names),
            num_rounds=len(villager_names),
            premise=activity_premises,
        )
    )

    # 6. Debrief scene - all players reflect on the year
    debrief_scene_type = scene_lib.SceneTypeSpec(
        name=f"debrief_{year_idx}",
        game_master_name="conversation rules",
        action_spec=entity_lib.free_action_spec(
            call_to_action=debrief_call_to_action,
            tag="reflection",
        ),
    )

    # Year 1: all players participate; subsequent years: skip debrief
    if year_idx == 0:
      debrief_participants = all_player_names
      # All participants need a premise entry (even if empty)
      debrief_premises = {name: [] for name in debrief_participants}

      scenes.append(
          scene_lib.SceneSpec(
              scene_type=debrief_scene_type,
              participants=debrief_participants,
              num_rounds=1,
              premise=debrief_premises,
          )
      )

  return scenes


def run_simulation(
    config: Any,
    model: Any,
    embedder: Any,
) -> dict[str, Any]:
  """Run the state formation simulation.

  Args:
    config: Config module with sample_parameters() function.
    model: Language model instance.
    embedder: Embedding function.

  Returns:
    Dictionary with simulation results including focal_scores, joint_action,
    focal_players, background_players, and structured_log.
  """
  # Load config from sample_parameters()
  config = config.sample_parameters()

  # Extract village names
  village_a_name = config.village_a_name
  village_b_name = config.village_b_name

  # Extract elder names with "Elder" prefix
  elder_a_raw = config.main_characters.a.name
  elder_b_raw = config.main_characters.b.name
  elder_names = [f"Elder {elder_a_raw}", f"Elder {elder_b_raw}"]

  # Extract supporting characters (tuples of (name, gender))
  supporting_a = list(config.supporting_characters.a)
  supporting_b = list(config.supporting_characters.b)
  villager_names_a = [name for name, _ in supporting_a]
  villager_names_b = [name for name, _ in supporting_b]
  villager_names = villager_names_a + villager_names_b

  # Create mapping from player name to their village (all players)
  village_assignments = {}
  for name in villager_names_a:
    village_assignments[name] = village_a_name
  for name in villager_names_b:
    village_assignments[name] = village_b_name
  village_assignments[elder_names[0]] = village_a_name
  village_assignments[elder_names[1]] = village_b_name

  # Activity options from config (free_time, farming, warrior)
  activity_options = config.activities

  # Number of years
  num_years = config.num_years

  # All players (elders + villagers) should be scored
  all_scored_players = list(elder_names) + villager_names

  # Create AgreementAnalyzer for analyzing elder post-negotiation responses
  agreement_analyzer = AgreementAnalyzer(
      model=model,
      elder_names=elder_names,
      villager_names=villager_names,
  )

  # Build event samplers from config's sample_event_* lambdas
  event_samplers = {
      "defense_fail": config.sample_event_of_failing_to_repel_barbarians,
      "defense_success": config.sample_event_of_success_repelling_barbarians,
      "food_fail": config.sample_event_of_failing_to_grow_food,
      "food_success": config.sample_event_of_success_growing_food,
      "no_treaty": config.sample_event_no_treaty_in_effect,
      "treaty": config.sample_event_treaty_in_effect,
  }

  # Create payoff handler
  payoff = StateFormationPayoff(
      player_names=all_scored_players,
      activity_options=activity_options,
      village_a_name=village_a_name,
      village_b_name=village_b_name,
      village_assignments=village_assignments,
      defense_threshold=config.defense_threshold,
      starvation_threshold=config.starvation_threshold,
      agreement_analyzer=agreement_analyzer,
      elder_names=elder_names,
      event_samplers=event_samplers,
  )

  # Negotiation phase premise - comes from config, already has embedded names
  # but still has {player_name} and {village_name} placeholders for per-elder
  # formatting in configure_scenes.
  negotiation_premise = config.negotiation_phase_premise

  # Call to action strings
  conversation_call_to_action = "What does {name} say next?"
  activity_call_to_action = (
      "How does {name} decide to allocate their time this season? "
      "Please specify proportions for farming, warrior training, and "
      "free time (must sum to 1.0). Example: "
      "farming: 0.5, warrior: 0.3, free time: 0.2"
  )
  post_negotiation_call_to_action = (
      "In {name}'s view, was there an agreement to pool "
      "agricultural products between villages such that a "
      "village with less food can be resupplied by a village "
      "with more food?"
  )
  debrief_call_to_action = (
      "How does {name} feel about how this year went? "
      "What went well? What did not go well? What does "
      "{name} feel could be improved for next year?"
  )

  # Mapping from shorthand keys (used in puppet configs to avoid dots in
  # ConfigDict keys) to actual call_to_action strings.
  call_to_action_key_map = {
      "post_negotiation": post_negotiation_call_to_action,
      "activity": activity_call_to_action,
      "conversation": conversation_call_to_action,
      "debrief": debrief_call_to_action,
  }

  # Activity premise template
  activity_premise = (
      "It is time for {name} to decide how to allocate their time "
      "this season between farming, warrior training, and free time."
  )

  # Configure scenes using config-derived premises
  scenes = configure_scenes(
      elder_names=elder_names,
      villager_names=villager_names,
      num_years=num_years,
      village_a_name=village_a_name,
      village_b_name=village_b_name,
      conversation_call_to_action=conversation_call_to_action,
      activity_call_to_action=activity_call_to_action,
      post_negotiation_call_to_action=post_negotiation_call_to_action,
      debrief_call_to_action=debrief_call_to_action,
      home_premise=config.home_scene_premise,
      negotiation_premise=negotiation_premise,
      activity_premise=activity_premise,
  )

  # Create AgreementDetector component for scene-end agreement analysis
  agreement_detector = AgreementDetector(
      model=model,
      payoff_handler=payoff,
      negotiation_scene_prefix="negotiation_",
  )

  # Build prefabs
  prefabs = {
      **helper_functions.get_package_classes(entity_prefabs),
      **helper_functions.get_package_classes(game_master_prefabs),
  }
  prefabs["rational__Entity"] = rational.Entity()
  # Conversation GM (plain prefab - extra_components passed via InstanceConfig)
  prefabs["conversation_rules__GameMaster"] = (
      dialogic_and_dramaturgic.GameMaster()
  )
  prefabs["decision_rules__GameMaster"] = (
      game_theoretic_and_dramaturgic.GameMaster()
  )

  prefabs["puppet__Entity"] = puppet.Entity()

  prefabs["minimal__Entity"] = minimal.Entity()

  focal_player_prefab = "rational__Entity"
  background_player_prefab = "minimal__Entity"

  instances = []

  # Create elder instances
  # Check if config provides fixed responses (for puppet tests)
  player_fixed_responses = getattr(config, "player_fixed_responses", {})
  if isinstance(player_fixed_responses, config_dict.ConfigDict):
    player_fixed_responses = dict(player_fixed_responses)

  elder_goal = (
      "Elder {name} wants to negotiate the best possible alliance for "
      "mutual defense and resource sharing between the villages."
  )

  for name in elder_names:
    goal = elder_goal.format(name=name)
    fixed_responses = {}

    if name in player_fixed_responses:
      raw_responses = player_fixed_responses[name]
      if isinstance(raw_responses, (dict, config_dict.ConfigDict)):
        raw_dict = dict(raw_responses)
        # Expand shorthand keys to actual call_to_action strings
        for key, value in raw_dict.items():
          actual_key = call_to_action_key_map.get(key, key)
          fixed_responses[actual_key] = value
      else:
        fixed_responses[activity_call_to_action] = raw_responses

    if fixed_responses:
      instances.append(
          prefab_lib.InstanceConfig(
              prefab="puppet__Entity",
              role=prefab_lib.Role.ENTITY,
              params={  # pyrefly: ignore[bad-argument-type]
                  "name": name,
                  "goal": goal,
                  "fixed_responses": fixed_responses,
              },
          )
      )
    else:
      instances.append(
          prefab_lib.InstanceConfig(
              prefab=focal_player_prefab,
              role=prefab_lib.Role.ENTITY,
              params={
                  "name": name,
                  "goal": goal,
              },
          )
      )

  # Create villager instances (supporting characters who make activity choices)
  # Villagers get a Constant component describing their perspective, matching
  # the v1.0 pattern of injecting villager_how_things_are_constant.
  villager_goal = "{name} wants to prosper."

  for villager_name in villager_names:
    goal = villager_goal.format(name=villager_name)
    fixed_responses = {}

    if villager_name in player_fixed_responses:
      raw_responses = player_fixed_responses[villager_name]
      if isinstance(raw_responses, (dict, config_dict.ConfigDict)):
        raw_dict = dict(raw_responses)
        # Expand shorthand keys to actual call_to_action strings
        for key, value in raw_dict.items():
          actual_key = call_to_action_key_map.get(key, key)
          fixed_responses[actual_key] = value
      else:
        fixed_responses[activity_call_to_action] = raw_responses

    # Build the Constant component for how_things_are
    village = village_assignments[villager_name]
    if village == village_a_name:
      how_things_are_text = (
          config.villager_how_things_are_constant.village_a.format(
              name=villager_name, village_name=village_a_name
          )
      )
    else:
      how_things_are_text = (
          config.villager_how_things_are_constant.village_b.format(
              name=villager_name, village_name=village_b_name
          )
      )

    how_things_are_component = agent_constant.Constant(
        state=how_things_are_text,
        pre_act_label="\nHow things are",
    )

    if fixed_responses:
      instances.append(
          prefab_lib.InstanceConfig(
              prefab="puppet__Entity",
              role=prefab_lib.Role.ENTITY,
              params={  # pyrefly: ignore[bad-argument-type]
                  "name": villager_name,
                  "goal": goal,
                  "fixed_responses": fixed_responses,
              },
          )
      )
    else:
      instances.append(
          prefab_lib.InstanceConfig(
              prefab=background_player_prefab,
              role=prefab_lib.Role.ENTITY,
              params={  # pyrefly: ignore[bad-argument-type]
                  "name": villager_name,
                  "goal": goal,
                  "extra_components": {
                      "how_things_are": how_things_are_component,
                  },
              },
          )
      )

  # Shared memories
  # Start with basic setting and barbarian raid info from config
  shared_memories = [config.basic_setting]
  shared_memories.extend(config.barbarian_raid_info)

  # Add per-village cultural elements (from config.villages)
  for text in config.villages.a:
    shared_memories.append(text)
  for text in config.villages.b:
    shared_memories.append(text)

  # Player-specific memories for elders
  player_specific_memories = {}
  for i, name in enumerate(elder_names):
    village = village_a_name if i == 0 else village_b_name
    player_specific_memories[name] = [
        f"{name} is the elder representative of {village}.",
        f"{name} negotiates on behalf of {village} at diplomatic meetings.",
    ]
    # Add the negotiation objective thought if present in config
    if hasattr(config, "negotiation_objective_thought"):
      player_specific_memories[name].append(
          config.negotiation_objective_thought
      )

  # Player-specific memories for villagers
  for villager_name in villager_names:
    village = village_assignments[villager_name]
    elder_name = elder_names[0] if village == village_a_name else elder_names[1]
    player_specific_memories[villager_name] = [
        f"{villager_name} is a villager from {village}.",
        (
            f"{villager_name} knows that {elder_name} will represent "
            f"{village} in negotiations with the other village about "
            "defense against the barbarian raiders."
        ),
    ]

  # Add initializer GM
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="formative_memories_initializer__GameMaster",
          role=prefab_lib.Role.INITIALIZER,
          params={  # pyrefly: ignore[bad-argument-type]
              "name": "initial setup rules",
              "next_game_master_name": "conversation rules",
              "shared_memories": shared_memories,
              "player_specific_memories": player_specific_memories,
          },
      )
  )

  # Create shared observation queue
  shared_observation_queue = make_observation.ObservationQueue()

  # Add conversation GM (handles negotiation scenes)
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="conversation_rules__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={  # pyrefly: ignore[bad-argument-type]
              "name": "conversation rules",
              "scenes": scenes,
              "external_queue": shared_observation_queue,
          },
      )
  )

  # Add decision GM with AgreementDetector (handles post-negotiation/activity)
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="decision_rules__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={  # pyrefly: ignore[bad-argument-type]
              "name": "decision rules",
              "scenes": scenes,
              "action_to_scores": payoff.action_to_scores,
              "scores_to_observation": payoff.scores_to_observation,
              "external_queue": shared_observation_queue,
              "extra_components": {
                  DEFAULT_AGREEMENT_DETECTOR_KEY: agreement_detector,
              },
          },
      )
  )

  # Create simulation config
  scenario_premise = (
      f"It is the year {config.times.setup.year}. {config.basic_setting}"
  )

  sim_config = prefab_lib.Config(
      default_premise=scenario_premise,
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

  sim_log = sim.play()

  # Return scores for test extraction
  joint_action = payoff.latest_joint_action
  cumulative_scores = payoff.get_cumulative_scores()

  return {
      "focal_scores": {
          name: cumulative_scores.get(name, 0.0) for name in elder_names
      },
      "background_scores": {
          name: cumulative_scores.get(name, 0.0) for name in villager_names
      },
      "joint_action": joint_action,
      "payoff": payoff,
      "focal_players": list(elder_names),
      "background_players": villager_names,
      "structured_log": sim_log,
  }
