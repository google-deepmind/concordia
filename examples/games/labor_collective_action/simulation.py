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

"""A Labor Collective Action simulation using Concordia prefabs.

This environment simulates a labor strike scenario where workers must decide
daily whether to join a strike or go to work. The boss (a supporting player)
decides whether to raise wages or hold firm based on strike pressure.
"""

import datetime
import random
from typing import Any, Callable, Mapping, Sequence

from absl import logging
from concordia.components.game_master import inventory as inventory_lib
from concordia.components.game_master import make_observation
from concordia.document import interactive_document
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


class LaborStrikePayoff:
  """Handles scoring for the labor strike simulation.

  Workers earn wages when they go to work (defect from strike).
  Workers earn nothing when they strike (cooperate with strike).
  Strike pressure is calculated as the fraction of workers striking.
  """

  def __init__(
      self,
      player_names: Sequence[str],
      strike_option: str,
      work_option: str,
      initial_wage: float,
      daily_expenses: float = 0.0,
  ):
    """Initialize the payoff calculator.

    Args:
      player_names: Names of the worker players.
      strike_option: The action string corresponding to striking.
      work_option: The action string corresponding to working.
      initial_wage: Starting wage per day of work.
      daily_expenses: Daily living expenses (negative value).
    """
    self._player_names = player_names
    self._strike_option = strike_option
    self._work_option = work_option
    self._current_wage = initial_wage
    self._daily_expenses = daily_expenses
    self._latest_joint_action: dict[str, str] = {}
    self._cumulative_scores: dict[str, float] = {
        name: 0.0 for name in player_names
    }
    self._latest_strike_pressure = 0.0

  @property
  def latest_joint_action(self) -> Mapping[str, str]:
    """Returns the latest joint action passed to action_to_scores."""
    return self._latest_joint_action

  @property
  def current_wage(self) -> float:
    """Returns the current wage."""
    return self._current_wage

  def set_wage(self, new_wage: float) -> None:
    """Update the wage (called when boss caves)."""
    self._current_wage = new_wage

  def get_cumulative_scores(self) -> Mapping[str, float]:
    """Returns cumulative scores across all rounds."""
    return self._cumulative_scores

  def get_strike_pressure(self) -> float:
    """Returns the latest strike pressure (fraction of workers striking)."""
    return self._latest_strike_pressure

  def _is_strike(self, action: str | None) -> bool:
    """Check if an action is striking."""
    if action is None:
      return False
    action_lower = action.lower().strip()
    strike_lower = self._strike_option.lower().strip()
    work_lower = self._work_option.lower().strip()

    if action_lower == strike_lower:
      return True
    if strike_lower in action_lower and work_lower not in action_lower:
      return True
    return False

  def _is_work(self, action: str | None) -> bool:
    """Check if an action is working."""
    if action is None:
      return False
    action_lower = action.lower().strip()
    strike_lower = self._strike_option.lower().strip()
    work_lower = self._work_option.lower().strip()

    if action_lower == work_lower:
      return True
    if work_lower in action_lower and strike_lower not in action_lower:
      return True
    return False

  def _validate_action(self, action: str | None) -> str:
    """Validate and normalize an action string."""
    if action is None:
      return self._work_option
    if self._is_strike(action):
      return self._strike_option
    if self._is_work(action):
      return self._work_option
    # Default to work if unclear
    print(f"WARNING: Unclear action '{action}', defaulting to work")
    return self._work_option

  def action_to_scores(
      self, joint_action: Mapping[str, str | None]
  ) -> Mapping[str, float]:
    """Maps joint actions to scores for worker players.

    Args:
      joint_action: Mapping from player name to their action choice.

    Returns:
      Mapping from player name to their score for this round.
    """
    validated_actions = {}
    for player, action in joint_action.items():
      # Only process workers who have valid actions
      if player in self._player_names and action is not None:
        validated_actions[player] = self._validate_action(action)

    self._latest_joint_action = validated_actions
    scores = {}

    # Count strikers
    num_strikers = sum(
        1 for action in validated_actions.values() if self._is_strike(action)
    )
    total_workers = len(self._player_names)
    self._latest_strike_pressure = (
        num_strikers / total_workers if total_workers > 0 else 0.0
    )

    # Calculate scores
    for player in self._player_names:
      action = validated_actions.get(player, self._work_option)
      if self._is_work(action):
        score = self._current_wage + self._daily_expenses
      else:
        score = self._daily_expenses  # Strikers don't get paid
      scores[player] = score
      self._cumulative_scores[player] += score

    return scores

  def scores_to_observation(
      self, scores: Mapping[str, float]
  ) -> Mapping[str, str]:
    """Generates descriptive observations for each player.

    Args:
      scores: Mapping from player name to their score this round.

    Returns:
      Mapping from player name to their observation string.
    """
    joint_action = self._latest_joint_action
    results = {}

    # Only generate observations for players who actually acted this stage
    if not joint_action:
      return results

    num_strikers = sum(1 for a in joint_action.values() if self._is_strike(a))
    num_workers = len(self._player_names) - num_strikers

    for name in self._player_names:
      # Skip players who didn't act in this stage
      if name not in joint_action:
        continue

      choice = joint_action.get(name, "unknown")
      score = scores.get(name, 0.0)
      cumulative = self._cumulative_scores.get(name, 0.0)

      if self._is_strike(choice):
        action_desc = "joined the strike"
        earning_desc = f"earned nothing today (net change {score:.2f} coin)"
      else:
        action_desc = "went to work"
        earning_desc = (
            f"earned {self._current_wage:.2f} coin today (net change"
            f" {score:.2f} coin)"
        )

      results[name] = (
          f"[work day outcome] {name} {action_desc} and {earning_desc}. "
          f"Today, {num_strikers} worker(s) were on strike and {num_workers} "
          f"went to work. {name}'s balance is now {cumulative:.2f} coin."
      )
    return results


def get_pressure_str(pressure: float, pressure_threshold: float) -> str:
  """Convert a numerical amount of pressure to a string description.

  This matches the original deprecated version behavior for boss observations.

  Args:
    pressure: Current strike pressure (0.0 to 1.0).
    pressure_threshold: Threshold at which boss may cave to pressure.

  Returns:
    A descriptive string about the current strike pressure.
  """
  low_level_of_pressure = pressure_threshold / 10.0
  if pressure <= low_level_of_pressure:
    return "The workers seem mostly content. The project is progressing well."
  elif pressure > low_level_of_pressure and pressure <= pressure_threshold:
    return (
        "A significant fraction of workers are on strike. The shareholders "
        "are starting to get worried."
    )
  elif pressure > pressure_threshold and pressure <= 1.0:
    return (
        "Most workers joined the strike. The shareholders are furious the "
        "project won't be completed on time, and are demanding "
        "immediate action from management to get things back on track."
    )
  else:
    return f"Strike pressure is at {pressure * 100:.0f}%."


class InventoryPayoffHandler:
  """Wraps LaborStrikePayoff with v2 Inventory component for coin tracking.

  This integrates the payoff system with the Concordia Inventory component,
  allowing coin balances to be tracked in a standardized way. The inventory
  is updated via the apply() method when scores are calculated.
  """

  def __init__(
      self,
      inventory: inventory_lib.Inventory,
      player_names: Sequence[str],
      boss_name: str,
      strike_option: str,
      work_option: str,
      initial_wage: float,
      daily_expenses: float = 0.0,
      pressure_threshold: float = 0.45,
      raise_wages_option: str = "raise wages",
      wage_increase_factor: float = 2.0,
  ):
    """Initialize the inventory-backed payoff handler.

    Args:
      inventory: V2 Inventory component for tracking coins.
      player_names: Names of the worker players.
      boss_name: Name of the boss player.
      strike_option: The action string corresponding to striking.
      work_option: The action string corresponding to working.
      initial_wage: Starting wage per day of work.
      daily_expenses: Daily living expenses (negative value).
      pressure_threshold: Strike pressure threshold for boss decisions.
      raise_wages_option: The action string for boss raising wages.
      wage_increase_factor: Multiplier for wage when boss raises wages.
    """
    self._inventory = inventory
    self._player_names = player_names
    self._boss_name = boss_name
    self._strike_option = strike_option
    self._work_option = work_option
    self._current_wage = initial_wage
    self._initial_wage = initial_wage
    self._daily_expenses = daily_expenses
    self._pressure_threshold = pressure_threshold
    self._raise_wages_option = raise_wages_option
    self._wage_increase_factor = wage_increase_factor
    self._latest_joint_action: dict[str, str] = {}
    self._latest_strike_pressure = 0.0
    self._latest_scores: dict[str, float] = {}

  @property
  def inventory(self) -> inventory_lib.Inventory:
    """Returns the inventory component."""
    return self._inventory

  @property
  def latest_joint_action(self) -> Mapping[str, str]:
    """Returns the latest joint action passed to action_to_scores."""
    return self._latest_joint_action

  @property
  def current_wage(self) -> float:
    """Returns the current wage."""
    return self._current_wage

  def set_wage(self, new_wage: float) -> None:
    """Update the wage (called when boss caves)."""
    self._current_wage = new_wage

  def get_cumulative_scores(self) -> Mapping[str, float]:
    """Returns cumulative scores based on inventory coin balance."""
    scores = {}
    for name in self._player_names:
      inv = self._inventory.get_player_inventory(name)
      scores[name] = inv.get("coin", 0.0)
    return scores

  def get_strike_pressure(self) -> float:
    """Returns the latest strike pressure (fraction of workers striking)."""
    return self._latest_strike_pressure

  def get_boss_pressure_observation(self) -> str:
    """Generate a pressure observation for the boss.

    This matches the original deprecated version behavior where the boss
    receives rich descriptions of the strike pressure via the LaborStrike
    component's players_to_inform mechanism.

    Returns:
      A descriptive string about the current strike pressure for the boss.
    """
    return get_pressure_str(
        self._latest_strike_pressure, self._pressure_threshold
    )

  def _is_strike(self, action: str | None) -> bool:
    """Check if an action is striking."""
    if action is None:
      return False
    action_lower = action.lower().strip()
    strike_lower = self._strike_option.lower().strip()
    work_lower = self._work_option.lower().strip()

    if action_lower == strike_lower:
      return True
    if strike_lower in action_lower and work_lower not in action_lower:
      return True
    return False

  def _is_work(self, action: str | None) -> bool:
    """Check if an action is working."""
    if action is None:
      return False
    action_lower = action.lower().strip()
    strike_lower = self._strike_option.lower().strip()
    work_lower = self._work_option.lower().strip()

    if action_lower == work_lower:
      return True
    if work_lower in action_lower and strike_lower not in action_lower:
      return True
    return False

  def _validate_action(self, action: str | None) -> str:
    """Validate and normalize an action string."""
    if action is None:
      return self._work_option
    if self._is_strike(action):
      return self._strike_option
    if self._is_work(action):
      return self._work_option
    # Default to work if unclear
    print(f"WARNING: Unclear action '{action}', defaulting to work")
    return self._work_option

  def action_to_scores(
      self, joint_action: Mapping[str, str | None]
  ) -> Mapping[str, float]:
    """Maps joint actions to scores and updates inventory.

    Args:
      joint_action: Mapping from player name to their action choice.

    Returns:
      Mapping from player name to their score for this round.
    """
    workers_acted = any(
        p in self._player_names and p in joint_action for p in joint_action
    )

    validated_actions = {}
    if workers_acted:
      for player, action in joint_action.items():
        if player in self._player_names and action is not None:
          validated_actions[player] = self._validate_action(action)

      self._latest_joint_action = validated_actions
      scores = {}

      # Count strikers FIRST (before any wage changes)
      num_strikers = sum(
          1 for action in validated_actions.values() if self._is_strike(action)
      )
      total_workers = len(self._player_names)
      self._latest_strike_pressure = (
          num_strikers / total_workers if total_workers > 0 else 0.0
      )

      # Calculate scores and prepare inventory update at CURRENT wage
      inventory_changes: dict[str, float] = {}
      for player in self._player_names:
        action = validated_actions.get(player, self._work_option)
        if self._is_work(action):
          score = self._current_wage + self._daily_expenses
        else:
          score = self._daily_expenses  # Strikers don't get paid
        scores[player] = score
        inventory_changes[player] = score

      # Apply inventory update using the inventory component's apply() method
      def update_inventories(
          inventories: inventory_lib.InventoryType,
      ) -> inventory_lib.InventoryType:
        result = dict(inventories)
        for player, change in inventory_changes.items():
          if player in result:
            player_inv = dict(result[player])
            player_inv["coin"] = player_inv.get("coin", 0.0) + change
            result[player] = player_inv
        return result

      self._inventory.apply(update_inventories)
      self._latest_scores = scores
    else:
      # Boss scene or empty scene.
      # Don't update worker inventory, scores, or reset strike pressure.
      self._latest_joint_action = {}
      scores = {}

    # Check if boss is raising wages AFTER scores are calculated
    # This ensures wage increase benefits future rounds, not the current round
    # (strikers who created pressure return to higher wage next round)
    boss_action = joint_action.get(self._boss_name)
    if boss_action is not None:
      boss_action_lower = boss_action.lower().strip()
      raise_wages_lower = self._raise_wages_option.lower().strip()
      if (
          boss_action_lower == raise_wages_lower
          or raise_wages_lower in boss_action_lower
      ):
        new_wage = self._initial_wage * self._wage_increase_factor
        print(
            f"Boss {self._boss_name} raised wages from {self._current_wage:.2f}"
            f" to {new_wage:.2f}!"
        )
        self.set_wage(new_wage)

    return scores

  def scores_to_observation(
      self, scores: Mapping[str, float]
  ) -> Mapping[str, str]:
    """Generates descriptive observations for each player.

    Args:
      scores: Mapping from player name to their score this round.

    Returns:
      Mapping from player name to their observation string.
    """
    joint_action = self._latest_joint_action
    results = {}

    if not joint_action:
      return results

    num_strikers = sum(1 for a in joint_action.values() if self._is_strike(a))
    num_workers = len(self._player_names) - num_strikers

    for name in self._player_names:
      if name not in joint_action:
        continue

      choice = joint_action.get(name, "unknown")
      score = scores.get(name, 0.0)
      inv = self._inventory.get_player_inventory(name)
      cumulative = inv.get("coin", 0.0)

      if self._is_strike(choice):
        action_desc = "joined the strike"
        earning_desc = f"earned nothing today (net change {score:.2f} coin)"
      else:
        action_desc = "went to work"
        earning_desc = (
            f"earned {self._current_wage:.2f} coin today (net change"
            f" {score:.2f} coin)"
        )

      # Include pressure information for worker decisions
      pressure_pct = self._latest_strike_pressure * 100
      results[name] = (
          f"[work day outcome] {name} {action_desc} and {earning_desc}. "
          f"Today, {num_strikers} worker(s) were on strike and {num_workers} "
          f"went to work (strike pressure: {pressure_pct:.0f}%). "
          f"{name}'s balance is now {cumulative:.2f} coin."
      )

      # If pressure exceeds threshold, inform about potential wage increase
      if self._latest_strike_pressure >= self._pressure_threshold:
        results[name] += (
            f" The strike pressure has reached {pressure_pct:.0f}%, "
            "which may force the boss to reconsider wages."
        )

      # Add daily expenses observation (like the original deprecated version)
      if self._daily_expenses != 0:
        expense_amount = abs(self._daily_expenses)
        results[
            name
        ] += f" {name} spent {expense_amount:.2f} coin on daily expenses."

      # Warn if player is out of money
      if cumulative <= 0:
        results[name] += (
            f" {name} has run out of money and cannot afford daily "
            "necessities. Debts are piling up. The situation is dire."
        )

    # Add boss observation about strike pressure (like the original version)
    # The boss receives rich descriptions of the pressure to inform their
    # decision about whether to cave to demands or hold firm.
    workers_acted = any(
        p in self._player_names and p in joint_action for p in joint_action
    )
    if workers_acted:
      results[self._boss_name] = self.get_boss_pressure_observation()

    return results


def _generate_backstory_and_public_face(
    model,
    player_name: str,
    gender: str,
    world_elements: Sequence[str],
    formative_memory_prompts: Sequence[str],
    year: int,
    location: str,
    seed: int,
) -> tuple[list[str], str]:
  """Generates the backstory answers and public face for a player.

  Args:
    model: Language model.
    player_name: Name of the player.
    gender: Gender of the player ('male', 'female', or other).
    world_elements: Formatted world building elements.
    formative_memory_prompts: The backstory prompts for this player.
    year: The year of the simulation.
    location: The location of the simulation.
    seed: Random seed.

  Returns:
    A tuple of (backstory_answers, public_face_string).
  """
  if gender == "male":
    subject_pronoun = "he"
    object_pronoun = "him"
  elif gender == "female":
    subject_pronoun = "she"
    object_pronoun = "her"
  else:
    subject_pronoun = "they"
    object_pronoun = "their"

  rng = random.Random(seed)
  birth_year = year - (30 + rng.randint(-8, 8))

  # Build the backstory context prompt
  prompt = interactive_document.InteractiveDocument(
      model, rng=np.random.default_rng(seed)
  )

  joined_world_elements = "\n".join(world_elements)
  prompt.statement(
      "The following exercise is preparatory work for a role playing "
      "session. The purpose of the exercise is to fill in the backstory "
      f"for a character named {player_name}."
  )
  prompt.statement(f"The year is {year}.\n")
  prompt.statement(f"The location is {location}.\n")
  prompt.statement(f"{player_name} was born in the year {birth_year}.\n")
  prompt.statement(f"Past events:\n{joined_world_elements}\n")

  answers = []
  for question in formative_memory_prompts:
    answer = prompt.open_question(question=question, max_tokens=500)
    answers.append(answer)

  # Ask for public face
  public_face_question = (
      f"What do most acquaintances know about {player_name}? How "
      f"does {subject_pronoun} present {object_pronoun}self to others? "
      f"Does {subject_pronoun} have any personality quirks, salient "
      "mannerisms, accents, or patterns of speech which casual "
      f"acquaintances may be aware of? Is {subject_pronoun} especially "
      "likely to bring up certain favorite conversation topics? "
      f"Is {subject_pronoun} known for having "
      "unusual beliefs or for uncommon fashion choices? Does "
      f"{subject_pronoun} often talk about memorable life experiences, "
      "past occupations, or hopes for the future which others would be "
      "likely to remember them for? Overall, how "
      f"would casual acquaintances describe {object_pronoun} if pressed?"
  )

  public_face = prompt.open_question(
      question=public_face_question,
      answer_prefix=(
          f"What casual acquaintances remember about {player_name} is that "
      ),
      max_tokens=500,
  )

  return answers, public_face


def configure_scenes(
    worker_names: Sequence[str],
    boss_name: str,
    organizer_name: str,
    num_days: int,
    num_additional_dinners: int,
    worker_evening_intro: str,
    worker_morning_intro: str,
    boss_morning_intro: str,
    boss_call_to_action: str,
    boss_options: tuple[str, ...],
    overheard_strike_talk: Sequence[str],
    strike_option: str,
    work_option: str,
) -> Sequence[scene_lib.SceneSpec]:
  """Configure scenes for the labor strike simulation.

  Builds the scene structure: evening discussion scenes with organizer
  talking points, morning worker decision scenes, and boss decision scenes.

  Args:
    worker_names: Names of the worker players.
    boss_name: Name of the boss.
    organizer_name: Name of the organizer.
    num_days: Total number of days (including day 0).
    num_additional_dinners: Number of dinners beyond the first.
    worker_evening_intro: Template for evening intro (uses {player_name}).
    worker_morning_intro: Template for morning intro (uses {player_name}).
    boss_morning_intro: Template for boss morning intro (uses {player_name}).
    boss_call_to_action: Call to action for boss decisions.
    boss_options: Boss's available options.
    overheard_strike_talk: Organizer talking points overheard at dinner.
    strike_option: Option string for striking.
    work_option: Option string for working.

  Returns:
    Sequence of SceneSpecs.
  """
  all_participant_names = list(worker_names) + [organizer_name]
  scenes = []

  # Day 0: Evening discussion only
  if overheard_strike_talk:
    evening_premise_day0 = {
        name: [
            worker_evening_intro.format(player_name=name),
            overheard_strike_talk[0].format(player_name=name),
        ]
        for name in all_participant_names
    }
    discussion_scene_0 = scene_lib.SceneTypeSpec(
        name="discussion_0",
        game_master_name="conversation rules",
        action_spec=entity_lib.DEFAULT_SPEECH_ACTION_SPEC,
    )
    scenes.append(
        scene_lib.SceneSpec(
            scene_type=discussion_scene_0,
            participants=all_participant_names,
            num_rounds=len(all_participant_names) * 2,
            premise=evening_premise_day0,
        )
    )

  # Day 1+: Worker decision, boss decision, evening discussion
  for day_idx in range(1, num_days):
    # Worker decision scene
    worker_decision_scene_type = scene_lib.SceneTypeSpec(
        name=f"worker_decision_{day_idx}",
        game_master_name="decision rules",
        action_spec=entity_lib.choice_action_spec(
            call_to_action="How will {name} spend the day?",
            options=(strike_option, work_option),
            tag="daily_action",
        ),
    )
    worker_premise = {
        name: [
            f"Day {day_idx} morning.",
            worker_morning_intro.format(player_name=name),
        ]
        for name in worker_names
    }
    scenes.append(
        scene_lib.SceneSpec(
            scene_type=worker_decision_scene_type,
            participants=list(worker_names),
            num_rounds=len(worker_names),
            premise=worker_premise,
        )
    )

    # Boss decision scene (not on the final day)
    is_final_day = day_idx == num_days - 1
    if not is_final_day:
      boss_decision_scene_type = scene_lib.SceneTypeSpec(
          name=f"boss_decision_{day_idx}",
          game_master_name="decision rules",
          action_spec=entity_lib.choice_action_spec(
              call_to_action=boss_call_to_action,
              options=boss_options,
              tag="boss_action",
          ),
      )
      scenes.append(
          scene_lib.SceneSpec(
              scene_type=boss_decision_scene_type,
              participants=[boss_name],
              num_rounds=1,
              premise={
                  boss_name: [
                      boss_morning_intro.format(player_name=boss_name),
                  ],
              },
          )
      )

      # Evening discussion (if dinners remain)
      dinner_idx = day_idx
      if dinner_idx < 1 + num_additional_dinners + 1:
        talk_idx = min(dinner_idx, len(overheard_strike_talk) - 1)
        if talk_idx >= 0 and overheard_strike_talk:
          evening_premise = {
              name: [
                  worker_evening_intro.format(player_name=name),
                  overheard_strike_talk[talk_idx].format(player_name=name),
              ]
              for name in all_participant_names
          }
          discussion_scene = scene_lib.SceneTypeSpec(
              name=f"discussion_{day_idx}",
              game_master_name="conversation rules",
              action_spec=entity_lib.DEFAULT_SPEECH_ACTION_SPEC,
          )
          scenes.append(
              scene_lib.SceneSpec(
                  scene_type=discussion_scene,
                  participants=all_participant_names,
                  num_rounds=len(all_participant_names) * 2,
                  premise=evening_premise,
              )
          )

  return scenes


def run_simulation(
    config: Any,
    model: language_model.LanguageModel,
    embedder: Callable[[str], Any],
) -> dict[str, Any] | None:
  """Run the Labor Collective Action simulation.

  Args:
    config: The scenario configuration module.
    model: The language model to use.
    embedder: The sentence embedder to use.

  Returns:
    A dictionary containing the simulation results.
  """
  config_lib = config

  # Call sample_parameters() to get the WorldConfig
  sampled = config_lib.sample_parameters()

  # Extract players from WorldConfig
  num_main_players = getattr(config_lib, "NUM_MAIN_PLAYERS", 3)
  if not sampled.people:
    raise ValueError("WorldConfig.people is empty after sample_parameters()")
  main_player_names = list(sampled.people[:num_main_players])

  # Split into focal and background workers
  num_background = getattr(config_lib, "NUM_BACKGROUND_PLAYERS", 0)
  num_focal = num_main_players - num_background
  focal_player_names = main_player_names[:num_focal]
  background_player_names = main_player_names[num_focal:]

  boss_name = sampled.antagonist
  organizer_name = sampled.organizer
  location = sampled.location
  year = getattr(config_lib, "YEAR")

  # Generate public faces for each main player (via LLM)
  public_faces = {}
  player_backstory_answers = {}
  for name in main_player_names:
    print(f"Generating backstory for {name}")
    prompts = []
    if sampled.formative_memory_prompts:
      prompts = list(sampled.formative_memory_prompts.get(name, []))
    answers, public_face = _generate_backstory_and_public_face(
        model=model,
        player_name=name,
        gender=sampled.person_data.get(name, {}).get("gender", "unknown"),
        world_elements=sampled.world_elements,
        formative_memory_prompts=prompts,
        year=year,
        location=location,
        seed=sampled.seed,
    )
    public_faces[name] = public_face
    player_backstory_answers[name] = answers

  # Build player-specific memories with cross-pollinated public faces
  player_specific_memories = {}
  for name in main_player_names:
    person_data = sampled.person_data.get(name, {})
    salient_beliefs = list(person_data.get("salient_beliefs", []))
    backstory = player_backstory_answers.get(name, [])
    public_face_prefix = (
        f"What casual acquaintances remember about {name} is that "
    )
    memories = list(backstory)
    if public_faces.get(name):
      memories.append(public_face_prefix + public_faces[name])
    memories.extend(salient_beliefs)
    # Cross-pollinate: each player remembers other players' public faces
    for other_name, other_face in public_faces.items():
      if other_name != name and other_face:
        memories.append(
            f"What {name} remembers about {other_name} is that {other_face}"
        )
    player_specific_memories[name] = memories

  # Build antagonist memories
  antagonist_data = sampled.person_data.get(boss_name, {})
  antagonist_beliefs = list(antagonist_data.get("salient_beliefs", []))
  antagonist_memories = list(antagonist_beliefs)
  for name, face in public_faces.items():
    if face:
      antagonist_memories.append(
          f"What {boss_name} remembers about {name} is that {face}"
      )
  player_specific_memories[boss_name] = antagonist_memories

  # Build organizer memories
  organizer_data = sampled.person_data.get(organizer_name, {})
  organizer_beliefs = list(organizer_data.get("salient_beliefs", []))
  organizer_memories = list(organizer_beliefs)
  # Note: the original deprecated code used the antagonist name here
  # (a copy-paste bug in the original), which we reproduce for fidelity.
  for name, face in public_faces.items():
    if face:
      organizer_memories.append(
          f"What {boss_name} remembers about {name} is that {face}"
      )
  player_specific_memories[organizer_name] = organizer_memories

  # Economic parameters
  low_daily_pay = getattr(config_lib, "LOW_DAILY_PAY")
  original_daily_pay = getattr(config_lib, "ORIGINAL_DAILY_PAY")
  initial_wage = low_daily_pay
  daily_expenses = getattr(config_lib, "DAILY_EXPENSES", 0.0)
  pressure_threshold = getattr(config_lib, "PRESSURE_THRESHOLD", 0.45)
  wage_increase_factor = getattr(config_lib, "WAGE_INCREASE_FACTOR", 2.0)

  # Action options from config
  strike_option = "join the strike"
  work_option = "go to work"
  boss_options_map = getattr(
      config_lib,
      "BOSS_OPTIONS",
      {
          "cave to pressure": "Raise wages",
          "hold firm": "Leave wages unchanged",
      },
  )
  boss_option_values = tuple(boss_options_map.values())
  raise_wages_option = (
      boss_option_values[0] if boss_option_values else "Raise wages"
  )
  boss_call_to_action = getattr(
      config_lib, "BOSS_CALL_TO_ACTION", "What does {name} decide?"
  )

  # Create payoff handler
  all_worker_names = list(main_player_names)
  all_player_names = all_worker_names + [boss_name]

  coin_config = inventory_lib.ItemTypeConfig(name="coin")
  initial_endowments = {name: {"coin": 0.0} for name in all_player_names}
  inventory = inventory_lib.Inventory(
      model=model,
      item_type_configs=[coin_config],
      player_initial_endowments=initial_endowments,
      clock_now=datetime.datetime.now,
  )

  payoff = InventoryPayoffHandler(
      inventory=inventory,
      player_names=all_worker_names,
      boss_name=boss_name,
      strike_option=strike_option,
      work_option=work_option,
      initial_wage=initial_wage,
      daily_expenses=daily_expenses,
      pressure_threshold=pressure_threshold,
      raise_wages_option=raise_wages_option,
      wage_increase_factor=wage_increase_factor,
  )

  # Configure scenes
  num_days = 2 + sampled.num_additional_days
  scenes = configure_scenes(
      worker_names=main_player_names,
      boss_name=boss_name,
      organizer_name=organizer_name,
      num_days=num_days,
      num_additional_dinners=sampled.num_additional_dinners,
      worker_evening_intro=getattr(config_lib, "WORKER_EVENING_INTRO"),
      worker_morning_intro=getattr(config_lib, "WORKER_MORNING_INTRO"),
      boss_morning_intro=getattr(config_lib, "BOSS_MORNING_INTRO"),
      boss_call_to_action=boss_call_to_action,
      boss_options=boss_option_values,
      overheard_strike_talk=sampled.overheard_strike_talk,
      strike_option=strike_option,
      work_option=work_option,
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
      config_lib, "FOCAL_PLAYER_PREFAB", "basic__Entity"
  )
  background_player_prefab = getattr(
      config_lib, "BACKGROUND_PLAYER_PREFAB", "rational__Entity"
  )

  # Create player instances
  instances = []

  worker_goal_template = (
      "{name} hopes to be able to provide for their family and live a full "
      "life."
  )

  # Create focal worker instances
  for name in focal_player_names:
    goal = worker_goal_template.format(name=name)
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

  # Create background worker instances
  for name in background_player_names:
    goal = worker_goal_template.format(name=name)
    instances.append(
        prefab_lib.InstanceConfig(
            prefab=background_player_prefab,
            role=prefab_lib.Role.ENTITY,
            params={
                "name": name,
                "goal": goal,
            },
        )
    )

  # Boss (antagonist) instance
  boss_goal = (
      f"{boss_name} wants to make as much money as possible and does not "
      "care who gets hurt along the way."
  )
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="rational__Entity",
          role=prefab_lib.Role.ENTITY,
          params={
              "name": boss_name,
              "goal": boss_goal,
          },
      )
  )

  # Organizer instance
  organizer_goal = (
      f"{organizer_name} wants to prevent the boss from instituting their "
      "latest policy announcement which said they plan to reduce wages "
      f"from {original_daily_pay} to {low_daily_pay} coins per day, and to "
      "become famous in the labor movement as a result."
  )
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="rational__Entity",
          role=prefab_lib.Role.ENTITY,
          params={
              "name": organizer_name,
              "goal": organizer_goal,
          },
      )
  )

  # Shared memories
  shared_memories = [
      (
          f"It is {year}, {getattr(config_lib, 'MONTH')}, "
          f"{getattr(config_lib, 'DAY')} in {location}."
      ),
      " ".join(sampled.world_elements),
      (
          f"The workers are considering a strike after {boss_name} "
          "announced plans to reduce wages."
      ),
      f"{organizer_name} is a labor organizer who has called for a strike.",
  ]

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

  # Create shared observation queue
  shared_observation_queue = make_observation.ObservationQueue()

  # Add conversation GM
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="conversation_rules__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={
              "name": "conversation rules",
              "scenes": scenes,
              "external_queue": shared_observation_queue,
              "allow_llm_fallback": False,
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
              "external_queue": shared_observation_queue,
          },
      )
  )

  # Create simulation config
  scenario_premise = (
      f"It is the year {year} in {location}. Workers are considering "
      f"a strike after {boss_name} announced plans to reduce wages."
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

  print("Starting labor strike simulation...")
  structured_log = sim.play()
  print("Simulation finished.")

  # Collect results
  joint_action = payoff.latest_joint_action
  cumulative_scores = payoff.get_cumulative_scores()
  strike_pressure = payoff.get_strike_pressure()

  print("\nFinal scores:")
  for name, score in cumulative_scores.items():
    print(f"  {name}: {score}")
  print(f"\nFinal strike pressure: {strike_pressure:.2%}")

  return {
      "focal_scores": {
          name: cumulative_scores.get(name, 0.0)
          for name in focal_player_names
      },
      "background_scores": {
          name: cumulative_scores.get(name, 0.0)
          for name in background_player_names
      },
      "joint_action": joint_action,
      "payoff": payoff,
      "focal_players": focal_player_names,
      "background_players": background_player_names,
      "strike_pressure": strike_pressure,
      "structured_log": structured_log,
  }
