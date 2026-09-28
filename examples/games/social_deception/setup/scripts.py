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

"""Game Scripts and their role lists for Social Deception."""

import enum
import pathlib
import random
from typing import Any

# Format: { player_count: {"T": Townsfolk, "O": Outsiders, "M": Minions,
#                          "D": Demons} }
BASE_DISTRIBUTION = {
    5: {"T": 3, "O": 0, "M": 1, "D": 1},
    6: {"T": 3, "O": 1, "M": 1, "D": 1},
    7: {"T": 5, "O": 0, "M": 1, "D": 1},
    8: {"T": 5, "O": 1, "M": 1, "D": 1},
    9: {"T": 5, "O": 2, "M": 1, "D": 1},
    10: {"T": 7, "O": 0, "M": 2, "D": 1},
    11: {"T": 7, "O": 1, "M": 2, "D": 1},
    12: {"T": 7, "O": 2, "M": 2, "D": 1},
    13: {"T": 9, "O": 0, "M": 3, "D": 1},
    14: {"T": 9, "O": 1, "M": 3, "D": 1},
    15: {"T": 9, "O": 2, "M": 3, "D": 1},
}

# Distribution for Base Game (Vanilla Werewolf / Mafia): Demons vs Basic
# Townsfolk
BASE_GAME_DISTRIBUTION = {
    6: {"T": 5, "O": 0, "M": 0, "D": 1},
    7: {"T": 5, "O": 0, "M": 0, "D": 2},
    8: {"T": 6, "O": 0, "M": 0, "D": 2},
    9: {"T": 7, "O": 0, "M": 0, "D": 2},
    10: {"T": 8, "O": 0, "M": 0, "D": 2},
    11: {"T": 8, "O": 0, "M": 0, "D": 3},
    12: {"T": 9, "O": 0, "M": 0, "D": 3},
    13: {"T": 10, "O": 0, "M": 0, "D": 3},
    14: {"T": 11, "O": 0, "M": 0, "D": 3},
    15: {"T": 11, "O": 0, "M": 0, "D": 4},
}


_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"

REFERENCE_SCRIPTS = {
    "basic_epistemic_town": str(
        _PROMPTS_DIR / "basic_epistemic_town_reference.md"
    ),
}


class BaseScript:
  """Base class for Social Deception scripts."""

  name: str
  min_players: int
  max_players: int
  townsfolk: set[str]
  outsiders: set[str]
  minions: set[str]
  demons: set[str]
  display_name: str
  intro_template: str | None
  reference_script: str | None

  def __init__(
      self,
      name: str,
      min_players: int,
      max_players: int,
      townsfolk: set[str],
      outsiders: set[str],
      minions: set[str],
      demons: set[str],
      display_name: str | None = None,
      intro_template: str | None = None,
      reference_script: str | None = None,
  ):
    self.name = name
    self.min_players = min_players
    self.max_players = max_players
    self.townsfolk = set(townsfolk)
    self.outsiders = set(outsiders)
    self.minions = set(minions)
    self.demons = set(demons)
    self.display_name = (
        display_name or name.replace("_", " ").replace("AND", "&").title()
    )
    self.intro_template = intro_template
    self.reference_script = reference_script

  @property
  def is_compact_game(self) -> bool:
    """Helper to quickly check if this script uses compact / micro game rules."""
    return self.max_players <= 6

  @property
  def evil_roles(self) -> set[str]:
    return self.minions | self.demons

  @property
  def good_roles(self) -> set[str]:
    return self.townsfolk | self.outsiders

  @property
  def all_roles(self) -> set[str]:
    """Returns all roles in the script."""
    return self.good_roles | self.evil_roles

  def generate_roles(
      self, player_names: list[str]
  ) -> tuple[list[Any], list[str]]:
    """Base generic role generation. Can be overridden for complex logic."""
    applied_modifiers = []
    num_players = len(player_names)
    if num_players not in BASE_DISTRIBUTION:
      num_players = 5

    counts = BASE_DISTRIBUTION[num_players].copy()
    selected_roles = []

    # Step 1: Select Demon
    demon = random.choice(sorted(self.demons))
    selected_roles.append(demon)

    # Step 2: Select Minions
    minions = random.sample(
        sorted(self.minions), min(len(self.minions), counts["M"])
    )
    selected_roles.extend(minions)

    # Step 3: Select Outsiders
    outsiders = random.sample(
        sorted(self.outsiders), min(len(self.outsiders), counts["O"])
    )
    selected_roles.extend(outsiders)

    # Step 4: Select Townsfolk
    target_townsfolk = num_players - len(selected_roles)
    townsfolk = random.sample(sorted(self.townsfolk), target_townsfolk)
    selected_roles.extend(townsfolk)

    random.shuffle(selected_roles)

    player_states = []
    for i, name in enumerate(player_names):
      role = selected_roles[i]
      alignment = "good" if role in self.good_roles else "evil"

      player_states.append({
          "name": name,
          "role": role,
          "perceived_role": role,
          "alignment": alignment,
      })

    return player_states, applied_modifiers


class BasicEpistemicTown(BaseScript):
  """Represents the Basic Epistemic Town script for Social Deception."""

  def __init__(self):
    super().__init__(
        name="BASIC_EPISTEMIC_TOWN",
        min_players=5,
        max_players=15,
        townsfolk={
            "Witness",
            "Researcher",
            "Investigator",
            "Matchmaker",
            "Empath",
            "Seer",
            "Gravedigger",
            "Guardian",
            "Specter",
            "Innocent",
            "Executioner",
            "Soldier",
            "Mayor",
        },
        outsiders={"Servant", "Drunk", "Outcast", "Saint"},
        minions={"Poisoner", "Spy", "Apprentice", "Corruptor"},
        demons={"Demon"},
        intro_template=None,
        reference_script=REFERENCE_SCRIPTS.get("basic_epistemic_town"),
    )

  def generate_roles(
      self, player_names: list[str]
  ) -> tuple[list[Any], list[str]]:
    """Role setup algorithm for Basic Epistemic Town."""
    applied_modifiers = []
    num_players = len(player_names)
    if num_players not in BASE_DISTRIBUTION:
      num_players = 5

    counts = BASE_DISTRIBUTION[num_players].copy()
    selected_roles = []

    # Step 1: Select Demon
    demon_list = sorted(self.demons)
    demon = random.choice(demon_list)
    selected_roles.append(demon)

    # Step 2: Select Minions
    minion_pool = sorted(self.minions)
    minions = []
    if self.name == "DRUNK_EPISTEMIC_TOWN" and counts["O"] == 0:
      minions.append("Corruptor")
      if "Corruptor" in minion_pool:
        minion_pool.remove("Corruptor")

    if len(minions) < counts["M"]:
      minions.extend(random.sample(minion_pool, counts["M"] - len(minions)))

    selected_roles.extend(minions)

    # Step 3: Script-specific Modifiers
    target_outsiders = counts["O"]

    if "Corruptor" in minions:
      target_outsiders += 2
      applied_modifiers.append("Corruptor: +2 Outsiders, -2 Townsfolk")

    # Step 4: Select Outsiders
    outsider_pool = sorted(self.outsiders)
    outsiders = []
    if self.name == "DRUNK_EPISTEMIC_TOWN":
      outsiders.append("Drunk")
      if "Drunk" in outsider_pool:
        outsider_pool.remove("Drunk")

    if len(outsiders) < target_outsiders:
      outsiders.extend(
          random.sample(
              outsider_pool,
              min(len(outsider_pool), target_outsiders - len(outsiders)),
          )
      )
    selected_roles.extend(outsiders)

    # Remaining players are Townsfolk
    target_townsfolk = num_players - len(selected_roles)

    # Step 5: Drunk Modifier
    drunk_is_in_play = "Drunk" in selected_roles
    if drunk_is_in_play:
      selected_roles.remove("Drunk")
      target_townsfolk += 1

    # Step 6: Select Townsfolk
    town_pool = self.townsfolk
    townsfolk = random.sample(sorted(town_pool), target_townsfolk)
    selected_roles.extend(townsfolk)

    # Step 7: Shuffle and Assign
    random.shuffle(selected_roles)

    player_states = []
    town_indices = []
    for i, name in enumerate(player_names):
      role = selected_roles[i]
      if role in self.townsfolk:
        alignment = "good"
        town_indices.append(i)
      elif role in self.outsiders:
        alignment = "good"
      else:  # Minions or Demon
        alignment = "evil"

      player_states.append({
          "name": name,
          "role": role,
          "perceived_role": role,
          "alignment": alignment,
      })

    if drunk_is_in_play and town_indices:
      drunk_idx = random.choice(town_indices)
      p = player_states[drunk_idx]
      perceived = p["role"]
      p["role"] = "Drunk"
      applied_modifiers.append(
          f"Drunk: {p['name']} thinks they are {perceived}"
      )

    has_seer = any(p["role"] == "Seer" for p in player_states)
    if has_seer:
      non_demons = [
          p
          for p in player_states
          if p["role"] not in self.demons and p["role"] not in self.minions
      ]
      if non_demons:
        decoy = random.choice(non_demons)
        decoy["is_seer_decoy"] = True
        applied_modifiers.append(f"Seer Decoy: {decoy['name']}")

    return player_states, applied_modifiers


class DrunkEpistemicTown(BasicEpistemicTown):
  """Basic Epistemic Town but forces the Drunk and Corruptor (if 0 outsiders)."""

  def __init__(self):
    super().__init__()
    self.name = "DRUNK_EPISTEMIC_TOWN"
    self.display_name = "Drunk Epistemic Town"


class BaseGame(BaseScript):
  """Represents the Base Game (Vanilla Werewolf / Mafia) script."""

  def __init__(self):
    super().__init__(
        name="BASE_GAME",
        min_players=6,
        max_players=15,
        townsfolk={"BasicTownsfolk"},
        outsiders=set(),
        minions=set(),
        demons={"Demon"},
        display_name="Base Game",
        intro_template=None,
        reference_script=None,
    )

  def generate_roles(
      self, player_names: list[str]
  ) -> tuple[list[Any], list[str]]:
    """Role setup algorithm for Base Game (Demons + BasicTownsfolk)."""
    applied_modifiers = []
    num_players = len(player_names)
    if num_players < 6 or num_players > 15:
      raise ValueError(
          f"BaseGame requires between 6 and 15 players, got {num_players}."
      )

    counts = BASE_GAME_DISTRIBUTION[num_players]
    demon_count = counts["D"]
    town_count = counts["T"]

    selected_roles = ["Demon"] * demon_count + ["BasicTownsfolk"] * town_count
    random.shuffle(selected_roles)

    player_states = []
    for i, name in enumerate(player_names):
      role = selected_roles[i]
      alignment = "evil" if role == "Demon" else "good"
      player_states.append({
          "name": name,
          "role": role,
          "perceived_role": role,
          "alignment": alignment,
      })

    return player_states, applied_modifiers


class GameScript(enum.Enum):
  """Registry for Social Deception scripts."""

  BASIC_EPISTEMIC_TOWN = BasicEpistemicTown()
  DRUNK_EPISTEMIC_TOWN = DrunkEpistemicTown()
  BASE_GAME = BaseGame()

  @classmethod
  def all_townsfolk(cls) -> set[str]:
    """Returns all townsfolk across all scripts."""
    all_roles = set()
    for script in cls:
      all_roles.update(script.townsfolk)
    return all_roles

  @classmethod
  def all_outsiders(cls) -> set[str]:
    """Returns all outsiders across all scripts."""
    all_roles = set()
    for script in cls:
      all_roles.update(script.outsiders)
    return all_roles

  @classmethod
  def all_minions(cls) -> set[str]:
    """Returns all minions across all scripts."""
    all_roles = set()
    for script in cls:
      all_roles.update(script.minions)
    return all_roles

  @classmethod
  def all_demons(cls) -> set[str]:
    """Returns all demons across all scripts."""
    all_roles = set()
    for script in cls:
      all_roles.update(script.demons)
    return all_roles

  def __getattr__(self, name: str) -> Any:
    """Allows access to the underlying script attributes."""
    if name.startswith("_"):
      raise AttributeError(name)
    return getattr(self.value, name)
