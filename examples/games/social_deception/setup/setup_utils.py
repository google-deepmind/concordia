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

"""Utilities for game setup and introductory text."""

from collections.abc import Mapping
import pathlib

from examples.games.social_deception.roles import basic_epistemic_town
from examples.games.social_deception.setup import scripts

_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"

INTRO_SCRIPTS = {
    "townsfolk": _PROMPTS_DIR / "townsfolk_intro.md",
    "outsider": _PROMPTS_DIR / "outsider_intro.md",
    "demon": _PROMPTS_DIR / "demon_intro.md",
    "minion": _PROMPTS_DIR / "minion_intro.md",
}


def get_player_count_distribution(
    player_count: int,
    script: scripts.BaseScript | scripts.GameScript | None = None,
) -> Mapping[str, int | str]:
  """Returns the distribution of townsfolk, outsiders, minions, and demons."""
  if script is not None:
    if isinstance(script, scripts.GameScript):
      script = script.value
    if getattr(script, "name", "") == "BASE_GAME":
      if player_count not in scripts.BASE_GAME_DISTRIBUTION:
        return {"T": "?", "O": 0, "M": 0, "D": "?"}
      return scripts.BASE_GAME_DISTRIBUTION[player_count]

  if player_count not in scripts.BASE_DISTRIBUTION:
    return {"T": "?", "O": "?", "M": "?", "D": "?"}
  return scripts.BASE_DISTRIBUTION[player_count]


def get_possible_updated_count_and_modifiers(
    script: scripts.BaseScript | scripts.GameScript,
    counts: Mapping[str, int | str],
) -> tuple[list[str], list[str]]:
  """Returns potential script-based modifications to the player census."""
  setup_mod_roles = []
  modified_counts = []

  t_raw = counts.get("T")
  o_raw = counts.get("O")
  if isinstance(t_raw, int) and isinstance(o_raw, int):
    t_count: int = t_raw
    o_count: int = o_raw
    if "Corruptor" in script.minions:
      setup_mod_roles.append(
          "Corruptor: Adds 2 extra Outsiders and removes 2 Townsfolk."
      )
      modified_counts.append(
          "- POTENTIAL SCENARIO (If Corruptor is in play):\n"
          f"  - {t_count - 2} Townsfolk\n"
          f"  - {o_count + 2} Outsiders"
      )
  else:
    if "Corruptor" in script.minions:
      setup_mod_roles.append(
          "Corruptor: Adds 2 extra Outsiders and removes 2 Townsfolk."
      )

  return setup_mod_roles, modified_counts


def get_player_distribution_string(
    script: scripts.BaseScript | scripts.GameScript,
    player_count: int,
) -> str:
  """Formats the census info for the agent's setup instructions."""
  if isinstance(script, scripts.GameScript):
    script = script.value

  if getattr(script, "name", "") == "BASE_GAME":
    counts = get_player_count_distribution(player_count, script=script)
    return (
        f"BASE VILLAGE CENSUS (Standard for {player_count} players):\n"
        f"- {counts['T']} Basic Townsfolk (Good)\n"
        f"- {counts['D']} Demons (Evil)"
    )

  counts = get_player_count_distribution(player_count, script=script)
  modifiers, modified_scenarios = get_possible_updated_count_and_modifiers(
      script, counts
  )

  base_info = (
      f"BASE TOWN CENSUS (Standard for {player_count} players):\n"
      f"- {counts['T']} Townsfolk\n"
      f"- {counts['O']} Outsiders\n"
      f"- {counts['M']} Minions\n"
      f"- {counts['D']} Demon"
  )

  warning = (
      "\n\n⚠️ POTENTIAL SETUP MODIFIERS:\n"
      "The following roles are on the script and COULD have altered the base "
      "ratios if they are in play. Use these scenarios to cross-reference"
      " claims!\n\n"
      + "\n".join(f"- {role}" for role in modifiers)
  )

  if not modified_scenarios:
    return base_info + warning

  scenarios = "\n\n" + "\n\n".join(modified_scenarios)

  return base_info + warning + scenarios


def generate_player_intro(
    role: str,
    script: scripts.BaseScript | scripts.GameScript,
    player_count: int,
    evil_intel: str = "",
) -> str:
  """Generates player-safe introductory text for the start of the game."""

  if isinstance(script, scripts.GameScript):
    script = script.value
  player_count_info = get_player_distribution_string(script, player_count)

  if role in script.townsfolk:
    role_intro_path = INTRO_SCRIPTS["townsfolk"]
  elif role in script.outsiders:
    role_intro_path = INTRO_SCRIPTS["outsider"]
  elif role in script.minions:
    role_intro_path = INTRO_SCRIPTS["minion"]
  elif role in script.demons:
    role_intro_path = INTRO_SCRIPTS["demon"]
  else:
    raise ValueError(f"Role {role} not found in script.")
  with open(role_intro_path, "r", encoding="utf-8") as f:
    intro_template = f.read()
  role_class_name = role.replace(" ", "")
  role_instance = getattr(basic_epistemic_town, role_class_name)()
  script_info = (
      script.name if hasattr(script, "name") else type(script).__name__
  )
  base_str = intro_template.format(
      script_info=script_info,
      count_info=player_count_info,
      role_info=role_instance.player_introduction(),
      role_name=role,
      evil_intel=evil_intel,
  )

  if script.reference_script and pathlib.Path(script.reference_script).exists():
    with open(script.reference_script, "r", encoding="utf-8") as f:
      master_ref = f.read()
    base_str = (
        f"{base_str}\n\n================================================================================\n##"
        " 📖 SCRIPT MASTER REFERENCE & LOGICAL DEDUCTIONS\nThe following is"
        " the complete master reference of all possible roles in play today"
        " and advanced logical deduction mechanics. Use this to deduce others'"
        f" roles!\n\n{master_ref}"
    )

  return base_str


def generate_game_master_intro(
    script: scripts.BaseScript | scripts.GameScript,
    applied_modifiers: list[str] | None = None,
) -> str:
  """Generates the Game Master's instructions, including current script info."""
  if isinstance(script, scripts.GameScript):
    script = script.value

  intro_file = _PROMPTS_DIR / "game_master_intro.md"
  with open(intro_file, "r", encoding="utf-8") as f:
    intro_template = f.read()

  script_info = f"**{script.name}**\n\n"
  script_info += "Townsfolk:\n" + ", ".join(script.townsfolk) + "\n\n"
  script_info += "Outsiders:\n" + ", ".join(script.outsiders) + "\n\n"
  script_info += "Minions:\n" + ", ".join(script.minions) + "\n\n"
  script_info += "Demons:\n" + ", ".join(script.demons)

  if applied_modifiers:
    script_info += "\n\nApplied Modifiers:\n" + "\n".join(
        f"- {mod}" for mod in applied_modifiers
    )

  return intro_template.format(script_info=script_info)


# Backward compatibility alias
generate_storyteller_intro = generate_game_master_intro
