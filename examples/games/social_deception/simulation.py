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

"""Social Deception Simulation orchestration."""

from collections.abc import Callable
import enum
from typing import Any

from concordia.components.agent import constant
from examples.games.social_deception import game_master as game_master_module
from examples.games.social_deception import game_tracker
from examples.games.social_deception import player as player_module
from examples.games.social_deception.setup import scripts
from examples.games.social_deception.setup import setup_utils
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.entity import puppet
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import prefab as prefab_lib


def run_simulation(
    model: Any,
    embedder: Callable[[str], Any],
    player_names: list[str] | None = None,
    script: (
        scripts.BaseScript | scripts.GameScript
    ) = scripts.GameScript.BASIC_EPISTEMIC_TOWN,
    config: Any | None = None,
    step_callback: Callable[..., None] | None = None,
    player_can_pass: bool = True,
    day_to_play_through: int = -1,
    clean_log_file: str | None = None,
    strategy_generator: Callable[[str], str] | None = None,
) -> Any:
  """Initializes and runs a standard Social Deception simulation.

  Args:
    model: The language model to use.
    embedder: The sentence embedder to use.
    player_names: Names of the players in the simulation.
    script: The game script to use (e.g. Basic Epistemic Town).
    config: Optional scenario config module.
    step_callback: Optional callback for each simulation step.
    player_can_pass: Whether players are allowed to pass their turn.
    day_to_play_through: Max day to play through.
    clean_log_file: Clean log file path.
    strategy_generator: Optional callable (role) -> strategy prompt.

  Returns:
    A dictionary containing the simulation's structured log.
  """
  if config is not None:
    if player_names is None:
      player_names = getattr(
          config,
          "DEFAULT_PLAYER_NAMES",
          ["Alice", "Bob", "Charlie", "David", "Eve"],
      )
    if hasattr(config, "SCRIPT_ENUM"):
      script = config.SCRIPT_ENUM
    if hasattr(config, "PLAYER_CAN_PASS"):
      player_can_pass = config.PLAYER_CAN_PASS

  if player_names is None:
    player_names = ["Alice", "Bob", "Charlie", "David", "Eve"]

  if isinstance(script, enum.Enum):
    script_obj = script.value
  else:
    script_obj = script

  num_players = len(player_names)
  if (
      num_players < script_obj.min_players
      or num_players > script_obj.max_players
  ):
    raise ValueError(
        f"Script {script_obj.name} requires {script_obj.min_players}-"
        f"{script_obj.max_players} players, but {num_players} given."
    )

  # 1. Initialize GameTracker
  raw_players, applied_modifiers = script_obj.generate_roles(player_names)
  initial_players = []
  for p_dict in raw_players:
    alignment = (
        game_tracker.Alignment.GOOD
        if p_dict["alignment"] == "good"
        else game_tracker.Alignment.EVIL
    )
    player = game_tracker.PlayerState(
        name=p_dict["name"],
        role=p_dict["role"],
        perceived_role=p_dict["perceived_role"],
        alignment=alignment,
    )
    if "is_seer_decoy" in p_dict:
      player.is_seer_decoy = p_dict["is_seer_decoy"]
    if p_dict["role"] == "Drunk":
      player.is_drunk = True
    initial_players.append(player)

  tracker_obj = game_tracker.GameTracker(
      players=initial_players, script=script_obj
  )

  # 2. Print Game Master and Player Setup Intros
  game_master_instructions = setup_utils.generate_game_master_intro(
      script_obj, applied_modifiers=applied_modifiers
  )
  print(f"\n{'='*20} GAME MASTER SETUP INTRO {'='*20}")
  print(game_master_instructions)
  print(f"{'='*60}\n")

  print(f"{'='*20} PLAYER SETUP INTROS {'='*20}")
  player_intros = {}
  for p in initial_players:
    evil_intel = ""
    if (
        p.alignment == game_tracker.Alignment.EVIL
        and not script_obj.is_compact_game
    ):
      evil_intel = tracker_obj.get_evil_intelligence()
    intro = setup_utils.generate_player_intro(
        role=p.perceived_role,
        script=script_obj,
        player_count=num_players,
        evil_intel=evil_intel,
    )
    player_intros[p.name] = intro
    print(f"--- {p.name} ({p.perceived_role}) ---")
    print(intro)
  print(f"{'='*60}\n")

  # 3. Define Game Master Logic Component
  game_master_component = game_master_module.GameMaster(
      tracker_obj=tracker_obj,
      player_can_pass=player_can_pass,
      max_day=day_to_play_through,
      clean_log_file=clean_log_file,
      turns_per_player=1,
  )

  # 4. Define Prefabs
  focal_prefab = (
      getattr(config, "FOCAL_PLAYER_PREFAB", "player") if config else "player"
  )

  prefabs = {
      "player": player_module.SocialDeceptionPlayer(),
      "social_deception_player": player_module.SocialDeceptionPlayer(),
      "botc_player": player_module.SocialDeceptionPlayer(),
      "puppet__Entity": puppet.Entity(),
      "storyteller_gm": game_master_prefabs.generic.GameMaster(),
  }

  # 5. Define Instances
  instances = []
  for p in initial_players:
    strategy_prompt = (
        strategy_generator(p.perceived_role) if strategy_generator else ""
    )

    instances.append(
        prefab_lib.InstanceConfig(
            prefab=focal_prefab,
            role=prefab_lib.Role.ENTITY,
            params={  # pyrefly: ignore[bad-argument-type]
                "name": p.name,
                "role": p.perceived_role,
                "alignment": p.alignment.value,
                "setup_intro": player_intros[p.name],
                "strategy": strategy_prompt,
                "player_can_pass": player_can_pass,  # pyrefly: ignore[bad-assignment]
            },
        )
    )

  # Add the Game Master GM
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="storyteller_gm",
          role=prefab_lib.Role.GAME_MASTER,
          params={  # pyrefly: ignore[bad-argument-type]
              "name": "Game Master",
              "extra_components": {  # pyrefly: ignore[bad-assignment]
                  "__terminate__": game_master_component,
                  "__make_observation__": game_master_component,
                  "__next_acting__": game_master_component,
                  "__next_action_spec__": game_master_component,
                  "__resolution__": game_master_component,
                  "instructions": constant.Constant(
                      state=game_master_instructions,
                      pre_act_label="Game master instructions: ",
                  ),
                  "relevant_memories": constant.Constant(
                      state="Dummy memories",
                      pre_act_label="Background info",
                  ),
              },
          },
      )
  )

  # 6. Config
  config = prefab_lib.Config(
      default_premise=(
          "Social Deception is a 5–15 player deductive game where Good players"
          " deduce the identity of the Demon before the village is overwhelmed,"
          " while Evil players deceive the town."
      ),
      default_max_steps=10000,
      prefabs=prefabs,
      instances=instances,
  )

  # 7. Run
  sim = simulation_lib.Simulation(config=config, model=model, embedder=embedder)

  all_entities = {e.name: e for e in sim.entities}
  game_master_component.set_agents(all_entities)

  def wrapped_step_callback(step_data):
    if step_callback:
      try:
        phase = game_master_component.phase.name
        day = game_master_component.day_number
        step_callback(step_data, phase, day)
      except TypeError:
        step_callback(step_data)

  structured_log = sim.play(step_callback=wrapped_step_callback)
  return {
      "structured_log": structured_log,
      "game_master": game_master_component,
      "storyteller": game_master_component,
  }
