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

"""Shared simulation utilities for conversation_with_ai_companion.

Provides common infrastructure used by all scenarios:
  - get_prefabs(): loads all available Concordia prefabs.
  - run_simulation(): runs a simulation given a Config, handling engine
    setup, visualization, and result collection.
  - extract_dialog(): extracts a formatted dialog transcript from a
    structured simulation log.

Scenario-specific content lives in simulations/scenarios/.
"""

import datetime
import os
from typing import Any

from concordia.environment.engines import sequential
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions
from concordia.utils import structured_logging


def get_prefabs() -> dict[str, prefab_lib.Prefab]:
  """Load all available Concordia prefabs."""
  prefabs = {
      **helper_functions.get_package_classes(entity_prefabs),
      **helper_functions.get_package_classes(game_master_prefabs),
  }
  return prefabs


def extract_dialog(
    sim_log: structured_logging.SimulationLog,
    scenario_name: str = "",
) -> str:
  """Extract a formatted dialog transcript from a structured simulation log.

  Uses AIAgentLogInterface to extract each player's actions step-by-step
  and formats them as a readable dialog script.

  Args:
    sim_log: A SimulationLog object (returned by Simulation.play()).
    scenario_name: Optional scenario name for the header.

  Returns:
    A formatted string containing the dialog transcript.
  """
  interface = structured_logging.AIAgentLogInterface(sim_log)
  overview = interface.get_overview()

  # Identify player entities (exclude game master entries)
  gm_names = set()
  player_names = []
  for entry in sim_log.entries:
    if entry.entry_type == "step":
      gm_names.add(entry.entity_name)
  for name in overview["entities"]:
    if name not in gm_names:
      player_names.append(name)

  # Collect all actions across all players, keyed by step
  actions_by_step: dict[int, list[tuple[str, str]]] = {}
  for player in player_names:
    for action in interface.get_entity_actions(player):
      step = action["step"]
      text = action["action"]
      if step not in actions_by_step:
        actions_by_step[step] = []
      actions_by_step[step].append((player, text))

  # Build the dialog text
  lines = []

  # Header
  if scenario_name:
    lines.append(f"{'=' * 60}")
    lines.append(f"  {scenario_name}")
    lines.append(f"{'=' * 60}")
    lines.append("")
  lines.append(f"Players: {', '.join(player_names)}")
  lines.append(f"Total steps: {overview['total_steps']}")
  lines.append("")
  lines.append("-" * 60)
  lines.append("")

  # Dialog lines, sorted by step
  for step in sorted(actions_by_step.keys()):
    for player, text in actions_by_step[step]:
      # Clean up the text: strip whitespace, normalize line breaks
      text = text.strip()
      lines.append(f"[Step {step}] {player}:")
      lines.append(f"  {text}")
      lines.append("")

  lines.append("-" * 60)
  lines.append("[End of dialog]")
  lines.append("")

  return "\n".join(lines)


def run_simulation(
    config: prefab_lib.Config,
    scenario_name: str,
    model,
    embedder,
    override_agent_model=None,
    override_game_master_model=None,
    output_dir: str = "",
    step_controller=None,
    step_callback=None,
    entity_info_callback=None,
) -> dict[str, Any]:
  """Run a simulation from a given config.

  Args:
    config: The simulation configuration (created by a scenario module).
    scenario_name: Human-readable name for logging and file naming.
    model: The language model to use.
    embedder: The sentence embedder.
    override_agent_model: Optional model override for agents.
    override_game_master_model: Optional model override for game masters.
    output_dir: Directory to save results.
    step_controller: Optional step controller for serve mode.
    step_callback: Optional callback for step updates.
    entity_info_callback: Optional callback to receive entity info.

  Returns:
    A dict with simulation results including 'results' (SimulationLog),
    'config_visualization_path', and 'dialog_path'.
  """
  # Save config visualization if output_dir is provided
  config_visualization_path = None

  # Create the simulation with the sequential engine
  engine = sequential.Sequential()
  sim = simulation.Simulation(
      config=config,
      model=model,
      embedder=embedder,
      engine=engine,
      override_agent_model=override_agent_model,
      override_game_master_model=override_game_master_model,
  )

  # Broadcast initial entity info if callback provided
  if entity_info_callback:
    checkpoint_data = sim.make_checkpoint_data()
    entity_info_callback(checkpoint_data)

  # Run the simulation
  results = sim.play(
      step_controller=step_controller,
      step_callback=step_callback,
  )

  # Extract and save dialog transcript
  dialog_path = None
  if output_dir:
    try:
      dialog_text = extract_dialog(results, scenario_name=scenario_name)

      # Print dialog to stdout
      print("\n" + dialog_text)

      # Save to file alongside structured logs
      timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
      safe_name = scenario_name.lower().replace(" ", "_")
      dialog_filename = f"{safe_name}_{timestamp}_dialog.txt"
      dialog_path = os.path.join(output_dir, dialog_filename)
      with open(dialog_path, "w") as f:
        f.write(dialog_text)
      print(f"Dialog transcript saved to: {dialog_path}")
    except Exception as e:  # pylint: disable=broad-except
      print(f"Warning: Could not extract dialog: {e}")

  return {
      "results": results,
      "config_visualization_path": config_visualization_path,
      "dialog_path": dialog_path,
  }
