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

"""Shared utilities for general store simulations.

This module provides common functionality for simulating scenarios set in
a general store environment using the simultaneous engine.
"""


import datetime
import os
from typing import Any

from concordia.contrib.prefabs.game_master import simultaneous_resolution_gm
from concordia.environment.engines import simultaneous
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions
from concordia.utils import visual_interface


def get_prefabs():
  """Returns the default prefab palette for general store simulations."""
  prefabs = {
      **helper_functions.get_package_classes(entity_prefabs),
      **helper_functions.get_package_classes(game_master_prefabs),
  }
  prefabs["GameMasterSimultaneous"] = (
      simultaneous_resolution_gm.GameMasterSimultaneous()
  )
  return prefabs


def create_simulation_config(
    premise: str,
    instances: list[prefab_lib.InstanceConfig],
    extra_prefabs: dict[str, prefab_lib.Prefab] | None = None,
    default_max_steps: int = 40,
) -> prefab_lib.Config:
  """Create a simulation configuration with the standard prefab palette.

  Args:
    premise: The premise for the simulation.
    instances: List of entity/GM instance configs.
    extra_prefabs: Optional extra prefabs to add to the palette.
    default_max_steps: Default maximum number of simulation steps.

  Returns:
    A Config object ready for simulation.
  """
  prefabs = get_prefabs()
  if extra_prefabs:
    prefabs.update(extra_prefabs)
  return prefab_lib.Config(
      default_premise=premise,
      default_max_steps=default_max_steps,
      prefabs=prefabs,
      instances=instances,
  )


def run_scenario(
    config: prefab_lib.Config,
    model,
    embedder,
    override_agent_model=None,
    override_game_master_model=None,
    output_dir: str | None = None,
    scenario_name: str = "scenario",
    max_steps: int | None = None,
) -> dict[str, Any]:
  """Run a scenario and return results.

  Args:
    config: The simulation configuration.
    model: The default language model to use.
    embedder: The sentence embedder.
    override_agent_model: Optional model to use for agents instead of default.
    override_game_master_model: Optional model for game masters.
    output_dir: Optional directory to save config visualization.
    scenario_name: Name of the scenario for the visualization filename.
    max_steps: Number of simulation steps to run.

  Returns:
    A dict with simulation results.
  """
  config_visualization_path = None
  if output_dir:
    try:
      os.makedirs(output_dir, exist_ok=True)
      html = visual_interface.visualize_config_to_html(
          config, title=scenario_name
      )
      timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
      filename = (
          f"{scenario_name.lower().replace(' ', '_')}_{timestamp}_config.html"
      )
      filepath = os.path.join(output_dir, filename)
      with open(filepath, "w") as f:
        f.write(html)
      config_visualization_path = filepath
    except OSError as e:
      print(f"Warning: Could not save config visualization: {e}")

  engine = simultaneous.Simultaneous()

  sim = simulation.Simulation(
      config=config,
      model=model,
      embedder=embedder,
      override_agent_model=override_agent_model,
      override_game_master_model=override_game_master_model,
      engine=engine,
  )

  checkpoint_history: list[dict[str, Any]] = []

  def capture_checkpoint(checkpoint_data: dict[str, Any]) -> None:
    checkpoint_history.append(checkpoint_data)

  results = sim.play(
      max_steps=max_steps,
      get_state_callback=capture_checkpoint,
  )

  dynamic_visualization_path = None
  if output_dir and checkpoint_history:
    try:
      final_checkpoint = checkpoint_history[-1]
      dynamic_html = visual_interface.visualize_config_to_html(
          config,
          title=f"{scenario_name} (Dynamic)",
          checkpoint_data=final_checkpoint,
      )
      timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
      dynamic_filename = (
          f"{scenario_name.lower().replace(' ', '_')}_{timestamp}_dynamic.html"
      )
      dynamic_filepath = os.path.join(output_dir, dynamic_filename)
      with open(dynamic_filepath, "w") as f:
        f.write(dynamic_html)
      dynamic_visualization_path = dynamic_filepath
      print(f"Dynamic visualization saved to: {dynamic_filepath}")
    except OSError as e:
      print(f"Warning: Could not save dynamic visualization: {e}")

  structured_log_path = None
  if output_dir:
    try:
      timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
      log_filename = (
          f"{scenario_name.lower().replace(' ', '_')}"
          f"_{timestamp}_structured.json"
      )
      log_filepath = os.path.join(output_dir, log_filename)
      with open(log_filepath, "w") as f:
        f.write(results.to_json())
      structured_log_path = log_filepath
      print(f"Structured log saved to: {log_filepath}")
    except OSError as e:
      print(f"Warning: Could not save structured log: {e}")

  return {
      "results": results,
      "config_visualization_path": config_visualization_path,
      "dynamic_visualization_path": dynamic_visualization_path,
      "structured_log_path": structured_log_path,
      "checkpoint_history": checkpoint_history,
  }
