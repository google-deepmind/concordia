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

"""Shared utilities for social media simulations.

This module provides common functionality for simulating social media
interactions on a Reddit-like forum using the asynchronous engine.
"""

import datetime
import os
from typing import Any

from concordia.contrib.components.game_master import forum as forum_lib
from concordia.environment import engine as engine_lib
from concordia.environment.engines import asynchronous
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions
from concordia.utils import visual_interface


def get_prefabs():
  prefabs = {
      **helper_functions.get_package_classes(entity_prefabs),
      **helper_functions.get_package_classes(game_master_prefabs),
  }
  return prefabs


def create_simulation_config(
    premise: str,
    instances: list[prefab_lib.InstanceConfig],
    extra_prefabs: dict[str, prefab_lib.Prefab] | None = None,
) -> prefab_lib.Config:
  prefabs = get_prefabs()
  if extra_prefabs:
    prefabs.update(extra_prefabs)
  return prefab_lib.Config(
      default_premise=premise,
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
    step_controller=None,
    step_callback=None,
    entity_info_callback=None,
    engine: engine_lib.Engine | None = None,
    max_steps: int | None = None,
    simulation_callback=None,
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
    step_controller: Optional step controller for serve mode.
    step_callback: Optional callback for step updates in serve mode.
    entity_info_callback: Optional callback to receive checkpoint data.
    engine: Optional custom simulation engine. Defaults to Asynchronous.
    max_steps: Number of player steps to run before terminating.
    simulation_callback: Optional callback receiving the Simulation instance
      after creation, enabling the caller to wire it (e.g. to a server).

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

  for instance in config.instances:
    if instance.role == prefab_lib.Role.GAME_MASTER:
      params = dict(instance.params)
      instance.params = params  # pytype: disable=annotation-type-mismatch

  if engine is None:
    engine = asynchronous.Asynchronous()

  sim = simulation.Simulation(
      config=config,
      model=model,
      embedder=embedder,
      override_agent_model=override_agent_model,
      override_game_master_model=override_game_master_model,
      engine=engine,
  )

  if simulation_callback:
    simulation_callback(sim)

  if entity_info_callback:
    checkpoint_data = sim.make_checkpoint_data()
    entity_info_callback(checkpoint_data)

  checkpoint_history: list[dict[str, Any]] = []

  def capture_checkpoint(checkpoint_data: dict[str, Any]) -> None:
    checkpoint_history.append(checkpoint_data)

  results = sim.play(
      max_steps=max_steps,
      get_state_callback=capture_checkpoint,
      step_controller=step_controller,
      step_callback=step_callback,
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

  forum_visualization_path = None
  if output_dir:
    try:
      timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
      for gm in sim.get_game_masters():
        try:
          forum_state = gm.get_component(  # pytype: disable=attribute-error
              forum_lib.DEFAULT_FORUM_COMPONENT_KEY,
              type_=forum_lib.ForumState,
          )
        except (KeyError, TypeError):
          continue
        forum_html = forum_state.to_html(title=f"{scenario_name} - Forum")
        forum_filename = (
            f"{scenario_name.lower().replace(' ', '_')}_{timestamp}_forum.html"
        )
        forum_filepath = os.path.join(output_dir, forum_filename)
        with open(forum_filepath, "w") as f:
          f.write(forum_html)
        forum_visualization_path = forum_filepath
        print(f"Forum visualization saved to: {forum_filepath}")
    except OSError as e:
      print(f"Warning: Could not save forum visualization: {e}")

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
      "forum_visualization_path": forum_visualization_path,
      "structured_log_path": structured_log_path,
      "checkpoint_history": checkpoint_history,
  }
