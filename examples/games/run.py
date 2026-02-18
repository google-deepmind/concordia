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

"""Unified runner for Concordia game simulations.

This script provides a single entry point for running any game simulation
(haggling, haggling_multi_item, pub_coordination). It handles the shared
boilerplate of argument parsing, model setup, config loading, and output
writing.

Usage:
  python -m concordia.examples.games.run --game=haggling --scenario=fruitville
  python -m concordia.examples.games.run --game=pub_coordination
  --scenario=london
  python -m concordia.examples.games.run --game=haggling_multi_item
  --scenario=fruitville_multi
"""

import argparse
import importlib
import sys

from concordia.contrib import language_models
import sentence_transformers


_AVAILABLE_GAMES = (
    "haggling",
    "haggling_multi_item",
    "pub_coordination",
)

_DEFAULT_SCENARIOS = {
    "haggling": "fruitville",
    "haggling_multi_item": "fruitville_multi",
    "pub_coordination": "london",
}


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Run a Concordia game simulation.",
  )
  parser.add_argument(
      "--game",
      type=str,
      required=True,
      choices=_AVAILABLE_GAMES,
      help=f"Which game to run. Options: {', '.join(_AVAILABLE_GAMES)}",
  )
  parser.add_argument(
      "--scenario",
      type=str,
      default=None,
      help=(
          "Scenario config to use. Defaults vary by game "
          f"({_DEFAULT_SCENARIOS})."
      ),
  )
  parser.add_argument(
      "--api_type",
      type=str,
      default="openai",
      help="Type of API to use for the language model.",
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="gpt-4o",
      help="Name of the language model to use.",
  )
  parser.add_argument(
      "--api_key",
      type=str,
      default=None,
      help="API key for the language model provider.",
  )
  parser.add_argument(
      "--disable_language_model",
      action="store_true",
      help="Run with a mock language model (for testing).",
  )
  parser.add_argument(
      "--focal_player_prefab",
      type=str,
      default=None,
      help=(
          "Override the focal player prefab from the config. "
          "Options: basic__Entity, rational__Entity, puppet__Entity, etc."
      ),
  )
  args = parser.parse_args()

  game = args.game
  scenario = args.scenario or _DEFAULT_SCENARIOS.get(game)
  if scenario is None:
    print(
        f"Error: No default scenario for game '{game}'. "
        "Please specify --scenario.",
        file=sys.stderr,
    )
    sys.exit(1)

  # Load the game's simulation module
  simulation_path = f"concordia.examples.games.{game}.simulation"
  try:
    simulation = importlib.import_module(simulation_path)
  except ImportError as e:
    print(
        f"Error: Could not load simulation for game '{game}': {e}",
        file=sys.stderr,
    )
    sys.exit(1)

  # Load the scenario config
  config_path = f"concordia.examples.games.{game}.configs.{scenario}"
  try:
    config_lib = importlib.import_module(config_path)
  except ImportError as e:
    print(
        f"Error: Could not load scenario '{scenario}' for game '{game}': {e}",
        file=sys.stderr,
    )
    sys.exit(1)

  if args.focal_player_prefab:
    config_lib.FOCAL_PLAYER_PREFAB = args.focal_player_prefab

  model = language_models.language_model_setup(
      api_type=args.api_type,
      model_name=args.model_name,
      api_key=args.api_key,
      disable_language_model=args.disable_language_model,
  )

  st_model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
  embedder = lambda x: st_model.encode(x, show_progress_bar=False)

  prefab_info = (
      f", prefab={args.focal_player_prefab}" if args.focal_player_prefab else ""
  )
  print(f"Starting {game} simulation with scenario: {scenario}{prefab_info}...")
  results = simulation.run_simulation(
      config=config_lib,
      model=model,
      embedder=embedder,
  )

  print("Simulation finished.")

  # Write output files
  if results and "structured_log" in results:
    structured_log = results["structured_log"]

    html_file = f"/tmp/simulation_results_{game}_{scenario}.html"
    with open(html_file, "w") as f:
      f.write(structured_log.to_html())
    print(f"HTML results written to {html_file}")

    json_file = f"/tmp/simulation_log_{game}_{scenario}.json"
    with open(json_file, "w") as f:
      f.write(structured_log.to_json())
    print(f"JSON log written to {json_file}")

  # Print score summary
  if results:
    focal_scores = results.get("focal_scores", {})
    if focal_scores:
      print("\nFocal player scores:")
      for name, score in focal_scores.items():
        print(f"  {name}: {score:.2f}")

    background_scores = results.get("background_scores", {})
    if background_scores:
      print("\nBackground player scores:")
      for name, score in background_scores.items():
        print(f"  {name}: {score:.2f}")


if __name__ == "__main__":
  main()
