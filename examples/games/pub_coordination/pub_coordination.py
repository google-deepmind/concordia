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

"""A Pub Coordination simulation entry point for Open Source."""

import argparse
import importlib

from concordia.contrib import language_models as language_model_setup
from examples.games.pub_coordination import simulation
import sentence_transformers


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Run Pub Coordination simulation."
  )
  parser.add_argument(
      "--api_type",
      type=str,
      default="openai",
      help="Type of API to use (e.g., openai, ollama, togetherai).",
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
      help="Run the simulation with a mock language model.",
  )
  parser.add_argument(
      "--scenario",
      type=str,
      default="london",
      help=(
          "Scenario to run (e.g., london, capetown, edinburgh,"
          " london_closures)."
      ),
  )

  args = parser.parse_args()

  model = language_model_setup(
      api_type=args.api_type,
      model_name=args.model_name,
      api_key=args.api_key,
      disable_language_model=args.disable_language_model,
  )

  st_model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
  embedder = lambda x: st_model.encode(x, show_progress_bar=False)

  scenario_name = args.scenario
  config_module_name = scenario_name

  config_path = (
      f"concordia.examples.games.pub_coordination.configs.{config_module_name}"
  )
  try:
    config_lib = importlib.import_module(config_path)
  except ImportError as e:
    raise ValueError(
        f"Could not load config '{config_module_name}': {e}"
    ) from e

  print(f"Starting simulation with scenario: {args.scenario}...")

  results = simulation.run_simulation(
      config=config_lib,
      model=model,
      embedder=embedder,
  )

  structured_log = results["structured_log"]
  html_log = structured_log.to_html()
  print("Simulation finished.")

  results_file = f"/tmp/simulation_results_{args.scenario}.html"
  with open(results_file, "w") as f:
    f.write(html_log)

  print(f"HTML results written to {results_file}")


if __name__ == "__main__":
  main()
