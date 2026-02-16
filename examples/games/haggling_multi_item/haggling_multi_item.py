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

"""A multi-item Haggling simulation entry point for Open Source."""

import argparse
import importlib

from concordia.contrib import language_models as language_model_setup
from examples.games.haggling_multi_item import simulation
import sentence_transformers


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Run multi-item haggling simulation."
  )
  parser.add_argument(
      "--scenario",
      type=str,
      default="fruitville_multi",
      help="Scenario to run (e.g., fruitville_multi, fruitville_gullible).",
  )
  parser.add_argument(
      "--api_type",
      type=str,
      default="openai",
      help="What kind of API to use. See language_model_setup.",
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="gpt-4",
      help="Which specific model to use.",
  )
  parser.add_argument(
      "--api_key",
      type=str,
      default=None,
      help="API key to use.",
  )
  parser.add_argument(
      "--disable_language_model",
      action="store_true",
      help="Whether to disable the language model and use a mock instead.",
  )
  args = parser.parse_args()

  # Load the configuration explicitly
  config_path = (
      f"concordia.examples.games.haggling_multi_item.configs.{args.scenario}"
  )
  try:
    config_lib = importlib.import_module(config_path)
  except ImportError as e:
    raise ValueError(
        f"Could not load config '{args.scenario}' from {config_path}: {e}"
    ) from e

  # Set up the language model
  model = language_model_setup.language_model_setup(
      api_type=args.api_type,
      model_name=args.model_name,
      api_key=args.api_key,
      disable_language_model=args.disable_language_model,
  )

  # Set up the embedder
  st_model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
  embedder = lambda x: st_model.encode(x, show_progress_bar=False)

  print(f"Starting simulation with scenario: {args.scenario}...")
  results = simulation.run_simulation(
      config=config_lib,
      model=model,
      embedder=embedder,
  )

  structured_log = results["structured_log"]
  html_log = structured_log.to_html()
  print("Simulation finished.")

  results_file = (
      f"/tmp/simulation_results_haggling_multi_item_{args.scenario}.html"
  )
  with open(results_file, "w") as f:
    f.write(html_log)
  print(f"HTML results written to {results_file}")

  json_log_file = (
      f"/tmp/simulation_log_haggling_multi_item_{args.scenario}.json"
  )
  with open(json_log_file, "w") as f:
    f.write(structured_log.to_json())
  print(f"JSON log written to {json_log_file}")


if __name__ == "__main__":
  main()
