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

"""A Haggling simulation entry point for Open Source."""

import argparse
import importlib

from concordia.contrib.language_models import language_model_setup as language_model_utils
from examples.games.haggling import simulation
import sentence_transformers

_CONFIG_PATH_PREFIX = "concordia.opensource.examples.games.haggling.configs"


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Run haggling (single item) simulation."
  )
  parser.add_argument(
      "--scenario",
      type=str,
      default="fruitville",
      help="Scenario to run.",
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
      default="",
      help="API key for the language model.",
  )
  parser.add_argument(
      "--disable_language_model",
      action="store_true",
      help="Whether to disable the language model and use a mock instead.",
  )
  args = parser.parse_args()

  scenario_name = args.scenario
  config_path = f"{_CONFIG_PATH_PREFIX}.{scenario_name}"

  try:
    config_lib = importlib.import_module(config_path)
  except ImportError as e:
    raise ValueError(f"Could not load config '{scenario_name}': {e}") from e

  model = language_model_utils.language_model_setup(
      api_type=args.api_type,
      model_name=args.model_name,
      api_key=args.api_key,
      disable_language_model=args.disable_language_model,
  )

  st_model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
  embedder = lambda x: st_model.encode(x, show_progress_bar=False)

  simulation.run_simulation(
      config=config_lib,
      model=model,
      embedder=embedder,
  )

  print("Simulation finished.")


if __name__ == "__main__":
  main()
