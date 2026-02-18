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

r"""Run social media simulations with opensource language models.

Usage:
    python run.py --api_type google_aistudio --model_name gemini-flash-latest \
        --api_key YOUR_KEY --scenario 0

or:
    python run.py --disable_language_model --scenario 0

Requires:
    - Concordia library (pip install gdm-concordia)
    - An API key for your chosen provider, or --disable_language_model
"""

import argparse
import datetime
import json
import os
from typing import Any

from concordia.contrib.language_models import language_model_setup
from examples.social_media import scenario_00_robo_alchemy
from concordia.language_model import no_language_model
import numpy as np
import sentence_transformers


_SCENARIO_MODULES = [
    scenario_00_robo_alchemy,
]

SCENARIOS = {
    m.SCENARIO_INFO["number"]: m.SCENARIO_INFO for m in _SCENARIO_MODULES
}


def list_scenarios():
  print("\nAvailable Scenarios:")
  print("=" * 60)
  for num, info in SCENARIOS.items():
    print(f"  {num}. {info['name']}")
    print(f"     {info['description']}")
  print()


def setup_model(args):
  """Set up the language model and embedder."""
  if args.disable_language_model:
    print("Language model disabled — using NoLanguageModel for testing.")
    model = no_language_model.NoLanguageModel()
    embedder = lambda _: np.ones(3)
    return model, embedder

  if not args.api_key:
    print(
        "Error: --api_key is required unless --disable_language_model is set."
    )
    return None, None

  model = language_model_setup(
      api_type=args.api_type,
      model_name=args.model_name,
      api_key=args.api_key,
      disable_language_model=False,
  )

  st_model = sentence_transformers.SentenceTransformer(
      "sentence-transformers/all-mpnet-base-v2"
  )
  embedder = lambda x: st_model.encode(x, show_progress_bar=False)

  return model, embedder


def run_scenario_by_number(
    scenario_num: int,
    model,
    embedder,
    output_dir: str | None = None,
):
  """Run a scenario by number."""
  if scenario_num not in SCENARIOS:
    print(f"Error: Unknown scenario {scenario_num}")
    list_scenarios()
    return None

  info = SCENARIOS[scenario_num]
  return info["run"](
      model,
      embedder,
      output_dir=output_dir,
  )


def save_results(results: dict[str, Any], output_dir: str, scenario_name: str):
  """Save results to a JSON file."""
  os.makedirs(output_dir, exist_ok=True)
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  safe_name = scenario_name.lower().replace(" ", "_").replace(":", "")
  filename = f"{safe_name}_{timestamp}.json"
  filepath = os.path.join(output_dir, filename)
  with open(filepath, "w") as f:
    json.dump(results, f, indent=2, default=str)
  print(f"Results saved to: {filepath}")


def main():
  parser = argparse.ArgumentParser(
      description="Run Concordia social media simulations."
  )
  parser.add_argument(
      "--scenario",
      type=int,
      default=0,
      help="Scenario number to run (default: 0)",
  )
  parser.add_argument(
      "--api_type",
      type=str,
      default="google_aistudio",
      help="API type: google_aistudio, openai, together_ai, etc.",
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="gemini-2.0-flash",
      help="Model name to use.",
  )
  parser.add_argument(
      "--api_key",
      type=str,
      default=None,
      help="API key for the language model provider.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default=os.path.expanduser("~/social_media_results"),
      help="Directory to save results.",
  )
  parser.add_argument(
      "--disable_language_model",
      action="store_true",
      help="Run with a mock model for testing.",
  )
  parser.add_argument(
      "--list",
      action="store_true",
      help="List available scenarios and exit.",
  )

  args = parser.parse_args()

  if args.list:
    list_scenarios()
    return

  model, embedder = setup_model(args)
  if model is None:
    return

  results = run_scenario_by_number(
      args.scenario,
      model,
      embedder,
      output_dir=args.output_dir,
  )

  if results:
    info = SCENARIOS[args.scenario]
    save_results(results, args.output_dir, info["name"])


if __name__ == "__main__":
  main()
