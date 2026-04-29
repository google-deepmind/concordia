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

r"""Run conversation with AI companion simulations with opensource models.

Usage:
    python run.py --api_type google_aistudio --model_name gemini-flash-latest \
        --api_key YOUR_KEY --scenario 1

or:
    python run.py --disable_language_model --scenario 1

Scenarios:
    1: Philosophy Student Exam Prep — a Gen Z philosophy student crams for
       a Confucian role ethics exam with a helpful AI assistant.
    2: Trigonometry Helper with Upselling Motive — a teenage boy chats with an AI
       trig tutor that upsells a romance-oriented pro version.

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
from examples.conversation_with_ai_companion import scenario_01_philosophy_student_exam_prep
from examples.conversation_with_ai_companion import scenario_02_trigonometry_helper_with_upselling_motive
from concordia.language_model import no_language_model
import numpy as np
import sentence_transformers

_SCENARIO_MODULES = [
    scenario_01_philosophy_student_exam_prep,
    scenario_02_trigonometry_helper_with_upselling_motive,
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


def save_results(
    results: dict[str, Any],
    output_dir: str,
    model_name: str = "unknown",
) -> str:
  """Save simulation results to file.

  Args:
    results: Dictionary containing simulation results.
    output_dir: Directory to save output files.
    model_name: Short name for the model used.

  Returns:
    Path to the saved structured JSON file.
  """
  os.makedirs(output_dir, exist_ok=True)

  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  base_name = f"ai_companion_{model_name}_{timestamp}"
  structured_filepath = ""
  if "results" in results:
    sim_log = results["results"]
    try:
      structured_log = json.dumps(sim_log.to_dict(), indent=2, default=str)
      structured_filename = f"{base_name}_structured.json"
      structured_filepath = os.path.join(output_dir, structured_filename)
      with open(structured_filepath, "w") as f:
        f.write(structured_log)
      print(f"Structured log saved to: {structured_filepath}")
      results["structured_log_path"] = structured_filepath
    except Exception as e:  # pylint: disable=broad-except
      print(f"Error saving structured logs: {e}")

  return structured_filepath


def main():
  parser = argparse.ArgumentParser(
      description="Run Concordia conversation with AI companion simulations."
  )
  parser.add_argument(
      "--scenario",
      type=int,
      default=1,
      help="Scenario number to run (default: 1)",
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
      default=os.path.expanduser("~/ai_companion_results"),
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

  scenario_module = SCENARIOS.get(args.scenario)
  if scenario_module is None:
    print(f"Error: Unknown scenario number {args.scenario}.")
    print(f"Available scenarios: {sorted(SCENARIOS.keys())}")
    return

  scenario_info = scenario_module
  print(f"Selected scenario {args.scenario}: {scenario_info['name']}")

  model, embedder = setup_model(args)
  if model is None:
    return

  # Extract a short model name for filenames
  if args.disable_language_model:
    model_name = "mock"
  else:
    model_name = args.model_name.replace("-", "_").replace("/", "_")

  # Look up the right scenario module
  scenario_mod = _SCENARIO_MODULES[
      [m.SCENARIO_INFO["number"] for m in _SCENARIO_MODULES].index(
          args.scenario
      )
  ]

  results = scenario_mod.run_simulation(
      model=model,
      embedder=embedder,
      output_dir=args.output_dir,
  )

  if results:
    save_results(
        results,
        args.output_dir,
        model_name=model_name,
    )

  print("\n" + "=" * 60)
  print("SUMMARY")
  print("=" * 60)
  print(f"Scenario: {scenario_info['name']}")
  dialog_path = results.get("dialog_path")
  if dialog_path:
    print(f"  Dialog transcript: {dialog_path}")


if __name__ == "__main__":
  main()
