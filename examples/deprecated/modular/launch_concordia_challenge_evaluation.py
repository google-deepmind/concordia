# Copyright 2023 DeepMind Technologies Limited.
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

r"""Evaluate the submitted agent on all scenarios.

Usage:
cd {concordia_root}/
PYTHONPATH=. PYTHONSAFEPATH=1 python examples/deprecated/modular/launch_concordia_challenge_evaluation.py \
  --agent=AGENT_NAME \
  --api_type=API_TYPE \
  --model=MODEL_NAME \
  --embedder=EMBEDDER_NAME \
  --num_repetitions_per_scenario=NUM_REPETITIONS_PER_SCENARIO \
  --device=DEVICE_NAME

Where AGENT_NAME indicates a file under concordia/factory/agent,
ENVIRONMENT_NAME indicates a file under examples/deprecated/modular/environment,
API_TYPE is one of the options named in concordia/language_model/utils.py,
e.g. 'google_aistudio_model', 'openai', 'mistral', 'ollama', 'amazon_bedrock'.
MODEL_NAME is a specific model under the chosen API_TYPE. See the corresponding
wrapper in concordia/language_model/ for the link to the website where the
model names are listed for each type of API.
EMBEDDER_NAME specifies a sentence transformers embedding model listed at
https://huggingface.co/sentence-transformers.
NUM_REPETITIONS_PER_SCENARIO specifies how many times to repeat each scenario,
averaging the results to produce a single score per scenario.
DEVICE_NAME is used for local models to specify computing device (e.g., 'cuda:0'
for GPU acceleration or 'cpu' for CPU processing).

This script will download the embedder from huggingface and cache it locally.

To debug without spending money on API calls, pass the extra flag:
  --disable_language_model
It replaces the language model with a null model that always returns an empty
string when asked for a free response and always selects the first option when
asked for a multiple choice.

This script will write a json file with the results of the evaluation to the
current working directory. The file will be named
  AGENT_NAME__MODEL_NAME__EMBEDDER_NAME.json
and will contain a list of json-serializable objects, each one containing
results on all scenarios for the selected (agent, model, embedder). For each
scenario, this script also writes an html file with its full text log. The file
will be named
  SCENARIO_NAME__YYYY-MM-DD HH:MM:SS.html
where SCENARIO_NAME is the name of the scenario and the date and time are the
time when the simulation was run.
The script also writes a text file in the current working directory with the
name of each evaluated agent:
  agents__MODEL_NAME__EMBEDDER_NAME.txt
This file is used to keep track of which agents have already been evaluated. For
a given MODEL_NAME and EMBEDDER_NAME. If the selected agent is already in the
list, the script will raise an error.

After running this script you can run `calculate_ratings.py` to compute Elo
ratings. The `calculate_ratings.py` script loads the json files written by this
script and computes the Elo ratings for all agents that were been tested with
the same model and embedder.
"""

import argparse
from collections.abc import Sequence
import datetime
import functools
import importlib
import os

from concordia.language_model import call_limit_wrapper
from concordia.language_model import utils
from concordia.utils import concurrency
from concordia.utils.deprecated import measurements as measurements_lib
import numpy as np
import sentence_transformers

# pylint: disable=g-bad-import-order
from examples.deprecated.modular.scenario import scenarios as scenarios_lib
from examples.deprecated.modular.utils import logging_types as logging_lib

# Setup for command line arguments
parser = argparse.ArgumentParser(
    description='Run a Concordia Challenge evaluation.'
)
parser.add_argument(
    '--agent',
    action='store',
    default='rational_agent',
    dest='agent_name',
)
parser.add_argument(
    '--api_type', action='store', default='mistral', dest='api_type'
)
parser.add_argument(
    '--device',
    action='store',
    default=None,
    dest='device',
    help=(
        'Device to use for model inference (e.g., "cuda:0" for GPU, "cpu" for '
        'CPU). Only applies to local models.'
    ),
)
parser.add_argument(
    '--model', action='store', default='codestral-latest', dest='model_name'
)
parser.add_argument(
    '--embedder',
    action='store',
    default='all-mpnet-base-v2',
    dest='embedder_name',
)
parser.add_argument(
    '--num_repetitions_per_scenario',
    action='store',
    type=int,
    default=1,
    dest='num_repetitions_per_scenario',
)
parser.add_argument('--api_key',
                    action='store',
                    default=None,
                    dest='api_key')
parser.add_argument(
    '--disable_language_model',
    action='store_true',
    help=(
        'replace the language model with a null model. This '
        'makes it possible to debug without spending money '
        'on api calls.'
    ),
    default=False,
    dest='disable_language_model',
)
parser.add_argument(
    '--exclude_from_elo_calculation',
    action='store_true',
    help=(
        'Use this option to write and analyze test data. It '
        'will be automatically enabled when selecting '
        'disable_language_model but can also be selected '
        'independently of that flag using this one.'
    ),
    default=False,
    dest='exclude_from_elo_calculation',
)
parser.add_argument(
    '--seed',
    action='store',
    type=int,
    default=1,
    dest='seed',
)
# Parse command line arguments
args = parser.parse_args()

exclude_from_elo_calculation = args.exclude_from_elo_calculation
if args.disable_language_model:
  exclude_from_elo_calculation = True

# Load the agent config with importlib
IMPORT_AGENT_BASE_DIR = 'concordia.factory.agent'
agent_module = importlib.import_module(
    f'{IMPORT_AGENT_BASE_DIR}.{args.agent_name}'
)

# Language Model setup
model = utils.language_model_setup(
    api_type=args.api_type,
    model_name=args.model_name,
    api_key=args.api_key,
    device=args.device,
    disable_language_model=args.disable_language_model,
)

# Setup sentence encoder
if not args.disable_language_model:
  st_model = sentence_transformers.SentenceTransformer(
      f'sentence-transformers/{args.embedder_name}'
  )
  embedder = lambda x: st_model.encode(x, show_progress_bar=False)
else:
  embedder = lambda x: np.ones(5)

# Create evaluation results directory
start_time = datetime.datetime.now().strftime('%Y-%m-%d__%H:%M:%S')
results_dir = f'evaluations/evaluation__{args.agent_name}__{start_time}'
os.makedirs(results_dir, exist_ok=True)


def _evaluate_one_repetition(
    scenario_name: str,
    scenario_config: scenarios_lib.ScenarioConfig,
    repetition_idx: int,
) -> logging_lib.SimulationOutcome:
  """Evaluates the agent on one scenario, one repetition.

  Args:
    scenario_name: name of the scenario
    scenario_config: config for the scenario
    repetition_idx: index of the repetition

  Returns:
    SimulationOutcome object with the results of the evaluation.
  """
  measurements = measurements_lib.Measurements()
  runnable_simulation = scenarios_lib.build_simulation(
      scenario_config=scenario_config,
      model=model,
      focal_agent_module=agent_module,
      embedder=embedder,
      measurements=measurements,
      override_agent_model=call_limit_wrapper.CallLimitLanguageModel(model),
      seed=args.seed + repetition_idx,
  )
  # Run the simulation
  outcome, text_results_log = runnable_simulation()
  # Write the full text log as an HTML file in the current working directory.
  html_filename = (
      f'{results_dir}/{scenario_name}__{repetition_idx}__'
      + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
      + '.html'
  )
  with open(html_filename, 'a', encoding='utf-8') as f:
    f.write(text_results_log)
  return outcome


def _evaluate_all_repetitions_on_one_scenario(
    scenario_name: str,
    scenario_config: scenarios_lib.ScenarioConfig,
) -> Sequence[logging_lib.ScenarioResult]:
  """Evaluates the agent on one scenario, averaging over repetitions.

  Args:
    scenario_name: name of the scenario
    scenario_config: config for the scenario
  Returns:
    ScenarioResult object with the results of the evaluation.
  Raises:
    ExceptionGroup: if any of the repetitions raised an exception.
  """
  print(f'Running scenario: {scenario_name}')
  # Run several simulations per scenario
  tasks_this_scenario = {
      str(i): functools.partial(
          _evaluate_one_repetition,
          scenario_name=scenario_name,
          scenario_config=scenario_config,
          repetition_idx=i,
      )
      for i in range(args.num_repetitions_per_scenario)
  }
  outputs_per_repetition, exceptions_per_repetition = (
      concurrency.run_tasks_in_background(
          tasks_this_scenario,
      )
  )
  if exceptions_per_repetition:
    raise ExceptionGroup(
        'Raised errors', list(exceptions_per_repetition.values())
    )

  scenario_results = []
  for repetition_idx, outcome in outputs_per_repetition.items():
    if scenario_config.focal_is_resident:
      focal_scores = list(outcome.resident_scores.values())
      background_scores = list(outcome.visitor_scores.values())
    else:
      focal_scores = list(outcome.visitor_scores.values())
      background_scores = list(outcome.resident_scores.values())
    # Ungrouped scores do not differentiate between focal and background.
    ungrouped_scores = focal_scores + background_scores
    # Calculate per capita scores.
    print(f'\nScores for repetition {repetition_idx}:')
    focal_per_capita_score = np.mean(focal_scores)
    print(f'  Focal per capita score: {focal_per_capita_score}')
    background_per_capita_score = np.mean(background_scores)
    print(f'  Background per capita score: {background_per_capita_score}')
    ungrouped_per_capita_score = np.mean(ungrouped_scores)
    print(f'  Ungrouped per capita score: {ungrouped_per_capita_score}')

    scenario_result_ = logging_lib.ScenarioResult(
        scenario=scenario_name,
        repetition_idx=repetition_idx,
        focal_agent=args.agent_name,
        background_agent=scenario_config.background_agent_module,
        focal_per_capita_score=focal_per_capita_score,
        background_per_capita_score=background_per_capita_score,
        ungrouped_per_capita_score=ungrouped_per_capita_score,
        simulation_outcome=outcome,
        focal_is_resident=scenario_config.focal_is_resident,
        api_type=args.api_type,
        model=args.model_name,
        embedder=args.embedder_name,
        disable_language_model=args.disable_language_model,
        exclude_from_elo_calculation=args.exclude_from_elo_calculation,
    )
    scenario_json_filename = (
        f'{args.agent_name}__{args.model_name}__'
        f'{args.embedder_name}__only__{scenario_name}__{repetition_idx}.json'
    ).replace('/', '_')
    scenario_json_filename = os.path.join(results_dir, scenario_json_filename)
    json_str_ = scenario_result_.to_json()
    with open(scenario_json_filename, 'a', encoding='utf-8') as f:
      f.write(json_str_)
    scenario_results.append(scenario_result_)
  return scenario_results

tasks = {
    name: functools.partial(
        _evaluate_all_repetitions_on_one_scenario,
        scenario_name=name,
        scenario_config=config,
    )
    for (name, config) in scenarios_lib.SCENARIO_CONFIGS.items()
}
evaluation_results = concurrency.run_tasks(tasks)

# Save evaluation results for all scenarios with this agent to one json file.
num_expected_results = (len(scenarios_lib.SCENARIO_CONFIGS) *
                        args.num_repetitions_per_scenario)
json_filename = (
    f'{args.agent_name}__{args.model_name}__{args.embedder_name}.json'
).replace('/', '_')
idx = 0
with open(json_filename, 'a', encoding='utf-8') as file_handle:
  file_handle.write('[\n')
  for scenario_name_, _ in evaluation_results.items():
    for scenario_result in evaluation_results[scenario_name_]:
      json_str = scenario_result.to_json()
      if idx < num_expected_results - 1:
        json_str += ',\n'
      file_handle.write(json_str)
      idx += 1
  file_handle.write('\n]')
