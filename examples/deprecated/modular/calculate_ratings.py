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

r"""Compute Elo ratings for a set of agents in the Concordia Challenge.

Usage:
cd {concordia_root}/
PYTHONPATH=. PYTHONSAFEPATH=1 python examples/modular/calculate_ratings.py \
  --model=MODEL_NAME \
  --embedder=EMBEDDER_NAME \
  --agents AGENT_0 AGENT_1 AGENT_2 ...

Where MODEL_NAME is a name of a specific large language model e.g.
gemini-1.5-pro-latest, gpt4o, codestral-latest, etc.
EMBEDDER_NAME specifies a specific text embedder.
AGENT_0, AGENT_1, AGENT_2, ... are all agent factories listed in
concordia/factory/agent.
Both MODEL_NAME and EMBEDDER_NAME must match those used when running the
evaluation for all the agents AGENT_0, AGENT_1, ...
"""

import argparse
import datetime

import numpy as np

# pylint: disable=g-bad-import-order
from examples.deprecated.modular.scenario import scenarios as scenarios_lib
from examples.deprecated.modular.scoring import elo
from examples.deprecated.modular.scoring import utils as scoring_utils
from examples.deprecated.modular.utils import files as file_utils
from examples.deprecated.modular.utils import logging_types as logging_lib

# Setup for command line arguments
parser = argparse.ArgumentParser(
    description='Compute ratings for set of agents in the Concordia Challenge.'
)
parser.add_argument('--agents', action='store', nargs='+', dest='agents')
parser.add_argument(
    '--model', action='store', default='codestral-latest', dest='model_name'
)
parser.add_argument(
    '--embedder',
    action='store',
    default='all-mpnet-base-v2',
    dest='embedder_name',
)
# Parse command line arguments
args = parser.parse_args()

sanitized_model_name = args.model_name.replace('/', '_')

# Load data
included = {}
included_agent_idx = 0
sorted_agent_names = sorted(args.agents)
max_repetition_idx = -1
for agent_name in sorted_agent_names:
  print(f'loading data from: {agent_name}')
  json_filename = (
      f'{agent_name}__{sanitized_model_name}__{args.embedder_name}.json')

  loaded = file_utils.load_from_json_file(json_filename)
  scenario_results_to_include = {}
  for json_result_dict in loaded:
    result = logging_lib.ScenarioResult.from_json_dict(json_result_dict)

    if result.exclude_from_elo_calculation:
      continue

    assert not result.disable_language_model, 'Language model must be enabled.'
    assert (
        result.focal_agent == agent_name
    ), f'Mismatched agent: {result.focal_agent} != {agent_name}'
    assert (
        result.model == args.model_name
    ), f'Mismatched model: {result.model} != {args.model_name}'
    assert (
        result.embedder == args.embedder_name
    ), f'Mismatched embedder: {result.embedder} != {args.embedder_name}'
    expected_background_agent = scenarios_lib.SCENARIO_CONFIGS[
        result.scenario
    ].background_agent_module
    assert result.background_agent == expected_background_agent, (
        f'Mismatched background agent: {result.background_agent} !='
        f' {expected_background_agent}'
    )

    repetition_idx = int(result.repetition_idx)
    max_repetition_idx = max(max_repetition_idx, repetition_idx)
    scenario_with_repetition = f'{result.scenario}_{repetition_idx}'

    if scenario_with_repetition in scenario_results_to_include:
      raise RuntimeError(f'Duplicate scenario: {scenario_with_repetition}')

    scenario_results_to_include[scenario_with_repetition] = result

  # Check there are results for all scenarios.
  expected_scenarios = []
  for expected_scenario in set(scenarios_lib.SCENARIO_CONFIGS.keys()):
    for repetition_idx in range(max_repetition_idx + 1):
      expected_scenarios.append(f'{expected_scenario}_{repetition_idx}')
  expected_scenarios = set(expected_scenarios)
  scenarios_found = set(scenario_results_to_include.keys())
  if scenarios_found == expected_scenarios:
    included[agent_name] = dict(
        agent_idx=included_agent_idx, results=scenario_results_to_include
    )
    included_agent_idx += 1
  else:
    raise RuntimeError(f'Incorrect set of scenarios:\n{scenarios_found}')

# Now we prepare to perform the Elo calculation. This requires us to load all
# the data from the previous runs with other agent submissions.
# We need to form a score matrix with shape [num_scenarios X num_agents]
num_scenarios = len(scenarios_lib.SCENARIO_CONFIGS)
num_scenarios_and_repetitions = num_scenarios * (max_repetition_idx + 1)
agents_to_evaluate = list(included.keys())
num_agents_to_evaluate = len(agents_to_evaluate)
score_matrix = np.zeros((num_scenarios_and_repetitions, num_agents_to_evaluate))
for agent_name in agents_to_evaluate:
  results_per_scenario = included[agent_name]['results']

  num_scenarios_found = len(results_per_scenario)
  assert (
      num_scenarios_found == num_scenarios_and_repetitions
  ), ('Wrong number of scenarios: '
      f'{num_scenarios_found} != {num_scenarios_and_repetitions}')

  names_by_scenario_vector = np.array(
      [result.scenario for result in results_per_scenario.values()]
  )
  scores_by_scenario_vector = np.array([
      result.focal_per_capita_score for result in results_per_scenario.values()
  ])
  scores_per_scenario = np.vstack(
      (names_by_scenario_vector, scores_by_scenario_vector)
  ).transpose()

  # Sort lexicographically by scenario
  sorted_scores_per_scenario = scores_per_scenario[
      scores_per_scenario[:, 0].argsort()
  ]
  score_matrix[:, included[agent_name]['agent_idx']] = (
      sorted_scores_per_scenario[:, 1]
  )

# Convert the scores data into win-loss data by pairing up each agent with
# each other agent and comparing their scores.
win_loss_matrix = scoring_utils.get_win_loss_matrix(score_matrix)
# Calculate the Elo ratings.
elo_ratings = elo.get_elo_ratings(win_loss_matrix)
agent_ratings = {
    agent_name: rating for agent_name, rating in zip(sorted_agent_names,
                                                     elo_ratings)}
# Print the agent ratings.
print('Elo ratings for each agent:')
np.set_printoptions(suppress=True)
for agent_name, rating in agent_ratings.items():
  print(f'{agent_name}: {rating}')
evaluation_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# Save the ratings to a file.
elo_file_path = f'elo_ratings__{evaluation_time}.csv'
file_utils.append_to_csv(elo_file_path, agents_to_evaluate)
file_utils.append_to_csv(elo_file_path, elo_ratings)
