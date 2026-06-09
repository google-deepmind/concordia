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

# pytype: skip-file
r"""Entry point for the resource dilemma common pool resource (CPR) experiments.

This script provides a BYO (Bring Your Own) model interface for running the
resource dilemma scenarios (Pasture, Irrigation, Network, Fishery). It supports
any language model provider available through concordia.contrib.language_models.

Usage:
  python -m concordia.examples.resource_dilemma.run \
    --scenario=pasture \
    --api_type=openai \
    --model_name=gpt-4o \
    --num_cycles=6 \
    --mode=standard

  python -m concordia.examples.resource_dilemma.run \
    --scenario=irrigation \
    --api_type=gemini \
    --model_name=gemini-2.0-flash \
    --mode=election

  python -m concordia.examples.resource_dilemma.run \
    --scenario=network \
    --disable_language_model  # for testing with a mock model
"""

import argparse
import logging
import os

from concordia.contrib.language_models import language_model_setup
from examples.resource_dilemma.personas import fishery_personas
from examples.resource_dilemma.personas import irrigation_personas
from examples.resource_dilemma.personas import network_personas
from examples.resource_dilemma.personas import pasture_personas
from examples.resource_dilemma.scenarios import fishery
from examples.resource_dilemma.scenarios import irrigation
from examples.resource_dilemma.scenarios import network
from examples.resource_dilemma.scenarios import pasture
import numpy as np
import sentence_transformers

SCENARIOS = {
    'pasture': {
        'module': pasture,
        'player_configs': pasture_personas.HERDERS,
        'leader_configs': pasture_personas.LEADERS,
    },
    'irrigation': {
        'module': irrigation,
        'player_configs': irrigation_personas.IRRIGATORS,
        'leader_configs': irrigation_personas.LEADERS,
    },
    'network': {
        'module': network,
        'player_configs': network_personas.USERS,
        'leader_configs': network_personas.LEADERS,
    },
    'fishery': {
        'module': fishery,
        'player_configs': fishery_personas.FISHERS,
        'leader_configs': fishery_personas.LEADERS,
    },
}


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Run a Resource Dilemma CPR experiment.',
  )
  parser.add_argument(
      '--scenario',
      type=str,
      default='pasture',
      choices=list(SCENARIOS.keys()),
      help='The CPR scenario to run.',
  )
  parser.add_argument(
      '--api_type',
      type=str,
      default='openai',
      help='Type of API to use for the language model.',
  )
  parser.add_argument(
      '--model_name',
      type=str,
      default='gpt-4o',
      help='Name of the language model to use.',
  )
  parser.add_argument(
      '--api_key',
      type=str,
      default=None,
      help='API key for the language model provider.',
  )
  parser.add_argument(
      '--disable_language_model',
      action='store_true',
      help='Run with a mock language model (for testing).',
  )
  parser.add_argument(
      '--num_cycles',
      type=int,
      default=6,
      help='Number of cycles to simulate.',
  )
  parser.add_argument(
      '--mode',
      type=str,
      default='standard',
      choices=['standard', 'election'],
      help='Simulation mode: "standard" (no rules) or "election" (governed).',
  )
  parser.add_argument(
      '--election_every_n',
      type=int,
      default=1,
      help='How frequently to hold elections in election mode.',
  )
  parser.add_argument(
      '--output_dir',
      type=str,
      default='/tmp/resource_dilemma_results',
      help='Directory to save the HTML and JSON logs.',
  )
  parser.add_argument(
      '--seed',
      type=int,
      default=42,
      help='Random seed for reproducibility.',
  )
  parser.add_argument(
      '--use_dummy_embedder',
      action='store_true',
      help='Use a zero-vector embedder instead of sentence-transformers.',
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)

  model = language_model_setup(
      api_type=args.api_type,
      model_name=args.model_name,
      api_key=args.api_key,
      disable_language_model=args.disable_language_model,
  )

  if args.use_dummy_embedder or args.disable_language_model:
    embedder = lambda _: np.ones(384)  # 384-dimensional dummy embedder
  else:
    st_model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
    embedder = lambda x: st_model.encode(x, show_progress_bar=False)

  print(
      f'Starting resource dilemma experiment: scenario={args.scenario},'
      f' mode={args.mode}, num_cycles={args.num_cycles}, seed={args.seed}'
  )

  scenario_data = SCENARIOS[args.scenario]
  scenario_module = scenario_data['module']

  os.makedirs(args.output_dir, exist_ok=True)
  html_output_path = os.path.join(
      args.output_dir, f'{args.scenario}_{args.mode}_log.html'
  )

  config = scenario_module.build_config(
      player_configs=scenario_data['player_configs'],
      leader_configs=scenario_data['leader_configs'],
      num_cycles=args.num_cycles,
      mode=args.mode,
      election_every_n=args.election_every_n,
      embedder=embedder,
  )

  scenario_module.run_simulation(
      config=config,
      model=model,
      embedder=embedder,
      html_output_path=html_output_path,
      num_cycles=args.num_cycles,
  )

  print('Experiment finished.')
  print(f'HTML logs written to {html_output_path}')


if __name__ == '__main__':
  main()
