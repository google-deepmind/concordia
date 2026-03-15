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
r"""Entry point for the signaling marketplace experiment.

This script provides a BYO (Bring Your Own) model interface for running the
signaling marketplace simulation. It supports any language model provider
available through concordia.contrib.language_models (OpenAI, Mistral,
Google AI Studio, Ollama, etc.).

Usage:
  python -m concordia.examples.signaling.run \\
    --api_type=openai --model_name=gpt-4o --num_days=3 --num_agents=10

  python -m concordia.examples.signaling.run \\
    --api_type=google_aistudio --model_name=gemini-2.0-flash \\
    --condition=asocial

  python -m concordia.examples.signaling.run \\
    --disable_language_model  # for testing with a mock model
"""

import argparse
import json
import logging

from concordia.contrib import language_models
from examples.signaling import simulation
from examples.signaling.configs import personas
import sentence_transformers


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Run the signaling marketplace experiment.',
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
      '--condition',
      type=str,
      default='social',
      choices=['social', 'asocial', 'asocial_personal'],
      help=(
          'Experimental condition. '
          '"social": marketplace + personal events + date conversation. '
          '"asocial": marketplace only. '
          '"asocial_personal": marketplace + personal events, no conversation.'
      ),
  )
  parser.add_argument(
      '--num_days',
      type=int,
      default=5,
      help='Number of days to simulate.',
  )
  parser.add_argument(
      '--num_agents',
      type=int,
      default=10,
      help='Number of agents (max 50).',
  )
  parser.add_argument(
      '--num_marketplace_rounds',
      type=int,
      default=5,
      help='Number of marketplace rounds per day.',
  )
  parser.add_argument(
      '--num_dial_rounds',
      type=int,
      default=80,
      help='Number of DIAL conversation rounds per dyad.',
  )
  parser.add_argument(
      '--item_list',
      type=str,
      default='original',
      choices=['original', 'synthetic', 'subculture', 'both'],
      help='Which set of marketplace goods to use.',
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

  model = language_models.language_model_setup(
      api_type=args.api_type,
      model_name=args.model_name,
      api_key=args.api_key,
      disable_language_model=args.disable_language_model,
  )

  if args.use_dummy_embedder or args.disable_language_model:
    embedder = personas.DummyEmbedder()
  else:
    st_model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
    embedder = lambda x: st_model.encode(x, show_progress_bar=False)

  print(
      f'Starting signaling experiment: condition={args.condition},'
      f' num_days={args.num_days}, num_agents={args.num_agents},'
      f' item_list={args.item_list}'
  )

  results = simulation.run_experiment(
      model=model,
      embedder=embedder,
      condition=args.condition,
      num_days=args.num_days,
      num_agents=args.num_agents,
      num_marketplace_rounds=args.num_marketplace_rounds,
      num_dial_rounds=args.num_dial_rounds,
      item_list=args.item_list,
      add_sellers=True,
      seed=args.seed,
  )

  print('Experiment finished.')

  for i, log in enumerate(results.get('marketplace_logs', [])):
    if log is not None:
      html_file = f'/tmp/signaling_marketplace_day_{i}.html'
      with open(html_file, 'w') as f:
        f.write(log.to_html())
      json_file = f'/tmp/signaling_marketplace_day_{i}.json'
      with open(json_file, 'w') as f:
        f.write(log.to_json())
      print(f'Day {i} marketplace log written to {html_file}')

  for entry in results.get('dial_logs', []):
    day = entry['day']
    dyad = entry['dyad']
    log = entry['log']
    if log is not None:
      html_file = f'/tmp/signaling_dial_day_{day}_{dyad}.html'
      with open(html_file, 'w') as f:
        f.write(log.to_html())
      print(f'DIAL log for day {day}, {dyad} written to {html_file}')

  trades_file = '/tmp/signaling_trades.json'
  with open(trades_file, 'w') as f:
    json.dump(results.get('trade_history', []), f, indent=2)
  print(f'Trade history written to {trades_file}')

  prices_file = '/tmp/signaling_prices.json'
  with open(prices_file, 'w') as f:
    json.dump(results.get('price_history', []), f, indent=2)
  print(f'Price history written to {prices_file}')


if __name__ == '__main__':
  main()
