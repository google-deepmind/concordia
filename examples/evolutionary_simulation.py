# Copyright 2023 DeepMind Technologies Limited.
# Copyright 2025 [SoyGema] - Modifications and additions with Claude Code
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

"""Evolutionary simulation framework for studying cooperation in public goods games."""

from collections import defaultdict
import datetime
from functools import partial
import logging
from operator import itemgetter
from pathlib import Path
import random
from typing import Any, Dict, Optional

from concordia.associative_memory import basic_associative_memory
from concordia.environment.engines import simultaneous
from concordia.language_model import language_model
from concordia.language_model import no_language_model
from concordia.language_model import utils as language_model_utils
from concordia.prefabs.entity import basic_with_plan
from concordia.prefabs.game_master.public_goods_game_master import PublicGoodsGameMaster
from concordia.prefabs.simulation import generic as simulation_generic
from concordia.utils import helper_functions as helper_functions_lib
from concordia.typing import evolutionary as evolutionary_types
from concordia.utils import checkpointing
from concordia.utils import measurements as measurements_lib
import numpy as np
import sentence_transformers

# === Setup logging ===
logger = logging.getLogger(__name__)


# === Language Model Setup Functions ===

def setup_language_model(config: evolutionary_types.EvolutionConfig) -> language_model.LanguageModel:
  """Setup language model based on configuration.
  
  Args:
    config: Evolution configuration containing LLM parameters
    
  Returns:
    Configured language model instance
  """
  if config.disable_language_model:
    logger.info('Using dummy language model (disabled)')
    return no_language_model.NoLanguageModel()
  
  try:
    logger.info(
        'Setting up language model: %s with model %s',
        config.api_type,
        config.model_name,
    )
    model = language_model_utils.language_model_setup(
        api_type=config.api_type,
        model_name=config.model_name,
        api_key=config.api_key,
        device=config.device,
        disable_language_model=False,
    )
    logger.info('Language model setup successful')
    return model
  except Exception as e:
    logger.warning(
        'Failed to setup language model (%s), falling back to dummy model: %s',
        config.api_type,
        e,
    )
    return no_language_model.NoLanguageModel()


def setup_embedder(config: evolutionary_types.EvolutionConfig):
  """Setup sentence embedder based on configuration.
  
  Args:
    config: Evolution configuration containing embedder parameters
    
  Returns:
    Embedder function that takes text and returns numpy array
  """
  if config.disable_language_model:
    logger.info('Using dummy embedder (language model disabled)')
    return lambda text: np.random.random(16)
  
  try:
    logger.info('Setting up sentence embedder: %s', config.embedder_name)
    st_model = sentence_transformers.SentenceTransformer(
        f'sentence-transformers/{config.embedder_name}'
    )
    embedder = lambda x: st_model.encode(x, show_progress_bar=False)
    logger.info('Sentence embedder setup successful')
    return embedder
  except Exception as e:
    logger.warning(
        'Failed to setup sentence embedder (%s), falling back to dummy: %s',
        config.embedder_name,
        e,
    )
    return lambda text: np.random.random(16)


# === Type aliases from concordia.typing.evolutionary ===
Strategy = evolutionary_types.Strategy
SelectionMethod = evolutionary_types.SelectionMethod
EvolutionConfig = evolutionary_types.EvolutionConfig
CheckpointData = evolutionary_types.CheckpointData


# === Helper functions moved to concordia.utils.checkpointing ===


# === Default configuration ===
DEFAULT_CONFIG = evolutionary_types.EvolutionConfig(
    # Use dummy model by default for backwards compatibility and testing
    disable_language_model=True
)

# === Example LLM configurations ===
GEMMA_CONFIG = evolutionary_types.EvolutionConfig(
    pop_size=4,
    num_generations=5,
    api_type='pytorch_gemma',
    model_name='google/gemma-2b-it',
    embedder_name='all-mpnet-base-v2',
    device='cpu',
    disable_language_model=False,
)

OPENAI_CONFIG = evolutionary_types.EvolutionConfig(
    pop_size=4,
    num_generations=5,
    api_type='openai',
    model_name='gpt-4o-mini',
    embedder_name='all-mpnet-base-v2',
    api_key=None,  # Set via environment variable OPENAI_API_KEY
    disable_language_model=False,
)

# === Measurement channels ===
MEASUREMENT_CHANNELS = {
    'generation_summary': 'evolutionary_generation_summary',
    'population_dynamics': 'evolutionary_population_dynamics',
    'selection_pressure': 'evolutionary_selection_pressure',
    'individual_scores': 'evolutionary_individual_scores',
    'strategy_distribution': 'evolutionary_strategy_distribution',
    'fitness_stats': 'evolutionary_fitness_statistics',
    'mutation_events': 'evolutionary_mutation_events',
    'convergence_metrics': 'evolutionary_convergence_metrics',
}

# === Checkpoint functionality (now in concordia.utils.checkpointing) ===


def _restore_agent_from_checkpoint_data(name: str, agent_data: Dict[str, Any]):
  """Helper to restore agent config from checkpoint data."""
  goal_str = agent_data['goal']
  strategy = (
      Strategy.COOPERATIVE
      if goal_str == Strategy.COOPERATIVE.value
      else Strategy.SELFISH
  )
  return make_agent_config(name, strategy)


def setup_measurements() -> measurements_lib.Measurements:
  """Setup measurement channels for evolutionary tracking."""
  measurements = measurements_lib.Measurements()

  # Initialize all measurement channels
  for channel_name in MEASUREMENT_CHANNELS.values():
    measurements.get_channel(
        channel_name
    )  # Creates channel if it doesn't exist

  return measurements


def make_agent_config(name: str, strategy: Strategy) -> basic_with_plan.Entity:
  """Helper to create a basic agent config with a name and strategy."""
  return basic_with_plan.Entity(
      params={
          'name': name,
          'goal': strategy.value,
      }
  )


def initialize_population(
    config: evolutionary_types.EvolutionConfig,
) -> Dict[str, basic_with_plan.Entity]:
  """Initialize a population with half cooperative, half selfish agents."""
  strategies = [Strategy.COOPERATIVE] * (config.pop_size // 2) + [
      Strategy.SELFISH
  ] * (config.pop_size - config.pop_size // 2)

  return {
      f'Agent_{i+1}': make_agent_config(f'Agent_{i+1}', strategy)
      for i, strategy in enumerate(strategies)
  }


def extract_scores_from_simulation(raw_log: list) -> Dict[str, float]:
  """Extract actual scores from simulation raw log."""
  # Find all "Player Scores" entries in the raw log
  score_entries = helper_functions_lib.find_data_in_nested_structure(
      raw_log, 'Player Scores'
  )

  if not score_entries:
    # Fallback to zero scores if no scores found
    return {}

  # Get the final scores (last entry should have cumulative scores)
  final_scores = score_entries[-1] if score_entries else {}

  # Ensure we return a dict with float values
  return {name: float(score) for name, score in final_scores.items()}


def run_generation(
    agent_configs: Dict[str, basic_with_plan.Entity],
    config: evolutionary_types.EvolutionConfig,
    measurements: Optional[measurements_lib.Measurements] = None,
) -> Dict[str, float]:
  """Run a simulation and return agent scores."""
  gm_key = 'game_master'
  gm_prefab = PublicGoodsGameMaster(
      params={
          'name': 'public_goods_rules',
          # Use default scenes and payoff logic from the prefab
      }
  )
  from concordia.typing import prefab as prefab_lib

  # Setup language model and embedder based on configuration
  model = setup_language_model(config)
  embedder = setup_embedder(config)

  sim_config = prefab_lib.Config(
      instances=[
          *[
              prefab_lib.InstanceConfig(
                  prefab=name,  # Use string key
                  role=prefab_lib.Role.ENTITY,
                  params={k: str(v) for k, v in agent_config.params.items()},
              )
              for name, agent_config in agent_configs.items()
          ],
          prefab_lib.InstanceConfig(
              prefab=gm_key,  # Use string key
              role=prefab_lib.Role.GAME_MASTER,
              params={k: str(v) for k, v in gm_prefab.params.items()},
          ),
      ],
      default_premise=(
          'A public goods game is played among four agents. Each round, agents'
          ' choose whether to contribute to a common pool. The pool is'
          ' multiplied and shared.'
      ),
      default_max_steps=config.num_rounds,
      prefabs={**agent_configs, gm_key: gm_prefab},  # All keys are strings
  )
  
  engine = simultaneous.Simultaneous()
  sim = simulation_generic.Simulation(
      config=sim_config,
      model=model,
      embedder=embedder,
      engine=engine,
  )

  # Create raw_log to capture simulation data
  raw_log = []
  sim.play(raw_log=raw_log)

  # Extract actual scores from the simulation
  scores = extract_scores_from_simulation(raw_log)

  # Fallback to agent names with zero scores if extraction fails
  if not scores:
    scores = {name: 0.0 for name in agent_configs}

  # Log individual scores to measurements
  if measurements is not None:
    for agent_name, score in scores.items():
      measurements.publish_datum(
          MEASUREMENT_CHANNELS['individual_scores'],
          {
              'agent_name': agent_name,
              'score': score,
              'goal': agent_configs[agent_name].params['goal'],
              'timestamp': datetime.datetime.now().isoformat(),
          },
      )

  return scores


def select_survivors(
    agent_configs: Dict[str, basic_with_plan.Entity],
    scores: Dict[str, float],
    method: SelectionMethod,
    k: int,
    measurements: Optional[measurements_lib.Measurements] = None,
) -> Dict[str, basic_with_plan.Entity]:
  """Select survivors using top-k or probabilistic selection."""
  if method == 'topk':
    sorted_agents = sorted(scores.items(), key=itemgetter(1), reverse=True)
    survivors = {name: agent_configs[name] for name, _ in sorted_agents[:k]}
  elif method == 'probabilistic':
    total_score = sum(max(0, s) for s in scores.values())
    if total_score == 0:
      # If all scores are zero, pick randomly
      survivors = dict(random.sample(list(agent_configs.items()), k))
    else:
      # Use random.choices with replacement, then deduplicate
      agent_names = list(agent_configs.keys())
      weights = [max(0, scores[name]) for name in agent_names]

      chosen = set()
      while len(chosen) < k:
        pick = random.choices(agent_names, weights=weights, k=1)[0]
        chosen.add(pick)
      survivors = {name: agent_configs[name] for name in chosen}
  else:
    raise ValueError(f'Unknown selection method: {method}')

  # Log selection pressure metrics
  if measurements is not None:
    survivor_scores = [scores[name] for name in survivors.keys()]
    eliminated_scores = [
        scores[name] for name in agent_configs.keys() if name not in survivors
    ]

    measurements.publish_datum(
        MEASUREMENT_CHANNELS['selection_pressure'],
        {
            'method': method,
            'survivors_count': len(survivors),
            'eliminated_count': len(eliminated_scores),
            'survivor_scores': survivor_scores,
            'eliminated_scores': eliminated_scores,
            'selection_intensity': (
                (max(survivor_scores) - min(eliminated_scores))
                if eliminated_scores
                else 0.0
            ),
            'timestamp': datetime.datetime.now().isoformat(),
        },
    )

  return survivors


def mutate_agents(
    survivors: Dict[str, basic_with_plan.Entity],
    config: evolutionary_types.EvolutionConfig,
    measurements: Optional[measurements_lib.Measurements] = None,
) -> Dict[str, basic_with_plan.Entity]:
  """Create next generation by mutating survivors and cloning to fill population."""
  survivor_names = list(survivors.keys())
  mutation_events = []
  new_agents = {}

  for i in range(config.pop_size):
    parent_name = random.choice(survivor_names)
    parent = survivors[parent_name]

    # Determine current strategy
    original_strategy = (
        Strategy.COOPERATIVE
        if parent.params['goal'] == Strategy.COOPERATIVE.value
        else Strategy.SELFISH
    )
    current_strategy = original_strategy
    mutated = False

    # Apply mutation
    if random.random() < config.mutation_rate:
      current_strategy = (
          Strategy.SELFISH
          if original_strategy == Strategy.COOPERATIVE
          else Strategy.COOPERATIVE
      )
      mutated = True

    name = f'Agent_{i+1}'
    new_agents[name] = make_agent_config(name, current_strategy)

    # Track mutation events
    if mutated:
      mutation_events.append({
          'agent_name': name,
          'parent_name': parent_name,
          'original_strategy': original_strategy.name,
          'mutated_strategy': current_strategy.name,
          'timestamp': datetime.datetime.now().isoformat(),
      })

  # Log mutation events
  if measurements is not None:
    for event in mutation_events:
      measurements.publish_datum(MEASUREMENT_CHANNELS['mutation_events'], event)

    # Log overall mutation statistics
    measurements.publish_datum(
        MEASUREMENT_CHANNELS['population_dynamics'],
        {
            'mutation_rate': config.mutation_rate,
            'mutations_occurred': len(mutation_events),
            'mutation_frequency': len(mutation_events) / config.pop_size,
            'parent_distribution': {
                name: sum(
                    1
                    for event in mutation_events
                    if event['parent_name'] == name
                )
                for name in survivor_names
            },
            'timestamp': datetime.datetime.now().isoformat(),
        },
    )

  return new_agents


def log_generation(
    generation: int,
    agent_configs: Dict[str, basic_with_plan.Entity],
    scores: Dict[str, float],
    measurements: Optional[measurements_lib.Measurements] = None,
):
  """Log generation statistics and publish to measurements."""
  # Group agents by strategy using defaultdict
  strategy_counts = defaultdict(int)
  strategy_scores = defaultdict(list)

  for name, agent in agent_configs.items():
    goal = agent.params['goal']
    strategy_counts[goal] += 1
    if name in scores:
      strategy_scores[goal].append(scores[name])

  coop = strategy_counts[Strategy.COOPERATIVE.value]
  selfish = strategy_counts[Strategy.SELFISH.value]
  total_agents = len(agent_configs)
  cooperation_rate = coop / total_agents if total_agents > 0 else 0.0

  # Calculate fitness statistics
  score_values = list(scores.values())
  avg_score = sum(score_values) / len(score_values) if score_values else 0.0
  max_score = max(score_values) if score_values else 0.0
  min_score = min(score_values) if score_values else 0.0

  # Calculate strategy-specific statistics
  coop_scores = strategy_scores[Strategy.COOPERATIVE.value]
  selfish_scores = strategy_scores[Strategy.SELFISH.value]

  avg_coop_score = sum(coop_scores) / len(coop_scores) if coop_scores else 0.0
  avg_selfish_score = (
      sum(selfish_scores) / len(selfish_scores) if selfish_scores else 0.0
  )

  # Console output
  logger.info('Generation %d:', generation)
  logger.info('  Cooperative: %d, Selfish: %d', coop, selfish)
  logger.info('  Scores: %s', scores)
  logger.info('  Fraction cooperative: %.2f', cooperation_rate)
  logger.info(
      '  Avg scores - Cooperative: %.2f, Selfish: %.2f',
      avg_coop_score,
      avg_selfish_score,
  )

  # Log to measurements
  if measurements is not None:
    # Generation summary
    measurements.publish_datum(
        MEASUREMENT_CHANNELS['generation_summary'],
        {
            'generation': generation,
            'total_agents': total_agents,
            'cooperative_agents': coop,
            'selfish_agents': selfish,
            'cooperation_rate': cooperation_rate,
            'avg_score': avg_score,
            'max_score': max_score,
            'min_score': min_score,
            'timestamp': datetime.datetime.now().isoformat(),
        },
    )

    # Strategy distribution
    measurements.publish_datum(
        MEASUREMENT_CHANNELS['strategy_distribution'],
        {
            'generation': generation,
            'cooperative_count': coop,
            'selfish_count': selfish,
            'cooperative_fraction': cooperation_rate,
            'selfish_fraction': 1.0 - cooperation_rate,
            'timestamp': datetime.datetime.now().isoformat(),
        },
    )

    # Fitness statistics
    measurements.publish_datum(
        MEASUREMENT_CHANNELS['fitness_stats'],
        {
            'generation': generation,
            'avg_cooperative_score': avg_coop_score,
            'avg_selfish_score': avg_selfish_score,
            'score_differential': avg_coop_score - avg_selfish_score,
            'fitness_variance': (
                sum((s - avg_score) ** 2 for s in score_values)
                / len(score_values)
                if score_values
                else 0.0
            ),
            'timestamp': datetime.datetime.now().isoformat(),
        },
    )


def evolutionary_main(
    config: evolutionary_types.EvolutionConfig = DEFAULT_CONFIG,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 5,
    resume_from_checkpoint: bool = False,
) -> measurements_lib.Measurements:
  """Run the evolutionary simulation with measurements tracking and checkpointing.

  Args:
      config: Evolution configuration parameters.
      checkpoint_dir: Directory to save/load checkpoints. If None, no checkpointing.
      checkpoint_interval: Save checkpoint every N generations.
      resume_from_checkpoint: If True, try to resume from latest checkpoint.
  """
  start_generation = 1
  measurements = setup_measurements()
  agent_configs = None

  # Try to resume from checkpoint if requested
  if resume_from_checkpoint and checkpoint_dir:
    latest_checkpoint = checkpointing.find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
      try:
        checkpoint_data = checkpointing.load_evolutionary_checkpoint(
            latest_checkpoint,
            setup_measurements_fn=setup_measurements,
            make_agent_config_fn=_restore_agent_from_checkpoint_data,
            measurement_channels=MEASUREMENT_CHANNELS
        )
        start_generation = checkpoint_data['generation'] + 1  # Resume from next generation
        agent_configs = checkpoint_data['agent_configs']
        measurements = checkpoint_data['measurements']
        logger.info('Resuming from generation %d', start_generation)
      except Exception as e:
        logger.error('Failed to load checkpoint: %s', e)
        logger.info('Starting fresh simulation...')
        agent_configs = None

  # Initialize population if not loaded from checkpoint
  if agent_configs is None:
    agent_configs = initialize_population(config)
    start_generation = 1

  # Log initial population state
  logger.info('Starting evolutionary simulation...')
  logger.info(
      'Population size: %d, Generations: %d',
      config.pop_size,
      config.num_generations,
  )
  logger.info(
      'Selection method: %s, Mutation rate: %.2f',
      config.selection_method,
      config.mutation_rate,
  )
  if checkpoint_dir:
    logger.info(
        'Checkpointing: Every %d generations to %s',
        checkpoint_interval,
        checkpoint_dir,
    )

  # Run evolution
  for generation in range(start_generation, config.num_generations + 1):
    scores = run_generation(agent_configs, config, measurements)
    log_generation(generation, agent_configs, scores, measurements)

    # Save checkpoint if requested
    if checkpoint_dir and generation % checkpoint_interval == 0:
      checkpointing.save_evolutionary_checkpoint(
          generation, agent_configs, measurements, checkpoint_dir, config,
          export_measurements_fn=export_measurements_to_dict
      )

    # Don't select survivors after the last generation
    if generation < config.num_generations:
      survivors = select_survivors(
          agent_configs,
          scores,
          config.selection_method,
          config.top_k,
          measurements,
      )
      agent_configs = mutate_agents(survivors, config, measurements)

  # Log final convergence metrics
  final_coop_rate = (
      sum(
          1
          for a in agent_configs.values()
          if a.params['goal'] == Strategy.COOPERATIVE.value
      )
      / config.pop_size
  )
  measurements.publish_datum(
      MEASUREMENT_CHANNELS['convergence_metrics'],
      {
          'final_generation': config.num_generations,
          'final_cooperation_rate': final_coop_rate,
          'converged_to_cooperation': final_coop_rate > 0.8,
          'converged_to_selfishness': final_coop_rate < 0.2,
          'simulation_parameters': {
              'population_size': config.pop_size,
              'generations': config.num_generations,
              'selection_method': config.selection_method,
              'mutation_rate': config.mutation_rate,
              'top_k': config.top_k,
          },
          'timestamp': datetime.datetime.now().isoformat(),
      },
  )

  # Save final checkpoint
  if checkpoint_dir:
    checkpointing.save_evolutionary_checkpoint(
        config.num_generations,
        agent_configs,
        measurements,
        checkpoint_dir,
        config,
        export_measurements_fn=export_measurements_to_dict
    )

  logger.info('Evolutionary simulation complete.')
  logger.info('Final cooperation rate: %.2f', final_coop_rate)
  logger.info(
      'Available measurement channels: %s',
      list(measurements.available_channels()),
  )

  return measurements


def get_measurement_summary(
    measurements: Optional[measurements_lib.Measurements],
) -> Dict[str, Any]:
  """Get a summary of all measurement data collected during evolution."""
  summary = {}

  if measurements is None:
    return summary

  for channel_key, channel_name in MEASUREMENT_CHANNELS.items():
    channel_data = measurements.get_channel(channel_name)
    summary[channel_key] = {
        'channel_name': channel_name,
        'total_entries': len(channel_data),
        'latest_entry': channel_data[-1] if channel_data else None,
        'sample_entries': (
            channel_data[:3] if len(channel_data) > 3 else channel_data
        ),
    }

  return summary


def export_measurements_to_dict(
    measurements: Optional[measurements_lib.Measurements],
) -> Dict[str, list]:
  """Export all measurement data to a dictionary for analysis."""
  export_data = {}

  if measurements is None:
    return export_data

  for channel_key, channel_name in MEASUREMENT_CHANNELS.items():
    export_data[channel_key] = measurements.get_channel(channel_name)

  return export_data


if __name__ == '__main__':
  # Configure logging
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  )

  # Choose configuration - modify this to test different LLM setups
  logger.info('=== Running Evolutionary Simulation ===')
  
  # Example 1: Default dummy model (fast, no LLM dependencies)
  logger.info('Running with dummy model configuration...')
  measurements = evolutionary_main(config=DEFAULT_CONFIG)
  
  # Example 2: Uncomment to test with Gemma (requires model download)
  # logger.info('Running with Gemma model configuration...')
  # measurements = evolutionary_main(config=GEMMA_CONFIG)
  
  # Example 3: Uncomment to test with OpenAI (requires API key)
  # import os
  # if os.getenv('OPENAI_API_KEY'):
  #   openai_config = OPENAI_CONFIG
  #   openai_config.api_key = os.getenv('OPENAI_API_KEY')
  #   logger.info('Running with OpenAI model configuration...')
  #   measurements = evolutionary_main(config=openai_config)

  # Print measurement summary
  logger.info('=== Measurement Summary ===')
  summary = get_measurement_summary(measurements)
  for channel_key, info in summary.items():
    logger.info('%s: %d entries', channel_key, info['total_entries'])

  # Example: Access specific measurements
  logger.info('=== Example: Final Generation Data ===')
  final_gen_data = measurements.get_last_datum(
      MEASUREMENT_CHANNELS['generation_summary']
  )
  if final_gen_data:
    logger.info(
        'Final cooperation rate: %.2f', final_gen_data['cooperation_rate']
    )
    logger.info('Average score: %.2f', final_gen_data['avg_score'])
    
  logger.info('=== LLM Integration Examples ===')
  logger.info('To use real language models, modify the configuration:')
  logger.info('  - For Gemma: Use GEMMA_CONFIG')
  logger.info('  - For OpenAI: Use OPENAI_CONFIG with API key')
  logger.info('  - Custom: Create your own EvolutionConfig with LLM settings')
