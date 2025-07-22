from collections import defaultdict
from dataclasses import dataclass
import datetime
from enum import Enum
from functools import partial
import json
import logging
from operator import itemgetter
from pathlib import Path
import random
from typing import Any, Dict, List, Literal, Optional, Tuple

from concordia.associative_memory import basic_associative_memory
from concordia.environment.engines import simultaneous
from concordia.language_model import no_language_model
from concordia.prefabs.entity import basic_with_plan
from concordia.prefabs.game_master.public_goods_game_master import PublicGoodsGameMaster
from concordia.prefabs.simulation import generic as simulation_generic
from concordia.utils import helper_functions as helper_functions_lib
from concordia.utils import measurements as measurements_lib

# === Setup logging ===
logger = logging.getLogger(__name__)


# === Enums and Data Classes ===
class Strategy(Enum):
  """Strategy types for agents."""

  COOPERATIVE = 'Maximize group reward in the public goods game'
  SELFISH = 'Maximize personal reward in the public goods game'


SelectionMethod = Literal['topk', 'probabilistic']


@dataclass
class EvolutionConfig:
  """Configuration parameters for evolutionary simulation."""

  pop_size: int = 4
  num_generations: int = 10
  selection_method: SelectionMethod = 'topk'
  top_k: int = 2
  mutation_rate: float = 0.2
  num_rounds: int = 10


@dataclass
class CheckpointData:
  """Data structure for checkpoint loading/saving."""

  generation: int
  agent_configs: Dict[str, basic_with_plan.Entity]
  measurements: measurements_lib.Measurements
  simulation_state: Dict[str, Any]


# === Helper functions for random state serialization ===
def _serialize_random_state(state):
  """Convert random state to JSON-serializable format."""
  return [state[0], list(state[1]), state[2]]


def _deserialize_random_state(state):
  """Convert JSON-serialized random state back to proper format."""
  return (state[0], tuple(state[1]), state[2])


# === Default configuration ===
DEFAULT_CONFIG = EvolutionConfig()

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

# === Checkpoint functionality ===


def save_evolutionary_checkpoint(
    generation: int,
    agent_configs: Dict[str, basic_with_plan.Entity],
    measurements: measurements_lib.Measurements,
    checkpoint_dir: Path,
    config: EvolutionConfig,
    simulation_state: Optional[Dict[str, Any]] = None,
) -> Path:
  """Save evolutionary state to checkpoint file."""
  checkpoint_dir = Path(checkpoint_dir)
  checkpoint_dir.mkdir(exist_ok=True)

  # Create evolutionary checkpoint data
  checkpoint_data = {
      'evolutionary_metadata': {
          'generation': generation,
          'timestamp': datetime.datetime.now().isoformat(),
          'parameters': {
              'population_size': config.pop_size,
              'num_generations': config.num_generations,
              'selection_method': config.selection_method,
              'top_k': config.top_k,
              'mutation_rate': config.mutation_rate,
              'num_rounds': config.num_rounds,
          },
          'random_state': _serialize_random_state(random.getstate()),
      },
      'population': {
          name: {
              'goal': agent.params['goal'],
              'params': agent.params,
          }
          for name, agent in agent_configs.items()
      },
      'measurements': export_measurements_to_dict(measurements),
      'simulation_state': simulation_state or {},
  }

  # Save to file
  checkpoint_file = checkpoint_dir / f'evolutionary_gen_{generation:03d}.json'
  try:
    checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
    logger.info('Evolutionary checkpoint saved: %s', checkpoint_file)
    return checkpoint_file
  except (OSError, json.JSONEncodeError) as e:
    logger.error('Error saving checkpoint: %s', e)
    return Path()


def load_evolutionary_checkpoint(checkpoint_file: Path) -> CheckpointData:
  """Load evolutionary state from checkpoint file."""
  checkpoint_file = Path(checkpoint_file)
  try:
    checkpoint_data = json.loads(checkpoint_file.read_text())

    # Extract evolutionary metadata
    metadata = checkpoint_data['evolutionary_metadata']
    generation = metadata['generation']

    # Restore random state for reproducibility
    if 'random_state' in metadata:
      random_state = _deserialize_random_state(metadata['random_state'])
      random.setstate(random_state)

    # Restore population
    population_data = checkpoint_data['population']
    agent_configs = {}
    for name, agent_data in population_data.items():
      # Convert goal string back to Strategy enum
      goal_str = agent_data['goal']
      strategy = (
          Strategy.COOPERATIVE
          if goal_str == Strategy.COOPERATIVE.value
          else Strategy.SELFISH
      )
      agent_configs[name] = make_agent_config(name, strategy)

    # Restore measurements
    measurements = setup_measurements()
    measurements_data = checkpoint_data.get('measurements', {})
    for channel_key, channel_data in measurements_data.items():
      if channel_key in MEASUREMENT_CHANNELS:
        channel_name = MEASUREMENT_CHANNELS[channel_key]
        for datum in channel_data:
          measurements.publish_datum(channel_name, datum)

    # Get simulation state
    simulation_state = checkpoint_data.get('simulation_state', {})

    logger.info('Evolutionary checkpoint loaded: Generation %d', generation)
    logger.info('Population: %d agents', len(agent_configs))
    logger.info(
        'Measurements: %d entries',
        sum(len(data) for data in measurements_data.values()),
    )

    return CheckpointData(
        generation=generation,
        agent_configs=agent_configs,
        measurements=measurements,
        simulation_state=simulation_state,
    )

  except (OSError, json.JSONDecodeError, KeyError) as e:
    logger.error('Error loading checkpoint: %s', e)
    raise


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
  """Find the latest evolutionary checkpoint in directory."""
  checkpoint_dir = Path(checkpoint_dir)
  if not checkpoint_dir.exists():
    return None

  checkpoint_files = [
      f
      for f in checkpoint_dir.iterdir()
      if f.name.startswith('evolutionary_gen_') and f.name.endswith('.json')
  ]

  if not checkpoint_files:
    return None

  # Sort by generation number
  latest_file = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[2]))

  logger.info('Found latest checkpoint: %s', latest_file)
  return latest_file


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
    config: EvolutionConfig,
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
    config: EvolutionConfig,
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

  config = prefab_lib.Config(
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
  model = no_language_model.NoLanguageModel()  # Use a dummy model for testing
  embedder = lambda text: [random.random() for _ in range(16)]  # Dummy embedder
  engine = simultaneous.Simultaneous()
  sim = simulation_generic.Simulation(
      config=config,
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
    config: EvolutionConfig,
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
    config: EvolutionConfig = DEFAULT_CONFIG,
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
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
      try:
        checkpoint_data = load_evolutionary_checkpoint(latest_checkpoint)
        start_generation = (
            checkpoint_data.generation + 1
        )  # Resume from next generation
        agent_configs = checkpoint_data.agent_configs
        measurements = checkpoint_data.measurements
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
      save_evolutionary_checkpoint(
          generation, agent_configs, measurements, checkpoint_dir, config
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
    save_evolutionary_checkpoint(
        config.num_generations,
        agent_configs,
        measurements,
        checkpoint_dir,
        config,
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

  # Run with default configuration
  measurements = evolutionary_main()

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
