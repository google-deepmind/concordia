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

"""Utilities for checkpointing evolutionary simulations."""

import datetime
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

from concordia.typing import evolutionary as evolutionary_types
from concordia.utils import measurements as measurements_lib

logger = logging.getLogger(__name__)


def serialize_random_state(state) -> list:
  """Convert random state to JSON-serializable format."""
  return [state[0], list(state[1]), state[2]]


def deserialize_random_state(state) -> tuple:
  """Convert JSON-serialized random state back to proper format."""
  return (state[0], tuple(state[1]), state[2])


def save_evolutionary_checkpoint(
    generation: int,
    agent_configs: evolutionary_types.AgentConfigDict,
    measurements: measurements_lib.Measurements,
    checkpoint_dir: Path,
    config: evolutionary_types.EvolutionConfig,
    simulation_state: Optional[Dict[str, Any]] = None,
    export_measurements_fn: Optional[callable] = None,
) -> Path:
  """Save evolutionary state to checkpoint file.
  
  Args:
    generation: Current generation number
    agent_configs: Dictionary of agent configurations
    measurements: Measurements object to save
    checkpoint_dir: Directory to save checkpoint files
    config: Evolution configuration object
    simulation_state: Optional additional simulation state
    export_measurements_fn: Function to export measurements to dict
    
  Returns:
    Path to the saved checkpoint file
  """
  checkpoint_dir = Path(checkpoint_dir)
  checkpoint_dir.mkdir(exist_ok=True)

  # Export measurements using provided function or default to empty dict
  measurements_data = {}
  if export_measurements_fn and measurements:
    measurements_data = export_measurements_fn(measurements)

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
          'random_state': serialize_random_state(random.getstate()),
      },
      'population': {
          name: {
              'goal': agent.params['goal'],
              'params': agent.params,
          }
          for name, agent in agent_configs.items()
      },
      'measurements': measurements_data,
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


def load_evolutionary_checkpoint(
    checkpoint_file: Path,
    setup_measurements_fn: Optional[callable] = None,
    make_agent_config_fn: Optional[callable] = None,
    measurement_channels: Optional[evolutionary_types.MeasurementChannels] = None,
) -> Dict[str, Any]:
  """Load evolutionary state from checkpoint file.
  
  Args:
    checkpoint_file: Path to checkpoint file
    setup_measurements_fn: Function to setup measurements object
    make_agent_config_fn: Function to create agent configs
    measurement_channels: Dictionary mapping channel keys to names
    
  Returns:
    Dictionary containing loaded checkpoint data
  """
  checkpoint_file = Path(checkpoint_file)
  try:
    checkpoint_data = json.loads(checkpoint_file.read_text())

    # Extract evolutionary metadata
    metadata = checkpoint_data['evolutionary_metadata']
    generation = metadata['generation']

    # Restore random state for reproducibility
    if 'random_state' in metadata:
      random_state = deserialize_random_state(metadata['random_state'])
      random.setstate(random_state)

    # Restore population
    population_data = checkpoint_data['population']
    agent_configs = {}
    if make_agent_config_fn:
      for name, agent_data in population_data.items():
        agent_configs[name] = make_agent_config_fn(name, agent_data)
    else:
      agent_configs = population_data

    # Restore measurements
    measurements = None
    if setup_measurements_fn:
      measurements = setup_measurements_fn()
      measurements_data = checkpoint_data.get('measurements', {})
      if measurement_channels:
        for channel_key, channel_data in measurements_data.items():
          if channel_key in measurement_channels:
            channel_name = measurement_channels[channel_key]
            for datum in channel_data:
              measurements.publish_datum(channel_name, datum)

    # Get simulation state
    simulation_state = checkpoint_data.get('simulation_state', {})

    logger.info('Evolutionary checkpoint loaded: Generation %d', generation)
    logger.info('Population: %d agents', len(agent_configs))
    if measurements_data:
      logger.info(
          'Measurements: %d entries',
          sum(len(data) for data in measurements_data.values()),
      )

    return {
        'generation': generation,
        'agent_configs': agent_configs,
        'measurements': measurements,
        'simulation_state': simulation_state,
    }

  except (OSError, json.JSONDecodeError, KeyError) as e:
    logger.error('Error loading checkpoint: %s', e)
    raise


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
  """Find the latest evolutionary checkpoint in directory.
  
  Args:
    checkpoint_dir: Directory to search for checkpoint files
    
  Returns:
    Path to the latest checkpoint file, or None if none found
  """
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