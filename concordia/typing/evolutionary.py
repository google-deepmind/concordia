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

"""Type definitions for evolutionary simulations."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Literal

from concordia.utils import measurements as measurements_lib


class Strategy(Enum):
  """Strategy types for evolutionary agents."""

  COOPERATIVE = 'Maximize group reward in the public goods game'
  SELFISH = 'Maximize personal reward in the public goods game'


# Type alias for selection methods
SelectionMethod = Literal['topk', 'probabilistic']


@dataclass
class EvolutionConfig:
  """Configuration parameters for evolutionary simulation.
  
  Attributes:
    pop_size: Number of agents in the population
    num_generations: Number of generations to simulate
    selection_method: Method for selecting survivors ('topk' or 'probabilistic')
    top_k: Number of survivors for top-k selection
    mutation_rate: Probability of strategy mutation per agent per generation
    num_rounds: Number of rounds per simulation game
    api_type: Language model API type ('pytorch_gemma', 'openai', etc.)
    model_name: Language model name (e.g., 'google/gemma-2b-it', 'gpt-4o')
    embedder_name: Sentence transformer model name (e.g., 'all-mpnet-base-v2')
    device: Device for local models ('cpu', 'cuda:0', etc.)
    api_key: Optional API key for cloud-based models
    disable_language_model: Use dummy model instead of real LLM
  """

  pop_size: int = 4
  num_generations: int = 10
  selection_method: SelectionMethod = 'topk'
  top_k: int = 2
  mutation_rate: float = 0.2
  num_rounds: int = 10
  
  # Language model configuration
  api_type: str = 'pytorch_gemma'
  model_name: str = 'google/gemma-2b-it'
  embedder_name: str = 'all-mpnet-base-v2'
  device: str = 'cpu'
  api_key: str | None = None
  disable_language_model: bool = False


@dataclass
class CheckpointData:
  """Data structure for evolutionary checkpoint loading/saving.
  
  This represents the complete state needed to resume an evolutionary
  simulation from a specific generation.
  
  Attributes:
    generation: The generation number this checkpoint represents
    agent_configs: Dictionary mapping agent names to their configurations
    measurements: Measurements object containing collected data
    simulation_state: Additional simulation state data
  """

  generation: int
  agent_configs: Dict[str, Any]  # Using Any to avoid circular imports
  measurements: measurements_lib.Measurements
  simulation_state: Dict[str, Any]


# Type aliases for common evolutionary data structures
AgentConfigDict = Dict[str, Any]
ScoreDict = Dict[str, float]
MeasurementChannels = Dict[str, str]


@dataclass 
class EvolutionResults:
  """Results from an evolutionary simulation run.
  
  Attributes:
    final_generation: The final generation number
    final_cooperation_rate: Cooperation rate in the final generation
    measurements: Complete measurements from the simulation
    converged_to_cooperation: Whether population converged to cooperation
    converged_to_selfishness: Whether population converged to selfishness
  """
  
  final_generation: int
  final_cooperation_rate: float
  measurements: measurements_lib.Measurements
  converged_to_cooperation: bool
  converged_to_selfishness: bool