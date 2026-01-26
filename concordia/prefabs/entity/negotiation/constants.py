"""Constants and enums for the negotiation module."""

from enum import Enum


class ModuleType(str, Enum):
  """Available cognitive enhancement modules for negotiation agents.

  Each module adds specific capabilities to the agent:
  - THEORY_OF_MIND: Opponent modeling, emotional intelligence, recursive beliefs
  - CULTURAL_ADAPTATION: Cultural awareness and communication style adaptation
  - TEMPORAL_STRATEGY: Multi-horizon planning, deadline handling, relationships
  - SWARM_INTELLIGENCE: Collective decision-making through sub-agents
  - UNCERTAINTY_AWARE: Probabilistic reasoning under incomplete information
  - STRATEGY_EVOLUTION: Meta-learning across negotiations, adaptive strategies

  Usage:
      from concordia.prefabs.entity.negotiation.constants import ModuleType

      # Use enum values
      modules = [ModuleType.THEORY_OF_MIND, ModuleType.CULTURAL_ADAPTATION]

      # String conversion works automatically
      if ModuleType.THEORY_OF_MIND in modules:
          ...

      # Can also use string values for backward compatibility
      modules = ['theory_of_mind', 'cultural_adaptation']
  """

  THEORY_OF_MIND = 'theory_of_mind'
  CULTURAL_ADAPTATION = 'cultural_adaptation'
  TEMPORAL_STRATEGY = 'temporal_strategy'
  SWARM_INTELLIGENCE = 'swarm_intelligence'
  UNCERTAINTY_AWARE = 'uncertainty_aware'
  STRATEGY_EVOLUTION = 'strategy_evolution'

  @classmethod
  def all_modules(cls) -> list['ModuleType']:
    """Return all available module types."""
    return list(cls)

  @classmethod
  def from_string(cls, value: str) -> 'ModuleType':
    """Convert string to ModuleType, case-insensitive.

    Args:
        value: Module name as string

    Returns:
        Corresponding ModuleType enum

    Raises:
        ValueError: If value doesn't match any module type
    """
    value_lower = value.lower().strip()
    for module in cls:
      if module.value == value_lower:
        return module
    valid = [m.value for m in cls]
    raise ValueError(f"Unknown module type: {value}. Valid types: {valid}")


# Mapping of module types to their component class names
MODULE_COMPONENT_NAMES = {
    ModuleType.THEORY_OF_MIND: 'TheoryOfMind',
    ModuleType.CULTURAL_ADAPTATION: 'CulturalAdaptation',
    ModuleType.TEMPORAL_STRATEGY: 'TemporalStrategy',
    ModuleType.SWARM_INTELLIGENCE: 'SwarmIntelligence',
    ModuleType.UNCERTAINTY_AWARE: 'UncertaintyAwareNegotiator',
    ModuleType.STRATEGY_EVOLUTION: 'StrategyEvolution',
}


# Default configurations for each module
DEFAULT_MODULE_CONFIGS = {
    ModuleType.THEORY_OF_MIND: {
        'max_recursion_depth': 3,
        'emotion_sensitivity': 0.7,
        'empathy_level': 0.8,
    },
    ModuleType.CULTURAL_ADAPTATION: {
        'own_culture': 'western_business',
        'adaptation_level': 0.7,
        'detect_culture': True,
    },
    ModuleType.TEMPORAL_STRATEGY: {
        'discount_factor': 0.9,
        'reputation_weight': 0.3,
        'relationship_investment_threshold': 0.6,
    },
    ModuleType.SWARM_INTELLIGENCE: {
        'consensus_threshold': 0.7,
        'max_iterations': 3,
    },
    ModuleType.UNCERTAINTY_AWARE: {
        'confidence_threshold': 0.7,
        'risk_tolerance': 0.5,
        'ambiguity_aversion': 0.3,
    },
    ModuleType.STRATEGY_EVOLUTION: {
        'population_size': 5,
        'learning_rate': 0.1,
        'strategy_memory': 10,
    },
}
