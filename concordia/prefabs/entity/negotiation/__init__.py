"""Negotiation agent prefabs and components for Concordia.

This module provides pre-built negotiation agents with modular cognitive
enhancements.

Quick Start:
    from concordia.prefabs.entity import negotiation

    # Build a basic negotiator
    agent = negotiation.build_agent(model, memory, name='Alice')

    # Build an advanced negotiator with modules
    agent = negotiation.build_advanced_agent(
        model, memory,
        name='Bob',
        modules=['theory_of_mind', 'cultural_adaptation'],
    )

    # Or use enum values for type safety
    from concordia.prefabs.entity.negotiation import ModuleType
    agent = negotiation.build_advanced_agent(
        model, memory,
        modules=[ModuleType.THEORY_OF_MIND, ModuleType.CULTURAL_ADAPTATION],
    )

Available Modules:
    - theory_of_mind: Opponent modeling, emotional intelligence
    - cultural_adaptation: Cultural awareness, communication adaptation
    - temporal_strategy: Multi-horizon planning, deadline handling
    - swarm_intelligence: Collective decision-making via sub-agents
    - uncertainty_aware: Probabilistic reasoning under uncertainty
    - strategy_evolution: Meta-learning across negotiations
"""

from concordia.prefabs.entity.negotiation import advanced_negotiator
from concordia.prefabs.entity.negotiation import base_negotiator
from concordia.prefabs.entity.negotiation.constants import (
    DEFAULT_MODULE_CONFIGS,
    MODULE_COMPONENT_NAMES,
    ModuleType,
)
from concordia.prefabs.entity.negotiation.config import (
    AlgorithmConfig,
    DeceptionDetectionConfig,
    EvaluationConfig,
    InterpretabilityConfig,
    ModuleDefaults,
    OutcomeConfig,
    ParsingConfig,
    RelationshipConfig,
    StrategyConfig,
    TheoryOfMindConfig,
)

# Convenience aliases for common operations
build_agent = base_negotiator.build_agent
build_advanced_agent = advanced_negotiator.build_agent

# Prefab dataclasses for Entity pattern
BaseNegotiator = base_negotiator.Entity
AdvancedNegotiator = advanced_negotiator.Entity

__all__ = [
    # Modules
    'base_negotiator',
    'advanced_negotiator',
    # Builder functions
    'build_agent',
    'build_advanced_agent',
    # Prefab classes
    'BaseNegotiator',
    'AdvancedNegotiator',
    # Constants
    'ModuleType',
    'MODULE_COMPONENT_NAMES',
    'DEFAULT_MODULE_CONFIGS',
    # Configuration
    'StrategyConfig',
    'OutcomeConfig',
    'AlgorithmConfig',
    'ModuleDefaults',
    'TheoryOfMindConfig',
    'DeceptionDetectionConfig',
    'EvaluationConfig',
    'InterpretabilityConfig',
    'RelationshipConfig',
    'ParsingConfig',
]
