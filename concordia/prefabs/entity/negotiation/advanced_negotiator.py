"""Advanced negotiation agent prefab with modular enhancements."""

from collections.abc import Mapping
import dataclasses
import json

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

# Import base negotiator and advanced components
from concordia.prefabs.entity.negotiation import base_negotiator
from concordia.prefabs.entity.negotiation.components import cultural_adaptation
from concordia.prefabs.entity.negotiation.components import temporal_strategy
from concordia.prefabs.entity.negotiation.components import swarm_intelligence
from concordia.prefabs.entity.negotiation.components import uncertainty_aware
from concordia.prefabs.entity.negotiation.components import strategy_evolution
from concordia.prefabs.entity.negotiation.components import theory_of_mind


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
    """An advanced negotiation agent with optional enhancement modules.

    This prefab extends the base negotiator with advanced capabilities
    that can be enabled/disabled through configuration.

    Available modules:
    - cultural_adaptation: Adapt to different cultural negotiation styles
    - temporal_strategy: Multi-horizon planning and relationship management
    - swarm_intelligence: Collective decision-making through specialized sub-agents
    - uncertainty_aware: Probabilistic reasoning under incomplete information
    - strategy_evolution: Meta-learning and continual adaptation across negotiations
    - theory_of_mind: Emotional intelligence and recursive reasoning
    """

    description: str = (
        'An advanced negotiation agent with modular enhancements. '
        'Supports cultural adaptation, temporal planning, collective '
        'intelligence, and other sophisticated negotiation capabilities.'
    )

    params: Mapping[str, str] = dataclasses.field(default_factory=lambda: {
        'name': 'AdvancedNegotiator',
        'goal': 'Achieve optimal negotiation outcomes',
        'negotiation_style': 'integrative',
        'reservation_value': '0.0',
        'ethical_constraints': 'Be honest and fair. Respect cultural differences.',
        'modules': '',  # Comma-separated list of module names
        'module_configs': '{}',  # JSON string of module configurations
        'extra_components': {},
    })

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the advanced negotiation agent with selected modules.

        Args:
            model: Language model for reasoning
            memory_bank: Memory bank for storing experiences

        Returns:
            Configured advanced negotiation agent
        """
        # Parse module list from params
        modules_str = self.params.get('modules', '')
        modules = [m.strip() for m in modules_str.split(',') if m.strip()]

        # Parse module configs from JSON string
        module_configs_str = self.params.get('module_configs', '{}')
        try:
            module_configs = json.loads(module_configs_str)
        except json.JSONDecodeError:
            module_configs = {}

        # Build extra components for selected modules
        extra_components = {}

        # Add selected modules
        if 'cultural_adaptation' in modules:
            config = module_configs.get('cultural_adaptation', {})
            cultural = cultural_adaptation.CulturalAdaptation(
                model=model,
                own_culture=config.get('own_culture', 'western_business'),
                adaptation_level=config.get('adaptation_level', 0.7),
                detect_culture=config.get('detect_culture', True),
            )
            extra_components['CulturalAdaptation'] = cultural

        if 'temporal_strategy' in modules:
            config = module_configs.get('temporal_strategy', {})
            temporal = temporal_strategy.TemporalStrategy(
                model=model,
                discount_factor=config.get('discount_factor', 0.9),
                reputation_weight=config.get('reputation_weight', 0.3),
                relationship_investment_threshold=config.get('relationship_investment_threshold', 0.6),
            )
            extra_components['TemporalStrategy'] = temporal

        if 'swarm_intelligence' in modules:
            config = module_configs.get('swarm_intelligence', {})
            swarm = swarm_intelligence.SwarmIntelligence(
                model=model,
                consensus_threshold=config.get('consensus_threshold', 0.7),
                max_iterations=config.get('max_iterations', 3),
                enable_sub_agents=config.get('enable_sub_agents', None),
            )
            extra_components['SwarmIntelligence'] = swarm

        if 'uncertainty_aware' in modules:
            config = module_configs.get('uncertainty_aware', {})
            uncertainty = uncertainty_aware.UncertaintyAware(
                model=model,
                confidence_threshold=config.get('confidence_threshold', 0.7),
                risk_tolerance=config.get('risk_tolerance', 0.3),
                information_gathering_budget=config.get('information_gathering_budget', 0.1),
            )
            extra_components['UncertaintyAware'] = uncertainty

        if 'strategy_evolution' in modules:
            config = module_configs.get('strategy_evolution', {})
            evolution = strategy_evolution.StrategyEvolution(
                model=model,
                population_size=config.get('population_size', 20),
                mutation_rate=config.get('mutation_rate', 0.1),
                crossover_rate=config.get('crossover_rate', 0.7),
                learning_rate=config.get('learning_rate', 0.01),
            )
            extra_components['StrategyEvolution'] = evolution

        if 'theory_of_mind' in modules:
            config = module_configs.get('theory_of_mind', {})
            tom = theory_of_mind.TheoryOfMind(
                model=model,
                max_recursion_depth=config.get('max_recursion_depth', 3),
                emotion_sensitivity=config.get('emotion_sensitivity', 0.7),
                empathy_level=config.get('empathy_level', 0.8),
            )
            extra_components['TheoryOfMind'] = tom

        # Update params to include extra components
        enhanced_params = dict(self.params)
        enhanced_params['extra_components'] = extra_components

        # Create base negotiator with enhanced params
        enhanced_prefab = base_negotiator.Entity()
        enhanced_prefab.params = enhanced_params

        return enhanced_prefab.build(model, memory_bank)
