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
from concordia.prefabs.entity.negotiation.components import uncertain_buyer
from concordia.prefabs.entity.negotiation.components import uncertain_seller


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

        if 'uncertain_buyer' in modules:
            config = module_configs.get('uncertain_buyer', {})
            uncertain_buyer_comp = uncertain_buyer.UncertainBuyer(
                model=model,
                confidence_threshold=config.get('confidence_threshold', 0.7),
                risk_tolerance=config.get('risk_tolerance', 0.3),
                preferences=config.get('preferences', {}),
                information_gathering_budget=config.get('information_gathering_budget', 0.1),
                own_reservation_=config.get('own_reservation_', 500000),
                own_reservation_std=config.get('own_reservation_std', 5000),
                mu=config.get('cp_reservation', 0),
                lambda_=config.get('lambda_', 1),
                a=config.get('a', 1),
                b=config.get('b', 1),
            )
            extra_components['UncertainBuyer'] = uncertain_buyer_comp

        if 'uncertain_seller' in modules:
            config = module_configs.get('uncertain_seller', {})
            uncertain_seller_comp = uncertain_seller.UncertainSeller(
                model=model,
                confidence_threshold=config.get('confidence_threshold', 0.7),
                risk_tolerance=config.get('risk_tolerance', 0.3),
                information_gathering_budget=config.get('information_gathering_budget', 0.1),
                own_reservation_=config.get('own_reservation_', 600000),
                mu=config.get('cp_reservation', 0),
                lambda_=config.get('lambda_', 1),
                a=config.get('a', 1),
                b=config.get('b', 1),
            )
            extra_components['UncertainSeller'] = uncertain_seller_comp

        # Update params to include extra components
        enhanced_params = dict(self.params)
        enhanced_params['extra_components'] = extra_components

        # Create base negotiator with enhanced params
        enhanced_prefab = base_negotiator.Entity()
        enhanced_prefab.params = enhanced_params

        return enhanced_prefab.build(model, memory_bank)


def build_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    name: str = 'AdvancedNegotiator',
    goal: str = 'Achieve optimal negotiation outcomes',
    negotiation_style: str = 'integrative',
    reservation_value: float = 0.0,
    ethical_constraints: str = 'Be honest and fair. Respect cultural differences.',
    modules: list = None,
    module_configs: dict = None,
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Convenience function to build an advanced negotiation agent.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        name: Name of the negotiation agent
        goal: Primary negotiation goal
        negotiation_style: Style of negotiation ('coopenegotrative', 'competitive', 'integrative')
        reservation_value: Minimum acceptable value
        ethical_constraints: Ethical guidelines for negotiation
        modules: List of module names to enable (e.g., ['cultural_adaptation', 'theory_of_mind'])
        module_configs: Dictionary of module-specific configurations
        **kwargs: Additional parameters for the agent
        
    Returns:
        Configured advanced negotiation agent
        
    Available modules:
        - 'cultural_adaptation': Adapt to different cultural negotiation styles
        - 'temporal_strategy': Multi-horizon planning and relationship management
        - 'swarm_intelligence': Collective decision-making through specialized sub-agents
        - 'uncertainty_aware': Probabilistic reasoning under incomplete information
        - 'strategy_evolution': Meta-learning and continual adaptation
        - 'theory_of_mind': Emotional intelligence and recursive reasoning
        
    Example:
        ```python
        agent = build_agent(
            model=my_model,
            memory_bank=my_memory,
            name="Sophie",
            goal="Negotiate international trade agreement",
            modules=['cultural_adaptation', 'theory_of_mind'],
            module_configs={
                'cultural_adaptation': {'own_culture': 'western_business'},
                'theory_of_mind': {'max_recursion_depth': 2}
            }
        )
        ```
    """
    if modules is None:
        modules = []
    if module_configs is None:
        module_configs = {}
    
    params = {
        'name': name,
        'goal': goal,
        'negotiation_style': negotiation_style,
        'reservation_value': str(reservation_value),
        'ethical_constraints': ethical_constraints,
        'modules': ','.join(modules),
        'module_configs': json.dumps(module_configs),
    }
    params.update(kwargs)
    
    prefab = Entity(params=params)
    return prefab.build(model=model, memory_bank=memory_bank)


def build_cultural_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    name: str = 'CulturalNegotiator',
    own_culture: str = 'western_business',
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent optimized for cross-cultural negotiations.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        name: Name of the negotiation agent
        own_culture: Agent's cultural background ('western_business', 'east_asian', etc.)
        **kwargs: Additional parameters for build_agent
        
    Returns:
        Agent with cultural adaptation capabilities
    """
    return build_agent(
        model=model,
        memory_bank=memory_bank,
        name=name,
        modules=['cultural_adaptation', 'theory_of_mind'],
        module_configs={
            'cultural_adaptation': {'own_culture': own_culture},
            'theory_of_mind': {'emotion_sensitivity': 0.8}
        },
        **kwargs
    )


def build_temporal_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    name: str = 'TemporalNegotiator',
    discount_factor: float = 0.9,
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent optimized for long-term relationship management.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        name: Name of the negotiation agent
        discount_factor: How much to value future outcomes (0-1)
        **kwargs: Additional parameters for build_agent
        
    Returns:
        Agent with temporal strategy capabilities
    """
    return build_agent(
        model=model,
        memory_bank=memory_bank,
        name=name,
        modules=['temporal_strategy', 'theory_of_mind'],
        module_configs={
            'temporal_strategy': {'discount_factor': discount_factor},
        },
        **kwargs
    )


def build_collective_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    name: str = 'CollectiveNegotiator',
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent optimized for multi-party negotiations.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        name: Name of the negotiation agent
        **kwargs: Additional parameters for build_agent
        
    Returns:
        Agent with swarm intelligence capabilities
    """
    return build_agent(
        model=model,
        memory_bank=memory_bank,
        name=name,
        modules=['swarm_intelligence', 'uncertainty_aware'],
        module_configs={
            'swarm_intelligence': {'consensus_threshold': 0.7},
        },
        **kwargs
    )


def build_adaptive_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    name: str = 'AdaptiveNegotiator',
    learning_rate: float = 0.01,
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent that learns and adapts strategies over time.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        name: Name of the negotiation agent
        learning_rate: How quickly to adapt strategies (0-1)
        **kwargs: Additional parameters for build_agent
        
    Returns:
        Agent with strategy evolution capabilities
    """
    return build_agent(
        model=model,
        memory_bank=memory_bank,
        name=name,
        modules=['strategy_evolution', 'uncertainty_aware'],
        module_configs={
            'strategy_evolution': {'learning_rate': learning_rate},
        },
        **kwargs
    )
