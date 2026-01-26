"""Integration framework for coordinating negotiation modules."""

import dataclasses
from typing import Any, Dict, List, Optional, Set
import logging

from concordia.prefabs.entity.negotiation.components import (
    cultural_adaptation,
    temporal_strategy,
    swarm_intelligence,
    uncertainty_aware,
    strategy_evolution,
    theory_of_mind
)


@dataclasses.dataclass
class ModuleConfig:
    """Configuration for a negotiation module."""
    enabled: bool
    priority: int  # Higher priority modules process first
    dependencies: Set[str] = dataclasses.field(default_factory=set)
    config_params: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModuleInteraction:
    """Represents interaction between modules."""
    source_module: str
    target_module: str
    interaction_type: str
    data_flow: Dict[str, Any]
    bidirectional: bool = False


class NegotiationModuleIntegrator:
    """Coordinates interactions between negotiation modules."""

    # Define module interaction protocols
    INTERACTION_PROTOCOLS = {
        ('theory_of_mind', 'cultural_adaptation'): {
            'type': 'emotional_cultural_context',
            'data': ['emotional_state', 'cultural_indicators'],
            'bidirectional': True
        },
        ('uncertainty_aware', 'swarm_intelligence'): {
            'type': 'collective_uncertainty_assessment',
            'data': ['belief_distributions', 'expert_confidence'],
            'bidirectional': True
        },
        ('temporal_strategy', 'strategy_evolution'): {
            'type': 'learning_from_relationships',
            'data': ['relationship_outcomes', 'temporal_patterns'],
            'bidirectional': True
        },
        ('theory_of_mind', 'swarm_intelligence'): {
            'type': 'collective_emotional_intelligence',
            'data': ['emotional_assessments', 'mental_models'],
            'bidirectional': False
        },
        ('cultural_adaptation', 'temporal_strategy'): {
            'type': 'cultural_relationship_dynamics',
            'data': ['cultural_time_orientation', 'relationship_norms'],
            'bidirectional': True
        },
        ('uncertainty_aware', 'strategy_evolution'): {
            'type': 'learning_under_uncertainty',
            'data': ['confidence_levels', 'information_gaps'],
            'bidirectional': True
        }
    }

    # Module processing order based on dependencies
    MODULE_DEPENDENCIES = {
        'theory_of_mind': set(),  # Base module - no dependencies
        'cultural_adaptation': set(),  # Base module
        'uncertainty_aware': set(),  # Base module
        'temporal_strategy': {'theory_of_mind'},  # Benefits from emotional understanding
        'swarm_intelligence': {'theory_of_mind', 'uncertainty_aware'},  # Needs both
        'strategy_evolution': {'temporal_strategy', 'uncertainty_aware'}  # Learns from both
    }

    def __init__(self):
        """Initialize the module integrator."""
        self.modules: Dict[str, ModuleConfig] = {}
        self.active_interactions: List[ModuleInteraction] = []
        self.logger = logging.getLogger(__name__)

    def register_module(self, module_name: str, config: ModuleConfig):
        """Register a module with its configuration."""
        self.modules[module_name] = config
        self.logger.info(f"Registered module: {module_name} with priority {config.priority}")

    def validate_configuration(self) -> List[str]:
        """Validate module configuration and dependencies."""
        issues = []

        # Check dependencies
        for module_name, config in self.modules.items():
            if not config.enabled:
                continue

            # Check if dependencies are satisfied
            required_deps = self.MODULE_DEPENDENCIES.get(module_name, set())
            for dep in required_deps:
                if dep not in self.modules or not self.modules[dep].enabled:
                    issues.append(f"{module_name} requires {dep} to be enabled")

        # Check for circular dependencies
        if self._has_circular_dependencies():
            issues.append("Circular dependencies detected in module configuration")

        return issues

    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in modules."""
        visited = set()
        rec_stack = set()

        def _visit(module: str) -> bool:
            visited.add(module)
            rec_stack.add(module)

            for dep in self.MODULE_DEPENDENCIES.get(module, set()):
                if dep not in visited:
                    if _visit(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(module)
            return False

        for module in self.modules:
            if module not in visited:
                if _visit(module):
                    return True

        return False

    def get_processing_order(self) -> List[str]:
        """Get optimal processing order based on dependencies and priorities."""
        enabled_modules = [m for m, c in self.modules.items() if c.enabled]

        # Topological sort based on dependencies
        sorted_modules = []
        visited = set()

        def _visit(module: str):
            if module in visited:
                return
            visited.add(module)

            # Visit dependencies first
            for dep in self.MODULE_DEPENDENCIES.get(module, set()):
                if dep in enabled_modules:
                    _visit(dep)

            sorted_modules.append(module)

        for module in enabled_modules:
            _visit(module)

        # Secondary sort by priority (stable sort)
        sorted_modules.sort(key=lambda m: self.modules[m].priority, reverse=True)

        return sorted_modules

    def identify_interactions(self) -> List[ModuleInteraction]:
        """Identify potential interactions between enabled modules."""
        interactions = []
        enabled_modules = [m for m, c in self.modules.items() if c.enabled]

        for module1 in enabled_modules:
            for module2 in enabled_modules:
                if module1 == module2:
                    continue

                # Check if interaction is defined
                interaction_key = (module1, module2)
                if interaction_key in self.INTERACTION_PROTOCOLS:
                    protocol = self.INTERACTION_PROTOCOLS[interaction_key]
                    interaction = ModuleInteraction(
                        source_module=module1,
                        target_module=module2,
                        interaction_type=protocol['type'],
                        data_flow={d: None for d in protocol['data']},
                        bidirectional=protocol['bidirectional']
                    )
                    interactions.append(interaction)

        return interactions

    def create_integration_report(self) -> Dict[str, Any]:
        """Create a comprehensive integration report."""
        enabled_modules = [m for m, c in self.modules.items() if c.enabled]
        processing_order = self.get_processing_order()
        interactions = self.identify_interactions()

        report = {
            'enabled_modules': enabled_modules,
            'processing_order': processing_order,
            'total_interactions': len(interactions),
            'interaction_types': {},
            'module_details': {},
            'dependency_graph': {},
            'synergy_score': 0.0
        }

        # Count interaction types
        for interaction in interactions:
            itype = interaction.interaction_type
            report['interaction_types'][itype] = report['interaction_types'].get(itype, 0) + 1

        # Module details
        for module, config in self.modules.items():
            if config.enabled:
                report['module_details'][module] = {
                    'priority': config.priority,
                    'dependencies': list(self.MODULE_DEPENDENCIES.get(module, set())),
                    'incoming_interactions': len([i for i in interactions if i.target_module == module]),
                    'outgoing_interactions': len([i for i in interactions if i.source_module == module])
                }

        # Build dependency graph
        for module in enabled_modules:
            deps = self.MODULE_DEPENDENCIES.get(module, set())
            report['dependency_graph'][module] = list(deps.intersection(enabled_modules))

        # Calculate synergy score (more interactions = more synergy)
        max_possible_interactions = len(enabled_modules) * (len(enabled_modules) - 1)
        if max_possible_interactions > 0:
            report['synergy_score'] = len(interactions) / max_possible_interactions

        return report


class ModuleCoordinator:
    """Coordinates data flow and decision making between modules."""

    def __init__(self, integrator: NegotiationModuleIntegrator):
        """Initialize the coordinator."""
        self.integrator = integrator
        self.shared_context: Dict[str, Any] = {}
        self.module_outputs: Dict[str, Any] = {}

    def update_shared_context(self, key: str, value: Any, source_module: str):
        """Update shared context that modules can access."""
        self.shared_context[key] = {
            'value': value,
            'source': source_module,
            'timestamp': 'current'
        }

    def get_shared_context(self, key: str) -> Optional[Any]:
        """Get value from shared context."""
        context_item = self.shared_context.get(key)
        return context_item['value'] if context_item else None

    def coordinate_modules(self, modules: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate module execution and data sharing."""
        processing_order = self.integrator.get_processing_order()

        # Process modules in order
        for module_name in processing_order:
            if module_name not in modules:
                continue

            module = modules[module_name]

            # Get module input from shared context
            module_input = self._prepare_module_input(module_name)

            # Module processing happens through Concordia's component system
            # Store any outputs for other modules
            self.module_outputs[module_name] = {
                'processed': True,
                'input_context': module_input
            }

            # Update shared context based on module type
            self._update_context_from_module(module_name, module)

        return self.module_outputs

    def _prepare_module_input(self, module_name: str) -> Dict[str, Any]:
        """Prepare input for a module based on its dependencies."""
        module_input = {}

        # Get data from dependencies
        dependencies = self.integrator.MODULE_DEPENDENCIES.get(module_name, set())
        for dep in dependencies:
            if dep in self.module_outputs:
                module_input[f"{dep}_output"] = self.module_outputs[dep]

        # Add relevant shared context
        if module_name == 'cultural_adaptation':
            module_input['cultural_indicators'] = self.get_shared_context('cultural_indicators')
        elif module_name == 'theory_of_mind':
            module_input['emotional_context'] = self.get_shared_context('emotional_context')
        elif module_name == 'uncertainty_aware':
            module_input['belief_state'] = self.get_shared_context('belief_state')

        return module_input

    def _update_context_from_module(self, module_name: str, module: Any):
        """Update shared context based on module outputs."""
        if module_name == 'theory_of_mind' and hasattr(module, '_mental_models'):
            self.update_shared_context('mental_models', module._mental_models, module_name)
        elif module_name == 'cultural_adaptation' and hasattr(module, '_cultural_profile'):
            self.update_shared_context('cultural_profile', module._cultural_profile, module_name)
        elif module_name == 'uncertainty_aware' and hasattr(module, '_beliefs'):
            self.update_shared_context('belief_state', module._beliefs, module_name)
        elif module_name == 'temporal_strategy' and hasattr(module, '_relationships'):
            self.update_shared_context('relationship_state', module._relationships, module_name)
        elif module_name == 'swarm_intelligence' and hasattr(module, '_last_analyses'):
            self.update_shared_context('collective_analysis', module._last_analyses, module_name)
        elif module_name == 'strategy_evolution' and hasattr(module, '_current_strategy'):
            self.update_shared_context('current_strategy', module._current_strategy, module_name)


def create_optimal_module_configuration(
    available_modules: List[str],
    negotiation_context: Dict[str, Any]
) -> Dict[str, ModuleConfig]:
    """Create optimal module configuration based on negotiation context."""
    configurations = {}

    # Always enable theory of mind for social intelligence
    if 'theory_of_mind' in available_modules:
        configurations['theory_of_mind'] = ModuleConfig(
            enabled=True,
            priority=90,  # High priority - foundational
            config_params={
                'max_recursion_depth': 3 if negotiation_context.get('complexity', 0.5) > 0.7 else 2,
                'emotion_sensitivity': 0.8,
                'empathy_level': 0.7 if negotiation_context.get('competitive', False) else 0.9
            }
        )

    # Cultural adaptation for cross-cultural contexts
    if 'cultural_adaptation' in available_modules and negotiation_context.get('cross_cultural', False):
        configurations['cultural_adaptation'] = ModuleConfig(
            enabled=True,
            priority=85,
            config_params={
                'adaptation_level': 0.8,
                'detect_culture': True
            }
        )

    # Uncertainty handling for incomplete information
    if 'uncertainty_aware' in available_modules and negotiation_context.get('information_completeness', 1.0) < 0.7:
        configurations['uncertainty_aware'] = ModuleConfig(
            enabled=True,
            priority=80,
            config_params={
                'confidence_threshold': 0.7,
                'risk_tolerance': negotiation_context.get('risk_tolerance', 0.3)
            }
        )

    # Temporal strategy for repeated interactions
    if 'temporal_strategy' in available_modules and negotiation_context.get('repeated_interaction', False):
        configurations['temporal_strategy'] = ModuleConfig(
            enabled=True,
            priority=75,
            dependencies={'theory_of_mind'},
            config_params={
                'discount_factor': 0.9,
                'reputation_weight': 0.4
            }
        )

    # Swarm intelligence for complex multi-issue negotiations
    if 'swarm_intelligence' in available_modules and negotiation_context.get('issue_count', 1) > 3:
        configurations['swarm_intelligence'] = ModuleConfig(
            enabled=True,
            priority=70,
            dependencies={'theory_of_mind', 'uncertainty_aware'},
            config_params={
                'consensus_threshold': 0.75,
                'enable_sub_agents': ['market_analysis', 'emotional_intelligence', 'game_theory', 'diplomatic_relations']
            }
        )

    # Strategy evolution for learning scenarios
    if 'strategy_evolution' in available_modules and negotiation_context.get('learning_enabled', True):
        configurations['strategy_evolution'] = ModuleConfig(
            enabled=True,
            priority=65,
            dependencies={'temporal_strategy', 'uncertainty_aware'},
            config_params={
                'population_size': 20,
                'mutation_rate': 0.1,
                'learning_rate': 0.02
            }
        )

    return configurations
