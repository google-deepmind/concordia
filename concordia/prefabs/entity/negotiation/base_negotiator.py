"""Base negotiation agent prefab with core negotiation capabilities."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

# Import our negotiation components
from concordia.prefabs.entity.negotiation.components import negotiation_instructions
from concordia.prefabs.entity.negotiation.components import negotiation_memory
from concordia.prefabs.entity.negotiation.components import negotiation_strategy


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
    """A basic negotiation agent with core negotiation capabilities.

    This prefab creates an agent that can engage in negotiations with:
    - Clear negotiation goals and constraints
    - Memory of offers and counteroffers
    - Basic negotiation strategies (cooperative, competitive, integrative)
    - Ethical guidelines for fair negotiation
    """
    role: prefab_lib.Role = prefab_lib.Role.ENTITY

    description: str = (
        'A negotiation agent with core capabilities for engaging in '
        'value-based negotiations. Supports different negotiation styles '
        'and maintains memory of negotiation history.'
    )

    params: Mapping[str, str] = dataclasses.field(default_factory=lambda: {
        'name': 'Negotiator',
        'goal': 'Reach a mutually beneficial agreement',
        'negotiation_style': 'integrative',
        'reservation_value': '0.0',
        'ethical_constraints': 'Be honest and fair. Do not deceive or manipulate.',
        'extra_components': {},
    })

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the base negotiation agent.

        Args:
            model: Language model for reasoning
            memory_bank: Memory bank for storing experiences

        Returns:
            Configured negotiation agent
        """
        # Extract parameters from params dict
        agent_name = self.params.get('name', 'Negotiator')
        goal = self.params.get('goal', 'Reach a mutually beneficial agreement')
        style = self.params.get('negotiation_style', 'integrative')
        reservation = float(self.params.get('reservation_value', '0.0'))
        ethics = self.params.get('ethical_constraints', 'Be honest and fair.')

        # Create memory component
        memory = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank,
        )

        # Create observation component
        observation_to_memory = agent_components.observation.ObservationToMemory()

        # Create observation retrieval component
        observation = agent_components.observation.LastNObservations(
            history_length=100,
            pre_act_label='Recent events in the negotiation:'
        )

        # Create negotiation-specific instructions
        instructions = negotiation_instructions.NegotiationInstructions(
            agent_name=agent_name,
            goal=goal,
            negotiation_style=style,
            reservation_value=reservation,
            ethical_constraints=ethics,
            verbose=True,
        )

        # Create negotiation memory
        neg_memory = negotiation_memory.NegotiationMemory(
            agent_name=agent_name,
            memory_bank=memory_bank,
            verbose=True,
        )

        # Create negotiation strategy
        strategy = negotiation_strategy.BasicNegotiationStrategy(
            agent_name=agent_name,
            negotiation_style=style,
            reservation_value=reservation,
            target_value=reservation * 2.0,  # Default target is 2x reservation
            verbose=True,
        )

        # Create question components for context and reasoning
        question_about_situation = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f'Current negotiation situation',
            question=f'What is the current negotiation situation that {agent_name} is in?',
            answer_prefix=f'{agent_name} is currently ',
            add_to_memory=False,
            memory_tag='[situation perception]',
        )

        question_about_self = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f'Self-perception as negotiator',
            question=f'What kind of negotiator is {agent_name}?',
            answer_prefix=f'{agent_name} is a negotiator who ',
            add_to_memory=False,
            memory_tag='[self perception]',
        )

        question_about_action = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f'Next negotiation action',
            question=f'Given the negotiation context, what should {agent_name} do?',
            answer_prefix=f'{agent_name} should ',
            add_to_memory=False,
            memory_tag='[action reasoning]',
        )

        # Recent memories for context
        recent_memories = agent_components.all_similar_memories.AllSimilarMemories(
            model=model,
            num_memories_to_retrieve=10,
        )

        # Assemble all components
        components_of_agent = {
            'observation_to_memory': observation_to_memory,
            'observation': observation,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            instructions.name: instructions,
            neg_memory.name: neg_memory,
            strategy.name: strategy,
            'situation_perception': question_about_situation,
            'self_perception': question_about_self,
            'action_reasoning': question_about_action,
            'AllSimilarMemories': recent_memories,
        }

        # Add any extra components
        extra_components = self.params.get('extra_components', {})
        if isinstance(extra_components, dict):
            components_of_agent.update(extra_components)

        # Define component order for context building
        component_order = [
            'observation_to_memory',
            'observation',
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
            instructions.name,
            neg_memory.name,
            strategy.name,
            'situation_perception',
            'self_perception',
            'action_reasoning',
            'AllSimilarMemories',
        ]

        # Add extra component names to order
        if isinstance(extra_components, dict):
            component_order.extend([
                name for name in extra_components.keys()
                if name not in component_order
            ])

        # Create the acting component
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
            prefix_entity_name=True,
        )

        # Create the agent
        agent = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=agent_name,
            act_component=act_component,
            context_components=components_of_agent,
        )

        return agent


def build_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    name: str = 'Negotiator',
    goal: str = 'Reach a mutually beneficial agreement',
    negotiation_style: str = 'integrative',
    reservation_value: float = 0.0,
    ethical_constraints: str = 'Be honest and fair. Do not deceive or manipulate.',
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Convenience function to build a base negotiation agent.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        name: Name of the negotiation agent
        goal: Primary negotiation goal
        negotiation_style: Style of negotiation ('cooperative', 'competitive', 'integrative')
        reservation_value: Minimum acceptable value
        ethical_constraints: Ethical guidelines for negotiation
        **kwargs: Additional parameters for the agent
        
    Returns:
        Configured base negotiation agent
        
    Example:
        ```python
        agent = build_agent(
            model=my_model,
            memory_bank=my_memory,
            name="Alice",
            goal="Secure the best possible deal for my company",
            negotiation_style="competitive",
            reservation_value=1000.0
        )
        ```
    """
    params = {
        'name': name,
        'goal': goal,
        'negotiation_style': negotiation_style,
        'reservation_value': str(reservation_value),
        'ethical_constraints': ethical_constraints,
    }
    params.update(kwargs)
    
    prefab = Entity(params=params)
    return prefab.build(model=model, memory_bank=memory_bank)
