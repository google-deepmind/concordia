# Copyright 2025 DeepMind Technologies Limited.
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

"""A prefab containing a negotiation game master."""

from collections.abc import Mapping, Sequence
import copy
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.prefabs.game_master.negotiation.components import negotiation_state
from concordia.prefabs.game_master.negotiation.components import negotiation_validation
from concordia.prefabs.game_master.negotiation.components import negotiation_modules
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains as thought_chains_lib
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class NegotiationGameMaster(prefab_lib.Prefab):
  """A prefab entity implementing a negotiation game master.

  This game master specializes in managing negotiation scenarios:
  - Tracks offers and counteroffers
  - Validates agreements against BATNAs
  - Manages negotiation phases and deadlines
  - Supports multiple negotiation protocols
  - Provides negotiation-specific observations
  """

  description: str = 'A game master specialized for negotiation scenarios.'
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Negotiation Mediator',
          'negotiation_type': 'price',  # 'price', 'contract', 'multi_issue'
          'protocol': 'alternating',  # 'alternating', 'simultaneous'
          'max_rounds': 20,
          'enable_deadlines': True,
          'enable_batna_validation': True,
          'enable_fairness_check': True,
          'instructions': (
              'You are a professional negotiation mediator. Your role is to:\n'
              '1. Facilitate negotiations between parties\n'
              '2. Ensure all offers are properly communicated\n'
              '3. Validate agreements for feasibility and fairness\n'
              '4. Track negotiation progress and deadlines\n'
              '5. Announce when agreements are reached or negotiations fail\n'
              '6. Maintain neutrality and professionalism'
          ),
          'extra_event_resolution_steps': '',
          'extra_components': {},
          'extra_components_index': {},
          'acting_order': 'alternating',  # For negotiations, usually alternating
          'gm_modules': [],  # List of GM module names to enable
          'gm_module_configs': {},  # Module-specific configurations
          'auto_detect_modules': False,  # Auto-detect and enable compatible modules
      }
  )
  entities: (
      Sequence[entity_agent_with_logging.EntityAgentWithLogging]
  ) = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a negotiation game master.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      An entity configured as a negotiation game master.
    """

    extra_components = self.params.get('extra_components', {})
    extra_components_index = self.params.get('extra_components_index', {})

    if extra_components_index and extra_components:
      if extra_components_index.keys() != extra_components.keys():
        raise ValueError(
            'extra_components_index must have the same keys as'
            ' extra_components.'
        )

    name = self.params.get('name')
    negotiation_type = self.params.get('negotiation_type', 'price')
    protocol = self.params.get('protocol', 'alternating')
    max_rounds = self.params.get('max_rounds', 20)
    enable_deadlines = self.params.get('enable_deadlines', True)
    enable_batna_validation = self.params.get('enable_batna_validation', True)
    enable_fairness_check = self.params.get('enable_fairness_check', True)
    custom_instructions = self.params.get('instructions')

    # Module support parameters
    gm_modules = self.params.get('gm_modules', [])
    gm_module_configs = self.params.get('gm_module_configs', {})
    auto_detect_modules = self.params.get('auto_detect_modules', False)

    extra_event_resolution_steps = self.params.get(
        'extra_event_resolution_steps', ''
    )
    assert isinstance(extra_event_resolution_steps, str)  # For pytype.

    if ',' in extra_event_resolution_steps:
      extra_event_resolution_steps = [
          step.strip()
          for step in extra_event_resolution_steps.split(',')
          if step
      ]
    else:
      extra_event_resolution_steps = [extra_event_resolution_steps]

    # Core components
    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    # Negotiation-specific instructions
    instructions_key = 'instructions'
    instructions = gm_components.instructions.Instructions()
    if custom_instructions:
      instructions.set_state(custom_instructions)

    # Player setup
    player_names = [entity.name for entity in self.entities]
    player_characters_key = 'player_characters'
    player_characters = gm_components.instructions.PlayerCharacters(
        player_characters=player_names,
    )

    # Observation components
    observation_to_memory_key = 'observation_to_memory'
    observation_to_memory = actor_components.observation.ObservationToMemory()

    observation_component_key = (
        actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = actor_components.observation.LastNObservations(
        history_length=1000,
    )

    # Negotiation state tracking
    negotiation_state_key = 'negotiation_state'
    negotiation_state_tracker = negotiation_state.NegotiationStateTracker(
        initial_phase='opening',
        max_rounds=max_rounds,
        enable_deadlines=enable_deadlines,
        track_relationships=True,
    )

    # Negotiation validation
    negotiation_validator_key = 'negotiation_validator'
    negotiation_validator = negotiation_validation.NegotiationValidator(
        domain_type=negotiation_type,
        enable_batna_check=enable_batna_validation,
        enable_fairness_check=enable_fairness_check,
        enable_feasibility_check=True,
    )

    # Display negotiation history
    display_events_key = 'display_events'
    display_events = gm_components.event_resolution.DisplayEvents(
        model=model,
        pre_act_label=(
            'Negotiation history (ordered from oldest to most recent events)'),
    )

    # Relevant memories for context
    relevant_memories_key = 'relevant_memories'
    relevant_memories = (
        actor_components.all_similar_memories.AllSimilarMemories(
            model=model,
            components=[
                display_events_key,
                negotiation_state_key,
            ],
            num_memories_to_retrieve=5,
            pre_act_label='Relevant negotiation context',
        )
    )

    # Make observation with negotiation context
    make_observation_key = (
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )
    observation_components = [
        instructions_key,
        player_characters_key,
        negotiation_state_key,
        negotiation_validator_key,
        relevant_memories_key,
        display_events_key,
    ]
    # Add GM modules to observation components
    observation_components.extend([f'gm_module_{m}' for m in gm_modules])

    make_observation = gm_components.make_observation.MakeObservation(
        model=model,
        player_names=player_names,
        components=observation_components,
        reformat_observations_in_specified_style=(
            'Format negotiation observations as: '
            '"//Round X//[Negotiation Phase] Current situation and any offers".'
        ),
    )

    # Next acting - handle negotiation protocols
    next_acting_kwargs = dict(
        model=model,
        components=[
            instructions_key,
            player_characters_key,
            negotiation_state_key,
            relevant_memories_key,
            display_events_key,
        ],
    )
    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY

    acting_order = self.params.get('acting_order', 'alternating')
    if protocol == 'alternating' or acting_order == 'alternating':
      # For alternating offers protocol
      next_actor = gm_components.next_acting.NextActingInFixedOrder(
          sequence=player_names,
      )
    elif protocol == 'simultaneous':
      # For simultaneous offers, use game master choice
      next_actor = gm_components.next_acting.NextActing(
          **next_acting_kwargs,
          player_names=player_names,
      )
    else:
      # Default to game master choice
      next_actor = gm_components.next_acting.NextActing(
          **next_acting_kwargs,
          player_names=player_names,
      )

    # Next action specification for negotiations
    next_action_spec_kwargs = copy.copy(next_acting_kwargs)
    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    next_action_spec = gm_components.next_acting.NextActionSpec(
        **next_action_spec_kwargs,
        player_names=player_names,
        call_to_next_action_spec=(
            'What type of negotiation action should {name} take next? '
            'Choose from: make offer, accept offer, reject offer, '
            'request information, make concession, or walk away.'
        ),
    )

    # Event resolution for negotiation actions
    account_for_agency_of_others = thought_chains_lib.AccountForAgencyOfOthers(
        model=model, players=self.entities, verbose=False
    )

    event_resolution_steps = [
        thought_chains_lib.maybe_inject_narrative_push,
        account_for_agency_of_others,
        thought_chains_lib.result_to_who_what_where,
    ]

    # Add extra resolution steps if provided
    if extra_event_resolution_steps:
      for step in extra_event_resolution_steps:
        if step:
          event_resolution_steps.append(getattr(thought_chains_lib, step))

    event_resolution_components = [
        instructions_key,
        player_characters_key,
        negotiation_state_key,
        negotiation_validator_key,
        relevant_memories_key,
        display_events_key,
    ]
    # Add GM modules to event resolution
    event_resolution_components.extend([f'gm_module_{m}' for m in gm_modules])

    event_resolution_key = (
        gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
    )
    event_resolution = gm_components.event_resolution.EventResolution(
        model=model,
        event_resolution_steps=event_resolution_steps,
        components=event_resolution_components,
        notify_observers=True,
    )

    # Initialize GM modules
    gm_module_instances = {}

    # Auto-detect modules if requested
    if auto_detect_modules:
      agent_modules = negotiation_modules.detect_agent_modules(self.entities)
      suggested_modules = negotiation_modules.suggest_gm_modules(agent_modules)
      gm_modules.extend([m for m in suggested_modules if m not in gm_modules])

    # Create module instances
    for module_name in gm_modules:
      config = gm_module_configs.get(module_name, {})
      module_instance = negotiation_modules.NegotiationGMModuleRegistry.create_module(
          module_name, config
      )
      if module_instance:
        gm_module_instances[f'gm_module_{module_name}'] = module_instance

        # Special initialization for specific modules
        if module_name == 'cultural_awareness':
          # Import the module to access its class
          from concordia.prefabs.game_master.negotiation.components import gm_cultural_awareness
          if isinstance(module_instance, gm_cultural_awareness.CulturalAwarenessGM):
            # Auto-detect participant cultures if available
            for entity in self.entities:
              if hasattr(entity, '_context_components'):
                components = entity._context_components
                if 'CulturalAdaptation' in components:
                  # Extract culture from component if possible
                  module_instance.set_participant_culture(
                      entity.name,
                      'western_business'  # Default, would extract from component
                  )

    # Assemble all components
    components_of_game_master = {
        instructions_key: instructions,
        player_characters_key: player_characters,
        negotiation_state_key: negotiation_state_tracker,
        negotiation_validator_key: negotiation_validator,
        relevant_memories_key: relevant_memories,
        observation_component_key: observation,
        observation_to_memory_key: observation_to_memory,
        display_events_key: display_events,
        memory_component_key: memory_component,
        make_observation_key: make_observation,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        event_resolution_key: event_resolution,
    }

    # Add GM modules to components
    components_of_game_master.update(gm_module_instances)

    component_order = list(components_of_game_master.keys())

    # Add extra components if provided
    if extra_components:
      components_of_game_master.update(extra_components)
      if extra_components_index:
        for component_name in extra_components.keys():
          component_order.insert(
              extra_components_index[component_name],
              component_name,
          )
      else:
        component_order = list(components_of_game_master.keys())

    # Create the switch act component
    act_component = gm_components.switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    # Build the game master entity
    game_master = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components_of_game_master,
    )

    return game_master


def build_game_master(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    name: str = 'Negotiation Mediator',
    negotiation_type: str = 'price',
    max_rounds: int = 10,
    gm_modules: list = None,
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Convenience function to build a negotiation game master.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        entities: List of negotiating agents
        name: Name of the game master
        negotiation_type: Type of negotiation ('price', 'contract', 'multi_issue')
        max_rounds: Maximum number of negotiation rounds
        gm_modules: List of GM module names to enable
        **kwargs: Additional parameters for the game master
        
    Returns:
        Configured negotiation game master
        
    Available GM modules:
        - 'social_intelligence': Tracks emotional dynamics and relationships
        - 'temporal_dynamics': Manages time pressure and deadlines
        - 'cultural_awareness': Handles cross-cultural negotiation protocols
        - 'uncertainty_management': Manages information asymmetry
        - 'collective_intelligence': Coordinates multi-party negotiations
        - 'strategy_evolution': Tracks strategy adaptation
        
    Example:
        ```python
        gm = build_game_master(
            model=my_model,
            memory_bank=my_memory,
            entities=[agent1, agent2],
            name="Trade Mediator",
            negotiation_type="contract",
            max_rounds=15,
            gm_modules=['cultural_awareness', 'social_intelligence']
        )
        ```
    """
    if gm_modules is None:
        gm_modules = []
    
    params = {
        'name': name,
        'negotiation_type': negotiation_type,
        'max_rounds': max_rounds,
        'gm_modules': gm_modules,
        'auto_detect_modules': True,  # Enable auto-detection
    }
    params.update(kwargs)
    
    prefab = NegotiationGameMaster(params=params)
    prefab.entities = entities
    return prefab.build(model=model, memory_bank=memory_bank)


def build_bilateral_negotiation(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    name: str = 'Bilateral Negotiation',
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a game master optimized for two-party negotiations.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        entities: List of exactly two negotiating agents
        name: Name of the negotiation session
        **kwargs: Additional parameters
        
    Returns:
        Game master configured for bilateral negotiation
    """
    if len(entities) != 2:
        raise ValueError("Bilateral negotiation requires exactly 2 entities")
    
    return build_game_master(
        model=model,
        memory_bank=memory_bank,
        entities=entities,
        name=name,
        negotiation_type='bilateral',
        gm_modules=['social_intelligence', 'temporal_dynamics'],
        **kwargs
    )


def build_multilateral_negotiation(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    name: str = 'Multilateral Negotiation',
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a game master optimized for multi-party negotiations.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        entities: List of multiple negotiating agents
        name: Name of the negotiation session
        **kwargs: Additional parameters
        
    Returns:
        Game master configured for multilateral negotiation
    """
    if len(entities) < 3:
        raise ValueError("Multilateral negotiation requires at least 3 entities")
    
    return build_game_master(
        model=model,
        memory_bank=memory_bank,
        entities=entities,
        name=name,
        negotiation_type='multilateral',
        gm_modules=['collective_intelligence', 'uncertainty_management', 'social_intelligence'],
        **kwargs
    )


def build_cultural_negotiation(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    name: str = 'Cross-Cultural Negotiation',
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a game master optimized for cross-cultural negotiations.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        entities: List of negotiating agents from different cultures
        name: Name of the negotiation session
        **kwargs: Additional parameters
        
    Returns:
        Game master with cultural mediation capabilities
    """
    return build_game_master(
        model=model,
        memory_bank=memory_bank,
        entities=entities,
        name=name,
        negotiation_type='cross_cultural',
        gm_modules=['cultural_awareness', 'social_intelligence', 'temporal_dynamics'],
        **kwargs
    )


def build_adaptive_negotiation(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    name: str = 'Adaptive Negotiation',
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a game master that adapts to agent strategies over time.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        entities: List of negotiating agents
        name: Name of the negotiation session
        **kwargs: Additional parameters
        
    Returns:
        Game master with strategy evolution tracking
    """
    return build_game_master(
        model=model,
        memory_bank=memory_bank,
        entities=entities,
        name=name,
        negotiation_type='adaptive',
        gm_modules=['strategy_evolution', 'uncertainty_management', 'social_intelligence'],
        max_rounds=20,  # Longer sessions for adaptation
        **kwargs
    )
