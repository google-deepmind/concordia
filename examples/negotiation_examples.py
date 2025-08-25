"""
Example scripts demonstrating the negotiation prefab framework.

This file contains practical examples showing how to use the negotiation
agents and game masters for different scenarios.
"""

import datetime
from typing import List, Sequence

# Core Concordia imports
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.language_model import no_language_model  # For testing without API
from concordia.utils import measurements as measurements_lib

# Negotiation prefab imports
from concordia.prefabs.entity.negotiation import base_negotiator
from concordia.prefabs.entity.negotiation import advanced_negotiator
from concordia.prefabs.game_master.negotiation import negotiation


def create_basic_price_negotiation():
    """
    Example 1: Simple bilateral price negotiation between buyer and seller.
    
    This demonstrates the most basic negotiation setup using base negotiators
    with different styles and reservation values.
    """
    print("=== Example 1: Basic Price Negotiation ===")
    
    # Set up language model and memory
    model = no_language_model.NoLanguageModel()
    clock = game_clock.FixedIntervalClock()
    memory_bank = basic_associative_memory.AssociativeMemoryBank()
    
    # Create buyer agent (competitive style, wants lower price)
    buyer = base_negotiator.build_agent(
        model=model,
        memory_bank=memory_bank,
        name="Alice_Buyer",
        goal="Purchase the item at the lowest possible price",
        negotiation_style="competitive",
        reservation_value=200.0,  # Won't pay more than $200
        ethical_constraints="Be firm but fair in negotiations",
    )
    
    # Create seller agent (cooperative style, wants higher price)
    seller = base_negotiator.build_agent(
        model=model,
        memory_bank=memory_bank,
        name="Bob_Seller", 
        goal="Sell the item at a fair market price",
        negotiation_style="cooperative",
        reservation_value=150.0,  # Won't sell for less than $150
        ethical_constraints="Be honest about product value and condition",
    )
    
    # Create bilateral negotiation game master
    gm = negotiation.build_bilateral_negotiation(
        model=model,
        memory_bank=memory_bank,
        entities=[buyer, seller],
        name="Price Negotiation Session",
        max_rounds=10,
    )
    
    print(f"Created buyer: {buyer._agent_name}")
    print(f"Created seller: {seller._agent_name}")
    print(f"Created game master: {gm._agent_name}")
    print("Negotiation setup complete!")
    
    return buyer, seller, gm


def create_cross_cultural_business_negotiation():
    """
    Example 2: Cross-cultural business negotiation.
    
    Demonstrates advanced agents with cultural adaptation capabilities
    for international business scenarios.
    """
    print("\n=== Example 2: Cross-Cultural Business Negotiation ===")
    
    # Set up environment
    model = no_language_model.NoLanguageModel()
    clock = game_clock.FixedIntervalClock()
    memory_bank = basic_associative_memory.AssociativeMemoryBank()
    
    # Create Western business representative
    western_rep = advanced_negotiator.build_cultural_agent(
        model=model,
        memory_bank=memory_bank,
        name="Sarah_Western",
        own_culture="western_business",
        goal="Negotiate a supply contract with favorable terms and quick resolution",
        negotiation_style="competitive",
        reservation_value=500000.0,
    )
    
    # Create Eastern business representative  
    eastern_rep = advanced_negotiator.build_cultural_agent(
        model=model,
        memory_bank=memory_bank,
        name="Tanaka_Eastern",
        own_culture="east_asian",
        goal="Establish long-term partnership with mutual respect and benefit",
        negotiation_style="integrative",
        reservation_value=450000.0,
    )
    
    # Create cultural-aware game master
    gm = negotiation.build_cultural_negotiation(
        model=model,
        memory_bank=memory_bank,
        entities=[western_rep, eastern_rep],
        name="International Trade Negotiation",
        negotiation_type="contract",
        max_rounds=20,  # Longer for relationship building
    )
    
    print(f"Western representative: {western_rep._agent_name}")
    print(f"Eastern representative: {eastern_rep._agent_name}")
    print(f"Game master with cultural mediation: {gm._agent_name}")
    print("Cross-cultural negotiation ready!")
    
    return western_rep, eastern_rep, gm


def create_multi_party_resource_allocation():
    """
    Example 3: Multi-party resource allocation negotiation.
    
    Shows how to set up complex negotiations with multiple parties,
    coalition formation, and collective intelligence.
    """
    print("\n=== Example 3: Multi-Party Resource Allocation ===")
    
    # Set up environment
    model = no_language_model.NoLanguageModel()
    clock = game_clock.FixedIntervalClock()
    memory_bank = basic_associative_memory.AssociativeMemoryBank()
    
    # Define participants with different capabilities and needs
    participants = [
        {
            "name": "TechCorp",
            "modules": ["swarm_intelligence", "uncertainty_aware"],
            "goal": "Secure technology resources for innovation projects",
            "reservation": 300000.0,
        },
        {
            "name": "ManufacturingInc", 
            "modules": ["temporal_strategy", "theory_of_mind"],
            "goal": "Obtain manufacturing resources for production scaling",
            "reservation": 250000.0,
        },
        {
            "name": "StartupLLC",
            "modules": ["strategy_evolution", "swarm_intelligence"], 
            "goal": "Access shared resources despite limited budget",
            "reservation": 100000.0,
        },
        {
            "name": "NonProfitOrg",
            "modules": ["cultural_adaptation", "theory_of_mind"],
            "goal": "Secure resources for community benefit programs", 
            "reservation": 75000.0,
        }
    ]
    
    # Create sophisticated agents
    agents = []
    for participant in participants:
        agent = advanced_negotiator.build_agent(
            model=model,
            memory_bank=memory_bank,
            name=participant["name"],
            goal=participant["goal"],
            negotiation_style="integrative",  # Collaborative for resource sharing
            reservation_value=participant["reservation"],
            modules=participant["modules"],
        )
        agents.append(agent)
    
    # Create multilateral game master with full capabilities
    gm = negotiation.build_multilateral_negotiation(
        model=model,
        memory_bank=memory_bank,
        entities=agents,
        name="Resource Allocation Summit",
        negotiation_type="multi_issue",
        max_rounds=30,
    )
    
    print(f"Created {len(agents)} participating organizations:")
    for agent in agents:
        print(f"  - {agent._agent_name}")
    print(f"Game master: {gm._agent_name}")
    print("Multi-party negotiation configured!")
    
    return agents, gm


def create_adaptive_learning_negotiation():
    """
    Example 4: Adaptive negotiation with strategy evolution.
    
    Demonstrates agents that learn and adapt their strategies over time,
    suitable for repeated negotiations or tournament scenarios.
    """
    print("\n=== Example 4: Adaptive Learning Negotiation ===")
    
    # Set up environment  
    model = no_language_model.NoLanguageModel()
    clock = game_clock.FixedIntervalClock()
    memory_bank = basic_associative_memory.AssociativeMemoryBank()
    
    # Create learning agents with different initial strategies
    adaptive_alice = advanced_negotiator.build_adaptive_agent(
        model=model,
        memory_bank=memory_bank,
        name="Alice_Learner",
        goal="Maximize long-term negotiation success through adaptation",
        negotiation_style="cooperative",  # Starting strategy
        learning_rate=0.1,  # Fast learning
        reservation_value=150.0,
    )
    
    adaptive_bob = advanced_negotiator.build_adaptive_agent(
        model=model,
        memory_bank=memory_bank, 
        name="Bob_Evolving",
        goal="Develop optimal negotiation strategies through experience",
        negotiation_style="competitive",  # Different starting point
        learning_rate=0.05,  # Slower, more conservative learning
        reservation_value=180.0,
    )
    
    # Create adaptive game master that tracks strategy evolution
    gm = negotiation.build_adaptive_negotiation(
        model=model,
        memory_bank=memory_bank,
        entities=[adaptive_alice, adaptive_bob],
        name="Learning Lab Negotiation",
        max_rounds=50,  # Longer for learning to occur
    )
    
    print(f"Adaptive learner: {adaptive_alice._agent_name}")
    print(f"Evolving negotiator: {adaptive_bob._agent_name}")
    print(f"Strategy evolution tracker: {gm._agent_name}")
    print("Adaptive negotiation laboratory ready!")
    
    return adaptive_alice, adaptive_bob, gm


def create_information_asymmetry_scenario():
    """
    Example 5: Negotiation with information asymmetry.
    
    Demonstrates handling of incomplete information, uncertainty,
    and strategic information sharing.
    """
    print("\n=== Example 5: Information Asymmetry Scenario ===")
    
    # Set up environment
    model = no_language_model.NoLanguageModel()
    clock = game_clock.FixedIntervalClock()
    memory_bank = basic_associative_memory.AssociativeMemoryBank()
    
    # Create agents with uncertainty awareness
    informed_agent = advanced_negotiator.build_agent(
        model=model,
        memory_bank=memory_bank,
        name="InfoAdvantage_Alice",
        goal="Leverage superior information while maintaining ethical standards",
        negotiation_style="integrative",
        modules=["uncertainty_aware", "theory_of_mind"],
        module_configs={
            "uncertainty_aware": {
                "confidence_threshold": 0.8,  # High confidence
                "information_gathering_budget": 0.05,  # Low need to gather info
            }
        },
        reservation_value=200.0,
    )
    
    uncertain_agent = advanced_negotiator.build_agent(
        model=model,
        memory_bank=memory_bank,
        name="InfoSeeking_Bob",
        goal="Make best decisions despite incomplete information",
        negotiation_style="cooperative",
        modules=["uncertainty_aware", "strategy_evolution"],
        module_configs={
            "uncertainty_aware": {
                "confidence_threshold": 0.4,  # Lower confidence
                "information_gathering_budget": 0.2,  # Higher info gathering
            }
        },
        reservation_value=180.0,
    )
    
    # Create game master with uncertainty management
    gm = negotiation.build_game_master(
        model=model,
        memory_bank=memory_bank,
        entities=[informed_agent, uncertain_agent],
        name="Information Asymmetry Experiment",
        negotiation_type="price",
        gm_modules=["uncertainty_management", "social_intelligence"],
        max_rounds=15,
    )
    
    print(f"Informed negotiator: {informed_agent._agent_name}")
    print(f"Information-seeking negotiator: {uncertain_agent._agent_name}")
    print(f"Uncertainty-aware game master: {gm._agent_name}")
    print("Information asymmetry scenario ready!")
    
    return informed_agent, uncertain_agent, gm


def demonstrate_gm_module_capabilities():
    """
    Example 6: Showcase of all GM module capabilities.
    
    Creates a comprehensive negotiation with all available GM modules
    to demonstrate the full range of framework capabilities.
    """
    print("\n=== Example 6: Complete GM Module Showcase ===")
    
    # Set up environment
    model = no_language_model.NoLanguageModel()
    clock = game_clock.FixedIntervalClock()
    memory_bank = basic_associative_memory.AssociativeMemoryBank()
    
    # Create diverse agents to trigger different modules
    comprehensive_alice = advanced_negotiator.build_agent(
        model=model,
        memory_bank=memory_bank,
        name="Comprehensive_Alice",
        goal="Demonstrate all negotiation capabilities",
        modules=[
            "cultural_adaptation", 
            "temporal_strategy",
            "theory_of_mind"
        ],
        module_configs={
            "cultural_adaptation": {"own_culture": "western_business"},
            "temporal_strategy": {"discount_factor": 0.9},
            "theory_of_mind": {"max_recursion_depth": 2},
        },
        reservation_value=300.0,
    )
    
    comprehensive_bob = advanced_negotiator.build_agent(
        model=model,
        memory_bank=memory_bank, 
        name="Comprehensive_Bob",
        goal="Showcase advanced negotiation modules",
        modules=[
            "swarm_intelligence",
            "uncertainty_aware", 
            "strategy_evolution"
        ],
        module_configs={
            "swarm_intelligence": {"consensus_threshold": 0.7},
            "uncertainty_aware": {"confidence_threshold": 0.6},
            "strategy_evolution": {"learning_rate": 0.08},
        },
        reservation_value=250.0,
    )
    
    comprehensive_charlie = advanced_negotiator.build_cultural_agent(
        model=model,
        memory_bank=memory_bank,
        name="Comprehensive_Charlie", 
        own_culture="middle_eastern",
        goal="Represent cultural mediation capabilities",
        reservation_value=275.0,
    )
    
    # Create game master with ALL modules enabled
    gm = negotiation.build_game_master(
        model=model,
        memory_bank=memory_bank,
        entities=[comprehensive_alice, comprehensive_bob, comprehensive_charlie],
        name="Complete Showcase Negotiation",
        negotiation_type="multi_issue",
        gm_modules=[
            "social_intelligence",      # Emotional dynamics, relationships
            "temporal_dynamics",        # Time management, commitments  
            "cultural_awareness",       # Cross-cultural mediation
            "uncertainty_management",   # Information asymmetry
            "collective_intelligence",  # Multi-party coordination
            "strategy_evolution",       # Learning and adaptation
        ],
        max_rounds=25,
    )
    
    print("Created comprehensive demonstration with:")
    print(f"  Agent 1: {comprehensive_alice._agent_name} (Cultural + Temporal + ToM)")
    print(f"  Agent 2: {comprehensive_bob._agent_name} (Swarm + Uncertainty + Evolution)")  
    print(f"  Agent 3: {comprehensive_charlie._agent_name} (Cultural specialist)")
    print(f"  Game Master: {gm._agent_name}")
    print("  All 6 GM modules: Social, Temporal, Cultural, Uncertainty, Collective, Evolution")
    print("Complete negotiation framework showcase ready!")
    
    return [comprehensive_alice, comprehensive_bob, comprehensive_charlie], gm


def run_quick_demonstration():
    """
    Quick demonstration showing basic usage of the negotiation framework.
    """
    print("🤝 Concordia Negotiation Framework Demonstration\n")
    print("This demonstrates key capabilities of the negotiation prefabs:")
    print("- Agent creation with different negotiation styles")
    print("- Game master configuration for different scenarios")  
    print("- Advanced modules for sophisticated negotiations\n")
    
    # Run all examples
    examples = [
        create_basic_price_negotiation,
        create_cross_cultural_business_negotiation,
        create_multi_party_resource_allocation,
        create_adaptive_learning_negotiation,
        create_information_asymmetry_scenario,
        demonstrate_gm_module_capabilities,
    ]
    
    results = []
    for example_func in examples:
        try:
            result = example_func()
            results.append(result)
            print("✅ Success!\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
            results.append(None)
    
    print("🎉 Demonstration complete!")
    print("\nFramework capabilities successfully showcased:")
    print("- ✅ Basic bilateral negotiations")  
    print("- ✅ Cross-cultural business scenarios")
    print("- ✅ Multi-party resource allocation")
    print("- ✅ Adaptive learning agents")
    print("- ✅ Information asymmetry handling") 
    print("- ✅ Complete GM module integration")
    
    return results


if __name__ == "__main__":
    # Run the demonstration
    demonstration_results = run_quick_demonstration()
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS:")
    print("="*60)
    print("""
To use these examples in your own code:

1. Import the negotiation prefabs:
   from concordia.prefabs.entity.negotiation import base_negotiator, advanced_negotiator
   from concordia.prefabs.game_master.negotiation import negotiation

2. Set up your language model and memory:
   model = your_language_model_here  # e.g., GPT, Gemma, etc.
   memory_bank = basic_associative_memory.AssociativeMemoryBank(model, clock)

3. Create agents using the convenience functions:
   agent = base_negotiator.build_agent(model, memory_bank, name="Alice", ...)
   
4. Create game master for your scenario:
   gm = negotiation.build_bilateral_negotiation(model, memory_bank, [agent1, agent2])

5. Run your simulation using Concordia's simulation environment.

For more details, see the documentation in the prefab files!
""")