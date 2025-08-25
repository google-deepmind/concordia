"""
Simple example showing how to run a simulation with prefab game master and entities.
This demonstrates the basic setup and execution of a Concordia simulation.
"""

import datetime
from typing import Sequence

# Core Concordia imports
from concordia import components as generic_components
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.environment import engine as simulation_engine
from concordia.environment.engines import sequential
from concordia.language_model import no_language_model  # For testing without API
from concordia.utils import measurements as measurements_lib

# Prefab imports for easy setup
from concordia.prefabs import game_master as gm_prefabs
from concordia.prefabs import entity as entity_prefabs

def create_simple_simulation():
    """Create a basic simulation with prefab components."""
    
    # Step 1: Set up the language model
    # For testing, use no_language_model. In production, use a real model like:
    # - gpt_model.GPTModel(api_key="your-key", model_name="gpt-4")
    # - google_aistudio_model.GoogleAIStudioModel(api_key="your-key")
    model = no_language_model.NoLanguageModel()
    
    # Step 2: Create a game clock
    clock = game_clock.GameClock(
        start=datetime.datetime(2024, 1, 1, 8, 0),  # Start at 8 AM
        step=datetime.timedelta(minutes=10)  # 10-minute increments
    )
    
    # Step 3: Define the simulation parameters
    simulation_name = "Simple Market Negotiation"
    num_players = 3
    player_names = ["Alice", "Bob", "Charlie"]
    
    # Step 4: Create agents using prefabs
    
    # ============================================================================
    # AVAILABLE ENTITY PREFABS (from concordia.prefabs.entity):
    # ============================================================================
    # 
    # BASIC AGENTS:
    # - entity_prefabs.basic.build_agent()
    #   Basic agent with memory, observation, and simple decision-making
    #   Good for: General purpose simulations, social interactions
    #
    # - entity_prefabs.basic_with_plan.build_agent()
    #   Agent with planning capabilities and goal-oriented behavior
    #   Good for: Task-oriented scenarios, project simulations
    #
    # - entity_prefabs.minimal.build_agent()
    #   Lightweight agent with minimal components
    #   Good for: Large-scale simulations, background characters
    #
    # - entity_prefabs.basic_scripted.build_agent()
    #   Agent that follows predefined scripts and behaviors
    #   Good for: NPCs, controlled scenarios, testing
    #
    # - entity_prefabs.fake_assistant_with_configurable_system_prompt.build_agent()
    #   Configurable test assistant for development
    #   Good for: Testing, debugging, development
    #
    # NEGOTIATION AGENTS (from concordia.prefabs.entity.negotiation):
    # - base_negotiator.build_agent()
    #   Basic negotiation capabilities with simple strategies
    #   Good for: Basic bargaining, simple negotiations
    #
    # - advanced_negotiator.build_agent()
    #   Sophisticated negotiation with multiple strategies, theory of mind
    #   Good for: Complex negotiations, multi-issue bargaining
    #
    # NEGOTIATION COMPONENTS (can be added to agents):
    # - cultural_adaptation: Adapts style based on cultural context
    # - negotiation_strategy: Core strategy implementation
    # - strategy_evolution: Learns and evolves strategies
    # - swarm_intelligence: Collective negotiation behavior
    # - temporal_strategy: Time-aware tactics
    # - theory_of_mind: Models opponent beliefs and intentions
    # - uncertainty_aware: Handles incomplete information
    # ============================================================================
    
    agents = []
    for name in player_names:
        # Using basic prefab entity with planning capabilities
        agent = entity_prefabs.basic_with_plan.build_agent(
            name=name,
            model=model,
            clock=clock,
            backstory=f"{name} is a merchant in the local market.",
            goal=f"Make profitable trades while maintaining good relationships.",
            initial_plan=f"Start by assessing what goods are available and their prices.",
        )
        agents.append(agent)
    
    # Alternative examples:
    
    # Example 1: Basic agent
    # from concordia.prefabs.entity import basic
    # agent = basic.build_agent(
    #     name="Simple Agent",
    #     model=model,
    #     clock=clock,
    #     backstory="A participant in the simulation.",
    #     traits=["friendly", "curious"]
    # )
    
    # Example 2: Minimal agent (for large-scale simulations)
    # from concordia.prefabs.entity import minimal
    # agent = minimal.build_agent(
    #     name="Background Character",
    #     model=model,
    #     clock=clock
    # )
    
    # Example 3: Scripted agent
    # from concordia.prefabs.entity import basic_scripted
    # agent = basic_scripted.build_agent(
    #     name="NPC",
    #     model=model,
    #     clock=clock,
    #     script=["Greet others", "Ask about weather", "Say goodbye"]
    # )
    
    # Example 4: Basic negotiator
    # from concordia.prefabs.entity.negotiation import base_negotiator
    # negotiator = base_negotiator.build_agent(
    #     name="Negotiator",
    #     model=model,
    #     clock=clock,
    #     negotiation_style="collaborative",
    #     reservation_value=100,
    # )
    
    # Example 5: Advanced negotiator with strategies
    # from concordia.prefabs.entity.negotiation import advanced_negotiator
    # negotiator = advanced_negotiator.build_agent(
    #     name="Strategic Negotiator",
    #     model=model,
    #     clock=clock,
    #     strategy="competitive",
    #     cultural_context="western",
    #     theory_of_mind_level=2,
    #     temporal_awareness=True
    # )
    
    # Step 5: Create game master using prefab
    
    # ============================================================================
    # AVAILABLE GAME MASTER PREFABS (from concordia.prefabs.game_master):
    # ============================================================================
    #
    # GENERAL PURPOSE:
    # - gm_prefabs.generic.build_game_master()
    #   Flexible, general-purpose game master for any scenario
    #   Good for: Custom simulations, prototyping, general experiments
    #
    # DIALOGUE & NARRATIVE:
    # - gm_prefabs.dialogic.build_game_master()
    #   Manages conversations and dialogue flow
    #   Good for: Social simulations, conversation studies
    #
    # - gm_prefabs.dialogic_and_dramaturgic.build_game_master()
    #   Combines dialogue with dramatic narrative elements
    #   Good for: Story-driven simulations, role-playing scenarios
    #
    # - gm_prefabs.game_theoretic_and_dramaturgic.build_game_master()
    #   Mixes game theory with narrative elements
    #   Good for: Strategic scenarios with narrative context
    #
    # RESEARCH & EXPERIMENTS:
    # - gm_prefabs.interviewer.build_game_master()
    #   Structured interview scenarios
    #   Good for: Research interviews, Q&A sessions
    #
    # - gm_prefabs.open_ended_interviewer.build_game_master()
    #   Flexible, open-ended interview format
    #   Good for: Exploratory research, unstructured interviews
    #
    # - gm_prefabs.psychology_experiment.build_game_master()
    #   Psychological experiment coordination
    #   Good for: Psychology studies, behavioral experiments
    #
    # ECONOMIC & MARKET:
    # - gm_prefabs.marketplace.build_game_master()
    #   Economic marketplace with supply/demand dynamics
    #   Good for: Trading simulations, economic experiments
    #
    # NEGOTIATION (from concordia.prefabs.game_master.negotiation):
    # - negotiation.build_game_master()
    #   Specialized negotiation management
    #   Good for: Bilateral/multilateral negotiations, bargaining
    #
    # CONTEXTUAL:
    # - gm_prefabs.situated.build_game_master()
    #   Context-aware environment management
    #   Good for: Location-based scenarios, environmental simulations
    #
    # - gm_prefabs.situated_in_time_and_place.build_game_master()
    #   Temporal and spatial context management
    #   Good for: Historical simulations, time-sensitive scenarios
    #
    # - gm_prefabs.scripted.build_game_master()
    #   Follows predefined scripts and scenarios
    #   Good for: Controlled experiments, reproducible scenarios
    #
    # SPECIALIZED COMPONENTS (can be added to game masters):
    # - gm_cultural_awareness: Cultural context management
    # - gm_social_intelligence: Social dynamics tracking
    # - gm_temporal_dynamics: Time-based mechanics
    # - negotiation_state: State tracking for negotiations
    # - negotiation_validation: Validation logic
    # ============================================================================
    
    # Using marketplace game master for economic simulation
    game_master = gm_prefabs.marketplace.build_game_master(
        name="Market Game Master",
        model=model,
        clock=clock,
        players=agents,
        goods=["apples", "oranges", "bananas"],
        initial_prices={"apples": 2.0, "oranges": 3.0, "bananas": 1.5},
        price_elasticity=0.1,
    )
    
    # Alternative examples:
    
    # Example 1: Generic game master (most flexible)
    # game_master = gm_prefabs.generic.build_game_master(
    #     name="General GM",
    #     model=model,
    #     clock=clock,
    #     players=agents,
    #     world_description="A collaborative workspace",
    #     objectives=["Complete tasks", "Maintain team cohesion"]
    # )
    
    # Example 2: Dialogue-focused game master
    # game_master = gm_prefabs.dialogic.build_game_master(
    #     name="Conversation Moderator",
    #     model=model,
    #     clock=clock,
    #     players=agents,
    #     topic="Project planning meeting",
    #     conversation_style="structured"
    # )
    
    # Example 3: Psychology experiment
    # game_master = gm_prefabs.psychology_experiment.build_game_master(
    #     name="Experiment Coordinator",
    #     model=model,
    #     clock=clock,
    #     players=agents,
    #     experiment_type="trust_game",
    #     conditions=["control", "treatment"],
    #     measurements=["cooperation_level", "trust_score"]
    # )
    
    # Example 4: Negotiation game master
    # from concordia.prefabs.game_master.negotiation import negotiation
    # game_master = negotiation.build_game_master(
    #     name="Negotiation Facilitator",
    #     model=model,
    #     clock=clock,
    #     players=agents,
    #     negotiation_type="multilateral",
    #     issues=["price", "quantity", "delivery"],
    #     max_rounds=15
    # )
    
    # Example 5: Situated environment with time and place
    # game_master = gm_prefabs.situated_in_time_and_place.build_game_master(
    #     name="Historical Simulation",
    #     model=model,
    #     clock=clock,
    #     players=agents,
    #     location="Victorian London",
    #     time_period="1890s",
    #     environmental_factors=["industrial_revolution", "social_hierarchy"]
    # )
    
    # Step 6: Create the simulation engine
    # Sequential engine - agents act one at a time
    engine = sequential.SequentialEngine(
        game_master=game_master,
        players=agents,
        clock=clock,
        name=simulation_name,
    )
    
    # Alternative engines:
    # - simultaneous.SimultaneousEngine() - All agents act at once
    # - parallel.ParallelEngine() - Parallel execution for performance
    
    return engine, agents, game_master

def run_simulation_with_logging():
    """Run a simulation with detailed logging and metrics."""
    
    # Create the simulation
    engine, agents, game_master = create_simple_simulation()
    
    # Step 7: Run the simulation
    num_steps = 10  # Run for 10 time steps
    
    print(f"Starting simulation: {engine.name}")
    print(f"Number of agents: {len(agents)}")
    print(f"Running for {num_steps} steps\n")
    
    for step in range(num_steps):
        print(f"--- Step {step + 1} ---")
        
        # Execute one simulation step
        engine.step()
        
        # Get observations from agents (what they perceived)
        for agent in agents:
            last_observation = agent.get_last_observation()
            print(f"{agent.name} observed: {last_observation}")
        
        # Get game state from game master
        game_state = game_master.get_state()
        print(f"Game state: {game_state}\n")
    
    print("Simulation complete!")
    
    # Step 8: Collect metrics and results
    # This depends on the specific game master and components used
    final_scores = game_master.get_scores()
    print(f"Final scores: {final_scores}")

def run_negotiation_simulation():
    """Example specifically for negotiation scenarios."""
    
    from concordia.prefabs.entity.negotiation import advanced_negotiator
    from concordia.prefabs.game_master.negotiation import negotiation
    from concordia.language_model import no_language_model
    
    # Setup
    model = no_language_model.NoLanguageModel()
    clock = game_clock.GameClock(
        start=datetime.datetime(2024, 1, 1, 9, 0),
        step=datetime.timedelta(minutes=5)
    )
    
    # Create negotiating agents with different strategies
    negotiator1 = advanced_negotiator.build_agent(
        name="Competitive Chris",
        model=model,
        clock=clock,
        strategy="competitive",
        initial_offer=150,
        reservation_value=100,
        concession_rate=0.05,
    )
    
    negotiator2 = advanced_negotiator.build_agent(
        name="Collaborative Carol",
        model=model,
        clock=clock,
        strategy="collaborative",
        initial_offer=120,
        reservation_value=80,
        concession_rate=0.1,
    )
    
    # Create negotiation game master
    negotiation_gm = negotiation.build_game_master(
        name="Negotiation Moderator",
        model=model,
        clock=clock,
        players=[negotiator1, negotiator2],
        negotiation_type="bilateral",
        max_rounds=20,
        deadline=datetime.timedelta(minutes=30),
        issues=["price", "delivery_time", "quantity"],
    )
    
    # Create engine
    engine = sequential.SequentialEngine(
        game_master=negotiation_gm,
        players=[negotiator1, negotiator2],
        clock=clock,
        name="Bilateral Negotiation",
    )
    
    # Run negotiation
    print("Starting negotiation simulation...")
    
    for round_num in range(20):  # Maximum 20 rounds
        print(f"\n--- Round {round_num + 1} ---")
        
        # Step the simulation
        engine.step()
        
        # Check if agreement reached
        if negotiation_gm.is_agreement_reached():
            agreement = negotiation_gm.get_agreement()
            print(f"Agreement reached! Terms: {agreement}")
            break
        
        # Show current offers
        current_offers = negotiation_gm.get_current_offers()
        print(f"Current offers: {current_offers}")
    
    if not negotiation_gm.is_agreement_reached():
        print("No agreement reached - negotiation failed.")
    
    # Get final metrics
    metrics = negotiation_gm.get_negotiation_metrics()
    print(f"\nNegotiation metrics: {metrics}")

def main():
    """Main entry point - choose which simulation to run."""
    
    print("Concordia Simulation Examples")
    print("=" * 40)
    print("1. Basic simulation with marketplace")
    print("2. Negotiation simulation")
    print("3. Custom configuration")
    
    # For this example, run the basic simulation
    print("\nRunning basic marketplace simulation...\n")
    run_simulation_with_logging()
    
    # Uncomment to run negotiation instead:
    # print("\nRunning negotiation simulation...\n")
    # run_negotiation_simulation()

if __name__ == "__main__":
    main()