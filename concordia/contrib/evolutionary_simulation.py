import numpy as np
from concordia.prefabs.simulation import generic as simulation_generic
from concordia.prefabs.game_master.public_goods_game_master import PublicGoodsGameMaster
from concordia.prefabs.entity import basic_with_plan
from concordia.environment.engines import simultaneous
from concordia.language_model import no_language_model
from concordia.associative_memory import basic_associative_memory
from concordia.utils import helper_functions as helper_functions_lib
from concordia.utils import measurements as measurements_lib
from typing import List, Dict, Tuple, Any, Optional
import random
import datetime

# === Configurable parameters ===
POP_SIZE = 4
NUM_GENERATIONS = 10
SELECTION_METHOD = 'topk'  # 'topk' or 'probabilistic'
TOP_K = 2  # Only used for top-k selection
MUTATION_RATE = 0.2  # Probability of flipping strategy per agent per generation
NUM_ROUNDS = 10  # Number of rounds per simulation

COOP_GOAL = "Maximize group reward in the public goods game"
SELFISH_GOAL = "Maximize personal reward in the public goods game"

# === Measurement channels ===
MEASUREMENT_CHANNELS = {
    'generation_summary': 'evolutionary_generation_summary',
    'population_dynamics': 'evolutionary_population_dynamics',
    'selection_pressure': 'evolutionary_selection_pressure',
    'individual_scores': 'evolutionary_individual_scores',
    'strategy_distribution': 'evolutionary_strategy_distribution',
    'fitness_stats': 'evolutionary_fitness_statistics',
    'mutation_events': 'evolutionary_mutation_events',
    'convergence_metrics': 'evolutionary_convergence_metrics',
}


def setup_measurements() -> measurements_lib.Measurements:
    """Setup measurement channels for evolutionary tracking."""
    measurements = measurements_lib.Measurements()

    # Initialize all measurement channels
    for channel_name in MEASUREMENT_CHANNELS.values():
        measurements.get_channel(channel_name)  # Creates channel if it doesn't exist

    return measurements


def make_agent_config(name, goal):
    """Helper to create a basic agent config with a name and goal."""
    return basic_with_plan.Entity(
        params={
            'name': name,
            'goal': goal,
        }
    )


def initialize_population(pop_size: int) -> Dict[str, basic_with_plan.Entity]:
    """Initialize a population with half cooperative, half selfish agents."""
    agent_configs = {}
    for i in range(pop_size):
        name = f"Agent_{i+1}"
        if i < pop_size // 2:
            goal = COOP_GOAL
        else:
            goal = SELFISH_GOAL
        agent_configs[name] = make_agent_config(name, goal)
    return agent_configs


def extract_scores_from_simulation(raw_log: list) -> Dict[str, float]:
    """Extract actual scores from simulation raw log."""
    # Find all "Player Scores" entries in the raw log
    score_entries = helper_functions_lib.find_data_in_nested_structure(raw_log, "Player Scores")

    if not score_entries:
        # Fallback to zero scores if no scores found
        return {}

    # Get the final scores (last entry should have cumulative scores)
    final_scores = score_entries[-1] if score_entries else {}

    # Ensure we return a dict with float values
    return {name: float(score) for name, score in final_scores.items()}


def run_generation(agent_configs: Dict[str, basic_with_plan.Entity], measurements: Optional[measurements_lib.Measurements] = None) -> Dict[str, float]:
    """Run a simulation and return agent scores."""
    gm_key = "game_master"
    gm_prefab = PublicGoodsGameMaster(
        params={
            'name': 'public_goods_rules',
            # Use default scenes and payoff logic from the prefab
        }
    )
    from concordia.typing import prefab as prefab_lib
    config = prefab_lib.Config(
        instances=[
            *[prefab_lib.InstanceConfig(
                prefab=name,  # Use string key
                role=prefab_lib.Role.ENTITY,
                params={k: str(v) for k, v in agent_config.params.items()},
            ) for name, agent_config in agent_configs.items()],
            prefab_lib.InstanceConfig(
                prefab=gm_key,  # Use string key
                role=prefab_lib.Role.GAME_MASTER,
                params={k: str(v) for k, v in gm_prefab.params.items()},
            ),
        ],
        default_premise="A public goods game is played among four agents. Each round, agents choose whether to contribute to a common pool. The pool is multiplied and shared.",
        default_max_steps=NUM_ROUNDS,
        prefabs={**agent_configs, gm_key: gm_prefab},  # All keys are strings
    )
    model = no_language_model.NoLanguageModel()  # Use a dummy model for testing
    embedder = lambda text: np.random.rand(16)  # Dummy embedder
    engine = simultaneous.Simultaneous()
    sim = simulation_generic.Simulation(
        config=config,
        model=model,
        embedder=embedder,
        engine=engine,
    )

    # Create raw_log to capture simulation data
    raw_log = []
    sim.play(raw_log=raw_log)

    # Extract actual scores from the simulation
    scores = extract_scores_from_simulation(raw_log)

    # Fallback to agent names with zero scores if extraction fails
    if not scores:
        scores = {name: 0.0 for name in agent_configs}

    # Log individual scores to measurements
    if measurements is not None:
        for agent_name, score in scores.items():
            measurements.publish_datum(
                MEASUREMENT_CHANNELS['individual_scores'],
                {
                    'agent_name': agent_name,
                    'score': score,
                    'goal': agent_configs[agent_name].params['goal'],
                    'timestamp': datetime.datetime.now().isoformat(),
                }
            )

    return scores


def select_survivors(agent_configs: Dict[str, basic_with_plan.Entity], scores: Dict[str, float], method: str, k: int, measurements: Optional[measurements_lib.Measurements] = None) -> Dict[str, basic_with_plan.Entity]:
    """Select survivors using top-k or probabilistic selection."""
    if method == 'topk':
        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        survivors = dict((name, agent_configs[name]) for name, _ in sorted_agents[:k])
    elif method == 'probabilistic':
        total_score = sum(max(0, s) for s in scores.values())
        if total_score == 0:
            # If all scores are zero, pick randomly
            survivors = dict(random.sample(list(agent_configs.items()), k))
        else:
            chosen = set()
            while len(chosen) < k:
                pick = random.choices(
                    population=list(agent_configs.keys()),
                    weights=[max(0, scores[name]) for name in agent_configs],
                    k=1
                )[0]
                chosen.add(pick)
            survivors = {name: agent_configs[name] for name in chosen}
    else:
        raise ValueError(f"Unknown selection method: {method}")

    # Log selection pressure metrics
    if measurements is not None:
        survivor_scores = [scores[name] for name in survivors.keys()]
        eliminated_scores = [scores[name] for name in agent_configs.keys() if name not in survivors]

        measurements.publish_datum(
            MEASUREMENT_CHANNELS['selection_pressure'],
            {
                'method': method,
                'survivors_count': len(survivors),
                'eliminated_count': len(eliminated_scores),
                'survivor_scores': survivor_scores,
                'eliminated_scores': eliminated_scores,
                'selection_intensity': (max(survivor_scores) - min(eliminated_scores)) if eliminated_scores else 0.0,
                'timestamp': datetime.datetime.now().isoformat(),
            }
        )

    return survivors


def mutate_agents(survivors: Dict[str, basic_with_plan.Entity], pop_size: int, mutation_rate: float, measurements: Optional[measurements_lib.Measurements] = None) -> Dict[str, basic_with_plan.Entity]:
    """Create next generation by mutating survivors and cloning to fill population."""
    new_agents = {}
    survivor_names = list(survivors.keys())
    mutation_events = []

    for i in range(pop_size):
        parent_name = random.choice(survivor_names)
        parent = survivors[parent_name]
        # Copy goal and possibly mutate
        original_goal = parent.params['goal']
        goal = original_goal
        mutated = False

        if random.random() < mutation_rate:
            goal = COOP_GOAL if goal == SELFISH_GOAL else SELFISH_GOAL
            mutated = True

        name = f"Agent_{i+1}"
        new_agents[name] = make_agent_config(name, goal)

        # Track mutation events
        if mutated:
            mutation_events.append({
                'agent_name': name,
                'parent_name': parent_name,
                'original_goal': original_goal,
                'mutated_goal': goal,
                'timestamp': datetime.datetime.now().isoformat(),
            })

    # Log mutation events
    if measurements is not None:
        for event in mutation_events:
            measurements.publish_datum(MEASUREMENT_CHANNELS['mutation_events'], event)

        # Log overall mutation statistics
        measurements.publish_datum(
            MEASUREMENT_CHANNELS['population_dynamics'],
            {
                'mutation_rate': mutation_rate,
                'mutations_occurred': len(mutation_events),
                'mutation_frequency': len(mutation_events) / pop_size,
                'parent_distribution': {name: sum(1 for event in mutation_events if event['parent_name'] == name) for name in survivor_names},
                'timestamp': datetime.datetime.now().isoformat(),
            }
        )

    return new_agents


def log_generation(generation: int, agent_configs: Dict[str, basic_with_plan.Entity], scores: Dict[str, float], measurements: Optional[measurements_lib.Measurements] = None):
    """Log generation statistics and publish to measurements."""
    coop = sum(1 for a in agent_configs.values() if a.params['goal'] == COOP_GOAL)
    selfish = sum(1 for a in agent_configs.values() if a.params['goal'] == SELFISH_GOAL)
    total_agents = len(agent_configs)
    cooperation_rate = coop / total_agents

    # Calculate fitness statistics
    score_values = list(scores.values())
    avg_score = sum(score_values) / len(score_values) if score_values else 0.0
    max_score = max(score_values) if score_values else 0.0
    min_score = min(score_values) if score_values else 0.0

    # Calculate strategy-specific statistics
    coop_scores = [score for name, score in scores.items() if agent_configs[name].params['goal'] == COOP_GOAL]
    selfish_scores = [score for name, score in scores.items() if agent_configs[name].params['goal'] == SELFISH_GOAL]

    avg_coop_score = sum(coop_scores) / len(coop_scores) if coop_scores else 0.0
    avg_selfish_score = sum(selfish_scores) / len(selfish_scores) if selfish_scores else 0.0

    # Console output
    print(f"\nGeneration {generation}:")
    print(f"  Cooperative: {coop}, Selfish: {selfish}")
    print(f"  Scores: {scores}")
    print(f"  Fraction cooperative: {cooperation_rate:.2f}")
    print(f"  Avg scores - Cooperative: {avg_coop_score:.2f}, Selfish: {avg_selfish_score:.2f}")

    # Log to measurements
    if measurements is not None:
        # Generation summary
        measurements.publish_datum(
            MEASUREMENT_CHANNELS['generation_summary'],
            {
                'generation': generation,
                'total_agents': total_agents,
                'cooperative_agents': coop,
                'selfish_agents': selfish,
                'cooperation_rate': cooperation_rate,
                'avg_score': avg_score,
                'max_score': max_score,
                'min_score': min_score,
                'timestamp': datetime.datetime.now().isoformat(),
            }
        )

        # Strategy distribution
        measurements.publish_datum(
            MEASUREMENT_CHANNELS['strategy_distribution'],
            {
                'generation': generation,
                'cooperative_count': coop,
                'selfish_count': selfish,
                'cooperative_fraction': cooperation_rate,
                'selfish_fraction': 1.0 - cooperation_rate,
                'timestamp': datetime.datetime.now().isoformat(),
            }
        )

        # Fitness statistics
        measurements.publish_datum(
            MEASUREMENT_CHANNELS['fitness_stats'],
            {
                'generation': generation,
                'avg_cooperative_score': avg_coop_score,
                'avg_selfish_score': avg_selfish_score,
                'score_differential': avg_coop_score - avg_selfish_score,
                'fitness_variance': sum((s - avg_score) ** 2 for s in score_values) / len(score_values) if score_values else 0.0,
                'timestamp': datetime.datetime.now().isoformat(),
            }
        )


def evolutionary_main():
    """Run the evolutionary simulation with measurements tracking."""
    # Setup measurements
    measurements = setup_measurements()

    # Initialize population
    agent_configs = initialize_population(POP_SIZE)

    # Log initial population state
    print("Starting evolutionary simulation...")
    print(f"Population size: {POP_SIZE}, Generations: {NUM_GENERATIONS}")
    print(f"Selection method: {SELECTION_METHOD}, Mutation rate: {MUTATION_RATE}")

    # Run evolution
    for generation in range(1, NUM_GENERATIONS + 1):
        scores = run_generation(agent_configs, measurements)
        log_generation(generation, agent_configs, scores, measurements)
        survivors = select_survivors(agent_configs, scores, SELECTION_METHOD, TOP_K, measurements)
        agent_configs = mutate_agents(survivors, POP_SIZE, MUTATION_RATE, measurements)

    # Log final convergence metrics
    final_coop_rate = sum(1 for a in agent_configs.values() if a.params['goal'] == COOP_GOAL) / POP_SIZE
    measurements.publish_datum(
        MEASUREMENT_CHANNELS['convergence_metrics'],
        {
            'final_generation': NUM_GENERATIONS,
            'final_cooperation_rate': final_coop_rate,
            'converged_to_cooperation': final_coop_rate > 0.8,
            'converged_to_selfishness': final_coop_rate < 0.2,
            'simulation_parameters': {
                'population_size': POP_SIZE,
                'generations': NUM_GENERATIONS,
                'selection_method': SELECTION_METHOD,
                'mutation_rate': MUTATION_RATE,
                'top_k': TOP_K,
            },
            'timestamp': datetime.datetime.now().isoformat(),
        }
    )

    print("\nEvolutionary simulation complete.")
    print(f"Final cooperation rate: {final_coop_rate:.2f}")
    print(f"Available measurement channels: {list(measurements.available_channels())}")

    return measurements


def get_measurement_summary(measurements: Optional[measurements_lib.Measurements]) -> Dict[str, Any]:
    """Get a summary of all measurement data collected during evolution."""
    summary = {}

    if measurements is None:
        return summary

    for channel_key, channel_name in MEASUREMENT_CHANNELS.items():
        channel_data = measurements.get_channel(channel_name)
        summary[channel_key] = {
            'channel_name': channel_name,
            'total_entries': len(channel_data),
            'latest_entry': channel_data[-1] if channel_data else None,
            'sample_entries': channel_data[:3] if len(channel_data) > 3 else channel_data,
        }

    return summary


def export_measurements_to_dict(measurements: Optional[measurements_lib.Measurements]) -> Dict[str, list]:
    """Export all measurement data to a dictionary for analysis."""
    export_data = {}

    if measurements is None:
        return export_data

    for channel_key, channel_name in MEASUREMENT_CHANNELS.items():
        export_data[channel_key] = measurements.get_channel(channel_name)

    return export_data


if __name__ == "__main__":
    measurements = evolutionary_main()

    # Print measurement summary
    print("\n=== Measurement Summary ===")
    summary = get_measurement_summary(measurements)
    for channel_key, info in summary.items():
        print(f"{channel_key}: {info['total_entries']} entries")

    # Example: Access specific measurements
    print("\n=== Example: Final Generation Data ===")
    final_gen_data = measurements.get_last_datum(MEASUREMENT_CHANNELS['generation_summary'])
    if final_gen_data:
        print(f"Final cooperation rate: {final_gen_data['cooperation_rate']:.2f}")
        print(f"Average score: {final_gen_data['avg_score']:.2f}")
