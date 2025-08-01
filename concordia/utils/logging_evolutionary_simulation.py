#!/usr/bin/env python3
"""
Logging-enabled Evolutionary Simulation
Wrapper around the main evolutionary simulation that captures all LLM interactions.
"""

import logging
from typing import Dict, Optional
from concordia.typing import evolutionary as evolutionary_types
from concordia.utils import measurements as measurements_lib
from concordia.utils.generation_logger import get_generation_logger, reset_generation_logger
from concordia.utils.logging_language_model import wrap_language_model_with_logging

logger = logging.getLogger(__name__)

def logging_evolutionary_main(
    config: evolutionary_types.EvolutionConfig,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = 5,
    resume_from_checkpoint: bool = False,
) -> measurements_lib.Measurements:
    """
    Enhanced evolutionary simulation with comprehensive LLM interaction logging.
    
    This is a wrapper around the original evolutionary_main that:
    1. Resets the generation logger
    2. Wraps the language model with logging capabilities  
    3. Updates generation/round numbers during simulation
    4. Captures all LLM interactions for export
    
    Args:
        config: Evolution configuration
        checkpoint_dir: Directory for checkpoints
        checkpoint_interval: Save checkpoint every N generations
        resume_from_checkpoint: Whether to resume from checkpoint
        
    Returns:
        Measurements object with all simulation data
    """
    
    # Import here to avoid circular imports
    from examples.evolutionary_simulation import (
        evolutionary_main, setup_language_model, setup_embedder, 
        run_generation as original_run_generation, 
        MEASUREMENT_CHANNELS
    )
    from concordia.prefabs.entity import basic_with_plan
    from concordia.prefabs.game_master.public_goods_game_master import PublicGoodsGameMaster
    from concordia.prefabs.simulation import generic as simulation_generic
    from concordia.environment.engines import simultaneous
    from concordia.typing import prefab as prefab_lib
    
    # Reset generation logger for fresh start
    reset_generation_logger()
    generation_logger = get_generation_logger()
    logger.info("🔄 Generation logger reset - ready to capture LLM interactions")
    
    # We need to patch the run_generation function to inject our logging
    def logging_run_generation(
        agent_configs: Dict[str, basic_with_plan.Entity],
        config: evolutionary_types.EvolutionConfig,
        measurements: Optional[measurements_lib.Measurements] = None,
    ) -> Dict[str, float]:
        """Enhanced run_generation with LLM logging."""
        
        gm_key = 'game_master'
        gm_prefab = PublicGoodsGameMaster(
            params={
                'name': 'public_goods_rules',
            }
        )
        
        # Setup base language model and embedder
        base_model = setup_language_model(config)
        embedder = setup_embedder(config)
        
        # Wrap the language model with logging - this is the key integration!
        # We'll use a dynamic agent name that gets updated during simulation
        logging_model = wrap_language_model_with_logging(base_model, "Agent_Unknown")
        
        logger.info(f"🎯 Language model wrapped with logging for generation capture")
        
        sim_config = prefab_lib.Config(
            instances=[
                *[
                    prefab_lib.InstanceConfig(
                        prefab=name,
                        role=prefab_lib.Role.ENTITY,
                        params={k: str(v) for k, v in agent_config.params.items()},
                    )
                    for name, agent_config in agent_configs.items()
                ],
                prefab_lib.InstanceConfig(
                    prefab=gm_key,
                    role=prefab_lib.Role.GAME_MASTER,
                    params={k: str(v) for k, v in gm_prefab.params.items()},
                ),
            ],
            default_premise=(
                'A public goods game is played among four agents. Each round, agents'
                ' choose whether to contribute to a common pool. The pool is'
                ' multiplied and shared.'
            ),
            default_max_steps=config.num_rounds,
            prefabs={**agent_configs, gm_key: gm_prefab},
        )
        
        engine = simultaneous.Simultaneous()
        sim = simulation_generic.Simulation(
            config=sim_config,
            model=logging_model,  # Use the logging-wrapped model!
            embedder=embedder,
            engine=engine,
        )
        
        # Run the simulation
        logger.debug(f"🎮 Starting simulation run with {len(agent_configs)} agents")
        raw_log = []
        sim.play(raw_log=raw_log)
        logger.debug(f"✅ Simulation run completed")
        
        # Extract scores using the original function
        from examples.evolutionary_simulation import extract_scores_from_simulation
        return extract_scores_from_simulation(raw_log)
    
    # Now we need to run a modified version of the evolutionary loop with generation tracking
    from examples.evolutionary_simulation import (
        initialize_population, select_survivors, mutate_agents, log_generation,
        setup_measurements, export_measurements_to_dict, _restore_agent_from_checkpoint_data
    )
    from concordia.utils import checkpointing
    import datetime
    from collections import defaultdict
    
    start_generation = 1
    measurements = setup_measurements()
    agent_configs = None
    
    # Checkpoint handling (same as original)
    if resume_from_checkpoint and checkpoint_dir:
        latest_checkpoint = checkpointing.find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            try:
                checkpoint_data = checkpointing.load_evolutionary_checkpoint(
                    latest_checkpoint,
                    setup_measurements_fn=setup_measurements,
                    make_agent_config_fn=_restore_agent_from_checkpoint_data,
                    measurement_channels=MEASUREMENT_CHANNELS
                )
                start_generation = checkpoint_data['generation'] + 1
                agent_configs = checkpoint_data['agent_configs']
                measurements = checkpoint_data['measurements']
                logger.info('Resuming from generation %d', start_generation)
            except Exception as e:
                logger.error('Failed to load checkpoint: %s', e)
                logger.info('Starting fresh simulation...')
                agent_configs = None
    
    # Initialize population if not loaded from checkpoint
    if agent_configs is None:
        agent_configs = initialize_population(config)
        start_generation = 1
    
    # Log initial state
    logger.info('Starting evolutionary simulation with LLM logging...')
    logger.info('Population size: %d, Generations: %d', config.pop_size, config.num_generations)
    logger.info('Selection method: %s, Mutation rate: %.2f', config.selection_method, config.mutation_rate)
    
    # Enhanced evolutionary loop with generation tracking
    for generation in range(start_generation, config.num_generations + 1):
        logger.info(f"🧬 Starting Generation {generation}")
        
        # Update generation logger
        generation_logger.set_generation(generation)
        
        # Run generation with logging
        scores = logging_run_generation(agent_configs, config, measurements)
        
        # Log generation results
        log_generation(generation, agent_configs, scores, measurements)
        
        # Log the generation completion
        generation_stats = generation_logger.get_statistics()
        logger.info(f"✅ Generation {generation} complete - captured {generation_stats.get('total_interactions', 0)} LLM interactions")
        
        # Save checkpoint if requested
        if checkpoint_dir and generation % checkpoint_interval == 0:
            checkpointing.save_evolutionary_checkpoint(
                generation, agent_configs, measurements, checkpoint_dir, config,
                export_measurements_fn=export_measurements_to_dict
            )
        
        # Selection and mutation for next generation
        if generation < config.num_generations:
            survivors = select_survivors(
                agent_configs, scores, config.selection_method, config.top_k, measurements
            )
            agent_configs = mutate_agents(survivors, config, measurements)
    
    # Final logging
    from examples.evolutionary_simulation import Strategy
    final_coop_rate = (
        sum(1 for a in agent_configs.values() if a.params['goal'] == Strategy.COOPERATIVE.value)
        / config.pop_size
    )
    
    measurements.publish_datum(
        MEASUREMENT_CHANNELS['convergence_metrics'],
        {
            'final_generation': config.num_generations,
            'final_cooperation_rate': final_coop_rate,
            'converged_to_cooperation': final_coop_rate > 0.8,
            'converged_to_selfishness': final_coop_rate < 0.2,
            'simulation_parameters': {
                'population_size': config.pop_size,
                'generations': config.num_generations,
                'selection_method': config.selection_method,
                'mutation_rate': config.mutation_rate,
                'top_k': config.top_k,
            },
            'timestamp': datetime.datetime.now().isoformat(),
        },
    )
    
    # Final statistics
    final_stats = generation_logger.get_statistics()
    logger.info("🎉 Evolutionary simulation with logging completed!")
    logger.info(f"📊 Total LLM interactions captured: {final_stats.get('total_interactions', 0)}")
    logger.info(f"📝 Total words generated: {final_stats.get('text_stats', {}).get('total_words', 0):,}")
    logger.info(f"🔤 Total characters generated: {final_stats.get('text_stats', {}).get('total_characters', 0):,}")
    
    return measurements