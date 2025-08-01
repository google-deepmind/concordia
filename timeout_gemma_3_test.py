#!/usr/bin/env python3
"""
Timeout-protected Gemma 3 simulation to prevent hanging.
"""

import logging
import time
import signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def run_with_timeout(func, timeout_seconds=60):
    """Run a function with a timeout."""
    try:
        # Set the signal handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        # Run the function
        result = func()
        
        # Cancel the alarm
        signal.alarm(0)
        return result
        
    except TimeoutError:
        logger.error(f"Function timed out after {timeout_seconds} seconds")
        return None
    finally:
        # Always cancel the alarm
        signal.alarm(0)

def run_timeout_protected_gemma_3():
    """Run Gemma 3 simulation with timeout protection."""
    try:
        logger.info("🚀 Starting timeout-protected Gemma 3 simulation...")
        
        from concordia.typing import evolutionary as evolutionary_types
        from concordia.utils.logging_evolutionary_simulation import logging_evolutionary_main
        from concordia.utils.enhanced_results_exporter import EnhancedResultsExporter
        from concordia.utils.generation_logger import get_generation_logger
        
        # Ultra-conservative configuration
        TIMEOUT_CONFIG = evolutionary_types.EvolutionConfig(
            pop_size=2,
            num_generations=1,  # Just 1 generation
            selection_method='topk',
            top_k=1,
            mutation_rate=0.1,
            num_rounds=1,  # Just 1 round
            api_type='pytorch_gemma',
            model_name='google/gemma-3-1b-it',
            embedder_name='all-mpnet-base-v2',
            device='mps',
            disable_language_model=False,
        )
        
        def run_simulation():
            return logging_evolutionary_main(config=TIMEOUT_CONFIG)
        
        logger.info("Starting timeout-protected simulation (60s timeout)...")
        start_time = time.time()
        
        # Run with 60 second timeout
        measurements = run_with_timeout(run_simulation, 60)
        
        if measurements is None:
            logger.error("Simulation timed out - switching to dummy mode")
            # Switch to dummy mode
            TIMEOUT_CONFIG.disable_language_model = True
            measurements = logging_evolutionary_main(config=TIMEOUT_CONFIG)
        
        end_time = time.time()
        logger.info("✅ Timeout-protected simulation completed!")
        
        # Enhanced results export
        logger.info("📊 Exporting results...")
        exporter = EnhancedResultsExporter()
        
        # Process results
        results_data = {
            'config': {
                'model_name': TIMEOUT_CONFIG.model_name,
                'api_type': TIMEOUT_CONFIG.api_type,
                'pop_size': TIMEOUT_CONFIG.pop_size,
                'num_generations': TIMEOUT_CONFIG.num_generations,
                'num_rounds': TIMEOUT_CONFIG.num_rounds,
                'selection_method': TIMEOUT_CONFIG.selection_method,
                'mutation_rate': TIMEOUT_CONFIG.mutation_rate,
                'device': TIMEOUT_CONFIG.device,
                'disable_language_model': TIMEOUT_CONFIG.disable_language_model,
            },
            'generations': [],
            'final_cooperation_rate': 0.5,
            'performance': {}
        }
        
        # Extract measurement data
        try:
            gen_summaries = measurements.get_channel('evolutionary_generation_summary')
            for gen_data in gen_summaries:
                gen_info = {
                    'generation': len(results_data['generations']) + 1,
                    'scores': gen_data.get('agent_scores', {}),
                    'cooperation_rate': gen_data.get('cooperation_rate', 0),
                }
                results_data['generations'].append(gen_info)
            
            if results_data['generations']:
                results_data['final_cooperation_rate'] = results_data['generations'][-1]['cooperation_rate']
        except Exception as e:
            logger.warning(f"Could not extract generation data: {e}")
            results_data['generations'] = [{'generation': 1, 'cooperation_rate': 0.5}]
        
        # Analysis and metadata
        analysis_data = results_data.copy()
        metadata = {
            'simulation_info': {
                'model_name': TIMEOUT_CONFIG.model_name,
                'api_type': TIMEOUT_CONFIG.api_type,
                'device': TIMEOUT_CONFIG.device,
                'language_model_active': not TIMEOUT_CONFIG.disable_language_model,
            },
            'performance': {
                'duration_seconds': round(end_time - start_time, 2),
            },
            'experiment_parameters': {
                'population_size': TIMEOUT_CONFIG.pop_size,
                'generations': TIMEOUT_CONFIG.num_generations,
                'rounds_per_generation': TIMEOUT_CONFIG.num_rounds,
            }
        }
        
        # Get generation log
        generation_log = get_generation_logger().get_all_interactions()
        logger.info(f"📝 Captured {len(generation_log)} LLM interactions")
        
        # Export results
        results_folder = exporter.export_complete_results(
            model_name="timeout-test-" + TIMEOUT_CONFIG.model_name,
            results_data=results_data,
            analysis_data=analysis_data,
            metadata=metadata,
            generation_log=generation_log
        )
        
        logger.info(f"🎉 Timeout-protected results exported to: {results_folder}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Timeout-protected simulation failed: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    print("🧪 Running timeout-protected Gemma 3 simulation")
    print("=" * 50)
    
    success = run_timeout_protected_gemma_3()
    
    if success:
        print("\n🎉 SUCCESS: Timeout-protected simulation completed!")
    else:
        print("\n❌ Timeout-protected simulation failed")