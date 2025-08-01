#!/usr/bin/env python3
"""
Test simulation with working Gemma 7B to verify the enhanced results export works.
This will create a baseline working simulation result.
"""

import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_working_gemma_7b_simulation():
    """Run a known working simulation with Gemma 7B to verify export pipeline."""
    try:
        logger.info("🚀 Starting working Gemma 7B simulation...")
        
        from concordia.typing import evolutionary as evolutionary_types
        from concordia.utils.logging_evolutionary_simulation import logging_evolutionary_main
        from concordia.utils.enhanced_results_exporter import EnhancedResultsExporter
        from concordia.utils.generation_logger import get_generation_logger
        
        # Known working Gemma 7B configuration
        WORKING_CONFIG = evolutionary_types.EvolutionConfig(
            pop_size=2,  # Small for speed
            num_generations=2,  # Just 2 generations
            selection_method='topk',
            top_k=1,
            mutation_rate=0.1,
            num_rounds=2,  # Minimal rounds
            api_type='pytorch_gemma',
            model_name='google/gemma-7b-it',  # Known working model
            embedder_name='all-mpnet-base-v2',
            device='mps',
            disable_language_model=False,
        )
        
        logger.info("Starting working Gemma 7B simulation...")
        start_time = time.time()
        measurements = logging_evolutionary_main(config=WORKING_CONFIG)
        end_time = time.time()
        
        logger.info("✅ Working simulation completed!")
        
        # Enhanced results export
        logger.info("📊 Exporting enhanced structured results...")
        exporter = EnhancedResultsExporter()
        
        # Simple results data preparation
        results_data = {
            'config': {
                'model_name': WORKING_CONFIG.model_name,
                'api_type': WORKING_CONFIG.api_type,
                'pop_size': WORKING_CONFIG.pop_size,
                'num_generations': WORKING_CONFIG.num_generations,
                'num_rounds': WORKING_CONFIG.num_rounds,
                'selection_method': WORKING_CONFIG.selection_method,
                'mutation_rate': WORKING_CONFIG.mutation_rate,
                'device': WORKING_CONFIG.device,
                'disable_language_model': WORKING_CONFIG.disable_language_model,
            },
            'generations': [],
            'final_cooperation_rate': 0.5,
            'performance': {}
        }
        
        # Process measurements
        try:
            gen_summaries = measurements.get_channel('evolutionary_generation_summary')
            for gen_data in gen_summaries:
                gen_info = {
                    'generation': len(results_data['generations']) + 1,
                    'scores': gen_data.get('agent_scores', {}),
                    'cooperative_count': gen_data.get('cooperative_count', 0),
                    'selfish_count': gen_data.get('selfish_count', 0),
                    'cooperation_rate': gen_data.get('cooperation_rate', 0),
                }
                results_data['generations'].append(gen_info)
            
            if results_data['generations']:
                results_data['final_cooperation_rate'] = results_data['generations'][-1]['cooperation_rate']
        except Exception as e:
            logger.warning(f"Could not extract generation data: {e}")
        
        # Metadata
        metadata = {
            'simulation_info': {
                'model_name': WORKING_CONFIG.model_name,
                'api_type': WORKING_CONFIG.api_type,
                'device': WORKING_CONFIG.device,
                'language_model_active': not WORKING_CONFIG.disable_language_model,
            },
            'performance': {
                'duration_seconds': round(end_time - start_time, 2),
                'avg_generation_time': round((end_time - start_time) / WORKING_CONFIG.num_generations, 2),
            },
            'experiment_parameters': {
                'population_size': WORKING_CONFIG.pop_size,
                'generations': WORKING_CONFIG.num_generations,
                'rounds_per_generation': WORKING_CONFIG.num_rounds,
                'selection_method': WORKING_CONFIG.selection_method,
                'mutation_rate': WORKING_CONFIG.mutation_rate,
            }
        }
        
        # Get generation log
        generation_log = get_generation_logger().get_all_interactions()
        logger.info(f"📝 Captured {len(generation_log)} LLM interactions")
        
        # Export results
        results_folder = exporter.export_complete_results(
            model_name=WORKING_CONFIG.model_name,
            results_data=results_data,
            analysis_data=results_data,  # Simple analysis
            metadata=metadata,
            generation_log=generation_log
        )
        
        logger.info(f"🎉 Working simulation results exported to: {results_folder}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Working simulation failed: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    print("🧪 Running working Gemma 7B simulation to verify export pipeline")
    print("=" * 70)
    
    success = run_working_gemma_7b_simulation()
    
    if success:
        print("\n🎉 SUCCESS: Working simulation completed!")
        print("🔍 Check simulation_results/ for new structured results!")
    else:
        print("\n❌ Working simulation failed")