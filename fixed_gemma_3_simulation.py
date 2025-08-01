#!/usr/bin/env python3
"""
Run a proper Gemma 3 simulation with the fixed token generation to verify quality.
"""

import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_fixed_gemma_3_simulation():
    """Run simulation with fixed Gemma 3 generation."""
    try:
        logger.info("🚀 Starting fixed Gemma 3 simulation...")
        
        from concordia.typing import evolutionary as evolutionary_types
        from concordia.utils.logging_evolutionary_simulation import logging_evolutionary_main
        from concordia.utils.enhanced_results_exporter import EnhancedResultsExporter
        from concordia.utils.generation_logger import get_generation_logger
        
        # Test configuration with fixed generation
        FIXED_CONFIG = evolutionary_types.EvolutionConfig(
            pop_size=2,  # Small for quick test
            num_generations=2,  
            selection_method='topk',
            top_k=1,
            mutation_rate=0.1,
            num_rounds=2,  
            api_type='pytorch_gemma',
            model_name='google/gemma-3-1b-it',
            embedder_name='all-mpnet-base-v2',
            device='mps',
            disable_language_model=False,
        )
        
        logger.info("Starting fixed Gemma 3 simulation...")
        start_time = time.time()
        measurements = logging_evolutionary_main(config=FIXED_CONFIG)
        end_time = time.time()
        
        logger.info("✅ Fixed simulation completed!")
        
        # Enhanced results export
        exporter = EnhancedResultsExporter()
        
        # Process results
        results_data = {
            'config': {
                'model_name': FIXED_CONFIG.model_name,
                'api_type': FIXED_CONFIG.api_type,
                'pop_size': FIXED_CONFIG.pop_size,
                'num_generations': FIXED_CONFIG.num_generations,
                'num_rounds': FIXED_CONFIG.num_rounds,
                'selection_method': FIXED_CONFIG.selection_method,
                'mutation_rate': FIXED_CONFIG.mutation_rate,
                'device': FIXED_CONFIG.device,
                'disable_language_model': FIXED_CONFIG.disable_language_model,
            },
            'generations': [],
            'final_cooperation_rate': 0.5,
            'performance': {}
        }
        
        # Extract generation data
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
        
        # Metadata
        metadata = {
            'simulation_info': {
                'model_name': FIXED_CONFIG.model_name,
                'api_type': FIXED_CONFIG.api_type,
                'device': FIXED_CONFIG.device,
                'language_model_active': True,
            },
            'performance': {
                'duration_seconds': round(end_time - start_time, 2),
            },
            'experiment_parameters': {
                'population_size': FIXED_CONFIG.pop_size,
                'generations': FIXED_CONFIG.num_generations,
                'rounds_per_generation': FIXED_CONFIG.num_rounds,
            }
        }
        
        # Get generation log to check quality
        generation_log = get_generation_logger().get_all_interactions()
        logger.info(f"📝 Captured {len(generation_log)} LLM interactions")
        
        # Analyze generation quality
        if generation_log:
            total_words = sum(len(interaction.get('response', '').split()) for interaction in generation_log)
            avg_words = total_words / len(generation_log) if generation_log else 0
            logger.info(f"📊 Average words per interaction: {avg_words:.1f}")
            logger.info(f"📈 Total words generated: {total_words}")
        
        # Export results
        results_folder = exporter.export_complete_results(
            model_name="fixed-" + FIXED_CONFIG.model_name,
            results_data=results_data,
            analysis_data=results_data,
            metadata=metadata,
            generation_log=generation_log
        )
        
        logger.info(f"🎉 Fixed simulation results exported to: {results_folder}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fixed simulation failed: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    print("🧪 Running Fixed Gemma 3 Simulation")
    print("=" * 50)
    
    success = run_fixed_gemma_3_simulation()
    
    if success:
        print("\n🎉 SUCCESS: Fixed Gemma 3 simulation completed!")
        print("🔍 Check generation log for improved response quality!")
    else:
        print("\n❌ Fixed simulation failed")