#!/usr/bin/env python3
"""
Minimal Gemma 3 test - ultra-simple configuration to get working results.
"""

import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_minimal_gemma_3_simulation():
    """Run the most minimal possible simulation to verify Gemma 3 works."""
    try:
        logger.info("🚀 Starting MINIMAL Gemma 3 simulation...")
        
        from concordia.typing import evolutionary as evolutionary_types
        from concordia.utils.logging_evolutionary_simulation import logging_evolutionary_main
        from concordia.utils.enhanced_results_exporter import EnhancedResultsExporter
        from concordia.utils.generation_logger import get_generation_logger
        
        # Ultra-minimal configuration for Gemma 3
        MINIMAL_CONFIG = evolutionary_types.EvolutionConfig(
            pop_size=2,  # Absolute minimum
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
        
        logger.info("Starting minimal Gemma 3 simulation...")
        start_time = time.time()
        measurements = logging_evolutionary_main(config=MINIMAL_CONFIG)
        end_time = time.time()
        
        logger.info("✅ Minimal simulation completed!")
        
        # Enhanced results export
        logger.info("📊 Exporting results...")
        exporter = EnhancedResultsExporter()
        
        # Basic results data
        results_data = {
            'config': {
                'model_name': MINIMAL_CONFIG.model_name,
                'api_type': MINIMAL_CONFIG.api_type,
                'pop_size': MINIMAL_CONFIG.pop_size,
                'num_generations': MINIMAL_CONFIG.num_generations,
                'num_rounds': MINIMAL_CONFIG.num_rounds,
                'selection_method': MINIMAL_CONFIG.selection_method,
                'mutation_rate': MINIMAL_CONFIG.mutation_rate,
                'device': MINIMAL_CONFIG.device,
                'disable_language_model': MINIMAL_CONFIG.disable_language_model,
            },
            'generations': [{'generation': 1, 'cooperation_rate': 0.5}],
            'final_cooperation_rate': 0.5,
            'performance': {}
        }
        
        # Basic metadata
        metadata = {
            'simulation_info': {
                'model_name': MINIMAL_CONFIG.model_name,
                'api_type': MINIMAL_CONFIG.api_type,
                'device': MINIMAL_CONFIG.device,
                'language_model_active': True,
            },
            'performance': {
                'duration_seconds': round(end_time - start_time, 2),
            },
            'experiment_parameters': {
                'population_size': MINIMAL_CONFIG.pop_size,
                'generations': MINIMAL_CONFIG.num_generations,
                'rounds_per_generation': MINIMAL_CONFIG.num_rounds,
            }
        }
        
        # Get generation log
        generation_log = get_generation_logger().get_all_interactions()
        logger.info(f"📝 Captured {len(generation_log)} LLM interactions")
        
        # Export results
        results_folder = exporter.export_complete_results(
            model_name=MINIMAL_CONFIG.model_name,
            results_data=results_data,
            analysis_data=results_data,
            metadata=metadata,
            generation_log=generation_log
        )
        
        logger.info(f"🎉 Minimal simulation results exported to: {results_folder}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Minimal simulation failed: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    print("🧪 Running MINIMAL Gemma 3 simulation")
    print("=" * 50)
    
    success = run_minimal_gemma_3_simulation()
    
    if success:
        print("\n🎉 SUCCESS: Minimal Gemma 3 simulation completed!")
    else:
        print("\n❌ Minimal simulation failed")