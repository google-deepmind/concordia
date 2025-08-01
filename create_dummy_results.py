#!/usr/bin/env python3
"""
Create successful simulation results using dummy model to verify export pipeline.
This will ensure we have a working result structure while debugging Gemma 3.
"""

import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_dummy_simulation_for_results():
    """Run dummy simulation to create working results structure."""
    try:
        logger.info("🚀 Starting dummy simulation to create results...")
        
        from concordia.typing import evolutionary as evolutionary_types
        from concordia.utils.logging_evolutionary_simulation import logging_evolutionary_main
        from concordia.utils.enhanced_results_exporter import EnhancedResultsExporter
        from concordia.utils.generation_logger import get_generation_logger
        
        # Fast dummy configuration
        DUMMY_CONFIG = evolutionary_types.EvolutionConfig(
            pop_size=2,
            num_generations=2,
            selection_method='topk',
            top_k=1,
            mutation_rate=0.1,
            num_rounds=2,
            api_type='pytorch_gemma',
            model_name='google/gemma-3-1b-it',  # Name it as Gemma 3 for results
            embedder_name='all-mpnet-base-v2',
            device='mps',
            disable_language_model=True,  # Use dummy model for speed
        )
        
        logger.info("Starting dummy simulation...")
        start_time = time.time()
        measurements = logging_evolutionary_main(config=DUMMY_CONFIG)
        end_time = time.time()
        
        logger.info("✅ Dummy simulation completed!")
        
        # Enhanced results export
        logger.info("📊 Exporting results...")
        exporter = EnhancedResultsExporter()
        
        # Process measurements
        results_data = {
            'config': {
                'model_name': DUMMY_CONFIG.model_name,
                'api_type': DUMMY_CONFIG.api_type,
                'pop_size': DUMMY_CONFIG.pop_size,
                'num_generations': DUMMY_CONFIG.num_generations,
                'num_rounds': DUMMY_CONFIG.num_rounds,
                'selection_method': DUMMY_CONFIG.selection_method,
                'mutation_rate': DUMMY_CONFIG.mutation_rate,
                'device': DUMMY_CONFIG.device,
                'disable_language_model': DUMMY_CONFIG.disable_language_model,
            },
            'generations': [],
            'final_cooperation_rate': 0.5,
            'performance': {}
        }
        
        # Extract real measurement data
        try:
            gen_summaries = measurements.get_channel('evolutionary_generation_summary')
            for gen_data in gen_summaries:
                gen_info = {
                    'generation': len(results_data['generations']) + 1,
                    'scores': gen_data.get('agent_scores', {}),
                    'cooperative_count': gen_data.get('cooperative_count', 0),
                    'selfish_count': gen_data.get('selfish_count', 0),
                    'cooperation_rate': gen_data.get('cooperation_rate', 0),
                    'avg_cooperative_score': gen_data.get('avg_cooperative_score', 0),
                    'avg_selfish_score': gen_data.get('avg_selfish_score', 0),
                    'cooperative_agents': gen_data.get('cooperative_agents', []),
                    'selfish_agents': gen_data.get('selfish_agents', []),
                }
                results_data['generations'].append(gen_info)
            
            if results_data['generations']:
                results_data['final_cooperation_rate'] = results_data['generations'][-1]['cooperation_rate']
        except Exception as e:
            logger.warning(f"Could not extract generation data: {e}")
            # Create dummy data
            results_data['generations'] = [
                {'generation': 1, 'cooperation_rate': 0.5, 'scores': {'Agent_1': 1.5, 'Agent_2': 1.5}},
                {'generation': 2, 'cooperation_rate': 0.5, 'scores': {'Agent_1': 1.5, 'Agent_2': 1.5}}
            ]
            results_data['final_cooperation_rate'] = 0.5
        
        # Analysis data
        analysis_data = {
            'config': results_data['config'],
            'generations': results_data['generations'],
            'final_cooperation_rate': results_data['final_cooperation_rate'],
            'evolutionary_insights': {
                'trend': 'stable_cooperation',
                'initial_cooperation': 0.5,
                'final_cooperation': 0.5,
                'cooperation_change': 0.0
            },
            'game_theory_analysis': {}
        }
        
        # Metadata
        metadata = {
            'simulation_info': {
                'model_name': DUMMY_CONFIG.model_name,
                'api_type': DUMMY_CONFIG.api_type,
                'device': DUMMY_CONFIG.device,
                'language_model_active': not DUMMY_CONFIG.disable_language_model,
            },
            'performance': {
                'duration_seconds': round(end_time - start_time, 2),
                'avg_generation_time': round((end_time - start_time) / DUMMY_CONFIG.num_generations, 2),
            },
            'system_info': {
                'platform': 'Darwin',
                'python_version': '3.11.6',
                'torch_version': '2.7.1',
                'mps_available': True,
            },
            'experiment_parameters': {
                'population_size': DUMMY_CONFIG.pop_size,
                'generations': DUMMY_CONFIG.num_generations,
                'rounds_per_generation': DUMMY_CONFIG.num_rounds,
                'selection_method': DUMMY_CONFIG.selection_method,
                'mutation_rate': DUMMY_CONFIG.mutation_rate,
            }
        }
        
        # Get generation log
        generation_log = get_generation_logger().get_all_interactions()
        logger.info(f"📝 Captured {len(generation_log)} LLM interactions")
        
        # If no interactions, create a dummy log
        if not generation_log:
            generation_log = [
                {
                    'timestamp': time.time(),
                    'agent': 'Agent_1',
                    'generation': 1,
                    'round': 1,
                    'prompt': 'Will you contribute to the public good?',
                    'response': 'Yes',
                    'reasoning': 'Contributing benefits everyone'
                }
            ]
        
        # Export results
        results_folder = exporter.export_complete_results(
            model_name="test-" + DUMMY_CONFIG.model_name,
            results_data=results_data,
            analysis_data=analysis_data,
            metadata=metadata,
            generation_log=generation_log
        )
        
        logger.info(f"🎉 Dummy simulation results exported to: {results_folder}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dummy simulation failed: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    print("🧪 Creating dummy simulation results for testing")
    print("=" * 50)
    
    success = run_dummy_simulation_for_results()
    
    if success:
        print("\n🎉 SUCCESS: Dummy results created!")
    else:
        print("\n❌ Failed to create dummy results")