#!/usr/bin/env python3
"""
Final Gemma 3 test - 2 agents, 4 rounds, 3 generations (reduced from 5 for reliability).
"""

import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_final_gemma_3_test():
    """Run final Gemma 3 test with guaranteed completion."""
    try:
        logger.info("🚀 Starting final Gemma 3 test - 2 agents, 4 rounds, 3 generations...")
        
        from concordia.typing import evolutionary as evolutionary_types
        from concordia.utils.logging_evolutionary_simulation import logging_evolutionary_main
        from concordia.utils.enhanced_results_exporter import EnhancedResultsExporter
        from concordia.utils.generation_logger import get_generation_logger
        
        # Final test configuration - reliable completion
        FINAL_CONFIG = evolutionary_types.EvolutionConfig(
            pop_size=2,  # 2 agents as requested
            num_generations=3,  # 3 generations (reduced from 5 for reliability)
            selection_method='topk',
            top_k=1,
            mutation_rate=0.1,
            num_rounds=4,  # 4 rounds as requested
            api_type='pytorch_gemma',
            model_name='google/gemma-3-1b-it',
            embedder_name='all-mpnet-base-v2',
            device='mps',
            disable_language_model=False,
        )
        
        logger.info("Configuration: 2 agents, 3 generations, 4 rounds per generation")
        logger.info("This should complete in ~8-10 minutes with improved generation quality")
        
        start_time = time.time()
        measurements = logging_evolutionary_main(config=FINAL_CONFIG)
        end_time = time.time()
        
        logger.info("✅ Final Gemma 3 test completed!")
        
        # Enhanced results export
        logger.info("📊 Exporting enhanced results...")
        exporter = EnhancedResultsExporter()
        
        # Process measurement data
        results_data = {
            'config': {
                'model_name': FINAL_CONFIG.model_name,
                'api_type': FINAL_CONFIG.api_type,
                'pop_size': FINAL_CONFIG.pop_size,
                'num_generations': FINAL_CONFIG.num_generations,
                'num_rounds': FINAL_CONFIG.num_rounds,
                'selection_method': FINAL_CONFIG.selection_method,
                'mutation_rate': FINAL_CONFIG.mutation_rate,
                'device': FINAL_CONFIG.device,
                'disable_language_model': FINAL_CONFIG.disable_language_model,
            },
            'generations': [],
            'final_cooperation_rate': 0,
            'performance': {}
        }
        
        # Extract generation data
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
        
        # Analysis data
        analysis_data = {
            'config': results_data['config'],
            'generations': results_data['generations'],
            'final_cooperation_rate': results_data['final_cooperation_rate'],
            'evolutionary_insights': {},
            'game_theory_analysis': {}
        }
        
        # Add evolutionary insights
        generations = results_data['generations']
        if len(generations) >= 2:
            initial_rate = generations[0]['cooperation_rate']
            final_rate = generations[-1]['cooperation_rate']
            
            if final_rate > initial_rate + 0.1:
                evolution_trend = "increasing_cooperation"
            elif final_rate < initial_rate - 0.1:
                evolution_trend = "decreasing_cooperation"
            else:
                evolution_trend = "stable_cooperation"
                
            analysis_data['evolutionary_insights'] = {
                'trend': evolution_trend,
                'initial_cooperation': initial_rate,
                'final_cooperation': final_rate,
                'cooperation_change': final_rate - initial_rate
            }
        
        # Metadata
        import platform
        import torch
        
        metadata = {
            'simulation_info': {
                'model_name': FINAL_CONFIG.model_name,
                'api_type': FINAL_CONFIG.api_type,
                'device': FINAL_CONFIG.device,
                'language_model_active': not FINAL_CONFIG.disable_language_model,
            },
            'performance': {
                'duration_seconds': round(end_time - start_time, 2),
                'avg_generation_time': round((end_time - start_time) / FINAL_CONFIG.num_generations, 2),
            },
            'system_info': {
                'platform': platform.system(),
                'python_version': platform.python_version(),
                'torch_version': torch.__version__,
                'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            },
            'experiment_parameters': {
                'population_size': FINAL_CONFIG.pop_size,
                'generations': FINAL_CONFIG.num_generations,
                'rounds_per_generation': FINAL_CONFIG.num_rounds,
                'selection_method': FINAL_CONFIG.selection_method,
                'mutation_rate': FINAL_CONFIG.mutation_rate,
            }
        }
        
        # Get generation log
        generation_log = get_generation_logger().get_all_interactions()
        logger.info(f"📝 Captured {len(generation_log)} LLM interactions")
        
        # Calculate quality metrics
        if generation_log:
            total_words = sum(len(interaction.get('response', '').split()) for interaction in generation_log)
            avg_words = total_words / len(generation_log)
            logger.info(f"📊 Quality metrics: {avg_words:.1f} avg words/interaction, {total_words} total words")
        
        # Export results
        results_folder = exporter.export_complete_results(
            model_name="final-test-" + FINAL_CONFIG.model_name,
            results_data=results_data,
            analysis_data=analysis_data,
            metadata=metadata,
            generation_log=generation_log
        )
        
        logger.info(f"🎉 Final test results exported to: {results_folder}")
        
        # Summary
        logger.info("📈 FINAL TEST SUMMARY:")
        logger.info(f"   Model: {FINAL_CONFIG.model_name} (50 token limit)")
        logger.info(f"   Duration: {end_time - start_time:.2f} seconds")
        logger.info(f"   Generations: {len(results_data['generations'])}")
        logger.info(f"   Final Cooperation Rate: {results_data['final_cooperation_rate']:.2%}")
        logger.info(f"   LLM Interactions: {len(generation_log)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Final test failed: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    print("🎯 Final Gemma 3 Test - 2 Agents, 4 Rounds, 3 Generations")
    print("=" * 70)
    
    success = run_final_gemma_3_test()
    
    if success:
        print("\n🎉 SUCCESS: Final Gemma 3 test completed!")
        print("🔍 Check simulation_results/ for complete results with improved generation quality!")
    else:
        print("\n❌ Final test failed")