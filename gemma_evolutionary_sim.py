#!/usr/bin/env python3
"""
Gemma 3 Evolutionary Simulation - Multi-model parameter study baseline.
"""

import gc
import logging
from typing import Optional

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_memory():
    """Clear GPU and system memory."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def test_gemma_generation(model_name: str = 'google/gemma-3-1b-it') -> bool:
    """Test Gemma 3 generation without problematic parameters."""
    try:
        logger.info(f"Testing {model_name} generation...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with proper accelerate support
        logger.info("Loading model with accelerate...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",  # Let accelerate handle device mapping
            low_cpu_mem_usage=True
        )

        # Test generation WITHOUT timeout parameter
        test_prompt = "Hello, I am"
        inputs = tokenizer(test_prompt, return_tensors="pt")

        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logger.info("Testing generation (without timeout)...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
                # NO timeout parameter - this was the issue!
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Generated: '{generated_text}'")

        # Test with evolutionary-style prompt
        evo_prompt = "Will Agent_1 contribute to the public good? Answer Yes or No:"
        inputs = tokenizer(evo_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        evo_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Evolutionary response: '{evo_response}'")

        # Clean up
        del model
        del tokenizer
        clear_memory()

        return True

    except Exception as e:
        logger.error(f"❌ Fixed generation test failed: {e}")
        clear_memory()
        return False

def prepare_results_data(config, measurements):
    """Prepare results data for enhanced export."""
    from concordia.utils import measurements as measurements_lib

    # Extract measurement data
    channels = {
        'generation_summary': 'evolutionary_generation_summary',
        'population_dynamics': 'evolutionary_population_dynamics',
        'selection_pressure': 'evolutionary_selection_pressure',
        'individual_scores': 'evolutionary_individual_scores',
        'strategy_distribution': 'evolutionary_strategy_distribution',
        'fitness_stats': 'evolutionary_fitness_statistics',
    }

    results_data = {
        'config': {
            'model_name': config.model_name,
            'api_type': config.api_type,
            'pop_size': config.pop_size,
            'num_generations': config.num_generations,
            'num_rounds': config.num_rounds,
            'selection_method': config.selection_method,
            'mutation_rate': config.mutation_rate,
            'device': config.device,
            'disable_language_model': config.disable_language_model,
        },
        'generations': [],
        'final_cooperation_rate': 0,
        'performance': {}
    }

    # Process generation data from measurements
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

        # Set final cooperation rate
        if results_data['generations']:
            results_data['final_cooperation_rate'] = results_data['generations'][-1]['cooperation_rate']

    except Exception as e:
        logger.warning(f"Could not extract generation data: {e}")

    return results_data

def prepare_analysis_data(config, measurements, results_data):
    """Prepare detailed analysis data - recreates the comprehensive breakdown."""
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

    return analysis_data

def prepare_metadata(config, start_time, end_time):
    """Prepare simulation metadata."""
    import platform
    import torch

    metadata = {
        'simulation_info': {
            'model_name': config.model_name,
            'api_type': config.api_type,
            'device': config.device,
            'language_model_active': not config.disable_language_model,
        },
        'performance': {
            'duration_seconds': round(end_time - start_time, 2),
            'avg_generation_time': round((end_time - start_time) / config.num_generations, 2),
        },
        'system_info': {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        },
        'experiment_parameters': {
            'population_size': config.pop_size,
            'generations': config.num_generations,
            'rounds_per_generation': config.num_rounds,
            'selection_method': config.selection_method,
            'mutation_rate': config.mutation_rate,
        }
    }

    return metadata

def run_gemma_3_1b_simulation():
    """Run simulation with Gemma 3 1B model and enhanced results export."""
    try:
        # Test model first
        if not test_gemma_generation('google/gemma-3-1b-it'):
            logger.error("Model test failed, cannot run simulation")
            return False

        logger.info("🚀 Starting Gemma 3 1B baseline simulation...")

        from concordia.typing import evolutionary as evolutionary_types
        from concordia.utils.logging_evolutionary_simulation import logging_evolutionary_main
        from concordia.utils.enhanced_results_exporter import EnhancedResultsExporter
        from concordia.utils.generation_logger import get_generation_logger
        import time

        # Gemma 3 1B Configuration - Test with smaller, faster model
        GEMMA_3_1B_CONFIG = evolutionary_types.EvolutionConfig(
            pop_size=2,  # Small population for testing
            num_generations=2,  # Just 2 generations to verify pipeline works
            selection_method='topk',
            top_k=1,
            mutation_rate=0.1,
            num_rounds=2,  # Even fewer rounds for faster execution
            api_type='pytorch_gemma',
            model_name='google/gemma-3-1b-it',  # Gemma 3 1B instruction-tuned model
            embedder_name='all-mpnet-base-v2',
            device='mps',  # Mac GPU acceleration
            disable_language_model=False,
        )

        logger.info("Starting Gemma 3 1B evolutionary simulation with full LLM logging...")

        start_time = time.time()
        measurements = logging_evolutionary_main(config=GEMMA_3_1B_CONFIG)
        end_time = time.time()

        logger.info("✅ Gemma 3 1B simulation completed!")

        # Enhanced results export
        logger.info("📊 Exporting enhanced structured results...")
        exporter = EnhancedResultsExporter()

        # Prepare results data
        results_data = prepare_results_data(GEMMA_3_1B_CONFIG, measurements)
        analysis_data = prepare_analysis_data(GEMMA_3_1B_CONFIG, measurements, results_data)
        metadata = prepare_metadata(GEMMA_3_1B_CONFIG, start_time, end_time)

        # Get generation log
        from concordia.utils.generation_logger import get_generation_logger
        generation_log = get_generation_logger().get_all_interactions()

        logger.info(f"📝 Captured {len(generation_log)} LLM interactions during simulation")

        # Export with structured folders including generation log
        results_folder = exporter.export_complete_results(
            model_name=GEMMA_3_1B_CONFIG.model_name,
            results_data=results_data,
            analysis_data=analysis_data,
            metadata=metadata,
            generation_log=generation_log
        )

        logger.info(f"🎉 Enhanced results exported to: {results_folder}")
        return True

    except Exception as e:
        logger.error(f"❌ Gemma 3 1B simulation failed: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    print("🎯 Gemma 3 Multi-Model Parameter Study - Testing 1B Model")
    print("=" * 70)

    success = run_gemma_3_1b_simulation()

    if success:
        print("\n🎉 SUCCESS: Gemma 3 1B baseline simulation completed!")
        print("🔍 Check simulation_results/ for structured results and generation logs!")
    else:
        print("\n❌ Simulation failed - check logs for details")
