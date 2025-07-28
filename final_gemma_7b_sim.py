#!/usr/bin/env python3
"""
Final fixed Gemma 7B simulation - addresses all identified issues.
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

def test_fixed_gemma_generation() -> bool:
    """Test fixed Gemma 7B generation without problematic parameters."""
    try:
        logger.info("Testing FIXED Gemma 7B generation...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = 'google/gemma-7b-it'

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

def run_real_gemma_7b_simulation():
    """Run simulation with real Gemma 7B model."""
    try:
        # Test model first
        if not test_fixed_gemma_generation():
            logger.error("Model test failed, cannot run simulation")
            return False

        logger.info("🚀 Starting REAL Gemma 7B simulation...")

        from concordia.typing import evolutionary as evolutionary_types
        from examples.evolutionary_simulation import evolutionary_main

        # Real Gemma 7B Configuration (no timeout issues)
        REAL_GEMMA_CONFIG = evolutionary_types.EvolutionConfig(
            pop_size=4,  # Conservative size
            num_generations=3,  # Quick test
            selection_method='topk',
            top_k=2,
            mutation_rate=0.1,
            num_rounds=6,  # Shorter rounds
            api_type='pytorch_gemma',
            model_name='google/gemma-7b-it',
            embedder_name='all-mpnet-base-v2',
            device='auto',  # Let accelerate decide
            disable_language_model=False,
        )

        logger.info("Starting real Gemma 7B evolutionary simulation...")
        measurements = evolutionary_main(config=REAL_GEMMA_CONFIG)

        logger.info("✅ REAL Gemma 7B simulation completed!")
        logger.info("📊 Results saved to simulation_results/")

        return True

    except Exception as e:
        logger.error(f"❌ Real Gemma simulation failed: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    print("🎯 FINAL Gemma 7B Fix - Addressing timeout & accelerate issues")
    print("=" * 60)

    success = run_real_gemma_7b_simulation()

    if success:
        print("\n🎉 SUCCESS: Real Gemma 7B simulation completed!")
        print("🔍 Check simulation_results/ for the REAL LLM results!")
    else:
        print("\n❌ Still have issues - but we identified the exact problems!")
