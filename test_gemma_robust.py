#!/usr/bin/env python3
"""
Robust Gemma 7B testing and simulation execution.
Addresses memory, MPS, and generation issues systematically.
"""

import gc
import logging
import sys
import torch
import warnings
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_memory():
    """Clear GPU and system memory."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    logger.info("Memory cleared")

def test_mps_compatibility() -> bool:
    """Test if MPS is actually working for model operations."""
    try:
        if not torch.backends.mps.is_available():
            logger.warning("MPS not available")
            return False
        
        # Test basic MPS operations
        test_tensor = torch.randn(100, 100).to('mps')
        result = torch.matmul(test_tensor, test_tensor)
        result.cpu()  # Move back to CPU
        
        logger.info("✅ MPS compatibility test passed")
        return True
    except Exception as e:
        logger.error(f"❌ MPS compatibility test failed: {e}")
        return False

def test_model_generation(device: str = 'mps') -> bool:
    """Test if Gemma model can actually generate text."""
    try:
        logger.info(f"Testing Gemma 7B generation on {device}...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = 'google/gemma-7b-it'
        
        # Load with reduced memory usage
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map=device if device != 'mps' else None,
            low_cpu_mem_usage=True
        )
        
        if device == 'mps':
            model = model.to('mps')
        
        # Test generation with simple prompt
        test_prompt = "Hello, I am"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        if device == 'mps':
            inputs = {k: v.to('mps') for k, v in inputs.items()}
        
        logger.info("Testing generation...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                timeout=30  # 30 second timeout
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Generated text: '{generated_text}'")
        
        # Clean up
        del model
        del tokenizer
        clear_memory()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model generation test failed: {e}")
        clear_memory()
        return False

def get_optimal_config() -> Dict[str, Any]:
    """Get optimal configuration based on system capabilities."""
    config = {
        'pop_size': 4,  # Start smaller
        'num_generations': 5,  # Reduce generations
        'selection_method': 'topk',
        'top_k': 2,
        'mutation_rate': 0.2,
        'num_rounds': 8,  # Reduce rounds
        'api_type': 'pytorch_gemma',
        'model_name': 'google/gemma-7b-it',
        'embedder_name': 'all-mpnet-base-v2',
        'disable_language_model': False,
    }
    
    # Test MPS compatibility
    if test_mps_compatibility() and test_model_generation('mps'):
        config['device'] = 'mps'
        logger.info("✅ Using MPS device")
    elif test_model_generation('cpu'):
        config['device'] = 'cpu'
        logger.info("⚠️  Falling back to CPU device")
    else:
        logger.error("❌ Neither MPS nor CPU generation working, using dummy model")
        config['disable_language_model'] = True
        config['device'] = 'cpu'
    
    return config

def run_robust_simulation():
    """Run simulation with robust error handling."""
    try:
        logger.info("🚀 Starting robust Gemma 7B simulation...")
        
        # Clear memory first
        clear_memory()
        
        # Get optimal configuration
        config = get_optimal_config()
        
        logger.info(f"Configuration: {config}")
        
        # Import after configuration to ensure environment is ready
        from examples.evolutionary_simulation import evolutionary_main
        from concordia.typing import evolutionary as evolutionary_types
        
        # Create configuration
        simulation_config = evolutionary_types.EvolutionConfig(**config)
        
        logger.info("Starting evolutionary simulation...")
        measurements = evolutionary_main(config=simulation_config)
        
        logger.info("✅ Simulation completed successfully!")
        logger.info("📊 Results should be saved to simulation_results/")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")
        logger.exception("Full error traceback:")
        return False
    finally:
        clear_memory()

if __name__ == "__main__":
    print("🧠 Ultra-Robust Gemma 7B Simulation")
    print("=" * 50)
    
    success = run_robust_simulation()
    
    if success:
        print("\n✅ SUCCESS: Check simulation_results/ for output!")
    else:
        print("\n❌ FAILED: Check logs for details")
        sys.exit(1)