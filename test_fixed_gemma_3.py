#!/usr/bin/env python3
"""
Test the fixed Gemma 3 generation with increased token limit.
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fixed_gemma_3_generation():
    """Test Gemma 3 with the fixed token limit."""
    try:
        logger.info("🧪 Testing fixed Gemma 3 generation...")
        
        from concordia.language_model.pytorch_gemma_model import PyTorchGemmaLanguageModel
        
        # Test with Gemma 3 1B
        model = PyTorchGemmaLanguageModel(
            model_name='google/gemma-3-1b-it',
            device='mps'
        )
        
        # Test with a typical Concordia prompt
        test_prompt = """Recent observations of Agent_1:
[observation] You are participating in a public goods game. Each round, you can choose to contribute 1 unit to a common pool or keep it for yourself. The total pool is multiplied by 1.6 and split equally among all players, regardless of contribution. Your payoff each round is your share of the pool plus any endowment you kept.
  
Question: What kind of person is Agent_1?
Answer: Agent_1 is"""
        
        logger.info("Testing text generation...")
        response = model.sample_text(test_prompt, max_tokens=50)
        
        logger.info(f"✅ Generated response: '{response}'")
        logger.info(f"📏 Response length: {len(response)} characters, {len(response.split())} words")
        
        # Test choice selection
        choice_prompt = "Will Agent_1 contribute to the public good?\n  (a) Yes\n  (b) No\nAnswer: ("
        responses = ['a', 'b']
        
        logger.info("Testing choice selection...")
        idx, choice, debug = model.sample_choice(choice_prompt, responses)
        
        logger.info(f"✅ Choice selected: {choice} (index {idx})")
        
        return len(response.split()) > 5  # Should generate more than 5 words
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Fixed Gemma 3 Generation")
    print("=" * 50)
    
    success = test_fixed_gemma_3_generation()
    
    if success:
        print("\n🎉 SUCCESS: Gemma 3 generation is now working properly!")
    else:
        print("\n❌ Generation still needs more work")