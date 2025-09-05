#!/usr/bin/env python3
"""
Test negotiation framework with Together AI models (Llama, Mistral, etc).

This demonstrates all negotiation components working with open-source models
through Together AI's API.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import hashlib
import numpy as np

# Import Together AI adapter
from together_ai_adapter import create_together_model, TogetherAIModel

# Core Concordia imports
from concordia.language_model import retry_wrapper
from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.typing import prefab as prefab_lib
from concordia.prefabs.simulation import generic as simulation
from concordia.utils import helper_functions

# Negotiation components
from concordia.prefabs.entity.negotiation import base_negotiator, advanced_negotiator
from concordia.prefabs.entity.negotiation.components import (
    negotiation_strategy,
    negotiation_memory,
    cultural_adaptation,
    theory_of_mind,
    temporal_strategy,
    uncertainty_aware,
    swarm_intelligence,
    strategy_evolution,
)

# Import prefab packages
import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as game_master_prefabs


def create_embedder():
    """Create an embedder for the memory system."""
    try:
        import sentence_transformers
        model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        return lambda x: model.encode(x, show_progress_bar=False)
    except ImportError:
        # Fallback to simple embedder
        def simple_embedder(text):
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(hash_val)
            return np.random.randn(384).astype(np.float32)
        return simple_embedder


class TogetherNegotiationTest:
    """Test suite for negotiation framework with Together AI."""
    
    def __init__(self, api_key: str, model_name: str = 'llama-8b'):
        """Initialize test suite with Together AI model."""
        self.api_key = api_key
        self.model_name = model_name
        self.setup_model()
        self.results = {}
        
    def setup_model(self):
        """Set up Together AI model and embedder."""
        print(f"🔧 Setting up Together AI model: {self.model_name}...")
        
        # Create Together AI model
        base_model = create_together_model(
            api_key=self.api_key,
            model=self.model_name,
            use_chat=False
        )
        
        # Wrap with retry logic for rate limits
        self.model = retry_wrapper.RetryLanguageModel(
            model=base_model,
            retry_tries=5,
            retry_delay=2.0,
            retry_on_exceptions=(Exception,),
        )
        
        self.embedder = create_embedder()
        self.clock = game_clock.FixedIntervalClock()
        
        print("✅ Together AI model ready!")
    
    def test_basic_components(self):
        """Test individual components with Together AI."""
        print("\n" + "="*60)
        print("TESTING COMPONENTS WITH TOGETHER AI")
        print("="*60)
        
        results = {}
        
        # 1. Test Strategy Component
        print("\n1️⃣ Testing NegotiationStrategy...")
        try:
            strategy = negotiation_strategy.BasicNegotiationStrategy(
                agent_name="TestAgent",
                negotiation_style='cooperative',
                reservation_value=1000,
                target_value=2000
            )
            context = strategy.get_strategic_context()
            print(f"   ✅ Strategy working: {len(context)} chars generated")
            results['strategy'] = 'PASS'
        except Exception as e:
            print(f"   ❌ Strategy failed: {e}")
            results['strategy'] = f'FAIL: {e}'
        
        # 2. Test Memory Component
        print("\n2️⃣ Testing NegotiationMemory...")
        try:
            memory_bank = basic_associative_memory.AssociativeMemoryBank()
            memory_bank.set_embedder(self.embedder)
            
            memory = negotiation_memory.NegotiationMemory(
                agent_name="TestAgent",
                memory_bank=memory_bank,
                verbose=False
            )
            
            offer = negotiation_memory.Offer(
                offerer="Seller",
                recipient="TestAgent",
                content="I offer $1500",
                value=1500,
                round_number=1
            )
            memory.remember_offer(offer)
            summary = memory.get_current_negotiation_summary()
            print(f"   ✅ Memory working: {summary[:50]}...")
            results['memory'] = 'PASS'
        except Exception as e:
            print(f"   ❌ Memory failed: {e}")
            results['memory'] = f'FAIL: {e}'
        
        # 3. Test with Language Model
        print("\n3️⃣ Testing LLM Response Generation...")
        try:
            prompt = "In a negotiation, what's a good opening offer if the item is worth $1000? Answer briefly:"
            response = self.model.sample_text(
                prompt,
                max_length=50,
                temperature=0.7
            )
            print(f"   Model response: {response[:100]}...")
            results['llm_generation'] = 'PASS' if response else 'FAIL'
        except Exception as e:
            print(f"   ❌ LLM generation failed: {e}")
            results['llm_generation'] = f'FAIL: {e}'
        
        return results
    
    def test_simple_negotiation(self):
        """Run a simple negotiation scenario."""
        print("\n" + "="*60)
        print("RUNNING SIMPLE NEGOTIATION WITH TOGETHER AI")
        print("="*60)
        
        # Create memory bank
        memory_bank = basic_associative_memory.AssociativeMemoryBank()
        memory_bank.set_embedder(self.embedder)
        
        # Create simple negotiator
        print("\n🤖 Creating negotiation agent...")
        agent = base_negotiator.build_agent(
            model=self.model,
            memory_bank=memory_bank,
            name="Alice_Buyer",
            goal="Buy equipment at a good price",
            negotiation_style='cooperative',
            reservation_value=10000,
        )
        
        print("✅ Agent created!")
        
        # Test agent response
        print("\n📝 Testing agent interaction...")
        agent.observe("The seller offers the equipment for $15,000")
        
        print("Agent's response:")
        action = agent.act()
        print(f"  {action}")
        
        return {'status': 'COMPLETE', 'response': action}
    
    def test_advanced_negotiation(self):
        """Test with advanced components enabled."""
        print("\n" + "="*60)
        print("TESTING ADVANCED NEGOTIATION WITH TOGETHER AI")
        print("="*60)
        
        memory_bank = basic_associative_memory.AssociativeMemoryBank()
        memory_bank.set_embedder(self.embedder)
        
        print("\n🤖 Creating advanced agents with all components...")
        
        try:
            # Create buyer with multiple components
            buyer = advanced_negotiator.build_advanced_agent(
                model=self.model,
                memory_bank=memory_bank,
                name="Advanced_Buyer",
                goal="Purchase with best terms",
                reservation_value=12000,
                enable_cultural_adaptation=True,
                enable_theory_of_mind=True,
                enable_temporal_dynamics=True,
            )
            
            # Create seller
            seller = advanced_negotiator.build_advanced_agent(
                model=self.model,
                memory_bank=memory_bank,
                name="Advanced_Seller",
                goal="Sell at good price",
                reservation_value=10000,
                enable_cultural_adaptation=True,
                enable_theory_of_mind=True,
                enable_temporal_dynamics=True,
            )
            
            print("✅ Advanced agents created!")
            
            # Run a few rounds
            print("\n🎭 Running negotiation rounds...")
            
            # Round 1
            print("\nRound 1:")
            seller.observe("Buyer is interested in the product")
            seller_action = seller.act()
            print(f"Seller: {seller_action[:200]}...")
            
            buyer.observe(seller_action)
            buyer_action = buyer.act()
            print(f"Buyer: {buyer_action[:200]}...")
            
            return {'status': 'COMPLETE', 'rounds': 1}
            
        except Exception as e:
            print(f"❌ Advanced test failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("\n" + "🎯"*30)
        print("NEGOTIATION FRAMEWORK TEST WITH TOGETHER AI")
        print("🎯"*30)
        print(f"\nModel: {self.model_name}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test basic components
        component_results = self.test_basic_components()
        
        # Test simple negotiation
        simple_result = self.test_simple_negotiation()
        
        # Test advanced negotiation
        advanced_result = self.test_advanced_negotiation()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        print("\n📊 Component Tests:")
        for comp, status in component_results.items():
            icon = "✅" if status == 'PASS' else "❌"
            print(f"  {icon} {comp}: {status}")
        
        print(f"\n📊 Simple Negotiation: {simple_result['status']}")
        print(f"📊 Advanced Negotiation: {advanced_result['status']}")
        
        print("\n✨ Testing complete with Together AI!")
        
        return {
            'components': component_results,
            'simple': simple_result,
            'advanced': advanced_result
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test negotiation framework with Together AI models"
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('TOGETHER_API_KEY'),
        help='Together AI API key (or set TOGETHER_API_KEY env var)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='llama-8b',
        choices=['llama-70b', 'llama-8b', 'mixtral', 'mistral', 'qwen'],
        help='Model to use (shorthand names)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test only'
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("❌ Error: Together AI API key required!")
        print("Get one at: https://api.together.xyz")
        print("Then set: export TOGETHER_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Create and run test
    test = TogetherNegotiationTest(args.api_key, args.model)
    
    if args.quick:
        print("🚀 Running quick test...")
        test.test_basic_components()
    else:
        print("🚀 Running full test suite...")
        test.run_all_tests()


if __name__ == "__main__":
    main()