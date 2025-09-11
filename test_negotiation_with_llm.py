#!/usr/bin/env python3
"""
Complete test suite for negotiation framework with real LLM.

This script demonstrates all negotiation components working together
with an actual language model (OpenAI GPT or compatible).
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'examples'))

# Core imports
from concordia.language_model import gpt_model
from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.typing import prefab as prefab_lib
from concordia.prefabs.simulation import generic as simulation
from concordia.utils import helper_functions

# Import negotiation components for testing
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

# For embeddings
try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence_transformers not installed. Using simple embedder.")

import hashlib
import numpy as np


def create_embedder():
    """Create an embedder for the memory system."""
    if HAS_SENTENCE_TRANSFORMERS:
        model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        return lambda x: model.encode(x, show_progress_bar=False)
    else:
        # Fallback to simple embedder
        def simple_embedder(text):
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(hash_val)
            return np.random.randn(384).astype(np.float32)
        return simple_embedder


class NegotiationTestSuite:
    """Comprehensive test suite for negotiation framework."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        """Initialize test suite with LLM."""
        self.api_key = api_key
        self.model_name = model_name
        self.setup_model()
        self.results = {}
        
    def setup_model(self):
        """Set up language model and embedder."""
        print(f"🔧 Setting up {self.model_name} model...")
        
        # Create language model with retry wrapper for rate limits
        from concordia.language_model import retry_wrapper
        
        base_model = gpt_model.GptLanguageModel(
            api_key=self.api_key,
            model_name=self.model_name
        )
        
        self.model = retry_wrapper.RetryLanguageModel(
            model=base_model,
            retry_tries=5,
            retry_delay=2.0,
            retry_on_exceptions=(Exception,),
        )
        
        self.embedder = create_embedder()
        self.clock = game_clock.FixedIntervalClock()
        
        print("✅ Model setup complete!")
    
    def test_individual_components(self):
        """Test each negotiation component individually with LLM."""
        print("\n" + "="*60)
        print("TESTING INDIVIDUAL COMPONENTS WITH LLM")
        print("="*60)
        
        results = {}
        
        # 1. Test NegotiationStrategy
        print("\n1️⃣ Testing NegotiationStrategy...")
        try:
            strategy = negotiation_strategy.BasicNegotiationStrategy(
                agent_name="TestAgent",
                negotiation_style='integrative',
                reservation_value=1000,
                target_value=2000
            )
            context = strategy.get_strategic_context()
            print(f"   ✅ Strategy context generated: {len(context)} chars")
            results['strategy'] = 'PASS'
        except Exception as e:
            print(f"   ❌ Strategy test failed: {e}")
            results['strategy'] = f'FAIL: {e}'
        
        # 2. Test NegotiationMemory
        print("\n2️⃣ Testing NegotiationMemory...")
        try:
            memory_bank = basic_associative_memory.AssociativeMemoryBank()
            memory_bank.set_embedder(self.embedder)
            
            memory = negotiation_memory.NegotiationMemory(
                agent_name="TestAgent",
                memory_bank=memory_bank,
                verbose=True
            )
            
            # Add test offer
            offer = negotiation_memory.Offer(
                offerer="Seller",
                recipient="TestAgent",
                content="Initial offer of $1500",
                value=1500,
                round_number=1
            )
            memory.remember_offer(offer)
            summary = memory.get_current_negotiation_summary()
            print(f"   ✅ Memory tracking offers: {summary[:100]}...")
            results['memory'] = 'PASS'
        except Exception as e:
            print(f"   ❌ Memory test failed: {e}")
            results['memory'] = f'FAIL: {e}'
        
        # 3. Test CulturalAdaptation
        print("\n3️⃣ Testing CulturalAdaptation...")
        try:
            cultural = cultural_adaptation.CulturalAdaptation(
                model=self.model,
                own_culture='western_business',
                adaptation_level=0.7
            )
            
            # Test cultural detection
            test_context = "The meeting begins with extensive relationship building and gift exchange"
            cultural.detect_cultural_cues(test_context)
            guidance = cultural.get_adaptation_guidance()
            print(f"   ✅ Cultural adaptation: {guidance[:100]}...")
            results['cultural'] = 'PASS'
        except Exception as e:
            print(f"   ❌ Cultural test failed: {e}")
            results['cultural'] = f'FAIL: {e}'
        
        # 4. Test TheoryOfMind
        print("\n4️⃣ Testing TheoryOfMind...")
        try:
            tom = theory_of_mind.TheoryOfMind(
                model=self.model,
                max_recursion_depth=2,
                emotion_sensitivity=0.8
            )
            
            # Test emotional analysis
            test_behavior = "The opponent hesitates and seems anxious about the price"
            emotional_state = tom.analyze_emotional_state(
                counterpart_id="Opponent",
                observed_behavior=test_behavior
            )
            print(f"   ✅ Emotional analysis complete: {emotional_state.dominant_emotion()}")
            results['theory_of_mind'] = 'PASS'
        except Exception as e:
            print(f"   ❌ Theory of Mind test failed: {e}")
            results['theory_of_mind'] = f'FAIL: {e}'
        
        # 5. Test TemporalStrategy
        print("\n5️⃣ Testing TemporalStrategy...")
        try:
            temporal = temporal_strategy.TemporalStrategy(
                model=self.model,
                discount_factor=0.95,
                reputation_weight=0.4
            )
            
            # Test relationship value assessment
            relationship_value = temporal.assess_relationship_value(
                counterpart_id="Partner",
                history_length=5
            )
            print(f"   ✅ Temporal planning: relationship value = {relationship_value:.2f}")
            results['temporal'] = 'PASS'
        except Exception as e:
            print(f"   ❌ Temporal test failed: {e}")
            results['temporal'] = f'FAIL: {e}'
        
        # 6. Test UncertaintyAware
        print("\n6️⃣ Testing UncertaintyAware...")
        try:
            uncertainty = uncertainty_aware.UncertaintyAware(
                model=self.model,
                confidence_threshold=0.7,
                risk_tolerance=0.3
            )
            
            # Test belief update
            uncertainty.update_belief(
                variable="opponent_reservation_price",
                evidence="They mentioned budget constraints",
                confidence=0.6
            )
            assessment = uncertainty.get_uncertainty_assessment()
            print(f"   ✅ Uncertainty management: {assessment[:100]}...")
            results['uncertainty'] = 'PASS'
        except Exception as e:
            print(f"   ❌ Uncertainty test failed: {e}")
            results['uncertainty'] = f'FAIL: {e}'
        
        # 7. Test SwarmIntelligence
        print("\n7️⃣ Testing SwarmIntelligence...")
        try:
            swarm = swarm_intelligence.SwarmIntelligence(
                model=self.model,
                consensus_threshold=0.7,
                max_iterations=2
            )
            
            # Test collective decision
            decision = swarm.collective_decision(
                question="Should we accept the current offer of $1500?",
                context="Market value is $1800, we need quick sale"
            )
            print(f"   ✅ Swarm decision: {decision['consensus_reached']}")
            results['swarm'] = 'PASS'
        except Exception as e:
            print(f"   ❌ Swarm test failed: {e}")
            results['swarm'] = f'FAIL: {e}'
        
        # 8. Test StrategyEvolution
        print("\n8️⃣ Testing StrategyEvolution...")
        try:
            evolution = strategy_evolution.StrategyEvolution(
                model=self.model,
                population_size=10,
                mutation_rate=0.1
            )
            
            # Test strategy adaptation
            evolution.record_outcome(
                strategy_genome={'aggressiveness': 0.7, 'patience': 0.3},
                fitness_score=0.8
            )
            new_strategy = evolution.generate_next_strategy()
            print(f"   ✅ Strategy evolved: {new_strategy}")
            results['evolution'] = 'PASS'
        except Exception as e:
            print(f"   ❌ Evolution test failed: {e}")
            results['evolution'] = f'FAIL: {e}'
        
        # Print summary
        print("\n" + "="*60)
        print("COMPONENT TEST SUMMARY")
        print("="*60)
        for component, status in results.items():
            icon = "✅" if status == 'PASS' else "❌"
            print(f"{icon} {component}: {status}")
        
        self.results['components'] = results
        return results
    
    def test_integrated_negotiation(self):
        """Test complete negotiation with all components integrated."""
        print("\n" + "="*60)
        print("TESTING INTEGRATED NEGOTIATION SCENARIO")
        print("="*60)
        
        # Create agents with all components
        print("\n🤖 Creating advanced negotiation agents...")
        
        memory_bank = basic_associative_memory.AssociativeMemoryBank()
        memory_bank.set_embedder(self.embedder)
        
        # Create buyer with full capabilities
        buyer = advanced_negotiator.build_advanced_agent(
            model=self.model,
            memory_bank=memory_bank,
            name="Alice_Buyer",
            goal="Purchase equipment at best price while building relationship",
            reservation_value=15000,
            enable_cultural_adaptation=True,
            enable_theory_of_mind=True,
            enable_temporal_dynamics=True,
            enable_uncertainty_management=True,
            enable_strategy_evolution=True,
            enable_swarm_intelligence=True,
        )
        
        # Create seller with full capabilities  
        seller = advanced_negotiator.build_advanced_agent(
            model=self.model,
            memory_bank=memory_bank,
            name="Bob_Seller",
            goal="Sell equipment at good price while establishing partnership",
            reservation_value=12000,
            enable_cultural_adaptation=True,
            enable_theory_of_mind=True,
            enable_temporal_dynamics=True,
            enable_uncertainty_management=True,
            enable_strategy_evolution=True,
            enable_swarm_intelligence=True,
        )
        
        print("✅ Advanced agents created with all components!")
        
        # Create negotiation scenario
        print("\n🎭 Setting up negotiation scenario...")
        
        prefabs = {
            **helper_functions.get_package_classes(entity_prefabs),
            **helper_functions.get_package_classes(game_master_prefabs),
        }
        
        config = prefab_lib.Config(
            default_premise="""
            Alice (Western company) and Bob (Eastern company) are negotiating 
            the sale of specialized manufacturing equipment. The negotiation involves:
            - Price: Equipment value estimated $12,000-$18,000
            - Delivery terms: Timing is critical for Alice
            - Future partnership: Both interested in long-term relationship
            - Cultural considerations: Different business cultures
            - Information asymmetry: Bob knows about upcoming model upgrade
            """,
            default_max_steps=10,
            prefabs=prefabs,
            instances=[
                # Use the pre-built agents
                buyer,
                seller,
                # Add game master
                prefab_lib.InstanceConfig(
                    prefab='generic__GameMaster',
                    role=prefab_lib.Role.GAME_MASTER,
                    params={
                        'name': 'Negotiation_Facilitator',
                    }
                ),
            ],
        )
        
        # Run negotiation
        print("\n🚀 Running negotiation simulation...")
        
        sim = simulation.Simulation(
            config=config,
            model=self.model,
            embedder=self.embedder,
        )
        
        raw_log = []
        html_log = sim.play(
            max_steps=5,
            raw_log=raw_log,
            return_html_log=True
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"negotiation_with_llm_{timestamp}.html"
        with open(filename, 'w') as f:
            f.write(html_log)
        
        print(f"\n✅ Negotiation complete! Log saved to: {filename}")
        print(f"   - Simulation steps: {len(raw_log)}")
        print(f"   - HTML log size: {len(html_log)} bytes")
        
        # Analyze results
        self.analyze_negotiation_log(raw_log)
        
        self.results['integrated'] = {
            'status': 'COMPLETE',
            'steps': len(raw_log),
            'log_file': filename
        }
        
        return filename
    
    def analyze_negotiation_log(self, raw_log):
        """Analyze the negotiation for component usage."""
        print("\n📊 Analyzing negotiation for component activity...")
        
        log_text = str(raw_log)
        
        # Check for evidence of each component
        component_evidence = {
            'Strategy': 'cooperative' in log_text.lower() or 'competitive' in log_text.lower(),
            'Memory': 'remember' in log_text.lower() or 'past' in log_text.lower(),
            'Cultural': 'culture' in log_text.lower() or 'customs' in log_text.lower(),
            'Theory of Mind': 'feel' in log_text.lower() or 'think' in log_text.lower(),
            'Temporal': 'future' in log_text.lower() or 'relationship' in log_text.lower(),
            'Uncertainty': 'uncertain' in log_text.lower() or 'probably' in log_text.lower(),
            'Swarm': 'collective' in log_text.lower() or 'team' in log_text.lower(),
            'Evolution': 'adapt' in log_text.lower() or 'learn' in log_text.lower(),
        }
        
        for component, found in component_evidence.items():
            status = "✅ Active" if found else "⚠️  Not detected"
            print(f"   {component}: {status}")
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("\n" + "🎯"*30)
        print("CONCORDIA NEGOTIATION FRAMEWORK - FULL LLM TEST SUITE")
        print("🎯"*30)
        print(f"\nModel: {self.model_name}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test components
        component_results = self.test_individual_components()
        
        # Test integrated scenario
        log_file = self.test_integrated_negotiation()
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL TEST SUMMARY")
        print("="*60)
        
        # Component results
        passed = sum(1 for r in component_results.values() if r == 'PASS')
        total = len(component_results)
        print(f"\n📊 Component Tests: {passed}/{total} passed")
        
        # Integration results
        if self.results.get('integrated', {}).get('status') == 'COMPLETE':
            print(f"✅ Integration Test: COMPLETE")
            print(f"   - Log file: {self.results['integrated']['log_file']}")
        else:
            print(f"❌ Integration Test: FAILED")
        
        print("\n🎉 Testing complete! Check the HTML log for detailed negotiation.")
        
        return self.results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Concordia negotiation framework with real LLM"
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('OPENAI_API_KEY'),
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4',
        choices=['gpt-4', 'gpt-4-turbo-preview', 'gpt-3.5-turbo'],
        help='Model to use for testing'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test (components only)'
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("❌ Error: OpenAI API key required!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Or pass it with: --api-key 'your-key-here'")
        sys.exit(1)
    
    # Create and run test suite
    suite = NegotiationTestSuite(args.api_key, args.model)
    
    if args.quick:
        print("🚀 Running quick component tests...")
        suite.test_individual_components()
    else:
        print("🚀 Running full test suite...")
        suite.run_all_tests()


if __name__ == "__main__":
    main()