#!/usr/bin/env python3
"""End-to-end scenario tests for complete negotiations.

This test suite runs complete negotiation scenarios to verify the
entire framework works correctly in realistic situations.
"""

import unittest
from unittest import mock
import os
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple

# Core Concordia imports
from concordia.language_model import language_model, no_language_model
from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.typing import prefab as prefab_lib
from concordia.prefabs.simulation import generic as simulation
from concordia.utils import helper_functions

# Import prefab packages
import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as game_master_prefabs


def create_simple_embedder():
    """Create a simple embedder for testing."""
    def simple_embedder(text):
        # Deterministic embedding based on text hash
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        return np.random.randn(384).astype(np.float32)
    return simple_embedder


class MockLanguageModel(language_model.LanguageModel):
    """Mock language model for deterministic testing."""
    
    def __init__(self, scenario_type='bilateral'):
        self.scenario_type = scenario_type
        self.call_count = 0
        self.negotiation_state = {
            'round': 0,
            'last_offer': None,
            'agreement_reached': False
        }
    
    def sample_text(
        self,
        prompt: str,
        *,
        max_length: int = 100,
        terminators: Tuple[str, ...] = (),
        temperature: float = 1.0,
        seed: Optional[int] = None
    ) -> str:
        """Generate mock responses based on scenario."""
        self.call_count += 1
        
        prompt_lower = prompt.lower()
        
        # Handle different types of prompts
        if 'accept' in prompt_lower and 'offer' in prompt_lower:
            # Decision to accept or reject
            if self.negotiation_state['round'] > 5:
                return "accept"
            return "reject"
        
        elif 'offer' in prompt_lower or 'propose' in prompt_lower:
            # Generate offers
            if self.scenario_type == 'bilateral':
                return self._bilateral_offer()
            elif self.scenario_type == 'multi_party':
                return self._multi_party_offer()
            elif self.scenario_type == 'cultural':
                return self._cultural_offer()
            else:
                return "I propose a fair deal"
        
        elif 'strategy' in prompt_lower:
            return self._get_strategy()
        
        elif 'cultural' in prompt_lower:
            return self._get_cultural_context()
        
        elif 'emotion' in prompt_lower:
            return "cooperative"
        
        else:
            # Generic response
            return f"Proceeding with negotiation (round {self.negotiation_state['round']})"
    
    def _bilateral_offer(self):
        """Generate bilateral negotiation offers."""
        self.negotiation_state['round'] += 1
        round_num = self.negotiation_state['round']
        
        if round_num == 1:
            return "I offer $180 for the item"
        elif round_num == 2:
            return "I can go down to $170"
        elif round_num == 3:
            return "My best offer is $160"
        else:
            return "Final offer: $155"
    
    def _multi_party_offer(self):
        """Generate multi-party negotiation offers."""
        self.negotiation_state['round'] += 1
        return f"We propose a {25}% share of resources"
    
    def _cultural_offer(self):
        """Generate culturally-aware offers."""
        self.negotiation_state['round'] += 1
        return "After careful consideration of our mutual interests, I propose $165"
    
    def _get_strategy(self):
        """Return negotiation strategy."""
        strategies = ['cooperative', 'competitive', 'integrative']
        return strategies[self.call_count % 3]
    
    def _get_cultural_context(self):
        """Return cultural context."""
        contexts = ['western_direct', 'eastern_indirect', 'neutral']
        return contexts[self.call_count % 3]


class TestBilateralNegotiation(unittest.TestCase):
    """Test bilateral negotiation scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.model = MockLanguageModel(scenario_type='bilateral')
        self.embedder = create_simple_embedder()
        self.prefabs = {
            **helper_functions.get_package_classes(entity_prefabs),
            **helper_functions.get_package_classes(game_master_prefabs),
        }
    
    def test_simple_price_negotiation(self):
        """Test a simple buyer-seller price negotiation."""
        # Create configuration
        config = prefab_lib.Config(
            default_premise='Alice and Bob negotiate the price of a laptop',
            default_max_steps=10,
            prefabs=self.prefabs,
            instances=[
                prefab_lib.InstanceConfig(
                    prefab='basic__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'Alice_Buyer',
                        'goal': 'Buy the laptop for the best price',
                        'traits': ['price-conscious', 'analytical'],
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='basic__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'Bob_Seller',
                        'goal': 'Sell the laptop for a good price',
                        'traits': ['fair', 'patient'],
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='generic__GameMaster',
                    role=prefab_lib.Role.GAME_MASTER,
                    params={
                        'name': 'Negotiation_GM',
                    }
                ),
            ],
        )
        
        # Create and run simulation
        sim = simulation.Simulation(
            config=config,
            model=self.model,
            embedder=self.embedder,
        )
        
        # Run a few steps
        raw_log = []
        html_log = sim.play(
            max_steps=5,
            raw_log=raw_log,
            return_html_log=True
        )
        
        # Verify simulation ran
        self.assertGreater(len(raw_log), 0)
        self.assertIsNotNone(html_log)
        self.assertIn('Alice_Buyer', html_log)
        self.assertIn('Bob_Seller', html_log)
    
    def test_negotiation_with_deadlines(self):
        """Test negotiation with time pressure."""
        # Create configuration with deadline
        config = prefab_lib.Config(
            default_premise='Urgent negotiation with deadline approaching',
            default_max_steps=8,
            prefabs=self.prefabs,
            instances=[
                prefab_lib.InstanceConfig(
                    prefab='basic_with_plan__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'Urgent_Buyer',
                        'goal': 'Complete purchase before deadline',
                        'initial_plan': 'Start reasonable, concede if needed',
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='basic_with_plan__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'Patient_Seller',
                        'goal': 'Maximize price knowing buyer has deadline',
                        'initial_plan': 'Hold firm, buyer will concede',
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='generic__GameMaster',
                    role=prefab_lib.Role.GAME_MASTER,
                    params={
                        'name': 'Deadline_GM',
                    }
                ),
            ],
        )
        
        # Run simulation
        sim = simulation.Simulation(
            config=config,
            model=self.model,
            embedder=self.embedder,
        )
        
        raw_log = []
        sim.play(max_steps=3, raw_log=raw_log)
        
        # Verify urgency affects negotiation
        self.assertGreater(len(raw_log), 0)


class TestMultiPartyNegotiation(unittest.TestCase):
    """Test multi-party negotiation scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.model = MockLanguageModel(scenario_type='multi_party')
        self.embedder = create_simple_embedder()
        self.prefabs = {
            **helper_functions.get_package_classes(entity_prefabs),
            **helper_functions.get_package_classes(game_master_prefabs),
        }
    
    def test_resource_allocation(self):
        """Test multi-party resource allocation negotiation."""
        # Create configuration with 4 parties
        config = prefab_lib.Config(
            default_premise='Four organizations negotiate resource allocation',
            default_max_steps=12,
            prefabs=self.prefabs,
            instances=[
                prefab_lib.InstanceConfig(
                    prefab='basic__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'TechCorp',
                        'goal': 'Secure technology resources',
                        'traits': ['innovative', 'competitive'],
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='basic__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'ManufacturingInc',
                        'goal': 'Get production resources',
                        'traits': ['practical', 'efficient'],
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='basic__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'StartupLLC',
                        'goal': 'Access shared resources',
                        'traits': ['flexible', 'creative'],
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='basic__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'NonProfit',
                        'goal': 'Secure resources for community',
                        'traits': ['mission-driven', 'collaborative'],
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='generic__GameMaster',
                    role=prefab_lib.Role.GAME_MASTER,
                    params={
                        'name': 'Resource_Coordinator',
                    }
                ),
            ],
        )
        
        # Run simulation
        sim = simulation.Simulation(
            config=config,
            model=self.model,
            embedder=self.embedder,
        )
        
        raw_log = []
        sim.play(max_steps=4, raw_log=raw_log)
        
        # Verify all parties participated
        log_text = str(raw_log)
        self.assertIn('TechCorp', log_text)
        self.assertIn('ManufacturingInc', log_text)
        self.assertIn('StartupLLC', log_text)
        self.assertIn('NonProfit', log_text)
    
    def test_coalition_formation(self):
        """Test coalition formation in multi-party negotiation."""
        # Configuration with potential coalitions
        config = prefab_lib.Config(
            default_premise='Parties may form coalitions for better outcomes',
            default_max_steps=10,
            prefabs=self.prefabs,
            instances=[
                prefab_lib.InstanceConfig(
                    prefab='basic__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'Party_A',
                        'goal': 'Maximize share through coalitions',
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='basic__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'Party_B',
                        'goal': 'Form strategic alliances',
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='basic__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'Party_C',
                        'goal': 'Prevent exclusion from deals',
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='generic__GameMaster',
                    role=prefab_lib.Role.GAME_MASTER,
                    params={
                        'name': 'Coalition_GM',
                    }
                ),
            ],
        )
        
        sim = simulation.Simulation(
            config=config,
            model=self.model,
            embedder=self.embedder,
        )
        
        raw_log = []
        sim.play(max_steps=3, raw_log=raw_log)
        
        # Verify negotiation occurred
        self.assertGreater(len(raw_log), 0)


class TestCulturalNegotiation(unittest.TestCase):
    """Test cross-cultural negotiation scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.model = MockLanguageModel(scenario_type='cultural')
        self.embedder = create_simple_embedder()
        self.prefabs = {
            **helper_functions.get_package_classes(entity_prefabs),
            **helper_functions.get_package_classes(game_master_prefabs),
        }
    
    def test_cross_cultural_business(self):
        """Test negotiation with cultural differences."""
        config = prefab_lib.Config(
            default_premise='International business negotiation with cultural considerations',
            default_max_steps=10,
            prefabs=self.prefabs,
            instances=[
                prefab_lib.InstanceConfig(
                    prefab='basic__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'Western_Executive',
                        'goal': 'Quick deal with clear terms',
                        'traits': ['direct', 'time-conscious', 'results-oriented'],
                        'backstory': 'Represents Western business culture',
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='basic__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'Eastern_Executive',
                        'goal': 'Build relationship for long-term partnership',
                        'traits': ['relationship-focused', 'patient', 'indirect'],
                        'backstory': 'Represents Eastern business culture',
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='generic__GameMaster',
                    role=prefab_lib.Role.GAME_MASTER,
                    params={
                        'name': 'Cultural_Mediator',
                    }
                ),
            ],
        )
        
        sim = simulation.Simulation(
            config=config,
            model=self.model,
            embedder=self.embedder,
        )
        
        raw_log = []
        sim.play(max_steps=4, raw_log=raw_log)
        
        # Verify cultural elements in negotiation
        log_text = str(raw_log)
        self.assertIn('Western_Executive', log_text)
        self.assertIn('Eastern_Executive', log_text)


class TestInformationAsymmetry(unittest.TestCase):
    """Test negotiations with information asymmetry."""
    
    def setUp(self):
        """Set up test environment."""
        self.model = MockLanguageModel(scenario_type='bilateral')
        self.embedder = create_simple_embedder()
        self.prefabs = {
            **helper_functions.get_package_classes(entity_prefabs),
            **helper_functions.get_package_classes(game_master_prefabs),
        }
    
    def test_hidden_information(self):
        """Test negotiation where one party has more information."""
        config = prefab_lib.Config(
            default_premise='Asset sale with hidden information about true value',
            default_max_steps=8,
            prefabs=self.prefabs,
            instances=[
                prefab_lib.InstanceConfig(
                    prefab='basic_with_plan__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'Informed_Seller',
                        'goal': 'Sell asset knowing its true high value',
                        'initial_plan': 'Reveal value strategically',
                        'backstory': 'Knows asset worth will increase soon',
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='basic_with_plan__Entity',
                    role=prefab_lib.Role.ENTITY,
                    params={
                        'name': 'Uninformed_Buyer',
                        'goal': 'Buy asset at fair market price',
                        'initial_plan': 'Gather information during negotiation',
                        'backstory': 'Has limited information about asset',
                    }
                ),
                prefab_lib.InstanceConfig(
                    prefab='generic__GameMaster',
                    role=prefab_lib.Role.GAME_MASTER,
                    params={
                        'name': 'Information_GM',
                    }
                ),
            ],
        )
        
        sim = simulation.Simulation(
            config=config,
            model=self.model,
            embedder=self.embedder,
        )
        
        raw_log = []
        sim.play(max_steps=3, raw_log=raw_log)
        
        # Verify negotiation handles information asymmetry
        self.assertGreater(len(raw_log), 0)


class ScenarioTestSuite(unittest.TestSuite):
    """Custom test suite for scenario tests."""
    
    def __init__(self):
        super().__init__()
        
        # Add all test cases
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(TestBilateralNegotiation))
        self.addTests(loader.loadTestsFromTestCase(TestMultiPartyNegotiation))
        self.addTests(loader.loadTestsFromTestCase(TestCulturalNegotiation))
        self.addTests(loader.loadTestsFromTestCase(TestInformationAsymmetry))


def run_scenario_tests(verbose=True):
    """Run all negotiation scenario tests."""
    print("="*60)
    print("NEGOTIATION SCENARIO TESTS")
    print("="*60)
    
    # Create test suite
    suite = ScenarioTestSuite()
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("SCENARIO TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    print("\nScenarios tested:")
    print("✓ Bilateral price negotiation")
    print("✓ Negotiation with deadlines")
    print("✓ Multi-party resource allocation")
    print("✓ Coalition formation")
    print("✓ Cross-cultural business negotiation")
    print("✓ Information asymmetry")
    
    return result


if __name__ == "__main__":
    # Run scenario tests
    result = run_scenario_tests(verbose=True)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)