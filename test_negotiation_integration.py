#!/usr/bin/env python3
"""Integration tests for negotiation component interactions.

This test suite verifies that negotiation components work correctly
when integrated together.
"""

import unittest
from unittest import mock
import numpy as np
from typing import Dict, List, Optional

# Core Concordia imports
from concordia.language_model import language_model
from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.agents import entity_agent_with_logging

# Negotiation framework
from concordia.prefabs.entity.negotiation import (
    base_negotiator,
    advanced_negotiator,
    integration_framework,
)

# Components
from concordia.prefabs.entity.negotiation.components import (
    negotiation_strategy,
    negotiation_memory,
    cultural_adaptation,
    theory_of_mind,
    temporal_strategy,
    uncertainty_aware,
)


class TestComponentIntegration(unittest.TestCase):
    """Test integration between different negotiation components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = mock.create_autospec(language_model.LanguageModel)
        self.clock = game_clock.FixedIntervalClock()
        self.memory_bank = basic_associative_memory.AssociativeMemoryBank()
        
        # Mock model responses
        def mock_sample_text(prompt, **kwargs):
            if 'cultural' in prompt.lower():
                return 'western_business'
            elif 'emotion' in prompt.lower():
                return 'cooperative'
            elif 'confidence' in prompt.lower():
                return '0.7'
            elif 'offer' in prompt.lower():
                return 'I offer $150'
            else:
                return 'neutral response'
        
        self.model.sample_text.side_effect = mock_sample_text
    
    def test_strategy_memory_integration(self):
        """Test integration between strategy and memory components."""
        # Create components
        strategy = negotiation_strategy.BasicNegotiationStrategy(
            agent_name="TestAgent",
            negotiation_style='cooperative',
            reservation_value=100,
            target_value=200
        )
        
        memory = negotiation_memory.NegotiationMemory(
            agent_name="TestAgent",
            clock_now=self.clock.now,
            memory_bank=self.memory_bank
        )
        
        # Simulate negotiation round
        memory.record_offer(1, "self", "initial", 170)
        memory.record_offer(1, "opponent", "counter", 130)
        
        # Strategy should adapt based on memory
        strategy.update_state(opponent_offer=130)
        context = strategy.get_strategic_context()
        
        # Verify integration
        self.assertIn("130", context)  # Opponent position tracked
        memory_context = memory.get_memory_context()
        self.assertIn("Round 1", memory_context)
    
    def test_cultural_theory_of_mind_integration(self):
        """Test integration between cultural adaptation and theory of mind."""
        # Create components
        cultural = cultural_adaptation.CulturalAdaptation(
            agent_name="TestAgent",
            model=self.model,
            cultural_background="western"
        )
        
        tom = theory_of_mind.TheoryOfMind(
            agent_name="TestAgent",
            model=self.model
        )
        
        # Simulate cross-cultural interaction
        cultural.detect_cultural_context("Meeting follows Japanese customs")
        tom.update_opponent_model(
            "Tanaka",
            "Bowed and exchanged business cards",
            "Initial meeting"
        )
        
        # Both should have relevant context
        cultural_context = cultural.get_cultural_context()
        tom_assessment = tom.get_opponent_assessment("Tanaka")
        
        self.assertIn("CULTURAL", cultural_context)
        self.assertIn("Tanaka", tom_assessment)
    
    def test_temporal_uncertainty_integration(self):
        """Test integration between temporal strategy and uncertainty awareness."""
        # Create components
        temporal = temporal_strategy.TemporalStrategy(
            agent_name="TestAgent",
            clock_now=self.clock.now,
            deadline_rounds=10
        )
        
        uncertainty = uncertainty_aware.UncertaintyAware(
            agent_name="TestAgent",
            model=self.model
        )
        
        # Simulate increasing time pressure with uncertainty
        for round_num in range(1, 8):
            temporal.update_round(round_num)
            uncertainty.assess_uncertainty(
                f"Round {round_num} negotiation",
                "time_pressure"
            )
        
        # Check urgency increases
        urgency = temporal.get_urgency_level()
        self.assertGreater(urgency, 0.5)  # Should be urgent by round 7
        
        # Uncertainty should reflect time pressure
        uncertainty_context = uncertainty.get_uncertainty_context()
        self.assertIn("UNCERTAINTY", uncertainty_context)
    
    def test_full_agent_integration(self):
        """Test full agent with all components integrated."""
        # Build complete agent
        agent = base_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name="IntegratedAgent",
            goal="Test all components",
            negotiation_style='integrative',
            reservation_value=100.0,
            enable_cultural_adaptation=True,
            enable_theory_of_mind=True,
        )
        
        # Verify agent has expected components
        self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)
        self.assertEqual(agent._agent_name, "IntegratedAgent")
        
        # Check key components exist
        component_names = [c.name for c in agent._context_components.values()]
        self.assertIn('NegotiationInstructions', component_names)
        self.assertIn('NegotiationMemory', component_names)
        self.assertIn('BasicNegotiationStrategy', component_names)
        
        # Test agent can process observations
        agent.observe("Opponent offers $150 for the item")
        
        # Test agent can generate actions
        action = agent.act()
        self.assertIsNotNone(action)


class TestModuleIntegrationFramework(unittest.TestCase):
    """Test the integration framework that coordinates modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integrator = integration_framework.NegotiationModuleIntegrator()
    
    def test_module_registration(self):
        """Test registering modules with the integrator."""
        # Register modules
        self.integrator.register_module(
            'theory_of_mind',
            integration_framework.ModuleConfig(
                enabled=True,
                priority=1,
                dependencies=set()
            )
        )
        
        self.integrator.register_module(
            'cultural_adaptation',
            integration_framework.ModuleConfig(
                enabled=True,
                priority=2,
                dependencies=set()
            )
        )
        
        # Verify registration
        self.assertEqual(len(self.integrator.modules), 2)
        self.assertIn('theory_of_mind', self.integrator.modules)
        self.assertIn('cultural_adaptation', self.integrator.modules)
    
    def test_dependency_validation(self):
        """Test validation of module dependencies."""
        # Register module with dependencies
        self.integrator.register_module(
            'swarm_intelligence',
            integration_framework.ModuleConfig(
                enabled=True,
                priority=3,
                dependencies={'theory_of_mind', 'uncertainty_aware'}
            )
        )
        
        # Validation should identify missing dependencies
        issues = self.integrator.validate_configuration()
        self.assertGreater(len(issues), 0)  # Should have dependency issues
        
        # Add missing dependencies
        self.integrator.register_module(
            'theory_of_mind',
            integration_framework.ModuleConfig(enabled=True, priority=1)
        )
        self.integrator.register_module(
            'uncertainty_aware',
            integration_framework.ModuleConfig(enabled=True, priority=1)
        )
        
        # Now validation should pass
        issues = self.integrator.validate_configuration()
        self.assertEqual(len(issues), 0)
    
    def test_interaction_protocols(self):
        """Test predefined interaction protocols between modules."""
        # Check that interaction protocols are defined
        protocols = integration_framework.NegotiationModuleIntegrator.INTERACTION_PROTOCOLS
        
        # Verify some key interactions exist
        self.assertIn(('theory_of_mind', 'cultural_adaptation'), protocols)
        self.assertIn(('uncertainty_aware', 'swarm_intelligence'), protocols)
        self.assertIn(('temporal_strategy', 'strategy_evolution'), protocols)
        
        # Check interaction properties
        tom_cultural = protocols[('theory_of_mind', 'cultural_adaptation')]
        self.assertEqual(tom_cultural['type'], 'emotional_cultural_context')
        self.assertTrue(tom_cultural['bidirectional'])


class TestAdvancedAgentIntegration(unittest.TestCase):
    """Test advanced negotiator with complex module interactions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = mock.create_autospec(language_model.LanguageModel)
        self.clock = game_clock.FixedIntervalClock()
        self.memory_bank = basic_associative_memory.AssociativeMemoryBank()
        
        # More sophisticated mock responses
        self.response_counter = 0
        def sophisticated_mock(prompt, **kwargs):
            self.response_counter += 1
            if 'emotional state' in prompt.lower():
                return 'cautiously optimistic'
            elif 'cultural' in prompt.lower():
                return 'high-context eastern culture'
            elif 'strategy' in prompt.lower():
                return 'win-win integrative approach'
            elif 'confidence' in prompt.lower():
                return str(0.6 + self.response_counter * 0.05)  # Increasing confidence
            else:
                return f'Response {self.response_counter}'
        
        self.model.sample_text.side_effect = sophisticated_mock
    
    def test_advanced_agent_creation(self):
        """Test creating an advanced negotiator with all modules."""
        agent = advanced_negotiator.build_advanced_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name="AdvancedNegotiator",
            enable_cultural_adaptation=True,
            enable_theory_of_mind=True,
            enable_temporal_dynamics=True,
            enable_uncertainty_management=True,
            enable_strategy_evolution=True,
            enable_swarm_intelligence=True,
        )
        
        # Verify agent creation
        self.assertIsNotNone(agent)
        self.assertEqual(agent._agent_name, "AdvancedNegotiator")
        
        # Check that advanced components are present
        component_names = [c.name for c in agent._context_components.values()]
        expected_components = [
            'TheoryOfMind',
            'CulturalAdaptation',
            'TemporalStrategy',
            'UncertaintyAware',
            'StrategyEvolution',
            'SwarmIntelligence'
        ]
        
        for component in expected_components:
            self.assertIn(component, component_names)
    
    def test_multi_module_coordination(self):
        """Test coordination between multiple modules during negotiation."""
        # Create agent with multiple modules
        agent = advanced_negotiator.build_advanced_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name="MultiModuleAgent",
            enable_theory_of_mind=True,
            enable_temporal_dynamics=True,
            enable_uncertainty_management=True,
        )
        
        # Simulate negotiation sequence
        observations = [
            "Opponent seems hesitant about the price",
            "They mentioned a deadline tomorrow",
            "Their offer is $150 but they seem uncertain",
            "They're consulting with their team"
        ]
        
        for obs in observations:
            agent.observe(obs)
        
        # Generate action with multi-module input
        action = agent.act()
        self.assertIsNotNone(action)
        
        # Verify modules processed the observations
        # (In real implementation, we'd check module states)


class ComponentInteractionTestSuite(unittest.TestSuite):
    """Custom test suite for component interaction tests."""
    
    def __init__(self):
        super().__init__()
        
        # Add all test cases
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(TestComponentIntegration))
        self.addTests(loader.loadTestsFromTestCase(TestModuleIntegrationFramework))
        self.addTests(loader.loadTestsFromTestCase(TestAdvancedAgentIntegration))


def run_integration_tests(verbose=True):
    """Run all component integration tests."""
    print("="*60)
    print("NEGOTIATION COMPONENT INTEGRATION TESTS")
    print("="*60)
    
    # Create test suite
    suite = ComponentInteractionTestSuite()
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    if result.failures:
        print("\nFailed tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nTests with errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result


if __name__ == "__main__":
    # Run integration tests
    result = run_integration_tests(verbose=True)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)