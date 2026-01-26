# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for negotiation agent builders."""

import datetime
import unittest
from unittest import mock

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.language_model import language_model
from concordia.prefabs.entity.negotiation import base_negotiator
from concordia.prefabs.entity.negotiation import advanced_negotiator


class BaseNegotiatorBuilderTest(unittest.TestCase):
  """Tests for base negotiator builder functions."""

  def setUp(self):
    """Set up test dependencies."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    self.clock = game_clock.GameClock()
    self.memory_bank = basic_associative_memory.AssociativeMemoryBank(
        model=self.model,
        clock=self.clock,
    )

  def test_build_agent_basic(self):
    """Test basic agent building."""
    agent = base_negotiator.build_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='TestAgent',
        goal='Test negotiation goal',
        negotiation_style='cooperative',
        reservation_value=100.0,
    )
    
    self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertEqual(agent._agent_name, 'TestAgent')
    
    # Check that key components exist
    self.assertIn('NegotiationInstructions', agent._context_components)
    self.assertIn('NegotiationMemory', agent._context_components)
    self.assertIn('BasicNegotiationStrategy', agent._context_components)

  def test_build_agent_different_styles(self):
    """Test building agents with different negotiation styles."""
    styles = ['cooperative', 'competitive', 'integrative']
    
    for style in styles:
      with self.subTest(style=style):
        agent = base_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name=f'Agent_{style}',
            negotiation_style=style,
        )
        
        self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)
        
        # Check that strategy component has correct style
        strategy_component = None
        for comp in agent._context_components.values():
          if hasattr(comp, '_negotiation_style'):
            strategy_component = comp
            break
        
        self.assertIsNotNone(strategy_component)
        self.assertEqual(strategy_component._negotiation_style, style)

  def test_build_agent_with_custom_parameters(self):
    """Test building agent with custom parameters."""
    agent = base_negotiator.build_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='CustomAgent',
        goal='Custom goal',
        reservation_value=500.0,
        ethical_constraints='Custom ethics',
        extra_components={'test_comp': mock.Mock()},
    )
    
    # Check custom parameters were applied
    instructions = None
    strategy = None
    for comp in agent._context_components.values():
      if hasattr(comp, '_goal'):
        instructions = comp
      elif hasattr(comp, '_reservation_value'):
        strategy = comp
    
    self.assertIsNotNone(instructions)
    self.assertIsNotNone(strategy)
    self.assertEqual(instructions._goal, 'Custom goal')
    self.assertEqual(strategy._reservation_value, 500.0)


class AdvancedNegotiatorBuilderTest(unittest.TestCase):
  """Tests for advanced negotiator builder functions."""

  def setUp(self):
    """Set up test dependencies."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    self.clock = game_clock.GameClock()
    self.memory_bank = basic_associative_memory.AssociativeMemoryBank(
        model=self.model,
        clock=self.clock,
    )

  def test_build_agent_basic(self):
    """Test basic advanced agent building."""
    agent = advanced_negotiator.build_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='AdvancedAgent',
    )
    
    self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertEqual(agent._agent_name, 'AdvancedAgent')

  def test_build_agent_with_modules(self):
    """Test building agent with specific modules."""
    agent = advanced_negotiator.build_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='ModularAgent',
        modules=['cultural_adaptation', 'theory_of_mind'],
        module_configs={
            'cultural_adaptation': {'own_culture': 'east_asian'},
            'theory_of_mind': {'max_recursion_depth': 2}
        }
    )
    
    self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)
    
    # Check that specified modules are present
    self.assertIn('CulturalAdaptation', agent._context_components)
    self.assertIn('TheoryOfMind', agent._context_components)

  def test_build_cultural_agent(self):
    """Test building culturally-aware agent."""
    agent = advanced_negotiator.build_cultural_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='CulturalAgent',
        own_culture='middle_eastern',
    )
    
    self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertIn('CulturalAdaptation', agent._context_components)
    self.assertIn('TheoryOfMind', agent._context_components)

  def test_build_temporal_agent(self):
    """Test building temporal strategy agent."""
    agent = advanced_negotiator.build_temporal_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='TemporalAgent',
        discount_factor=0.8,
    )
    
    self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertIn('TemporalStrategy', agent._context_components)
    self.assertIn('TheoryOfMind', agent._context_components)

  def test_build_collective_agent(self):
    """Test building collective intelligence agent."""
    agent = advanced_negotiator.build_collective_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='CollectiveAgent',
    )
    
    self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertIn('SwarmIntelligence', agent._context_components)
    self.assertIn('UncertaintyAware', agent._context_components)

  def test_build_adaptive_agent(self):
    """Test building adaptive strategy agent."""
    agent = advanced_negotiator.build_adaptive_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='AdaptiveAgent',
        learning_rate=0.05,
    )
    
    self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertIn('StrategyEvolution', agent._context_components)
    self.assertIn('UncertaintyAware', agent._context_components)

  def test_all_modules_combination(self):
    """Test building agent with all available modules."""
    agent = advanced_negotiator.build_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='FullyLoadedAgent',
        modules=[
            'cultural_adaptation',
            'temporal_strategy',
            'swarm_intelligence',
            'uncertainty_aware',
            'strategy_evolution',
            'theory_of_mind',
        ],
    )
    
    self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)
    
    # Check that all modules are present
    expected_components = [
        'CulturalAdaptation',
        'TemporalStrategy',
        'SwarmIntelligence',
        'UncertaintyAware',
        'StrategyEvolution',
        'TheoryOfMind',
    ]
    
    for component in expected_components:
      self.assertIn(component, agent._context_components, 
                   f'Missing component: {component}')

  def test_module_configuration_propagation(self):
    """Test that module configurations are properly propagated."""
    config = {
        'cultural_adaptation': {
            'own_culture': 'northern_european',
            'adaptation_level': 0.9,
        },
        'theory_of_mind': {
            'max_recursion_depth': 1,
            'emotion_sensitivity': 0.5,
        }
    }
    
    agent = advanced_negotiator.build_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='ConfiguredAgent',
        modules=['cultural_adaptation', 'theory_of_mind'],
        module_configs=config,
    )
    
    # Check that configurations were applied
    cultural_comp = agent._context_components['CulturalAdaptation']
    self.assertEqual(cultural_comp._own_profile.name, 'Northern European')
    self.assertEqual(cultural_comp._adaptation_level, 0.9)
    
    tom_comp = agent._context_components['TheoryOfMind']
    self.assertEqual(tom_comp._max_recursion_depth, 1)
    self.assertEqual(tom_comp._emotion_sensitivity, 0.5)


class AgentBuilderErrorHandlingTest(unittest.TestCase):
  """Tests for error handling in agent builders."""

  def setUp(self):
    """Set up test dependencies."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    self.clock = game_clock.GameClock()
    self.memory_bank = basic_associative_memory.AssociativeMemoryBank(
        model=self.model,
        clock=self.clock,
    )

  def test_invalid_module_name(self):
    """Test handling of invalid module names."""
    # This should not raise an error, but should skip invalid modules
    agent = advanced_negotiator.build_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='TestAgent',
        modules=['invalid_module_name', 'theory_of_mind'],
    )
    
    self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertIn('TheoryOfMind', agent._context_components)
    self.assertNotIn('InvalidModuleName', agent._context_components)

  def test_invalid_json_config(self):
    """Test handling of invalid JSON configuration."""
    # This should fall back to empty config
    prefab = advanced_negotiator.Entity(params={
        'modules': 'theory_of_mind',
        'module_configs': 'invalid json',
    })
    
    agent = prefab.build(self.model, self.memory_bank)
    self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)

  def test_missing_required_parameters(self):
    """Test behavior with missing required parameters."""
    # Should use defaults
    agent = base_negotiator.build_agent(
        model=self.model,
        memory_bank=self.memory_bank,
    )
    
    self.assertIsInstance(agent, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertEqual(agent._agent_name, 'Negotiator')  # Default name


class ComponentInteractionTest(unittest.TestCase):
  """Tests for interaction between different components."""

  def setUp(self):
    """Set up test dependencies."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    self.clock = game_clock.GameClock()
    self.memory_bank = basic_associative_memory.AssociativeMemoryBank(
        model=self.model,
        clock=self.clock,
    )

  def test_component_ordering(self):
    """Test that components are in correct order."""
    agent = advanced_negotiator.build_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='OrderedAgent',
        modules=['theory_of_mind', 'cultural_adaptation'],
    )
    
    # Check that act component has correct component order
    act_component = agent._act_component
    self.assertIsNotNone(act_component)
    
    # Component order should include both base and advanced components
    component_names = list(agent._context_components.keys())
    
    # Base components should be present
    base_components = ['observation_to_memory', 'observation', 'memory']
    for comp in base_components:
      self.assertTrue(any(comp in name for name in component_names))
    
    # Advanced components should be present
    self.assertIn('TheoryOfMind', component_names)
    self.assertIn('CulturalAdaptation', component_names)

  def test_component_state_interaction(self):
    """Test that components can interact through shared state."""
    agent = advanced_negotiator.build_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='InteractionAgent',
        modules=['negotiation_memory', 'theory_of_mind'],
    )
    
    # Components should be able to access shared memory
    memory_comp = None
    tom_comp = None
    
    for comp in agent._context_components.values():
      if hasattr(comp, 'record_offer'):
        memory_comp = comp
      elif hasattr(comp, 'detect_emotion'):
        tom_comp = comp
    
    # Both components should use the same memory bank
    if memory_comp and tom_comp:
      self.assertEqual(memory_comp._memory_bank, tom_comp._memory_bank)


if __name__ == '__main__':
  unittest.main()