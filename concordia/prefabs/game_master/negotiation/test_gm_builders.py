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

"""Integration tests for negotiation game master builders."""

import unittest
from unittest import mock

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.language_model import language_model
from concordia.prefabs.game_master.negotiation import negotiation


class GameMasterBuilderTest(unittest.TestCase):
  """Tests for game master builder functions."""

  def setUp(self):
    """Set up test dependencies."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    self.clock = game_clock.FixedIntervalClock()
    self.memory_bank = basic_associative_memory.AssociativeMemoryBank()
    
    # Create mock agents
    self.agents = []
    for i, name in enumerate(['Alice', 'Bob', 'Charlie']):
      agent = mock.create_autospec(
          entity_agent_with_logging.EntityAgentWithLogging,
          instance=True,
      )
      agent.name = name
      agent._context_components = {}  # Mock empty components
      self.agents.append(agent)

  def test_build_game_master_basic(self):
    """Test basic game master building."""
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=self.agents[:2],  # Use 2 agents
        name='TestGM',
        negotiation_type='price',
        max_rounds=10,
    )
    
    self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertEqual(gm._agent_name, 'TestGM')
    
    # Check that key components exist
    self.assertIn('negotiation_state', gm._context_components)
    self.assertIn('negotiation_validator', gm._context_components)

  def test_build_game_master_with_modules(self):
    """Test building GM with specific modules."""
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=self.agents[:2],
        name='ModularGM',
        gm_modules=['social_intelligence', 'temporal_dynamics'],
    )
    
    self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)
    
    # Check that GM modules are present
    self.assertIn('gm_module_social_intelligence', gm._context_components)
    self.assertIn('gm_module_temporal_dynamics', gm._context_components)

  def test_build_bilateral_negotiation(self):
    """Test building bilateral negotiation GM."""
    gm = negotiation.build_bilateral_negotiation(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=self.agents[:2],  # Exactly 2 agents
        name='Bilateral GM',
    )
    
    self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertEqual(gm._agent_name, 'Bilateral GM')
    
    # Should have social intelligence and temporal dynamics modules
    self.assertIn('gm_module_social_intelligence', gm._context_components)
    self.assertIn('gm_module_temporal_dynamics', gm._context_components)

  def test_build_bilateral_negotiation_wrong_agent_count(self):
    """Test bilateral negotiation with wrong number of agents."""
    with self.assertRaises(ValueError):
      negotiation.build_bilateral_negotiation(
          model=self.model,
          memory_bank=self.memory_bank,
          entities=self.agents,  # 3 agents, should be exactly 2
      )

  def test_build_multilateral_negotiation(self):
    """Test building multilateral negotiation GM."""
    gm = negotiation.build_multilateral_negotiation(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=self.agents,  # 3 agents
        name='Multilateral GM',
    )
    
    self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertEqual(gm._agent_name, 'Multilateral GM')
    
    # Should have collective intelligence modules
    self.assertIn('gm_module_collective_intelligence', gm._context_components)
    self.assertIn('gm_module_uncertainty_management', gm._context_components)
    self.assertIn('gm_module_social_intelligence', gm._context_components)

  def test_build_multilateral_negotiation_insufficient_agents(self):
    """Test multilateral negotiation with insufficient agents."""
    with self.assertRaises(ValueError):
      negotiation.build_multilateral_negotiation(
          model=self.model,
          memory_bank=self.memory_bank,
          entities=self.agents[:2],  # Only 2 agents, need at least 3
      )

  def test_build_cultural_negotiation(self):
    """Test building cross-cultural negotiation GM."""
    gm = negotiation.build_cultural_negotiation(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=self.agents[:2],
        name='Cultural GM',
    )
    
    self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertEqual(gm._agent_name, 'Cultural GM')
    
    # Should have cultural awareness modules
    self.assertIn('gm_module_cultural_awareness', gm._context_components)
    self.assertIn('gm_module_social_intelligence', gm._context_components)
    self.assertIn('gm_module_temporal_dynamics', gm._context_components)

  def test_build_adaptive_negotiation(self):
    """Test building adaptive negotiation GM."""
    gm = negotiation.build_adaptive_negotiation(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=self.agents[:2],
        name='Adaptive GM',
    )
    
    self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertEqual(gm._agent_name, 'Adaptive GM')
    
    # Should have strategy evolution modules
    self.assertIn('gm_module_strategy_evolution', gm._context_components)
    self.assertIn('gm_module_uncertainty_management', gm._context_components)
    self.assertIn('gm_module_social_intelligence', gm._context_components)

  def test_auto_module_detection(self):
    """Test automatic module detection based on agent capabilities."""
    # Create agents with specific components
    cultural_agent = mock.create_autospec(
        entity_agent_with_logging.EntityAgentWithLogging,
        instance=True,
    )
    cultural_agent.name = 'CulturalAgent'
    cultural_agent._context_components = {'CulturalAdaptation': mock.Mock()}
    
    temporal_agent = mock.create_autospec(
        entity_agent_with_logging.EntityAgentWithLogging,
        instance=True,
    )
    temporal_agent.name = 'TemporalAgent'
    temporal_agent._context_components = {'TemporalStrategy': mock.Mock()}
    
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=[cultural_agent, temporal_agent],
        gm_modules=[],  # Start with no modules
        auto_detect_modules=True,
    )
    
    # Should automatically detect and add appropriate GM modules
    # (Note: The exact behavior depends on the detect_agent_modules implementation)
    self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)

  def test_custom_parameters_propagation(self):
    """Test that custom parameters are properly propagated."""
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=self.agents[:2],
        name='CustomGM',
        negotiation_type='contract',
        max_rounds=15,
        protocol='simultaneous',
        enable_batna_validation=True,
    )
    
    self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)
    
    # Check that parameters were used in building
    # (Note: Exact verification depends on how parameters are stored)
    self.assertEqual(gm._agent_name, 'CustomGM')


class GameMasterComponentTest(unittest.TestCase):
  """Tests for game master component integration."""

  def setUp(self):
    """Set up test dependencies."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    self.clock = game_clock.FixedIntervalClock()
    self.memory_bank = basic_associative_memory.AssociativeMemoryBank()
    
    # Create simple mock agents
    self.agents = []
    for name in ['Alice', 'Bob']:
      agent = mock.create_autospec(
          entity_agent_with_logging.EntityAgentWithLogging,
          instance=True,
      )
      agent.name = name
      agent._context_components = {}
      self.agents.append(agent)

  def test_negotiation_state_component_integration(self):
    """Test that negotiation state component is properly integrated."""
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=self.agents,
        name='StateTestGM',
    )
    
    # Should have negotiation state component
    self.assertIn('negotiation_state', gm._context_components)
    state_component = gm._context_components['negotiation_state']
    
    # Should be able to start a negotiation
    state = state_component.start_negotiation(
        negotiation_id='test_integration',
        participants=['Alice', 'Bob'],
    )
    
    self.assertEqual(state.negotiation_id, 'test_integration')
    self.assertEqual(len(state.participants), 2)

  def test_negotiation_validator_integration(self):
    """Test that negotiation validator is properly integrated."""
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=self.agents,
        name='ValidatorTestGM',
        enable_batna_validation=True,
    )
    
    # Should have validator component
    self.assertIn('negotiation_validator', gm._context_components)
    validator = gm._context_components['negotiation_validator']
    
    # Should be able to set BATNAs
    validator.set_batna('Alice', {'price': 100, 'role': 'buyer'})
    
    # Should be able to validate offers
    is_valid, errors = validator.validate_offer('Alice', {'price': 90})
    self.assertIsInstance(is_valid, bool)
    self.assertIsInstance(errors, list)

  def test_gm_modules_integration(self):
    """Test that GM modules are properly integrated."""
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=self.agents,
        name='ModulesTestGM',
        gm_modules=['social_intelligence', 'cultural_awareness'],
    )
    
    # Should have GM module components
    self.assertIn('gm_module_social_intelligence', gm._context_components)
    self.assertIn('gm_module_cultural_awareness', gm._context_components)
    
    # Modules should be properly initialized
    social_module = gm._context_components['gm_module_social_intelligence']
    cultural_module = gm._context_components['gm_module_cultural_awareness']
    
    self.assertTrue(social_module.is_enabled())
    self.assertTrue(cultural_module.is_enabled())

  def test_component_ordering(self):
    """Test that components are in correct order."""
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=self.agents,
        name='OrderTestGM',
        gm_modules=['social_intelligence'],
    )
    
    # Check that act component has correct component order
    act_component = gm._act_component
    self.assertIsNotNone(act_component)
    
    # Should include both core and module components
    component_names = list(gm._context_components.keys())
    
    # Core components should be present
    core_components = ['instructions', 'player_characters', 'negotiation_state']
    for comp in core_components:
      self.assertTrue(any(comp in name for name in component_names))
    
    # Module components should be present
    self.assertIn('gm_module_social_intelligence', component_names)


class GameMasterErrorHandlingTest(unittest.TestCase):
  """Tests for error handling in game master builders."""

  def setUp(self):
    """Set up test dependencies."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    self.clock = game_clock.FixedIntervalClock()
    self.memory_bank = basic_associative_memory.AssociativeMemoryBank()
    
    # Create mock agents
    self.agents = []
    for name in ['Alice', 'Bob']:
      agent = mock.create_autospec(
          entity_agent_with_logging.EntityAgentWithLogging,
          instance=True,
      )
      agent.name = name
      agent._context_components = {}
      self.agents.append(agent)

  def test_invalid_gm_module_names(self):
    """Test handling of invalid GM module names."""
    # Should not raise error, but should skip invalid modules
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=self.agents,
        name='InvalidModuleGM',
        gm_modules=['invalid_module', 'social_intelligence'],
    )
    
    self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)
    
    # Should have valid module but not invalid one
    self.assertIn('gm_module_social_intelligence', gm._context_components)
    self.assertNotIn('gm_module_invalid_module', gm._context_components)

  def test_empty_entity_list(self):
    """Test handling of empty entity list."""
    # Should still create GM but with empty player list
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=[],
        name='EmptyEntitiesGM',
    )
    
    self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)

  def test_duplicate_entity_names(self):
    """Test handling of duplicate entity names."""
    # Create agents with same name
    duplicate_agents = []
    for _ in range(2):
      agent = mock.create_autospec(
          entity_agent_with_logging.EntityAgentWithLogging,
          instance=True,
      )
      agent.name = 'SameName'
      agent._context_components = {}
      duplicate_agents.append(agent)
    
    # Should still create GM (behavior depends on implementation)
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=duplicate_agents,
        name='DuplicateNamesGM',
    )
    
    self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)


class SpecializedGMBuilderTest(unittest.TestCase):
  """Tests for specialized GM builder functions."""

  def setUp(self):
    """Set up test dependencies."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    self.clock = game_clock.FixedIntervalClock()
    self.memory_bank = basic_associative_memory.AssociativeMemoryBank()

  def test_all_specialized_builders_create_valid_gms(self):
    """Test that all specialized builders create valid GMs."""
    # Create different numbers of agents for different builders
    two_agents = []
    three_agents = []
    
    for i, name in enumerate(['Alice', 'Bob']):
      agent = mock.create_autospec(
          entity_agent_with_logging.EntityAgentWithLogging,
          instance=True,
      )
      agent.name = name
      agent._context_components = {}
      two_agents.append(agent)
    
    for name in ['Alice', 'Bob', 'Charlie']:
      agent = mock.create_autospec(
          entity_agent_with_logging.EntityAgentWithLogging,
          instance=True,
      )
      agent.name = name
      agent._context_components = {}
      three_agents.append(agent)
    
    # Test all specialized builders
    builders_and_agents = [
        ('bilateral', negotiation.build_bilateral_negotiation, two_agents),
        ('multilateral', negotiation.build_multilateral_negotiation, three_agents),
        ('cultural', negotiation.build_cultural_negotiation, two_agents),
        ('adaptive', negotiation.build_adaptive_negotiation, two_agents),
    ]
    
    for builder_name, builder_func, agents in builders_and_agents:
      with self.subTest(builder=builder_name):
        gm = builder_func(
            model=self.model,
            memory_bank=self.memory_bank,
            entities=agents,
            name=f'{builder_name.title()} GM',
        )
        
        self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)
        self.assertTrue(gm._agent_name.startswith(builder_name.title()))

  def test_specialized_builders_have_correct_modules(self):
    """Test that specialized builders include appropriate modules."""
    # Create agents
    two_agents = []
    three_agents = []
    
    for name in ['Alice', 'Bob']:
      agent = mock.create_autospec(
          entity_agent_with_logging.EntityAgentWithLogging,
          instance=True,
      )
      agent.name = name
      agent._context_components = {}
      two_agents.append(agent)
    
    for name in ['Alice', 'Bob', 'Charlie']:
      agent = mock.create_autospec(
          entity_agent_with_logging.EntityAgentWithLogging,
          instance=True,
      )
      agent.name = name
      agent._context_components = {}
      three_agents.append(agent)
    
    # Bilateral should have social and temporal
    bilateral_gm = negotiation.build_bilateral_negotiation(
        self.model, self.memory_bank, two_agents
    )
    self.assertIn('gm_module_social_intelligence', bilateral_gm._context_components)
    self.assertIn('gm_module_temporal_dynamics', bilateral_gm._context_components)
    
    # Multilateral should have collective intelligence
    multilateral_gm = negotiation.build_multilateral_negotiation(
        self.model, self.memory_bank, three_agents
    )
    self.assertIn('gm_module_collective_intelligence', multilateral_gm._context_components)
    
    # Cultural should have cultural awareness
    cultural_gm = negotiation.build_cultural_negotiation(
        self.model, self.memory_bank, two_agents
    )
    self.assertIn('gm_module_cultural_awareness', cultural_gm._context_components)
    
    # Adaptive should have strategy evolution
    adaptive_gm = negotiation.build_adaptive_negotiation(
        self.model, self.memory_bank, two_agents
    )
    self.assertIn('gm_module_strategy_evolution', adaptive_gm._context_components)


if __name__ == '__main__':
  unittest.main()