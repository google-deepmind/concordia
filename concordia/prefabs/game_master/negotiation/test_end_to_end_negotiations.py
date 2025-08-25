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

"""End-to-end tests for complete negotiation simulations."""

import datetime
import unittest
from unittest import mock

from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.language_model import language_model
from concordia.environment import simulation_environment
from concordia.prefabs.entity.negotiation import base_negotiator
from concordia.prefabs.entity.negotiation import advanced_negotiator
from concordia.prefabs.game_master.negotiation import negotiation


class EndToEndNegotiationTest(unittest.TestCase):
  """End-to-end tests for complete negotiation scenarios."""

  def setUp(self):
    """Set up test environment."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    
    # Mock reasonable responses for different types of queries
    def mock_sample_text(prompt, **kwargs):
      if 'emotion' in prompt.lower():
        return 'neutral'
      elif 'cultural' in prompt.lower():
        return 'western_business'
      elif 'strategy' in prompt.lower():
        return 'cooperative'
      elif 'offer' in prompt.lower() or 'price' in prompt.lower():
        return 'I propose a price of $150'
      elif 'accept' in prompt.lower():
        return 'accept'
      else:
        return 'I understand the situation'
    
    self.model.sample_text.side_effect = mock_sample_text
    
    self.clock = game_clock.FixedIntervalClock()
    self.memory_bank = basic_associative_memory.AssociativeMemoryBank()

  def create_simple_agents(self, names, agent_builder=base_negotiator.build_agent):
    """Helper to create simple test agents."""
    agents = []
    for name in names:
      agent = agent_builder(
          model=self.model,
          memory_bank=self.memory_bank,
          name=name,
          reservation_value=100.0,
      )
      agents.append(agent)
    return agents

  def test_basic_bilateral_negotiation(self):
    """Test basic bilateral price negotiation."""
    # Create agents
    agents = self.create_simple_agents(['Buyer', 'Seller'])
    
    # Create game master
    gm = negotiation.build_bilateral_negotiation(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=agents,
        name='Price Negotiation',
    )
    
    # Verify GM was created successfully
    self.assertIsNotNone(gm)
    self.assertEqual(gm._agent_name, 'Price Negotiation')
    
    # Check that appropriate components are present
    self.assertIn('negotiation_state', gm._context_components)
    self.assertIn('gm_module_social_intelligence', gm._context_components)
    
    # Test that we can get the negotiation state component
    state_component = gm._context_components['negotiation_state']
    
    # Start a negotiation
    state = state_component.start_negotiation(
        negotiation_id='test_bilateral',
        participants=['Buyer', 'Seller'],
    )
    
    self.assertEqual(state.negotiation_id, 'test_bilateral')
    self.assertEqual(len(state.participants), 2)
    self.assertEqual(state.phase, 'opening')

  def test_multilateral_contract_negotiation(self):
    """Test multi-party contract negotiation."""
    # Create three agents with different goals
    agent_configs = [
        ('CompanyA', 200.0),
        ('CompanyB', 150.0),
        ('CompanyC', 180.0),
    ]
    
    agents = []
    for name, reservation in agent_configs:
      agent = base_negotiator.build_agent(
          model=self.model,
          memory_bank=self.memory_bank,
          name=name,
          goal=f'Secure contract terms worth at least ${reservation}',
          reservation_value=reservation,
      )
      agents.append(agent)
    
    # Create multilateral GM
    gm = negotiation.build_multilateral_negotiation(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=agents,
        name='Contract Negotiation',
    )
    
    # Verify GM has collective intelligence capabilities
    self.assertIn('gm_module_collective_intelligence', gm._context_components)
    self.assertIn('gm_module_uncertainty_management', gm._context_components)
    
    # Test coalition detection capability
    collective_module = gm._context_components['gm_module_collective_intelligence']
    
    # Simulate coordination actions
    recent_actions = [
        ('CompanyA', 'We should work together with CompanyB on this proposal'),
        ('CompanyB', 'I agree with CompanyA, let\'s coordinate our approach'),
    ]
    
    # Test coalition formation detection
    coalition = collective_module.detect_coalition_formation(
        ['CompanyA', 'CompanyB', 'CompanyC'],
        recent_actions,
        mock.Mock(current_round=5, participants=['CompanyA', 'CompanyB', 'CompanyC'])
    )
    
    # Coalition formation is probabilistic, so we just check it doesn't error
    # If a coalition is formed, it should have the right members
    if coalition:
      self.assertIn('CompanyA', coalition.members)
      self.assertIn('CompanyB', coalition.members)

  def test_cross_cultural_negotiation(self):
    """Test cross-cultural negotiation with cultural adaptation."""
    # Create agents with cultural awareness
    western_agent = advanced_negotiator.build_cultural_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='WesternRep',
        own_culture='western_business',
        goal='Negotiate trade agreement efficiently',
    )
    
    eastern_agent = advanced_negotiator.build_cultural_agent(
        model=self.model,
        memory_bank=self.memory_bank,
        name='EasternRep',
        own_culture='east_asian',
        goal='Build lasting relationship while securing favorable terms',
    )
    
    agents = [western_agent, eastern_agent]
    
    # Create cultural-aware GM
    gm = negotiation.build_cultural_negotiation(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=agents,
        name='Trade Negotiation',
    )
    
    # Verify cultural awareness is enabled
    self.assertIn('gm_module_cultural_awareness', gm._context_components)
    cultural_module = gm._context_components['gm_module_cultural_awareness']
    
    # Test cultural profile setting
    cultural_module.set_participant_culture('WesternRep', 'western_business')
    cultural_module.set_participant_culture('EasternRep', 'east_asian')
    
    # Test cultural violation detection
    violation = cultural_module.detect_cultural_violation(
        'WesternRep',
        'No, that proposal is completely wrong and unacceptable',
        'EasternRep'
    )
    
    # Should detect violation due to direct criticism to face-saving culture
    self.assertIsNotNone(violation)
    self.assertIn('face', violation.lower())

  def test_adaptive_strategy_negotiation(self):
    """Test negotiation with strategy evolution over time."""
    # Create adaptive agents
    adaptive_agents = []
    for name in ['AdaptiveAlice', 'AdaptiveBob']:
      agent = advanced_negotiator.build_adaptive_agent(
          model=self.model,
          memory_bank=self.memory_bank,
          name=name,
          learning_rate=0.1,
      )
      adaptive_agents.append(agent)
    
    # Create adaptive GM
    gm = negotiation.build_adaptive_negotiation(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=adaptive_agents,
        name='Learning Negotiation',
        max_rounds=20,
    )
    
    # Verify strategy evolution capability
    self.assertIn('gm_module_strategy_evolution', gm._context_components)
    evolution_module = gm._context_components['gm_module_strategy_evolution']
    
    # Test strategy analysis
    context = mock.Mock(
        current_round=5,
        current_phase='bargaining',
        participants=['AdaptiveAlice', 'AdaptiveBob']
    )
    
    snapshot = evolution_module.analyze_strategy_from_action(
        'AdaptiveAlice',
        'I think we should collaborate to find a mutually beneficial solution',
        context
    )
    
    self.assertEqual(snapshot.participant, 'AdaptiveAlice')
    self.assertIn(snapshot.strategy_type, ['cooperative', 'competitive', 'integrative', 'mixed'])
    
    # Test strategy change detection by creating a different strategy
    self.assertEqual(len(evolution_module._strategy_snapshots), 1)
    evolution_module._strategy_snapshots.append(snapshot)
    
    # Create a second snapshot with different strategy
    different_snapshot = evolution_module.analyze_strategy_from_action(
        'AdaptiveAlice',
        'I demand we accept my terms immediately',
        context
    )
    
    if different_snapshot.strategy_type != snapshot.strategy_type:
      transition = evolution_module.detect_strategy_change('AdaptiveAlice', different_snapshot)
      if transition:
        self.assertEqual(transition.participant, 'AdaptiveAlice')
        self.assertNotEqual(transition.from_strategy, transition.to_strategy)

  def test_information_asymmetry_scenario(self):
    """Test negotiation with information asymmetry handling."""
    # Create agents with uncertainty awareness
    uncertain_agents = []
    for name in ['InfoAlice', 'InfoBob']:
      agent = advanced_negotiator.build_agent(
          model=self.model,
          memory_bank=self.memory_bank,
          name=name,
          modules=['uncertainty_aware', 'theory_of_mind'],
      )
      uncertain_agents.append(agent)
    
    # Create GM with uncertainty management
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=uncertain_agents,
        name='Info Asymmetry Test',
        gm_modules=['uncertainty_management', 'social_intelligence'],
    )
    
    # Test uncertainty management
    self.assertIn('gm_module_uncertainty_management', gm._context_components)
    uncertainty_module = gm._context_components['gm_module_uncertainty_management']
    
    # Test information asymmetry tracking
    uncertainty_module.track_information_asymmetry(
        'InfoAlice',
        'reservation_value',
        0.9,  # Alice knows her reservation well
        {'InfoAlice'},  # Only Alice knows
        0.8   # High strategic value
    )
    
    # Test information request processing
    context = mock.Mock(
        current_round=3,
        current_phase='bargaining',
        participants=['InfoAlice', 'InfoBob']
    )
    
    granted, message = uncertainty_module.process_information_request(
        'InfoBob',
        'InfoAlice', 
        'Can you tell me your minimum acceptable price?',
        context
    )
    
    # Should return valid response
    self.assertIsInstance(granted, bool)
    self.assertIsInstance(message, str)

  def test_temporal_dynamics_long_term_negotiation(self):
    """Test negotiation with temporal dynamics and relationship building."""
    # Create agents with temporal strategy
    temporal_agents = []
    for name in ['LongTermAlice', 'LongTermBob']:
      agent = advanced_negotiator.build_temporal_agent(
          model=self.model,
          memory_bank=self.memory_bank,
          name=name,
          discount_factor=0.9,  # High value on future outcomes
      )
      temporal_agents.append(agent)
    
    # Create GM with temporal dynamics
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=temporal_agents,
        gm_modules=['temporal_dynamics', 'social_intelligence'],
    )
    
    # Test temporal dynamics
    self.assertIn('gm_module_temporal_dynamics', gm._context_components)
    temporal_module = gm._context_components['gm_module_temporal_dynamics']
    
    # Test relationship tracking
    temporal_module.update_relationship(
        'LongTermAlice', 'LongTermBob', 
        trust_delta=0.1, commitment_delta=0.05
    )
    
    relationship = temporal_module.get_relationship('LongTermAlice', 'LongTermBob')
    self.assertGreater(relationship.trust_level, 0.5)  # Should be above initial 0.5
    
    # Test commitment tracking
    temporal_module.record_commitment(
        'LongTermAlice',
        'delivery',
        {'date': '2024-01-15', 'quality': 'high'},
        'medium',
        current_round=5,
        deadline_round=10
    )
    
    self.assertEqual(len(temporal_module._commitments), 1)
    commitment = temporal_module._commitments[0]
    self.assertEqual(commitment.committer, 'LongTermAlice')

  def test_complex_multi_module_scenario(self):
    """Test complex scenario with multiple GM modules active."""
    # Create sophisticated agents
    agents = []
    configs = [
        ('Alice', ['cultural_adaptation', 'theory_of_mind', 'temporal_strategy']),
        ('Bob', ['swarm_intelligence', 'uncertainty_aware', 'strategy_evolution']),
        ('Charlie', ['theory_of_mind', 'cultural_adaptation']),
    ]
    
    for name, modules in configs:
      agent = advanced_negotiator.build_agent(
          model=self.model,
          memory_bank=self.memory_bank,
          name=name,
          modules=modules,
      )
      agents.append(agent)
    
    # Create comprehensive GM
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=agents,
        name='Complex Negotiation',
        gm_modules=[
            'social_intelligence',
            'temporal_dynamics', 
            'cultural_awareness',
            'uncertainty_management',
            'collective_intelligence',
            'strategy_evolution'
        ],
    )
    
    # Verify all modules are present
    expected_modules = [
        'gm_module_social_intelligence',
        'gm_module_temporal_dynamics',
        'gm_module_cultural_awareness',
        'gm_module_uncertainty_management',
        'gm_module_collective_intelligence',
        'gm_module_strategy_evolution',
    ]
    
    for module_name in expected_modules:
      self.assertIn(module_name, gm._context_components)
    
    # Test that modules can work together
    context = mock.Mock(
        negotiation_id='complex_test',
        participants=['Alice', 'Bob', 'Charlie'],
        current_phase='bargaining',
        current_round=8,
        active_modules={
            'Alice': {'cultural_adaptation', 'theory_of_mind', 'temporal_strategy'},
            'Bob': {'swarm_intelligence', 'uncertainty_aware', 'strategy_evolution'},
            'Charlie': {'theory_of_mind', 'cultural_adaptation'},
        },
        shared_data={}
    )
    
    # Test each module can process the same event without conflicts
    event = "Alice proposes a collaborative approach to resolve cultural differences"
    
    for module_name in expected_modules:
      module = gm._context_components[module_name]
      
      # Each module should be able to validate the action
      is_valid, message = module.validate_action('Alice', event, context)
      self.assertIsInstance(is_valid, bool)
      if not is_valid:
        self.assertIsInstance(message, str)
      
      # Each module should be able to update its state
      module.update_state(event, 'Alice', context)
      
      # Each module should be able to provide observation context
      obs_context = module.get_observation_context('Bob', context)
      self.assertIsInstance(obs_context, str)
      
      # Each module should be able to generate a report
      report = module.get_module_report()
      self.assertIsInstance(report, str)


class SimulationRunnerTest(unittest.TestCase):
  """Test running complete simulations with the negotiation framework."""

  def setUp(self):
    """Set up simulation environment."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    
    # Mock responses for simulation
    responses = [
        'I propose we start with a fair offer',
        'That sounds reasonable, let me consider it',
        'I accept your proposal',
        'Great, I accept too',
    ]
    self.model.sample_text.side_effect = responses * 10  # Repeat for multiple rounds
    
    self.clock = game_clock.FixedIntervalClock()
    self.memory_bank = basic_associative_memory.AssociativeMemoryBank()

  def test_simulation_initialization_and_basic_run(self):
    """Test that simulation can be initialized and run basic steps."""
    # Create agents
    agents = []
    for name in ['Buyer', 'Seller']:
      agent = base_negotiator.build_agent(
          model=self.model,
          memory_bank=self.memory_bank,
          name=name,
      )
      agents.append(agent)
    
    # Create GM
    gm = negotiation.build_bilateral_negotiation(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=agents,
    )
    
    # Test that all components exist and can be accessed
    self.assertIsNotNone(gm._act_component)
    self.assertTrue(len(gm._context_components) > 0)
    
    # Test that we can get the next acting component
    acting_component = gm._context_components.get('acting')
    if acting_component:
      # Should be able to get next acting entity
      self.assertTrue(hasattr(acting_component, 'get_next_acting'))

  def test_negotiation_state_lifecycle(self):
    """Test complete negotiation state lifecycle."""
    agents = []
    for name in ['Alice', 'Bob']:
      agent = base_negotiator.build_agent(
          model=self.model,
          memory_bank=self.memory_bank,
          name=name,
      )
      agents.append(agent)
    
    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=agents,
        name='Lifecycle Test',
        max_rounds=5,
    )
    
    # Get state component
    state_component = gm._context_components['negotiation_state']
    
    # 1. Start negotiation
    state = state_component.start_negotiation('lifecycle_test', ['Alice', 'Bob'])
    self.assertEqual(state.phase, 'opening')
    self.assertEqual(state.current_round, 0)
    
    # 2. Record offers
    offer1 = state_component.record_offer(
        'lifecycle_test', 'Alice', 'Bob', 'initial', {'price': 150}
    )
    self.assertEqual(len(state.offers_history), 1)
    self.assertEqual(state.active_offer, offer1)
    
    # 3. Record counter-offer
    offer2 = state_component.record_offer(
        'lifecycle_test', 'Bob', 'Alice', 'counter', {'price': 130}
    )
    self.assertEqual(len(state.offers_history), 2)
    self.assertEqual(state.active_offer, offer2)
    
    # 4. Progress through phases
    state_component.advance_phase('lifecycle_test')
    updated_state = state_component.get_negotiation_state('lifecycle_test')
    self.assertEqual(updated_state.phase, 'bargaining')
    
    # 5. Record agreement
    agreement = state_component.record_agreement(
        'lifecycle_test', ['Alice', 'Bob'], {'price': 140}, 'final'
    )
    self.assertEqual(updated_state.agreement, agreement)
    self.assertTrue(updated_state.concluded)


if __name__ == '__main__':
  unittest.main()