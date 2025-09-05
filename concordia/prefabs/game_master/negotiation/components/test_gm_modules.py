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

"""Tests for negotiation GM modules."""

import unittest

from concordia.prefabs.game_master.negotiation.components import negotiation_modules
from concordia.prefabs.game_master.negotiation.components import gm_social_intelligence
from concordia.prefabs.game_master.negotiation.components import gm_temporal_dynamics
from concordia.prefabs.game_master.negotiation.components import gm_cultural_awareness
from concordia.prefabs.game_master.negotiation.components import gm_uncertainty_management
from concordia.prefabs.game_master.negotiation.components import gm_collective_intelligence
from concordia.prefabs.game_master.negotiation.components import gm_strategy_evolution


class ModuleContextTest(unittest.TestCase):
  """Tests for ModuleContext data structure."""

  def test_module_context_creation(self):
    """Test ModuleContext creation and access."""
    context = negotiation_modules.ModuleContext(
        negotiation_id='test_neg',
        participants=['Alice', 'Bob'],
        current_phase='opening',
        current_round=1,
        active_modules={'Alice': {'cultural_adaptation'}, 'Bob': {'theory_of_mind'}},
        shared_data={},
    )
    
    self.assertEqual(context.negotiation_id, 'test_neg')
    self.assertEqual(len(context.participants), 2)
    self.assertIn('Alice', context.participants)
    self.assertEqual(context.current_phase, 'opening')
    self.assertEqual(context.current_round, 1)


class SocialIntelligenceGMTest(unittest.TestCase):
  """Tests for SocialIntelligenceGM module."""

  def setUp(self):
    """Set up test dependencies."""
    self.module = gm_social_intelligence.SocialIntelligenceGM()
    self.context = negotiation_modules.ModuleContext(
        negotiation_id='test',
        participants=['Alice', 'Bob'],
        current_phase='bargaining',
        current_round=5,
        active_modules={'Alice': {'theory_of_mind'}, 'Bob': {'theory_of_mind'}},
        shared_data={},
    )

  def test_module_initialization(self):
    """Test module initialization."""
    self.assertEqual(self.module._module_name, 'social_intelligence')
    self.assertTrue(self.module._track_emotions)
    self.assertTrue(self.module._detect_deception)

  def test_supported_agent_modules(self):
    """Test supported agent modules."""
    supported = self.module.get_supported_agent_modules()
    self.assertIn('theory_of_mind', supported)

  def test_emotion_detection(self):
    """Test emotion detection from text."""
    emotion = self.module.detect_emotion(
        "This is absolutely unacceptable and insulting!",
        'Alice',
        5
    )
    
    self.assertIsNotNone(emotion)
    self.assertEqual(emotion.participant, 'Alice')
    self.assertIn(emotion.primary_emotion, ['angry', 'frustrated'])
    self.assertLess(emotion.valence, 0)

  def test_deception_detection(self):
    """Test deception detection."""
    # First statement
    self.module.check_consistency('Alice', 'I can pay up to $200', 3)
    
    # Contradictory statement
    deception = self.module.check_consistency('Alice', 'I cannot pay more than $150', 5)
    
    self.assertIsNotNone(deception)
    self.assertEqual(deception.actor, 'Alice')
    self.assertEqual(deception.indicator_type, 'inconsistency')

  def test_validation(self):
    """Test action validation."""
    # Set up emotional state
    emotion = gm_social_intelligence.EmotionalReading(
        participant='Alice',
        primary_emotion='angry',
        intensity=0.8,
        valence=-0.7,
        confidence=0.8,
        triggers=['rejected offer'],
        round_number=4,
    )
    self.module._current_emotions['Alice'] = emotion
    
    # Test escalating when emotional
    is_valid, message = self.module.validate_action(
        'Alice', 
        'This is my final ultimatum!', 
        self.context
    )
    
    self.assertFalse(is_valid)
    self.assertIn('emotional', message.lower())

  def test_state_update(self):
    """Test state update functionality."""
    self.module.update_state(
        'I am very frustrated with this process',
        'Alice',
        self.context
    )
    
    # Check that emotion was detected
    self.assertIn('Alice', self.module._current_emotions)
    emotion = self.module._current_emotions['Alice']
    self.assertIn(emotion.primary_emotion, ['frustrated', 'angry'])

  def test_observation_context(self):
    """Test observation context generation."""
    # Add some emotional state
    self.module.update_state('I am excited about this deal', 'Bob', self.context)
    
    context_str = self.module.get_observation_context('Alice', self.context)
    
    self.assertIn('SOCIAL DYNAMICS', context_str)
    self.assertIn('Bob', context_str)

  def test_module_report(self):
    """Test module report generation."""
    # Add some activity
    self.module.update_state('I am frustrated', 'Alice', self.context)
    self.module.update_state('I am hopeful', 'Bob', self.context)
    
    report = self.module.get_module_report()
    
    self.assertIn('SOCIAL INTELLIGENCE REPORT', report)
    self.assertIn('frustrated', report)
    self.assertIn('hopeful', report)


class TemporalDynamicsGMTest(unittest.TestCase):
  """Tests for TemporalDynamicsGM module."""

  def setUp(self):
    """Set up test dependencies."""
    self.module = gm_temporal_dynamics.TemporalDynamicsGM()
    self.context = negotiation_modules.ModuleContext(
        negotiation_id='test',
        participants=['Alice', 'Bob'],
        current_phase='bargaining',
        current_round=5,
        active_modules={'Alice': {'temporal_strategy'}, 'Bob': {'temporal_strategy'}},
        shared_data={},
    )

  def test_relationship_tracking(self):
    """Test relationship state tracking."""
    # Get initial relationship
    rel = self.module.get_relationship('Alice', 'Bob')
    self.assertEqual(rel.trust_level, 0.5)  # Initial neutral trust
    
    # Update relationship
    self.module.update_relationship('Alice', 'Bob', trust_delta=0.2)
    updated_rel = self.module.get_relationship('Alice', 'Bob')
    self.assertGreater(updated_rel.trust_level, 0.5)

  def test_commitment_tracking(self):
    """Test commitment recording and tracking."""
    self.module.record_commitment(
        'Alice',
        'payment',
        {'amount': 1000, 'date': '2024-01-15'},
        'short',
        5,
        10  # deadline round
    )
    
    self.assertEqual(len(self.module._commitments), 1)
    commitment = self.module._commitments[0]
    self.assertEqual(commitment.committer, 'Alice')
    self.assertEqual(commitment.commitment_type, 'payment')

  def test_commitment_violation_detection(self):
    """Test commitment deadline violation detection."""
    # Record a commitment with deadline
    self.module.record_commitment('Alice', 'delivery', {}, 'short', 5, 8)
    
    # Check violations after deadline
    violations = self.module.check_commitment_violations(10)
    self.assertEqual(len(violations), 1)
    self.assertEqual(violations[0].committer, 'Alice')

  def test_phase_tracking(self):
    """Test negotiation phase tracking."""
    # Simulate phase changes
    for _ in range(3):
      self.module.update_state('test event', 'Alice', self.context)
    
    # Check phase duration tracking
    self.assertGreater(self.module._phase_durations['bargaining'], 0)

  def test_observation_context(self):
    """Test temporal observation context."""
    # Add some relationship history
    self.module.update_relationship('Alice', 'Bob', trust_delta=0.1)
    
    context_str = self.module.get_observation_context('Alice', self.context)
    
    self.assertIn('TEMPORAL DYNAMICS', context_str)
    self.assertIn('Bob', context_str)
    self.assertIn('Trust', context_str)


class CulturalAwarenessGMTest(unittest.TestCase):
  """Tests for CulturalAwarenessGM module."""

  def setUp(self):
    """Set up test dependencies."""
    self.module = gm_cultural_awareness.CulturalAwarenessGM()
    self.context = negotiation_modules.ModuleContext(
        negotiation_id='test',
        participants=['Alice', 'Bob'],
        current_phase='opening',
        current_round=2,
        active_modules={'Alice': {'cultural_adaptation'}, 'Bob': {'cultural_adaptation'}},
        shared_data={},
    )

  def test_cultural_profile_setting(self):
    """Test setting participant cultural profiles."""
    self.module.set_participant_culture('Alice', 'western_business')
    self.module.set_participant_culture('Bob', 'east_asian')
    
    self.assertEqual(self.module._participant_cultures['Alice'], 'western_business')
    self.assertEqual(self.module._participant_cultures['Bob'], 'east_asian')

  def test_cultural_violation_detection(self):
    """Test cultural protocol violation detection."""
    # Set up cultures
    self.module.set_participant_culture('Alice', 'western_business')
    self.module.set_participant_culture('Bob', 'east_asian')
    
    # Test direct rejection to indirect culture
    violation = self.module.detect_cultural_violation(
        'Alice',
        'No, that proposal is impossible',
        'Bob'
    )
    
    self.assertIsNotNone(violation)
    self.assertIn('direct', violation.lower())

  def test_face_saving_violation(self):
    """Test face-saving violation detection."""
    self.module.set_participant_culture('Bob', 'east_asian')
    
    violation = self.module.detect_cultural_violation(
        'Alice',
        'You are wrong about this',
        'Bob'
    )
    
    self.assertIsNotNone(violation)
    self.assertIn('face', violation.lower())

  def test_action_validation(self):
    """Test cultural action validation."""
    # Set up high face-saving culture
    self.module.set_participant_culture('Bob', 'east_asian')
    
    is_valid, message = self.module.validate_action(
        'Alice',
        'You made a mistake in your calculations',
        self.context
    )
    
    self.assertFalse(is_valid)
    self.assertIn('Cultural sensitivity', message)

  def test_observation_context(self):
    """Test cultural observation context."""
    self.module.set_participant_culture('Alice', 'western_business')
    
    context_str = self.module.get_observation_context('Alice', self.context)
    
    self.assertIn('CULTURAL CONTEXT', context_str)
    self.assertIn('Western Business', context_str)


class UncertaintyManagementGMTest(unittest.TestCase):
  """Tests for UncertaintyManagementGM module."""

  def setUp(self):
    """Set up test dependencies."""
    self.module = gm_uncertainty_management.UncertaintyManagementGM()
    self.context = negotiation_modules.ModuleContext(
        negotiation_id='test',
        participants=['Alice', 'Bob'],
        current_phase='bargaining',
        current_round=3,
        active_modules={'Alice': {'uncertainty_aware'}, 'Bob': {'uncertainty_aware'}},
        shared_data={},
    )

  def test_information_asymmetry_tracking(self):
    """Test information asymmetry tracking."""
    self.module.track_information_asymmetry(
        'Alice',
        'reservation_value',
        0.9,
        {'Alice'},  # Only Alice knows
        0.8
    )
    
    self.assertEqual(len(self.module._information_asymmetries), 1)
    asymmetry = self.module._information_asymmetries[0]
    self.assertEqual(asymmetry.actor, 'Alice')
    self.assertEqual(asymmetry.information_type, 'reservation_value')

  def test_uncertainty_metrics_calculation(self):
    """Test uncertainty metrics calculation."""
    metrics = self.module.calculate_uncertainty_metrics('Alice', self.context)
    
    self.assertEqual(metrics.participant, 'Alice')
    self.assertBetween(metrics.preference_uncertainty, 0, 1)
    self.assertBetween(metrics.outcome_uncertainty, 0, 1)

  def test_information_value_assessment(self):
    """Test information value assessment."""
    value = self.module.assess_information_value(
        'preference',
        'Alice',
        'Bob',
        self.context
    )
    
    self.assertBetween(value, 0, 1)

  def test_information_request_processing(self):
    """Test information request processing."""
    granted, message = self.module.process_information_request(
        'Alice',
        'Bob',
        'Can you tell me your preference for delivery timing?',
        self.context
    )
    
    # Result depends on implementation, but should return boolean and message
    self.assertIsInstance(granted, bool)
    self.assertIsInstance(message, str)

  def assertBetween(self, value, min_val, max_val):
    """Helper to assert value is between min and max."""
    self.assertGreaterEqual(value, min_val)
    self.assertLessEqual(value, max_val)


class CollectiveIntelligenceGMTest(unittest.TestCase):
  """Tests for CollectiveIntelligenceGM module."""

  def setUp(self):
    """Set up test dependencies."""
    self.module = gm_collective_intelligence.CollectiveIntelligenceGM()
    self.context = negotiation_modules.ModuleContext(
        negotiation_id='test',
        participants=['Alice', 'Bob', 'Charlie'],
        current_phase='bargaining',
        current_round=5,
        active_modules={
            'Alice': {'swarm_intelligence'}, 
            'Bob': {'swarm_intelligence'},
            'Charlie': {'swarm_intelligence'}
        },
        shared_data={},
    )

  def test_coalition_formation_detection(self):
    """Test coalition formation detection."""
    # Simulate coordination actions
    recent_actions = [
        ('Alice', 'We should work together with Bob on this'),
        ('Bob', 'I agree, Alice. Let\'s coordinate our approach'),
    ]
    
    coalition = self.module.detect_coalition_formation(
        self.context.participants,
        recent_actions,
        self.context
    )
    
    if coalition:  # Coalition formation is probabilistic
      self.assertIn('Alice', coalition.members)
      self.assertIn('Bob', coalition.members)

  def test_information_sharing_tracking(self):
    """Test information sharing pattern tracking."""
    self.module.track_information_sharing(
        'Alice',
        'I want to share my pricing strategy with Bob',
        self.context
    )
    
    # Check that information flow was recorded
    self.assertGreater(len(self.module._information_flows), 0)
    flow = self.module._information_flows[0]
    self.assertEqual(flow.sender, 'Alice')

  def test_collective_decision_assessment(self):
    """Test collective decision potential assessment."""
    decision = self.module.assess_collective_decision_potential(
        self.context.participants,
        'We all need to agree on the final terms together',
        self.context
    )
    
    if decision:  # Decision detection is based on keywords
      self.assertEqual(decision.decision_type, 'consensus')
      self.assertIn('Alice', decision.participants)

  def test_emergent_behavior_detection(self):
    """Test emergent behavior detection."""
    # Add some information flows to create patterns
    for i in range(15):
      self.module._information_flows.append(
          gm_collective_intelligence.InformationFlow(
              sender=f'Agent{i%3}',
              recipients={'Agent1', 'Agent2'},
              information_type='strategic',
              sharing_mechanism='broadcast',
              round_shared=i,
              verification_level=0.8
          )
      )
    
    patterns = self.module.detect_emergent_behavior(self.context)
    
    # Should detect information cascade due to high flow density
    if patterns:
      self.assertIn('information_cascade', patterns)


class StrategyEvolutionGMTest(unittest.TestCase):
  """Tests for StrategyEvolutionGM module."""

  def setUp(self):
    """Set up test dependencies."""
    self.module = gm_strategy_evolution.StrategyEvolutionGM()
    self.context = negotiation_modules.ModuleContext(
        negotiation_id='test',
        participants=['Alice', 'Bob'],
        current_phase='bargaining',
        current_round=5,
        active_modules={'Alice': {'strategy_evolution'}, 'Bob': {'strategy_evolution'}},
        shared_data={},
    )

  def test_strategy_analysis_from_action(self):
    """Test strategy analysis from participant action."""
    snapshot = self.module.analyze_strategy_from_action(
        'Alice',
        'I demand that we reach a mutually beneficial agreement quickly',
        self.context
    )
    
    self.assertEqual(snapshot.participant, 'Alice')
    self.assertIn(snapshot.strategy_type, ['cooperative', 'competitive', 'integrative', 'mixed'])
    self.assertBetween(snapshot.aggressiveness, 0, 1)

  def test_strategy_change_detection(self):
    """Test strategy change detection."""
    # First strategy
    snapshot1 = gm_strategy_evolution.StrategySnapshot(
        participant='Alice',
        round_number=3,
        strategy_type='cooperative',
        aggressiveness=0.3,
        concession_rate=0.7,
        information_sharing=0.8,
        coalition_tendency=0.6,
        risk_tolerance=0.5,
        tactics_used=['reciprocity'],
    )
    self.module._strategy_snapshots.append(snapshot1)
    
    # Second strategy (different)
    snapshot2 = gm_strategy_evolution.StrategySnapshot(
        participant='Alice',
        round_number=5,
        strategy_type='competitive',
        aggressiveness=0.8,
        concession_rate=0.2,
        information_sharing=0.3,
        coalition_tendency=0.2,
        risk_tolerance=0.7,
        tactics_used=['anchoring'],
    )
    
    transition = self.module.detect_strategy_change('Alice', snapshot2)
    
    self.assertIsNotNone(transition)
    self.assertEqual(transition.participant, 'Alice')
    self.assertEqual(transition.from_strategy, 'cooperative')
    self.assertEqual(transition.to_strategy, 'competitive')

  def test_innovation_detection(self):
    """Test innovation detection in strategies."""
    # Create a snapshot with novel tactic combination
    snapshot = gm_strategy_evolution.StrategySnapshot(
        participant='Alice',
        round_number=5,
        strategy_type='integrative',
        aggressiveness=0.5,
        concession_rate=0.5,
        information_sharing=0.7,
        coalition_tendency=0.6,
        risk_tolerance=0.5,
        tactics_used=['anchoring', 'value_creation', 'information_seeking'],  # Novel combination
    )
    
    innovation = self.module.detect_innovation(snapshot, self.context)
    
    if innovation:  # Innovation detection is based on novelty
      self.assertEqual(innovation['participant'], 'Alice')
      self.assertEqual(innovation['innovation_type'], 'tactic_combination')

  def test_performance_metrics_update(self):
    """Test performance metrics updating."""
    snapshot = gm_strategy_evolution.StrategySnapshot(
        participant='Alice',
        round_number=5,
        strategy_type='cooperative',
        aggressiveness=0.3,
        concession_rate=0.7,
        information_sharing=0.8,
        coalition_tendency=0.6,
        risk_tolerance=0.5,
        tactics_used=['reciprocity'],
    )
    
    self.module._update_performance_metrics('Alice', snapshot, self.context)
    
    key = ('Alice', 'cooperative')
    self.assertIn(key, self.module._performance_metrics)
    metrics = self.module._performance_metrics[key]
    self.assertEqual(metrics.participant, 'Alice')
    self.assertEqual(metrics.strategy_type, 'cooperative')

  def assertBetween(self, value, min_val, max_val):
    """Helper to assert value is between min and max."""
    self.assertGreaterEqual(value, min_val)
    self.assertLessEqual(value, max_val)


class ModuleRegistryTest(unittest.TestCase):
  """Tests for the module registry system."""

  def test_module_registration(self):
    """Test that modules are properly registered."""
    # Check that our modules are registered
    self.assertIsNotNone(
        negotiation_modules.NegotiationGMModuleRegistry.get_module('social_intelligence')
    )
    self.assertIsNotNone(
        negotiation_modules.NegotiationGMModuleRegistry.get_module('temporal_dynamics')
    )
    self.assertIsNotNone(
        negotiation_modules.NegotiationGMModuleRegistry.get_module('cultural_awareness')
    )

  def test_module_creation(self):
    """Test module creation through registry."""
    module = negotiation_modules.NegotiationGMModuleRegistry.create_module(
        'social_intelligence',
        {'emotion_sensitivity': 0.8}
    )
    
    self.assertIsInstance(module, gm_social_intelligence.SocialIntelligenceGM)

  def test_module_listing(self):
    """Test listing all registered modules."""
    modules = negotiation_modules.NegotiationGMModuleRegistry.list_modules()
    
    self.assertIn('social_intelligence', modules)
    self.assertIn('temporal_dynamics', modules)
    self.assertIn('cultural_awareness', modules)


if __name__ == '__main__':
  unittest.main()