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

"""Tests for negotiation agent components."""

import datetime
import unittest
from unittest import mock

from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.language_model import language_model
from concordia.prefabs.entity.negotiation.components import cultural_adaptation
from concordia.prefabs.entity.negotiation.components import negotiation_memory
from concordia.prefabs.entity.negotiation.components import negotiation_strategy
from concordia.prefabs.entity.negotiation.components import theory_of_mind


class NegotiationStrategyTest(unittest.TestCase):
  """Tests for NegotiationStrategy component."""

  def setUp(self):
    """Set up test dependencies."""
    self.agent_name = 'TestAgent'
    self.strategy = negotiation_strategy.BasicNegotiationStrategy(
        agent_name=self.agent_name,
        negotiation_style='integrative',
        reservation_value=100.0,
        target_value=200.0,
    )

  def test_cooperative_strategy_selection(self):
    """Test cooperative strategy behavior."""
    strategy = negotiation_strategy.BasicNegotiationStrategy(
        agent_name=self.agent_name,
        negotiation_style='cooperative',
        reservation_value=100.0,
        target_value=200.0,
    )

    context = strategy.pre_act('test action spec')
    self.assertIn('cooperative', context.lower())
    self.assertIn('mutual benefit', context.lower())

  def test_competitive_strategy_selection(self):
    """Test competitive strategy behavior."""
    strategy = negotiation_strategy.BasicNegotiationStrategy(
        agent_name=self.agent_name,
        negotiation_style='competitive',
        reservation_value=100.0,
        target_value=200.0,
    )

    context = strategy.pre_act('test action spec')
    self.assertIn('competitive', context.lower())
    self.assertIn('maximize', context.lower())

  def test_integrative_strategy_selection(self):
    """Test integrative strategy behavior."""
    context = self.strategy.pre_act('test action spec')
    self.assertIn('integrative', context.lower())
    self.assertIn('value', context.lower())

  def test_offer_evaluation(self):
    """Test offer evaluation logic."""
    # Offer above target should be excellent
    evaluation = self.strategy._evaluate_offer(250.0)
    self.assertIn('excellent', evaluation.lower())

    # Offer at reservation should be acceptable
    evaluation = self.strategy._evaluate_offer(100.0)
    self.assertIn('acceptable', evaluation.lower())

    # Offer below reservation should be unacceptable
    evaluation = self.strategy._evaluate_offer(50.0)
    self.assertIn('unacceptable', evaluation.lower())

  def test_concession_calculation(self):
    """Test concession amount calculation."""
    # First concession should be larger
    concession1 = self.strategy._calculate_concession_amount(1, 5)
    concession2 = self.strategy._calculate_concession_amount(4, 5)
    
    self.assertGreater(concession1, concession2)
    self.assertGreater(concession1, 0)
    self.assertGreater(concession2, 0)


class NegotiationMemoryTest(unittest.TestCase):
  """Tests for NegotiationMemory component."""

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
    
    self.memory = negotiation_memory.NegotiationMemory(
        agent_name='TestAgent',
        memory_bank=self.memory_bank,
    )

  def test_offer_tracking(self):
    """Test offer tracking functionality."""
    # Record an offer
    self.memory.record_offer(
        offerer='Alice',
        recipient='Bob',
        terms={'price': 100},
        offer_type='initial',
    )

    context = self.memory.pre_act('test action spec')
    self.assertIn('Alice', context)
    self.assertIn('100', context)

  def test_agreement_tracking(self):
    """Test agreement tracking functionality."""
    # Record an agreement
    self.memory.record_agreement(
        parties=['Alice', 'Bob'],
        terms={'price': 150, 'delivery': '1 week'},
        agreement_type='final',
    )

    context = self.memory.pre_act('test action spec')
    self.assertIn('agreement', context.lower())
    self.assertIn('150', context)

  def test_negotiation_history(self):
    """Test negotiation history retrieval."""
    # Add multiple events
    self.memory.record_offer('Alice', 'Bob', {'price': 100}, 'initial')
    self.memory.record_offer('Bob', 'Alice', {'price': 120}, 'counter')
    
    context = self.memory.pre_act('test action spec')
    self.assertIn('Alice', context)
    self.assertIn('Bob', context)
    self.assertIn('100', context)
    self.assertIn('120', context)


class CulturalAdaptationTest(unittest.TestCase):
  """Tests for CulturalAdaptation component."""

  def setUp(self):
    """Set up test dependencies."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    
    self.adaptation = cultural_adaptation.CulturalAdaptation(
        model=self.model,
        own_culture='western_business',
        adaptation_level=0.7,
    )

  def test_cultural_profile_access(self):
    """Test access to cultural profiles."""
    western = cultural_adaptation.CULTURAL_PROFILES['western_business']
    self.assertEqual(western.name, 'Western Business (USA/UK)')
    self.assertGreater(western.directness, 0.5)

    eastern = cultural_adaptation.CULTURAL_PROFILES['east_asian']
    self.assertEqual(eastern.name, 'East Asian (Japan/China)')
    self.assertLess(eastern.directness, 0.5)

  def test_cultural_distance_calculation(self):
    """Test cultural distance calculation."""
    western = cultural_adaptation.CULTURAL_PROFILES['western_business']
    eastern = cultural_adaptation.CULTURAL_PROFILES['east_asian']
    
    distance = western.get_distance_from(eastern)
    self.assertGreater(distance, 0.5)  # Should be significant distance

  def test_culture_detection(self):
    """Test culture detection from communication."""
    # Mock the language model response
    self.model.sample_text.return_value = 'east_asian'
    
    detected = self.adaptation.detect_cultural_style(
        "Please consider our humble proposal with great respect"
    )
    self.assertEqual(detected, 'east_asian')

  def test_adaptation_context(self):
    """Test cultural adaptation context."""
    # Set a counterpart culture
    self.adaptation._detected_culture = 'east_asian'
    self.adaptation._counterpart_profile = cultural_adaptation.CULTURAL_PROFILES['east_asian']
    
    context = self.adaptation.pre_act('test action spec')
    self.assertIn('CULTURAL ADAPTATION', context)
    self.assertIn('East Asian', context)


class TheoryOfMindTest(unittest.TestCase):
  """Tests for TheoryOfMind component."""

  def setUp(self):
    """Set up test dependencies."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    
    self.tom = theory_of_mind.TheoryOfMind(
        model=self.model,
        max_recursion_depth=2,
        emotion_sensitivity=0.7,
    )

  def test_emotion_detection(self):
    """Test emotion detection from text."""
    # Mock responses for emotion detection
    self.model.sample_text.side_effect = [
        'frustrated',  # Primary emotion
        'This offer is insulting',  # Emotional context
    ]
    
    emotion = self.tom.detect_emotion(
        "This offer is completely unacceptable and insulting!",
        'Alice'
    )
    
    self.assertIsNotNone(emotion)
    self.assertEqual(emotion.participant, 'Alice')
    self.assertLess(emotion.valence, 0)  # Negative emotion

  def test_belief_modeling(self):
    """Test belief modeling functionality."""
    # Mock belief inference
    self.model.sample_text.return_value = 'Bob believes Alice wants $150'
    
    belief = self.tom.infer_belief(
        'Bob',
        'Alice',
        "Alice said she really needs at least $150"
    )
    
    self.assertIn('150', belief)
    self.assertIn('Alice', belief)

  def test_intention_prediction(self):
    """Test intention prediction."""
    # Mock intention prediction
    self.model.sample_text.return_value = 'Alice intends to make a counteroffer'
    
    intention = self.tom.predict_intention(
        'Alice',
        "I need to think about this offer..."
    )
    
    self.assertIn('counteroffer', intention)
    self.assertIn('Alice', intention)

  def test_empathic_response(self):
    """Test empathic response generation."""
    # Create an emotion
    emotion = theory_of_mind.EmotionalState(
        participant='Bob',
        primary_emotion='frustrated',
        intensity=0.8,
        valence=-0.7,
        context=['offer rejection'],
    )
    
    # Mock empathic response
    self.model.sample_text.return_value = 'I understand your frustration'
    
    response = self.tom.generate_empathic_response(emotion)
    self.assertIn('understand', response)


if __name__ == '__main__':
  unittest.main()