# Copyright 2024 DeepMind Technologies Limited.
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

"""Tests for emotional_state component."""

import datetime
import unittest
from unittest import mock

from concordia.components.agent import memory as memory_component
from concordia.contrib.components.agent import emotional_state
from concordia.language_model import language_model
from concordia.testing import mock_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class EmotionalStateTest(unittest.TestCase):
  """Tests for the EmotionalState component."""

  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    self.model = mock_model.MockModel()
    self.clock = mock.Mock(
        return_value=datetime.datetime(2024, 1, 1, 12, 0, 0)
    )

  def test_initialization(self):
    """Tests that EmotionalState can be initialized."""
    component = emotional_state.EmotionalState(
        model=self.model,
        clock_now=self.clock,
    )
    self.assertIsNotNone(component)
    self.assertEqual(component.get_pre_act_label(), 'Emotional state')

  def test_emotional_state_with_no_memories(self):
    """Tests emotional state when agent has no memories."""
    # Create a mock entity with a memory component
    mock_entity = mock.Mock(spec=entity_component.EntityWithComponents)
    mock_entity.name = 'Alice'
    
    mock_memory = mock.Mock(spec=memory_component.Memory)
    mock_memory.retrieve_recent.return_value = []
    mock_entity.get_component.return_value = mock_memory
    
    component = emotional_state.EmotionalState(
        model=self.model,
        clock_now=self.clock,
        add_to_memory=False,
    )
    component.set_entity(mock_entity)
    
    result = component._make_pre_act_value()
    self.assertIn('Alice', result)
    self.assertIn('no clear emotional state', result.lower())

  def test_emotional_state_with_memories(self):
    """Tests emotional state analysis with memories."""
    # Create a mock entity
    mock_entity = mock.Mock(spec=entity_component.EntityWithComponents)
    mock_entity.name = 'Bob'
    
    # Create mock memory with some sample memories
    mock_memory = mock.Mock(spec=memory_component.Memory)
    mock_memory.retrieve_recent.return_value = [
        'Bob won the lottery today.',
        'Bob celebrated with friends.',
        'Bob felt very excited about the future.',
    ]
    mock_entity.get_component.return_value = mock_memory
    
    # Set up the mock model to return reasonable responses
    self.model.sample_text_responses = [
        'Bob is feeling happy because he won the lottery and celebrated with friends.'
    ]
    
    component = emotional_state.EmotionalState(
        model=self.model,
        clock_now=self.clock,
        include_intensity=False,
        add_to_memory=False,
    )
    component.set_entity(mock_entity)
    
    result = component._make_pre_act_value()
    self.assertIn('Bob', result)
    # The result should mention an emotional state
    self.assertTrue(len(result) > 0)

  def test_get_current_emotion(self):
    """Tests getting the current emotion."""
    component = emotional_state.EmotionalState(
        model=self.model,
        clock_now=self.clock,
    )
    
    # Initially should be empty
    self.assertEqual(component.get_current_emotion(), '')
    
    # After calling _make_pre_act_value, it should be set
    mock_entity = mock.Mock(spec=entity_component.EntityWithComponents)
    mock_entity.name = 'Charlie'
    mock_memory = mock.Mock(spec=memory_component.Memory)
    mock_memory.retrieve_recent.return_value = ['Charlie had a good day.']
    mock_entity.get_component.return_value = mock_memory
    
    component.set_entity(mock_entity)
    component._make_pre_act_value()
    
    # Should now have a value
    self.assertNotEqual(component.get_current_emotion(), '')

  def test_state_persistence(self):
    """Tests that state can be saved and restored."""
    component = emotional_state.EmotionalState(
        model=self.model,
        clock_now=self.clock,
    )
    
    # Set a mock emotional state
    test_state = 'Test emotional state'
    component.set_state({'current_emotional_state': test_state})
    
    # Retrieve the state
    state = component.get_state()
    self.assertEqual(state['current_emotional_state'], test_state)


class EmotionalAppraisalTest(unittest.TestCase):
  """Tests for the EmotionalAppraisal component."""

  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    self.model = mock_model.MockModel()

  def test_initialization(self):
    """Tests that EmotionalAppraisal can be initialized."""
    component = emotional_state.EmotionalAppraisal(
        model=self.model,
    )
    self.assertIsNotNone(component)
    self.assertEqual(component.get_pre_act_label(), 'Emotional appraisal')

  def test_appraisal_with_no_memories(self):
    """Tests appraisal when agent has no memories."""
    mock_entity = mock.Mock(spec=entity_component.EntityWithComponents)
    mock_entity.name = 'Diana'
    
    mock_memory = mock.Mock(spec=memory_component.Memory)
    mock_memory.retrieve_recent.return_value = []
    
    # Mock the get_component method to return appropriate mocks
    def get_component_side_effect(name, type_=None):
      if type_ == emotional_state.EmotionalState:
        raise ValueError('Component not found')
      elif type_ == memory_component.Memory:
        return mock_memory
      return mock.Mock()
    
    mock_entity.get_component.side_effect = get_component_side_effect
    
    component = emotional_state.EmotionalAppraisal(
        model=self.model,
    )
    component.set_entity(mock_entity)
    
    result = component._make_pre_act_value()
    self.assertIn('Diana', result)
    self.assertIn('insufficient context', result.lower())

  def test_appraisal_with_emotional_state(self):
    """Tests appraisal with emotional state component present."""
    mock_entity = mock.Mock(spec=entity_component.EntityWithComponents)
    mock_entity.name = 'Eve'
    
    # Create mock emotional state component
    mock_emotional_state = mock.Mock(spec=emotional_state.EmotionalState)
    mock_emotional_state.get_current_emotion.return_value = (
        'Eve is feeling happy.'
    )
    
    # Create mock memory
    mock_memory = mock.Mock(spec=memory_component.Memory)
    mock_memory.retrieve_recent.return_value = [
        'Eve just received good news.',
        'Eve is meeting a friend.',
    ]
    
    # Mock get_component to return appropriate components
    def get_component_side_effect(name, type_=None):
      if type_ == emotional_state.EmotionalState:
        return mock_emotional_state
      elif type_ == memory_component.Memory:
        return mock_memory
      return mock.Mock()
    
    mock_entity.get_component.side_effect = get_component_side_effect
    
    self.model.sample_text_responses = [
        "Eve's next action might reinforce her positive mood if she "
        "continues to engage positively with her friend."
    ]
    
    component = emotional_state.EmotionalAppraisal(
        model=self.model,
    )
    component.set_entity(mock_entity)
    
    result = component._make_pre_act_value()
    self.assertIn('Eve', result)
    self.assertIn('appraisal', result.lower())


if __name__ == '__main__':
  unittest.main()
