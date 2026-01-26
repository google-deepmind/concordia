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

"""Test for the negotiation game master prefab."""

import datetime
import unittest
from unittest import mock

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.language_model import language_model
from concordia.prefabs.game_master.negotiation import negotiation
from concordia.prefabs.entity.negotiation import base_negotiator


class NegotiationGameMasterTest(unittest.TestCase):
  """Tests for the negotiation game master."""

  def setUp(self):
    """Set up test dependencies."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    self.clock = game_clock.FixedIntervalClock()
    self.memory_bank = basic_associative_memory.AssociativeMemoryBank()

  def test_basic_instantiation(self):
    """Test that we can create a negotiation game master."""
    # Create mock entities
    buyer = mock.create_autospec(
        entity_agent_with_logging.EntityAgentWithLogging,
        instance=True,
    )
    buyer.name = 'Buyer'

    seller = mock.create_autospec(
        entity_agent_with_logging.EntityAgentWithLogging,
        instance=True,
    )
    seller.name = 'Seller'

    # Create game master
    gm_prefab = negotiation.NegotiationGameMaster(
        params={
            'name': 'Test Negotiation GM',
            'negotiation_type': 'price',
            'protocol': 'alternating',
        },
        entities=[buyer, seller],
    )

    # Build the game master
    gm = gm_prefab.build(self.model, self.memory_bank)

    # Verify it's created correctly
    self.assertIsInstance(gm, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertEqual(gm._agent_name, 'Test Negotiation GM')

    # Check that key components exist
    self.assertIn('negotiation_state', gm._context_components)
    self.assertIn('negotiation_validator', gm._context_components)

  def test_negotiation_state_tracking(self):
    """Test that negotiation state is properly tracked."""
    # Create simple mock agents
    buyer = mock.create_autospec(
        entity_agent_with_logging.EntityAgentWithLogging,
        instance=True,
    )
    buyer.name = 'Alice'

    seller = mock.create_autospec(
        entity_agent_with_logging.EntityAgentWithLogging,
        instance=True,
    )
    seller.name = 'Bob'

    # Create game master
    gm_prefab = negotiation.NegotiationGameMaster(
        params={
            'name': 'State Test GM',
            'negotiation_type': 'price',
            'max_rounds': 5,
        },
        entities=[buyer, seller],
    )

    gm = gm_prefab.build(self.model, self.memory_bank)

    # Get state tracker component
    state_tracker = gm._context_components['negotiation_state']

    # Start a negotiation
    state = state_tracker.start_negotiation(
        negotiation_id='test_neg',
        participants=['Alice', 'Bob'],
    )

    # Verify initial state
    self.assertEqual(state.phase, 'opening')
    self.assertEqual(state.current_round, 0)
    self.assertEqual(len(state.participants), 2)

    # Record an offer
    offer = state_tracker.record_offer(
        negotiation_id='test_neg',
        offerer='Alice',
        recipient='Bob',
        offer_type='initial',
        terms={'price': 100},
    )

    # Verify offer was recorded
    self.assertEqual(len(state.offers_history), 1)
    self.assertEqual(state.active_offer, offer)

  def test_negotiation_validation(self):
    """Test that offers are properly validated."""
    # Create mock agents
    buyer = mock.create_autospec(
        entity_agent_with_logging.EntityAgentWithLogging,
        instance=True,
    )
    buyer.name = 'Buyer'

    seller = mock.create_autospec(
        entity_agent_with_logging.EntityAgentWithLogging,
        instance=True,
    )
    seller.name = 'Seller'

    # Create game master with validation
    gm_prefab = negotiation.NegotiationGameMaster(
        params={
            'name': 'Validation Test GM',
            'negotiation_type': 'price',
            'enable_batna_validation': True,
        },
        entities=[buyer, seller],
    )

    gm = gm_prefab.build(self.model, self.memory_bank)

    # Get validator component
    validator = gm._context_components['negotiation_validator']

    # Set BATNAs
    validator.set_batna('Buyer', {'price': 150, 'role': 'buyer'})
    validator.set_batna('Seller', {'price': 100, 'role': 'seller'})

    # Test valid offer
    is_valid, errors = validator.validate_offer(
        'Buyer',
        {'price': 130},  # Within buyer's BATNA
    )
    self.assertTrue(is_valid)
    self.assertEqual(len(errors), 0)

    # Test invalid offer (exceeds BATNA)
    is_valid, errors = validator.validate_offer(
        'Buyer',
        {'price': 200},  # Exceeds buyer's BATNA
    )
    self.assertFalse(is_valid)
    self.assertTrue(any('BATNA' in error for error in errors))

  def test_different_protocols(self):
    """Test different negotiation protocols."""
    # Create mock agents
    agents = []
    for i in range(3):
      agent = mock.create_autospec(
          entity_agent_with_logging.EntityAgentWithLogging,
          instance=True,
      )
      agent.name = f'Agent{i}'
      agents.append(agent)

    # Test alternating protocol
    gm_alternating = negotiation.NegotiationGameMaster(
        params={
            'name': 'Alternating GM',
            'protocol': 'alternating',
        },
        entities=agents[:2],
    ).build(self.model, self.memory_bank)

    # Check that NextActingInFixedOrder is used
    next_actor = gm_alternating._context_components['acting']
    self.assertEqual(next_actor.__class__.__name__, 'NextActingInFixedOrder')

    # Test simultaneous protocol
    gm_simultaneous = negotiation.NegotiationGameMaster(
        params={
            'name': 'Simultaneous GM',
            'protocol': 'simultaneous',
        },
        entities=agents,
    ).build(self.model, self.memory_bank)

    # Check that standard NextActing is used
    next_actor = gm_simultaneous._context_components['acting']
    self.assertEqual(next_actor.__class__.__name__, 'NextActing')


if __name__ == '__main__':
  unittest.main()
