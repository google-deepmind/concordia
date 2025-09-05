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

"""Performance and stress tests for negotiation framework."""

import datetime
import time
import unittest
from unittest import mock

from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.language_model import language_model
from concordia.prefabs.entity.negotiation import base_negotiator
from concordia.prefabs.entity.negotiation import advanced_negotiator
from concordia.prefabs.game_master.negotiation import negotiation


class PerformanceTestBase(unittest.TestCase):
  """Base class for performance tests."""

  def setUp(self):
    """Set up performance test environment."""
    self.model = mock.create_autospec(
        language_model.LanguageModel, instance=True
    )
    
    # Fast mock responses for performance testing
    self.model.sample_text.return_value = 'test response'
    
    self.clock = game_clock.FixedIntervalClock()
    self.memory_bank = basic_associative_memory.AssociativeMemoryBank()

  def time_operation(self, operation, description="Operation"):
    """Time an operation and assert it completes within reasonable time."""
    start_time = time.time()
    result = operation()
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"{description} took {elapsed:.3f} seconds")
    
    return result, elapsed


class AgentCreationPerformanceTest(PerformanceTestBase):
  """Test performance of agent creation."""

  def test_base_agent_creation_performance(self):
    """Test performance of creating base negotiation agents."""
    def create_base_agents():
      agents = []
      for i in range(10):
        agent = base_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name=f'Agent{i}',
        )
        agents.append(agent)
      return agents

    agents, elapsed = self.time_operation(
        create_base_agents, 
        "Creating 10 base agents"
    )
    
    self.assertEqual(len(agents), 10)
    self.assertLess(elapsed, 5.0, "Base agent creation should be under 5 seconds")

  def test_advanced_agent_creation_performance(self):
    """Test performance of creating advanced agents with modules."""
    def create_advanced_agents():
      agents = []
      modules = ['cultural_adaptation', 'theory_of_mind']
      for i in range(5):
        agent = advanced_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name=f'AdvancedAgent{i}',
            modules=modules,
        )
        agents.append(agent)
      return agents

    agents, elapsed = self.time_operation(
        create_advanced_agents,
        "Creating 5 advanced agents with modules"
    )
    
    self.assertEqual(len(agents), 5)
    self.assertLess(elapsed, 10.0, "Advanced agent creation should be under 10 seconds")

  def test_massive_agent_creation(self):
    """Test creating large numbers of agents."""
    def create_many_agents():
      agents = []
      for i in range(50):
        agent = base_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name=f'MassAgent{i}',
        )
        agents.append(agent)
      return agents

    agents, elapsed = self.time_operation(
        create_many_agents,
        "Creating 50 base agents"
    )
    
    self.assertEqual(len(agents), 50)
    self.assertLess(elapsed, 30.0, "Mass agent creation should be under 30 seconds")


class GameMasterPerformanceTest(PerformanceTestBase):
  """Test performance of game master creation and operations."""

  def test_gm_creation_performance(self):
    """Test performance of GM creation with different module configurations."""
    # Create test agents first
    agents = []
    for i in range(3):
      agent = base_negotiator.build_agent(
          model=self.model,
          memory_bank=self.memory_bank,
          name=f'TestAgent{i}',
      )
      agents.append(agent)

    def create_basic_gm():
      return negotiation.build_game_master(
          model=self.model,
          memory_bank=self.memory_bank,
          entities=agents,
          name='BasicGM',
      )

    gm, elapsed = self.time_operation(create_basic_gm, "Creating basic GM")
    self.assertIsNotNone(gm)
    self.assertLess(elapsed, 3.0, "Basic GM creation should be under 3 seconds")

  def test_gm_with_all_modules_performance(self):
    """Test GM creation with all available modules."""
    agents = []
    for i in range(3):
      agent = advanced_negotiator.build_agent(
          model=self.model,
          memory_bank=self.memory_bank,
          name=f'FullAgent{i}',
          modules=['theory_of_mind', 'cultural_adaptation'],
      )
      agents.append(agent)

    def create_full_gm():
      return negotiation.build_game_master(
          model=self.model,
          memory_bank=self.memory_bank,
          entities=agents,
          name='FullGM',
          gm_modules=[
              'social_intelligence',
              'temporal_dynamics',
              'cultural_awareness',
              'uncertainty_management',
              'collective_intelligence',
              'strategy_evolution',
          ],
      )

    gm, elapsed = self.time_operation(
        create_full_gm, 
        "Creating GM with all modules"
    )
    self.assertIsNotNone(gm)
    self.assertLess(elapsed, 10.0, "Full GM creation should be under 10 seconds")

  def test_multiple_gm_creation(self):
    """Test creating multiple GMs."""
    agents = []
    for i in range(2):
      agent = base_negotiator.build_agent(
          model=self.model,
          memory_bank=self.memory_bank,
          name=f'MultiAgent{i}',
      )
      agents.append(agent)

    def create_multiple_gms():
      gms = []
      for i in range(10):
        gm = negotiation.build_bilateral_negotiation(
            model=self.model,
            memory_bank=self.memory_bank,
            entities=agents,
            name=f'GM{i}',
        )
        gms.append(gm)
      return gms

    gms, elapsed = self.time_operation(
        create_multiple_gms,
        "Creating 10 bilateral GMs"
    )
    
    self.assertEqual(len(gms), 10)
    self.assertLess(elapsed, 15.0, "Creating 10 GMs should be under 15 seconds")


class ModulePerformanceTest(PerformanceTestBase):
  """Test performance of individual GM modules."""

  def setUp(self):
    """Set up module performance tests."""
    super().setUp()
    
    # Import modules for testing
    from concordia.prefabs.game_master.negotiation.components import gm_social_intelligence
    from concordia.prefabs.game_master.negotiation.components import gm_temporal_dynamics
    from concordia.prefabs.game_master.negotiation.components import gm_cultural_awareness
    from concordia.prefabs.game_master.negotiation.components import negotiation_modules
    
    self.social_module = gm_social_intelligence.SocialIntelligenceGM()
    self.temporal_module = gm_temporal_dynamics.TemporalDynamicsGM()
    self.cultural_module = gm_cultural_awareness.CulturalAwarenessGM()
    
    self.context = negotiation_modules.ModuleContext(
        negotiation_id='perf_test',
        participants=['Alice', 'Bob', 'Charlie'],
        current_phase='bargaining',
        current_round=50,
        active_modules={
            'Alice': {'theory_of_mind'},
            'Bob': {'cultural_adaptation'},
            'Charlie': {'temporal_strategy'}
        },
        shared_data={}
    )

  def test_module_update_performance(self):
    """Test performance of module state updates."""
    events = [
        'I propose a collaborative approach',
        'I disagree with that assessment',
        'Let me consider your cultural perspective',
        'Time is running out for this deal',
        'I commit to this long-term partnership',
    ] * 20  # 100 events total

    def update_all_modules():
      for i, event in enumerate(events):
        actor = f'Agent{i % 3}'
        context = self.context
        context.current_round = i
        
        self.social_module.update_state(event, actor, context)
        self.temporal_module.update_state(event, actor, context)
        self.cultural_module.update_state(event, actor, context)

    _, elapsed = self.time_operation(
        update_all_modules,
        "Processing 100 events through 3 modules"
    )
    
    self.assertLess(elapsed, 5.0, "Module updates should be under 5 seconds")

  def test_module_validation_performance(self):
    """Test performance of action validation."""
    actions = [
        'I propose we work together',
        'That is completely unacceptable',
        'Let me understand your position',
        'I demand immediate agreement',
        'We should respect cultural differences',
    ] * 50  # 250 validations total

    def validate_all_actions():
      results = []
      for i, action in enumerate(actions):
        actor = f'Agent{i % 3}'
        
        # Validate with each module
        results.append(self.social_module.validate_action(actor, action, self.context))
        results.append(self.temporal_module.validate_action(actor, action, self.context))
        results.append(self.cultural_module.validate_action(actor, action, self.context))
      return results

    results, elapsed = self.time_operation(
        validate_all_actions,
        "Validating 750 actions (250 x 3 modules)"
    )
    
    self.assertEqual(len(results), 750)
    self.assertLess(elapsed, 3.0, "Action validation should be under 3 seconds")

  def test_observation_generation_performance(self):
    """Test performance of observation context generation."""
    def generate_observations():
      observations = []
      for i in range(100):
        observer = f'Observer{i % 3}'
        
        obs1 = self.social_module.get_observation_context(observer, self.context)
        obs2 = self.temporal_module.get_observation_context(observer, self.context)
        obs3 = self.cultural_module.get_observation_context(observer, self.context)
        
        observations.extend([obs1, obs2, obs3])
      return observations

    observations, elapsed = self.time_operation(
        generate_observations,
        "Generating 300 observation contexts"
    )
    
    self.assertEqual(len(observations), 300)
    self.assertLess(elapsed, 2.0, "Observation generation should be under 2 seconds")


class StressTest(PerformanceTestBase):
  """Stress tests for the negotiation framework."""

  def test_large_scale_negotiation(self):
    """Test negotiation with many participants."""
    # Create many agents
    num_agents = 10
    agents = []
    
    def create_many_agents():
      for i in range(num_agents):
        agent = base_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name=f'LargeAgent{i}',
            reservation_value=100.0 + i * 10,
        )
        agents.append(agent)

    _, creation_time = self.time_operation(
        create_many_agents,
        f"Creating {num_agents} agents"
    )

    def create_large_gm():
      return negotiation.build_multilateral_negotiation(
          model=self.model,
          memory_bank=self.memory_bank,
          entities=agents,
          name='Large Scale Negotiation',
      )

    gm, gm_creation_time = self.time_operation(
        create_large_gm,
        f"Creating GM for {num_agents} agents"
    )

    self.assertIsNotNone(gm)
    self.assertEqual(len(agents), num_agents)
    
    # Total setup should be reasonable even for large negotiations
    total_time = creation_time + gm_creation_time
    self.assertLess(total_time, 30.0, "Large scale setup should be under 30 seconds")

  def test_long_running_negotiation_simulation(self):
    """Test simulation of long negotiations."""
    agents = []
    for i in range(3):
      agent = base_negotiator.build_agent(
          model=self.model,
          memory_bank=self.memory_bank,
          name=f'LongAgent{i}',
      )
      agents.append(agent)

    gm = negotiation.build_game_master(
        model=self.model,
        memory_bank=self.memory_bank,
        entities=agents,
        name='Long Running Test',
        max_rounds=100,
        gm_modules=['social_intelligence', 'temporal_dynamics'],
    )

    # Get modules for direct testing
    social_module = gm._context_components['gm_module_social_intelligence']
    temporal_module = gm._context_components['gm_module_temporal_dynamics']

    def simulate_long_negotiation():
      from concordia.prefabs.game_master.negotiation.components import negotiation_modules
      
      context = negotiation_modules.ModuleContext(
          negotiation_id='long_test',
          participants=[agent.name for agent in agents],
          current_phase='bargaining',
          current_round=0,
          active_modules={agent.name: {'basic'} for agent in agents},
          shared_data={}
      )

      events = [
          'I propose a starting price',
          'That seems reasonable',
          'Let me counter with a different offer',
          'I need to think about this',
          'Time is a concern for me',
      ]

      # Simulate 500 rounds
      for round_num in range(500):
        context.current_round = round_num
        event = events[round_num % len(events)]
        actor = agents[round_num % len(agents)].name

        # Update modules
        social_module.update_state(event, actor, context)
        temporal_module.update_state(event, actor, context)

        # Validate actions periodically
        if round_num % 10 == 0:
          social_module.validate_action(actor, event, context)
          temporal_module.validate_action(actor, event, context)

    _, elapsed = self.time_operation(
        simulate_long_negotiation,
        "Simulating 500 rounds of negotiation"
    )

    self.assertLess(elapsed, 20.0, "Long simulation should be under 20 seconds")

  def test_memory_usage_stability(self):
    """Test that memory usage doesn't grow excessively."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Create and destroy many objects
    for iteration in range(10):
      agents = []
      for i in range(5):
        agent = advanced_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name=f'MemAgent{i}_{iteration}',
            modules=['theory_of_mind'],
        )
        agents.append(agent)

      gm = negotiation.build_game_master(
          model=self.model,
          memory_bank=self.memory_bank,
          entities=agents,
          name=f'MemoryTestGM_{iteration}',
          gm_modules=['social_intelligence'],
      )

      # Use the GM briefly
      social_module = gm._context_components['gm_module_social_intelligence']
      for i in range(10):
        social_module.update_state(f'Test event {i}', 'TestActor', mock.Mock(
            current_round=i, participants=['TestActor'], current_phase='test'
        ))

      # Clear references
      del agents
      del gm
      del social_module

    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory
    memory_growth_mb = memory_growth / (1024 * 1024)

    print(f"Memory growth: {memory_growth_mb:.1f} MB")
    
    # Allow some memory growth but not excessive
    self.assertLess(memory_growth_mb, 100, "Memory growth should be under 100 MB")

  def test_concurrent_negotiations(self):
    """Test handling multiple concurrent negotiations."""
    negotiations = []

    def create_concurrent_negotiations():
      for i in range(5):
        agents = []
        for j in range(2):
          agent = base_negotiator.build_agent(
              model=self.model,
              memory_bank=self.memory_bank,
              name=f'ConcurrentAgent{i}_{j}',
          )
          agents.append(agent)

        gm = negotiation.build_bilateral_negotiation(
            model=self.model,
            memory_bank=self.memory_bank,
            entities=agents,
            name=f'ConcurrentNeg{i}',
        )
        negotiations.append((agents, gm))

    _, elapsed = self.time_operation(
        create_concurrent_negotiations,
        "Creating 5 concurrent negotiations"
    )

    self.assertEqual(len(negotiations), 5)
    self.assertLess(elapsed, 10.0, "Concurrent negotiation setup should be under 10 seconds")

    # Verify all negotiations are independent
    for i, (agents, gm) in enumerate(negotiations):
      self.assertEqual(len(agents), 2)
      self.assertIsNotNone(gm)
      self.assertEqual(gm._agent_name, f'ConcurrentNeg{i}')


class BenchmarkTest(PerformanceTestBase):
  """Benchmark tests to establish performance baselines."""

  def test_baseline_agent_throughput(self):
    """Establish baseline for agent creation throughput."""
    num_agents = 100
    start_time = time.time()

    agents = []
    for i in range(num_agents):
      agent = base_negotiator.build_agent(
          model=self.model,
          memory_bank=self.memory_bank,
          name=f'BenchAgent{i}',
      )
      agents.append(agent)

    end_time = time.time()
    elapsed = end_time - start_time
    throughput = num_agents / elapsed

    print(f"Agent creation throughput: {throughput:.1f} agents/second")
    self.assertEqual(len(agents), num_agents)
    self.assertGreater(throughput, 5, "Should create at least 5 agents per second")

  def test_baseline_module_throughput(self):
    """Establish baseline for module operation throughput."""
    from concordia.prefabs.game_master.negotiation.components import gm_social_intelligence
    from concordia.prefabs.game_master.negotiation.components import negotiation_modules
    
    module = gm_social_intelligence.SocialIntelligenceGM()
    context = negotiation_modules.ModuleContext(
        negotiation_id='benchmark',
        participants=['Alice', 'Bob'],
        current_phase='bargaining',
        current_round=0,
        active_modules={'Alice': {'theory_of_mind'}, 'Bob': {'theory_of_mind'}},
        shared_data={}
    )

    num_operations = 1000
    start_time = time.time()

    for i in range(num_operations):
      context.current_round = i
      module.update_state(f'Test event {i}', 'Alice', context)

    end_time = time.time()
    elapsed = end_time - start_time
    throughput = num_operations / elapsed

    print(f"Module operation throughput: {throughput:.1f} operations/second")
    self.assertGreater(throughput, 100, "Should process at least 100 operations per second")


if __name__ == '__main__':
  # Run with timing information
  unittest.main(verbosity=2)