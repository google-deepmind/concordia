# Copyright 2026 DeepMind Technologies Limited.
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

"""Smoke tests for resource dilemma CPR scenarios.

These tests run each scenario with a mock language model and dummy
embedder to verify that the simulation builds and runs to completion
without errors. No LLM is required.
"""

from absl.testing import absltest
from absl.testing import parameterized
from examples.resource_dilemma import run
from examples.resource_dilemma.personas import fishery_personas
from examples.resource_dilemma.personas import irrigation_personas
from examples.resource_dilemma.personas import network_personas
from examples.resource_dilemma.personas import pasture_personas
from examples.resource_dilemma.scenarios import fishery
from examples.resource_dilemma.scenarios import irrigation
from examples.resource_dilemma.scenarios import network
from examples.resource_dilemma.scenarios import pasture
from concordia.typing import prefab as prefab_lib
import numpy as np


def _mock_embedder(text: str) -> np.ndarray:
  del text
  return np.ones(384)


# Each scenario module, its build_config kwargs, and a human-readable name.
_SCENARIOS = [
    ('pasture', pasture, dict(
        player_configs=pasture_personas.HERDERS,
        leader_configs=pasture_personas.LEADERS,
    )),
    ('irrigation', irrigation, dict(
        player_configs=irrigation_personas.IRRIGATORS,
        leader_configs=irrigation_personas.LEADERS,
    )),
    ('network', network, dict(
        player_configs=network_personas.USERS,
        leader_configs=network_personas.LEADERS,
    )),
    ('fishery', fishery, dict(
        player_configs=fishery_personas.FISHERS,
        leader_configs=fishery_personas.LEADERS,
    )),
]


class ResourceDilemmaTest(parameterized.TestCase):
  """Smoke tests for resource dilemma scenarios."""

  def _assert_harvest_completed(self, config, num_cycles):
    harvest_instances = [
        instance for instance in config.instances
        if instance.prefab in ('HarvestingGameMaster', 'ResourceHarvestGameMaster')
    ]
    self.assertLen(harvest_instances, 1)
    state = harvest_instances[0].params['sim_state']
    participant_count = sum(
        instance.role == prefab_lib.Role.ENTITY for instance in config.instances
    )
    self.assertGreater(participant_count, 0)
    self.assertEqual(state.cycle_harvest_total, participant_count)
    self.assertTrue(state.terminated)
    self.assertEqual(state.current_cycle, num_cycles)
    logger = harvest_instances[0].params['logger_state']
    self.assertEqual(
        sum(logger.cumulative_harvests.values()), participant_count * num_cycles
    )
    summaries = [row for row in logger.step_logs if row['phase'] == 'summary']
    self.assertLen(summaries, num_cycles)

  @parameterized.named_parameters(
      dict(
          testcase_name=f'{name}_{num_cycles}_cycles',
          scenario_module=mod,
          config_kwargs=kwargs,
          num_cycles=num_cycles,
      )
      for name, mod, kwargs in _SCENARIOS
      for num_cycles in (1, 2)
  )
  def test_standard_mode_runs_to_completion(
      self, scenario_module, config_kwargs, num_cycles
  ):
    """Verifies standard mode runs without errors."""
    model = run.MockHarvestModel()
    config = scenario_module.build_config(
        **config_kwargs,
        num_cycles=num_cycles,
        mode='standard',
        embedder=_mock_embedder,
    )
    result = scenario_module.run_simulation(
        config=config,
        model=model,
        embedder=_mock_embedder,
        num_cycles=num_cycles,
    )
    self.assertIsNotNone(result)
    self._assert_harvest_completed(config, num_cycles)

  @parameterized.named_parameters(
      dict(
          testcase_name=f'{name}_{num_cycles}_cycles',
          scenario_module=mod,
          config_kwargs=kwargs,
          num_cycles=num_cycles,
      )
      for name, mod, kwargs in _SCENARIOS
      for num_cycles in (1, 2)
  )
  def test_election_mode_runs_to_completion(
      self, scenario_module, config_kwargs, num_cycles
  ):
    """Verifies election mode runs without errors."""
    model = run.MockHarvestModel()
    config = scenario_module.build_config(
        **config_kwargs,
        num_cycles=num_cycles,
        mode='election',
        election_every_n=1,
        embedder=_mock_embedder,
    )
    result = scenario_module.run_simulation(
        config=config,
        model=model,
        embedder=_mock_embedder,
        num_cycles=num_cycles,
    )
    self.assertIsNotNone(result)
    self._assert_harvest_completed(config, num_cycles)


if __name__ == '__main__':
  absltest.main()

