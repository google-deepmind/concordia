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
from examples.resource_dilemma.personas import fishery_personas
from examples.resource_dilemma.personas import irrigation_personas
from examples.resource_dilemma.personas import network_personas
from examples.resource_dilemma.personas import pasture_personas
from examples.resource_dilemma.scenarios import fishery
from examples.resource_dilemma.scenarios import irrigation
from examples.resource_dilemma.scenarios import network
from examples.resource_dilemma.scenarios import pasture
from concordia.language_model import no_language_model
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

  @parameterized.named_parameters(
      dict(
          testcase_name=name,
          scenario_module=mod,
          config_kwargs=kwargs,
      )
      for name, mod, kwargs in _SCENARIOS
  )
  def test_standard_mode_runs_to_completion(
      self, scenario_module, config_kwargs
  ):
    """Verifies standard mode runs without errors."""
    model = no_language_model.NoLanguageModel()
    config = scenario_module.build_config(
        **config_kwargs,
        num_cycles=1,
        mode='standard',
        embedder=_mock_embedder,
    )
    result = scenario_module.run_simulation(
        config=config,
        model=model,
        embedder=_mock_embedder,
        num_cycles=1,
    )
    self.assertIsNotNone(result)

  @parameterized.named_parameters(
      dict(
          testcase_name=name,
          scenario_module=mod,
          config_kwargs=kwargs,
      )
      for name, mod, kwargs in _SCENARIOS
  )
  def test_election_mode_runs_to_completion(
      self, scenario_module, config_kwargs
  ):
    """Verifies election mode runs without errors."""
    model = no_language_model.NoLanguageModel()
    config = scenario_module.build_config(
        **config_kwargs,
        num_cycles=1,
        mode='election',
        election_every_n=1,
        embedder=_mock_embedder,
    )
    result = scenario_module.run_simulation(
        config=config,
        model=model,
        embedder=_mock_embedder,
        num_cycles=1,
    )
    self.assertIsNotNone(result)


if __name__ == '__main__':
  absltest.main()

