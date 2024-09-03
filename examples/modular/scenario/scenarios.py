# Copyright 2023 DeepMind Technologies Limited.
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

"""Define specific scenarios for the Concordia Challenge."""

from collections.abc import Callable, Collection, Mapping
import dataclasses
import importlib
import pathlib
import sys
import types

from examples.modular.utils import logging_types as logging_lib
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
import immutabledict
import numpy as np

concordia_root_dir = pathlib.Path(
    __file__
).parent.parent.parent.parent.parent.resolve()
print('concordia root dir: ', concordia_root_dir)
sys.path.append(f'{concordia_root_dir}')

Runnable = Callable[[], tuple[logging_lib.SimulationOutcome, str]]

IMPORT_ENV_BASE_DIR = 'environment'
IMPORT_AGENT_BASE_DIR = 'concordia.factory.agent'
IMPORT_SUPPORT_AGENT_DIR = (
    'examples.modular.environment.supporting_agent_factory')


@dataclasses.dataclass(frozen=True)
class SubstrateConfig:
  """Class for configuring a substrate."""

  description: str
  environment: str
  supporting_agent_module: str | None


@dataclasses.dataclass(frozen=True)
class ScenarioConfig:
  """Class for configuring a scenario."""

  description: str
  substrate_config: SubstrateConfig
  background_agent_module: str
  time_and_place_module: str | None
  focal_is_resident: bool
  tags: Collection[str]


SUBSTRATE_CONFIGS: Mapping[str, SubstrateConfig] = immutabledict.immutabledict(
    # keep-sorted start numeric=yes block=yes
    labor_collective_action__rational_boss=SubstrateConfig(
        description='labor organization collective action with a rational boss',
        environment='labor_collective_action',
        supporting_agent_module='rational_agent',
    ),
)

SCENARIO_CONFIGS: Mapping[str, ScenarioConfig] = immutabledict.immutabledict(
    # keep-sorted start numeric=yes block=yes
    labor_collective_action__rational_boss_0=ScenarioConfig(
        description=(
            'resident population of focal agents in a labor '
            'organization collective action scenario with a boss '
            'who is rational and a visitor agent '
            'who is rational'
        ),
        substrate_config=SUBSTRATE_CONFIGS[
            'labor_collective_action__rational_boss'
        ],
        background_agent_module='rational_agent',
        time_and_place_module='wild_west_railroad_construction_labor',
        focal_is_resident=True,
        tags=('discouraging antisocial behavior',),
    ),
    labor_collective_action__rational_boss_1=ScenarioConfig(
        description=(
            'visitor focal agent in a labor organization collective '
            'action scenario with a boss who is rational '
            'and a resident population of rational agents'
        ),
        substrate_config=SUBSTRATE_CONFIGS[
            'labor_collective_action__rational_boss'
        ],
        background_agent_module='rational_agent',
        time_and_place_module='wild_west_railroad_construction_labor',
        focal_is_resident=False,
        tags=(
            'convention following',
            'leadership',
            'persuasion',
        ),
    ),
)


def build_simulation(
    scenario_config: ScenarioConfig,
    model: language_model.LanguageModel,
    focal_agent_module: types.ModuleType,
    embedder: Callable[[str], np.ndarray],
    measurements: measurements_lib.Measurements,
) -> Runnable:
  """Builds a simulation from a scenario configuration."""
  substrate_config = scenario_config.substrate_config
  # Load the environment config with importlib
  root_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
  sys.path.append(f'{root_dir}')
  simulation = importlib.import_module(
      f'{IMPORT_ENV_BASE_DIR}.{substrate_config.environment}'
  )
  background_agent_module = importlib.import_module(
      f'{IMPORT_AGENT_BASE_DIR}.{scenario_config.background_agent_module}'
  )
  if scenario_config.focal_is_resident:
    resident_agent_module = focal_agent_module
    visitor_agent_module = background_agent_module
  else:
    visitor_agent_module = focal_agent_module
    resident_agent_module = background_agent_module

  supporting_agent_module = importlib.import_module(
      f'{IMPORT_SUPPORT_AGENT_DIR}.{substrate_config.supporting_agent_module}'
  )
  runnable_simulation = simulation.Simulation(
      model=model,
      embedder=embedder,
      measurements=measurements,
      resident_visitor_modules=(resident_agent_module, visitor_agent_module),
      supporting_agent_module=supporting_agent_module,
      time_and_place_module=scenario_config.time_and_place_module,
  )
  return runnable_simulation
