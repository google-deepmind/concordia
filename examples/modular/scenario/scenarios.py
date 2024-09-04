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
import types

from examples.modular import environment as environment_lib
from examples.modular.environment import supporting_agent_factory
from examples.modular.utils import logging_types as logging_lib
from concordia.factory import agent as agent_lib
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
import immutabledict
import numpy as np

Runnable = Callable[[], tuple[logging_lib.SimulationOutcome, str]]

DEFAULT_IMPORT_ENV_BASE_MODULE = environment_lib.__name__
DEFAULT_IMPORT_AGENT_BASE_MODULE = agent_lib.__name__
DEFAULT_IMPORT_SUPPORT_AGENT_MODULE = supporting_agent_factory.__name__


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
    labor_collective_action__paranoid_boss=SubstrateConfig(
        description='labor organization collective action with a paranoid boss',
        environment='labor_collective_action',
        supporting_agent_module='paranoid_agent',
    ),
    pub_coordination=SubstrateConfig(
        description=(
            'pub attendance coordination with supporting agent being stubborn'
            ' and always choosing their preference.'
        ),
        environment='pub_coordination',
        supporting_agent_module='basic_puppet_agent',
    ),
    pub_coordination_closures=SubstrateConfig(
        description=(
            'pub attendance coordination with one pub sometimes being closed'
        ),
        environment='pub_coordination_closures',
        supporting_agent_module='basic_puppet_agent',
    ),
    pub_coordination_friendships=SubstrateConfig(
        description='pub attendance coordination with friendship network',
        environment='pub_coordination_friendships',
        supporting_agent_module='basic_puppet_agent',
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
    labor_collective_action__paranoid_boss_0=ScenarioConfig(
        description=(
            'resident population of focal agents in a labor '
            'organization collective action scenario with a boss '
            'who is paranoid and a visitor agent who is basic'
        ),
        substrate_config=SUBSTRATE_CONFIGS[
            'labor_collective_action__paranoid_boss'
        ],
        background_agent_module='basic_agent',
        time_and_place_module='wild_west_railroad_construction_labor',
        focal_is_resident=True,
        tags=(
            'discouraging antisocial behavior',
            'role playing',
        ),
    ),
    labor_collective_action__paranoid_boss_1=ScenarioConfig(
        description=(
            'visitor focal agent in a labor organization collective '
            'action scenario with a boss who is paranoid '
            'and a resident population of basic agents'
        ),
        substrate_config=SUBSTRATE_CONFIGS[
            'labor_collective_action__paranoid_boss'
        ],
        background_agent_module='basic_agent',
        time_and_place_module='wild_west_railroad_construction_labor',
        focal_is_resident=False,
        tags=(
            'convention following',
            'leadership',
            'persuasion',
            'role playing',
        ),
    ),
    pub_coordination_0=ScenarioConfig(
        description=(
            'resident population of focal agents in a pub coordination scenario'
            ' with a supporting agent who is stubborn and a'
            ' rational visitor agent'
        ),
        substrate_config=SUBSTRATE_CONFIGS['pub_coordination'],
        background_agent_module='rational_agent',
        time_and_place_module='pub_coordination_london',
        focal_is_resident=True,
        tags=('coordination', 'persuasion'),
    ),
    pub_coordination_1=ScenarioConfig(
        description=(
            'visitor focal agent in a pub coordination '
            'scenario with a supporting agent who is a stubborn '
            'and a visitor who is rational'
        ),
        substrate_config=SUBSTRATE_CONFIGS['pub_coordination'],
        background_agent_module='rational_agent',
        time_and_place_module='pub_coordination_london',
        focal_is_resident=False,
        tags=('coordination', 'persuasion'),
    ),
    pub_coordination_closures_0=ScenarioConfig(
        description=(
            'resident population of focal agents in a pub coordination with'
            ' pub closures and with a supporting agent who is stubborn and a'
            ' rational visitor agent'
        ),
        substrate_config=SUBSTRATE_CONFIGS['pub_coordination_closures'],
        background_agent_module='rational_agent',
        time_and_place_module='pub_coordination_london',
        focal_is_resident=True,
        tags=('coordination', 'persuasion', 'private information'),
    ),
    pub_coordination_closures_1=ScenarioConfig(
        description=(
            'visitor focal agent in a pub coordination scenario with closures'
            'and with a supporting agent who is stubborn and a'
            ' rational visitor agent'
        ),
        substrate_config=SUBSTRATE_CONFIGS['pub_coordination_closures'],
        background_agent_module='rational_agent',
        time_and_place_module='pub_coordination_london',
        focal_is_resident=False,
        tags=('coordination', 'persuasion', 'private information'),
    ),
    pub_coordination_friendships_0=ScenarioConfig(
        description=(
            'resident population of focal agents in a pub coordination scenario'
            ' with friendships with a supporting agent who is stubborn and a'
            ' rational visitor agent'
        ),
        substrate_config=SUBSTRATE_CONFIGS['pub_coordination_friendships'],
        background_agent_module='rational_agent',
        time_and_place_module='pub_coordination_london',
        focal_is_resident=True,
        tags=('coordination', 'persuasion', 'social networks'),
    ),
    pub_coordination_friendships_1=ScenarioConfig(
        description=(
            'visitor focal agent in a pub coordination with friendships'
            'scenario with a supporting agent who is stubborn '
            'and a rational visitor agent'
        ),
        substrate_config=SUBSTRATE_CONFIGS['pub_coordination_friendships'],
        background_agent_module='rational_agent',
        time_and_place_module='pub_coordination_london',
        focal_is_resident=False,
        tags=('coordination', 'persuasion', 'social networks'),
    ),
)


def build_simulation(
    scenario_config: ScenarioConfig,
    model: language_model.LanguageModel,
    focal_agent_module: types.ModuleType,
    embedder: Callable[[str], np.ndarray],
    measurements: measurements_lib.Measurements,
    agent_base_module: str = DEFAULT_IMPORT_AGENT_BASE_MODULE,
    support_agent_base_module: str = DEFAULT_IMPORT_SUPPORT_AGENT_MODULE,
    env_base_module: str = DEFAULT_IMPORT_ENV_BASE_MODULE,
) -> Runnable:
  """Builds a simulation from a scenario configuration."""
  substrate_config = scenario_config.substrate_config
  # Load the environment config with importlib
  simulation = importlib.import_module(
      f'{env_base_module}.{substrate_config.environment}'
  )
  background_agent_module = importlib.import_module(
      f'{agent_base_module}.{scenario_config.background_agent_module}'
  )
  if scenario_config.focal_is_resident:
    resident_agent_module = focal_agent_module
    visitor_agent_module = background_agent_module
  else:
    visitor_agent_module = focal_agent_module
    resident_agent_module = background_agent_module

  supporting_agent_module = importlib.import_module(
      f'{support_agent_base_module}.{substrate_config.supporting_agent_module}'
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
