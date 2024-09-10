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

"""Configure specific supporting agent instances for use in substrates."""

from collections.abc import Mapping
import dataclasses
from typing import Any

from examples.modular.utils import supporting_agent_factory_with_overrides as bots_lib
import immutabledict


@dataclasses.dataclass(frozen=True)
class SupportingAgentConfig:
  """Class for configuring a supporting agent.

  Attributes:
    module_name: the name of the supporting agent module to load and use.
    overrides: a mapping of kwarg names to values to override.
  """

  module_name: str
  overrides: Mapping[str, Any] | None = None


SUPPORTING_AGENT_CONFIGS: Mapping[str, bots_lib.SupportingAgentFactory] = (
    immutabledict.immutabledict(
        # keep-sorted start numeric=yes block=yes
        labor_collective_action__fixed_rule_boss=SupportingAgentConfig(
            module_name='basic_puppet_agent',
            overrides=immutabledict.immutabledict(
                fixed_response_by_call_to_action=immutabledict.immutabledict({
                    # Note: it would be better to get these strings from the
                    # environment config somehow, but that's not trivial since
                    # it would introduce a cyclic dependency. Though it probably
                    # would work if we moved the variable in question down to
                    # the time_and_place_module. Will try that later.
                    'What does {name} decide?': 'Leave wages unchanged'
                })
            ),
        )
    )
)
