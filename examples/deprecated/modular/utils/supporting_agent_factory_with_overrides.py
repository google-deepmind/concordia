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

"""Supporting agent factory wrapper with overrides.
"""

from collections.abc import Mapping
import types
from typing import Any

from concordia.agents.deprecated import entity_agent_with_logging


class SupportingAgentFactory:
  """Class for configuring a supporting agent.

  Wrapping a supporting agent module with this allows overriding of any
  kwarg used as a parameter of build_agent.
  """

  def __init__(
      self, module: types.ModuleType, overrides: Mapping[str, Any] | None = None
  ):
    """Initialize a supporting agent factory wrapper.

    Args:
      module: the supporting agent module to wrap.
      overrides: a mapping of kwarg names to values to override.
    """
    self._module = module
    self._overrides = overrides

  def build_agent(
      self,
      *args,
      **kwargs) -> entity_agent_with_logging.EntityAgentWithLogging:
    kwargs.update(self._overrides)
    return self._module.build_agent(*args, **kwargs)
