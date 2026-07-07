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

"""Policy interfaces for tool execution in interactive documents."""

from collections.abc import Mapping
import dataclasses
import enum
from typing import Any, Protocol

from concordia.document import tool as tool_module


class PolicyAction(enum.Enum):
  """Action selected by a tool policy."""

  ALLOW = 'allow'
  DENY = 'deny'
  EDIT = 'edit'


@dataclasses.dataclass(frozen=True)
class PolicyDecision:
  """Result of a policy evaluation."""

  action: PolicyAction = PolicyAction.ALLOW
  reason: str = ''
  edited_args: dict[str, Any] | None = None
  tags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ToolCall:
  """Structured tool call data passed to policy implementations."""

  tool_name: str
  args: dict[str, Any]
  raw_response: str
  attempt_index: int


class ToolPolicy(Protocol):
  """Protocol for policy implementations used by tool-enabled documents."""

  def evaluate(
      self,
      call: ToolCall,
      available_tools: Mapping[str, tool_module.Tool],
  ) -> PolicyDecision:
    """Evaluates a tool call and returns a decision."""
    raise NotImplementedError
