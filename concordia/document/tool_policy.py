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


def _is_valid_type(value: Any, json_type: str) -> bool:
  """Checks whether a value matches a limited JSON-schema primitive type."""
  if json_type == 'string':
    return isinstance(value, str)
  if json_type == 'number':
    return isinstance(value, (int, float)) and not isinstance(value, bool)
  if json_type == 'integer':
    return isinstance(value, int) and not isinstance(value, bool)
  if json_type == 'boolean':
    return isinstance(value, bool)
  if json_type == 'object':
    return isinstance(value, Mapping)
  if json_type == 'array':
    return isinstance(value, list)
  return True


def validate_input_schema(
    args: Mapping[str, Any], schema: Mapping[str, Any]
) -> str | None:
  """Validates tool args against a minimal JSON-schema subset.

  Supported subset:
    - top-level type check for "object"
    - required keys
    - property primitive types: string, number, integer, boolean, object, array

  Unsupported schema keywords are intentionally ignored.

  Args:
    args: Tool arguments to validate.
    schema: JSON-schema-like mapping.

  Returns:
    None when valid, otherwise a human-readable validation error.
  """
  schema_type = schema.get('type')
  if schema_type is not None and schema_type != 'object':
    return 'Schema type must be "object".'

  required = schema.get('required', ())
  if isinstance(required, list):
    required_keys = tuple(required)
  else:
    required_keys = ()
  for key in required_keys:
    if isinstance(key, str) and key not in args:
      return f'Missing required argument "{key}".'

  properties = schema.get('properties')
  if not isinstance(properties, Mapping):
    return None

  for key, value in args.items():
    property_schema = properties.get(key)
    if not isinstance(property_schema, Mapping):
      continue
    expected_type = property_schema.get('type')
    if not isinstance(expected_type, str):
      continue
    if not _is_valid_type(value, expected_type):
      return (
          f'Argument "{key}" must have type "{expected_type}", '
          f'got "{type(value).__name__}".'
      )

  return None


class SchemaValidatingPolicy:
  """Policy wrapper that validates args against Tool.input_schema."""

  def __init__(self, delegate: ToolPolicy | None = None):
    self._delegate = delegate

  def evaluate(
      self,
      call: ToolCall,
      available_tools: Mapping[str, tool_module.Tool],
  ) -> PolicyDecision:
    tool = available_tools.get(call.tool_name)
    if tool is not None and tool.input_schema is not None:
      error = validate_input_schema(call.args, tool.input_schema)
      if error:
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason=f'Input schema validation failed: {error}',
        )

    if self._delegate is None:
      return PolicyDecision()
    return self._delegate.evaluate(call, available_tools)
