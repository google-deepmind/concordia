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

"""Tests for tool policy primitives."""

from collections.abc import Mapping
from typing import Any

from absl.testing import absltest
from concordia.document import tool as tool_module
from concordia.document import tool_policy


class _StubTool(tool_module.Tool):
  """Minimal tool implementation used by tests."""

  @property
  def name(self) -> str:
    return 'stub'

  @property
  def description(self) -> str:
    return 'Stub tool.'

  def execute(self, **kwargs: Any) -> str:
    del kwargs
    return 'ok'


class _SchemaTool(_StubTool):
  """Tool stub with configurable input schema."""

  def __init__(self, schema: Mapping[str, Any]):
    self._schema = schema

  @property
  def input_schema(self) -> Mapping[str, Any] | None:
    return self._schema


class _DenyPolicy:
  """Policy stub that always denies tool calls."""

  def evaluate(
      self,
      call: tool_policy.ToolCall,
      available_tools: Mapping[str, tool_module.Tool],
  ) -> tool_policy.PolicyDecision:
    del available_tools
    return tool_policy.PolicyDecision(
        action=tool_policy.PolicyAction.DENY,
        reason=f'Denied {call.tool_name}.',
        tags=('test_tag',),
    )


class _RecordingPolicy:
  """Policy stub that records calls and returns a configured decision."""

  def __init__(self, decision: tool_policy.PolicyDecision):
    self._decision = decision
    self.calls: list[tool_policy.ToolCall] = []

  def evaluate(
      self,
      call: tool_policy.ToolCall,
      available_tools: Mapping[str, tool_module.Tool],
  ) -> tool_policy.PolicyDecision:
    del available_tools
    self.calls.append(call)
    return self._decision


class ToolPolicyTest(absltest.TestCase):

  def test_policy_decision_defaults_to_allow(self):
    decision = tool_policy.PolicyDecision()
    self.assertEqual(decision.action, tool_policy.PolicyAction.ALLOW)
    self.assertEqual(decision.reason, '')
    self.assertIsNone(decision.edited_args)
    self.assertEqual(decision.tags, ())

  def test_tool_call_captures_input(self):
    call = tool_policy.ToolCall(
        tool_name='search',
        args={'query': 'weather'},
        raw_response='{"tool":"search"}',
        attempt_index=2,
    )
    self.assertEqual(call.tool_name, 'search')
    self.assertEqual(call.args, {'query': 'weather'})
    self.assertEqual(call.raw_response, '{"tool":"search"}')
    self.assertEqual(call.attempt_index, 2)

  def test_tool_policy_protocol_shape(self):
    policy = _DenyPolicy()
    decision = policy.evaluate(
        tool_policy.ToolCall(
            tool_name='stub',
            args={},
            raw_response='{}',
            attempt_index=1,
        ),
        {'stub': _StubTool()},
    )
    self.assertEqual(decision.action, tool_policy.PolicyAction.DENY)
    self.assertEqual(decision.tags, ('test_tag',))

  def test_validate_input_schema_accepts_valid_args(self):
    schema = {
        'type': 'object',
        'required': ['query'],
        'properties': {
            'query': {'type': 'string'},
            'limit': {'type': 'integer'},
            'active': {'type': 'boolean'},
        },
    }

    self.assertIsNone(
        tool_policy.validate_input_schema(
            {'query': 'weather', 'limit': 3, 'active': True},
            schema,
        )
    )

  def test_validate_input_schema_rejects_missing_required_key(self):
    schema = {
        'type': 'object',
        'required': ['query'],
        'properties': {'query': {'type': 'string'}},
    }

    error = tool_policy.validate_input_schema({}, schema)

    self.assertEqual(error, 'Missing required argument "query".')

  def test_validate_input_schema_rejects_wrong_primitive_type(self):
    schema = {
        'type': 'object',
        'properties': {'limit': {'type': 'integer'}},
    }

    error = tool_policy.validate_input_schema({'limit': 'many'}, schema)

    self.assertEqual(
        error, 'Argument "limit" must have type "integer", got "str".'
    )

  def test_schema_validating_policy_denies_when_validation_fails(self):
    schema_tool = _SchemaTool(
        {
            'type': 'object',
            'required': ['query'],
            'properties': {'query': {'type': 'string'}},
        }
    )
    delegate = _RecordingPolicy(tool_policy.PolicyDecision())
    policy = tool_policy.SchemaValidatingPolicy(delegate)

    decision = policy.evaluate(
        tool_policy.ToolCall(
            tool_name='stub',
            args={},
            raw_response='{}',
            attempt_index=1,
        ),
        {'stub': schema_tool},
    )

    self.assertEqual(decision.action, tool_policy.PolicyAction.DENY)
    self.assertIn('Input schema validation failed', decision.reason)
    self.assertEmpty(delegate.calls)

  def test_schema_validating_policy_delegates_after_validation(self):
    schema_tool = _SchemaTool(
        {
            'type': 'object',
            'required': ['query'],
            'properties': {'query': {'type': 'string'}},
        }
    )
    delegate_decision = tool_policy.PolicyDecision(
        action=tool_policy.PolicyAction.EDIT,
        edited_args={'query': 'edited'},
        reason='delegate rewrite',
    )
    delegate = _RecordingPolicy(delegate_decision)
    policy = tool_policy.SchemaValidatingPolicy(delegate)

    decision = policy.evaluate(
        tool_policy.ToolCall(
            tool_name='stub',
            args={'query': 'weather'},
            raw_response='{}',
            attempt_index=1,
        ),
        {'stub': schema_tool},
    )

    self.assertEqual(decision, delegate_decision)
    self.assertLen(delegate.calls, 1)

  def test_schema_validating_policy_allows_when_no_delegate(self):
    schema_tool = _SchemaTool(
        {
            'type': 'object',
            'required': ['query'],
            'properties': {'query': {'type': 'string'}},
        }
    )
    policy = tool_policy.SchemaValidatingPolicy()

    decision = policy.evaluate(
        tool_policy.ToolCall(
            tool_name='stub',
            args={'query': 'weather'},
            raw_response='{}',
            attempt_index=1,
        ),
        {'stub': schema_tool},
    )

    self.assertEqual(decision.action, tool_policy.PolicyAction.ALLOW)


if __name__ == '__main__':
  absltest.main()
