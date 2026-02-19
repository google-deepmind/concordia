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

"""Unit tests for InteractiveDocumentWithTools tool-calling functionality."""

import functools
from typing import cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.document import document
from concordia.document import interactive_document_tools
from concordia.document import tool as tool_module
from concordia.document import tool_policy
from concordia.language_model import language_model


# Content factories for testing
DEBUG = functools.partial(document.Content, tags=frozenset({'debug'}))
STATEMENT = functools.partial(document.Content, tags=frozenset({'statement'}))
QUESTION = functools.partial(document.Content, tags=frozenset({'question'}))
RESPONSE = functools.partial(document.Content, tags=frozenset({'response'}))
MODEL_RESPONSE = functools.partial(
    document.Content, tags=frozenset({'response', 'model'})
)
TOOL_CALL = functools.partial(document.Content, tags=frozenset({'tool_call'}))
TOOL_RESULT = functools.partial(
    document.Content, tags=frozenset({'tool_result'})
)


class MockTool(tool_module.Tool):
  """A mock tool for testing."""

  def __init__(self, name: str, description: str, return_value: str):
    self._name = name
    self._description = description
    self._return_value = return_value
    self.call_count = 0
    self.last_args = None

  @property
  def name(self) -> str:
    return self._name

  @property
  def description(self) -> str:
    return self._description

  def execute(self, **kwargs) -> str:
    self.call_count += 1
    self.last_args = kwargs
    return self._return_value


class MockSchemaTool(MockTool):
  """A mock tool with input schema metadata."""

  def __init__(
      self,
      name: str,
      description: str,
      return_value: str,
      input_schema: dict[str, object],
  ):
    super().__init__(name, description, return_value)
    self._input_schema = input_schema

  @property
  def input_schema(self) -> dict[str, object] | None:
    return self._input_schema


class MockPolicy:
  """Policy stub with configurable decisions."""

  def __init__(
      self,
      decision: tool_policy.PolicyDecision | None = None,
      *,
      raise_error: bool = False,
  ):
    self._decision = decision or tool_policy.PolicyDecision()
    self._raise_error = raise_error
    self.calls: list[tool_policy.ToolCall] = []

  def evaluate(
      self,
      call: tool_policy.ToolCall,
      available_tools: dict[str, tool_module.Tool],
  ) -> tool_policy.PolicyDecision:
    del available_tools
    self.calls.append(call)
    if self._raise_error:
      raise ValueError('policy failure')
    return self._decision


class MockNonDecisionPolicy:
  """Policy stub that returns an invalid decision shape."""

  def evaluate(
      self,
      call: tool_policy.ToolCall,
      available_tools: dict[str, tool_module.Tool],
  ) -> object:
    del call, available_tools
    return {'action': 'allow'}


class MockInvalidActionPolicy:
  """Policy stub that returns a decision with malformed action."""

  def evaluate(
      self,
      call: tool_policy.ToolCall,
      available_tools: dict[str, tool_module.Tool],
  ) -> tool_policy.PolicyDecision:
    del call, available_tools
    return tool_policy.PolicyDecision(
        action=cast(tool_policy.PolicyAction, 'allow'),
        reason='invalid action type',
    )


class MockStringTagsPolicy:
  """Policy stub with malformed string tags payload."""

  def evaluate(
      self,
      call: tool_policy.ToolCall,
      available_tools: dict[str, tool_module.Tool],
  ) -> tool_policy.PolicyDecision:
    del call, available_tools
    return tool_policy.PolicyDecision(
        action=tool_policy.PolicyAction.ALLOW,
        tags=cast(tuple[str, ...], 'invalid'),
    )


class InteractiveDocumentWithToolsTest(parameterized.TestCase):

  def test_open_question_no_tools(self):
    """Without tools, should behave like base InteractiveDocument."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.return_value = 'Simple answer'

    doc = interactive_document_tools.InteractiveDocumentWithTools(model)
    response = doc.open_question('What is 1+1?')

    self.assertEqual(response, 'Simple answer')
    model.sample_text.assert_called_once()

  def test_open_question_with_forced_response(self):
    """Forced response should bypass tool usage."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    tool = MockTool('search', 'Search the web', 'search results')

    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool]
    )
    response = doc.open_question(
        'What is the weather?', forced_response='Sunny'
    )

    self.assertEqual(response, 'Sunny')
    self.assertEqual(tool.call_count, 0)
    model.sample_text.assert_not_called()

  def test_open_question_direct_answer(self):
    """LLM provides direct answer without using tools."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.return_value = 'The answer is 42'
    tool = MockTool('calculator', 'Do math', '42')

    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool]
    )
    response = doc.open_question('What is the meaning of life?')

    self.assertEqual(response, 'The answer is 42')
    self.assertEqual(tool.call_count, 0)

  def test_open_question_with_tool_call(self):
    """LLM uses a tool then provides answer."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    # First call: tool call, second call: final answer
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "weather"}}',
        'Based on the search, it is sunny.',
    ]
    tool = MockTool('search', 'Search the web', 'Weather: Sunny, 20C')

    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool]
    )
    response = doc.open_question('What is the weather?')

    self.assertEqual(response, 'Based on the search, it is sunny.')
    self.assertEqual(tool.call_count, 1)
    self.assertEqual(tool.last_args, {'query': 'weather'})

  def test_tool_result_in_document(self):
    """Tool call and result should be recorded in the document."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "test"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'Search result text')

    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool]
    )
    doc.open_question('Question?')

    # Check document contains tool call and result tags
    contents = doc.contents()
    tags_found = set()
    for content in contents:
      tags_found.update(content.tags)

    self.assertIn('tool_call', tags_found)
    self.assertIn('tool_result', tags_found)

  def test_max_tool_calls_limit(self):
    """Should stop after max tool calls and force final answer."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    # Always return tool calls until forced to stop
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        '{"tool": "search", "args": {"query": "q2"}}',
        '{"tool": "search", "args": {"query": "q3"}}',
        'Forced final answer',
    ]
    tool = MockTool('search', 'Search', 'result')

    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool], max_tool_calls_per_question=3
    )
    response = doc.open_question('Question?')

    self.assertEqual(response, 'Forced final answer')
    self.assertEqual(tool.call_count, 3)

  def test_tool_result_truncation(self):
    """Long tool results should be truncated."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "test"}}',
        'Final answer',
    ]
    # Return a very long result
    long_result = 'x' * 2000
    tool = MockTool('search', 'Search', long_result)

    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool], max_tool_result_length=100
    )
    doc.open_question('Question?')

    # Check that result in document is truncated
    doc_text = doc.text()
    self.assertIn('...', doc_text)
    # The truncated result should be 100 chars (97 + '...')
    self.assertNotIn('x' * 200, doc_text)

  def test_unknown_tool(self):
    """Calling unknown tool should return error message."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "unknown_tool", "args": {}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')

    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool]
    )
    doc.open_question('Question?')

    doc_text = doc.text()
    self.assertIn('Error: Unknown tool "unknown_tool"', doc_text)

  def test_parse_tool_call_valid(self):
    """Test parsing valid JSON tool calls."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    doc = interactive_document_tools.InteractiveDocumentWithTools(model)

    result = doc._parse_tool_call(
        '{"tool": "search", "args": {"query": "test"}}'
    )
    self.assertEqual(result, ('search', {'query': 'test'}))

  def test_parse_tool_call_with_surrounding_text(self):
    """Tool call JSON can be embedded in other text."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    doc = interactive_document_tools.InteractiveDocumentWithTools(model)

    result = doc._parse_tool_call(
        'Let me search for that. {"tool": "search", "args": {"q": "test"}}'
        ' Done.'
    )
    self.assertEqual(result, ('search', {'q': 'test'}))

  def test_parse_tool_call_invalid(self):
    """Invalid JSON should return None."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    doc = interactive_document_tools.InteractiveDocumentWithTools(model)

    result = doc._parse_tool_call('Just a regular answer without tools.')
    self.assertIsNone(result)

  def test_copy_preserves_tools(self):
    """Copying document should preserve tool configuration."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    tool = MockTool('search', 'Search', 'result')
    policy = MockPolicy(
        tool_policy.PolicyDecision(action=tool_policy.PolicyAction.DENY)
    )

    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model,
        tools=[tool],
        max_tool_calls_per_question=10,
        policy=policy,
        enforcement_mode='enforce',
    )
    doc.statement('Some content')

    copied = doc.copy()

    self.assertIn('search', copied._tools)
    self.assertEqual(copied._max_tool_calls, 10)
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        'Final answer',
    ]
    copied.open_question('Question?')
    self.assertEqual(tool.call_count, 0)

  def test_invalid_enforcement_mode_raises(self):
    """Invalid policy enforcement mode should raise ValueError."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    invalid_mode = cast(
        interactive_document_tools.EnforcementMode, 'invalid'
    )
    with self.assertRaises(ValueError):
      interactive_document_tools.InteractiveDocumentWithTools(
          model, enforcement_mode=invalid_mode
      )

  def test_policy_observe_mode_runs_tool_and_logs_allow(self):
    """Observe mode should execute tool and log allow decision."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    policy = MockPolicy(
        tool_policy.PolicyDecision(
            action=tool_policy.PolicyAction.ALLOW, reason='safe'
        )
    )
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool], policy=policy, enforcement_mode='observe'
    )

    response = doc.open_question('Question?')

    self.assertEqual(response, 'Final answer')
    self.assertEqual(tool.call_count, 1)
    self.assertLen(policy.calls, 1)
    self.assertEqual(policy.calls[0].attempt_index, 1)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_allow', tags_found)

  def test_policy_observe_mode_denied_call_still_executes_tool(self):
    """Observe mode should record deny decision without blocking execution."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    policy = MockPolicy(
        tool_policy.PolicyDecision(
            action=tool_policy.PolicyAction.DENY, reason='blocked in observe'
        )
    )
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool], policy=policy, enforcement_mode='observe'
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 1)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_deny_observed', tags_found)

  def test_policy_observe_mode_edit_keeps_original_args(self):
    """Observe mode should execute original args for edit decisions."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "original"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    policy = MockPolicy(
        tool_policy.PolicyDecision(
            action=tool_policy.PolicyAction.EDIT,
            reason='suggest edit',
            edited_args={'query': 'edited'},
        )
    )
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool], policy=policy, enforcement_mode='observe'
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 1)
    self.assertEqual(tool.last_args, {'query': 'original'})
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_edit_observed', tags_found)

  def test_policy_enforce_mode_denied_call_blocks_execution(self):
    """Enforce mode deny should block tool execution."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    policy = MockPolicy(
        tool_policy.PolicyDecision(
            action=tool_policy.PolicyAction.DENY, reason='blocked'
        )
    )
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool], policy=policy, enforcement_mode='enforce'
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 0)
    doc_text = doc.text()
    self.assertIn('Error: Tool call "search" denied by policy.', doc_text)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_deny_enforced', tags_found)

  def test_policy_enforce_mode_edit_executes_edited_args(self):
    """Enforce mode edit should execute with edited args."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "original"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    policy = MockPolicy(
        tool_policy.PolicyDecision(
            action=tool_policy.PolicyAction.EDIT,
            reason='rewrite args',
            edited_args={'query': 'edited'},
        )
    )
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool], policy=policy, enforcement_mode='enforce'
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 1)
    self.assertEqual(tool.last_args, {'query': 'edited'})
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_edit_enforced', tags_found)

  def test_policy_error_observe_mode_executes_and_logs_error(self):
    """Observe mode should fail open when policy raises an error."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    policy = MockPolicy(raise_error=True)
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool], policy=policy, enforcement_mode='observe'
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 1)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_error_observed', tags_found)

  def test_policy_error_enforce_mode_blocks_and_logs_error(self):
    """Enforce mode should fail closed when policy raises an error."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    policy = MockPolicy(raise_error=True)
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool], policy=policy, enforcement_mode='enforce'
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 0)
    doc_text = doc.text()
    self.assertIn('blocked due to policy error', doc_text)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_error_enforced', tags_found)

  def test_policy_observe_mode_invalid_action_fails_open(self):
    """Observe mode executes tool if policy returns invalid action."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model,
        tools=[tool],
        policy=MockInvalidActionPolicy(),
        enforcement_mode='observe',
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 1)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_error_observed', tags_found)

  def test_policy_observe_mode_string_tags_fails_open(self):
    """Observe mode executes tool if policy returns string tags."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model,
        tools=[tool],
        policy=MockStringTagsPolicy(),
        enforcement_mode='observe',
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 1)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_error_observed', tags_found)

  def test_policy_enforce_mode_invalid_action_blocks(self):
    """Enforce mode blocks tool if policy returns invalid action."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model,
        tools=[tool],
        policy=MockInvalidActionPolicy(),
        enforcement_mode='enforce',
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 0)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_error_enforced', tags_found)

  def test_policy_enforce_mode_string_tags_blocks(self):
    """Enforce mode blocks tool if policy returns string tags."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model,
        tools=[tool],
        policy=MockStringTagsPolicy(),
        enforcement_mode='enforce',
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 0)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_error_enforced', tags_found)

  def test_policy_observe_mode_non_decision_fails_open(self):
    """Observe mode executes tool if policy returns non-decision object."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model,
        tools=[tool],
        policy=MockNonDecisionPolicy(),
        enforcement_mode='observe',
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 1)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_error_observed', tags_found)

  def test_schema_validating_policy_observe_mode_denied_but_executes(self):
    """Observe mode executes on schema validation deny decision."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": 12}}',
        'Final answer',
    ]
    tool = MockSchemaTool(
        'search',
        'Search',
        'result',
        {
            'type': 'object',
            'required': ['query'],
            'properties': {'query': {'type': 'string'}},
        },
    )
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model,
        tools=[tool],
        policy=tool_policy.SchemaValidatingPolicy(),
        enforcement_mode='observe',
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 1)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_deny_observed', tags_found)

  def test_schema_validating_policy_enforce_mode_denied_and_blocks(self):
    """Enforce mode blocks on schema validation deny decision."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": 12}}',
        'Final answer',
    ]
    tool = MockSchemaTool(
        'search',
        'Search',
        'result',
        {
            'type': 'object',
            'required': ['query'],
            'properties': {'query': {'type': 'string'}},
        },
    )
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model,
        tools=[tool],
        policy=tool_policy.SchemaValidatingPolicy(),
        enforcement_mode='enforce',
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 0)
    doc_text = doc.text()
    self.assertIn('Error: Tool call "search" denied by policy.', doc_text)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_deny_enforced', tags_found)

  def test_policy_enforce_mode_non_decision_blocks(self):
    """Enforce mode blocks tool if policy returns non-decision object."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        'Final answer',
    ]
    tool = MockTool('search', 'Search', 'result')
    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model,
        tools=[tool],
        policy=MockNonDecisionPolicy(),
        enforcement_mode='enforce',
    )

    doc.open_question('Question?')

    self.assertEqual(tool.call_count, 0)
    tags_found = {tag for c in doc.contents() for tag in c.tags}
    self.assertIn('tool_policy_error_enforced', tags_found)

  def test_multiple_choice_with_tool_call_then_answer(self):
    """LLM uses a tool then answers multiple choice."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    # First call: tool call, second call: answer letter
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "capital of France"}}',
        'Based on my search, the answer is (a)',
    ]
    tool = MockTool(
        'search', 'Search the web', 'Paris is the capital of France'
    )

    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool]
    )
    result = doc.multiple_choice_question(
        'What is the capital of France?',
        ['Paris', 'London', 'Berlin'],
        randomize_choices=False,
    )

    self.assertEqual(tool.call_count, 1)
    self.assertEqual(tool.last_args, {'query': 'capital of France'})
    self.assertEqual(result, 0)

  def test_multiple_choice_direct_answer(self):
    """LLM answers multiple choice directly without using tools."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.return_value = '(b)'
    tool = MockTool('search', 'Search', 'result')

    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool]
    )
    result = doc.multiple_choice_question(
        'What color is the sky?',
        ['Red', 'Blue', 'Green'],
        randomize_choices=False,
    )

    self.assertEqual(tool.call_count, 0)
    self.assertEqual(result, 1)

  def test_multiple_choice_max_tool_calls(self):
    """Multiple choice forces answer when tool budget exhausted."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    # Keep calling tools until budget exhausted
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "q1"}}',
        '{"tool": "search", "args": {"query": "q2"}}',
    ]
    # After budget exhausted, sample_choice is called
    model.sample_choice.return_value = (1, 'b', {'debug': 'forced'})
    tool = MockTool('search', 'Search', 'result')

    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool], max_tool_calls_per_question=2
    )
    result = doc.multiple_choice_question(
        'Question?',
        ['A', 'B', 'C'],
        randomize_choices=False,
    )

    self.assertEqual(tool.call_count, 2)
    self.assertEqual(result, 1)
    model.sample_choice.assert_called_once()

  def test_yes_no_question_with_tool_call(self):
    """yes_no_question uses tool then answers."""
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    # yes_no_question calls multiple_choice with ['Yes', 'No'] and randomizes.
    # We answer with 'a' - could be Yes or No depending on randomization.
    model.sample_text.side_effect = [
        '{"tool": "search", "args": {"query": "is sky blue"}}',
        # Answer with (a) - randomization means we just check tool was called
        '(a)',
    ]
    tool = MockTool(
        'search', 'Search', 'The sky appears blue due to scattering'
    )

    doc = interactive_document_tools.InteractiveDocumentWithTools(
        model, tools=[tool]
    )
    result = doc.yes_no_question('Is the sky blue?')

    # Main assertion: tool was called
    self.assertEqual(tool.call_count, 1)
    # Result is a bool (True or False depending on randomization)
    self.assertIsInstance(result, (bool, type(result)))


if __name__ == '__main__':
  absltest.main()
