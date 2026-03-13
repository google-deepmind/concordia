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

"""Interactive document with tool use capabilities.

This module extends InteractiveDocument to support tool calling by the LLM.
When asking questions, the LLM can invoke registered tools to gather
information, and the tool calls and results are recorded in the document.
"""

from collections.abc import Collection, Iterable, Sequence
import json
import re
from typing import Any, Literal

from concordia.document import document
from concordia.document import interactive_document
from concordia.document import tool as tool_module
from concordia.document import tool_policy
from concordia.language_model import language_model
import numpy as np


# Default configuration
DEFAULT_MAX_TOOL_CALLS_PER_QUESTION = 5
DEFAULT_MAX_TOOL_RESULT_LENGTH = 1000

# New tags for tool-related content
TOOL_CALL_TAG = 'tool_call'
TOOL_RESULT_TAG = 'tool_result'
TOOL_POLICY_TAG = 'tool_policy'
TOOL_POLICY_ALLOW_TAG = 'tool_policy_allow'
TOOL_POLICY_DENY_OBSERVED_TAG = 'tool_policy_deny_observed'
TOOL_POLICY_EDIT_OBSERVED_TAG = 'tool_policy_edit_observed'
TOOL_POLICY_DENY_ENFORCED_TAG = 'tool_policy_deny_enforced'
TOOL_POLICY_EDIT_ENFORCED_TAG = 'tool_policy_edit_enforced'
TOOL_POLICY_ERROR_OBSERVED_TAG = 'tool_policy_error_observed'
TOOL_POLICY_ERROR_ENFORCED_TAG = 'tool_policy_error_enforced'

_VALID_ENFORCEMENT_MODES = frozenset({'observe', 'enforce'})

EnforcementMode = Literal['observe', 'enforce']

# Regex pattern to find JSON tool calls in LLM output
# Matches {"tool": "name", "args": {...}}
_TOOL_CALL_PATTERN = re.compile(
    r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{[^}]*\})\s*\}',
    re.DOTALL,
)


class InteractiveDocumentWithTools(interactive_document.InteractiveDocument):
  """An interactive document that supports tool use by the LLM.

  This extends InteractiveDocument with the ability for the LLM to call
  registered tools during question answering. Tool calls and their results
  are recorded in the document for transparency and reproducibility.

  Example usage:
    # Define a tool
    class WebSearchTool:
      @property
      def name(self) -> str:
        return "web_search"

      @property
      def description(self) -> str:
        return "Search the web. Args: query (str)"

      def execute(self, *, query: str) -> str:
        return "Search results for: " + query

    # Create document with tools
    doc = InteractiveDocumentWithTools(model, tools=[WebSearchTool()])

    # Ask a question - the LLM can use tools if needed
    answer = doc.open_question("What is the weather in London?")
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      tools: Sequence[tool_module.Tool] = (),
      contents: Iterable[document.Content] = (),
      rng: np.random.Generator | None = None,
      max_tool_calls_per_question: int = DEFAULT_MAX_TOOL_CALLS_PER_QUESTION,
      max_tool_result_length: int = DEFAULT_MAX_TOOL_RESULT_LENGTH,
      policy: tool_policy.ToolPolicy | None = None,
      enforcement_mode: EnforcementMode = 'observe',
  ) -> None:
    """Initializes the instance.

    Args:
      model: Language model to interact with.
      tools: Sequence of tools available for the LLM to use.
      contents: Initial contents of the document.
      rng: Randomization source.
      max_tool_calls_per_question: Maximum number of tool calls allowed per
        question before forcing a final answer.
      max_tool_result_length: Maximum character length for tool results. Results
        exceeding this will be truncated.
      policy: Optional policy used to evaluate tool calls.
      enforcement_mode: Whether policy decisions are observed or enforced.
    """
    super().__init__(model=model, contents=contents, rng=rng)
    if enforcement_mode not in _VALID_ENFORCEMENT_MODES:
      raise ValueError(
          'enforcement_mode must be "observe" or "enforce", '
          f'got "{enforcement_mode}".'
      )
    self._tools = {t.name: t for t in tools}
    self._max_tool_calls = max_tool_calls_per_question
    self._max_result_length = max_tool_result_length
    self._policy = policy
    self._enforcement_mode = enforcement_mode

  def copy(self) -> 'InteractiveDocumentWithTools':
    """See base class."""
    return InteractiveDocumentWithTools(
        model=self._model,
        tools=list(self._tools.values()),
        contents=self.contents(),
        rng=self._rng,
        max_tool_calls_per_question=self._max_tool_calls,
        max_tool_result_length=self._max_result_length,
        policy=self._policy,
        enforcement_mode=self._enforcement_mode,
    )

  def _format_tool_descriptions(self) -> str:
    """Formats tool descriptions for injection into the prompt."""
    if not self._tools:
      return ''

    lines = ['\nAvailable tools (use JSON format to call):']
    for name, t in self._tools.items():
      lines.append(f'- {name}: {t.description}')
    lines.append(
        '\nTo use a tool, respond with: {"tool": "<name>", "args": {...}}'
    )
    lines.append('After receiving tool results, provide your final answer.')
    return '\n'.join(lines)

  def _parse_tool_call(self, text: str) -> tuple[str, dict[str, Any]] | None:
    """Parses a tool call from LLM output.

    Args:
      text: The LLM's response text.

    Returns:
      A tuple of (tool_name, args_dict) if a tool call is found, else None.
    """
    match = _TOOL_CALL_PATTERN.search(text)
    if not match:
      return None

    tool_name = match.group(1)
    args_str = match.group(2)

    try:
      args = json.loads(args_str)
      if not isinstance(args, dict):
        return None
      return (tool_name, args)
    except json.JSONDecodeError:
      return None

  def _truncate_result(self, result: str) -> str:
    """Truncates a tool result to the maximum allowed length."""
    if len(result) <= self._max_result_length:
      return result
    return result[: self._max_result_length - 3] + '...'

  def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
    """Executes a tool and returns the (possibly truncated) result.

    Args:
      tool_name: Name of the tool to execute.
      args: Arguments to pass to the tool.

    Returns:
      The tool's result string, truncated if necessary.
    """
    if tool_name not in self._tools:
      return f'Error: Unknown tool "{tool_name}"'

    tool = self._tools[tool_name]
    try:
      result = tool.execute(**args)
      return self._truncate_result(result)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f'Error executing tool: {e}'

  def _tool_call(
      self, text: str, *, tags: Collection[str] = (), end: str = ''
  ) -> None:
    """Appends a tool call record to the document."""
    self.append(text + end, tags=[TOOL_CALL_TAG, *tags])

  def _tool_result(
      self, text: str, *, tags: Collection[str] = (), end: str = ''
  ) -> None:
    """Appends a tool result record to the document."""
    self.append(text + end, tags=[TOOL_RESULT_TAG, *tags])

  def _tool_policy(
      self, text: str, *, tags: Collection[str] = (), end: str = ''
  ) -> None:
    """Appends a tool policy record to the document."""
    self.append(text + end, tags=[TOOL_POLICY_TAG, *tags])

  @staticmethod
  def _merge_tags(*tag_groups: Collection[str]) -> tuple[str, ...]:
    """Merges tags while preserving order and removing duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    for tag_group in tag_groups:
      for tag in tag_group:
        if tag not in seen:
          merged.append(tag)
          seen.add(tag)
    return tuple(merged)

  def _serialize_args(self, args: dict[str, Any]) -> str:
    """Converts args to a stable string for logs."""
    try:
      return json.dumps(args, sort_keys=True)
    except TypeError:
      return str(args)

  def _format_policy_note(
      self,
      *,
      action: str,
      reason: str = '',
      edited_args: dict[str, Any] | None = None,
  ) -> str:
    """Formats a policy note for structured logging."""
    segments = [f'action={action}']
    if reason:
      segments.append(f'reason={reason}')
    if edited_args is not None:
      segments.append(f'edited_args={self._serialize_args(edited_args)}')
    return '[Tool Policy: ' + '; '.join(segments) + ']'

  def _evaluate_policy_decision(
      self,
      *,
      tool_name: str,
      args: dict[str, Any],
      raw_response: str,
      attempt_index: int,
  ) -> tool_policy.PolicyDecision:
    """Evaluates the configured policy for one tool call."""
    if self._policy is None:
      raise RuntimeError('Policy is not configured.')
    return self._policy.evaluate(
        tool_policy.ToolCall(
            tool_name=tool_name,
            args=dict(args),
            raw_response=raw_response,
            attempt_index=attempt_index,
        ),
        self._tools,
    )

  def _coerce_policy_decision(
      self, decision: Any
  ) -> tool_policy.PolicyDecision:
    """Validates and normalizes the policy decision shape."""
    if not isinstance(decision, tool_policy.PolicyDecision):
      raise ValueError(
          'Policy must return tool_policy.PolicyDecision, got '
          f'{type(decision).__name__}.'
      )
    if not isinstance(decision.action, tool_policy.PolicyAction):
      raise ValueError(
          'Policy decision action must be PolicyAction, got '
          f'{type(decision.action).__name__}.'
      )

    if isinstance(decision.tags, str):
      raise ValueError(
          'Policy decision tags must be an iterable of strings, not str.'
      )
    try:
      tags = tuple(decision.tags)
    except TypeError as error:
      raise ValueError(
          'Policy decision tags must be an iterable of strings.'
      ) from error
    if any(not isinstance(tag, str) for tag in tags):
      raise ValueError('Policy decision tags must contain only strings.')

    reason = str(decision.reason)
    edited_args = decision.edited_args
    if (
        decision.action is tool_policy.PolicyAction.EDIT
        and not isinstance(edited_args, dict)
    ):
      raise ValueError(
          'Policy decision action EDIT requires edited_args as dict.'
      )
    return tool_policy.PolicyDecision(
        action=decision.action,
        reason=reason,
        edited_args=edited_args,
        tags=tags,
    )

  def _policy_error_outcome(
      self, *, tool_name: str, args: dict[str, Any], error: Exception
  ) -> tuple[dict[str, Any], str, tuple[str, ...], str]:
    """Converts policy failures into observe/enforce runtime outcomes."""
    if self._enforcement_mode == 'observe':
      tags = (TOOL_POLICY_ERROR_OBSERVED_TAG,)
      note = self._format_policy_note(
          action='error_observed', reason=str(error)
      )
      return args, self._execute_tool(tool_name, args), tags, note

    tags = (TOOL_POLICY_ERROR_ENFORCED_TAG,)
    note = self._format_policy_note(action='error_enforced', reason=str(error))
    result = f'Error: Tool call "{tool_name}" blocked due to policy error.'
    return args, result, tags, note

  def _apply_policy_decision(
      self,
      *,
      tool_name: str,
      args: dict[str, Any],
      decision: tool_policy.PolicyDecision,
  ) -> tuple[dict[str, Any], str, tuple[str, ...], str]:
    """Applies a validated policy decision to produce execution outcome."""
    tags = decision.tags
    reason = decision.reason

    if decision.action is tool_policy.PolicyAction.ALLOW:
      merged_tags = self._merge_tags((TOOL_POLICY_ALLOW_TAG,), tags)
      note = self._format_policy_note(action='allow', reason=reason)
      return args, self._execute_tool(tool_name, args), merged_tags, note

    if decision.action is tool_policy.PolicyAction.DENY:
      if self._enforcement_mode == 'observe':
        merged_tags = self._merge_tags((TOOL_POLICY_DENY_OBSERVED_TAG,), tags)
        note = self._format_policy_note(action='deny_observed', reason=reason)
        result = self._execute_tool(tool_name, args)
        return args, result, merged_tags, note
      merged_tags = self._merge_tags((TOOL_POLICY_DENY_ENFORCED_TAG,), tags)
      note = self._format_policy_note(action='deny_enforced', reason=reason)
      result = f'Error: Tool call "{tool_name}" denied by policy.'
      return args, result, merged_tags, note

    if decision.action is tool_policy.PolicyAction.EDIT:
      if self._enforcement_mode == 'observe':
        merged_tags = self._merge_tags((TOOL_POLICY_EDIT_OBSERVED_TAG,), tags)
        note = self._format_policy_note(
            action='edit_observed',
            reason=reason,
            edited_args=decision.edited_args,
        )
        return args, self._execute_tool(tool_name, args), merged_tags, note

      edited_args = decision.edited_args
      if not isinstance(edited_args, dict):
        return self._policy_error_outcome(
            tool_name=tool_name,
            args=args,
            error=ValueError(
                'Policy decision action EDIT requires edited_args as dict.'
            ),
        )

      merged_tags = self._merge_tags((TOOL_POLICY_EDIT_ENFORCED_TAG,), tags)
      note = self._format_policy_note(
          action='edit_enforced', reason=reason, edited_args=edited_args
      )
      return (
          edited_args,
          self._execute_tool(tool_name, edited_args),
          merged_tags,
          note,
      )

    return self._policy_error_outcome(
        tool_name=tool_name,
        args=args,
        error=ValueError(
            'Policy decision action must be one of ALLOW, DENY, EDIT.'
        ),
    )

  def _run_policy(
      self,
      *,
      tool_name: str,
      args: dict[str, Any],
      raw_response: str,
      attempt_index: int,
  ) -> tuple[dict[str, Any], str, tuple[str, ...], str | None]:
    """Evaluates policy and executes tool call according to mode."""
    if self._policy is None:
      return args, self._execute_tool(tool_name, args), (), None

    try:
      decision = self._evaluate_policy_decision(
          tool_name=tool_name,
          args=args,
          raw_response=raw_response,
          attempt_index=attempt_index,
      )
      decision = self._coerce_policy_decision(decision)
    except Exception as error:  # pylint: disable=broad-exception-caught
      return self._policy_error_outcome(
          tool_name=tool_name, args=args, error=error
      )

    return self._apply_policy_decision(
        tool_name=tool_name, args=args, decision=decision
    )

  def _record_tool_interaction(
      self,
      *,
      raw_response: str,
      tool_name: str,
      original_args: dict[str, Any],
      executed_args: dict[str, Any],
      result: str,
      policy_tags: tuple[str, ...],
      policy_note: str | None,
  ) -> None:
    """Records tool call, policy decision, and tool result in the document."""
    self._model_response(raw_response)
    self._response('\n')

    call_text = (
        f'[Tool Call: {tool_name}({self._serialize_args(original_args)})]'
    )
    if executed_args != original_args:
      call_text += (
          ' [Executed with args: '
          f'{self._serialize_args(executed_args)}]'
      )
    self._tool_call(call_text + '\n', tags=policy_tags)

    if policy_note:
      self._tool_policy(policy_note + '\n', tags=policy_tags)

    self._tool_result(f'[Tool Result: {result}]\n', tags=policy_tags)

  def open_question(
      self,
      question: str,
      *,
      forced_response: str | None = None,
      answer_prefix: str = '',
      answer_suffix: str = '',
      max_tokens: int = interactive_document.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = ('\n',),
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      question_label: str = 'Question',
      answer_label: str = 'Answer',
  ) -> str:
    """Asks an open question, allowing the LLM to use tools if needed.

    This overrides the base class method to add tool-use capability. The LLM
    can respond with a JSON tool call, which will be executed and its result
    appended to the document. The LLM then continues until it provides a
    final answer (without a tool call) or reaches the tool call limit.

    Args:
      question: The question to ask.
      forced_response: Forces the document to provide this response.
      answer_prefix: A prefix to append to the model's prompt.
      answer_suffix: A suffix to append to the model's response.
      max_tokens: Maximum tokens to sample from the model.
      terminators: Strings that terminate the response.
      temperature: Sampling temperature.
      top_p: Top-p sampling parameter.
      top_k: Top-k sampling parameter.
      question_label: Label for the question.
      answer_label: Label for the answer.

    Returns:
      The agent's final answer (after any tool usage).
    """
    # If no tools or forced response, use base implementation
    if not self._tools or forced_response is not None:
      return super().open_question(
          question=question,
          forced_response=forced_response,
          answer_prefix=answer_prefix,
          answer_suffix=answer_suffix,
          max_tokens=max_tokens,
          terminators=terminators,
          temperature=temperature,
          top_p=top_p,
          top_k=top_k,
          question_label=question_label,
          answer_label=answer_label,
      )

    # Inject tool descriptions as a preamble (system-prompt style)
    tool_info = self._format_tool_descriptions()
    self.statement(tool_info)

    self._question(f'{question_label}: {question}\n')
    self._response(f'{answer_label}: {answer_prefix}')

    tool_calls_made = 0

    while tool_calls_made < self._max_tool_calls:
      # Sample from the model
      response = self._model.sample_text(
          prompt=self._model_view.text(),
          max_tokens=max_tokens,
          terminators=[],  # Don't terminate early - we need full JSON
          temperature=temperature,
          top_p=top_p,
          top_k=top_k,
      )

      # Check if this is a tool call
      tool_call = self._parse_tool_call(response)

      if tool_call is None:
        # No tool call - this is the final answer
        response = response.removeprefix(answer_prefix)
        # Apply terminators to the final answer
        for terminator in terminators:
          if terminator in response:
            response = response[: response.index(terminator)]
        self._model_response(response)
        self._response(f'{answer_suffix}\n')
        return response

      # Execute the tool call
      tool_name, args = tool_call
      tool_calls_made += 1

      executed_args, result, policy_tags, policy_note = self._run_policy(
          tool_name=tool_name,
          args=args,
          raw_response=response,
          attempt_index=tool_calls_made,
      )
      self._record_tool_interaction(
          raw_response=response,
          tool_name=tool_name,
          original_args=args,
          executed_args=executed_args,
          result=result,
          policy_tags=policy_tags,
          policy_note=policy_note,
      )

      # Prepare for next iteration
      self._response(f'{answer_label}: ')

    # Exceeded tool call limit - force a final answer
    self._response('[Max tool calls reached. Provide final answer.]\n')
    self._response(f'{answer_label}: ')

    response = self._model.sample_text(
        prompt=self._model_view.text(),
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    response = response.removeprefix(answer_prefix)
    self._model_response(response)
    self._response(f'{answer_suffix}\n')
    return response

  def multiple_choice_question(
      self,
      question: str,
      answers: Sequence[str],
      randomize_choices: bool = True,
  ) -> int:
    """Presents a multiple choice question, allowing tool use before answering.

    The LLM can either:
    1. Call tools to gather information before answering
    2. Answer the question directly

    If the tool call budget is exhausted, the LLM is forced to answer
    immediately without further tool calls.

    Args:
      question: The question to ask.
      answers: The available answer choices.
      randomize_choices: Whether to randomize choice order.

    Returns:
      The index of the selected answer in the original answers list.
    """
    # If no tools, use base implementation directly
    if not self._tools:
      return super().multiple_choice_question(
          question=question,
          answers=answers,
          randomize_choices=randomize_choices,
      )

    # Set up randomized choices
    if randomize_choices:
      original_indices = self._rng.permutation(len(answers))
    else:
      original_indices = list(range(len(answers)))

    letters = 'abcdefghijklmnopqrstuvwxyz'
    options = {
        letters[i]: answers[idx] for i, idx in enumerate(original_indices)
    }

    # Inject tool descriptions as a preamble (system-prompt style)
    tool_info = self._format_tool_descriptions()
    tool_info += '\nAlternatively, answer with the letter of your choice.'
    self.statement(tool_info)

    # Format the question with choices
    self._question(f'Question: {question}\n')
    for key, option in options.items():
      self._question(f'  ({key}) {option}\n')
    self._response('Answer: ')

    tool_calls_made = 0

    while tool_calls_made < self._max_tool_calls:
      # Sample from the model - allow space for tool call JSON
      response = self._model.sample_text(
          prompt=self._model_view.text(),
          max_tokens=500,
          terminators=[],
          temperature=0.0,
      )

      # Check if this is a tool call
      tool_call = self._parse_tool_call(response)

      if tool_call is None:
        # Not a tool call - try to parse as answer choice
        # Look for a letter choice in the response
        response_lower = response.lower().strip()
        for letter in options.keys():
          if letter in response_lower:
            self._model_response(f'({letter})\n')
            # Find the original index
            letter_idx = list(options.keys()).index(letter)
            return original_indices[letter_idx]

        # Couldn't parse - treat as final response and use sample_choice
        self._model_response(response + '\n')
        self._response('Please select from the options: (')
        idx, choice_response, debug = self._model.sample_choice(
            prompt=self._model_view.text(),
            responses=list(options.keys()),
        )
        self._model_response(choice_response)
        self._response(')\n')
        self.debug(f'[{debug}]')
        return original_indices[idx]

      # Execute the tool call
      tool_name, args = tool_call
      tool_calls_made += 1

      executed_args, result, policy_tags, policy_note = self._run_policy(
          tool_name=tool_name,
          args=args,
          raw_response=response,
          attempt_index=tool_calls_made,
      )
      self._record_tool_interaction(
          raw_response=response,
          tool_name=tool_name,
          original_args=args,
          executed_args=executed_args,
          result=result,
          policy_tags=policy_tags,
          policy_note=policy_note,
      )

      # Prepare for next iteration
      self._response('Answer: ')

    # Exceeded tool call limit - force answer using sample_choice
    self._response('[Max tool calls reached. Select your answer.]\n')
    self._response('Answer: (')
    idx, response, debug = self._model.sample_choice(
        prompt=self._model_view.text(),
        responses=list(options.keys()),
    )
    self._model_response(response)
    self._response(')\n')
    self.debug(f'[{debug}]')
    return original_indices[idx]

  def yes_no_question(self, question: str) -> bool:
    """Presents a yes/no question, allowing tool use before answering.

    Args:
      question: The question to ask.

    Returns:
      True if answered Yes, False if answered No.
    """
    result = self.multiple_choice_question(question, ['Yes', 'No'])
    return result == 0
