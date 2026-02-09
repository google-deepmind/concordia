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
from typing import Any

from concordia.document import document
from concordia.document import interactive_document
from concordia.document import tool as tool_module
from concordia.language_model import language_model
import numpy as np


# Default configuration
DEFAULT_MAX_TOOL_CALLS_PER_QUESTION = 5
DEFAULT_MAX_TOOL_RESULT_LENGTH = 1000

# New tags for tool-related content
TOOL_CALL_TAG = 'tool_call'
TOOL_RESULT_TAG = 'tool_result'

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
    """
    super().__init__(model=model, contents=contents, rng=rng)
    self._tools = {t.name: t for t in tools}
    self._max_tool_calls = max_tool_calls_per_question
    self._max_result_length = max_tool_result_length

  def copy(self) -> 'InteractiveDocumentWithTools':
    """See base class."""
    return InteractiveDocumentWithTools(
        model=self._model,
        tools=list(self._tools.values()),
        contents=self.contents(),
        rng=self._rng,
        max_tool_calls_per_question=self._max_tool_calls,
        max_tool_result_length=self._max_result_length,
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

      # Record the raw LLM output (contains tool call)
      self._model_response(response)
      self._response('\n')

      # Record the tool call
      args_json = json.dumps(args)
      self._tool_call(f'[Tool Call: {tool_name}({args_json})]\n')

      # Execute and record result
      result = self._execute_tool(tool_name, args)
      self._tool_result(f'[Tool Result: {result}]\n')

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

      # Record the raw LLM output (contains tool call)
      self._model_response(response)
      self._response('\n')

      # Record the tool call
      args_json = json.dumps(args)
      self._tool_call(f'[Tool Call: {tool_name}({args_json})]\n')

      # Execute and record result
      result = self._execute_tool(tool_name, args)
      self._tool_result(f'[Tool Result: {result}]\n')

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
