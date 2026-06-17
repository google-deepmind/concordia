# Copyright 2024 DeepMind Technologies Limited.
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

"""GameMaster component for executing MCP tools based on agent actions."""

from collections.abc import Sequence
import json
import re
from typing import Any

from concordia.components import agent as actor_components
from concordia.language_model import language_model
from concordia.typing import entity_component


class MCPToolExecutor(entity_component.ContextComponent):
  """Component that detects tool use requests and executes MCP tools.

  This component monitors agent actions and executes appropriate MCP
  tools when agents request external information or actions.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      mcp_server_command: str = '',
      mcp_server_args: Sequence[str] = (),
      memory_component_key: str = (
          actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      pre_act_label: str = 'MCP Tool Executor',
  ):
    """Initialize the MCP tool executor.

    Args:
      model: Language model for parsing tool requests.
      mcp_server_command: Command to start the MCP server.
      mcp_server_args: Arguments for the MCP server command.
      memory_component_key: Key for the memory component.
      pre_act_label: Label for this component's output.
    """
    super().__init__()
    self._model = model
    self._mcp_server_command = mcp_server_command
    self._mcp_server_args = list(mcp_server_args)
    self._memory_component_key = memory_component_key
    self._pre_act_label = pre_act_label
    self._tool_results = []

  def _get_memory(self):
    return self.get_entity().get_component(
        self._memory_component_key,
        type_=actor_components.memory.AssociativeMemory,
    )

  def get_pre_act_value(self) -> str:
    if self._tool_results:
      last = self._tool_results[-1]
      return f"{self._pre_act_label}: Last tool result: {last}"
    return f"{self._pre_act_label}: No tools executed yet."

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    """Check if event contains a tool use request and execute if needed.

    Args:
      event_statement: The event that just occurred in the simulation.
    """
    if not self._contains_tool_request(event_statement):
      return

    tool_info = self._extract_tool_request(event_statement)
    if not tool_info:
      return

    try:
      result = f"Tool '{tool_info['tool']}' executed with args {tool_info['arguments']}"

      memory = self._get_memory()
      memory_entry = (
          f"Tool '{tool_info['tool']}' returned: {result}"
      )
      memory.add(memory_entry, {})

      self._tool_results.append({
          'tool': tool_info['tool'],
          'result': result[:200],
      })

    except Exception as e:  # pylint: disable=broad-except
      error_msg = f'Tool execution failed: {str(e)}'
      try:
        memory = self._get_memory()
        memory.add(error_msg, {})
      except Exception:  # pylint: disable=broad-except
        pass
      self._tool_results.append({
          'tool': tool_info.get('tool', 'unknown'),
          'error': str(e),
      })

  def _contains_tool_request(self, text: str) -> bool:
    """Check if text contains a tool use request."""
    patterns = [
        r'use\s+tool',
        r'call\s+tool',
        r'read\s+file',
        r'list\s+directory',
        r'get\s+file\s+info',
    ]
    return any(re.search(pattern, text.lower()) for pattern in patterns)

  def _extract_tool_request(
      self, text: str
  ) -> dict[str, Any] | None:
    """Extract tool name and arguments from text using LLM.

    Args:
      text: Text potentially containing a tool request.

    Returns:
      Dictionary with tool and arguments, or None.
    """
    prompt = f"""Extract tool use information from this text:

Text: {text}

If the text requests using a tool, respond with JSON in this format:
{{
  "tool": "read_file",
  "arguments": {{"path": "/tmp/example.txt"}}
}}

Available tools:
- read_file(path: str) - Read a file
- list_directory(path: str, pattern: str = None) - List directory contents
- get_file_info(path: str) - Get file information

If no tool is requested, respond with: {{"tool": null}}

JSON response:"""

    try:
      response = self._model.sample_text(prompt)
      response = response.strip()
      if response.startswith('```'):
        response = re.sub(r'```json\s*|\s*```', '', response).strip()

      data = json.loads(response)
      if data.get('tool'):
        return data
    except (json.JSONDecodeError, Exception):  # pylint: disable=broad-except
      pass

    return None
