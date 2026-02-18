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

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.tools import mcp_client
from concordia.typing import component


class MCPToolExecutor(component.Component):
    """Component that detects tool use requests and executes MCP tools.
    
    This component monitors agent actions and executes appropriate MCP
    tools when agents request external information or actions.
    """
    
    def __init__(
        self,
        model: language_model.LanguageModel,
        mcp_client_manager: mcp_client.MCPClientManager,
        memory: associative_memory.AssociativeMemory,
        component_name: str = 'MCP Tool Executor',
    ):
        """Initialize the MCP tool executor.
        
        Args:
            model: Language model for parsing tool requests
            mcp_client_manager: Manager for MCP clients
            memory: Memory to store tool results
            component_name: Name of this component
        """
        self._model = model
        self._mcp_manager = mcp_client_manager
        self._memory = memory
        self._component_name = component_name
        self._tool_results = []
        
    def name(self) -> str:
        return self._component_name
        
    def get_last_log(self):
        if self._tool_results:
            return {
                'Summary': f'Executed {len(self._tool_results)} tool(s)',
                'Results': self._tool_results[-5:]  # Last 5 results
            }
        return {}
        
    def update_after_event(
        self,
        event_statement: str,
    ) -> None:
        """Check if event contains a tool use request and execute if needed.
        
        Args:
            event_statement: The event that just occurred in the simulation
        """
        # Check if event contains tool use pattern
        if not self._contains_tool_request(event_statement):
            return
            
        # Extract tool request details
        tool_info = self._extract_tool_request(event_statement)
        if not tool_info:
            return
            
        # Execute the tool
        try:
            result = mcp_client.run_sync(
                self._mcp_manager.call_tool(
                    client_name=tool_info['client'],
                    tool_name=tool_info['tool'],
                    arguments=tool_info['arguments'],
                )
            )
            
            # Store result in memory
            memory_entry = (
                f"Tool '{tool_info['tool']}' returned: {result}"
            )
            self._memory.add(memory_entry)
            
            # Track result
            self._tool_results.append({
                'tool': tool_info['tool'],
                'result': result[:200]  # Truncate long results
            })
            
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            self._memory.add(error_msg)
            self._tool_results.append({
                'tool': tool_info.get('tool', 'unknown'),
                'error': str(e)
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
        
    def _extract_tool_request(self, text: str) -> dict[str, Any] | None:
        """Extract tool name and arguments from text using LLM.
        
        Args:
            text: Text potentially containing a tool request
            
        Returns:
            Dictionary with client, tool, and arguments, or None
        """
        prompt = f"""Extract tool use information from this text:

Text: {text}

If the text requests using a tool, respond with JSON in this format:
{{
  "client": "file-tools",
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
            # Clean response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith('```'):
                response = re.sub(r'```json\s*|\s*```', '', response).strip()
                
            data = json.loads(response)
            if data.get('tool'):
                return data
        except (json.JSONDecodeError, Exception):
            pass
            
        return None
