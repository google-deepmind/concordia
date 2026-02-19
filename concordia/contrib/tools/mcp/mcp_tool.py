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

"""MCP tool wrapper that implements Concordia's Tool interface."""

from typing import Any

from concordia.document import tool
from concordia.contrib.tools.mcp import mcp_client


class MCPTool(tool.Tool):
    """Wrapper for MCP tools that implements Concordia's Tool interface.
    
    This allows MCP tools to be used anywhere Concordia's Tool abstraction
    is expected, providing a unified interface for all tool types.
    
    Example:
```python
        # Create MCP client
        client = mcp_client.MCPClient(
            server_command='python',
            server_args=['server.py']
        )
        mcp_client.run_sync(client.connect())
        
        # Wrap as Concordia Tool
        read_file_tool = MCPTool(
            client=client,
            tool_name='read_file',
            tool_description='Read contents of a file'
        )
        
        # Use like any other Tool
        result = read_file_tool.execute({'path': '/tmp/file.txt'})
```
    """
    
    def __init__(
        self,
        client: mcp_client.MCPClient,
        tool_name: str,
        tool_description: str,
    ):
        """Initialize MCPTool wrapper.
        
        Args:
            client: Connected MCP client instance
            tool_name: Name of the tool on the MCP server
            tool_description: Human-readable description of what the tool does
        """
        self._client = client
        self._tool_name = tool_name
        self._tool_description = tool_description
        
    @property
    def name(self) -> str:
        """The unique name of the tool."""
        return self._tool_name
        
    @property
    def description(self) -> str:
        """A description of what the tool does."""
        return self._tool_description
        
    def execute(self, *args: Any, **kwargs: Any) -> str:
        """Execute the MCP tool.
        
        Args:
            *args: Positional arguments (converted to dict for MCP)
            **kwargs: Keyword arguments passed to the MCP tool
            
        Returns:
            Tool execution result as string
        """
        # Convert args to kwargs if provided
        if args:
            # Assume single dict argument if provided
            if len(args) == 1 and isinstance(args[0], dict):
                arguments = args[0]
            else:
                # This shouldn't happen with proper usage
                arguments = kwargs
        else:
            arguments = kwargs
            
        # Execute via MCP client
        result = mcp_client.run_sync(
            self._client.call_tool(self._tool_name, arguments)
        )
        
        return result


def create_mcp_tools_from_client(
    client: mcp_client.MCPClient,
) -> list[MCPTool]:
    """Create MCPTool wrappers for all tools from an MCP client.
    
    Args:
        client: Connected MCP client instance
        
    Returns:
        List of MCPTool instances, one for each tool the client exposes
        
    Example:
```python
        client = mcp_client.MCPClient(...)
        mcp_client.run_sync(client.connect())
        
        # Get all tools as Concordia Tool instances
        tools = create_mcp_tools_from_client(client)
        
        # Use with interactive documents or other Tool-based systems
        for tool in tools:
            print(f"Available: {tool.name} - {tool.description}")
```
    """
    tool_list = mcp_client.run_sync(client.list_tools())
    
    mcp_tools = []
    for tool_info in tool_list:
        mcp_tool = MCPTool(
            client=client,
            tool_name=tool_info['name'],
            tool_description=tool_info['description'],
        )
        mcp_tools.append(mcp_tool)
        
    return mcp_tools
