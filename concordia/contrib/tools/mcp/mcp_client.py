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

"""MCP (Model Context Protocol) client for Concordia agents."""

import asyncio
import json
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """Client for connecting to MCP servers and calling tools.
    
    This client allows Concordia's GameMaster to connect to external MCP
    servers and expose their tools to agents in the simulation.
    
    Example usage:
        client = MCPClient(server_command="python", server_args=["my_server.py"])
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})
        await client.disconnect()
    """
    
    def __init__(
        self,
        server_command: str,
        server_args: Optional[List[str]] = None,
        server_env: Optional[Dict[str, str]] = None,
    ):
        """Initialize MCP client.
        
        Args:
            server_command: Command to start the MCP server (e.g., "python")
            server_args: Arguments for the server command (e.g., ["server.py"])
            server_env: Environment variables for the server process
        """
        self.server_params = StdioServerParameters(
            command=server_command,
            args=server_args or [],
            env=server_env,
        )
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._connected = False
        
    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self._connected:
            return
            
        self._exit_stack = AsyncExitStack()
        
        # Start stdio transport
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(self.server_params)
        )
        stdio, write = stdio_transport
        
        # Create session
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        
        # Initialize connection
        await self._session.initialize()
        self._connected = True
        
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if not self._connected:
            return
            
        if self._exit_stack:
            await self._exit_stack.aclose()
        
        self._session = None
        self._exit_stack = None
        self._connected = False
        
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from the MCP server.
        
        Returns:
            List of tool definitions with name, description, and input schema.
        """
        if not self._connected or not self._session:
            raise RuntimeError("Client not connected. Call connect() first.")
            
        response = await self._session.list_tools()
        
        tools = []
        for tool in response.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            })
        return tools
        
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments for the tool
            
        Returns:
            String representation of the tool's result
        """
        if not self._connected or not self._session:
            raise RuntimeError("Client not connected. Call connect() first.")
            
        result = await self._session.call_tool(tool_name, arguments)
        
        # Format result as string
        content_parts = []
        for item in result.content:
            if hasattr(item, 'text'):
                content_parts.append(item.text)
            else:
                content_parts.append(str(item))
                
        return "\n".join(content_parts)
        
    def is_connected(self) -> bool:
        """Check if client is connected to server."""
        return self._connected


class MCPClientManager:
    """Manager for multiple MCP clients.
    
    Allows GameMaster to work with multiple MCP servers simultaneously.
    """
    
    def __init__(self):
        """Initialize the manager."""
        self._clients: Dict[str, MCPClient] = {}
        
    def add_client(self, name: str, client: MCPClient) -> None:
        """Add a named MCP client.
        
        Args:
            name: Unique name for this client
            client: MCPClient instance
        """
        self._clients[name] = client
        
    async def connect_all(self) -> None:
        """Connect all registered clients."""
        for client in self._clients.values():
            await client.connect()
            
    async def disconnect_all(self) -> None:
        """Disconnect all registered clients."""
        for client in self._clients.values():
            await client.disconnect()
            
    async def list_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """List tools from all connected clients.
        
        Returns:
            Dictionary mapping client name to list of tools
        """
        all_tools = {}
        for name, client in self._clients.items():
            if client.is_connected():
                all_tools[name] = await client.list_tools()
        return all_tools
        
    async def call_tool(
        self,
        client_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """Call a tool from a specific client.
        
        Args:
            client_name: Name of the client to use
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result as string
        """
        if client_name not in self._clients:
            raise ValueError(f"No client named '{client_name}'")
            
        client = self._clients[client_name]
        return await client.call_tool(tool_name, arguments)
        
    def get_client(self, name: str) -> Optional[MCPClient]:
        """Get a client by name."""
        return self._clients.get(name)


def run_sync(coro):
    """Helper to run async code synchronously.
    
    Useful for integrating MCP client into Concordia's synchronous flow.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)
