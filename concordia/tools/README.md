# MCP (Model Context Protocol) Integration for Concordia

This module enables Concordia agents to use external tools via the Model Context Protocol.

## Overview

The MCP integration allows agents in Concordia simulations to:
- Read and access external files
- Query databases
- Make API calls
- Search the web
- Use any tool exposed via an MCP server

## Components

### 1. MCP Client (`mcp_client.py`)

Wrapper for connecting to MCP servers and executing tools.
```python
from concordia.tools import mcp_client

# Create client
client = mcp_client.MCPClient(
    server_command='python',
    server_args=['path/to/server.py']
)

# Connect and use
await client.connect()
tools = await client.list_tools()
result = await client.call_tool('read_file', {'path': '/tmp/file.txt'})
await client.disconnect()
```

### 2. MCP Servers (`mcp_servers/`)

Pre-built MCP servers for common operations:

- **file_reader_server.py**: Read files, list directories, get file info
  - Tools: `read_file`, `list_directory`, `get_file_info`
  - Security: Only allows access to `/tmp` and current working directory

### 3. GameMaster Integration

#### Tool Executor Component (`components/game_master/mcp_tool_executor.py`)

Detects when agents request tools and executes them automatically.

#### Tool-Use GameMaster Prefab (`prefabs/game_master/tool_use_gm.py`)

Complete GameMaster with MCP tool support built-in.
```python
from concordia.prefabs.game_master import tool_use_gm

gm = tool_use_gm.build_tool_use_game_master(
    model=language_model,
    memory=memory_bank,
    mcp_server_command=sys.executable,
    mcp_server_args=['concordia/tools/mcp_servers/file_reader_server.py'],
    clock=clock,
    players=[alice, bob],
)
```

## Quick Start

See `examples/mcp_tool_use_example.ipynb` for a complete working example.

## Creating Custom MCP Servers
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def my_custom_tool(arg: str) -> str:
    """Tool description here."""
    return f"Result: {arg}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## Security Considerations

- MCP servers should validate and sanitize all inputs
- File operations should be restricted to safe directories
- API keys and credentials should be managed securely
- Consider rate limiting for external API calls

## Future Enhancements

- Web search MCP server
- Database query MCP server
- REST API client MCP server
- Multi-modal tool support (images, audio)

## References

- [MCP Specification](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Concordia Documentation](https://github.com/google-deepmind/concordia)
