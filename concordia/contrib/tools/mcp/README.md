# MCP (Model Context Protocol) Integration for Concordia

**Note:** This is a contrib module. It wraps MCP tools to work with Concordia's existing `Tool` abstraction from `concordia/document/tool.py`.

## Quick Start
```python
from concordia.contrib.tools.mcp import mcp_client, mcp_tool

# Create and connect MCP client
client = mcp_client.MCPClient(
    server_command='python',
    server_args=['path/to/mcp_server.py']
)
mcp_client.run_sync(client.connect())

# Get all tools as Concordia Tool instances
tools = mcp_tool.create_mcp_tools_from_client(client)

# Use with interactive documents
for tool in tools:
    result = tool.execute({'arg': 'value'})
```

## Architecture

This module wraps MCP tools to implement Concordia's `Tool` interface, allowing seamless integration with existing tool-based systems like interactive documents.

**Key files:**
- `mcp_client.py` - MCP protocol client
- `mcp_tool.py` - Wrapper implementing `concordia.document.tool.Tool`
- `servers/` - Example MCP servers

## Example Servers

### File Reader Server
```bash
python concordia/contrib/tools/mcp/servers/file_reader_server.py
```

Tools: `read_file`, `list_directory`, `get_file_info`

## Creating Custom Servers

See `servers/file_reader_server.py` for an example.

## References

- [MCP Specification](https://modelcontextprotocol.io/)
- [Concordia Tool Interface](../../../document/tool.py)
