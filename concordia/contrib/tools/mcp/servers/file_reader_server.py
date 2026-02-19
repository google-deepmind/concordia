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

"""Simple MCP server for file reading operations.

This server provides basic file system tools that Concordia agents can use
to read files and list directories during simulations.
"""

import os
import pathlib
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("file-reader")


@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a text file.
    
    Args:
        path: Path to the file to read (absolute or relative)
        
    Returns:
        Contents of the file as a string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be read
    """
    file_path = pathlib.Path(path).expanduser().resolve()
    
    # Security: only allow reading from /tmp or current working directory
    allowed_dirs = [
        pathlib.Path("/tmp").resolve(),
        pathlib.Path.cwd().resolve(),
    ]
    
    if not any(str(file_path).startswith(str(d)) for d in allowed_dirs):
        raise PermissionError(
            f"Access denied. Can only read from: {', '.join(str(d) for d in allowed_dirs)}"
        )
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


@mcp.tool()
def list_directory(path: str, pattern: Optional[str] = None) -> str:
    """List contents of a directory.
    
    Args:
        path: Path to the directory
        pattern: Optional glob pattern to filter files (e.g., "*.txt")
        
    Returns:
        Formatted list of files and directories
    """
    dir_path = pathlib.Path(path).expanduser().resolve()
    
    # Security: only allow listing from /tmp or current working directory
    allowed_dirs = [
        pathlib.Path("/tmp").resolve(),
        pathlib.Path.cwd().resolve(),
    ]
    
    if not any(str(dir_path).startswith(str(d)) for d in allowed_dirs):
        raise PermissionError(
            f"Access denied. Can only list: {', '.join(str(d) for d in allowed_dirs)}"
        )
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
        
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    # Get entries
    if pattern:
        entries = sorted(dir_path.glob(pattern))
    else:
        entries = sorted(dir_path.iterdir())
    
    # Format output
    lines = [f"Contents of {dir_path}:"]
    for entry in entries:
        type_marker = "/" if entry.is_dir() else ""
        size = entry.stat().st_size if entry.is_file() else 0
        lines.append(f"  {entry.name}{type_marker} ({size} bytes)")
    
    return "\n".join(lines)


@mcp.tool()
def get_file_info(path: str) -> str:
    """Get information about a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Formatted file information (size, type, modification time)
    """
    file_path = pathlib.Path(path).expanduser().resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    stat = file_path.stat()
    
    info_lines = [
        f"File: {file_path.name}",
        f"Path: {file_path}",
        f"Type: {'Directory' if file_path.is_dir() else 'File'}",
        f"Size: {stat.st_size} bytes",
        f"Modified: {stat.st_mtime}",
    ]
    
    return "\n".join(info_lines)


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")
