# Copyright 2026 DeepMind Technologies Limited.
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

"""Tests for the MCP client."""

import asyncio
import importlib
import importlib.util
import os
import sys
from unittest import mock

from absl.testing import absltest

_MCP_INSTALLED = importlib.util.find_spec('mcp') is not None

_SCHEMA = {
    'type': 'object',
    'properties': {'path': {'type': 'string'}},
    'required': ['path'],
}


@absltest.skipIf(not _MCP_INSTALLED, 'mcp is not installed')
class MCPClientTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Imported here (not guarded by try/except) so that an incompatible mcp
    # API fails the test instead of skipping it.
    self.mcp_client = importlib.import_module(
        'concordia.contrib.tools.mcp.mcp_client'
    )
    self.types = importlib.import_module('mcp.types')

  def _connected_client(self, session):
    client = self.mcp_client.MCPClient(server_command='unused')
    client._session = session  # pylint: disable=protected-access
    client._connected = True  # pylint: disable=protected-access
    return client

  def test_list_tools_maps_tool_definitions(self):
    session = mock.Mock()
    session.list_tools = mock.AsyncMock(
        return_value=self.types.ListToolsResult(
            tools=[
                self.types.Tool(
                    name='read_file',
                    description='Read a file.',
                    input_schema=_SCHEMA,
                ),
                self.types.Tool(name='no_description', input_schema=_SCHEMA),
            ]
        )
    )
    client = self._connected_client(session)

    tools = asyncio.run(client.list_tools())

    self.assertEqual(
        tools,
        [
            {
                'name': 'read_file',
                'description': 'Read a file.',
                'input_schema': _SCHEMA,
            },
            {
                'name': 'no_description',
                'description': '',
                'input_schema': _SCHEMA,
            },
        ],
    )

  def test_call_tool_joins_text_content(self):
    session = mock.Mock()
    session.call_tool = mock.AsyncMock(
        return_value=self.types.CallToolResult(
            content=[
                self.types.TextContent(type='text', text='first'),
                self.types.TextContent(type='text', text='second'),
            ]
        )
    )
    client = self._connected_client(session)

    result = asyncio.run(client.call_tool('read_file', {'path': 'x'}))

    self.assertEqual(result, 'first\nsecond')
    session.call_tool.assert_awaited_once_with('read_file', {'path': 'x'})

  def test_round_trip_with_file_reader_server(self):
    server_path = os.path.join(
        os.path.dirname(__file__), 'servers', 'file_reader_server.py'
    )
    workdir = self.create_tempdir()
    workdir.create_file('notes.txt', content='hello over stdio')
    old_cwd = os.getcwd()
    os.chdir(workdir.full_path)
    self.addCleanup(os.chdir, old_cwd)

    async def round_trip():
      client = self.mcp_client.MCPClient(
          server_command=sys.executable, server_args=[server_path]
      )
      await client.connect()
      try:
        tools = await client.list_tools()
        text = await client.call_tool('read_file', {'path': 'notes.txt'})
      finally:
        await client.disconnect()
      return tools, text

    tools, text = asyncio.run(round_trip())

    tools_by_name = {tool['name']: tool for tool in tools}
    self.assertCountEqual(
        tools_by_name, ['read_file', 'list_directory', 'get_file_info']
    )
    self.assertEqual(
        tools_by_name['read_file']['input_schema']['required'], ['path']
    )
    self.assertEqual(text, 'hello over stdio')


if __name__ == '__main__':
  absltest.main()
