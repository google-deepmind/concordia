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

"""Tests for the example file reader MCP server."""

import asyncio
import importlib
import importlib.util
import os

from absl.testing import absltest

_MCP_INSTALLED = importlib.util.find_spec('mcp') is not None


@absltest.skipIf(not _MCP_INSTALLED, 'mcp is not installed')
class FileReaderServerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Imported here (not guarded by try/except) so that an incompatible mcp
    # API fails the test instead of skipping it.
    self.server = importlib.import_module(
        'concordia.contrib.tools.mcp.servers.file_reader_server'
    )
    # The server only reads from the working directory (or /tmp).
    self.workdir = self.create_tempdir()
    old_cwd = os.getcwd()
    os.chdir(self.workdir.full_path)
    self.addCleanup(os.chdir, old_cwd)

  def test_registers_tools_with_input_schemas(self):
    tools = asyncio.run(self.server.mcp.list_tools())
    schemas = {tool.name: tool.input_schema for tool in tools}
    self.assertCountEqual(
        schemas, ['read_file', 'list_directory', 'get_file_info']
    )
    self.assertEqual(schemas['read_file']['required'], ['path'])

  def test_read_file_returns_file_contents(self):
    self.workdir.create_file('notes.txt', content='hello from concordia')
    result = asyncio.run(
        self.server.mcp.call_tool('read_file', {'path': 'notes.txt'})
    )
    self.assertEqual(result.content[0].text, 'hello from concordia')

  def test_list_directory_lists_entries(self):
    self.workdir.create_file('a.txt', content='abc')
    result = asyncio.run(
        self.server.mcp.call_tool('list_directory', {'path': '.'})
    )
    self.assertIn('a.txt (3 bytes)', result.content[0].text)


if __name__ == '__main__':
  absltest.main()
