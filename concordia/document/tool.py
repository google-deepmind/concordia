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

"""Tool interface for interactive documents with tool use."""

import abc
from collections.abc import Mapping
from typing import Any


class Tool(abc.ABC):
  """Abstract base class for tools that can be called by an LLM.

  Tools are callable utilities that an LLM can invoke during interactive
  document sessions to gather information or perform actions. Each tool
  has a name, description (for the LLM to understand when to use it),
  and an execute method that performs the actual work.

  Example usage:
    class WebSearchTool(Tool):
      @property
      def name(self) -> str:
        return "web_search"

      @property
      def description(self) -> str:
        return "Search the web. Args: query (str)"

      def execute(self, *, query: str) -> str:
        return search_web(query)
  """

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """The unique name of the tool.

    This name is used by the LLM to reference the tool in tool call requests.
    Should be a valid identifier (lowercase, underscores).
    """
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def description(self) -> str:
    """Human-readable description for the LLM.

    Should explain what the tool does and what arguments it accepts.
    The LLM uses this to decide when and how to invoke the tool.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def execute(self, **kwargs: Any) -> str:
    """Execute the tool with the given arguments.

    Args:
      **kwargs: Tool-specific arguments as keyword arguments.

    Returns:
      A string result that will be shown to the LLM. Results may be
      truncated by the calling code to fit context limits.
    """
    raise NotImplementedError

  @property
  def input_schema(self) -> Mapping[str, Any] | None:
    """Schema describing tool arguments.

    This is optional metadata that policy implementations can use to validate
    arguments before execution.
    """
    return None

  @property
  def risk_level(self) -> str:
    """Risk category for tool execution.

    Recommended values are "read", "sensitive", and "destructive".
    """
    return 'read'
