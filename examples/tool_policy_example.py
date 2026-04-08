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

"""Minimal policy-aware tool runtime example."""

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from concordia.document import interactive_document_tools
from concordia.document import tool
from concordia.document import tool_policy
from concordia.language_model import language_model


class DeterministicModel(language_model.LanguageModel):
  """LanguageModel stub that returns predefined text outputs."""

  def __init__(self, responses: Sequence[str]):
    self._responses = list(responses)

  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    del (
        prompt,
        max_tokens,
        terminators,
        temperature,
        top_p,
        top_k,
        timeout,
        seed,
    )
    if not self._responses:
      raise ValueError('No more predefined model responses.')
    return self._responses.pop(0)

  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    del prompt, seed
    return 0, responses[0], {'strategy': 'deterministic'}


class SensitiveEchoTool(tool.Tool):
  """Example tool with schema and sensitive risk metadata."""

  def __init__(self):
    self.call_count = 0

  @property
  def name(self) -> str:
    return 'sensitive_echo'

  @property
  def description(self) -> str:
    return 'Echoes a message. Args: message (str).'

  @property
  def input_schema(self) -> Mapping[str, Any] | None:
    return {
        'type': 'object',
        'required': ['message'],
        'properties': {'message': {'type': 'string'}},
    }

  @property
  def risk_level(self) -> str:
    return 'sensitive'

  def execute(self, **kwargs: Any) -> str:
    self.call_count += 1
    return f'echo={kwargs.get("message")}'


class RiskAwarePolicy:
  """Blocks tools marked as sensitive or destructive."""

  def evaluate(
      self,
      call: tool_policy.ToolCall,
      available_tools: Mapping[str, tool.Tool],
  ) -> tool_policy.PolicyDecision:
    evaluated_tool = available_tools[call.tool_name]
    if evaluated_tool.risk_level in ('sensitive', 'destructive'):
      return tool_policy.PolicyDecision(
          action=tool_policy.PolicyAction.DENY,
          reason='Risk policy blocks sensitive tool usage.',
      )
    return tool_policy.PolicyDecision(action=tool_policy.PolicyAction.ALLOW)


def _policy_lines(doc: interactive_document_tools.InteractiveDocumentWithTools):
  for content in doc.contents():
    if interactive_document_tools.TOOL_POLICY_TAG in content.tags:
      yield content.text.strip()


def main():
  model = DeterministicModel(
      responses=[
          (
              '{"tool": "sensitive_echo", "args": {"message": 123}}'
          ),  # Invalid schema.
          'First answer after schema block.',
          (
              '{"tool": "sensitive_echo", "args": {"message": "hello"}}'
          ),  # Valid schema, blocked by risk delegate.
          'Second answer after risk block.',
      ]
  )
  tool_instance = SensitiveEchoTool()
  policy = tool_policy.SchemaValidatingPolicy(delegate=RiskAwarePolicy())
  doc = interactive_document_tools.InteractiveDocumentWithTools(
      model=model,
      tools=[tool_instance],
      policy=policy,
      enforcement_mode='enforce',
  )

  first_answer = doc.open_question('Try tool call with malformed args.')
  second_answer = doc.open_question('Try tool call with valid args.')

  print('first_answer:', first_answer)
  print('second_answer:', second_answer)
  print('tool_execute_count:', tool_instance.call_count)
  print('policy_events:')
  for line in _policy_lines(doc):
    print('  -', line)


if __name__ == '__main__':
  main()
