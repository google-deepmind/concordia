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

"""Unit tests for Dialogic Game Master conversation termination and cleanup."""

from typing import Any

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.language_model import language_model
from concordia.typing import entity_component

from examples.concordia_island.prefabs import island_gm
from examples.concordia_island.sim import conversation as async_conv


class _MockActComponent(entity_component.ContextComponent):

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass


class _MockLanguageModel(language_model.LanguageModel):

  def __init__(self, sample_choice: int = 0):
    self._sample_choice = sample_choice
    self.queries: list[str] = []

  def sample_text(self, prompt: str, **kwargs) -> str:
    self.queries.append(prompt)
    return "Yes, the conversation is over"

  def sample_choice(
      self, prompt: str, responses: list[str], **kwargs
  ) -> tuple[int, str, dict[str, Any]]:
    self.queries.append(prompt)
    idx = self._sample_choice
    return idx, responses[idx], {}


class ConversationTerminationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.player_names = ["Jon Huang", "Tara Lee"]
    self.conv_state = async_conv.AsyncConversationState(
        player_names=self.player_names,
        max_turns=16,
    )

  def test_end_all_conversations(self):
    conv_id = self.conv_state.create_conversation(
        participants=("Jon Huang", "Tara Lee"),
        location="sunset_cafe",
        max_turns=16,
    )
    self.assertTrue(self.conv_state.is_in_conversation("Jon Huang"))
    self.assertTrue(self.conv_state.is_in_conversation("Tara Lee"))

    self.conv_state.end_all_conversations(reason="Day Transition Test")
    self.assertFalse(self.conv_state.is_in_conversation("Jon Huang"))
    self.assertFalse(self.conv_state.is_in_conversation("Tara Lee"))
    conv = self.conv_state._conversations[conv_id]
    self.assertFalse(conv.active)

  def test_dialogic_gm_termination_with_llm_yes(self):
    mock_model = _MockLanguageModel(sample_choice=0)  # 0 = Yes, over
    resolver = async_conv.AsyncConversationResolution(
        player_names=self.player_names,
        model=mock_model,
        terminate_boring=True,
    )

    entity_agent_with_logging.EntityAgentWithLogging(
        agent_name="GM",
        act_component=_MockActComponent(),
        context_components={
            async_conv.DEFAULT_ASYNC_CONVERSATION_KEY: self.conv_state,
            "resolution": resolver,
        },
    )

    self.conv_state.create_conversation(
        participants=("Jon Huang", "Tara Lee"),
        location="sunset_cafe",
        max_turns=16,
        terminate_check_min_turn=2,
    )

    self.conv_state.add_utterance("Jon Huang", "Hello Tara.")
    self.conv_state.add_utterance("Tara Lee", "I am leaving now, goodbye.")

    conv = self.conv_state.get_conversation_for("Tara Lee")
    should_term = resolver._should_terminate_conversation(conv)
    self.assertTrue(should_term)

  def test_dialogic_gm_termination_with_llm_no(self):
    mock_model = _MockLanguageModel(sample_choice=1)  # 1 = No, continue
    resolver = async_conv.AsyncConversationResolution(
        player_names=self.player_names,
        model=mock_model,
        terminate_boring=True,
    )

    entity_agent_with_logging.EntityAgentWithLogging(
        agent_name="GM",
        act_component=_MockActComponent(),
        context_components={
            async_conv.DEFAULT_ASYNC_CONVERSATION_KEY: self.conv_state,
            "resolution": resolver,
        },
    )

    self.conv_state.create_conversation(
        participants=("Jon Huang", "Tara Lee"),
        location="sunset_cafe",
        max_turns=16,
        terminate_check_min_turn=4,
    )

    self.conv_state.add_utterance("Jon Huang", "How is your coffee?")
    self.conv_state.add_utterance("Tara Lee", "It is very good, thanks.")
    self.conv_state.add_utterance("Jon Huang", "The weather is lovely today.")
    self.conv_state.add_utterance("Tara Lee", "Yes, sunny and crisp.")

    conv = self.conv_state.get_conversation_for("Tara Lee")
    should_term = resolver._should_terminate_conversation(conv)
    self.assertFalse(should_term)

  def test_island_event_resolution_dialogic_check(self):
    mock_model = _MockLanguageModel(sample_choice=0)
    resolver = island_gm.IslandEventResolution(
        model=mock_model,
        player_names=self.player_names,
        terminate_boring=True,
    )

    entity_agent_with_logging.EntityAgentWithLogging(
        agent_name="GM",
        act_component=_MockActComponent(),
        context_components={
            async_conv.DEFAULT_ASYNC_CONVERSATION_KEY: self.conv_state,
            "resolution": resolver,
        },
    )

    self.conv_state.create_conversation(
        participants=("Jon Huang", "Tara Lee"),
        location="sunset_cafe",
        max_turns=16,
        terminate_check_min_turn=2,
    )

    self.conv_state.add_utterance("Jon Huang", "I will head home now.")
    self.conv_state.add_utterance("Tara Lee", "Good night, Jon.")

    conv = self.conv_state.get_conversation_for("Jon Huang")
    self.assertTrue(resolver._should_terminate_conversation(conv))


if __name__ == "__main__":
  absltest.main()
