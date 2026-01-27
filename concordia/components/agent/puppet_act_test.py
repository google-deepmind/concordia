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

"""Tests for puppet_act component."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.components.agent import puppet_act
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib


class PuppetActComponentTest(parameterized.TestCase):
  """Tests for PuppetActComponent."""

  def setUp(self):
    super().setUp()
    self._model = no_language_model.NoLanguageModel()
    self._mock_entity = mock.MagicMock()
    self._mock_entity.name = "Alice"

  def _create_component(self, fixed_responses):
    """Creates a PuppetActComponent and attaches it to a mock entity."""
    component = puppet_act.PuppetActComponent(
        model=self._model,
        fixed_responses=fixed_responses,
    )
    component.set_entity(self._mock_entity)
    return component

  def test_returns_fixed_response_exact_match(self):
    """Tests that exact CTA matches return the fixed response."""
    component = self._create_component({
        "What does Alice say?": "Hello world!",
    })

    action_spec = entity_lib.ActionSpec(
        call_to_action="What does Alice say?",
        output_type=entity_lib.OutputType.FREE,
    )

    result = component.get_action_attempt(contexts={}, action_spec=action_spec)
    self.assertEqual(result, "Hello world!")

  def test_returns_fixed_response_with_name_placeholder(self):
    """Tests that CTA with {name} placeholder is formatted and matched."""
    component = self._create_component({
        "What does {name} say?": "I am a puppet!",
    })

    action_spec = entity_lib.ActionSpec(
        call_to_action="What does {name} say?",
        output_type=entity_lib.OutputType.FREE,
    )

    result = component.get_action_attempt(contexts={}, action_spec=action_spec)
    self.assertEqual(result, "I am a puppet!")

  def test_returns_fixed_response_formatted_cta_match(self):
    """Tests matching when CTA is formatted with entity name."""
    component = self._create_component({
        "Where would {name} go?": "The Red Lion",
    })

    action_spec = entity_lib.ActionSpec(
        call_to_action="Where would Alice go?",
        output_type=entity_lib.OutputType.FREE,
    )

    result = component.get_action_attempt(contexts={}, action_spec=action_spec)
    self.assertEqual(result, "The Red Lion")

  def test_no_match_falls_back_to_empty_for_no_language_model(self):
    """Tests that unmatched CTAs with NoLanguageModel return entity prefix."""
    component = self._create_component({
        "Some other question": "Some answer",
    })

    action_spec = entity_lib.ActionSpec(
        call_to_action="Unmatched question",
        output_type=entity_lib.OutputType.FREE,
    )

    result = component.get_action_attempt(contexts={}, action_spec=action_spec)
    self.assertTrue(result.startswith("Alice "))

  @parameterized.parameters(
      ("pub_a", ["pub_a", "pub_b", "pub_c"], "pub_a"),
      ("pub_b", ["pub_a", "pub_b", "pub_c"], "pub_b"),
  )
  def test_fixed_response_with_choice_output_type(
      self, fixed_answer, options, expected
  ):
    """Tests fixed responses don't apply to CHOICE output type directly."""
    component = self._create_component({
        "Which pub?": fixed_answer,
    })

    action_spec = entity_lib.ActionSpec(
        call_to_action="Which pub?",
        output_type=entity_lib.OutputType.CHOICE,
        options=options,
    )

    result = component.get_action_attempt(contexts={}, action_spec=action_spec)
    self.assertEqual(result, expected)

  def test_get_state_returns_responses(self):
    """Tests that get_state returns the fixed responses."""
    responses = {"question": "answer"}
    component = self._create_component(responses)

    state = component.get_state()
    self.assertEqual(state, {"responses": responses})

  def test_multiple_fixed_responses(self):
    """Tests multiple fixed responses work correctly."""
    component = self._create_component({
        "Question 1": "Answer 1",
        "Question 2": "Answer 2",
        "Where would {name} go?": "The Anchor",
    })

    action_spec_1 = entity_lib.ActionSpec(
        call_to_action="Question 1",
        output_type=entity_lib.OutputType.FREE,
    )
    action_spec_2 = entity_lib.ActionSpec(
        call_to_action="Question 2",
        output_type=entity_lib.OutputType.FREE,
    )
    action_spec_3 = entity_lib.ActionSpec(
        call_to_action="Where would Alice go?",
        output_type=entity_lib.OutputType.FREE,
    )

    self.assertEqual(
        component.get_action_attempt(contexts={}, action_spec=action_spec_1),
        "Answer 1",
    )
    self.assertEqual(
        component.get_action_attempt(contexts={}, action_spec=action_spec_2),
        "Answer 2",
    )
    self.assertEqual(
        component.get_action_attempt(contexts={}, action_spec=action_spec_3),
        "The Anchor",
    )


if __name__ == "__main__":
  absltest.main()
