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

"""Test agent components."""
import datetime
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import all_similar_memories
from concordia.components.agent import concat_act_component
from concordia.components.agent import constant
from concordia.components.agent import instructions
from concordia.components.agent import memory
from concordia.components.agent import observation
from concordia.components.agent import plan
from concordia.components.agent import pubsub
from concordia.components.agent import puppet_act
from concordia.components.agent import question_of_recent_memories
from concordia.components.agent import report_function
from concordia.components.agent import scripted_act
from concordia.components.game_master import next_game_master
from concordia.components.game_master import switch_act
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.utils import helper_functions
import numpy as np

DEFAULT_SKIP_KEYS = {"_model", "_lock"}

embedder = lambda x: np.random.rand(3)
memory_bank = basic_associative_memory.AssociativeMemoryBank(
    sentence_embedder=embedder
)
memory_bank.add("Fake memory")

deep_compare_components = helper_functions.deep_compare_components
COMPONENT_FACTORIES = {
    "plan": {
        "component_class": plan.Plan,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "components": ["comp1_key", "comp2_key"],
            "goal_component_key": "comp1",
            "force_time_horizon": "24 hours",
            "pre_act_label": "Test Plan",
        },
        "state_example": {"current_plan": "test plan"},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "plan_with_defaults": {
        "component_class": plan.Plan,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "components": ["comp1_key", "comp2_key"],
        },
        "state_example": {"current_plan": "test plan"},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "all_similar_memories": {
        "component_class": all_similar_memories.AllSimilarMemories,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "components": ["comp1_key", "comp2_key"],
            "memory_component_key": "comp1",
            "num_memories_to_retrieve": 10,
            "pre_act_label": "Test All Similar Memories",
        },
        "state_example": {},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "constant": {
        "component_class": constant.Constant,
        "kwargs": {
            "state": "test state",
            "pre_act_label": "Test Constant",
        },
        "state_example": {},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "associative_memory": {
        "component_class": memory.AssociativeMemory,
        "kwargs": {
            "memory_bank": memory_bank,
        },
        "state_example": {
            "memory_bank": memory_bank.get_state(),
            "buffer": ["Fake memory"],
        },
        "skip_keys": {"_model", "_lock", "_memory_bank_lock"},
    },
    "report_function": {
        "component_class": report_function.ReportFunction,
        "kwargs": {
            "function": lambda: "test function",
            "pre_act_label": "Test Report Function",
        },
        "state_example": {},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "concat_act_component": {
        "component_class": concat_act_component.ConcatActComponent,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "component_order": ["comp1_key", "comp2_key"],
        },
        "state_example": {},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "instructions": {
        "component_class": instructions.Instructions,
        "kwargs": {
            "agent_name": "test agent",
            "pre_act_label": "Test Instructions",
        },
        "state_example": {},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "observation": {
        "component_class": observation.ObservationsSinceLastPreAct,
        "kwargs": {},
        "state_example": {"num_since_last_pre_act": 10},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "question_of_recent_memories": {
        "component_class": question_of_recent_memories.QuestionOfRecentMemories,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "memory_component_key": "memory",
            "components": ["comp1_key", "comp2_key"],
            "clock_now": lambda: datetime.datetime(2024, 1, 1, 12, 0, 0),
            "pre_act_label": "Test Question",
            "question": "What would {agent_name} do?",
            "answer_prefix": "{agent_name} would ",
            "add_to_memory": False,
        },
        "state_example": {},
        "skip_keys": {"_model", "_lock", "_clock_now"},
    },
    "puppet_act": {
        "component_class": puppet_act.PuppetActComponent,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "fixed_responses": {"test_action": "test_response"},
        },
        "state_example": {},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "scripted_act": {
        "component_class": scripted_act.ScriptedActComponent,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "script": [{"name": "test_agent", "line": "test_line"}],
            "component_order": ["comp1_key"],
        },
        "state_example": {},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "publisher": {
        "component_class": pubsub.Publisher,
        "kwargs": {
            "function": lambda: "published value",
            "pre_act_label": "Test Publisher",
            "skip_if_busy": True,
        },
        "state_example": {
            "latest_value": "cached value",
            "skip_if_busy": False,
        },
        "skip_keys": (
            DEFAULT_SKIP_KEYS
            | {
                "_condition_lock",
                "_publish_lock",
                "_subject",
                "_scheduler",
                "_trigger_subject",
                "_trigger_subscription",
            }
        ),
    },
    "subscriber": {
        "component_class": pubsub.Subscriber,
        "kwargs": {
            "publisher_component_key": "__publisher__",
            "initial_value": "initial",
            "pre_act_label": "Test Subscriber",
        },
        "state_example": {
            "current_value": "updated",
        },
        "skip_keys": (
            DEFAULT_SKIP_KEYS
            | {
                "_value_lock",
                "_subject",
                "_subscription",
            }
        ),
    },
}


class AgentComponentTest(parameterized.TestCase):
  """Tests for agent components."""

  @parameterized.named_parameters(
      dict(testcase_name="plan", component_name="plan"),
      dict(
          testcase_name="plan_with_defaults",
          component_name="plan_with_defaults",
      ),
      dict(
          testcase_name="all_similar_memories",
          component_name="all_similar_memories",
      ),
      dict(
          testcase_name="constant",
          component_name="constant",
      ),
      dict(
          testcase_name="associative_memory",
          component_name="associative_memory",
      ),
      dict(
          testcase_name="report_function",
          component_name="report_function",
      ),
      dict(
          testcase_name="concat_act_component",
          component_name="concat_act_component",
      ),
      dict(
          testcase_name="instructions",
          component_name="instructions",
      ),
      dict(
          testcase_name="observation",
          component_name="observation",
      ),
      dict(
          testcase_name="question_of_recent_memories",
          component_name="question_of_recent_memories",
      ),
      dict(
          testcase_name="puppet_act",
          component_name="puppet_act",
      ),
      dict(
          testcase_name="scripted_act",
          component_name="scripted_act",
      ),
      dict(
          testcase_name="publisher",
          component_name="publisher",
      ),
      dict(
          testcase_name="subscriber",
          component_name="subscriber",
      ),
  )
  def test_get_and_set_state(self, component_name: str):
    """Tests getting and setting the state of a component."""

    # Initialize component A from config
    component_config = COMPONENT_FACTORIES[component_name]
    component_class = component_config["component_class"]
    kwargs = component_config["kwargs"]
    state_example = component_config["state_example"]
    skip_keys = component_config["skip_keys"]

    component_a = component_class(**kwargs)  # pyrefly: ignore[bad-argument-type]
    component_a.set_state(state_example)
    state_a = component_a.get_state()

    # Initialize component B, then set the state to the state of component A
    component_b = component_class(**kwargs)  # pyrefly: ignore[bad-argument-type]
    component_b.set_state(state_a)
    state_b = component_b.get_state()

    # Simple check for the state firsts
    self.assertEqual(state_a, state_b)

    # Deeper check for the entire component
    deep_compare_components(component_a, component_b, self, skip_keys)


def _in_pre_act_phase(
    component: action_spec_ignored.ActionSpecIgnored,
) -> action_spec_ignored.ActionSpecIgnored:
  """Attaches `component` to a fake entity that is in the `PRE_ACT` phase."""
  entity = mock.MagicMock()
  entity.get_phase.return_value = entity_component.Phase.PRE_ACT
  component.set_entity(entity)
  return component


class ActionSpecIgnoredPreActTest(absltest.TestCase):
  """Tests how `ActionSpecIgnored` formats the context `pre_act` returns."""

  def test_pre_act_prefixes_the_value_with_its_label_by_default(self):
    component = _in_pre_act_phase(constant.Constant("rules", "Next"))

    self.assertEqual(
        component.pre_act(entity_lib.DEFAULT_ACTION_SPEC), "Next:\nrules\n"
    )

  def test_pre_act_returns_the_bare_value_when_asked_not_to_prefix(self):
    component = _in_pre_act_phase(
        constant.Constant("rules", "Next", prefix_pre_act=False)
    )

    self.assertEqual(component.pre_act(entity_lib.DEFAULT_ACTION_SPEC), "rules")
    self.assertEqual(component.get_pre_act_label(), "Next")
    self.assertEqual(component.get_pre_act_value(), "rules")

  def test_derived_classes_may_override_the_format(self):

    class Bracketed(action_spec_ignored.ActionSpecIgnored):

      def _make_pre_act_value(self) -> str:
        return "value"

      def _format_pre_act(self, label: str, value: str) -> str:
        return f"[{label}] {value}"

    component = _in_pre_act_phase(Bracketed("label"))

    self.assertEqual(
        component.pre_act(entity_lib.DEFAULT_ACTION_SPEC), "[label] value"
    )

  def test_unprefixed_constant_names_the_next_game_master(self):
    model = no_language_model.NoLanguageModel()
    game_master = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name="rules",
        act_component=switch_act.SwitchAct(model=model, entity_names=()),
        context_components={
            next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY: (
                constant.Constant("rules", prefix_pre_act=False)
            ),
        },
    )

    chosen = game_master.act(
        entity_lib.ActionSpec(
            call_to_action=next_game_master.DEFAULT_CALL_TO_NEXT_GAME_MASTER,
            output_type=entity_lib.OutputType.NEXT_GAME_MASTER,
            options=("setup", "rules"),
        )
    )

    self.assertEqual(chosen, "rules")


if __name__ == "__main__":
  absltest.main()
