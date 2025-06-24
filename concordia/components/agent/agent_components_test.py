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
from absl.testing import absltest
from absl.testing import parameterized
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import all_similar_memories
from concordia.components.agent import concat_act_component
from concordia.components.agent import constant
from concordia.components.agent import instructions
from concordia.components.agent import memory
from concordia.components.agent import observation
from concordia.components.agent import plan
from concordia.components.agent import report_function
from concordia.language_model import no_language_model
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
}


class AgentComponentTest(parameterized.TestCase):
  """Tests for agent components."""

  @parameterized.named_parameters(
      dict(testcase_name="plan", component_name="plan"),
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
  )
  def test_get_and_set_state(self, component_name: str):
    """Tests getting and setting the state of a component."""

    # Initialize component A from config
    component_config = COMPONENT_FACTORIES[component_name]
    component_class = component_config["component_class"]
    kwargs = component_config["kwargs"]
    state_example = component_config["state_example"]
    skip_keys = component_config["skip_keys"]

    component_a = component_class(**kwargs)
    component_a.set_state(state_example)
    state_a = component_a.get_state()

    # Initialize component B, then set the state to the state of component A
    component_b = component_class(**kwargs)
    component_b.set_state(state_a)
    state_b = component_b.get_state()

    # Simple check for the state firsts
    self.assertEqual(state_a, state_b)

    # Deeper check for the entire component
    deep_compare_components(component_a, component_b, self, skip_keys)


if __name__ == "__main__":
  absltest.main()
