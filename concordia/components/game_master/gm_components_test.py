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

"""Test game master components."""
import datetime
from absl.testing import absltest
from absl.testing import parameterized
from concordia.components.game_master import event_resolution
from concordia.components.game_master import inventory
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting
from concordia.components.game_master import next_game_master
from concordia.components.game_master import payoff_matrix
from concordia.components.game_master import questionnaire
from concordia.components.game_master import scene_tracker
from concordia.components.game_master import switch_act
from concordia.components.game_master import terminate
from concordia.components.game_master import world_state
from concordia.language_model import no_language_model
from concordia.utils import helper_functions
import numpy as np

DEFAULT_SKIP_KEYS = {"_model", "_lock"}

embedder = lambda x: np.random.rand(3)

deep_compare_components = helper_functions.deep_compare_components
COMPONENT_FACTORIES = {
    "event_resolution": {
        "component_class": event_resolution.EventResolution,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "event_resolution_steps": None,
            "components": [],
            "notify_observers": False,
        },
        "state_example": {
            "_active_entity_name": "test_name",
            "_putative_action": "test_action",
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "display_events": {
        "component_class": event_resolution.DisplayEvents,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
        },
        "state_example": {},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "send_event_to_relevant_players": {
        "component_class": event_resolution.SendEventToRelevantPlayers,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "player_names": ["Alice", "Bob"],
            "make_observation_component_key": "make_observation",
        },
        "state_example": {
            "_queue": {"fake_queue": []},
            "_last_action_spec": "test_action_spec",
            "_map_names_to_previous_observations": {
                "Alice": "fake observation",
            },
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "next_acting": {
        "component_class": next_acting.NextActing,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "player_names": ["Alice", "Bob"],
            "components": [],
            "pre_act_label": "Test Next Acting",
        },
        "state_example": {
            "currently_active_player": "Alice",
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "next_acting_in_fixed_order": {
        "component_class": next_acting.NextActingInFixedOrder,
        "kwargs": {
            "sequence": ["Alice", "Bob"],
            "pre_act_label": "Test Next Acting",
        },
        "state_example": {
            "currently_active_player_idx": 0,
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "next_acting_in_random_order": {
        "component_class": next_acting.NextActingInRandomOrder,
        "kwargs": {
            "player_names": ["Alice", "Bob"],
            "replace": False,
            "pre_act_label": "Test Next Acting",
        },
        "state_example": {
            "currently_active_player_idx": 0,
            "currently_available_indices": [0, 1, 2],
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "next_acting_from_scene_spec": {
        "component_class": next_acting.NextActingFromSceneSpec,
        "kwargs": {
            "memory_component_key": "memory",
            "scene_tracker_component_key": "scene_tracker",
            "pre_act_label": "Test Next Acting",
        },
        "state_example": {
            "currently_active_player": "Alice",
            "counter": 0,
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "next_action_spec": {
        "component_class": next_acting.NextActionSpec,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "player_names": ["Alice", "Bob"],
            "components": [],
        },
        "state_example": {},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "next_action_spec_from_spec_scene": {
        "component_class": next_acting.NextActionSpecFromSceneSpec,
        "kwargs": {},
        "state_example": {},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "fixed_action_spec": {
        "component_class": next_acting.FixedActionSpec,
        "kwargs": {
            "action_spec": "test_action_spec",
            "pre_act_label": "Test Fixed Action Spec",
        },
        "state_example": {},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "inventory": {
        "component_class": inventory.Inventory,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "item_type_configs": [],
            "player_initial_endowments": {},
            "clock_now": lambda: datetime.datetime.now,
            "observations_component_name": "observations",
            "memory_component_name": "memory",
            "chain_of_thought_prefix": "Test Inventory",
            "financial": False,
            "never_increase": False,
        },
        "state_example": {
            "inventories": {
                "test_item": 1,
            },
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "score": {
        "component_class": inventory.Score,
        "kwargs": {
            "inventory": inventory.Inventory(
                model=no_language_model.NoLanguageModel(),
                item_type_configs=[],
                player_initial_endowments={},
                clock_now=lambda: datetime.datetime.now,
            ),
            "player_names": ["Alice", "Bob"],
            "targets": {},
        },
        "state_example": {
            "inventory": {
                "inventories": {
                    "test_item": 1,
                },
            },
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "make_observation": {
        "component_class": make_observation.MakeObservation,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "player_names": ["Alice", "Bob"],
            "components": [],
            "pre_act_label": "Test Make Observation",
        },
        "state_example": {
            "queue": {"fake_queue": []},
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "next_game_master": {
        "component_class": next_game_master.NextGameMaster,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "map_game_master_names_to_choices": {
                "test_game_master": "test_choice",
            },
        },
        "state_example": {
            "currently_active_game_master": "test_game_master",
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "formative_memories_initializer": {
        "component_class": next_game_master.FormativeMemoriesInitializer,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "next_game_master_name": "test_game_master",
            "player_names": ["Alice", "Bob"],
            "pre_act_label": "Test Formative Memories Initializer",
        },
        "state_example": {"initialized": False},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "payoff_matrix": {
        "component_class": payoff_matrix.PayoffMatrix,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "acting_player_names": ["Alice", "Bob"],
            "action_to_scores": lambda x: x,
            "scores_to_observation": lambda x: x,
        },
        "state_example": {
            "stage_idx": 0,
            "partial_joint_action": "test_action",
            "player_scores": {
                "Alice": 0.0,
                "Bob": 0.0,
            },
            "history": [],
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "questionnaire": {
        "component_class": questionnaire.Questionnaire,
        "kwargs": {"questionnaires": {}},
        "state_example": {
            "questionnaire_idx": 0,
            "question_idx": -1,
            "answers": {
                "Alice": {"fake_questionnaire": {}},
            },
            "last_observation": None,
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "scene_tracker": {
        "component_class": scene_tracker.SceneTracker,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "scenes": [],
        },
        "state_example": {
            "round_idx_to_scene": {},
            "max_rounds": 10,
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "switch_act": {
        "component_class": switch_act.SwitchAct,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "entity_names": ["Alice", "Bob"],
        },
        "state_example": {},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "terminate": {
        "component_class": terminate.Terminate,
        "kwargs": {},
        "state_example": {"terminate_now": False},
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "world_state": {
        "component_class": world_state.WorldState,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
        },
        "state_example": {
            "state": "",
            "latest_action_spec": "",
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "locations": {
        "component_class": world_state.Locations,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "entity_names": ["Alice", "Bob"],
            "prompt": "Test Locations",
        },
        "state_example": {
            "locations": {},
            "entity_locations": {},
            "latest_action_spec": "",
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
    "generative_clock": {
        "component_class": world_state.GenerativeClock,
        "kwargs": {
            "model": no_language_model.NoLanguageModel(),
            "prompt": "Test Generative Clock",
            "start_time": "12PM",
        },
        "state_example": {
            "num_steps": 8,
            "time": "8PM",
            "prompt_to_log": "",
            "latest_action_spec": "",
        },
        "skip_keys": DEFAULT_SKIP_KEYS,
    },
}


class GMComponentTest(parameterized.TestCase):
  """Tests for game master components."""

  @parameterized.named_parameters(
      dict(testcase_name="event_resolution", component_name="event_resolution"),
      dict(
          testcase_name="display_events",
          component_name="display_events",
      ),
      dict(
          testcase_name="send_event_to_relevant_players",
          component_name="send_event_to_relevant_players",
      ),
      dict(
          testcase_name="next_acting",
          component_name="next_acting",
      ),
      dict(
          testcase_name="next_acting_in_fixed_order",
          component_name="next_acting_in_fixed_order",
      ),
      dict(
          testcase_name="next_acting_in_random_order",
          component_name="next_acting_in_random_order",
      ),
      dict(
          testcase_name="next_acting_from_scene_spec",
          component_name="next_acting_from_scene_spec",
      ),
      dict(
          testcase_name="next_action_spec",
          component_name="next_action_spec",
      ),
      dict(
          testcase_name="next_action_spec_from_spec_scene",
          component_name="next_action_spec_from_spec_scene",
      ),
      dict(
          testcase_name="fixed_action_spec",
          component_name="fixed_action_spec",
      ),
      dict(
          testcase_name="inventory",
          component_name="inventory",
      ),
      dict(
          testcase_name="score",
          component_name="score",
      ),
      dict(
          testcase_name="make_observation",
          component_name="make_observation",
      ),
      dict(
          testcase_name="next_game_master",
          component_name="next_game_master",
      ),
      dict(
          testcase_name="formative_memories_initializer",
          component_name="formative_memories_initializer",
      ),
      dict(
          testcase_name="payoff_matrix",
          component_name="payoff_matrix",
      ),
      dict(
          testcase_name="questionnaire",
          component_name="questionnaire",
      ),
      dict(
          testcase_name="scene_tracker",
          component_name="scene_tracker",
      ),
      dict(
          testcase_name="switch_act",
          component_name="switch_act",
      ),
      dict(
          testcase_name="terminate",
          component_name="terminate",
      ),
      dict(
          testcase_name="world_state",
          component_name="world_state",
      ),
      dict(
          testcase_name="locations",
          component_name="locations",
      ),
      dict(
          testcase_name="generative_clock",
          component_name="generative_clock",
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
