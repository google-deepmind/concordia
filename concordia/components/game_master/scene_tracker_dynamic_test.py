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

"""Literal dynamic state editing preserves the standard tracker and memory cursor."""

import copy
import datetime
import json
from unittest import mock

from concordia.components.agent import memory as memory_lib
from concordia.components.game_master import scene_tracker
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
from concordia.typing import scene as scene_lib
from concordia.utils import project_test_support as fixtures
import pytest


@pytest.fixture(autouse=True)
def no_execution():
  with mock.patch(
      'concordia.prefabs.simulation.generic.Simulation.play',
      side_effect=AssertionError('No simulation'),
  ), mock.patch.object(
      no_language_model.NoLanguageModel,
      'sample_text',
      side_effect=AssertionError('No model'),
  ):
    yield


def tracker():
  spec = scene_lib.SceneSpec(
      scene_type=scene_lib.SceneTypeSpec(
          name='Discussion',
          game_master_name='Conversation',
          default_premise={'Alice': ['Welcome'], 'Bob': []},
          action_spec=entity_lib.DEFAULT_ACTION_SPEC,
          possible_participants=None,
      ),
      participants=['Alice', 'Bob'],
      num_rounds=4,
      start_time=datetime.datetime(2026, 1, 1),
  )
  return scene_tracker.SceneTracker(no_language_model.NoLanguageModel(), [spec])


def test_literal_roundtrip_partial_update_and_atomic_validation():
  component = tracker()
  original = component.get_state()
  component.set_state(original)
  assert component.get_state() == original
  component.set_state({'scenes': [{'num_rounds': 6}]})
  updated = component.get_state()
  assert updated['scenes'][0]['num_rounds'] == 6
  assert (
      updated['scenes'][0]['start_time'] == original['scenes'][0]['start_time']
  )
  assert (
      updated['scenes'][0]['scene_type'] == original['scenes'][0]['scene_type']
  )
  for record, path in [
      ({'num_rounds': 0}, 'num_rounds'),
      ({'participants': ['Missing']}, 'premise'),
      ({'scene_type': {'name': ''}}, 'scene_type.name'),
      ({'start_time': 'yesterday'}, 'start_time'),
      ({'extra': 3}, 'scenes'),
      ({'participants': ['Alice', 'Alice']}, 'participants'),
  ]:
    with pytest.raises(ValueError, match=path):
      component.set_state({'scenes': [record]})
    assert component.get_state() == updated
  component.set_state({})
  assert component.get_state() == updated


def test_callables_are_not_stringified_or_editable():
  component = tracker()
  component._scenes[0].scene_type.default_premise['Alice'] = [lambda name: name]
  assert component.get_dynamic_state() == {}
  assert component.get_state() == {}
  with pytest.raises(ValueError, match='not editable'):
    component.set_state({'scenes': []})


def test_existing_memory_cursor_is_preserved_and_schedule_rebuilt():
  registry = fixtures.scene_registry()
  simulation = fixtures.build(
      registry.to_config(registry.default_document('scenes-v1'))
  )
  gm = fixtures.as_agent(simulation.get_game_masters()[0])
  component = gm.get_component(
      '__next_game_master__', type_=scene_tracker.SceneTracker
  )
  memory = gm.get_component('__memory__', type_=memory_lib.Memory)
  memory.add('[scene counter](1)')
  memory.update()
  before = copy.deepcopy(memory.get_state())
  simulation.set_component_dynamic_state(
      'Conversation', '__next_game_master__', 'scenes', [{'num_rounds': 3}]
  )
  assert component._get_scene_step_and_scene()[0] == 1
  assert component._max_rounds == 3
  assert memory.get_state() == before
  memory.add('[scene counter](2)')
  memory.update()
  stable = component.get_state()
  with pytest.raises(ValueError, match='current progress'):
    component.set_state({'scenes': [{'num_rounds': 1}]})
  assert component.get_state() == stable
  assert component._get_scene_step_and_scene()[0] == 2


@pytest.mark.parametrize(
    'action_spec',
    [
        None,
        {
            'Alice': entity_lib.DEFAULT_ACTION_SPEC,
            'action_spec': entity_lib.DEFAULT_SPEECH_ACTION_SPEC,
        },
    ],
)
def test_action_spec_mapping_and_none_roundtrip(action_spec):
  component = tracker()
  value = json.loads(json.dumps(component.get_dynamic_state()))
  if action_spec is None:
    value['scenes'][0]['scene_type']['action_spec'] = None
  else:
    value['scenes'][0]['scene_type']['action_spec'] = {
        name: {'action_spec': spec.to_dict()}
        for name, spec in action_spec.items()
    }
  component.set_state(value)
  assert component.get_dynamic_state() == value
  assert component._scenes[0].scene_type.action_spec == action_spec
  for invalid in [
      {'Alice': None},
      {
          'Alice': {
              'Bob': {'action_spec': entity_lib.DEFAULT_ACTION_SPEC.to_dict()}
          }
      },
      {'Alice': []},
  ]:
    proposed = copy.deepcopy(value)
    proposed['scenes'][0]['scene_type']['action_spec'] = invalid
    with pytest.raises(ValueError, match='action_spec.Alice'):
      component.set_state(proposed)
    assert component.get_dynamic_state() == value


def test_empty_configuration_json_roundtrip_and_legacy_checkpoint():
  component = scene_tracker.SceneTracker(
      no_language_model.NoLanguageModel(), []
  )
  state = json.loads(json.dumps(component.get_state()))
  assert state == {'scenes': []}
  component.set_state(state)
  component.set_state({})
  assert component.get_state() == state
  for malformed in (None, {}, [None], [{'num_rounds': 1}]):
    with pytest.raises(ValueError, match='scenes'):
      component.set_state({'scenes': malformed})
    assert component.get_state() == state


def test_empty_schedule_done_and_progress_constraint():
  registry = fixtures.scene_registry()
  simulation = fixtures.build(
      registry.to_config(registry.default_document('scenes-v1'))
  )
  gm = fixtures.as_agent(simulation.get_game_masters()[0])
  component = gm.get_component(
      '__next_game_master__', type_=scene_tracker.SceneTracker
  )
  memory = gm.get_component('__memory__', type_=memory_lib.Memory)
  original = component.get_state()
  assert not component.is_done()
  component.set_state({'scenes': []})
  component.set_state(json.loads(json.dumps(component.get_state())))
  assert component.is_done()
  component.set_state(original)
  assert not component.is_done()
  memory.add('[scene counter](1)')
  memory.update()
  before = copy.deepcopy(memory.get_state())
  with pytest.raises(ValueError, match='current progress'):
    component.set_state({'scenes': []})
  assert component.get_state() == original
  assert memory.get_state() == before
  assert not component.is_done()
