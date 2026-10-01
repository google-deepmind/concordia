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

"""Scene action-spec regression tests; no engine loop or model calls."""

from unittest import mock

from concordia.components.game_master import next_acting
from concordia.environment import engine
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.typing import entity as entity_lib
from concordia.utils import project_test_support
import pytest


def test_editor_scene_without_override_uses_default_action_spec():
  registry = project_test_support.scene_registry()
  config = registry.to_config(registry.default_document('scenes-v1'))
  with (
      mock.patch.object(
          generic.Simulation, 'play', side_effect=AssertionError('No run')
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_text',
          side_effect=AssertionError('No model'),
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_choice',
          side_effect=AssertionError('No model'),
      ),
  ):
    simulation = project_test_support.build(config)
    component = simulation.game_masters[0].get_component(
        next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    assert component._get_current_scene_type().action_spec is None
    result = component.pre_act(
        entity_lib.ActionSpec(
            call_to_action='Next action spec',
            output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
        )
    )
    assert engine.action_spec_parser(result) == entity_lib.DEFAULT_ACTION_SPEC


@pytest.mark.parametrize('mapping', [False, True])
def test_explicit_scene_action_spec_is_preserved(mapping):
  component = next_acting.NextActionSpecFromSceneSpec()
  spec = entity_lib.choice_action_spec(
      call_to_action='Choose', options=('a', 'b')
  )
  scene = mock.Mock(action_spec={'Alice': spec} if mapping else spec)
  with (
      mock.patch.object(
          component, '_get_current_scene_type', return_value=scene
      ),
      mock.patch.object(
          component, 'get_current_active_player', return_value='Alice'
      ),
  ):
    result = component.pre_act(
        entity_lib.ActionSpec(
            call_to_action='Next',
            output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
        )
    )
  assert engine.action_spec_parser(result) == spec


def test_missing_player_override_has_actionable_error():
  component = next_acting.NextActionSpecFromSceneSpec()
  scene = mock.Mock(name='scene', action_spec={})
  with (
      mock.patch.object(
          component, '_get_current_scene_type', return_value=scene
      ),
      mock.patch.object(
          component, 'get_current_active_player', return_value='Alice'
      ),
      pytest.raises(ValueError, match='Alice'),
  ):
    component.pre_act(
        entity_lib.ActionSpec(
            call_to_action='Next',
            output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
        )
    )
