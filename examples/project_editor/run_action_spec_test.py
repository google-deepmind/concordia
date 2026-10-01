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

"""Exercise the example's first scene action-spec lookup without running it."""

from unittest import mock

from concordia.components.game_master import next_acting
from concordia.environment import engine
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.typing import entity
from examples.project_editor import run
from examples.project_editor import template


def test_default_project_scene_can_request_action_spec_without_model_calls():
  registry = template.registry()
  document = registry.default_document(template.TEMPLATE_KEY)
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
    simulation = run.build(registry.to_config(document))
    component = simulation.game_masters[0].get_component(
        next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    assert component._get_current_scene_type().action_spec is None
    result = component.pre_act(
        entity.ActionSpec(
            call_to_action='Next',
            output_type=entity.OutputType.NEXT_ACTION_SPEC,
        )
    )
    assert engine.action_spec_parser(result) == entity.DEFAULT_ACTION_SPEC
