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

"""Example setup and export checks; no simulation/model execution."""

import copy
import json
from pathlib import Path
from unittest import mock

from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
import pytest

from examples.project_editor import run
from examples.project_editor import template


def test_open_builds_preview_without_running():
  with (
      mock.patch.object(generic.Simulation, 'play', side_effect=AssertionError),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_text',
          side_effect=AssertionError,
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_choice',
          side_effect=AssertionError,
      ),
  ):
    server = run.create_editor(port=0)
    assert not server.is_serving
    assert server.simulation is None
    service = server.operation_service
    assert service is not None
    state = service.snapshot('developer')['result']
    assert state['state'] == 'ready'
    fields = state['definition']['inspector']['conversation']
    assert fields['acting_order']['choices'] == [
        'fixed',
        'random',
        'game_master_choice',
    ]
    assert (
        'Goal'
        in state['definition']['entities']['entity_1']['component_info'][
            'context_components'
        ]
    )


@pytest.mark.parametrize('delay', [-1, float('inf'), float('nan'), 6])
def test_pacing_is_finite_and_bounded(delay):
  with pytest.raises(ValueError, match='delay'):
    run.create_editor(step_delay=delay)


def test_each_export_preserves_definition_and_previous_log(tmp_path: Path):
  registry = template.registry()
  document = registry.default_document(template.TEMPLATE_KEY)
  log = mock.Mock()
  log.to_json.return_value = '{"fixture":true}'
  log.to_html.return_value = '<p>Fixture</p>'
  first = run.save_result(registry, document, log, tmp_path)
  second = run.save_result(registry, document, log, tmp_path)
  assert first != second
  for directory in (first, second):
    assert (
        json.loads((directory / 'initial-project.json').read_text()) == document
    )
    assert (directory / 'log.json').read_text() == '{"fixture":true}'


def test_builder_and_legacy_documents_remain_distinct():
  registry = template.registry()
  builder = registry.default_document(template.TEMPLATE_KEY)
  assert builder['schema_version'] == 2
  assert [x['prototype'] for x in builder['instances']] == [
      'alice',
      'bob',
      'conversation',
  ]
  assert len(registry.catalog(builder)) == 3
  for key in (template.LEGACY_TEMPLATE_KEY, template.PREVIOUS_TEMPLATE_KEY):
    legacy = registry.default_document(key)
    assert legacy['schema_version'] == 1
    assert registry.loads(registry.dumps(legacy)) == legacy
    assert registry.catalog(legacy) == []
    assert all('prototype' not in item for item in legacy['instances'])


def test_builder_validates_each_duplicated_gm():
  registry = template.registry()
  document = registry.default_document(template.TEMPLATE_KEY)
  gm = copy.deepcopy(document['instances'][2])
  gm['id'] = 'new-gm'
  gm['params']['name'] = 'New GM'
  gm['params']['acting_order'] = 'unsupported'
  document['instances'].append(gm)
  with pytest.raises(ValueError, match='new-gm'):
    registry.normalize(document)
