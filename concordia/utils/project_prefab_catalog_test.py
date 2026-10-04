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

"""Explicit prefab creation keys preserve saved presets and use shared drafts."""

import copy
import dataclasses

from concordia.utils import project_components
from concordia.utils import project_config
from concordia.utils import project_test_support as fixtures
from concordia.utils import session_commands
from concordia.utils import session_draft
import pytest


def registry(bindings=None, factory=fixtures.make_config):
  original = fixtures.builder_registry()._templates['builder-v1']
  return project_config.Registry({
      'builder-v1': dataclasses.replace(
          original, factory=factory, prefab_prototypes=bindings or {}
      )
  })


def test_prefab_defaults_and_legacy_presets_use_distinct_creation_keys():
  old = registry()
  saved = old.default_document('builder-v1')
  new = registry(
      {'minimal': 'alice', 'basic': 'bob', 'dialogic': 'conversation'}
  )
  assert new.loads(old.dumps(saved)) == saved
  assert new.to_config(saved) == old.to_config(saved)
  document = copy.deepcopy(saved)
  document['instances'][0]['params']['goal'] = 'Edited Alice only'
  catalog = new.catalog(document)
  assert [
      x['key']
      for x in catalog
      if not x['instance']['prototype'].startswith('installed:')
  ] == [
      'minimal',
      'basic',
      'dialogic',
      'alice',
      'bob',
      'conversation',
  ]
  journal = {'document': document, 'metadata': {'catalog': catalog}}

  def execute(line):
    nonlocal journal
    journal = session_draft.apply(journal, session_commands.parse(line))[
        'journal'
    ]

  execute('add instance minimal --id Charlie')
  added = journal['document']['instances'][-1]
  assert added['id'] == 'Charlie'
  assert added['prototype'] == 'alice'
  standard = fixtures.make_config().prefabs['minimal'].params
  assert {k: v for k, v in added['params'].items() if k != 'name'} == {
      k: standard[k] for k in added['params'] if k != 'name'
  }
  assert added['params']['goal'] == ''
  assert new.normalize(journal['document']) == journal['document']
  execute('add instance alice --id OriginalPreset')
  assert (
      journal['document']['instances'][-1]['params']['custom_instructions']
      == saved['instances'][0]['params']['custom_instructions']
  )
  execute('add instance dialogic --id Rules')
  gm = journal['document']['instances'][-1]
  assert gm['role'] == 'game_master'
  assert gm['params']['next_game_master_name'] == 'Rules'
  assert (
      new.to_config(journal['document'])
      .instances[-1]
      .params['next_game_master_name']
      == gm['params']['name']
  )
  execute('duplicate alice')
  assert (
      journal['document']['instances'][-1]['params']['goal']
      == 'Edited Alice only'
  )
  # Returned catalogs are owned copies; no edits leak into subsequent instances.
  catalog[0]['instance']['params']['goal'] = 'Mutated catalog'
  assert new.catalog(document)[0]['instance']['params']['goal'] == ''
  assert new.default_document('builder-v1') == saved
  assert new.loads(new.dumps(journal['document'])) == journal['document']


def test_multiple_presets_require_explicit_binding_and_collision_is_rejected():
  # This factory has two minimal presets. There is no implicit first match.
  original = registry(factory=fixtures._legacy_config)
  document = original.default_document('builder-v1')
  journal = {
      'document': document,
      'metadata': {'catalog': original.catalog(document)},
  }
  with pytest.raises(ValueError, match='unambiguous'):
    session_draft.apply(
        journal, session_commands.parse('add instance minimal --id Charlie')
    )
  selected = registry({'minimal': 'bob'}, factory=fixtures._legacy_config)
  entries = selected.catalog(document)
  assert entries[0]['instance']['prototype'] == 'bob'
  assert entries[0]['key'] == 'minimal'
  assert [x['key'] for x in entries[1:] if x['kind'] == 'preset'] == [
      'alice',
      'bob',
      'conversation',
  ]
  # Even malformed/stale client metadata cannot pick the first duplicate key.
  journal['metadata']['catalog'] = [entries[0], copy.deepcopy(entries[0])]
  with pytest.raises(ValueError, match='unambiguous'):
    session_draft.apply(
        journal, session_commands.parse('add instance minimal --id Charlie')
    )
  assert len(journal['document']['instances']) == 3


@pytest.mark.parametrize(
    'bindings,message',
    [
        ({'alice': 'alice'}, 'collides'),
        ({'minimal': 'missing'}, 'unknown'),
        ({'basic': 'alice'}, 'must use'),
    ],
)
def test_invalid_creation_registration_fails_explicitly(bindings, message):
  configured = registry(bindings)
  with pytest.raises(project_config.ValidationError, match=message):
    configured.catalog(configured.default_document('builder-v1'))


def test_prefab_creation_and_duplicate_keep_component_state_semantics():
  configured = registry({'minimal': 'alice'})
  configured = project_config.Registry({
      'builder-v1': dataclasses.replace(
          configured._templates['builder-v1'],
          component_types=project_components.standard_types(
              ('alice',), ('alice',)
          ),
      )
  })
  document = configured.default_document('builder-v1')
  document['dynamic_states'] = {
      'alice': {'Instructions': {'state': 'Edited definition'}}
  }
  journal = {
      'document': document,
      'metadata': {
          'catalog': configured.catalog(document),
          'component_catalog': configured.component_catalog(document),
      },
  }

  def execute(line):
    nonlocal journal
    journal = session_draft.apply(journal, session_commands.parse(line))[
        'journal'
    ]

  execute('add component constant alice --id reminder')
  execute('duplicate alice')
  duplicate_id = journal['selectedId']
  execute('add instance minimal --id Charlie')
  result = journal['document']
  assert (
      len([x for x in result['components'] if x['instance'] == duplicate_id])
      == 1
  )
  assert (
      result['dynamic_states'][duplicate_id]['Instructions']['state']
      == 'Edited definition'
  )
  assert not any(x['instance'] == 'Charlie' for x in result['components'])
  assert 'Charlie' not in result['dynamic_states']
  assert configured.normalize(result) == result
