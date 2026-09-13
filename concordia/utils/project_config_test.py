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

"""Initial project contracts against the supported ordinary template."""

import copy
import dataclasses
import json
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import entity_agent_with_logging
from concordia.utils import project_config

from examples.project_editor import run
from examples.project_editor import template


class ProjectConfigTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.registry = template.registry()
    self.doc = self.registry.default_document(template.TEMPLATE_KEY)

  @parameterized.parameters(
      '"quotes" & <tags> café 🎵',
      'first\n\nsecond\n',
      '',
      '</script><script>throw Error(1)</script>',
  )
  def test_literal_roundtrip_and_types(self, value):
    self.doc['instances'][0]['params']['custom_instructions'] = value
    self.doc['premise'] = value
    self.doc['instances'][2]['params']['can_terminate_simulation'] = True
    self.doc['max_steps'] = 2
    reopened = self.registry.loads(self.registry.dumps(self.doc))
    self.assertEqual(reopened, self.doc)
    config = self.registry.to_config(reopened)
    self.assertEqual(config.instances[0].params['custom_instructions'], value)
    self.assertIs(config.instances[2].params['can_terminate_simulation'], True)
    self.assertEqual(config.default_max_steps, 2)

  def test_stable_ids_survive_reorder_and_rename_reference(self):
    self.doc['instances'][2]['params']['name'] = 'Renamed GM'
    self.doc['instances'].reverse()
    normalized = self.registry.normalize(self.doc)
    self.assertEqual(
        [x['id'] for x in normalized['instances']],
        ['alice', 'bob', 'conversation'],
    )
    config = self.registry.to_config(normalized)
    self.assertEqual(
        config.instances[2].params['next_game_master_name'], 'Renamed GM'
    )
    self.assertEqual(
        normalized['instances'][2]['params']['next_game_master_name'],
        'conversation',
    )

  @parameterized.parameters(
      ('schema_version', True, '$.schema_version'),
      ('template', 'os.system', '$.template'),
      ('max_steps', False, '$.max_steps'),
      ('max_steps', 0, '$.max_steps'),
      ('max_steps', 1001, '$.max_steps'),
      ('premise', [], '$.premise'),
      ('instances', [], '$.instances'),
  )
  def test_invalid_document_path(self, key, value, path):
    self.doc[key] = value
    with self.assertRaises(project_config.ValidationError) as error:
      self.registry.normalize(self.doc)
    self.assertEqual(error.exception.path, path)

  @parameterized.parameters(
      ('randomize_choices', 'false'),
      ('goal', {}),
      ('custom_instructions', None),
      ('name', 'Bob'),
      ('name', ''),
      ('extra_components', {'x': 'not a component'}),
  )
  def test_invalid_actor_params(self, key, value):
    self.doc['instances'][0]['params'][key] = value
    with self.assertRaisesRegex(
        project_config.ValidationError, r'instances\[[01]\].params'
    ):
      self.registry.normalize(self.doc)

  @parameterized.parameters(
      ('acting_order', 'nonsense'),
      ('next_game_master_name', 'alice'),
      ('next_game_master_name', 'missing'),
  )
  def test_invalid_gm_params(self, key, value):
    self.doc['instances'][2]['params'][key] = value
    with self.assertRaisesRegex(
        project_config.ValidationError, 'instances.*params.' + key
    ):
      self.registry.normalize(self.doc)

  def test_invalid_unicode_has_field_path(self):
    self.doc['instances'][0]['params']['goal'] = '\ud800'
    with self.assertRaisesRegex(
        project_config.ValidationError, r'params.goal.*Unicode'
    ):
      self.registry.normalize(self.doc)

  def test_unknown_and_duplicate_instance_identity(self):
    for field, value in [
        ('id', 'bob'),
        ('id', 'missing'),
        ('prefab', 'evil.module'),
        ('role', 'initializer'),
    ]:
      bad = copy.deepcopy(self.doc)
      bad['instances'][0][field] = value
      with self.assertRaises(project_config.ValidationError):
        self.registry.normalize(bad)

  def test_no_silent_duplicate_json_keys(self):
    with self.assertRaisesRegex(
        project_config.ValidationError, 'duplicate JSON field'
    ):
      self.registry.loads('{"template":"a", "template":"b"}')

  def test_object_bearing_template_cannot_be_exported(self):
    def unsupported():
      config = template.make_config()
      params: dict[str, Any] = dict(config.instances[0].params)
      params['extra_components'] = {'callable': lambda: None}
      config.instances[0].params = params
      return config

    registry = project_config.Registry(
        {'unsupported': project_config.Template(unsupported, ('a', 'b', 'gm'))}
    )
    with self.assertRaisesRegex(
        project_config.ValidationError, 'extra_components.*objects unsupported'
    ):
      registry.default_document('unsupported')

  def test_editor_headless_config_parity_and_owned_builds(self):
    saved = self.registry.dumps(self.doc)
    first = self.registry.to_config(self.doc)
    second = self.registry.to_config(self.registry.loads(saved))
    self.assertEqual(dataclasses.asdict(first), dataclasses.asdict(second))
    sim_a, sim_b = run.build(first), run.build(second)
    for a, b in zip(
        sim_a.get_entities() + sim_a.get_game_masters(),
        sim_b.get_entities() + sim_b.get_game_masters(),
    ):
      assert isinstance(a, entity_agent_with_logging.EntityAgentWithLogging)
      assert isinstance(b, entity_agent_with_logging.EntityAgentWithLogging)
      self.assertIsNot(a, b)
      self.assertEqual(
          set(a.get_all_context_components()),
          set(b.get_all_context_components()),
      )
      for key in a.get_all_context_components():
        self.assertIsNot(a.get_component(key), b.get_component(key))
    sim_a.set_component_dynamic_state(
        'Alice', 'Instructions', 'state', 'A only'
    )
    a = sim_a.get_entities()[0]
    b = sim_b.get_entities()[0]
    assert isinstance(a, entity_agent_with_logging.EntityAgentWithLogging)
    assert isinstance(b, entity_agent_with_logging.EntityAgentWithLogging)
    self.assertNotEqual(a.get_state(), b.get_state())
    self.assertEqual(json.loads(saved), self.doc)


if __name__ == '__main__':
  absltest.main()
