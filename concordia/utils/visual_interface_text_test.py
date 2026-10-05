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

"""Tests for literal component data in the editor's initial HTML document."""

from html import parser
import json

from absl.testing import absltest
from absl.testing import parameterized
from concordia.typing import prefab as prefab_lib
from concordia.utils import visual_interface


class _Scripts(parser.HTMLParser):
  """Extracts script text with the browser's raw-text parsing boundaries."""

  def __init__(self):
    super().__init__()
    self.scripts = []
    self.in_script = False

  def handle_starttag(self, tag, attrs):
    if tag == 'script':
      self.scripts.append('')
      self.in_script = True

  def handle_endtag(self, tag):
    if tag == 'script':
      self.in_script = False

  def handle_data(self, data):
    if self.in_script:
      self.scripts[-1] += data


class LiteralComponentDataTest(parameterized.TestCase):

  @parameterized.parameters(
      'Explain "ren" and \'li\'.',
      '\nFirst line\n\nSecond line\n',
      '<b>literal</b> & &amp;',
      '仁 λ 🛰️ e\u0301',
      '',
      '</textarea><span id="text-markup-probe">literal</span>',
      '</script><script>globalThis.textProbe = true;</script>',
      '<!-- </ScRiPt> literal closing tag',
  )
  def test_initial_script_preserves_component_state(self, value):
    config = prefab_lib.Config(
        prefabs={},
        instances=[
            prefab_lib.InstanceConfig(
                prefab='minimal',
                role=prefab_lib.Role.ENTITY,
                params={'name': 'Alice'},
            )
        ],
    )
    checkpoint = {
        'entities': {
            'Alice': {
                'component_info': {
                    'context_components': {
                        'Instructions': {
                            'class_name': 'Constant',
                            'state': {
                                'state': value,
                                'pre_act_label': '\nInstructions',
                            },
                            'dynamic_state': {'state': value},
                        }
                    },
                }
            }
        },
        'game_masters': {},
    }
    html = visual_interface.visualize_config_to_html(
        config, checkpoint_data=checkpoint
    )
    document = _Scripts()
    document.feed(html)
    self.assertLen(document.scripts, 1)
    script = document.scripts[0]
    data_start = script.index('const entityData = ') + len(
        'const entityData = '
    )
    data, _ = json.JSONDecoder().raw_decode(script[data_start:])
    component = data['entity_0']['component_info']['context_components'][
        'Instructions'
    ]
    self.assertEqual(component['state']['state'], value)
    self.assertEqual(component['dynamic_state'], {'state': value})
    self.assertEqual(component['state']['pre_act_label'], '\nInstructions')


if __name__ == '__main__':
  absltest.main()
