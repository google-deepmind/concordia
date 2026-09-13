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

"""Optional Chromium round-trip checks using the standard editor and server.

Install Playwright and its Chromium browser, then run:
  python -m pytest -n 0 concordia/utils/visual_interface_browser_test.py

Only component editing is exercised: no simulation steps or model calls run.
The module is skipped when the optional Playwright package is absent.
"""

import copy
import json
from unittest import mock

from concordia.environment.engines import sequential
from concordia.language_model import no_language_model
from concordia.prefabs.entity import minimal
from concordia.prefabs.simulation import generic
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.utils import simulation_server
from concordia.utils import visual_interface
import numpy as np
import pytest

browser_api = pytest.importorskip('playwright.sync_api')

_VALUES = (
    'Explain "ren" and \'li\'.',
    '\nFirst line\n\nSecond line\n',
    '<b>literal</b> & &amp;',
    '仁 λ 🛰️ e\u0301',
    '',
    ' \t leading and trailing \t ',
    '</textarea><span id="text-markup-probe">literal</span>',
    '</script><script>globalThis.textProbe = true;</script>',
)


@pytest.fixture(scope='module')
def browser():
  with browser_api.sync_playwright() as playwright:
    chromium = playwright.chromium.launch()
    yield chromium
    chromium.close()


def _states(simulation):
  return {
      entity.name: entity.get_state() for entity in simulation.get_entities()
  }


@pytest.mark.parametrize('index', range(len(_VALUES)))
def test_literal_prose_roundtrip(browser, index, tmp_path):
  initial = _VALUES[index]
  edited = _VALUES[(index + 1) % len(_VALUES)]
  # Players are already in visual order: this does not depend on mixed-role
  # card identity behavior, which is a separate concern.
  config = prefab_lib.Config(
      prefabs={'minimal': minimal.Entity()},
      instances=[
          prefab_lib.InstanceConfig(
              prefab='minimal',
              role=prefab_lib.Role.ENTITY,
              params={'name': name, 'custom_instructions': value},
          )
          for name, value in [('Alice', initial), ('Bob', 'Do not edit Bob.')]
      ],
  )
  with (
      mock.patch.object(generic.Simulation, 'play', side_effect=AssertionError),
      mock.patch.object(
          sequential.Sequential, 'run_loop', side_effect=AssertionError
      ),
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
    simulation = generic.Simulation(
        config=config,
        model=no_language_model.NoLanguageModel(),
        embedder=lambda _: np.ones(3),
        engine=sequential.Sequential(),
    )
    before = _states(simulation)
    checkpoint = simulation.make_checkpoint_data()
    server = simulation_server.SimulationServer(
        port=0,
        html_content=visual_interface.visualize_config_to_html(
            config, checkpoint_data=checkpoint
        ),
    )
    server.set_simulation(simulation)
    server.broadcast_entity_info(checkpoint)
    server.start()
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    try:
      page.goto(f'http://127.0.0.1:{server.bound_port}/')
      browser_api.expect(page.locator('#console-output')).to_contain_text(
          'Connected to simulation server'
      )
      page.locator('[data-entity-id="entity_0"]').click()
      browser_api.expect(page.locator('#inspector-title')).to_have_text('Alice')
      control = page.locator('#dyn_Instructions_state')
      browser_api.expect(control).to_have_value(initial)
      page.locator('#toggle_comp_Instructions').click()
      control.fill(edited)
      button = page.locator('[data-input-id="dyn_Instructions_state"]')
      with page.expect_response('**/cmd/set_component_state') as response:
        button.click()
      assert response.value.json()['status'] == 'ok'
      # Wait for the real SSE checkpoint, not just the locally edited control.
      page.wait_for_function(
          """value => entityData.entity_0.component_info.context_components
              .Instructions.state.state === value""",
          arg=edited,
      )
      browser_api.expect(control).to_have_value(edited)
      expected = copy.deepcopy(before)
      expected['Alice']['context_components']['Instructions']['state'] = edited
      assert _states(simulation) == expected
      entity = simulation.get_entities()[0]
      assert isinstance(entity, entity_component.EntityWithComponents)
      component = entity.get_component('Instructions')
      assert component.get_state()['state'] == edited
      assert component.get_state()['pre_act_label'] == '\nInstructions'
      # Read-only metadata has no edit control; backend validation still rejects
      # attempts to change it, even when bypassing the UI.
      assert page.locator('[data-state-key="pre_act_label"]').count() == 0
      with pytest.raises(ValueError):
        simulation.set_component_dynamic_state(
            'Alice', 'Instructions', 'pre_act_label', 'not permitted'
        )
      # The HTTP document still has the initial checkpoint. The existing SSE
      # reconnect must replace it with the saved value on reload.
      page.reload()
      page.wait_for_function(
          """value => entityData.entity_0.component_info.context_components
              .Instructions.state.state === value""",
          arg=edited,
      )
      page.locator('[data-entity-id="entity_0"]').click()
      browser_api.expect(control).to_have_value(edited)
      page.locator('#toggle_comp_Instructions').click()
      page.screenshot(path=str(tmp_path / 'saved-and-reloaded.png'))
      # A no-op Save after repaint/reload must not silently alter prose.
      with page.expect_response('**/cmd/set_component_state') as response:
        button.click()
      assert response.value.json()['status'] == 'ok'
      assert component.get_state()['state'] == edited
      page.wait_for_function(
          """value => entityData.entity_0.component_info.context_components
              .Instructions.state.state === value""",
          arg=edited,
      )
      (tmp_path / 'roundtrip.json').write_text(
          json.dumps(
              {
                  'initial': initial,
                  'saved': edited,
                  'reloaded': control.input_value(),
                  'component_state': component.get_state(),
                  'other_entity_unchanged': (
                      _states(simulation)['Bob'] == before['Bob']
                  ),
                  'simulation_steps': 0,
                  'model_calls': 0,
              },
              ensure_ascii=False,
              indent=2,
          ),
          encoding='utf-8',
      )
      assert page.locator('#text-markup-probe').count() == 0
      assert not page.evaluate('Boolean(globalThis.textProbe)')
      assert not errors
      assert not server.current_step_data
    finally:
      page.close()
      server.stop()
