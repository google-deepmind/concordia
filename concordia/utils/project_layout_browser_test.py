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

"""Intercepted-browser editor scrollport, sizing and typography contracts."""

import argparse
import json
from unittest import mock
import urllib.parse

from concordia.command_line_interface import concordia_session
from concordia.utils import project_test_support
from concordia.utils import simulation_server
import pytest

# pylint: disable=redefined-outer-name


@pytest.fixture
def editor_browser():
  browser_api = pytest.importorskip('playwright.sync_api')
  registry = project_test_support.scene_registry()
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry,
      registry.default_document('scenes-v1'),
      integrated=True,
      run=mock.Mock(side_effect=AssertionError('No simulation')),
      preview=lambda _: {
          'entities': {
              'Alice': {
                  'component_info': {
                      'context_components': {
                          'Fixture': {
                              'class_name': 'FixtureComponent',
                              'state': {'text': 'Inspector detail fixture'},
                          }
                      }
                  }
              }
          }
      },
  )
  service = server.operation_service
  assert service is not None

  def intercept(route):
    path = urllib.parse.urlsplit(route.request.url).path
    if path == '/':
      route.fulfill(content_type='text/html', body=server.html_content)
    elif path == '/api/state':
      route.fulfill(json=service.snapshot('developer'))
    elif path == '/api/dispatch':
      route.fulfill(
          json=service.dispatch('developer', route.request.post_data_json)
      )
    else:
      route.fulfill(status=404, body='No external network')

  with (
      mock.patch.object(
          server, 'start', side_effect=AssertionError('No listener')
      ),
      browser_api.sync_playwright() as playwright,
  ):
    browser = playwright.chromium.launch()

    def open_page(**kwargs):
      context = browser.new_context(**kwargs)
      context.route('**/*', intercept)
      context.add_init_script('window.EventSource=class {close(){}};')
      page = context.new_page()
      page.goto('http://localhost/')
      browser_api.expect(page.locator('#editor-status')).to_contain_text(
          'saved definition'
      )
      return context, page

    yield open_page, browser_api.expect
    browser.close()


def test_terminal_scrollback_and_accessible_splitters(editor_browser, tmp_path):
  open_page, expect = editor_browser
  context, page = open_page(viewport={'width': 1400, 'height': 950})
  errors = []
  page.on('pageerror', lambda error: errors.append(str(error)))
  output = page.get_by_role('log', name='Simulation log')
  prompt = page.locator('#editor-command-form')
  command = page.get_by_role('textbox', name='Simulation log command')
  assert prompt.evaluate('node=>node.parentElement.id') == 'console-output'
  page.evaluate(
      "for(let i=0;i<100;i++)logConsole('Scroll fixture"
      " '+i);document.getElementById('console-output').scrollTop=0;"
  )
  before = output.evaluate('node=>node.scrollTop')
  page.evaluate("logConsole('New output while reading')")
  assert output.evaluate('node=>node.scrollTop') == before
  assert (
      output.evaluate('node=>node.lastElementChild.id') == 'editor-command-form'
  )
  assert (
      prompt.bounding_box()['y']
      > output.bounding_box()['y'] + output.bounding_box()['height']
  )
  output.evaluate('node=>node.scrollTop=node.scrollHeight')
  page.evaluate("logConsole('Follow bottom')")
  assert (
      output.evaluate(
          'node=>node.scrollHeight-node.clientHeight-node.scrollTop'
      )
      < 2
  )
  command.fill('help')
  command.press('Enter')
  expect(output).to_contain_text('Run starts a fresh')
  command.press('ArrowUp')
  expect(command).to_have_value('help')
  page.evaluate("logConsole('Output while composing')")
  expect(command).to_have_value('help')
  expect(command).to_be_focused()
  assert (
      output.evaluate('node=>node.lastElementChild.id') == 'editor-command-form'
  )

  left = page.get_by_role('separator', name='Hierarchy width', exact=True)
  right = page.get_by_role('separator', name='Inspector width', exact=True)
  terminal = page.get_by_role(
      'separator', name='Simulation log height', exact=True
  )
  initial = float(left.get_attribute('aria-valuenow'))
  box = left.bounding_box()
  page.mouse.move(box['x'] + 4, box['y'] + 30)
  page.mouse.down()
  page.mouse.move(box['x'] + 84, box['y'] + 30, steps=4)
  page.mouse.up()
  assert float(left.get_attribute('aria-valuenow')) >= initial + 75
  left.focus()
  left.press('End')
  assert left.get_attribute('aria-valuenow') == left.get_attribute(
      'aria-valuemax'
  )
  left.press('Home')
  assert left.get_attribute('aria-valuenow') == left.get_attribute(
      'aria-valuemin'
  )
  right.focus()
  right.press('Home')
  assert right.get_attribute('aria-valuenow') == right.get_attribute(
      'aria-valuemin'
  )
  right.press('ArrowLeft')
  assert (
      float(right.get_attribute('aria-valuenow'))
      == float(right.get_attribute('aria-valuemin')) + 10
  )
  # Drag terminal upwards to its bound, then verify keyboard resizing too.
  box = terminal.bounding_box()
  page.mouse.move(box['x'] + 100, box['y'] + 4)
  page.mouse.down()
  page.mouse.move(box['x'] + 100, 0, steps=5)
  page.mouse.up()
  assert terminal.get_attribute('aria-valuenow') == terminal.get_attribute(
      'aria-valuemax'
  )
  assert page.locator('.bottom-panel').bounding_box()['height'] > 950 * 0.60
  terminal.focus()
  terminal.press('ArrowDown')
  assert (
      float(terminal.get_attribute('aria-valuenow'))
      == float(terminal.get_attribute('aria-valuemax')) - 10
  )
  saved = float(terminal.get_attribute('aria-valuenow'))
  page.reload()
  expect(terminal).to_have_attribute('aria-valuenow', str(round(saved)))
  page.set_viewport_size({'width': 720, 'height': 500})
  expect(left).to_be_visible()
  assert page.locator('.center-panel').bounding_box()['width'] >= 159
  assert page.locator('.right-sidebar').bounding_box()['width'] >= 179
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
  page.set_viewport_size({'width': 1400, 'height': 950})
  page.screenshot(path=str(tmp_path / 'layout-desktop.png'))
  assert not errors
  context.close()


def test_responsive_typography_touch_and_untrusted_saved_sizes(
    editor_browser, tmp_path
):
  open_page, expect = editor_browser
  desktop, page = open_page(viewport={'width': 1200, 'height': 900})
  font = lambda selector: page.locator(selector).first.evaluate(
      'node=>getComputedStyle(node).fontSize'
  )
  assert font('.left-sidebar button') == font('.center-panel') == '12px'
  assert font('.param-row .param-value') == '11px'
  assert font('.component-state .state-value') == '10px'
  assert font('.component-item-class') == '10px'
  assert font('.inspector-title') == '14px'
  page.locator('.component-item-header').first.click()
  expect(page.locator('.component-state .state-value').first).to_be_visible()
  assert page.locator('#config-svg').evaluate(
      'node=>node.getScreenCTM().a'
  ) == pytest.approx(1)
  page.set_viewport_size({'width': 2400, 'height': 1000})
  assert page.locator('#config-svg').evaluate(
      'node=>node.getScreenCTM().a'
  ) == pytest.approx(1)
  page.evaluate("document.documentElement.style.fontSize='20px'")
  assert font('.center-panel') == '15px'
  assert font('.param-row .param-value') == '13.75px'
  assert font('.component-state .state-value') == '12.5px'
  assert page.locator('#config-svg').evaluate(
      'node=>node.getScreenCTM().a'
  ) == pytest.approx(1.25)
  page.evaluate("document.documentElement.style.fontSize=''")
  page.set_viewport_size({'width': 1400, 'height': 950})
  page.screenshot(path=str(tmp_path / 'layout-typography-desktop.png'))
  page.evaluate(
      "localStorage.setItem('concordia.editor.layout.v1',JSON.stringify({left:999,right:-3,terminal:'bad'}))"
  )
  page.reload()
  assert page.locator('.center-panel').bounding_box()['width'] >= 160
  desktop.close()
  touch, page = open_page(
      viewport={'width': 1100, 'height': 850}, has_touch=True
  )
  assert page.evaluate("matchMedia('(pointer:coarse)').matches")
  assert (
      font('.left-sidebar button')
      == font('.center-panel')
      == font('.right-sidebar')
      == '16px'
  )
  assert font('.param-row .param-value') == '16px'
  assert font('.component-state .state-value') == '16px'
  assert font('.component-item-class') == '16px'
  assert page.locator('#config-svg').evaluate(
      'node=>node.getScreenCTM().a'
  ) == pytest.approx(1.4)
  handle = page.get_by_role('separator', name='Hierarchy width', exact=True)
  box = handle.bounding_box()
  initial = float(handle.get_attribute('aria-valuenow'))
  cdp = touch.new_cdp_session(page)
  cdp.send(
      'Input.dispatchTouchEvent',
      {
          'type': 'touchStart',
          'touchPoints': [{'x': box['x'] + 4, 'y': box['y'] + 30}],
      },
  )
  cdp.send(
      'Input.dispatchTouchEvent',
      {
          'type': 'touchMove',
          'touchPoints': [{'x': box['x'] + 54, 'y': box['y'] + 30}],
      },
  )
  cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
  assert float(handle.get_attribute('aria-valuenow')) >= initial + 45
  page.set_viewport_size({'width': 412, 'height': 915})
  expect(handle).to_be_hidden()
  assert font('.param-row .param-value') == '16px'
  assert font('.component-state .state-value') == '16px'
  page.locator('[data-tab=inspector]').click()
  page.locator('.component-item-header').first.click()
  detail = page.locator('.component-state .state-value').first
  detail.scroll_into_view_if_needed()
  expect(detail).to_be_visible()
  page.screenshot(path=str(tmp_path / 'layout-typography-mobile.png'))
  page.locator('[data-tab=log]').click()
  expect(
      page.get_by_role('textbox', name='Simulation log command')
  ).to_be_visible()
  assert font('#editor-command') == '16px'
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
  assert page.locator('#editor-command').bounding_box()['height'] >= 44
  page.screenshot(path=str(tmp_path / 'layout-mobile.png'))
  page.set_viewport_size({'width': 1100, 'height': 850})
  expect(handle).to_be_visible()
  assert page.locator('.center-panel').bounding_box()['width'] >= 160
  touch.close()


def test_shared_catalog_inspection_creation_and_layout_commands(editor_browser):
  open_page, expect = editor_browser
  context, page = open_page(viewport={'width': 1400, 'height': 950})
  command = page.get_by_role('textbox', name='Simulation log command')
  output = page.get_by_role('log', name='Simulation log')

  def send(line):
    command.fill(line)
    command.press('Enter')
    expect(page.get_by_role('button', name='Send', exact=True)).to_be_enabled()

  send('catalog components')
  expect(output).to_contain_text('dependencies')
  send('inspect alice Fixture')
  expect(output).to_contain_text('Inspector detail fixture')
  send('add instance alice --id charlie')
  send('set . params.name \'"Charlie"\'')
  send('inspect charlie')
  expect(output).to_contain_text('Charlie')
  send('layout terminal 99999')
  separator = page.get_by_role('separator', name='Simulation log height')
  assert separator.get_attribute('aria-valuenow') == separator.get_attribute(
      'aria-valuemax'
  )
  send('layout left 260')
  assert (
      page.get_by_role('separator', name='Hierarchy width').get_attribute(
          'aria-valuenow'
      )
      == '260'
  )
  send("locate '$.instances[charlie].params.name: name needed'")
  expect(page.locator('#editor-charlie-name')).to_be_focused()
  context.close()


def test_loaded_reordered_draft_keeps_runtime_actor_identity(tmp_path):
  """Typed inspection follows snapshot identity before and after draft Save."""
  browser_api = pytest.importorskip('playwright.sync_api')
  registry = project_test_support.scene_registry()
  original = registry.default_document('scenes-v1')
  checkpoint = {
      'entities': {
          name: {
              'component_info': {
                  'context_components': {
                      name
                      + 'Only': {
                          'class_name': 'FixtureComponent',
                          'state': {'text': name + ' retained runtime'},
                      }
                  }
              }
          }
          for name in ('Alice', 'Bob')
      }
  }
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry,
      original,
      integrated=True,
      run=mock.Mock(side_effect=AssertionError('No simulation')),
      preview=lambda _: checkpoint,
  )
  adapter = server._project_editor
  assert adapter is not None
  # Populate the real snapshot pipeline directly; never start a runner/engine.
  adapter.begin(registry.to_config(original))
  adapter.receive_checkpoint(checkpoint)
  service = server.operation_service
  assert service is not None
  reordered = json.loads(json.dumps(original))
  reordered['instances'][:2] = list(reversed(reordered['instances'][:2]))
  source = tmp_path / 'reordered.json'
  source.write_text(json.dumps(reordered))

  def intercept(route):
    path = urllib.parse.urlsplit(route.request.url).path
    if path == '/':
      route.fulfill(content_type='text/html', body=server.html_content)
    elif path == '/api/state':
      route.fulfill(json=service.snapshot('developer'))
    elif path == '/api/dispatch':
      route.fulfill(
          json=service.dispatch('developer', route.request.post_data_json)
      )
    else:
      route.fulfill(status=404, body='No external network')

  with (
      mock.patch.object(
          server, 'start', side_effect=AssertionError('No listener')
      ),
      browser_api.sync_playwright() as playwright,
  ):
    browser = playwright.chromium.launch()
    with browser.new_context(
        viewport={'width': 1400, 'height': 950}
    ) as context:
      context.route('**/*', intercept)
      context.add_init_script('window.EventSource=class {close(){}};')
      page = context.new_page()
      errors = []
      page.on('pageerror', lambda error: errors.append(str(error)))
      page.goto('http://localhost/')
      expect = browser_api.expect
      expect(page.locator('#editor-status')).to_contain_text('saved definition')
      command = page.get_by_role('textbox', name='Simulation log command')
      output = page.get_by_role('log', name='Simulation log')

      def send(line):
        command.fill(line)
        command.press('Enter')
        expect(
            page.get_by_role('button', name='Send', exact=True)
        ).to_be_enabled()

      with page.expect_file_chooser() as chooser:
        send('load')
      chooser.value.set_files(source)
      expect(output).to_contain_text('Loaded local draft')
      assert server.get_project()['document'] == original
      send('inspect bob BobOnly')
      expect(output.locator('.console-line').last).to_contain_text(
          'Bob retained runtime'
      )
      send('view runtime')
      send('select bob BobOnly')
      expect(output).to_contain_text('Selected bob · BobOnly')
      send('inspect bob BobOnly')
      expect(output.locator('.console-line').last).to_contain_text(
          'Bob retained runtime'
      )
      send('inspect alice AliceOnly')
      expect(output.locator('.console-line').last).to_contain_text(
          'Alice retained runtime'
      )
      send('view definition')
      send('save')
      expect(page.locator('#editor-status')).to_contain_text('saved definition')
      assert server.get_project()['document']['instances'][0]['id'] == 'bob'
      assert adapter.runtime_view is not None
      assert adapter.runtime_view['document']['instances'][0]['id'] == 'alice'
      send('view runtime')
      send('inspect bob BobOnly')
      expect(output.locator('.console-line').last).to_contain_text(
          'Bob retained runtime'
      )
      assert not errors
    browser.close()

  # External CLI consumes the same retained mapping after the saved order
  # changes.
  with mock.patch.object(
      concordia_session.urllib.request,
      'urlopen',
      side_effect=project_test_support.fake_http(server),
  ):
    args = argparse.Namespace(
        url='http://fixture',
        timeout=1,
        line='view runtime',
        draft=tmp_path / 'journal.json',
        file=None,
        output=None,
    )
    concordia_session.friendly(args)
    args.line = 'inspect bob BobOnly'
    assert (
        concordia_session.friendly(args)['state']['text']
        == 'Bob retained runtime'
    )

  adapter.begin(registry.to_config(reordered))
  assert adapter.runtime_view is None
  adapter.receive_checkpoint(checkpoint)
  assert adapter.runtime_view is not None
  assert adapter.runtime_view['document']['instances'][0]['id'] == 'bob'
  assert adapter.runtime_view['entities']['entity_0']['name'] == 'Bob'


if __name__ == '__main__':
  raise SystemExit(pytest.main([__file__]))
