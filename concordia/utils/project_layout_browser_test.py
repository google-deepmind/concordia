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

from unittest import mock
from urllib.parse import urlsplit

from concordia.utils import project_test_support
from concordia.utils import simulation_server
import pytest


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
    path = urlsplit(route.request.url).path
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

  with mock.patch.object(
      server, 'start', side_effect=AssertionError('No listener')
  ), browser_api.sync_playwright() as playwright:
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
