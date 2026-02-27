# Copyright 2025 DeepMind Technologies Limited.
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

"""Tests for concordia_log CLI tool."""

import io
import json
import os
import sys
import tempfile

from absl.testing import absltest
from concordia.command_line_interface import concordia_log
from concordia.utils import structured_logging


def _create_sample_log() -> structured_logging.SimulationLog:
  log = structured_logging.SimulationLog()
  log.add_entry(
      step=1,
      timestamp='2024-01-01T10:00:00',
      entity_name='Alice',
      component_name='ActComponent',
      entry_type='entity',
      summary='Alice said hello to Bob',
      raw_data={
          'key': 'Entity [Alice]',
          'value': {
              '__act__': {
                  'Key': 'action',
                  'Value': 'Alice said "Hello Bob, nice to meet you."',
                  'Prompt': 'What does Alice do next?',
              },
              '__observation__': {
                  'Key': 'Observation',
                  'Value': ['Bob is standing nearby.'],
              },
              'Instructions': {
                  'Key': 'Instructions',
                  'Value': 'You are Alice, a friendly baker.',
              },
          },
      },
  )
  log.add_entry(
      step=1,
      timestamp='2024-01-01T10:00:00',
      entity_name='Bob',
      component_name='ActComponent',
      entry_type='entity',
      summary='Bob waved at Alice',
      raw_data={
          'key': 'Entity [Bob]',
          'value': {
              '__act__': {
                  'Key': 'action',
                  'Value': 'Bob waved back warmly.',
                  'Prompt': 'What does Bob do next?',
              },
          },
      },
  )
  log.add_entry(
      step=2,
      timestamp='2024-01-01T10:01:00',
      entity_name='Alice',
      component_name='ActComponent',
      entry_type='entity',
      summary='Alice offered coffee',
      raw_data={
          'key': 'Entity [Alice]',
          'value': {
              '__act__': {
                  'Key': 'action',
                  'Value': 'Alice offered to buy coffee for Bob.',
                  'Prompt': 'What does Alice do next?',
              },
          },
      },
  )
  log.add_entry(
      step=2,
      timestamp='2024-01-01T10:01:00',
      entity_name='default rules',
      component_name='game_master',
      entry_type='step',
      summary='Step 2: Alice and Bob are having coffee',
      raw_data={
          'key': 'default rules',
          'value': {
              'resolve': {
                  '__resolution__': {
                      'Value': 'Alice and Bob sat down for coffee.',
                  },
                  'tension_tracker': {
                      '__act__': {
                          'Value': '0.2',
                      },
                  },
              },
          },
      },
  )
  log.attach_memories(
      entity_memories={
          'Alice': [
              'Alice loves hiking in the mountains.',
              'Alice works as a baker.',
              'Alice met Bob at the coffee shop.',
          ],
          'Bob': [
              'Bob is a journalist.',
              'Bob met Alice today.',
          ],
      },
      game_master_memories=['The simulation started at 10:00 AM.'],
  )
  return log


def _create_log_with_images() -> structured_logging.SimulationLog:
  log = structured_logging.SimulationLog()
  fake_base64 = 'iVBORw0KGgoAAAANSUhEUg' + 'A' * 200
  image_md = f'![image](data:image/png;base64,{fake_base64})'
  log.add_entry(
      step=1,
      timestamp='2024-01-01T10:00:00',
      entity_name='Alice',
      component_name='ActComponent',
      entry_type='entity',
      summary='Alice posted an image',
      raw_data={
          'key': 'Entity [Alice]',
          'value': {
              '__act__': {
                  'Key': 'action',
                  'Value': f'Alice posted: "Check this out!" {image_md}',
              },
          },
      },
  )
  return log


def _write_log_to_file(
    log: structured_logging.SimulationLog,
) -> str:
  fd, path = tempfile.mkstemp(suffix='.json')
  with os.fdopen(fd, 'w') as f:
    f.write(log.to_json())
  return path


def _capture_output(func, *args, **kwargs) -> str:
  buf = io.StringIO()
  old_stdout = sys.stdout
  sys.stdout = buf  # type: ignore[assignment]
  try:
    func(*args, **kwargs)
    return buf.getvalue()
  finally:
    sys.stdout = old_stdout


def _capture_stderr(func, *args, **kwargs) -> str:
  buf = io.StringIO()
  old_stderr = sys.stderr
  sys.stderr = buf  # type: ignore[assignment]
  try:
    func(*args, **kwargs)
    return buf.getvalue()
  finally:
    sys.stderr = old_stderr


class OverviewTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._log = _create_sample_log()
    self._path = _write_log_to_file(self._log)

  def tearDown(self):
    super().tearDown()
    os.unlink(self._path)

  def test_overview_text(self):
    output = _capture_output(concordia_log.main, ['overview', self._path])
    self.assertIn('Steps:', output)
    self.assertIn('Entities', output)
    self.assertIn('Alice', output)
    self.assertIn('Bob', output)

  def test_overview_json(self):
    output = _capture_output(
        concordia_log.main, ['--json', 'overview', self._path]
    )
    data = json.loads(output)
    self.assertIn('total_steps', data)
    self.assertIn('total_entries', data)


class EntitiesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._log = _create_sample_log()
    self._path = _write_log_to_file(self._log)

  def tearDown(self):
    super().tearDown()
    os.unlink(self._path)

  def test_entities_text(self):
    output = _capture_output(concordia_log.main, ['entities', self._path])
    self.assertIn('Alice', output)
    self.assertIn('Bob', output)

  def test_entities_json(self):
    output = _capture_output(
        concordia_log.main, ['--json', 'entities', self._path]
    )
    data = json.loads(output)
    self.assertIsInstance(data, list)
    self.assertIn('Alice', data)


class ActionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._log = _create_sample_log()
    self._path = _write_log_to_file(self._log)

  def tearDown(self):
    super().tearDown()
    os.unlink(self._path)

  def test_actions_text(self):
    output = _capture_output(
        concordia_log.main, ['actions', self._path, 'Alice']
    )
    self.assertIn('Step 1', output)
    self.assertIn('Step 2', output)

  def test_actions_json(self):
    output = _capture_output(
        concordia_log.main, ['--json', 'actions', self._path, 'Alice']
    )
    data = json.loads(output)
    self.assertIsInstance(data, list)
    self.assertLen(data, 2)

  def test_actions_unknown_entity(self):
    stderr = _capture_stderr(
        concordia_log.main, ['actions', self._path, 'NonExistent']
    )
    self.assertIn('No actions found', stderr)


class ContextTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._log = _create_sample_log()
    self._path = _write_log_to_file(self._log)

  def tearDown(self):
    super().tearDown()
    os.unlink(self._path)

  def test_context_text(self):
    output = _capture_output(
        concordia_log.main,
        ['context', self._path, 'Alice', '--step', '1'],
    )
    self.assertIn('Action:', output)

  def test_context_json(self):
    output = _capture_output(
        concordia_log.main,
        ['--json', 'context', self._path, 'Alice', '--step', '1'],
    )
    data = json.loads(output)
    self.assertIsInstance(data, dict)

  def test_context_missing_step(self):
    stderr = _capture_stderr(
        concordia_log.main,
        ['context', self._path, 'Alice', '--step', '99'],
    )
    self.assertIn('No context found', stderr)


class StepTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._log = _create_sample_log()
    self._path = _write_log_to_file(self._log)

  def tearDown(self):
    super().tearDown()
    os.unlink(self._path)

  def test_step_text(self):
    output = _capture_output(concordia_log.main, ['step', self._path, '1'])
    self.assertIn('Alice', output)
    self.assertIn('Bob', output)

  def test_step_json(self):
    output = _capture_output(
        concordia_log.main, ['--json', 'step', self._path, '1']
    )
    data = json.loads(output)
    self.assertIsInstance(data, list)


class TimelineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._log = _create_sample_log()
    self._path = _write_log_to_file(self._log)

  def tearDown(self):
    super().tearDown()
    os.unlink(self._path)

  def test_timeline_text(self):
    output = _capture_output(
        concordia_log.main, ['timeline', self._path, 'Alice']
    )
    self.assertIn('Step 1', output)
    self.assertIn('Step 2', output)


class SearchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._log = _create_sample_log()
    self._path = _write_log_to_file(self._log)

  def tearDown(self):
    super().tearDown()
    os.unlink(self._path)

  def test_search_found(self):
    output = _capture_output(
        concordia_log.main, ['search', self._path, 'hello']
    )
    self.assertIn('Alice', output)

  def test_search_not_found(self):
    stderr = _capture_stderr(
        concordia_log.main, ['search', self._path, 'xyznotfound']
    )
    self.assertIn('No entries matching', stderr)


class MemoriesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._log = _create_sample_log()
    self._path = _write_log_to_file(self._log)

  def tearDown(self):
    super().tearDown()
    os.unlink(self._path)

  def test_memories_text(self):
    output = _capture_output(
        concordia_log.main, ['memories', self._path, 'Alice']
    )
    self.assertIn('hiking', output)
    self.assertIn('baker', output)

  def test_memories_json(self):
    output = _capture_output(
        concordia_log.main, ['--json', 'memories', self._path, 'Alice']
    )
    data = json.loads(output)
    self.assertIsInstance(data, list)
    self.assertLen(data, 3)


class ComponentsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._log = _create_sample_log()
    self._path = _write_log_to_file(self._log)

  def tearDown(self):
    super().tearDown()
    os.unlink(self._path)

  def test_components_list_all(self):
    output = _capture_output(
        concordia_log.main,
        ['components', self._path, '--entity', 'Alice'],
    )
    self.assertIn('Components for Alice', output)
    self.assertIn('__act__', output)

  def test_components_list_keys(self):
    output = _capture_output(
        concordia_log.main,
        [
            'components',
            self._path,
            '--entity',
            'Alice',
            '--component',
            '__act__',
        ],
    )
    self.assertIn('Keys for __act__', output)

  def test_components_extract_values(self):
    output = _capture_output(
        concordia_log.main,
        [
            'components',
            self._path,
            '--entity',
            'Alice',
            '--component',
            '__act__',
            '--key',
            'Value',
        ],
    )
    self.assertIn('Alice', output)
    self.assertIn('Step', output)


class DumpTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._log = _create_sample_log()
    self._path = _write_log_to_file(self._log)

  def tearDown(self):
    super().tearDown()
    os.unlink(self._path)

  def test_dump_all(self):
    output = _capture_output(concordia_log.main, ['dump', self._path])
    data = json.loads(output)
    self.assertIsInstance(data, list)
    self.assertLen(data, 4)

  def test_dump_filter_step(self):
    output = _capture_output(
        concordia_log.main, ['dump', self._path, '--step', '1']
    )
    data = json.loads(output)
    self.assertTrue(all(e['step'] == 1 for e in data))

  def test_dump_filter_entity(self):
    output = _capture_output(
        concordia_log.main, ['dump', self._path, '--entity', 'Alice']
    )
    data = json.loads(output)
    self.assertTrue(all(e['entity_name'] == 'Alice' for e in data))


class ImageStrippingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._log = _create_log_with_images()
    self._path = _write_log_to_file(self._log)

  def tearDown(self):
    super().tearDown()
    os.unlink(self._path)

  def test_actions_strip_images_by_default(self):
    output = _capture_output(
        concordia_log.main, ['actions', self._path, 'Alice']
    )
    self.assertNotIn('iVBORw0KGgo', output)
    self.assertIn('[image:', output)
    self.assertIn('bytes]', output)

  def test_actions_include_images(self):
    output = _capture_output(
        concordia_log.main,
        ['--include-images', 'actions', self._path, 'Alice'],
    )
    self.assertIn('iVBORw0KGgo', output)

  def test_dump_strips_images_by_default(self):
    output = _capture_output(concordia_log.main, ['dump', self._path])
    self.assertNotIn('iVBORw0KGgo', output)
    self.assertIn('[image:', output)

  def test_dump_include_images(self):
    output = _capture_output(
        concordia_log.main, ['--include-images', 'dump', self._path]
    )
    self.assertIn('iVBORw0KGgo', output)


class StripImagesTest(absltest.TestCase):

  def test_strip_simple_image(self):
    text = 'Hello ![img](data:image/png;base64,abc123) world'
    result = concordia_log._strip_images(text)
    self.assertNotIn('abc123', result)
    self.assertIn('[img:', result)
    self.assertIn('bytes]', result)
    self.assertIn('Hello', result)
    self.assertIn('world', result)

  def test_strip_no_images(self):
    text = 'No images here, just text.'
    result = concordia_log._strip_images(text)
    self.assertEqual(result, text)

  def test_strip_multiple_images(self):
    text = (
        '![a](data:image/png;base64,abc) and ![b](data:image/jpeg;base64,def)'
    )
    result = concordia_log._strip_images(text)
    self.assertNotIn('abc', result)
    self.assertNotIn('def', result)
    self.assertIn('[a:', result)
    self.assertIn('[b:', result)


if __name__ == '__main__':
  absltest.main()
