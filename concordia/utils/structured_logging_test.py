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

"""Tests for structured_logging module."""

import json
import threading

from absl.testing import absltest
from concordia.utils import structured_logging


class ContentStoreTest(absltest.TestCase):
  """Tests for ContentStore class."""

  def test_add_returns_id(self):
    """Adding content returns a string ID."""
    store = structured_logging.ContentStore()
    content_id = store.add('test content')
    self.assertIsInstance(content_id, str)
    self.assertNotEmpty(content_id)

  def test_same_content_same_id(self):
    """Adding the same content twice returns the same ID."""
    store = structured_logging.ContentStore()
    id1 = store.add('duplicate content')
    id2 = store.add('duplicate content')
    self.assertEqual(id1, id2)

  def test_different_content_different_id(self):
    """Different content gets different IDs."""
    store = structured_logging.ContentStore()
    id1 = store.add('content one')
    id2 = store.add('content two')
    self.assertNotEqual(id1, id2)

  def test_get_retrieves_content(self):
    """get() returns the original content."""
    store = structured_logging.ContentStore()
    content = 'hello world'
    content_id = store.add(content)
    self.assertEqual(store.get(content_id), content)

  def test_get_unknown_id_raises(self):
    """get() raises KeyError for unknown ID."""
    store = structured_logging.ContentStore()
    with self.assertRaises(KeyError):
      store.get('nonexistent_id')

  def test_get_or_none_returns_none_for_none(self):
    """get_or_none(None) returns None."""
    store = structured_logging.ContentStore()
    self.assertIsNone(store.get_or_none(None))

  def test_get_or_none_returns_none_for_unknown(self):
    """get_or_none() returns None for unknown ID."""
    store = structured_logging.ContentStore()
    self.assertIsNone(store.get_or_none('unknown'))

  def test_get_or_none_returns_content(self):
    """get_or_none() returns content for valid ID."""
    store = structured_logging.ContentStore()
    content_id = store.add('test')
    self.assertEqual(store.get_or_none(content_id), 'test')

  def test_contains(self):
    """__contains__ returns True for stored content."""
    store = structured_logging.ContentStore()
    content_id = store.add('test')
    self.assertIn(content_id, store)
    self.assertNotIn('fake_id', store)

  def test_len(self):
    """__len__ returns number of unique items."""
    store = structured_logging.ContentStore()
    self.assertEmpty(store)
    store.add('one')
    self.assertLen(store, 1)
    store.add('two')
    self.assertLen(store, 2)
    store.add('one')  # Duplicate
    self.assertLen(store, 2)

  def test_to_dict(self):
    """to_dict exports all content."""
    store = structured_logging.ContentStore()
    id1 = store.add('content1')
    id2 = store.add('content2')
    data = store.to_dict()
    self.assertEqual(data[id1], 'content1')
    self.assertEqual(data[id2], 'content2')

  def test_from_dict(self):
    """from_dict creates store from exported data."""
    original = structured_logging.ContentStore()
    id1 = original.add('test1')
    id2 = original.add('test2')

    restored = structured_logging.ContentStore.from_dict(original.to_dict())
    self.assertEqual(restored.get(id1), 'test1')
    self.assertEqual(restored.get(id2), 'test2')

  def test_unicode_content(self):
    """Unicode content is handled correctly."""
    store = structured_logging.ContentStore()
    content = '你好世界 🌍 émojis'
    content_id = store.add(content)
    self.assertEqual(store.get(content_id), content)

  def test_concurrent_add(self):
    """Concurrent adds are thread-safe."""
    store = structured_logging.ContentStore()
    results = []

    def add_content(i):
      content_id = store.add(f'content_{i}')
      results.append((i, content_id))

    threads = [
        threading.Thread(target=add_content, args=(i,)) for i in range(100)
    ]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertLen(results, 100)
    self.assertLen(store, 100)


class StructuredLogEntryTest(absltest.TestCase):
  """Tests for StructuredLogEntry dataclass."""

  def test_create_minimal_entry(self):
    """Create entry with required fields only."""
    entry = structured_logging.StructuredLogEntry(
        step=1,
        timestamp='2024-01-01T10:00:00',
        entity_name='Alice',
        component_name='ActComponent',
        entry_type='action',
    )
    self.assertEqual(entry.step, 1)
    self.assertEqual(entry.entity_name, 'Alice')
    self.assertEqual(entry.summary, '')
    self.assertEqual(entry.deduplicated_data, {})

  def test_create_full_entry(self):
    """Create entry with all fields."""
    entry = structured_logging.StructuredLogEntry(
        step=5,
        timestamp='2024-01-01T10:00:00',
        entity_name='Bob',
        component_name='ObservationComponent',
        entry_type='observation',
        summary='Bob observed something',
        deduplicated_data={'Key': 'value', 'nested': {'_ref': 'abc123'}},
    )
    self.assertEqual(entry.deduplicated_data['Key'], 'value')
    self.assertEqual(entry.deduplicated_data['nested'], {'_ref': 'abc123'})

  def test_to_dict(self):
    """to_dict produces serializable dictionary."""
    entry = structured_logging.StructuredLogEntry(
        step=1,
        timestamp='2024-01-01T10:00:00',
        entity_name='Alice',
        component_name='Act',
        entry_type='action',
        summary='test',
        deduplicated_data={'prompt': {'_ref': 'p1'}},
    )
    data = entry.to_dict()
    self.assertEqual(data['step'], 1)
    self.assertEqual(data['entity_name'], 'Alice')
    self.assertEqual(data['deduplicated_data']['prompt'], {'_ref': 'p1'})
    # Should be JSON serializable
    json.dumps(data)

  def test_from_dict(self):
    """from_dict recreates entry."""
    original = structured_logging.StructuredLogEntry(
        step=2,
        timestamp='2024-01-01',
        entity_name='Test',
        component_name='Comp',
        entry_type='test',
        deduplicated_data={'context': ['a', 'b']},
    )
    restored = structured_logging.StructuredLogEntry.from_dict(
        original.to_dict()
    )
    self.assertEqual(restored.step, 2)
    self.assertEqual(restored.deduplicated_data['context'], ['a', 'b'])

  def test_from_dict_with_missing_optional_fields(self):
    """from_dict handles missing optional fields."""
    data = {
        'step': 1,
        'timestamp': '2024-01-01',
        'entity_name': 'E',
        'component_name': 'C',
        'entry_type': 'e',
    }
    entry = structured_logging.StructuredLogEntry.from_dict(data)
    self.assertEqual(entry.summary, '')
    self.assertEqual(entry.deduplicated_data, {})


class SimulationLogTest(absltest.TestCase):
  """Tests for SimulationLog class."""

  def test_empty_log(self):
    """New log is empty."""
    log = structured_logging.SimulationLog()
    self.assertEmpty(log)
    self.assertEqual(log.get_entity_names(), [])

  def test_add_entry(self):
    """Adding entry increases log size."""
    log = structured_logging.SimulationLog()
    log.add_entry(
        step=1,
        timestamp='2024-01-01T10:00:00',
        entity_name='Alice',
        component_name='Act',
        entry_type='action',
    )
    self.assertLen(log, 1)

  def test_content_deduplication(self):
    """Same content is deduplicated."""
    log = structured_logging.SimulationLog()
    # Must be >50 chars to trigger deduplication
    repeated_content = 'A' * 100

    log.add_entry(
        step=1,
        timestamp='t1',
        entity_name='A',
        component_name='C',
        entry_type='e',
        raw_data={'prompt': repeated_content},
    )
    log.add_entry(
        step=2,
        timestamp='t2',
        entity_name='A',
        component_name='C',
        entry_type='e',
        raw_data={'prompt': repeated_content},
    )

    # Only one unique content string stored
    self.assertLen(log.content_store, 1)
    # But two entries
    self.assertLen(log, 2)

  def test_get_entries_by_entity(self):
    """Can query entries by entity name."""
    log = structured_logging.SimulationLog()
    log.add_entry(
        step=1,
        timestamp='t',
        entity_name='Alice',
        component_name='C',
        entry_type='e',
    )
    log.add_entry(
        step=2,
        timestamp='t',
        entity_name='Bob',
        component_name='C',
        entry_type='e',
    )
    log.add_entry(
        step=3,
        timestamp='t',
        entity_name='Alice',
        component_name='C',
        entry_type='e',
    )

    alice_entries = log.get_entries_by_entity('Alice')
    self.assertLen(alice_entries, 2)
    self.assertTrue(all(e.entity_name == 'Alice' for e in alice_entries))

  def test_get_entries_by_step(self):
    """Can query entries by step number."""
    log = structured_logging.SimulationLog()
    log.add_entry(
        step=1,
        timestamp='t',
        entity_name='A',
        component_name='C1',
        entry_type='e',
    )
    log.add_entry(
        step=1,
        timestamp='t',
        entity_name='A',
        component_name='C2',
        entry_type='e',
    )
    log.add_entry(
        step=2,
        timestamp='t',
        entity_name='A',
        component_name='C1',
        entry_type='e',
    )

    step1_entries = log.get_entries_by_step(1)
    self.assertLen(step1_entries, 2)

  def test_get_entries_by_component(self):
    """Can query entries by component name."""
    log = structured_logging.SimulationLog()
    log.add_entry(
        step=1,
        timestamp='t',
        entity_name='A',
        component_name='ActComponent',
        entry_type='action',
    )
    log.add_entry(
        step=1,
        timestamp='t',
        entity_name='A',
        component_name='ObsComponent',
        entry_type='observation',
    )

    act_entries = log.get_entries_by_component('ActComponent')
    self.assertLen(act_entries, 1)
    self.assertEqual(act_entries[0].component_name, 'ActComponent')

  def test_get_entity_names(self):
    """get_entity_names returns all entities."""
    log = structured_logging.SimulationLog()
    log.add_entry(
        step=1,
        timestamp='t',
        entity_name='Alice',
        component_name='C',
        entry_type='e',
    )
    log.add_entry(
        step=1,
        timestamp='t',
        entity_name='Bob',
        component_name='C',
        entry_type='e',
    )

    names = log.get_entity_names()
    self.assertIn('Alice', names)
    self.assertIn('Bob', names)

  def test_get_steps(self):
    """get_steps returns sorted step numbers."""
    log = structured_logging.SimulationLog()
    log.add_entry(
        step=3,
        timestamp='t',
        entity_name='A',
        component_name='C',
        entry_type='e',
    )
    log.add_entry(
        step=1,
        timestamp='t',
        entity_name='A',
        component_name='C',
        entry_type='e',
    )
    log.add_entry(
        step=2,
        timestamp='t',
        entity_name='A',
        component_name='C',
        entry_type='e',
    )

    steps = log.get_steps()
    self.assertEqual(steps, [1, 2, 3])

  def test_to_dict_and_from_dict(self):
    """Log can be serialized and deserialized."""
    log = structured_logging.SimulationLog()
    log.add_entry(
        step=1,
        timestamp='2024-01-01',
        entity_name='Alice',
        component_name='Act',
        entry_type='action',
        summary='Test',
        raw_data={'prompt': 'What?', 'response': 'Answer'},
    )

    data = log.to_dict()
    restored = structured_logging.SimulationLog.from_dict(data)

    self.assertLen(restored, 1)
    self.assertEqual(restored.entries[0].summary, 'Test')

    # Content should be restored
    dedup_data = restored.entries[0].deduplicated_data
    prompt_ref = dedup_data.get('prompt', {})
    if isinstance(prompt_ref, dict) and '_ref' in prompt_ref:
      self.assertEqual(restored.content_store.get(prompt_ref['_ref']), 'What?')
    else:
      self.assertEqual(prompt_ref, 'What?')

  def test_to_json_and_from_json(self):
    """Log can be exported to and restored from JSON."""
    log = structured_logging.SimulationLog()
    log.add_entry(
        step=1,
        timestamp='t',
        entity_name='A',
        component_name='C',
        entry_type='e',
        raw_data={'prompt': 'Hello'},
    )

    json_str = log.to_json()
    self.assertIsInstance(json_str, str)

    restored = structured_logging.SimulationLog.from_json(json_str)
    self.assertLen(restored, 1)

  def test_get_summary(self):
    """get_summary returns useful statistics."""
    log = structured_logging.SimulationLog()
    log.add_entry(
        step=1,
        timestamp='t',
        entity_name='Alice',
        component_name='C',
        entry_type='action',
    )
    log.add_entry(
        step=1,
        timestamp='t',
        entity_name='Bob',
        component_name='C',
        entry_type='observation',
    )
    log.add_entry(
        step=2,
        timestamp='t',
        entity_name='Alice',
        component_name='C',
        entry_type='action',
    )

    summary = log.get_summary()
    self.assertEqual(summary['total_entries'], 3)
    self.assertEqual(summary['total_steps'], 2)
    self.assertIn('Alice', summary['entities'])
    self.assertIn('Bob', summary['entities'])
    self.assertEqual(summary['entry_type_counts']['action'], 2)
    self.assertEqual(summary['entry_type_counts']['observation'], 1)

  def test_concurrent_add_entries(self):
    """Concurrent entry additions are thread-safe."""
    log = structured_logging.SimulationLog()

    def add_entries(entity_name):
      for i in range(50):
        log.add_entry(
            step=i,
            timestamp=f't{i}',
            entity_name=entity_name,
            component_name='C',
            entry_type='e',
            raw_data={'prompt': f'Prompt for {entity_name} step {i}'},
        )

    threads = [
        threading.Thread(target=add_entries, args=(name,))
        for name in ['Alice', 'Bob', 'Charlie']
    ]

    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertLen(log, 150)
    self.assertLen(log.get_entries_by_entity('Alice'), 50)

  def test_reconstruct_value(self):
    """reconstruct_value resolves references back to strings."""
    log = structured_logging.SimulationLog()
    log.add_entry(
        step=1,
        timestamp='t',
        entity_name='A',
        component_name='C',
        entry_type='e',
        raw_data={'long_text': 'A' * 100},
    )

    entry = log.entries[0]
    reconstructed = log.reconstruct_value(entry.deduplicated_data)
    self.assertEqual(reconstructed['long_text'], 'A' * 100)


class FromRawLogTest(absltest.TestCase):
  """Tests for from_raw_log method."""

  def test_from_raw_log_empty(self):
    """from_raw_log handles empty list."""
    log = structured_logging.SimulationLog.from_raw_log([])
    self.assertEmpty(log)

  def test_from_raw_log_basic_entry(self):
    """from_raw_log parses basic raw_log entries."""
    raw_log = [
        {
            'Step': 1,
            'Summary': 'Alice acted',
            'Entity [Alice]': {'Action': 'said hello'},
            'game_master': {'resolve': {'key': 'value'}},
        },
    ]
    log = structured_logging.SimulationLog.from_raw_log(raw_log)
    # Should have entries for both entity and game_master keys
    self.assertGreaterEqual(len(log), 2)

  def test_from_raw_log_preserves_step(self):
    """from_raw_log preserves step numbers."""
    raw_log = [
        {'Step': 5, 'Summary': 'Test', 'Entity [Bob]': {}},
    ]
    log = structured_logging.SimulationLog.from_raw_log(raw_log)
    self.assertEqual(log.get_steps(), [5])

  def test_from_raw_log_extracts_entity_name(self):
    """from_raw_log extracts entity name from key."""
    raw_log = [
        {'Step': 1, 'Summary': 'Test', 'Entity [Alice]': {}},
    ]
    log = structured_logging.SimulationLog.from_raw_log(raw_log)
    self.assertIn('Alice', log.get_entity_names())


class AIAgentLogInterfaceTest(absltest.TestCase):
  """Tests for AIAgentLogInterface class."""

  def _create_sample_log(self) -> structured_logging.SimulationLog:
    """Create a sample log for testing."""
    log = structured_logging.SimulationLog()
    log.add_entry(
        step=1,
        timestamp='t1',
        entity_name='Alice',
        component_name='ActComponent',
        entry_type='action',
        summary='Alice said hello',
        raw_data={'prompt': 'What does Alice do?', 'response': 'said hello'},
    )
    log.add_entry(
        step=1,
        timestamp='t1',
        entity_name='Bob',
        component_name='ObsComponent',
        entry_type='observation',
        summary='Bob observed Alice',
    )
    log.add_entry(
        step=2,
        timestamp='t2',
        entity_name='Alice',
        component_name='ActComponent',
        entry_type='action',
        summary='Alice walked away',
    )
    return log

  def test_get_overview(self):
    """get_overview returns simulation statistics."""
    log = self._create_sample_log()
    interface = structured_logging.AIAgentLogInterface(log)

    overview = interface.get_overview()

    self.assertEqual(overview['total_entries'], 3)
    self.assertEqual(overview['total_steps'], 2)
    self.assertIn('Alice', overview['entities'])
    self.assertIn('Bob', overview['entities'])

  def test_get_entity_timeline(self):
    """get_entity_timeline returns entries for an entity."""
    log = self._create_sample_log()
    interface = structured_logging.AIAgentLogInterface(log)

    timeline = interface.get_entity_timeline('Alice')

    self.assertLen(timeline, 2)
    self.assertTrue(all(e['entity_name'] == 'Alice' for e in timeline))

  def test_get_entity_timeline_with_content(self):
    """get_entity_timeline with include_content=True includes text."""
    log = self._create_sample_log()
    interface = structured_logging.AIAgentLogInterface(log)

    timeline = interface.get_entity_timeline('Alice', include_content=True)

    # First entry has prompt/response in deduplicated_data
    self.assertIn('deduplicated_data', timeline[0])

  def test_get_step_summary(self):
    """get_step_summary returns all entries for a step."""
    log = self._create_sample_log()
    interface = structured_logging.AIAgentLogInterface(log)

    step1 = interface.get_step_summary(1)

    self.assertLen(step1, 2)

  def test_filter_entries_by_entry_type(self):
    """filter_entries can filter by entry_type."""
    log = self._create_sample_log()
    interface = structured_logging.AIAgentLogInterface(log)

    actions = interface.filter_entries(entry_type='action')

    self.assertLen(actions, 2)
    self.assertTrue(all(e['entry_type'] == 'action' for e in actions))

  def test_filter_entries_by_step_range(self):
    """filter_entries can filter by step range."""
    log = self._create_sample_log()
    interface = structured_logging.AIAgentLogInterface(log)

    step1_only = interface.filter_entries(step_range=(1, 1))

    self.assertLen(step1_only, 2)

  def test_search_entries(self):
    """search_entries finds entries by summary text."""
    log = self._create_sample_log()
    interface = structured_logging.AIAgentLogInterface(log)

    results = interface.search_entries('hello')

    self.assertLen(results, 1)
    self.assertIn('hello', results[0]['summary'])

  def test_search_entries_case_insensitive(self):
    """search_entries is case-insensitive."""
    log = self._create_sample_log()
    interface = structured_logging.AIAgentLogInterface(log)

    results = interface.search_entries('ALICE')

    self.assertNotEmpty(results)

  def test_get_entry_content(self):
    """get_entry_content retrieves full content for an entry."""
    log = self._create_sample_log()
    interface = structured_logging.AIAgentLogInterface(log)

    content = interface.get_entry_content(0)

    # The interface returns 'data' with reconstructed values
    self.assertIn('data', content)

  def test_get_entry_content_out_of_range(self):
    """get_entry_content raises IndexError for invalid index."""
    log = self._create_sample_log()
    interface = structured_logging.AIAgentLogInterface(log)

    with self.assertRaises(IndexError):
      interface.get_entry_content(100)


if __name__ == '__main__':
  absltest.main()
