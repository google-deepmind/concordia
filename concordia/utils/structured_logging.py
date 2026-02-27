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

"""Structured logging module for efficient, deduplicated simulation logs.

This module provides:
- ContentStore: Content-addressable storage for deduplication
- StructuredLogEntry: Dataclass for log entries with content references
- SimulationLog: Container for simulation logs with multiple output formats
- AIAgentLogInterface: High-level API for AI agents to query logs
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import re
import threading
from typing import Any

from concordia.utils import structured_logging_html

_DEFAULT_MIN_CHUNK_LENGTH = 50

_IMAGE_MD_REGEX = re.compile(
    r'!\[([^\]]*)\]\((data:image/[a-zA-Z0-9.\-+]+;base64,[^)]+)\)'
)
_CONTENT_REF_REGEX = re.compile(r'!\[([^\]]*)\]\(content_ref:([a-f0-9]+)\)')


class ContentStore:
  """Content-addressable storage for log strings.

  Stores content by its hash, enabling deduplication of repeated text
  like observations and memories that appear multiple times in logs.

  Thread-safe for concurrent access.

  Example:
    store = ContentStore()
    id1 = store.add("Hello world")
    id2 = store.add("Hello world")  # Same content, same ID
    assert id1 == id2
    assert store.get(id1) == "Hello world"
  """

  def __init__(self, hash_prefix_length: int = 16):
    """Initialize the content store.

    Args:
      hash_prefix_length: Number of hex characters to use for content IDs.
        Default is 16 (64 bits), which has negligible collision probability for
        typical log sizes.
    """
    self._content_by_hash: dict[str, str] = {}
    self._lock = threading.Lock()
    self._hash_prefix_length = hash_prefix_length

  def add(self, content: str) -> str:
    """Store content and return its hash ID.

    If the content already exists, returns the existing ID without
    storing a duplicate.

    Args:
      content: The string content to store.

    Returns:
      A hex string ID that can be used to retrieve the content.
    """
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    content_id = content_hash[: self._hash_prefix_length]

    with self._lock:
      if content_id not in self._content_by_hash:
        self._content_by_hash[content_id] = content

    return content_id

  def get(self, content_id: str) -> str:
    """Retrieve content by its ID.

    Args:
      content_id: The hash ID returned by add().

    Returns:
      The original content string.

    Raises:
      KeyError: If the content_id is not found.
    """
    with self._lock:
      return self._content_by_hash[content_id]

  def get_or_none(self, content_id: str | None) -> str | None:
    """Retrieve content by ID, returning None if not found or ID is None.

    Args:
      content_id: The hash ID, or None.

    Returns:
      The content string, or None if not found or ID was None.
    """
    if content_id is None:
      return None
    with self._lock:
      return self._content_by_hash.get(content_id)

  def __contains__(self, content_id: str) -> bool:
    """Check if a content ID exists in the store."""
    with self._lock:
      return content_id in self._content_by_hash

  def __len__(self) -> int:
    """Return the number of unique content items stored."""
    with self._lock:
      return len(self._content_by_hash)

  def to_dict(self) -> dict[str, str]:
    """Export all content as a dictionary for serialization."""
    with self._lock:
      return dict(self._content_by_hash)

  @classmethod
  def from_dict(cls, data: Mapping[str, str]) -> ContentStore:
    """Create a ContentStore from a serialized dictionary."""
    store = cls()
    store._content_by_hash = dict(data)
    return store


@dataclasses.dataclass
class StructuredLogEntry:
  """A log entry with content deduplicated via ContentStore references.

  Instead of storing full text inline, this entry stores the original log
  dict with long strings replaced by {'_ref': content_id} references that
  point to the ContentStore. This enables significant storage savings when
  the same content appears in multiple log entries.

  Attributes:
    step: The simulation step number when this entry was created.
    timestamp: ISO format timestamp string.
    entity_name: Name of the entity that generated this entry.
    component_name: Name of the component that logged this entry.
    entry_type: Category of log entry (e.g., 'entity', 'step').
    summary: Short human-readable summary of the entry.
    deduplicated_data: The original log dict with long strings replaced by
      {'_ref': content_id} references. Use SimulationLog.reconstruct_value() to
      get the original data back.
  """

  step: int
  timestamp: str
  entity_name: str
  component_name: str
  entry_type: str
  summary: str = ''
  deduplicated_data: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    """Convert entry to a serializable dictionary."""
    return {
        'step': self.step,
        'timestamp': self.timestamp,
        'entity_name': self.entity_name,
        'component_name': self.component_name,
        'entry_type': self.entry_type,
        'summary': self.summary,
        'deduplicated_data': dict(self.deduplicated_data),
    }

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> StructuredLogEntry:
    """Create an entry from a serialized dictionary."""
    return cls(
        step=data['step'],
        timestamp=data['timestamp'],
        entity_name=data['entity_name'],
        component_name=data['component_name'],
        entry_type=data['entry_type'],
        summary=data.get('summary', ''),
        deduplicated_data=data.get('deduplicated_data', {}),
    )


class SimulationLog:
  """Container for simulation logs with efficient storage and multiple views.

  This class collects log entries during simulation, automatically deduplicating
  content, and provides multiple output formats:
  - JSON for AI agent access
  - HTML for human viewing
  - Raw data for programmatic analysis

  Thread-safe for concurrent logging from multiple components.

  Example:
    log = SimulationLog()
    log.add_entry(
        step=1,
        timestamp='2025-01-01T12:00:00',
        entity_name='Alice',
        component_name='ActComponent',
        entry_type='action',
        summary='Alice said hello',
        raw_data={'Key': 'action', 'Value': 'Alice said hello'},
    )

    # Export in different formats
    json_data = log.to_json()
    html_data = log.to_html()
  """

  def __init__(self):
    """Initialize an empty simulation log."""
    self._content_store = ContentStore()
    self._entries: list[StructuredLogEntry] = []
    self._lock = threading.Lock()

    # Indexes for efficient queries
    self._entity_index: dict[str, list[int]] = {}
    self._step_index: dict[int, list[int]] = {}
    self._component_index: dict[str, list[int]] = {}

    # Optional memory storage (populated when created from Simulation.play)
    self._entity_memories: dict[str, list[str]] = {}
    self._game_master_memories: list[str] = []

  @property
  def content_store(self) -> ContentStore:
    """Access the underlying content store."""
    return self._content_store

  @property
  def entries(self) -> list[StructuredLogEntry]:
    """Access the list of log entries (read-only view recommended)."""
    return self._entries

  def attach_memories(
      self,
      entity_memories: dict[str, list[str]] | None = None,
      game_master_memories: Sequence[str] | None = None,
  ) -> None:
    """Attach entity and game master memories for HTML rendering.

    This method allows memories to be stored with the log so they can be
    rendered to HTML tabs via to_html().

    Args:
      entity_memories: Dict mapping entity names to lists of memory strings.
      game_master_memories: Sequence of game master memory strings.
    """
    if entity_memories is not None:
      self._entity_memories = entity_memories
    if game_master_memories is not None:
      self._game_master_memories = list(game_master_memories)

  def _deduplicate_value(
      self, value: Any, min_length: int = _DEFAULT_MIN_CHUNK_LENGTH
  ) -> Any:
    """Recursively deduplicate strings in a value.

    Scans through the value tree (dicts, lists, strings) and replaces
    strings longer than min_length with {'_ref': content_id} references.
    Embedded images are extracted and stored separately so that identical
    images shared across different text contexts are deduplicated.

    Args:
      value: The value to deduplicate (can be str, list, dict, or other).
      min_length: Minimum string length to deduplicate. Shorter strings are kept
        inline to avoid overhead.

    Returns:
      The value with long strings replaced by references.
    """
    if isinstance(value, str):
      if len(value) >= min_length:
        value = self._extract_and_store_images(value)
        return {'_ref': self._content_store.add(value)}
      return value
    elif isinstance(value, list):
      return [self._deduplicate_value(v, min_length) for v in value]
    elif isinstance(value, dict):
      return {
          k: self._deduplicate_value(v, min_length) for k, v in value.items()
      }
    else:
      try:
        json.dumps(value)
        return value
      except TypeError:
        return str(value)

  def _extract_and_store_images(self, text: str) -> str:
    """Extract embedded images from text, store them, and replace with refs."""

    def _replace_image(match: re.Match[str]) -> str:
      alt = match.group(1)
      data_uri = match.group(2)
      content_id = self._content_store.add(data_uri)
      return f'![{alt}](content_ref:{content_id})'

    return _IMAGE_MD_REGEX.sub(_replace_image, text)

  def reconstruct_value(self, value: Any) -> Any:
    """Recursively reconstruct strings from references.

    Scans through the value tree and replaces {'_ref': content_id}
    references with the original string content from ContentStore.
    Also restores embedded image data from content_ref: placeholders.

    Args:
      value: The value with references to reconstruct.

    Returns:
      The value with references replaced by original strings.
    """
    if isinstance(value, dict):
      if '_ref' in value and len(value) == 1:
        text = self._content_store.get(value['_ref'])
        return self._restore_images(text)
      return {k: self.reconstruct_value(v) for k, v in value.items()}
    elif isinstance(value, list):
      return [self.reconstruct_value(v) for v in value]
    return value

  def _restore_images(self, text: str) -> str:
    """Restore embedded images from content_ref: placeholders."""

    def _replace_ref(match: re.Match[str]) -> str:
      alt = match.group(1)
      content_id = match.group(2)
      data_uri = self._content_store.get_or_none(content_id)
      if data_uri is not None:
        return f'![{alt}]({data_uri})'
      return match.group(0)

    return _CONTENT_REF_REGEX.sub(_replace_ref, text)

  def add_entry(
      self,
      step: int,
      timestamp: str,
      entity_name: str,
      component_name: str,
      entry_type: str,
      summary: str = '',
      raw_data: Mapping[str, Any] | None = None,
  ) -> StructuredLogEntry:
    """Add a log entry, automatically deduplicating content.

    Args:
      step: Simulation step number.
      timestamp: ISO format timestamp.
      entity_name: Name of the entity.
      component_name: Name of the component.
      entry_type: Type of event being logged.
      summary: Short human-readable summary.
      raw_data: The raw log data dict (e.g., {'Key': ..., 'Value': ...}). Long
        strings will be deduplicated automatically.

    Returns:
      The created StructuredLogEntry.
    """
    deduplicated_data = {}
    if raw_data:
      deduplicated_data = self._deduplicate_value(raw_data)

    entry = StructuredLogEntry(
        step=step,
        timestamp=timestamp,
        entity_name=entity_name,
        component_name=component_name,
        entry_type=entry_type,
        summary=summary,
        deduplicated_data=deduplicated_data,
    )

    with self._lock:
      entry_idx = len(self._entries)
      self._entries.append(entry)

      if entity_name not in self._entity_index:
        self._entity_index[entity_name] = []
      self._entity_index[entity_name].append(entry_idx)

      if step not in self._step_index:
        self._step_index[step] = []
      self._step_index[step].append(entry_idx)

      if component_name not in self._component_index:
        self._component_index[component_name] = []
      self._component_index[component_name].append(entry_idx)

    return entry

  def get_entries_by_entity(self, entity_name: str) -> list[StructuredLogEntry]:
    """Get all entries for a specific entity."""
    with self._lock:
      indices = self._entity_index.get(entity_name, [])
      return [self._entries[i] for i in indices]

  def get_entries_by_step(self, step: int) -> list[StructuredLogEntry]:
    """Get all entries for a specific simulation step."""
    with self._lock:
      indices = self._step_index.get(step, [])
      return [self._entries[i] for i in indices]

  def get_entries_by_component(
      self, component_name: str
  ) -> list[StructuredLogEntry]:
    """Get all entries from a specific component."""
    with self._lock:
      indices = self._component_index.get(component_name, [])
      return [self._entries[i] for i in indices]

  def get_entity_names(self) -> list[str]:
    """Get list of all entity names that have log entries."""
    with self._lock:
      return list(self._entity_index.keys())

  def get_steps(self) -> list[int]:
    """Get list of all step numbers with log entries."""
    with self._lock:
      return sorted(self._step_index.keys())

  def get_entity_memories(self, entity_name: str) -> list[str]:
    """Get memories for a specific entity if available.

    Args:
      entity_name: Name of the entity.

    Returns:
      List of memory strings, or empty list if not available.
    """
    return self._entity_memories.get(entity_name, [])

  def get_game_master_memories(self) -> list[str]:
    """Get game master memories if available.

    Returns:
      List of memory strings, or empty list if not available.
    """
    return self._game_master_memories

  def __len__(self) -> int:
    """Return the number of log entries."""
    with self._lock:
      return len(self._entries)

  def to_dict(self) -> dict[str, Any]:
    """Export the full log as a serializable dictionary.

    Returns:
      Dictionary with 'content_store', 'entries', 'entity_memories', and
      'game_master_memories' keys.
    """

    with self._lock:
      result = {
          'content_store': self._content_store.to_dict(),
          'entries': [e.to_dict() for e in self._entries],
      }
      if self._entity_memories:
        result['entity_memories'] = self._entity_memories
      if self._game_master_memories:
        result['game_master_memories'] = self._game_master_memories
      return result

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> SimulationLog:
    """Create a SimulationLog from a serialized dictionary."""
    log = cls()
    log._content_store = ContentStore.from_dict(data['content_store'])

    for entry_data in data['entries']:
      entry = StructuredLogEntry.from_dict(entry_data)
      log._entries.append(entry)

      # Rebuild indexes
      idx = len(log._entries) - 1
      if entry.entity_name not in log._entity_index:
        log._entity_index[entry.entity_name] = []
      log._entity_index[entry.entity_name].append(idx)

      if entry.step not in log._step_index:
        log._step_index[entry.step] = []
      log._step_index[entry.step].append(idx)

      if entry.component_name not in log._component_index:
        log._component_index[entry.component_name] = []
      log._component_index[entry.component_name].append(idx)

    # Load attached memories if present
    if 'entity_memories' in data:
      log._entity_memories = data['entity_memories']
    if 'game_master_memories' in data:
      log._game_master_memories = data['game_master_memories']

    return log

  def to_json(self, indent: int | None = 2) -> str:
    """Export the log as a JSON string.

    Args:
      indent: JSON indentation level. None for compact output.

    Returns:
      JSON string representation of the log.
    """
    return json.dumps(self.to_dict(), indent=indent)

  @classmethod
  def from_json(cls, json_str: str) -> SimulationLog:
    """Create a SimulationLog from a JSON string."""
    return cls.from_dict(json.loads(json_str))

  def get_summary(self) -> dict[str, Any]:
    """Get a high-level summary of the log for AI agent inspection.

    Returns:
      Dictionary with summary statistics and structure info.
    """
    with self._lock:
      entry_types: dict[str, int] = {}
      for entry in self._entries:
        entry_types[entry.entry_type] = entry_types.get(entry.entry_type, 0) + 1

      return {
          'total_entries': len(self._entries),
          'total_steps': len(self._step_index),
          'entities': list(self._entity_index.keys()),
          'components': list(self._component_index.keys()),
          'entry_type_counts': entry_types,
          'unique_content_items': len(self._content_store),
      }

  @classmethod
  def from_raw_log(cls, raw_log: list[Mapping[str, Any]]) -> SimulationLog:
    """Create a SimulationLog from the old raw_log format.

    This method converts the raw simulation log format (list of dicts
    with Step, entity keys, Summary, etc.) into the new structured format.
    Supports both sequential/simultaneous logs and asynchronous logs
    (which include a 'thread' key identifying the entity thread).

    Args:
      raw_log: List of log entries in the raw format.

    Returns:
      A new SimulationLog populated with the raw_log data.
    """

    log = cls()

    for entry in raw_log:
      step = entry.get('Step', 0)
      summary = entry.get('Summary', '')
      thread = entry.get('thread', '')

      for key, value in entry.items():
        if key in ('Step', 'Summary', 'date', 'thread'):
          continue

        if 'Entity' in key:
          entity_name = key.replace('Entity [', '').replace(']', '').strip()
          if entity_name == 'Entity':
            entity_name = thread or 'Unknown'

          raw_data = {'key': key, 'value': value} if value else {}
          if thread:
            raw_data['thread'] = thread

          log.add_entry(
              step=step,
              timestamp='',
              entity_name=entity_name,
              component_name='entity_action',
              entry_type='entity',
              summary=summary,
              raw_data=raw_data,
          )
        else:
          gm_name = key.split(' --- ')[0] if ' --- ' in key else key

          raw_data = {'key': key, 'value': value} if value else {}
          if thread:
            raw_data['thread'] = thread

          log.add_entry(
              step=step,
              timestamp='',
              entity_name=gm_name,
              component_name='game_master',
              entry_type='step',
              summary=summary,
              raw_data=raw_data,
          )

    return log

  def to_html(self, title: str = 'Simulation Log') -> str:
    """Render the log to HTML with JavaScript-based deduplication.

    Args:
      title: Title for the HTML page.

    Returns:
      Complete HTML string with embedded data and dynamic rendering.
    """
    return structured_logging_html.render_dynamic_html(
        simulation_log=self,
        entity_memories=self._entity_memories or None,
        game_master_memories=self._game_master_memories or None,
        title=title,
    )


class AIAgentLogInterface:
  """High-level interface for AI agents to query and analyze simulation logs.

  This class provides a clean, semantic API for AI agents to access log data
  without needing to understand the underlying data structures. It supports:
  - Filtering by entity, step, component, or event type
  - Retrieving full content (prompts, responses) with deduplication resolved
  - Timeline and entity-centric views
  - Summary statistics

  Example:
    log = simulation.play(return_structured_log=True)
    interface = AIAgentLogInterface(log)

    # Get summary of the entire simulation
    print(interface.get_overview())

    # Get all actions by a specific entity
    actions = interface.get_entity_timeline('Alice')

    # Find all entries matching a filter
    observations = interface.filter_entries(entry_type='observation')
  """

  def __init__(self, simulation_log: SimulationLog):
    """Initialize the interface with a SimulationLog.

    Args:
      simulation_log: The log to provide access to.
    """
    self._log = simulation_log

  def get_overview(self) -> dict[str, Any]:
    """Get a high-level overview of the simulation.

    Returns:
      Dictionary with simulation statistics and structure.
    """
    summary = self._log.get_summary()
    return {
        'total_entries': summary['total_entries'],
        'total_steps': summary['total_steps'],
        'entities': summary['entities'],
        'components': summary['components'],
        'entry_types': list(summary['entry_type_counts'].keys()),
        'entry_type_counts': summary['entry_type_counts'],
        'deduplication_savings': (
            f"{summary['unique_content_items']} unique items stored"
        ),
    }

  def get_entity_timeline(
      self,
      entity_name: str,
      include_content: bool = False,
  ) -> list[dict[str, Any]]:
    """Get a chronological timeline of all entries for an entity.

    Args:
      entity_name: Name of the entity.
      include_content: If True, include full prompt/response text.

    Returns:
      List of entry dictionaries in chronological order.
    """
    entries = self._log.get_entries_by_entity(entity_name)
    return [self._entry_to_dict(entry, include_content) for entry in entries]

  def get_step_summary(
      self,
      step: int,
      include_content: bool = False,
  ) -> list[dict[str, Any]]:
    """Get all entries for a specific simulation step.

    Args:
      step: The step number.
      include_content: If True, include full prompt/response text.

    Returns:
      List of entry dictionaries for that step.
    """
    entries = self._log.get_entries_by_step(step)
    return [self._entry_to_dict(entry, include_content) for entry in entries]

  def filter_entries(
      self,
      entity_name: str | None = None,
      component_name: str | None = None,
      entry_type: str | None = None,
      step_range: tuple[int, int] | None = None,
      include_content: bool = False,
  ) -> list[dict[str, Any]]:
    """Filter entries by various criteria.

    Args:
      entity_name: Filter by entity name.
      component_name: Filter by component name.
      entry_type: Filter by entry type.
      step_range: Filter by step range (inclusive).
      include_content: If True, include full reconstructed content.

    Returns:
      List of matching entry dictionaries.
    """
    results = []
    for entry in self._log.entries:
      if entity_name and entry.entity_name != entity_name:
        continue
      if component_name and entry.component_name != component_name:
        continue
      if entry_type and entry.entry_type != entry_type:
        continue
      if step_range:
        if entry.step < step_range[0] or entry.step > step_range[1]:
          continue
      results.append(self._entry_to_dict(entry, include_content))
    return results

  def get_component_values(
      self,
      component_key: str = '__act__',
      value_key: str = 'Value',
      entity_name: str | None = None,
      entry_type: str | None = None,
      step_range: tuple[int, int] | None = None,
  ) -> list[dict[str, Any]]:
    """Extract a specific component value across matching log entries.

    Resolves deduplicated references and navigates into each entry's data
    to extract the value at
    ``deduplicated_data['value'][component_key][value_key]``.

    This is useful for extracting entity actions (component_key='__act__',
    value_key='Value'), resource tracker outputs, or any other named
    component value without manually resolving references.

    Args:
      component_key: The key of the component within the entry's value dict.
        Defaults to '__act__' which holds entity actions.
      value_key: The key within the component dict to extract. Defaults to
        'Value' which holds the action text.
      entity_name: If provided, only include entries for this entity.
      entry_type: If provided, only include entries of this type.
      step_range: If provided, only include entries in this step range
        (inclusive).

    Returns:
      List of dicts, each with 'step', 'entity_name', and 'value' keys.
      Only entries where the requested component and value key exist are
      included.
    """
    results = []
    for entry in self._log.entries:
      if entity_name and entry.entity_name != entity_name:
        continue
      if entry_type and entry.entry_type != entry_type:
        continue
      if step_range:
        if entry.step < step_range[0] or entry.step > step_range[1]:
          continue

      full_data = self._log.reconstruct_value(entry.deduplicated_data)
      value_dict = full_data.get('value', {})
      if not isinstance(value_dict, dict):
        continue

      component = value_dict.get(component_key, {})
      if not isinstance(component, dict):
        continue

      if value_key not in component:
        continue

      results.append({
          'step': entry.step,
          'entity_name': entry.entity_name,
          'value': component[value_key],
      })

    return results

  def get_entry_content(self, entry_index: int) -> dict[str, Any]:
    """Get the full content for a specific entry by index.

    Args:
      entry_index: Index of the entry in the log.

    Returns:
      Dictionary with 'data' containing the reconstructed entry content.

    Raises:
      IndexError: If the entry_index is out of range.
    """
    if entry_index < 0 or entry_index >= len(self._log.entries):
      raise IndexError(f'Entry index {entry_index} out of range')

    entry = self._log.entries[entry_index]

    return {
        'data': self._log.reconstruct_value(entry.deduplicated_data),
    }

  def search_summaries(
      self,
      query: str,
      include_content: bool = False,
  ) -> list[dict[str, Any]]:
    """Search entries by text in the summary field only.

    This is a fast search that only checks the short summary string
    attached to each entry (e.g., 'Step 3 forum_rules --- ...').

    Args:
      query: Text to search for (case-insensitive).
      include_content: If True, include full reconstructed content in results.

    Returns:
      List of matching entry dictionaries.
    """
    query_lower = query.lower()
    results = []
    for entry in self._log.entries:
      if query_lower in entry.summary.lower():
        results.append(self._entry_to_dict(entry, include_content))
    return results

  def search_entries(
      self,
      query: str,
      include_content: bool = True,
  ) -> list[dict[str, Any]]:
    """Search entries by text in all reconstructed content.

    Searches through the full reconstructed content of each entry,
    including prompts, values, observations, component data, etc.
    This is slower than search_summaries() but finds matches in all
    logged data.

    Args:
      query: Text to search for (case-insensitive).
      include_content: If True, include full reconstructed content in results.

    Returns:
      List of matching entry dictionaries.
    """
    query_lower = query.lower()
    results = []
    for entry in self._log.entries:
      full_data = self._log.reconstruct_value(entry.deduplicated_data)
      full_str = json.dumps(full_data, default=str)
      if query_lower in full_str.lower():
        results.append(self._entry_to_dict(entry, include_content))
    return results

  def get_entity_memories(self, entity_name: str) -> list[str]:
    """Get memories for a specific entity if available.

    Args:
      entity_name: Name of the entity.

    Returns:
      List of memory strings, or empty list if not available.
    """
    return self._log.get_entity_memories(entity_name)

  def get_game_master_memories(self) -> list[str]:
    """Get game master memories if available.

    Returns:
      List of memory strings, or empty list if not available.
    """
    return self._log.get_game_master_memories()

  def get_entity_actions(
      self,
      entity_name: str,
  ) -> list[dict[str, Any]]:
    """Get a concise timeline of an entity's actions across all steps.

    Extracts the __act__ Value from each entity entry, providing a
    quick overview of what the entity did at each step.

    Args:
      entity_name: Name of the entity.

    Returns:
      List of dicts with 'step' and 'action' keys.
    """
    results = []
    for entry in self._log.entries:
      if entry.entity_name != entity_name:
        continue
      if entry.entry_type != 'entity':
        continue
      full_data = self._log.reconstruct_value(entry.deduplicated_data)
      value_dict = full_data.get('value', {})
      if not isinstance(value_dict, dict):
        continue
      act_component = value_dict.get('__act__', {})
      if isinstance(act_component, dict) and 'Value' in act_component:
        results.append({
            'step': entry.step,
            'action': act_component['Value'],
        })
    return results

  def get_entity_action_context(
      self,
      entity_name: str,
      step: int,
  ) -> dict[str, Any] | None:
    """Get an entity's full action context at a specific step.

    Returns the entity's action, observations, and prompt at the time
    it produced its action. This is the primary method for understanding
    why an entity did what it did.

    Args:
      entity_name: Name of the entity.
      step: The simulation step to look up.

    Returns:
      Dictionary with 'step', 'entity_name', 'action', 'action_prompt',
      'observations', and 'all_components' keys, or None if no matching
      entry was found.
    """
    for entry in self._log.entries:
      if (
          entry.entity_name != entity_name
          or entry.entry_type != 'entity'
          or entry.step != step
      ):
        continue

      full_data = self._log.reconstruct_value(entry.deduplicated_data)
      value_dict = full_data.get('value', {})
      if not isinstance(value_dict, dict):
        continue

      act_component = value_dict.get('__act__', {})
      obs_component = value_dict.get('__observation__', {})

      action = ''
      action_prompt = ''
      if isinstance(act_component, dict):
        action = act_component.get('Value', '')
        action_prompt = act_component.get('Prompt', '')

      observations = []
      if isinstance(obs_component, dict):
        obs_value = obs_component.get('Value', [])
        if isinstance(obs_value, list):
          observations = obs_value
        elif obs_value:
          observations = [obs_value]

      # Collect all component names and their values
      all_components = {}
      for comp_name, comp_data in value_dict.items():
        if isinstance(comp_data, dict) and 'Value' in comp_data:
          all_components[comp_name] = comp_data['Value']

      return {
          'step': step,
          'entity_name': entity_name,
          'action': action,
          'action_prompt': action_prompt,
          'observations': observations,
          'all_components': all_components,
      }

    return None

  def _entry_to_dict(
      self,
      entry: StructuredLogEntry,
      include_content: bool = False,
  ) -> dict[str, Any]:
    """Convert an entry to a dictionary representation.

    Args:
      entry: The entry to convert.
      include_content: If True, resolve content IDs to actual text.

    Returns:
      Dictionary representation of the entry.
    """
    result: dict[str, Any] = {
        'step': entry.step,
        'timestamp': entry.timestamp,
        'entity_name': entry.entity_name,
        'component_name': entry.component_name,
        'entry_type': entry.entry_type,
        'summary': entry.summary,
        'deduplicated_data': dict(entry.deduplicated_data),
    }

    if include_content:
      result['data'] = self._log.reconstruct_value(entry.deduplicated_data)

    return result
