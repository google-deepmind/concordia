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

"""Local artifact data source for the Concordia map dashboard & Studio.

Reads simulation artifacts from a local run directory (`simulation_state.json`,
`location_history.json`, `simulation_structured.json`, `entity_memories.json`,
`entity_states.json`, `raw_log.json`, or the per-agent `*_memories.json` files
in `data/paper_runs/`).

When a run has no `location_history.json` (the paper runs do not), agent
trajectories are reconstructed from the `// place [time]` tags on observation
memories and carried forward between observations; each snapshot entry's `age`
is the number of ticks since that agent was last observed.

Failures are reported explicitly in `error` and `partial_errors`, never hidden
behind synthetic fallback data.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import threading
import time
import traceback
from typing import Any

_STOPWORDS = frozenset({
    'their', 'her', 'his', 'my', 'our', 'the', 'a', 'an', 'another', 'at',
    'home', 'apartment', 'current', 'location', 'desk', 'room', 'place',
    'is', 'allows', 'its', 'event', 'schedule', 'system', 'announcement',
})

_ALIASES = {
    'brecksville': 'brecksville_commons',
    'millbrook': 'millbrook_apts',
    'chippewa': 'chippewa_ridge',
    'riverview': 'riverview_estates',
    'timber': 'timber_creek',
    'sunset': 'sunset_apartments_common_room',
    'swell': 'swell_bar',
}

_MOVE_RE = re.compile(
    r'(?:stay at|move to|go to|head to|walk to|remain at|going to|went to|'
    r'visit|heading to|current location,?)\s+'
    r'(?:the\s+)?(?:their\s+|her\s+|his\s+|our\s+|my\s+)?'
    r'(?:apartment in\s+|home in\s+|unit in\s+|desk at\s+|desk in\s+)?'
    r'([a-z0-9_]+)',
    re.IGNORECASE)

_OBS_RE = re.compile(r'//\s*([a-z_0-9]+)\s*\[')
_STATE_RE = re.compile(r'"State"\s*:\s*"([^"]+)"')
_FULL_TS_RE = re.compile(
    r'//\s*([a-z_0-9]+)\s*\[([A-Za-z]+),?\s+([A-Za-z]+)\s+(\d+)(?:st|nd|rd|th)?,?\s+'
    r'(\d+):(\d+)\s+([AP]M)\]'
)


def _normalise_location(raw: str) -> str | None:
  loc = (raw or '').lower().strip()
  if not loc or loc in _STOPWORDS:
    return None
  loc = re.sub(r'_unit_\d*$', '', loc)
  return _ALIASES.get(loc, loc)


def parse_start_time(text: str) -> datetime.datetime:
  """Parses the --start_time flag into a real datetime."""
  text = (text or '').strip()
  if not text:
    return datetime.datetime(2025, 1, 1, 7, 0)
  try:
    return datetime.datetime.fromisoformat(text)
  except ValueError:
    pass

  m = re.search(
      r'(?:(?P<wd>\w+),\s*)?(?P<mon>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?'
      r'(?:,\s*(?P<hour>\d{1,2}):(?P<min>\d{2})\s*(?P<ampm>[AaPp][Mm])?)?',
      text)
  if not m:
    return datetime.datetime(2025, 1, 1, 7, 0)
  months = {mn.lower(): i for i, mn in enumerate(
      ['January', 'February', 'March', 'April', 'May', 'June', 'July',
       'August', 'September', 'October', 'November', 'December'], start=1)}
  month = months.get(m.group('mon').lower(), 1)
  day = int(m.group('day'))
  hour = int(m.group('hour') or 7)
  minute = int(m.group('min') or 0)
  ampm = (m.group('ampm') or '').lower()
  if ampm == 'pm' and hour != 12:
    hour += 12
  elif ampm == 'am' and hour == 12:
    hour = 0

  wd = (m.group('wd') or '').lower()
  weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
              'saturday', 'sunday']
  if wd in weekdays:
    want = weekdays.index(wd)
    for year in range(2024, 2041):
      try:
        cand = datetime.datetime(year, month, day, hour, minute)
      except ValueError:
        continue
      if cand.weekday() == want:
        return cand
  return datetime.datetime(2025, month, day, hour, minute)


def _extract_location(content: str) -> str | None:
  """Recovers a place id from a log or memory content string."""
  m = _OBS_RE.search(content)
  if m:
    raw_loc = m.group(1).lower()
    if raw_loc == 'marketplace':
      return None
    loc = _normalise_location(raw_loc)
    if loc:
      return loc

  try:
    parsed = json.loads(content)
  except (json.JSONDecodeError, TypeError):
    parsed = None
  if isinstance(parsed, dict):
    for key, entity in parsed.items():
      if not key.startswith('Entity [') or not isinstance(entity, dict):
        continue
      for comp in ('CurrentLocation', 'DynamicLocation'):
        val = entity.get(comp)
        if isinstance(val, dict):
          cand = val.get('current_location') or val.get('Value')
          if isinstance(cand, str):
            loc = _normalise_location(cand)
            if loc:
              return loc
        elif isinstance(val, str) and val:
          loc = _normalise_location(val)
          if loc:
            return loc
      md = entity.get('MovementDecision')
      if isinstance(md, dict):
        mm = _MOVE_RE.search(str(md.get('State', '')))
        if mm:
          loc = _normalise_location(mm.group(1))
          if loc:
            return loc
      break

  sm = _STATE_RE.search(content)
  if sm:
    mm = _MOVE_RE.search(sm.group(1))
    if mm:
      return _normalise_location(mm.group(1))
  return None


# The paper runs used the concordia_island place ids (sunset_apartments,
# coral_village, ...), so they are drawn on that atlas.
DEFAULT_RUN_GEOGRAPHY = 'concordia_island'


def default_run_dir() -> str:
  """Returns the bundled Gemini 2.5 Flash ESA job-loss paper run if present."""
  here = os.path.dirname(os.path.abspath(__file__))
  cand = os.path.join(
      os.path.dirname(here),
      'data',
      'paper_runs',
      'job_loss',
      'gemini_2_5_flash_esa',
  )
  return cand if os.path.isdir(cand) else ''


class LocalFileSource:
  """Reads simulation state and location history from local run files."""

  def __init__(
      self,
      run_id: str = '',
      db: str = '',
      expected_agents: int = 20,
      expected_ticks: int = 80,
      run_dir: str = '',
      data_ttl: float = 8.0,
      history_ttl: float = 60.0,
  ):
    self.run_dir = run_dir or ('' if run_id else default_run_dir())
    self.run_id = run_id or os.path.basename(self.run_dir.rstrip('/'))
    self.db = db
    self.expected_agents = expected_agents
    self.expected_ticks = expected_ticks
    self._data_ttl = data_ttl
    self._history_ttl = history_ttl
    self._lock = threading.Lock()
    self._data = None
    self._data_at = 0.0
    self._hist_lock = threading.Lock()
    self._hist = None
    self._hist_at = 0.0

  @property
  def enabled(self) -> bool:
    """Returns True if a run directory or run ID is configured."""
    return bool(self.run_dir or self.run_id)

  def _resolve_dir(self) -> str:
    """Resolves the active run directory on disk."""
    if self.run_dir and os.path.isdir(self.run_dir):
      return self.run_dir
    if self.run_id:
      for cand in (
          os.path.join('./local_runs', self.run_id),
          os.path.join('./local_runs', f'run_{self.run_id}'),
      ):
        if os.path.isdir(cand):
          return cand
    return self.run_dir

  def data(self) -> dict[str, Any]:
    """Returns cached or freshly loaded summary data for the run."""
    now = time.time()
    with self._lock:
      if self._data and (now - self._data_at) < self._data_ttl:
        return self._data
    fresh = self._fetch_data()
    with self._lock:
      self._data = fresh
      self._data_at = time.time()
    return fresh

  def _fetch_data(self) -> dict[str, Any]:
    """Loads run summary, locations, conversations, and events from disk."""
    rdir = self._resolve_dir()
    out: dict[str, Any] = {
        'enabled': bool(rdir and os.path.isdir(rdir)),
        'run_id': self.run_id,
        'expected_agents': self.expected_agents,
        'expected_ticks': self.expected_ticks,
        'server_time': datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        'error': None,
        'partial_errors': [],
        'tick_progress': [],
        'agent_locations': [],
        'conversations': [],
        'location_events': [],
        'sim_clock': None,
        'clock': None,
        'run_info': None,
        'unique_agents': 0,
        'total_entries': 0,
        'steps_per_minute': [],
        'recent_activity': [],
    }
    if not rdir or not os.path.isdir(rdir):
      out['error'] = (
          f'Run directory not found ({rdir!r}). Pass --run_dir=/path/to/run '
          'or run python -m examples.concordia_island.run first.'
      )
      return out

    try:
      hist = self.location_history()
      state_path = os.path.join(rdir, 'simulation_state.json')
      perf_path = os.path.join(rdir, 'performance.json')
      state = {}
      if os.path.isfile(state_path):
        with open(state_path, 'r', encoding='utf-8') as f:
          state = json.load(f)
      perf = {}
      if os.path.isfile(perf_path):
        with open(perf_path, 'r', encoding='utf-8') as f:
          perf = json.load(f)

      latest = hist.get('latest_locations', [])
      if not latest and isinstance(state.get('locations'), dict):
        latest = [
            {'agent': a, 'location': l, 'updated': state.get('final_sim_time')}
            for a, l in sorted(state['locations'].items())
        ]
      else:
        latest = [
            {
                'agent': item['agent'],
                'location': item['location'],
                'updated': state.get('final_sim_time'),
            }
            for item in latest
        ]

      tp = state.get('tick_progress')
      if not tp and hist.get('ticks'):
        tp = [
            {'step': t, 'agents': len(hist['snapshots'].get(str(t), []))}
            for t in hist['ticks']
        ]
      tp = tp or []

      last_hist_tick = hist['ticks'][-1] if hist.get('ticks') else 0
      max_ticks = int(
          state.get('ticks') or last_hist_tick or self.expected_ticks
      )
      cur_tick = int(
          state.get('final_tick', hist['ticks'][-1] if hist.get('ticks') else 0)
          or 0
      )
      unique_agents = int(state.get('agents', len(latest)) or len(latest))
      total_entries = int(
          state.get('log_entries', perf.get('total_steps', 0)) or 0
      )

      clock_obj = {
          'current_tick': cur_tick,
          'max_ticks': max_ticks,
          'acted_count': unique_agents,
          'tick_start': state.get('final_sim_time', state.get('start_time')),
      }
      out['expected_agents'] = unique_agents or self.expected_agents
      out['expected_ticks'] = max_ticks
      out['tick_progress'] = tp
      out['agent_locations'] = latest
      out['conversations'] = state.get('conversations', [])
      out['location_events'] = state.get('location_events', [])
      out['sim_clock'] = clock_obj
      out['clock'] = clock_obj
      out['unique_agents'] = unique_agents
      out['total_entries'] = total_entries
      out['run_info'] = {
          'name': f'Local Run ({os.path.basename(rdir.rstrip("/"))})',
          'model_config': perf.get('agent_prefab', 'local'),
          'expected_agents': unique_agents,
          'total_steps': total_entries,
          'elapsed_seconds': state.get(
              'elapsed_seconds', perf.get('wall_time_seconds')
          ),
          'created_at': state.get('start_time'),
      }
      for ev in out['location_events'][:25]:
        out['recent_activity'].append({
            'step': cur_tick,
            'agent': ev.get('agent', ''),
            'component': 'Observation',
            'snippet': ev.get('text', '')[:200],
            'time': ev.get('time', ''),
        })
    except Exception:  # pylint: disable=broad-except
      out['error'] = (
          f'Error reading local run directory {rdir!r}:\n'
          f'{traceback.format_exc()}'
      )
    return out

  def location_history(self) -> dict[str, Any]:
    """Returns cached or freshly loaded per-tick agent location snapshots."""
    now = time.time()
    with self._hist_lock:
      if self._hist and (now - self._hist_at) < self._history_ttl:
        return self._hist
    fresh = self._fetch_history()
    with self._hist_lock:
      self._hist = fresh
      self._hist_at = time.time()
    return fresh

  def _fetch_history(self) -> dict[str, Any]:
    """Loads or reconstructs per-tick location snapshots from disk."""
    rdir = self._resolve_dir()
    out: dict[str, Any] = {
        'ticks': [],
        'snapshots': {},
        'latest_locations': [],
        'error': None,
        'observed_counts': {},
    }
    if not rdir or not os.path.isdir(rdir):
      out['error'] = f'Run directory not found ({rdir!r}).'
      return out

    loc_hist_path = os.path.join(rdir, 'location_history.json')
    if os.path.isfile(loc_hist_path):
      try:
        with open(loc_hist_path, 'r', encoding='utf-8') as f:
          loc_data = json.load(f)
        part2_path = os.path.join(rdir, 'location_history_part2.json')
        if os.path.isfile(part2_path):
          with open(part2_path, 'r', encoding='utf-8') as pf:
            part2_data = json.load(pf)
          if isinstance(loc_data.get('snapshots'), dict) and isinstance(
              part2_data.get('snapshots'), dict
          ):
            loc_data['snapshots'].update(part2_data['snapshots'])
        return loc_data
      except Exception:  # pylint: disable=broad-except
        out['error'] = f'Malformed {loc_hist_path}:\n{traceback.format_exc()}'
        return out

    # No saved trajectory: reconstruct one from the "// place [time]" tags in
    # the agents' observation memories. Sources, in order of preference:
    # entity_memories.json, simulation_structured.json, or one
    # <first>_<last>_memories.json file per agent (the data/paper_runs format).
    per_tick: dict[int, dict[str, str]] = {}
    hours_order = {7: 0, 9: 1, 11: 2, 13: 3, 15: 4, 17: 5, 19: 6, 21: 7}
    mems_dict = {}
    read_errors = []
    for cand in ('entity_memories.json', 'simulation_structured.json'):
      p = os.path.join(rdir, cand)
      if os.path.isfile(p):
        try:
          with open(p, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
          mems_dict = loaded.get('entity_memories', loaded)
          if isinstance(mems_dict, dict) and mems_dict:
            break
        except Exception:  # pylint: disable=broad-except
          read_errors.append(f'{p}: {traceback.format_exc(limit=1)}')
    if not mems_dict:
      mems_dict = {}
      for fname in sorted(os.listdir(rdir)):
        if not fname.endswith('_memories.json'):
          continue
        p = os.path.join(rdir, fname)
        try:
          with open(p, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        except Exception:  # pylint: disable=broad-except
          read_errors.append(f'{p}: {traceback.format_exc(limit=1)}')
          continue
        if isinstance(loaded, dict) and isinstance(
            loaded.get('agent'), str
        ) and isinstance(loaded.get('memories'), list):
          mems_dict[loaded['agent']] = loaded['memories']
    if read_errors:
      out['error'] = 'Unreadable memory files:\n' + '\n'.join(read_errors)

    for agent, mems in (mems_dict or {}).items():
      if not isinstance(mems, list):
        continue
      for m_str in mems:
        m = _FULL_TS_RE.search(str(m_str))
        if m:
          day_num = int(m.group(4))
          hr = int(m.group(5))
          ampm = m.group(7)
          if ampm == 'PM' and hr != 12:
            hr += 12
          elif ampm == 'AM' and hr == 12:
            hr = 0
          slot = hours_order.get(hr, max(0, min(7, (hr - 7) // 2)))
          tick = (day_num - 1) * 8 + slot + 1
          loc = _normalise_location(m.group(1))
          if loc and loc != 'marketplace':
            per_tick.setdefault(tick, {})[agent] = loc

    ticks = sorted(per_tick.keys())
    running: dict[str, str] = {}
    last_seen: dict[str, int] = {}
    snapshots: dict[str, list[dict[str, Any]]] = {}
    observed: dict[str, int] = {}
    for t in ticks:
      running.update(per_tick[t])
      for a in per_tick[t]:
        last_seen[a] = t
      snapshots[str(t)] = [
          {'agent': a, 'location': l, 'age': t - last_seen[a]}
          for a, l in sorted(running.items())
      ]
      observed[str(t)] = len(per_tick[t])
    out['ticks'] = ticks
    out['snapshots'] = snapshots
    out['observed_counts'] = observed
    out['latest_locations'] = snapshots.get(str(ticks[-1]), []) if ticks else []
    return out
