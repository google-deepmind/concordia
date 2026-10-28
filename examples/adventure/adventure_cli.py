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
"""Interactive CLI for an Agent (or a developer) to play and evaluate Adventure.

Connects to a running `AdventureServer` (`http://127.0.0.1:8765` by default) to:
- `status` / `look`: Inspect the current room, props, exits, co-located NPCs,
  inventory, discovered clues, story arc/tension, engine telemetry, and recent
  narrative cards.
- `act "<action>"`: Submit a player action, wait synchronously until both the
  player's turn and any immediate Hot NPC reaction turn finish, and display all
  new narrative cards, state diffs, telemetry, and updated room state.
- `debug` / `telemetry`: Display full Engine Telemetry, Latency Tracker
  (`p50`/`p95` across player resolve, NPC resolve, Warm step, and Frontier
  expansion), Three-Tier NPC Roster (`HOT`/`WARM`/`COLD`), Story Director state,
  TopicBus state, and recent Concordia `SimulationLog` entries.
- `stream`: Print the full canonical `narrative_log` with duplicate checks.
- `npcs`: Inspect all `HOT`, `WARM`, and `COLD` NPCs, their goals, secrets,
  relationships, and live Concordia `AssociativeMemory` entries.
- `map`: Inspect the discovered world graph (`expanded` vs `stub` rooms).
- `evaluate`: Export live `SimulationLog` (`simulation_structured.json`,
  `simulation_log.html`) and `world_ledger.json` from `/api/logs` and
  `/api/state`, run `evaluate_hard_invariants` (and optional LLM autorater),
  and print the scorecard.
- `reset`: Reset the live server simulation back to Tick 0 (`POST /api/reset`).
"""

from collections.abc import Mapping, Sequence
import json
import os
import re
import time
from typing import Any
import urllib.error
import urllib.request

from absl import app
from absl import flags
from examples.adventure import evaluate_run
from examples.adventure import runner
from examples.adventure import story_director
from examples.adventure import world_ledger
from concordia.utils import structured_logging


def _define_flags() -> None:
  """Registers command-line flags for the Adventure CLI."""
  flags.DEFINE_string(
      'url',
      'http://127.0.0.1:8765',
      'Base URL of the running AdventureServer.',
  )
  flags.DEFINE_string(
      'cmd',
      '',
      'Command override (status, act, debug, stream, npcs, map, evaluate,'
      ' reset).',
  )
  flags.DEFINE_string(
      'action',
      '',
      'Player action text when running `act`.',
  )
  flags.DEFINE_string(
      'npc',
      '',
      'Optional NPC name filter for `npcs`.',
  )
  flags.DEFINE_integer(
      'last',
      10,
      'Number of recent narrative log entries to display in `status` or'
      ' `stream`.',
  )
  flags.DEFINE_float(
      'timeout',
      45.0,
      'Maximum seconds to wait for a turn and NPC reaction to settle in `act`.',
  )
  flags.DEFINE_bool(
      'memories',
      True,
      'Whether to include recent entity memories when running `npcs`.',
  )
  flags.DEFINE_bool(
      'autorater',
      False,
      'Whether to run the LLM rubric autorater in `evaluate`.',
  )
  flags.DEFINE_bool(
      'json_output',
      False,
      'If True, emit raw JSON output instead of formatted terminal text.',
  )
  flags.DEFINE_string(
      'run_dirs',
      '',
      'Comma-separated list of run directories for `compare_voices`.',
  )
  flags.DEFINE_string(
      'output_dir',
      '/tmp/adventure_cli_eval',
      'Directory for exported evaluation artifacts.',
  )
  flags.DEFINE_bool(
      'use_mock',
      False,
      'Use RichLocalMockModel when running the autorater.',
  )
  flags.DEFINE_string(
      'api_type',
      'google_aistudio_model',
      'Language model API type when running the autorater.',
  )
  flags.DEFINE_string(
      'model_name',
      'gemini-2.5-flash',
      'Language model name when running the autorater.',
  )
  flags.DEFINE_string(
      'api_key',
      '',
      'Optional API key override when running the autorater.',
  )


class AdventureCliClient:
  """HTTP client for interacting with a live `AdventureServer`."""

  def __init__(self, base_url: str = 'http://127.0.0.1:8765'):
    """Initializes the client."""
    self._base_url = base_url.rstrip('/')

  def get_state(self) -> dict[str, Any]:
    """Fetches `/api/state` from the server."""
    req = urllib.request.Request(f'{self._base_url}/api/state', method='GET')
    with urllib.request.urlopen(req, timeout=10.0) as resp:
      return json.loads(resp.read().decode('utf-8'))

  def get_logs(self) -> dict[str, Any]:
    """Fetches `/api/logs` (`SimulationLog.to_dict()`) from the server."""
    req = urllib.request.Request(f'{self._base_url}/api/logs', method='GET')
    with urllib.request.urlopen(req, timeout=15.0) as resp:
      return json.loads(resp.read().decode('utf-8'))

  def reset(self) -> dict[str, Any]:
    """Calls `POST /api/reset` to restart the simulation at Tick 0."""
    req = urllib.request.Request(
        f'{self._base_url}/api/reset',
        data=b'{}',
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=15.0) as resp:
      return json.loads(resp.read().decode('utf-8'))

  def act_and_wait(
      self, action_text: str, *, timeout_seconds: float = 45.0
  ) -> dict[str, Any]:
    """Submits `action_text` and waits until the turn and NPC reaction settle.

    Args:
      action_text: Free-form action or dialogue string for the player.
      timeout_seconds: Maximum wall-clock seconds to wait for turn completion.

    Returns:
      Dictionary containing `pre_state`, `post_state`, `new_entries`, `diff`,
      and `wall_elapsed_ms`.
    """
    pre_state = self.get_state()
    pre_ledger = pre_state.get('ledger', {})
    pre_log = pre_ledger.get('narrative_log', [])
    last_id = int(pre_log[-1]['id']) if pre_log else 0
    pre_steps = int(pre_state.get('player_steps_completed', 0))

    start_wall = time.perf_counter()
    body = json.dumps({'action': action_text}).encode('utf-8')
    req = urllib.request.Request(
        f'{self._base_url}/api/action',
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
      post_ack = json.loads(resp.read().decode('utf-8'))
    post_state = post_ack.get('state', pre_state)

    deadline = time.time() + timeout_seconds
    settled_polls = 0
    while time.time() < deadline:
      time.sleep(0.12)
      post_state = self.get_state()
      cur_steps = int(post_state.get('player_steps_completed', 0))
      turn_busy = bool(post_state.get('turn_busy', False))
      if cur_steps > pre_steps and not turn_busy:
        settled_polls += 1
        if settled_polls >= 2:
          break
      else:
        settled_polls = 0

    wall_elapsed_ms = round((time.perf_counter() - start_wall) * 1000.0, 2)
    post_ledger = post_state.get('ledger', {})
    post_log = post_ledger.get('narrative_log', [])
    new_entries = [e for e in post_log if int(e.get('id', 0)) > last_id]

    diff = _compute_state_diff(pre_ledger, post_ledger)
    return {
        'action': action_text,
        'wall_elapsed_ms': wall_elapsed_ms,
        'pre_state': pre_state,
        'post_state': post_state,
        'new_entries': new_entries,
        'diff': diff,
    }


def _extract_discovered_clues(story: Mapping[str, Any]) -> list[str]:
  """Extracts discovered clue IDs from a serialized StoryState dictionary."""
  found: list[str] = list(story.get('discovered_clues', []) or [])
  clues_map = story.get('clues')
  if isinstance(clues_map, Mapping):
    for cid, cdict in clues_map.items():
      if (
          isinstance(cdict, Mapping)
          and cdict.get('discovered')
          and str(cid) not in found
      ):
        found.append(str(cid))
  return found


def _compute_state_diff(
    pre_ledger: Mapping[str, Any], post_ledger: Mapping[str, Any]
) -> dict[str, Any]:
  """Computes a structured summary of changes between two ledger snapshots."""
  pre_loc = str(pre_ledger.get('player_location_id', ''))
  post_loc = str(post_ledger.get('player_location_id', ''))
  pre_inv = set(pre_ledger.get('player_inventory', []) or [])
  post_inv = set(post_ledger.get('player_inventory', []) or [])

  pre_story = pre_ledger.get('story') or {}
  post_story = post_ledger.get('story') or {}
  pre_clues = set(_extract_discovered_clues(pre_story))
  post_clues = set(_extract_discovered_clues(post_story))

  pre_room = (pre_ledger.get('locations') or {}).get(pre_loc, {})
  post_room = (post_ledger.get('locations') or {}).get(post_loc, {})
  pre_props = set(pre_room.get('affordances', []) or [])
  post_props = set(post_room.get('affordances', []) or [])

  return {
      'tick_before': int(pre_ledger.get('tick', 0)),
      'tick_after': int(post_ledger.get('tick', 0)),
      'clock_before': str(
          pre_ledger.get('clock_formatted') or pre_ledger.get('clock', '')
      ),
      'clock_after': str(
          post_ledger.get('clock_formatted') or post_ledger.get('clock', '')
      ),
      'moved': pre_loc != post_loc,
      'location_before': pre_loc,
      'location_after': post_loc,
      'inventory_added': sorted(post_inv - pre_inv),
      'inventory_removed': sorted(pre_inv - post_inv),
      'clues_discovered': sorted(post_clues - pre_clues),
      'props_added': (
          sorted(post_props - pre_props) if pre_loc == post_loc else []
      ),
      'props_removed': (
          sorted(pre_props - post_props) if pre_loc == post_loc else []
      ),
      'tension_before': pre_story.get('tension', 0),
      'tension_after': post_story.get('tension', 0),
  }


def _format_compact_telemetry(state: Mapping[str, Any]) -> str:
  """Returns a 1-line summary of Engine Telemetry and Three-Tier NPC counts."""
  ledger = state.get('ledger', {}) or {}
  telemetry = ledger.get('telemetry', {}) or {}
  latency = state.get('latency', {}) or {}
  p_res = latency.get('player_resolve', {}) or {}
  tiers = telemetry.get('tier_counts', {}) or {}
  return (
      'ENGINE TELEMETRY: '
      f"Resolve p50={p_res.get('p50_ms', 0)}ms "
      f"p95={p_res.get('p95_ms', 0)}ms | "
      f"ControlCalls={latency.get('control_hook_llm_calls', 0)} | "
      f"MaxResolveCalls={latency.get('max_llm_calls_per_player_resolve', 0)} | "
      f"FrontierHitRate={round(float(telemetry.get('frontier_cache_hit_rate', 1.0)) * 100, 1)}% | "
      f"Tiers(H/W/C)={tiers.get('hot', 0)}/{tiers.get('warm', 0)}/{tiers.get('cold', 0)} | "
      f"Hib={telemetry.get('hibernations_count', 0)} "
      f"Rehyd={telemetry.get('rehydrations_count', 0)} "
      f"Warm={telemetry.get('warm_events_count', 0)}"
  )


def format_status(state: Mapping[str, Any], *, last_n: int = 6) -> str:
  """Formats the current simulation state for terminal reading."""
  ledger = state.get('ledger', {})
  loc_id = ledger.get('player_location_id', '')
  locations = ledger.get('locations', {})
  loc = locations.get(loc_id, {})
  story = ledger.get('story') or {}
  npcs = ledger.get('npcs', {})
  telemetry = ledger.get('telemetry') or {}

  lines: list[str] = []
  lines.append('=' * 76)
  act_title = story.get('current_act_title') or f"Act {telemetry.get('act', 1)}"
  tension = story.get('tension', telemetry.get('tension', 0))
  clock_str = (
      ledger.get('clock_formatted')
      or ledger.get('clock')
      or telemetry.get('clock', '')
  )
  lines.append(
      f'CLOCK: {clock_str} | TICK: {ledger.get("tick", 0)} | '
      f'ACT: {act_title} | TENSION: {tension}/10'
  )
  lines.append(
      f"LOCATION: {loc.get('name', loc_id)} ({loc_id}) | "
      f"STATUS: {ledger.get('player_status', 'Healthy')}"
  )
  lines.append('=' * 76)
  lines.append(f"Description: {loc.get('description', '')}")

  props = loc.get('affordances', []) or []
  lines.append(f'Affordances / Props ({len(props)}):')
  for p in props:
    lines.append(f'  * {p}')

  exits = loc.get('exits', {}) or {}
  exit_strs = []
  for direction, target_id in exits.items():
    target_node = locations.get(target_id, {})
    target_name = target_node.get('name', target_id)
    exp_tag = 'expanded' if target_node.get('expanded') else 'stub'
    exit_strs.append(f'{direction} -> {target_name} [{exp_tag}]')
  lines.append('Exits: ' + (', '.join(exit_strs) if exit_strs else 'none'))

  co_located = [
      rec for rec in npcs.values() if rec.get('location_id') == loc_id
  ]
  lines.append(f'Co-located NPCs ({len(co_located)}):')
  if co_located:
    for rec in co_located:
      rel = (rec.get('relationships') or {}).get(
          ledger.get('player_name', 'Traveler'), 'Neutral'
      )
      lead_voice = rec.get('last_winning_voice') or 'The Mask'
      sub_minds = rec.get('sub_minds') or {}
      lead_sm = sub_minds.get(lead_voice) or {}
      rank_str = f"Rank {lead_sm.get('rank', 1)}"
      intr_str = ', INTRUSIVE' if lead_sm.get('intrusive') else ''
      lines.append(
          f"  * {rec.get('name')} [{rec.get('tier', '').upper()}] — "
          f"{rec.get('role_summary')} (Rel: {rel} | "
          f'Leading Voice: {lead_voice} [{rank_str}{intr_str}])'
      )
  else:
    lines.append('  (none)')

  inv = ledger.get('player_inventory', []) or []
  clues = _extract_discovered_clues(story)
  lines.append(f"Inventory: {', '.join(inv) if inv else '(empty)'}")
  lines.append(f"Clues Discovered: {', '.join(clues) if clues else '(none)'}")
  lines.append('-' * 76)
  lines.append(_format_compact_telemetry(state))

  nlog = ledger.get('narrative_log', []) or []
  recent = nlog[-last_n:] if last_n > 0 else nlog
  lines.append('-' * 76)
  lines.append(
      f'RECENT NARRATIVE STREAM (showing {len(recent)} of {len(nlog)}):'
  )
  for entry in recent:
    lines.append(
        f"  [#{entry.get('id')} | Tick {entry.get('tick')} | "
        f"{entry.get('kind')} | {entry.get('actor')}]"
    )
    lines.append(f"    {entry.get('text')}")
  lines.append('=' * 76)
  return '\n'.join(lines)


def format_act_result(result: Mapping[str, Any]) -> str:
  """Formats the output of `act_and_wait` for terminal reading."""
  diff = result.get('diff', {})
  new_entries = result.get('new_entries', [])
  post_state = result.get('post_state', {})
  post_ledger = post_state.get('ledger', {})
  loc_id = post_ledger.get('player_location_id', '')
  locations = post_ledger.get('locations', {})
  loc = locations.get(loc_id, {})

  lines: list[str] = []
  lines.append('=' * 76)
  lines.append(
      f"ACTION SUBMITTED: {result.get('action')} "
      f"(settled in {result.get('wall_elapsed_ms')} ms)"
  )
  lines.append(
      f"TICK: {diff.get('tick_before')} -> {diff.get('tick_after')} | "
      f"TENSION: {diff.get('tension_before')} -> {diff.get('tension_after')}"
  )
  if diff.get('moved'):
    lines.append(
        f"MOVED: {diff.get('location_before')} -> {diff.get('location_after')}"
    )
  if diff.get('clues_discovered'):
    lines.append(f"NEW CLUES: {', '.join(diff['clues_discovered'])}")
  if diff.get('inventory_added'):
    lines.append(f"INVENTORY +: {', '.join(diff['inventory_added'])}")
  if diff.get('props_added') or diff.get('props_removed'):
    lines.append(
        f"PROPS DELTA: +{diff.get('props_added')} /"
        f" -{diff.get('props_removed')}"
    )
  lines.append('-' * 76)
  lines.append(f'NEW STREAM CARDS ({len(new_entries)}):')
  for entry in new_entries:
    lines.append(
        f"  [#{entry.get('id')} | Tick {entry.get('tick')} | "
        f"{entry.get('kind')} | {entry.get('actor')}]"
    )
    lines.append(f"    {entry.get('text')}")
  lines.append('-' * 76)
  exits = loc.get('exits', {}) or {}
  exit_strs = [
      f"{d} -> {(locations.get(tid) or {}).get('name', tid)}"
      for d, tid in exits.items()
  ]
  co_located = [
      rec.get('name')
      for rec in (post_ledger.get('npcs') or {}).values()
      if rec.get('location_id') == loc_id
  ]
  lines.append(
      f"NOW AT: {loc.get('name', loc_id)} | "
      f"Present: {', '.join(co_located) if co_located else 'none'} | "
      f"Exits: {', '.join(exit_strs) if exit_strs else 'none'}"
  )
  lines.append(_format_compact_telemetry(post_state))
  lines.append('=' * 76)
  return '\n'.join(lines)


def format_stream(state: Mapping[str, Any], *, last_n: int = 0) -> str:
  """Formats the canonical `narrative_log` and runs duplicate checks."""
  ledger = state.get('ledger', {})
  nlog = ledger.get('narrative_log', []) or []
  ids = [entry.get('id') for entry in nlog]
  unique_ids = len(set(ids)) == len(ids)
  adjacent_dups = 0
  for i in range(1, len(nlog)):
    if nlog[i].get('text') == nlog[i - 1].get('text'):
      adjacent_dups += 1

  counts: dict[str, int] = {}
  for entry in nlog:
    k = str(entry.get('kind', 'unknown'))
    counts[k] = counts.get(k, 0) + 1

  shown = nlog[-last_n:] if last_n > 0 else nlog
  lines = [
      '=' * 76,
      (
          f'NARRATIVE STREAM ({len(nlog)} total entries | '
          f'unique_ids={unique_ids} | adjacent_dups={adjacent_dups})'
      ),
      f'Kind breakdown: {json.dumps(counts, sort_keys=True)}',
      '-' * 76,
  ]
  for entry in shown:
    lines.append(
        f"[#{entry.get('id')} | Tick {entry.get('tick')} | "
        f"{entry.get('clock')} | {entry.get('kind')} | {entry.get('actor')}]"
    )
    lines.append(f"  {entry.get('text')}")
  lines.append('=' * 76)
  return '\n'.join(lines)


def format_npcs(
    state: Mapping[str, Any],
    *,
    npc_filter: str = '',
    include_memories: bool = True,
) -> str:
  """Formats all NPCs and their Concordia memories."""
  ledger = state.get('ledger', {})
  npcs = ledger.get('npcs', {}) or {}
  memories = state.get('memories', {}) or {}
  lines = ['=' * 76, f'NPC ROSTER ({len(npcs)} total):', '-' * 76]
  for name, rec in sorted(npcs.items()):
    if npc_filter and npc_filter.lower() not in name.lower():
      continue
    lines.append(
        f"* {name} [{rec.get('tier', '').upper()}] @ {rec.get('location_id')} "
        f"(last_tick={rec.get('last_active_tick')}, "
        f"addressed={rec.get('directly_addressed')})"
    )
    lines.append(f"  Role: {rec.get('role_summary')}")
    lines.append(f"  Goal: {rec.get('goal')}")
    if rec.get('secret'):
      lines.append(f"  Secret: {rec.get('secret')}")
    if rec.get('relationships'):
      lines.append(f"  Relationships: {json.dumps(rec.get('relationships'))}")
    sub_minds = rec.get('sub_minds') or {}
    if sub_minds:
      lead_voice = rec.get('last_winning_voice') or ''
      lines.append(f'  Inner Parliament (Leading: {lead_voice}):')
      for v_name, sm in sub_minds.items():
        if not isinstance(sm, Mapping):
          continue
        intr = ' [INTRUSIVE]' if sm.get('intrusive') else ''
        scars = sm.get('scars') or []
        scar_tail = f' | Latest Scar: {scars[-1]}' if scars else ''
        lines.append(
            f"    - {v_name} [Rank {sm.get('rank', 1)}, "
            f"{sm.get('domain', '')}{intr}]: "
            f"obsession=\"{sm.get('obsession', '')}\"{scar_tail}"
        )
    active_opts = rec.get('active_voice_options') or []
    if active_opts:
      lines.append('  Latest Council Deliberation:')
      for opt in active_opts:
        if isinstance(opt, Mapping):
          lines.append(
              f"    - {opt.get('voice')} [Rank {opt.get('affinity', 1)}]: "
              f"\"{opt.get('perception', '')}\" -> {opt.get('action', '')}"
          )
    if include_memories:
      entity_mems = memories.get(name, []) or []
      lines.append(f'  Recent Memories ({len(entity_mems)} total):')
      for m in entity_mems[-5:]:
        lines.append(f'    > {str(m)[:160]}')
  lines.append('=' * 76)
  return '\n'.join(lines)


def format_map(state: Mapping[str, Any]) -> str:
  """Formats the discovered location graph."""
  ledger = state.get('ledger', {})
  player_loc = ledger.get('player_location_id', '')
  locations = ledger.get('locations', {}) or {}
  npcs = ledger.get('npcs', {}) or {}
  lines = [
      '=' * 76,
      f'WORLD MAP ({len(locations)} rooms in ledger):',
      '-' * 76,
  ]
  for loc_id, node in sorted(locations.items()):
    marker = ' [PLAYER HERE]' if loc_id == player_loc else ''
    status = 'EXPANDED' if node.get('expanded') else 'STUB'
    here_npcs = [
        n.get('name') for n in npcs.values() if n.get('location_id') == loc_id
    ]
    exits = node.get('exits', {}) or {}
    exit_str = ', '.join(f'{d}->{t}' for d, t in exits.items())
    lines.append(f"* {node.get('name')} ({loc_id}) [{status}]{marker}")
    lines.append(f'  Exits: {exit_str or "none"}')
    if here_npcs:
      lines.append(f"  NPCs: {', '.join(here_npcs)}")
  lines.append('=' * 76)
  return '\n'.join(lines)


def format_debug(
    state: Mapping[str, Any], logs_dict: Mapping[str, Any] | None = None
) -> str:
  """Formats Engine Telemetry, Three-Tier NPC Roster, Story & Bus diagnostics."""
  ledger = state.get('ledger', {}) or {}
  telemetry = ledger.get('telemetry', {}) or {}
  latency = state.get('latency', {}) or {}
  npcs = ledger.get('npcs', {}) or {}
  memories = state.get('memories', {}) or {}
  story = ledger.get('story', {}) or {}
  bus_state = ledger.get('bus_state', {}) or {}
  rumors = ledger.get('rumors', []) or []
  world_facts = ledger.get('world_facts', {}) or {}

  invariants = evaluate_run.evaluate_hard_invariants(
      telemetry=telemetry,
      latency=latency,
      entity_memories=memories,
  )

  p_res = latency.get('player_resolve', {}) or {}
  npc_res = latency.get('npc_resolve', {}) or {}
  warm_lat = latency.get('warm_step', {}) or {}
  front_lat = latency.get('frontier_expand', {}) or {}
  tiers = telemetry.get('tier_counts', {}) or {}

  lines: list[str] = []
  lines.append('=' * 76)
  lines.append('1. ENGINE TELEMETRY & LATENCY TRACKER')
  lines.append('-' * 76)
  lines.append(
      f"  Player Steps Completed : {state.get('player_steps_completed', 0)} "
      f"(turn_busy={state.get('turn_busy', False)}, "
      f"pending_queue={state.get('pending_player_actions', False)})"
  )
  lines.append(
      f"  Player Resolve Latency : p50={p_res.get('p50_ms', 0)}ms | "
      f"p95={p_res.get('p95_ms', 0)}ms | "
      f"mean={p_res.get('mean_ms', 0)}ms (count={p_res.get('count', 0)})"
  )
  lines.append(
      f"  NPC Resolve Latency    : p50={npc_res.get('p50_ms', 0)}ms | "
      f"mean={npc_res.get('mean_ms', 0)}ms (count={npc_res.get('count', 0)})"
  )
  lines.append(
      f"  Frontier Gen Latency   : p50={front_lat.get('p50_ms', 0)}ms |"
      f" mean={front_lat.get('mean_ms', 0)}ms"
      f" (count={front_lat.get('count', 0)})"
  )
  lines.append(
      f"  Warm Step Latency      : p50={warm_lat.get('p50_ms', 0)}ms | "
      f"mean={warm_lat.get('mean_ms', 0)}ms (count={warm_lat.get('count', 0)})"
  )
  lines.append(
      f"  Control Hook LLM Calls : {latency.get('control_hook_llm_calls', 0)} |"
      f" Director LLM Calls: {telemetry.get('director_llm_calls', 0)} | Max LLM"
      ' Calls / Player Resolve:'
      f" {latency.get('max_llm_calls_per_player_resolve', 0)}"
  )
  lines.append(
      '  Frontier Cache Hit Rate: '
      f"{round(float(telemetry.get('frontier_cache_hit_rate', 1.0)) * 100, 1)}% "
      f"(hits={telemetry.get('frontier_cache_hits', 0)}, "
      f"traversals={telemetry.get('frontier_traversals', 0)})"
  )
  lines.append(
      f"  Locations              : {telemetry.get('expanded_locations', 0)}"
      f" expanded / {telemetry.get('total_locations', 0)} total"
  )
  lines.append(
      '  Lifecycle Counters     :'
      f" Hibernations={telemetry.get('hibernations_count', 0)} |"
      f" Rehydrations={telemetry.get('rehydrations_count', 0)} |"
      f" WarmEvents={telemetry.get('warm_events_count', 0)}"
  )
  lines.append(
      f"  Hard Invariants        : all_passed={invariants.get('all_passed')} "
      f"{json.dumps(invariants.get('checks', {}), sort_keys=True)}"
  )

  lines.append('=' * 76)
  lines.append(
      '2. THREE-TIER NPC ROSTER '
      f"(HOT={tiers.get('hot', 0)} | "
      f"WARM={tiers.get('warm', 0)} | "
      f"COLD={tiers.get('cold', 0)})"
  )
  lines.append('-' * 76)
  for tier_label in ('hot', 'warm', 'cold'):
    tier_npcs = [
        rec
        for rec in npcs.values()
        if str(rec.get('tier', '')).lower() == tier_label
    ]
    lines.append(f'  [{tier_label.upper()} TIER — {len(tier_npcs)} NPCs]')
    if not tier_npcs:
      lines.append('    (none)')
      continue
    for rec in tier_npcs:
      name = str(rec.get('name', ''))
      mems = memories.get(name, []) or []
      has_cold = bool(rec.get('cold_state'))
      lines.append(
          f"    * {name} @ {rec.get('location_id')} | "
          f"role={rec.get('role_summary')} | "
          f"last_tick={rec.get('last_active_tick')} | "
          f"addressed={rec.get('directly_addressed')} | "
          f'memories={len(mems)} | cold_snapshot={has_cold}'
      )
      lines.append(f"      Goal: {rec.get('goal')}")
      sub_minds = rec.get('sub_minds') or {}
      if sub_minds:
        sm_summary = ', '.join(
            f"{vn}:R{sm.get('rank', 1)}{'!' if sm.get('intrusive') else ''}"
            for vn, sm in sub_minds.items()
            if isinstance(sm, Mapping)
        )
        lines.append(
            f"      Inner Parliament (Lead={rec.get('last_winning_voice')}): "
            f'{sm_summary}'
        )
      if mems:
        lines.append(f'      Latest Memory: {str(mems[-1])[:140]}')

  lines.append('=' * 76)
  lines.append('3. STORY DIRECTOR, RUMORS & TOPIC BUS')
  lines.append('-' * 76)
  lines.append(
      f"  Act: {story.get('current_act_title', telemetry.get('act', 1))} | "
      f"Tension: {story.get('tension', telemetry.get('tension', 0))}/100 | "
      f"Storylets Fired: {telemetry.get('storylets_fired', 0)} | "
      f"Clues Discovered: {telemetry.get('clues_discovered', 0)}"
  )
  lines.append(
      f"  Factions: {json.dumps(story.get('factions', {}), sort_keys=True)}"
  )
  if world_facts:
    lines.append(f'  World Facts: {json.dumps(world_facts, sort_keys=True)}')
  lines.append(f'  Off-Screen Rumors ({len(rumors)}):')
  for r in rumors[-5:]:
    lines.append(f'    * {r}')
  lines.append(
      f"  TopicBus: seq={bus_state.get('seq', 0)} | "
      f"topics={list((bus_state.get('topics') or {}).keys())}"
  )

  if logs_dict is not None:
    entries = logs_dict.get('entries', []) or []
    lines.append('=' * 76)
    lines.append(
        f'4. CONCORDIA STRUCTURED LOG SUMMARY ({len(entries)} entries)'
    )
    lines.append('-' * 76)
    for e in entries[-6:]:
      lines.append(
          f"  Step {e.get('step')} | {e.get('entry_type')} | "
          f"{e.get('entity_name')}: {str(e.get('summary', ''))[:120]}"
      )

  lines.append('=' * 76)
  return '\n'.join(lines)


def run_evaluate_command(
    client: AdventureCliClient,
    *,
    output_dir: str,
    run_autorater: bool,
    use_mock: bool = False,
    api_type: str = 'google_aistudio_model',
    model_name: str = 'gemini-2.5-flash',
    api_key: str = '',
) -> dict[str, Any]:
  """Exports live logs and runs `evaluate_run` on the current session."""
  os.makedirs(output_dir, exist_ok=True)
  state = client.get_state()
  log_dict = client.get_logs()
  sim_log = structured_logging.SimulationLog.from_dict(log_dict)

  structured_path = os.path.join(output_dir, 'simulation_structured.json')
  with open(structured_path, 'w', encoding='utf-8') as f:
    f.write(sim_log.to_json())

  html_path = os.path.join(output_dir, 'simulation_log.html')
  with open(html_path, 'w', encoding='utf-8') as f:
    f.write(sim_log.to_html())

  ledger_dict = state.get('ledger', {}) or {}
  ledger_path = os.path.join(output_dir, 'world_ledger.json')
  with open(ledger_path, 'w', encoding='utf-8') as f:
    json.dump(ledger_dict, f, indent=2)

  latency_path = os.path.join(output_dir, 'latency.json')
  with open(latency_path, 'w', encoding='utf-8') as f:
    json.dump(state.get('latency', {}), f, indent=2)

  meta_path = os.path.join(output_dir, evaluate_run.PLAYTEST_META_FILENAME)
  if not os.path.exists(meta_path):
    story_dict = ledger_dict.get('story') or {}
    with open(meta_path, 'w', encoding='utf-8') as f:
      json.dump(
          {
              'archetype': 'ChaosBoundaryTester',
              'info_policy': 'player_visible_only',
              'bible': str(story_dict.get('title', '')),
          },
          f,
          indent=2,
      )

  if run_autorater:
    model = runner.setup_language_model(
        use_mock=use_mock,
        api_type=api_type,
        model_name=model_name,
        api_key=api_key,
    )
    return evaluate_run.evaluate_artifacts_directory(output_dir, model=model)

  transcript = evaluate_run.extract_transcript_from_log(sim_log)
  deterministic = evaluate_run.compute_deterministic_metrics(
      ledger_dict=ledger_dict,
      transcript=transcript,
  )
  telemetry = ledger_dict.get('telemetry', {})
  latency = state.get('latency', {})
  memories = state.get('memories', {})
  invariants = evaluate_run.evaluate_hard_invariants(
      telemetry=telemetry,
      latency=latency,
      entity_memories=memories,
      deterministic=deterministic,
  )
  return {
      'output_dir': output_dir,
      'structured_log_path': structured_path,
      'html_log_path': html_path,
      'invariants': invariants,
      'deterministic': deterministic,
  }


def _norm_voice(name: str) -> str:
  return re.sub(r'[^a-z0-9]+', '', str(name).lower())


def compare_voice_runs(run_dirs: Sequence[str]) -> dict[str, Any]:
  """Compares multiple playtest directories across Sub-Mind trajectories."""
  runs_summary: list[dict[str, Any]] = []
  for rdir in run_dirs:
    rdir = rdir.strip()
    if not rdir or not os.path.isdir(rdir):
      continue
    ledger_path = os.path.join(rdir, 'world_ledger.json')
    meta_path = os.path.join(rdir, evaluate_run.PLAYTEST_META_FILENAME)
    struct_path = os.path.join(rdir, 'simulation_structured.json')
    if not os.path.isfile(ledger_path):
      continue
    with open(ledger_path, 'r', encoding='utf-8') as f:
      ledger = json.load(f)
    meta: dict[str, Any] = {}
    if os.path.isfile(meta_path):
      with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    log_steps = 0
    if os.path.isfile(struct_path):
      try:
        with open(struct_path, 'r', encoding='utf-8') as f:
          sim_log = structured_logging.SimulationLog.from_json(f.read())
        log_steps = len(sim_log.entries)
      except Exception:  # pylint: disable=broad-exception-caught
        log_steps = 0

    story = ledger.get('story', {})
    target_voice = str(meta.get('target_voice') or os.path.basename(rdir))
    norm_target = _norm_voice(target_voice)
    sub_minds = story.get('sub_minds', {})
    sm_info = sub_minds.get(target_voice, {})
    if not sm_info:
      for k, v in sub_minds.items():
        if _norm_voice(k) == norm_target or (
            isinstance(v, Mapping)
            and _norm_voice(v.get('id', '')) == norm_target
        ):
          sm_info = v
          target_voice = k
          break
    emergent_minds = [
        k
        for k, v in sub_minds.items()
        if isinstance(v, Mapping) and v.get('origin') == 'emergent'
    ]
    all_sub_minds_ranks = {
        k: {
            'rank': v.get('rank', 1),
            'xp': v.get('xp', 0),
            'scars_count': len(v.get('scars', [])),
            'intrusive': bool(v.get('intrusive', False)),
        }
        for k, v in sub_minds.items()
        if isinstance(v, Mapping)
    }
    factions = {
        fid: (
            str(fobj.get('disposition') or '').strip()
            or f"standing {int(fobj.get('standing_with_player', 0)):+d}"
        )
        for fid, fobj in story.get('factions', {}).items()
        if isinstance(fobj, Mapping)
    }
    faction_details = {
        fid: {
            'standing': int(fobj.get('standing_with_player', 0)),
            'disposition': fobj.get('disposition', ''),
            'relations': dict(fobj.get('relations', {})),
            'recent_actions': list(fobj.get('recent_actions', []))[-5:],
        }
        for fid, fobj in story.get('factions', {}).items()
        if isinstance(fobj, Mapping)
    }
    clues_found = [
        cid
        for cid, cobj in story.get('clues', {}).items()
        if isinstance(cobj, Mapping) and cobj.get('discovered')
    ]
    clues_details = {
        cid: {
            'title': cobj.get('title', ''),
            'step': cobj.get('discovered_at_step', 0),
        }
        for cid, cobj in story.get('clues', {}).items()
        if isinstance(cobj, Mapping) and cobj.get('discovered')
    }
    truths_found = [
        tid
        for tid, tobj in story.get('truths', {}).items()
        if isinstance(tobj, Mapping) and tobj.get('discovered')
    ]
    fired_beats = list(story.get('fired', {}).keys())
    fired_log = list(story.get('fired_log', []))
    turf_war_log = list(story.get('turf_war_log', []))
    visited_rooms = list(story.get('visited_location_ids', []))
    nar_log = ledger.get('narrative_log', [])
    player_actions = [
        e.get('text', '')
        for e in nar_log
        if isinstance(e, Mapping) and e.get('kind') == 'player_action'
    ]
    scene_resolutions = [
        e.get('text', '')
        for e in nar_log
        if isinstance(e, Mapping) and e.get('kind') == 'scene_resolution'
    ]
    npc_resolutions = [
        f"{e.get('actor', 'NPC')}: {e.get('text', '')}"
        for e in nar_log
        if isinstance(e, Mapping) and e.get('kind') == 'npc_resolution'
    ]
    scars = (
        list(sm_info.get('scars', [])) if isinstance(sm_info, Mapping) else []
    )
    vindicated_count = sum(1 for s in scars if 'Vindicated:' in str(s))
    burned_count = sum(1 for s in scars if 'Burned:' in str(s))
    npcs_summary = {
        nid: {
            'location': (
                nobj.get('location_id', '')
                or nobj.get('current_location_id', '')
            ),
            'tier': nobj.get('tier', ''),
            'relationships': nobj.get('relationships', {}),
            'last_winning_voice': nobj.get('last_winning_voice', ''),
            'sub_minds': {
                sm_k: {
                    'rank': sm_v.get('rank', 1),
                    'intrusive': bool(sm_v.get('intrusive', False)),
                    'scars': list(sm_v.get('scars', [])),
                    'obsession': sm_v.get('obsession', ''),
                }
                for sm_k, sm_v in (nobj.get('sub_minds') or {}).items()
                if isinstance(sm_v, Mapping)
            },
            'memories_count': len(nobj.get('memories', [])),
            'recent_memories': list(nobj.get('memories', []))[-3:],
        }
        for nid, nobj in ledger.get('npcs', {}).items()
        if isinstance(nobj, Mapping)
    }

    eval_path = os.path.join(rdir, 'evaluation_report.json')
    eval_summary: dict[str, Any] = {}
    if os.path.isfile(eval_path):
      try:
        with open(eval_path, 'r', encoding='utf-8') as f:
          eval_rep = json.load(f)
        autorater = eval_rep.get('autorater', {})
        det = eval_rep.get('deterministic', {})
        inv = eval_rep.get('invariants', {})
        eval_summary = {
            'weighted_total': autorater.get('weighted_total'),
            'lowest_dimension': autorater.get('lowest_dimension'),
            'top_improvement_target': autorater.get('top_improvement_target'),
            'caps_applied': autorater.get('caps_applied', {}),
            'invariants_all_passed': inv.get('all_passed'),
            'dimension_scores': {
                dk: dv.get('score')
                for dk, dv in (autorater.get('dimensions') or {}).items()
                if isinstance(dv, Mapping)
            },
            'dimensions_full': autorater.get('dimensions', {}),
            'narrative_coherence': det.get('narrative_coherence', {}),
            'parliament_npc': (det.get('parliament') or {}).get('npc', {}),
            'voices_options': (det.get('voices') or {}).get('options', {}),
        }
      except Exception:  # pylint: disable=broad-exception-caught
        pass

    runs_summary.append({
        'dir': rdir,
        'target_voice': target_voice,
        'evaluation': eval_summary,
        'domain': (
            sm_info.get('domain', '') if isinstance(sm_info, Mapping) else ''
        ),
        'final_rank': (
            sm_info.get('rank', 1) if isinstance(sm_info, Mapping) else 1
        ),
        'xp': sm_info.get('xp', 0) if isinstance(sm_info, Mapping) else 0,
        'intrusive': bool(
            sm_info.get('intrusive', False)
            if isinstance(sm_info, Mapping)
            else False
        ),
        'obsession': (
            sm_info.get('obsession', '') if isinstance(sm_info, Mapping) else ''
        ),
        'scars_total': len(scars),
        'vindicated_count': vindicated_count,
        'burned_count': burned_count,
        'scars': scars,
        'all_sub_minds_ranks': all_sub_minds_ranks,
        'voice_affinity': story.get('voice_affinity', {}),
        'emergent_minds': emergent_minds,
        'player_turns': len(player_actions),
        'npc_turns': len(npc_resolutions),
        'concordia_log_steps': log_steps,
        'final_act': story.get('act', 1),
        'final_tension': story.get('tension', 0.0),
        'doom_stage': story.get('doom_stage', 0),
        'ended': bool(story.get('ended', False)),
        'ending_archetype': story.get('ending_archetype', ''),
        'ending_title': story.get('ending_title', ''),
        'ending_summary': story.get('ending_summary', ''),
        'ending_slides': list(story.get('ending_slides', [])),
        'factions': factions,
        'faction_details': faction_details,
        'turf_war_log': turf_war_log,
        'clues_count': len(clues_found),
        'clues_found': clues_found,
        'clues_details': clues_details,
        'truths_count': len(truths_found),
        'truths_found': truths_found,
        'storylets_fired_count': len(fired_beats),
        'storylets_fired': fired_beats,
        'fired_log': fired_log,
        'visited_rooms_count': len(visited_rooms),
        'visited_rooms': visited_rooms,
        'inventory': list(ledger.get('player_inventory', [])),
        'npcs_summary': npcs_summary,
        'sample_actions': (
            player_actions[:4] + player_actions[20:23] + player_actions[-3:]
        ),
        'sample_resolutions': (
            scene_resolutions[:3]
            + scene_resolutions[20:22]
            + scene_resolutions[-2:]
        ),
        'sample_npc_reactions': npc_resolutions[:3] + npc_resolutions[-3:],
    })
  return {'runs': runs_summary}


def finalize_epilogues(run_dirs: Sequence[str]) -> dict[str, Any]:
  """Synthesizes 3-slide epilogues and ending cards for completed run dirs."""
  updated: list[dict[str, Any]] = []
  for rdir in run_dirs:
    rdir = rdir.strip()
    if not rdir or not os.path.isdir(rdir):
      continue
    ledger_path = os.path.join(rdir, 'world_ledger.json')
    struct_path = os.path.join(rdir, 'simulation_structured.json')
    html_path = os.path.join(rdir, 'simulation_log.html')
    eval_path = os.path.join(rdir, 'evaluation_report.json')
    if not os.path.isfile(ledger_path):
      continue
    with open(ledger_path, 'r', encoding='utf-8') as f:
      raw_ledger = json.load(f)
    ledger = world_ledger.WorldLedger(premise='')
    ledger.set_state(raw_ledger)
    for sm in ledger.story.sub_minds.values():
      if not sm.domain or sm.domain == 'instinct':
        def_dom, _, _ = world_ledger.story_state.infer_submind_defaults(
            sm.name, sm.worldview
        )
        if def_dom != 'instinct':
          sm.domain = def_dom
    director = story_director.StoryDirector(ledger)
    ledger.set_director(director)

    dom_sm = director._dominant_player_sub_mind()  # pylint: disable=protected-access
    lead_domain = dom_sm.domain if dom_sm is not None else ''
    wars = ledger.story.active_turf_wars()
    cracked_wounds = sum(
        1
        for r in ledger.all_npcs().values()
        if any(
            (sm.name == 'The Wound' or sm.domain == 'self_preservation')
            and (sm.rank >= 4 or sm.scars)
            for sm in r.sub_minds.values()
        )
    )
    corrupt_truth = (
        ledger.story.truths.get('t_butchers_bill')
        or ledger.story.truths.get('t_contract')
        or next(
            (t for t in ledger.story.truths.values() if t.discovered),
            None,
        )
    )
    if lead_domain in ('coercion', 'force', 'tactical_force', 'predation') and (
        len(wars) >= 1
        or cracked_wounds >= 1
        or ledger.story.truths_discovered() >= 1
    ):
      outcome_hint = 'chaos'
    elif (
        lead_domain in ('procedure', 'forensics', 'cybernetics')
        and corrupt_truth is not None
        and corrupt_truth.discovered
    ):
      outcome_hint = 'reckoning'
    elif lead_domain == 'empathy':
      outcome_hint = 'defection'
    elif lead_domain in ('leverage', 'deception'):
      outcome_hint = 'compromise'
    else:
      outcome_hint = 'unresolved'

    climax_loc = (
        ledger.story.visited_location_ids[0]
        if ledger.story.visited_location_ids
        else ledger.player_location_id
    )
    epilogue = director.resolve_emergent_climax(
        climax_loc,
        outcome=outcome_hint,
        trigger_reason='finalized_epilogue',
    )
    with ledger.lock:
      raw_log = ledger._narrative_log  # pylint: disable=protected-access
      raw_log[:] = [e for e in raw_log if e.get('kind') != 'ending']
    ledger.append_narrative_entry(
        kind='ending',
        actor='Epilogue',
        text=epilogue,
        location_id=climax_loc,
    )
    new_ledger_dict = ledger.get_state()
    with open(ledger_path, 'w', encoding='utf-8') as f:
      json.dump(new_ledger_dict, f, indent=2)

    if os.path.isfile(eval_path):
      try:
        with open(eval_path, 'r', encoding='utf-8') as f:
          eval_rep = json.load(f)
        det = eval_rep.get('deterministic', {})
        if isinstance(det, dict):
          fid = det.get('fidelity', {})
          if isinstance(fid, dict):
            fid['ended'] = True
            fid['ending_summary'] = epilogue
            fid['ending_archetype'] = ledger.story.ending_archetype
            fid['ending_title'] = ledger.story.ending_title
        with open(eval_path, 'w', encoding='utf-8') as f:
          json.dump(eval_rep, f, indent=2)
      except Exception:  # pylint: disable=broad-exception-caught
        pass

    if os.path.isfile(struct_path):
      try:
        with open(struct_path, 'r', encoding='utf-8') as f:
          sim_log = structured_logging.SimulationLog.from_json(f.read())
        rendered_html = sim_log.to_html()
        with open(html_path, 'w', encoding='utf-8') as f:
          f.write(rendered_html)
      except Exception:  # pylint: disable=broad-exception-caught
        pass

    updated.append({
        'dir': rdir,
        'ended': ledger.story.ended,
        'ending_archetype': ledger.story.ending_archetype,
        'ending_title': ledger.story.ending_title,
        'ending_slides_count': len(ledger.story.ending_slides),
    })
  return {'finalized': updated}


def inspect_offline_run_dirs(
    run_dirs: Sequence[str],
    *,
    cmd: str = 'stream',
    last_n: int = 0,
    npc_filter: str = '',
) -> str:
  """Formats offline Concordia structured logs and narrative streams via CLI."""
  blocks: list[str] = []
  for rdir in run_dirs:
    rdir = rdir.strip()
    if not rdir or not os.path.isdir(rdir):
      continue
    ledger_path = os.path.join(rdir, 'world_ledger.json')
    struct_path = os.path.join(rdir, 'simulation_structured.json')
    ledger_dict: dict[str, Any] = {}
    logs_dict: dict[str, Any] = {}
    memories: dict[str, Any] = {}
    if os.path.isfile(ledger_path):
      with open(ledger_path, 'r', encoding='utf-8') as f:
        ledger_dict = json.load(f)
    if os.path.isfile(struct_path):
      with open(struct_path, 'r', encoding='utf-8') as f:
        logs_dict = json.load(f)
      memories = dict(logs_dict.get('entity_memories') or {})
    state = {'ledger': ledger_dict, 'memories': memories}
    blocks.append(f'### RUN DIRECTORY: {rdir} ###')
    if cmd in ('stream', 'feed', 'log', 'inspect_logs'):
      blocks.append(format_stream(state, last_n=last_n))
    if cmd in ('npcs', 'inspect-npc', 'inspect_logs'):
      blocks.append(
          format_npcs(state, npc_filter=npc_filter, include_memories=True)
      )
    if cmd in ('debug', 'telemetry', 'diag'):
      blocks.append(format_debug(state, logs_dict=logs_dict or None))
    if cmd == 'inspect_logs' and os.path.isfile(struct_path):
      sim_log = structured_logging.SimulationLog.from_dict(logs_dict)
      interface = structured_logging.AIAgentLogInterface(sim_log)
      overview = interface.get_overview()
      blocks.append(
          'CONCORDIA AI-AGENT LOG OVERVIEW: '
          + json.dumps(overview, ensure_ascii=False)
      )
      for ent_name in overview.get('entities', []):
        if ent_name == 'AdventureGM':
          continue
        actions = interface.get_entity_actions(ent_name)
        act_lines = [f'  --- Entity Actions: {ent_name} ({len(actions)}) ---']
        for a in actions:
          act_text = str(a.get('action', '')).replace('\n', ' ').strip()
          act_lines.append(f"    Step {a.get('step', '?')}: {act_text}")
        blocks.append('\n'.join(act_lines))
  return '\n\n'.join(blocks)


def main(argv: Sequence[str]) -> None:
  pos_args = list(argv[1:])
  cmd = (flags.FLAGS.cmd or (pos_args[0] if pos_args else 'status')).lower()
  if cmd in ('finalize_epilogues', 'finalize'):
    raw_dirs = flags.FLAGS.run_dirs or ','.join(pos_args[1:])
    dirs = [d.strip() for d in raw_dirs.split(',') if d.strip()]
    res = finalize_epilogues(dirs)
    print(json.dumps(res, indent=2))
    return
  if cmd in ('compare_voices', 'compare_runs', 'compare'):
    raw_dirs = flags.FLAGS.run_dirs or ','.join(pos_args[1:])
    dirs = [d.strip() for d in raw_dirs.split(',') if d.strip()]
    comparison = compare_voice_runs(dirs)
    print(json.dumps(comparison, indent=2))
    return
  if flags.FLAGS.run_dirs and cmd in (
      'stream',
      'feed',
      'log',
      'npcs',
      'inspect-npc',
      'debug',
      'telemetry',
      'diag',
      'inspect_logs',
  ):
    dirs = [d.strip() for d in flags.FLAGS.run_dirs.split(',') if d.strip()]
    print(
        inspect_offline_run_dirs(
            dirs,
            cmd=cmd,
            last_n=flags.FLAGS.last,
            npc_filter=flags.FLAGS.npc,
        )
    )
    return

  client = AdventureCliClient(flags.FLAGS.url)

  try:
    if cmd in ('status', 'look', 'observe'):
      state = client.get_state()
      if flags.FLAGS.json_output:
        print(json.dumps(state, indent=2))
      else:
        print(format_status(state, last_n=flags.FLAGS.last))
      return

    if cmd in ('act', 'do', 'say', 'step', 'batch', 'play'):
      action_text = flags.FLAGS.action or ' '.join(pos_args[1:]).strip()
      if not action_text:
        raise app.UsageError(
            'Provide an action string via `adventure_cli -- act "<action>"` '
            'or `--action="<action>"`.'
        )
      actions = [a.strip() for a in action_text.split('||') if a.strip()]
      results = []
      for act_item in actions:
        res = client.act_and_wait(act_item, timeout_seconds=flags.FLAGS.timeout)
        results.append(res)
        if not flags.FLAGS.json_output:
          print(format_act_result(res), flush=True)
      if flags.FLAGS.json_output:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
      elif len(results) > 1:
        print(
            format_status(client.get_state(), last_n=flags.FLAGS.last),
            flush=True,
        )
      return

    if cmd in ('debug', 'telemetry', 'diag'):
      state = client.get_state()
      logs_dict = client.get_logs()
      if flags.FLAGS.json_output:
        print(
            json.dumps(
                {
                    'telemetry': state.get('ledger', {}).get('telemetry', {}),
                    'latency': state.get('latency', {}),
                    'npcs': state.get('ledger', {}).get('npcs', {}),
                    'story': state.get('ledger', {}).get('story', {}),
                    'log_entries_count': len(logs_dict.get('entries', [])),
                },
                indent=2,
            )
        )
      else:
        print(format_debug(state, logs_dict=logs_dict))
      return

    if cmd in ('stream', 'feed', 'log'):
      state = client.get_state()
      if flags.FLAGS.json_output:
        print(
            json.dumps(
                state.get('ledger', {}).get('narrative_log', []), indent=2
            )
        )
      else:
        print(format_stream(state, last_n=flags.FLAGS.last))
      return

    if cmd in ('npcs', 'inspect-npc'):
      npc_name = flags.FLAGS.npc or (
          ' '.join(pos_args[1:]).strip() if len(pos_args) > 1 else ''
      )
      state = client.get_state()
      if flags.FLAGS.json_output:
        print(json.dumps(state.get('ledger', {}).get('npcs', {}), indent=2))
      else:
        print(
            format_npcs(
                state,
                npc_filter=npc_name,
                include_memories=flags.FLAGS.memories,
            )
        )
      return

    if cmd in ('map', 'world'):
      state = client.get_state()
      if flags.FLAGS.json_output:
        print(
            json.dumps(state.get('ledger', {}).get('locations', {}), indent=2)
        )
      else:
        print(format_map(state))
      return

    if cmd in ('evaluate', 'eval'):
      out_dir = flags.FLAGS.output_dir or '/tmp/adventure_cli_eval'
      report = run_evaluate_command(
          client,
          output_dir=out_dir,
          run_autorater=flags.FLAGS.autorater,
          use_mock=bool(flags.FLAGS.use_mock),
          api_type=str(flags.FLAGS.api_type),
          model_name=str(flags.FLAGS.model_name),
          api_key=str(flags.FLAGS.api_key),
      )
      if flags.FLAGS.json_output:
        print(json.dumps(report, indent=2))
      else:
        print('=' * 76)
        print(f'EVALUATION SNAPSHOT EXPORTED TO: {out_dir}')
        print(json.dumps(report, indent=2))
        print('-' * 76)
        print(
            'Inspect exported Concordia logs with CLI:\n'
            '  python -m examples.adventure.adventure_cli '
            f'--run_dirs={out_dir} --cmd=inspect_logs'
        )
        print('=' * 76)
      return

    if cmd == 'reset':
      res = client.reset()
      if flags.FLAGS.json_output:
        print(json.dumps(res, indent=2))
      else:
        print('Simulation reset to Tick 0.')
        print(format_status(res.get('state', {}), last_n=flags.FLAGS.last))
      return

    raise app.UsageError(
        f'Unknown command {cmd!r}. Valid commands: status, act, debug, '
        'telemetry, stream, npcs, map, evaluate, reset.'
    )
  except urllib.error.URLError as err:
    raise SystemExit(
        f'Failed to connect to AdventureServer at {flags.FLAGS.url}: {err}'
    ) from err


if __name__ == '__main__':
  _define_flags()
  app.run(main)
