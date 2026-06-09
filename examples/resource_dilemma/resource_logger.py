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

"""Structured step-level logging for resource dilemma CPR simulations.

Provides a ``ResourceStepLoggerComponent`` — a *purely observational* Game
Master component that hooks into the Concordia entity-component lifecycle to
record structured simulation data after each resolved event:

  1. Current phase (harvesting, voting, discussion)
  2. Per-agent action (harvest amount, vote, or utterance)
  3. Resource stock levels (parsed from GM announcements, **not** computed)
  4. Current leader of the resource
  5. Cumulative harvests per agent (parsed from event text)
  6. Results of the last election

**This component does NOT apply any functional simulation logic** (e.g. stock
regeneration, harvest capping). All resource dynamics are the
responsibility of the Game Master's LLM-based reasoning.  The logger only
*observes* and *records*.

The component is designed to be attached to each Game Master in a multi-GM
simulation. All component instances share a single ``ResourceLoggerState``
object to maintain consistent state across phases.

Usage::

  # Create shared logger state
  logger_state = ResourceLoggerState(
      initial_resources=100.0,
      carrying_capacity=120.0,
      num_cycles=6,
  )

  # Pass logger_state to each GM prefab (via prefab.logger_state field).
  # The prefab's build() method creates a ResourceStepLoggerComponent and
  # wires it into the GM's component dict.
"""

import collections
import datetime
import html as html_lib
import json
import os
import re
from typing import Any, cast

from absl import logging
from concordia.associative_memory import basic_associative_memory as associative_memory
from examples.resource_dilemma import simulation_state as sim_state_lib
from examples.resource_dilemma.gamemaster import resource_components
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

# Component key for registering on GMs
DEFAULT_RESOURCE_LOGGER_COMPONENT_KEY = '__resource_step_logger__'

# ---------------------------------------------------------------------------
# Regex patterns for extracting structured data from LLM-generated text
# ---------------------------------------------------------------------------

# Votes: YES / NO / ABSTAIN
_VOTE_PATTERN = re.compile(r'\b(YES|NO|ABSTAIN)\b', re.IGNORECASE)

# Resource stock level: "stock is 85", "stock at 90", "stock: 72"
_STOCK_PATTERN = re.compile(
    r'(?:(?:fish|water|pasture|bandwidth|resource)\s+)?'
    r'stock\s*(?:is|at|:|=)?\s*(\d+(?:\.\d+)?)',
    re.IGNORECASE,
)

# Election winner: "WINNER: Name with 3 votes"
_LEADER_PATTERN = re.compile(
    r'WINNER:\s*(.+?)(?:\s+with\s+\d+|\s*$)', re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Shared mutable state (observational only — no simulation logic)
# ---------------------------------------------------------------------------


class ResourceLoggerState:
  """Shared mutable state for the resource step logger.

  This is a plain Python object (not a component) that can be shared across
  multiple ``ResourceStepLoggerComponent`` instances on different Game Masters.
  Each GM's component reads and writes to the same state so that cumulative
  harvests, cycle counts, etc. stay consistent across phases.

  Operational simulation state (resource level, cycle tracking, termination)
  is delegated to a ``ResourceSimulationState`` instance when provided.  This
  ensures that GM components (e.g. ``ResourceTerminate``) never depend on the
  logger — they depend on the simulation state directly.
  """

  def __init__(
      self,
      initial_resources: float = 100.0,
      carrying_capacity: float = 120.0,
      num_cycles: int = 6,
      min_resources_for_continuation: float = 5.0,
      html_output_path: str | None = None,
      max_steps: int = 0,
      player_names: list[str] | None = None,
      sim_state: sim_state_lib.ResourceSimulationState | None = None,
  ):
    """Initialise the logger state.

    Args:
      initial_resources: Starting resource stock (used as initial display value
        until the GM announces an updated stock level).
      carrying_capacity: Maximum resource stock (used for display only).
      num_cycles: Total number of cycles in the simulation.
      min_resources_for_continuation: Minimum resources required at the end of a
        cycle to continue the simulation. Matches the threshold used by
        `ResourceTerminate`.
      html_output_path: Optional CNS/local path for a live HTML log.
      max_steps: Total number of simulation steps (for progress display).
      player_names: Optional list of all player names in the simulation.
      sim_state: Optional shared simulation state.  When provided, operational
        fields (resource_level, carrying_capacity, current_cycle, total_cycles,
        terminated, cycle_harvest_total) are proxied to this object.
    """
    # Wire up the simulation state (source of truth for operational values)
    self._sim_state: sim_state_lib.ResourceSimulationState | None = sim_state

    self.min_resources_for_continuation = min_resources_for_continuation

    # Logging-only fields
    self.step_logs: list[dict[str, Any]] = []
    self.cumulative_harvests: dict[str, float] = collections.defaultdict(float)
    if player_names:
      for name in player_names:
        self.cumulative_harvests[name] = 0.0
    self.current_leader: str | None = None
    self.last_election_results: dict[str, Any] = {}
    self.last_phase: str | None = None
    self.step_count: int = 0
    self.phase_step_count: int = 0
    self.cycle_harvests: dict[str, float] = collections.defaultdict(float)
    self.cycle_votes: dict[str, str] = {}
    self.proposed_policies: dict[str, str] = {}
    self.html_output_path: str | None = html_output_path
    self.max_steps: int = max_steps or num_cycles * 30

    # Fallback local storage when no sim_state is provided (e.g. tests)
    if sim_state is None:
      self._resource_level: float = initial_resources
      self._carrying_capacity: float = carrying_capacity
      self._current_cycle: int = 1
      self._total_cycles: int = num_cycles
      self._terminated: bool = False
      self._cycle_harvest_total: float = 0.0

  # --- Properties reading from sim_state when available ---

  @property
  def resource_level(self) -> float:
    if self._sim_state is not None:
      return self._sim_state.resource_level
    return self._resource_level

  @resource_level.setter
  def resource_level(self, value: float) -> None:
    if self._sim_state is not None:
      return  # Ignore — sim state is updated by GM components
    self._resource_level = value

  @property
  def carrying_capacity(self) -> float:
    if self._sim_state is not None:
      return self._sim_state.carrying_capacity
    return self._carrying_capacity

  @carrying_capacity.setter
  def carrying_capacity(self, value: float) -> None:
    if self._sim_state is not None:
      return
    self._carrying_capacity = value

  @property
  def current_cycle(self) -> int:
    if self._sim_state is not None:
      return self._sim_state.current_cycle
    return self._current_cycle

  @current_cycle.setter
  def current_cycle(self, value: int) -> None:
    if self._sim_state is not None:
      return
    self._current_cycle = value

  @property
  def total_cycles(self) -> int:
    if self._sim_state is not None:
      return self._sim_state.total_cycles
    return self._total_cycles

  @total_cycles.setter
  def total_cycles(self, value: int) -> None:
    if self._sim_state is not None:
      return
    self._total_cycles = value

  @property
  def terminated(self) -> bool:
    if self._sim_state is not None:
      return self._sim_state.terminated
    return self._terminated

  @terminated.setter
  def terminated(self, value: bool) -> None:
    if self._sim_state is not None:
      return
    self._terminated = value

  @property
  def cycle_harvest_total(self) -> float:
    if self._sim_state is not None:
      return self._sim_state.cycle_harvest_total
    return self._cycle_harvest_total

  @cycle_harvest_total.setter
  def cycle_harvest_total(self, value: float) -> None:
    if self._sim_state is not None:
      return
    self._cycle_harvest_total = value

  def get_summary_json(self) -> str:
    """Return the full step log history as a JSON string."""
    return json.dumps(self.step_logs, indent=2, default=str)


# ---------------------------------------------------------------------------
# GM Component (purely observational)
# ---------------------------------------------------------------------------


class ResourceStepLoggerComponent(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """GM component that logs structured resource simulation data.

  Attach this component to each Game Master in the resource simulation.
  All instances should share the same ``ResourceLoggerState`` object so that
  state is consistent across GM transitions.

  **This is a logging-only component.** It extracts structured data from
  resolved events but does NOT apply any functional simulation logic
  (regeneration, harvest capping, etc.). All resource dynamics are handled by
  the GM's LLM-based reasoning.

  The component participates in the standard lifecycle:

  - ``pre_act``: Provides the current resource state summary as context for
    the GM's LLM reasoning (only during RESOLVE).
  - ``post_act``: Parses the resolved event text to extract structured data
    (harvests, votes, stock levels, leader changes), updates shared
    observational state, logs via the logging channel, optionally writes HTML.
  """

  def __init__(
      self,
      state: ResourceLoggerState,
      phase: str = 'unknown',
      memory_bank: associative_memory.AssociativeMemoryBank | None = None,
  ):
    """Initialise the logger component.

    Args:
      state: Shared ``ResourceLoggerState`` object.
      phase: The phase this GM represents ('harvesting', 'discussion',
        'voting').
      memory_bank: The shared memory bank of the GM.
    """
    super().__init__()
    self._state = state
    self._phase = phase
    self._memory_bank = memory_bank
    self._latest_action_spec: entity_lib.ActionSpec | None = None

  def _update_state_from_memory(self) -> None:
    """Update logger-only state by querying the GM shared memory.

    Updates logging-only fields (current_leader, cumulative_harvests,
    cycle_harvests).  Does NOT modify sim state fields (current_cycle,
    resource_level, cycle_harvest_total) — those are owned by GM components.
    """
    if not self._memory_bank:
      return

    s = self._state

    # Derive current leader (logging-only)
    election_memories = self._memory_bank.scan(
        lambda x: 'ELECTION RESULT: Winner is' in x
    )
    if election_memories:
      latest_election = election_memories[-1]
      match = re.search(r'Winner is\s+(.+?)(?:\s+with|\s*$)', latest_election)
      if match:
        s.current_leader = match.group(1).strip()

    # Derive cumulative harvests (logging-only)
    harvest_memories = self._memory_bank.scan(lambda x: 'decided to use:' in x)
    s.cumulative_harvests.clear()
    for memory in harvest_memories:
      match = re.search(
          r'(?:\[.*?\]\s*)?(.+?)\s+decided to use:\s*(.*)', memory
      )
      if match:
        name = match.group(1)
        amount = resource_components.extract_harvest_amount(match.group(2))
        if amount is not None:
          s.cumulative_harvests[name] += amount

    # Derive cycle harvests (logging-only)
    all_relevant = self._memory_bank.scan(
        lambda x: 'proposed policy:' in x or 'decided to use:' in x
    )

    s.cycle_harvests.clear()

    for memory in all_relevant:
      if 'proposed policy:' in memory:
        s.cycle_harvests.clear()
      elif 'decided to use:' in memory:
        match = re.search(
            r'(?:\[.*?\]\s*)?(.+?)\s+decided to use:\s*(.*)', memory
        )
        if match:
          name = match.group(1)
          amount = resource_components.extract_harvest_amount(match.group(2))
          if amount is not None:
            s.cycle_harvests[name] += amount

  # --- Lifecycle hooks ---------------------------------------------------

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    self._latest_action_spec = action_spec
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      s = self._state
      result = (
          f'Resource State: Resource level: {s.resource_level:.1f} '
          f'/ {s.carrying_capacity:.0f} capacity. '
          f'Cycle {s.current_cycle}/{s.total_cycles}. '
          f'Leader: {s.current_leader or "None elected"}.'
      )
      self._logging_channel({
          'Key': 'resource_state',
          'Summary': result,
          'Value': result,
      })
      return result
    return ''

  def post_act(
      self,
      action_attempt: str,
  ) -> str:
    if (
        self._latest_action_spec is None
        or self._latest_action_spec.output_type != entity_lib.OutputType.RESOLVE
    ):
      return ''

    s = self._state
    s.step_count += 1
    phase = self._phase

    logging.info(
        'RESOURCE_LOGGER: phase=%s, last_phase=%s', phase, s.last_phase
    )

    # Update state from GM shared memory
    self._update_state_from_memory()

    # Reset phase step counter on transition
    if s.last_phase is not None and s.last_phase != phase:
      s.phase_step_count = 0

    s.phase_step_count += 1

    # --- Parse resource stock from GM announcements (logging-only) ---
    stock_match = _STOCK_PATTERN.search(action_attempt)
    if stock_match:
      try:
        s.resource_level = float(stock_match.group(1))
      except ValueError:
        pass

    try:
      data = json.loads(action_attempt)
    except json.JSONDecodeError:
      data = None

    result = ''
    if data is not None:
      result = data.get('summary', '')
      individual_actions = data.get('individual_actions', {})

      for agent_name, action in individual_actions.items():

        parsed_action = {
            'raw': f'{agent_name}: {action}',
            'agent_name': agent_name,
        }

        if phase in ('harvesting', 'fishing'):
          harvest = resource_components.extract_harvest_amount(action)
          parsed_action['harvest_amount'] = harvest
          if harvest is not None and not self._memory_bank:
            s.cumulative_harvests[agent_name] += harvest
            s.cycle_harvests[agent_name] += harvest
            s.cycle_harvest_total += harvest
        elif phase == 'voting':
          parsed_action['vote'] = action
          s.cycle_votes[agent_name] = action
        elif phase == 'policy generation':
          s.proposed_policies[agent_name] = action

        step_log = {
            'step': s.step_count,
            'phase_step': s.phase_step_count,
            'phase': phase,
            'game_master': (
                self._latest_action_spec.tag
                if self._latest_action_spec
                else 'unknown'
            ),
            'timestamp': datetime.datetime.now().isoformat(),
            'action': parsed_action,
            'resource_level': s.resource_level,
            'cumulative_harvests': dict(s.cumulative_harvests),
            'current_leader': s.current_leader,
            'carrying_capacity': s.carrying_capacity,
            'cycle': s.current_cycle,
            'total_cycles': s.total_cycles,
        }
        s.step_logs.append(step_log)

    else:
      # --- Fallback: parse the resolved event as plain text ---
      parsed: dict[str, Any] = {'raw': action_attempt}

      if phase in ('harvesting', 'fishing'):
        harvest = resource_components.extract_harvest_amount(action_attempt)
        parsed['harvest_amount'] = harvest
        if harvest is not None:
          agent_name = _extract_agent_name(
              action_attempt, list(s.cumulative_harvests.keys())
          )
          if agent_name:
            s.cumulative_harvests[agent_name] += harvest
            parsed['agent_name'] = agent_name

      elif phase == 'voting':
        vote = _extract_vote(action_attempt)
        parsed['vote'] = vote
        if 'VOTE RESULT' in action_attempt:
          s.last_election_results['summary'] = action_attempt
          yes_m = re.search(r'YES\s*=\s*(\d+)', action_attempt)
          no_m = re.search(r'NO\s*=\s*(\d+)', action_attempt)
          if yes_m:
            s.last_election_results['yes'] = int(yes_m.group(1))
          if no_m:
            s.last_election_results['no'] = int(no_m.group(1))
          if 'PASSED' in action_attempt.upper():
            s.last_election_results['passed'] = (
                'DID NOT' not in action_attempt.upper()
            )

      elif phase == 'discussion':
        parsed['type'] = 'utterance'

      # Check for leader/winner announcements
      leader_match = _LEADER_PATTERN.search(action_attempt)
      if leader_match:
        candidate = leader_match.group(1).strip()
        if '[' not in candidate and ']' not in candidate:
          s.current_leader = candidate

      # --- Assemble step log ---
      step_log = {
          'timestamp': datetime.datetime.now().isoformat(),
          'step': s.step_count,
          'phase_step': s.phase_step_count,
          'total_steps': s.max_steps,
          'phase': phase,
          'game_master': self._get_gm_name(),
          'action': parsed,
          'current_leader': s.current_leader,
          'resource_level': s.resource_level,
          'carrying_capacity': s.carrying_capacity,
          'cycle': s.current_cycle,
          'total_cycles': s.total_cycles,
          'cumulative_harvests': dict(s.cumulative_harvests),
          'last_election_results': dict(s.last_election_results),
      }

      s.step_logs.append(step_log)
      _print_step_log(step_log)

      self._logging_channel({
          'Key': 'resource_step',
          'Summary': (
              f'Cycle {s.current_cycle}, step {s.phase_step_count}:'
              f' {phase} — {action_attempt[:120]}'
          ),
          'Value': json.dumps(step_log, default=str),
      })

    # --- Common post-processing (both JSON and fallback paths) ---

    s.last_phase = phase

    if phase in ('harvesting', 'fishing'):
      summary_log = self._log_cycle_summary()
      s.step_logs.append(summary_log)

    _write_html(s)

    return result

  def _log_cycle_summary(self) -> dict[str, Any]:
    s = self._state
    winner = s.current_leader or 'None'
    policy = s.proposed_policies.get(winner, 'N/A')

    # Read stock level and termination flag from sim state
    stock_remaining = s.resource_level
    continue_sim = (
        not s.terminated
        and stock_remaining > s.min_resources_for_continuation
        and s.current_cycle < s.total_cycles
    )
    if s.terminated:
      continue_str = 'NO (simulation terminated)'
    elif stock_remaining <= s.min_resources_for_continuation:
      continue_str = f'NO (stock below {s.min_resources_for_continuation:.1f})'
    elif not continue_sim:
      continue_str = 'NO (depleted or reached limit)'
    else:
      continue_str = 'YES'

    # Compute projected stock after regeneration (doubling, capped).
    if continue_sim:
      stock_after_regen = min(stock_remaining * 2.0, s.carrying_capacity)
      stock_line = (
          f'Stock remaining: {stock_remaining:.1f}'
          f' -> {stock_after_regen:.1f} (after regeneration)'
      )
    else:
      stock_after_regen = stock_remaining
      stock_line = f'Stock remaining: {stock_remaining:.1f}'

    summary_text = (
        f'--- CYCLE {s.current_cycle} SUMMARY ---\n'
        f'Leader: {winner}\n'
        f'Policy: {policy}\n'
        f'Votes: {dict(s.cycle_votes)}\n'
        f'Harvests: {dict(s.cycle_harvests)}\n'
        f'{stock_line}\n'
        f'Continue simulation: {continue_str}'
    )

    s.step_count += 1

    step_log = {
        'timestamp': datetime.datetime.now().isoformat(),
        'step': s.step_count,
        'phase_step': 0,
        'phase': 'summary',
        'game_master': 'logger',
        'action': {
            'raw': summary_text,
            'type': 'cycle_summary',
            'leader': winner,
            'policy': policy,
            'votes': dict(s.cycle_votes),
            'harvests': dict(s.cycle_harvests),
            'stock': stock_remaining,
            'stock_after_regen': stock_after_regen,
            'continue': continue_sim,
        },
        'current_leader': s.current_leader,
        'resource_level': s.resource_level,
        'carrying_capacity': s.carrying_capacity,
        'cycle': s.current_cycle,
        'total_cycles': s.total_cycles,
        'cumulative_harvests': dict(s.cumulative_harvests),
    }
    return step_log

  # --- State serialisation -----------------------------------------------

  def get_state(self) -> entity_component.ComponentState:
    s = self._state
    return {
        'step_count': s.step_count,
        'resource_level': s.resource_level,
        'cumulative_harvests': dict(s.cumulative_harvests),
        'current_leader': s.current_leader,
        'current_cycle': s.current_cycle,
        'last_phase': s.last_phase,
        'cycle_harvest_total': s.cycle_harvest_total,
        'last_election_results': dict(s.last_election_results),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    s = self._state
    s.step_count = cast(int, state.get('step_count', s.step_count))
    if 'cumulative_harvests' in state:
      harvests = cast(dict[str, float], state['cumulative_harvests'])
      s.cumulative_harvests = collections.defaultdict(float, harvests)
    s.current_leader = cast(
        str | None, state.get('current_leader', s.current_leader)
    )
    s.last_phase = cast(str | None, state.get('last_phase', s.last_phase))
    if 'last_election_results' in state:
      s.last_election_results = dict(
          cast(dict[str, Any], state['last_election_results'])
      )

  # --- Internal helpers ---------------------------------------------------

  def _get_gm_name(self) -> str:
    """Return the name of the parent GM entity, if available."""
    try:
      return self.get_entity().name
    except RuntimeError:
      return 'unknown'


# ---------------------------------------------------------------------------
# Extraction helpers (module-level, stateless)
# ---------------------------------------------------------------------------


def _extract_vote(action_text: str) -> str:
  """Extract a YES/NO/ABSTAIN vote from voting action text."""
  match = _VOTE_PATTERN.search(action_text)
  if match:
    return match.group(1).upper()
  return 'UNKNOWN'


def _extract_agent_name(text: str, known_names: list[str]) -> str | None:
  """Try to identify which agent is referenced in the event text."""
  for name in known_names:
    if name in text:
      return name
  return None


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def _print_step_log(step_log: dict[str, Any]) -> None:
  """Log a formatted summary of the step as a single atomic block."""
  sep = '=' * 60
  action = step_log.get('action', {})
  raw = action.get('raw', '')
  phase = step_log['phase']

  cycle = step_log.get('cycle', '?')
  phase_step = step_log.get('phase_step', step_log['step'])
  lines = [
      '',
      sep,
      (
          f'CYCLE {cycle}, STEP {phase_step} '
          f'| Phase: {phase.upper()} '
          f'| GM: {step_log["game_master"]} '
          f'| Time: {step_log.get("timestamp", "")}'
      ),
      sep,
  ]

  if phase in ('harvesting', 'fishing'):
    harvest = action.get('harvest_amount')
    harvest_str = f'{harvest}' if harvest is not None else '?'
    agent = action.get('agent_name', '?')
    lines.append(f'  [HARVEST] {agent}: {raw} — extracted: {harvest_str}')
  elif phase == 'voting':
    vote = action.get('vote', 'N/A')
    lines.append(f'  [VOTE] {raw} — extracted: {vote}')
  elif phase == 'discussion':
    lines.append(f'  [TALK] {raw}')
  elif phase == 'policy generation':
    lines.append(f'  [POLICY] {raw}')
  else:
    lines.append(f'  [????] {raw}')

  # State summary
  leader = step_log['current_leader'] or 'None elected'
  lines.append('  --- Simulation State ---')
  lines.append(f'  Resource level: {step_log["resource_level"]:.1f}')
  lines.append(f'  Current leader: {leader}')

  if step_log.get('cumulative_harvests'):
    harvests_str = ', '.join(
        f'{n}: {c:.1f}'
        for n, c in sorted(step_log['cumulative_harvests'].items())
    )
    lines.append(f'  Cumulative harvests: {harvests_str}')

  if step_log.get('last_election_results'):
    election = step_log['last_election_results'].get('summary', 'N/A')
    lines.append(f'  Last election: {election}')

  logging.info('\n'.join(lines))


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
  """HTML-escape a string."""
  return html_lib.escape(str(text))


def _truncate(text: str, max_len: int = 80) -> str:
  """Truncate text to max_len characters with ellipsis."""
  if len(text) > max_len:
    return text[:max_len] + '\u2026'
  return text


_PHASE_NAME_MAPPING = {
    'policy generation': 'policy',
    'voting': 'election',
    'fishing': 'harvest',
    'harvesting': 'harvest',
    'discussion': 'discuss',
}

_PHASE_ICONS = {
    'harvest': '\U0001f33e',  # Sheaf of rice — generic harvest
    'discuss': '\U0001f5e3',
    'election': '\U0001f5f3',
    'policy': '\U0001f4dc',
}


def _write_html(state: ResourceLoggerState) -> None:
  """Render and write both full and phase-specific HTML logs to disk."""
  phase = state.last_phase or 'unknown'
  clean_phase = _PHASE_NAME_MAPPING.get(phase, phase)
  path = state.html_output_path

  if not path:
    return

  # Write phase-specific log
  content_phase = _render_html(state, phase_filter=phase)
  try:
    path_phase = path.replace('.html', f'_{clean_phase}.html')
    os.makedirs(os.path.dirname(path_phase), exist_ok=True)
    with open(path_phase, 'w', encoding='utf-8') as f:
      f.write(content_phase)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning('Failed to write phase HTML log: %s', e)

  # Write full log
  content_full = _render_html(state)
  try:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
      f.write(content_full)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning('Failed to write full HTML log: %s', e)


def _render_html(
    state: ResourceLoggerState, phase_filter: str | None = None
) -> str:
  """Render all accumulated step logs as a self-contained HTML page."""
  is_final = state.step_logs and state.step_logs[-1]['step'] >= state.max_steps
  refresh_tag = '' if is_final else '<meta http-equiv="refresh" content="15">'
  now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

  show_summaries = phase_filter is None

  if phase_filter:
    logs_to_render = [
        log for log in state.step_logs if log.get('phase') == phase_filter
    ]
  else:
    logs_to_render = [
        log for log in state.step_logs if log.get('phase') == 'summary'
    ]

  latest = logs_to_render[-1] if logs_to_render else {}
  progress_pct = (
      int(100 * latest.get('step', 0) / state.max_steps)
      if state.max_steps
      else 0
  )
  status_text = 'COMPLETED' if is_final else 'RUNNING'

  # --- Build step cards ---
  cards_html = []
  for log in logs_to_render:
    cards_html.append(_render_step_card(log))

  # --- Cumulative harvests table ---
  harvests_rows = ''
  if state.cumulative_harvests:
    for name, total in sorted(
        state.cumulative_harvests.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
      harvests_rows += f'<tr><td>{_esc(name)}</td><td>{total:.1f}</td></tr>'

  leader_text = state.current_leader or 'None elected'
  if len(leader_text) > 80:
    leader_text = leader_text[:80] + '\u2026'
  leader_html = _esc(leader_text)
  election_html = _esc(state.last_election_results.get('summary', 'N/A'))

  state_panel_html = ''
  if show_summaries:
    state_panel_html = f"""
<div class="state-panel">
  <div class="state-item">
    <span class="state-label">\U0001f33e Resource Level</span>
    <span class="state-value">{state.resource_level:.1f} / {state.carrying_capacity:.0f}</span>
  </div>
  <div class="state-item">
    <span class="state-label">\U0001f504 Cycle</span>
    <span class="state-value">{state.current_cycle} / {state.total_cycles}</span>
  </div>
  <div class="state-item">
    <span class="state-label">\U0001f451 Current Leader</span>
    <span class="state-value">{leader_html}</span>
  </div>
  <div class="state-item">
    <span class="state-label">\U0001f5f3 Last Election</span>
    <span class="state-value">{election_html}</span>
  </div>
</div>
"""

  harvests_table_html = ''
  if show_summaries and harvests_rows:
    harvests_table_html = (
        '<div class="catches-table"><h3>Cumulative'
        f' Harvests</h3><table><tr><th>Agent</th><th>Total</th></tr>{harvests_rows}</table></div>'
    )

  return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Resource Dilemma — Simulation Log</title>
{refresh_tag}
<style>
{_CSS}
</style>
</head>
<body>
<header>
  <h1>\U0001f33e Resource Dilemma — Simulation Log</h1>
  <div class="meta">
    Status: <strong>{status_text}</strong> |
    Step {latest.get('step', 0)}/{state.max_steps}
    ({progress_pct}%) |
    Cycle <strong>{state.current_cycle}/{state.total_cycles}</strong> |
    Updated: {now}
  </div>
  <div class="progress-bar">
    <div class="progress-fill" style="width:{progress_pct}%"></div>
  </div>
</header>

{state_panel_html}

{harvests_table_html}

<div class="steps">
{''.join(cards_html)}
</div>

<footer>Auto-generated by ResourceStepLoggerComponent</footer>
</body>
</html>
"""


def _render_step_card(step_log: dict[str, Any]) -> str:
  """Render a single step as a collapsible HTML card."""
  phase = step_log['phase']
  clean_phase = _PHASE_NAME_MAPPING.get(phase, phase)
  valid_phases = ('harvest', 'discuss', 'election', 'policy')
  phase_class = clean_phase if clean_phase in valid_phases else 'unknown'
  phase_icon = _PHASE_ICONS.get(clean_phase, '\u2753')
  cycle = step_log.get('cycle', '?')
  phase_step = step_log.get('phase_step', step_log['step'])
  gm = _esc(step_log['game_master'])
  action = step_log.get('action', {})
  raw = _esc(action.get('raw', ''))
  timestamp = _esc(step_log.get('timestamp', ''))

  if phase in ('harvesting', 'fishing'):
    harvest = action.get('harvest_amount')
    agent = _esc(action.get('agent_name', '?'))
    badge = (
        f'<span class="badge fish">{harvest}</span>'
        if harvest is not None
        else '<span class="badge unknown">?</span>'
    )
    action_html = (
        '<div class="action-row">'
        f'<strong>{agent}</strong> {badge}'
        f'<div class="raw">{raw}</div></div>'
    )
  elif action.get('type') == 'cycle_summary':
    leader = _esc(action.get('leader', 'N/A'))
    policy = _esc(action.get('policy', 'N/A'))
    votes = action.get('votes', {})
    harvests = action.get('harvests', action.get('catches', {}))
    stock = action.get('stock', 'N/A')
    stock_after_regen = action.get('stock_after_regen')

    if (
        stock_after_regen is not None
        and stock_after_regen != stock
        and action.get('continue')
    ):
      stock_display = (
          f'{stock} \u2192 {stock_after_regen:.1f} (after regeneration)'
      )
    else:
      stock_display = f'{stock}'

    votes_list = ''.join(
        [f'<li>{_esc(k)}: {_esc(v)}</li>' for k, v in votes.items()]
    )
    harvests_list = ''.join(
        [f'<li>{_esc(k)}: {v}</li>' for k, v in harvests.items()]
    )

    action_html = f"""
<div class="cycle-summary">
  <h3>Cycle {cycle} summary</h3>
  <p><strong>Leader:</strong> {leader}</p>
  <p><strong>Policy:</strong> {policy}</p>
  <p><strong>Stock Remaining:</strong> {stock_display}</p>
  <div class="summary-details">
    <div class="summary-col">
      <h4>Votes</h4>
      <ul>{votes_list}</ul>
    </div>
    <div class="summary-col">
      <h4>Harvests</h4>
      <ul>{harvests_list}</ul>
    </div>
  </div>
</div>
"""
  elif phase == 'voting':
    vote = action.get('vote', 'N/A')
    vote_lower = vote.lower()
    action_html = (
        '<div class="action-row">'
        f'<span class="badge {vote_lower}">{vote}</span>'
        f'<div class="raw">{raw}</div></div>'
    )
  else:
    action_html = f'<div class="action-row"><div class="raw">{raw}</div></div>'

  return f"""
<details class="step-card {phase_class}" open>
  <summary>
    {phase_icon} Cycle {cycle}, step {phase_step} — {clean_phase.upper()}
    <span class="gm-label">GM: {gm}</span>
    <span class="gm-label">Time: {timestamp}</span>
  </summary>
  <div class="step-body">
    {action_html}
    <div class="step-state">
      <span>\U0001f33e Resource: {step_log['resource_level']:.1f} / {step_log.get('carrying_capacity', 120):.0f}</span>
      <span class="sep">|</span>
      <span>\U0001f504 Cycle {step_log.get('cycle', '?')}/{step_log.get('total_cycles', '?')}</span>
      <span class="sep">|</span>
      <span>\U0001f451 {_esc(_truncate(step_log['current_leader'] or 'None', 60))}</span>
    </div>
  </div>
</details>
"""


_CSS = """
:root {
  --fishing: #2196F3;
  --discussion: #4CAF50;
  --voting: #FF9800;
  --policy: #9C27B0;
  --unknown: #9E9E9E;
  --bg: #f5f5f5;
}
body {
  font-family: 'Segoe UI', Roboto, sans-serif;
  max-width: 900px; margin: 0 auto;
  padding: 16px; background: var(--bg); color: #222;
}
header { margin-bottom: 20px; }
h1 { margin: 0 0 4px; font-size: 1.5em; }
.meta { font-size: 0.9em; color: #555; margin-bottom: 8px; }
.progress-bar {
  height: 8px; background: #ddd; border-radius: 4px;
  overflow: hidden;
}
.progress-fill {
  height: 100%; background: var(--fishing); border-radius: 4px;
  transition: width 0.3s;
}
.state-panel {
  display: flex; gap: 12px; flex-wrap: wrap;
  margin-bottom: 16px;
}
.state-item {
  flex: 1 1 180px; background: #fff;
  border-radius: 8px; padding: 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.state-label { display: block; font-size: 0.85em; color: #666; }
.state-value { font-size: 1.2em; font-weight: 600; }
.catches-table {
  background: #fff; border-radius: 8px; padding: 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px;
}
.catches-table h3 { margin: 0 0 8px; font-size: 1em; }
.catches-table table { width: 100%; border-collapse: collapse; }
.catches-table th, .catches-table td {
  text-align: left; padding: 4px 8px;
  border-bottom: 1px solid #eee;
}
.steps { display: flex; flex-direction: column; gap: 8px; }
.step-card {
  background: #fff; border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  border-left: 4px solid var(--unknown);
}
.step-card.harvest { border-left-color: var(--fishing); }
.step-card.discuss { border-left-color: var(--discussion); }
.step-card.election { border-left-color: var(--voting); }
.step-card.policy { border-left-color: var(--policy); }
.step-card summary {
  padding: 10px 14px; cursor: pointer;
  font-weight: 600; font-size: 0.95em;
  list-style: none;
}
.step-card summary::-webkit-details-marker { display: none; }
.gm-label {
  float: right; font-weight: 400; font-size: 0.85em; color: #888;
}
.step-body { padding: 0 14px 12px; }
.action-row {
  padding: 6px 0; border-bottom: 1px solid #f0f0f0;
}
.action-row:last-child { border-bottom: none; }
.raw {
  font-size: 0.85em; color: #555; margin-top: 2px;
  word-break: break-word;
}
.badge {
  display: inline-block; padding: 2px 8px;
  border-radius: 12px; font-size: 0.8em;
  font-weight: 600; color: #fff;
}
.badge.fish { background: var(--fishing); }
.badge.yes { background: #4CAF50; }
.badge.no { background: #f44336; }
.badge.abstain { background: #9E9E9E; }
.badge.unknown { background: #bbb; }
.step-state {
  margin-top: 8px; font-size: 0.85em; color: #666;
  padding-top: 6px; border-top: 1px solid #eee;
  display: flex; align-items: center; gap: 4px; flex-wrap: wrap;
}
.sep { color: #ccc; margin: 0 4px; }
.cycle-summary {
  padding: 12px;
  background: #f9f9f9;
  border-radius: 6px;
  margin-top: 8px;
}
.cycle-summary p { margin: 4px 0; }
.summary-details {
  display: flex; gap: 20px; flex-wrap: wrap; margin-top: 10px;
}
.summary-col {
  flex: 1 1 200px;
}
.summary-col h4 { margin: 0 0 6px; font-size: 0.95em; color: #333; }
.summary-col ul { margin: 0; padding-left: 16px; font-size: 0.85em; color: #555; }
footer {
  margin-top: 20px; text-align: center;
  font-size: 0.8em; color: #999;
}
"""
