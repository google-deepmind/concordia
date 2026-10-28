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
"""Adventure-specific Game Master and Sub-Mind components.

Separates Game Master work into:
1. Pure-Python control hooks (`AdventureTerminate`, `AdventureNextActing`)
   paired with Concordia's stock `SwitchAct` and `Asynchronous` engine for zero
   control-hook LLM calls.
2. `FastSceneResolver`: Priority-locked synchronous LLM call per foreground
   `RESOLVE` (and Warm off-screen outcome routing) that outputs a vivid
   narrative outcome plus a structured JSON delta applied to `WorldLedger`.
3. `FrontierArchitectPublisher`: Asynchronous `pubsub.Publisher` context
   component mounted on the Game Master that expands adjacent unexpanded exit
   stubs and seeds Cold/Warm NPCs one hop ahead of the player.
"""

from collections.abc import Callable, Mapping, Sequence
import concurrent.futures
import datetime
import json
import math
import random
import re
import threading
import time
from typing import Any, override

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import pubsub
from concordia.components.game_master import event_resolution
from concordia.components.game_master import interrupt_scheduling
from concordia.components.game_master import interrupt_time_model
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting as next_acting_components
from concordia.components.game_master import terminate as terminate_components
from examples.adventure import roster_manager
from examples.adventure import story_director
from examples.adventure import story_state
from examples.adventure import world_ledger
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_FRONTIER_COMPONENT_KEY = '__frontier_architect__'
DEFAULT_SUBMIND_COUNCIL_COMPONENT_KEY = '__submind_council__'
DEFAULT_NPC_SUBMIND_COUNCIL_COMPONENT_KEY = '__npc_submind_council__'

_JSON_BLOCK_RE = re.compile(
    r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL | re.IGNORECASE
)


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
  """Extracts and parses the first valid JSON object from an LLM response."""
  if not raw_text:
    return None
  fence_match = _JSON_BLOCK_RE.search(raw_text)
  candidates = []
  if fence_match:
    candidates.append(fence_match.group(1))
  start = raw_text.find('{')
  end = raw_text.rfind('}')
  if start != -1 and end != -1 and end > start:
    candidates.append(raw_text[start : end + 1])

  for candidate in candidates:
    for variant in (candidate, candidate.replace("\\'", "'")):
      try:
        parsed = json.loads(variant)
        if isinstance(parsed, dict):
          return parsed
      except json.JSONDecodeError:
        continue
  return None


def sanitize_narrative_text(raw_text: str) -> str:
  """Extracts clean narrative prose from raw/truncated LLM output without JSON."""
  if not raw_text:
    return ''
  m = re.search(r'"narrative_event"\s*:\s*"((?:[^"\\]|\\.)*)', raw_text)
  if m:
    extracted = m.group(1).replace('\\"', '"').replace('\\n', ' ').strip()
    if extracted:
      return extracted
  cleaned = re.sub(r'```(?:json)?', '', raw_text, flags=re.IGNORECASE)
  if '{' in raw_text or '"delta"' in raw_text:
    cleaned = re.sub(r'"[a-zA-Z0-9_]+"\s*:', '', cleaned)
    cleaned = re.sub(r'[\{\}\[\]"`]', '', cleaned)
  else:
    cleaned = re.sub(r'[\{\}\[\]`]', '', cleaned)
  cleaned = re.sub(r'\s+', ' ', cleaned).strip()
  return cleaned


def dedup_leading_actor_name(actor: str, text: str) -> str:
  """Removes duplicate leading NPC name prefixes from resolved prose.

  Args:
    actor: Name of the acting NPC.
    text: Candidate narrative string.

  Returns:
    The narrative string with duplicate leading actor names stripped.
  """
  cleaned = text.strip()
  if not cleaned or not actor.strip():
    return cleaned
  actor_clean = actor.strip()
  cleaned = re.sub(
      r'^(?:'
      + re.escape(actor_clean)
      + r'\s+)?(?:answer|response|action)\s*:\s*',
      '',
      cleaned,
      flags=re.IGNORECASE,
  ).strip()
  tokens = [t for t in actor_clean.split() if len(t) >= 3]
  if cleaned.lower().startswith(actor_clean.lower()):
    rest = cleaned[len(actor_clean) :].lstrip(' :')
    rest_lower = rest.lower()
    if rest and rest_lower.startswith(actor_clean.lower()):
      cleaned = rest
    elif rest:
      for tok in tokens:
        tok_l = tok.lower()
        if rest_lower.startswith(tok_l + ' '):
          cleaned = f'{actor_clean} ' + rest[len(tok) :].lstrip()
          break
        if rest_lower.startswith((tok_l + "'", tok_l + '’', tok_l + ':')):
          cleaned = rest
          break
  return cleaned


def strip_leading_actor_name(actor: str, text: str) -> str:
  """Strips a leading actor name (and colon) so verb clauses can be embedded."""
  cleaned = dedup_leading_actor_name(actor, text)
  actor_clean = actor.strip()
  if actor_clean and cleaned.lower().startswith(actor_clean.lower()):
    rest = cleaned[len(actor_clean) :].lstrip(' :,-—')
    if rest and not rest.startswith(("'s", '’s')):
      return rest
  return cleaned


def truncate_at_word_boundary(text: str, max_chars: int = 90) -> str:
  """Truncates `text` at a word or sentence boundary without mid-word cuts."""
  cleaned = re.sub(r'\s+', ' ', text).strip()
  if len(cleaned) <= max_chars:
    return cleaned.rstrip(' ,;:-—')
  window = cleaned[:max_chars]
  for punct in ('. ', '; ', ', ', ' — ', ' - '):
    idx = window.rfind(punct)
    if idx >= max_chars // 2:
      return window[:idx].rstrip(' ,;:-—')
  space_idx = window.rfind(' ')
  if space_idx > 0:
    return window[:space_idx].rstrip(' ,;:-—')
  return window.rstrip(' ,;:-—')


class LatencyTracker:
  """Thread-safe tracker for LLM call counts and resolution latencies."""

  def __init__(self):
    """Initializes the instance."""
    self._lock = threading.Lock()
    self._player_resolve_ms: list[float] = []
    self._npc_resolve_ms: list[float] = []
    self._warm_step_ms: list[float] = []
    self._frontier_expand_ms: list[float] = []
    self._submind_council_ms: list[float] = []
    self._submind_parallel_calls: list[int] = []
    self._control_hook_llm_calls: int = 0
    self._resolve_llm_calls_per_turn: list[int] = []

  def record_resolve(
      self, duration_ms: float, *, is_player: bool, llm_calls: int = 1
  ) -> None:
    """Records a foreground scene resolution latency measurement."""
    with self._lock:
      if is_player:
        self._player_resolve_ms.append(duration_ms)
        self._resolve_llm_calls_per_turn.append(llm_calls)
      else:
        self._npc_resolve_ms.append(duration_ms)

  def record_warm_step(self, duration_ms: float) -> None:
    """Records a background Warm NPC step duration."""
    with self._lock:
      self._warm_step_ms.append(duration_ms)

  def record_frontier_expansion(self, duration_ms: float) -> None:
    """Records a background frontier room expansion duration."""
    with self._lock:
      self._frontier_expand_ms.append(duration_ms)

  def record_submind_council(
      self, duration_ms: float, *, parallel_calls: int = 0
  ) -> None:
    """Records a parallel Sub-Mind council deliberation duration."""
    with self._lock:
      self._submind_council_ms.append(duration_ms)
      self._submind_parallel_calls.append(parallel_calls)

  def record_control_hook_llm_call(self) -> None:
    """Increments the counter if a control hook unexpectedly invokes an LLM."""
    with self._lock:
      self._control_hook_llm_calls += 1

  def summary(self) -> dict[str, Any]:
    """Returns summary statistics (mean, p50, p95, call counts)."""
    with self._lock:

      def _stats(values: list[float]) -> dict[str, float]:
        if not values:
          return {'count': 0.0, 'mean_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}
        ordered = sorted(values)
        count = len(ordered)
        p50 = ordered[int(0.50 * (count - 1))]
        p95 = ordered[int(0.95 * (count - 1))]
        return {
            'count': float(count),
            'mean_ms': round(sum(ordered) / count, 2),
            'p50_ms': round(p50, 2),
            'p95_ms': round(p95, 2),
        }

      max_calls = (
          max(self._resolve_llm_calls_per_turn)
          if self._resolve_llm_calls_per_turn
          else 0
      )
      return {
          'control_hook_llm_calls': self._control_hook_llm_calls,
          'max_llm_calls_per_player_resolve': max_calls,
          'player_resolve': _stats(self._player_resolve_ms),
          'npc_resolve': _stats(self._npc_resolve_ms),
          'warm_step': _stats(self._warm_step_ms),
          'frontier_expand': _stats(self._frontier_expand_ms),
          'submind_council': _stats(self._submind_council_ms),
          'submind_parallel_calls_total': sum(self._submind_parallel_calls),
      }


class AdventureTerminate(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Pure-Python TERMINATE component for Concordia's `Asynchronous` engine.

  Ends the simulation when the player completes `max_player_steps`, when the
  authored story ends (`story.ended`), or when `request_stop()` is called.
  """

  def __init__(
      self,
      ledger: world_ledger.WorldLedger | None = None,
      *,
      max_player_steps: int = 30,
  ):
    """Initializes the instance."""
    super().__init__()
    self._lock = threading.Lock()
    self._explicit_ledger = ledger
    self._max_player_steps = max(1, max_player_steps)
    self._stop_requested = False
    self._epilogue_appended = False
    self._require_full_steps = False
    self._deferred_ended = False
    self._explicit_resolver: FastSceneResolver | None = None

  @property
  def _ledger(self) -> world_ledger.WorldLedger:
    if self._explicit_ledger is not None:
      return self._explicit_ledger
    return self.get_entity().get_component(
        world_ledger.DEFAULT_WORLD_LEDGER_COMPONENT_KEY,
        type_=world_ledger.WorldLedger,
    )

  @property
  def _resolver(self) -> 'FastSceneResolver | None':
    if self._explicit_resolver is not None:
      return self._explicit_resolver
    if self._entity is not None:
      try:
        return self._entity.get_component(
            event_resolution.DEFAULT_RESOLUTION_COMPONENT_KEY,
            type_=FastSceneResolver,
        )
      except KeyError:
        return None
    return None

  def set_resolver(self, resolver: 'FastSceneResolver') -> None:
    """Links the FastSceneResolver so player step counts can be checked."""
    self._explicit_resolver = resolver

  def set_max_player_steps(self, max_player_steps: int) -> None:
    """Sets the player step target and resets stop state for a new run."""
    with self._lock:
      self._max_player_steps = max(1, max_player_steps)
      self._stop_requested = False
      self._epilogue_appended = False
      self._deferred_ended = False

  def set_require_full_steps(self, require_full: bool) -> None:
    """When True, continues running until `max_player_steps` is reached."""
    with self._lock:
      self._require_full_steps = bool(require_full)

  def request_stop(self) -> None:
    """Requests all `Asynchronous` engine loops to exit on their next check."""
    with self._lock:
      self._stop_requested = True

  @property
  def stop_requested(self) -> bool:
    """Returns True if `request_stop()` has been called."""
    with self._lock:
      return self._stop_requested

  def should_terminate(self) -> bool:
    """Returns True when the simulation has reached a termination condition."""
    with self._lock:
      if self._stop_requested:
        return True
      resolver = self._resolver
      steps_done = (
          resolver.player_steps_completed if resolver is not None else 0
      )
      ledger = self._ledger
      if self._require_full_steps and steps_done < self._max_player_steps:
        if ledger.story.ended:
          self._deferred_ended = True
          ledger.story.ended = False
        return False
      story_ended = ledger.story.ended
      if not story_ended and steps_done < self._max_player_steps:
        return False
      director = ledger.director
      director_active = director is not None and getattr(
          director, 'active', False
      )
      if not story_ended and (
          self._deferred_ended
          or bool(ledger.story.ending_summary)
          or bool(ledger.story.ending_archetype)
          or (
              director_active
              and (self._require_full_steps or ledger.story.act >= 3)
          )
      ):
        ledger.story.ended = True
        story_ended = True
        if director_active and director is not None:
          arch = ledger.story.ending_archetype
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
          corrupt_truth = ledger.story.truths.get(
              't_butchers_bill'
          ) or ledger.story.truths.get('t_contract')
          if lead_domain in (
              'coercion',
              'force',
              'tactical_force',
              'predation',
          ) and (len(wars) >= 1 or cracked_wounds >= 1):
            preserved_outcome = 'chaos'
          elif (
              lead_domain in ('procedure', 'forensics', 'cybernetics')
              and corrupt_truth is not None
              and corrupt_truth.discovered
          ):
            preserved_outcome = 'reckoning'
          elif lead_domain in ('leverage', 'deception'):
            preserved_outcome = 'compromise'
          elif arch == 'eleventh_hour_bell':
            preserved_outcome = 'failure'
          else:
            preserved_outcome = 'unresolved'
          climax_loc = (
              ledger.story.visited_location_ids[0]
              if ledger.story.visited_location_ids
              else ledger.player_location_id
          )
          ledger.story.ending_summary = director.compose_epilogue(
              outcome=preserved_outcome,
              location_id=climax_loc,
          )
      if (
          story_ended
          and ledger.story.ending_summary
          and not self._epilogue_appended
      ):
        self._epilogue_appended = True
        ledger.append_narrative_entry(
            kind='ending',
            actor='Epilogue',
            text=ledger.story.ending_summary,
            location_id=ledger.player_location_id,
        )
      return True

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.TERMINATE:
      return ''
    done = self.should_terminate()
    val = (
        entity_lib.BINARY_OPTIONS['affirmative']
        if done
        else entity_lib.BINARY_OPTIONS['negative']
    )
    self._logging_channel({'Key': '__terminate__', 'Value': val})
    return val

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'max_player_steps': self._max_player_steps,
          'stop_requested': self._stop_requested,
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      raw_max = state.get('max_player_steps', self._max_player_steps)
      if isinstance(raw_max, (int, float, str)):
        self._max_player_steps = max(1, int(raw_max))
      self._stop_requested = bool(state.get('stop_requested', False))

  @override
  def get_dynamic_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {'max_player_steps': self._max_player_steps}


class AdventureNextActing(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Pure-Python NEXT_ACTING and NEXT_ACTION_SPEC component (zero LLM calls).

  Gates the player thread on `player_ready_fn()`, gates Hot NPC threads on
  scene presence, direct address, and cooldown pacing, and schedules background
  Warm-tier off-screen NPC steps via `EntityScheduler`.
  """

  def __init__(
      self,
      ledger: world_ledger.WorldLedger | None = None,
      *,
      player_ready_fn: Callable[[], bool] | None = None,
      observation_queue: make_observation.ObservationQueue | None = None,
      hot_cooldown_seconds: float = 0.05,
      hot_ambient_tick_interval: int = 1,
      max_consecutive_npc_turns: int = 1,
      warm_tick_interval: int = 3,
      time_model: interrupt_time_model.TimeModel | None = None,
  ):
    """Initializes the instance.

    Args:
      ledger: Optional explicit WorldLedger (otherwise looked up on GM entity).
      player_ready_fn: Returns True when the player has a pending action ready.
      observation_queue: Queue feeding MakeObservation for Warm off-screen
        steps.
      hot_cooldown_seconds: Minimum wall-clock gap between ambient NPC actions.
      hot_ambient_tick_interval: Minimum tick gap between ambient NPC actions.
      max_consecutive_npc_turns: Max NPC actions allowed between player turns.
      warm_tick_interval: Player steps between Warm off-screen NPC steps.
      time_model: Optional shared TimeModel for Warm scheduling.
    """
    super().__init__()
    self._lock = threading.RLock()
    self._explicit_ledger = ledger
    self._has_external_player_ready = player_ready_fn is not None
    self._player_ready_fn = player_ready_fn or (lambda: True)
    self._observation_queue = observation_queue
    self._hot_cooldown_seconds = hot_cooldown_seconds
    self._hot_ambient_tick_interval = hot_ambient_tick_interval
    self._max_consecutive_npc_turns = max_consecutive_npc_turns
    self._warm_tick_interval = max(1, warm_tick_interval)
    self._npc_turns_since_player = 0
    self._has_player_acted = False
    self._last_warm_player_step = 0
    self._in_flight_warm_turns: dict[str, tuple[Any, str]] = {}
    self._turns_in_flight: set[str] = set()
    self._explicit_resolver: FastSceneResolver | None = None
    self._explicit_terminate: AdventureTerminate | None = None
    self._explicit_frontier_publisher: Any | None = None
    self._on_turn_state_change: Callable[[], None] | None = None
    self._initial_frontier_triggered = False

    self._time_model = time_model or interrupt_time_model.DatetimeTimeModel(
        start_time=datetime.datetime(2026, 1, 1, 8, 0, 0)
    )
    self._fallback_warm_scheduler = interrupt_scheduling.EntityScheduler(
        player_names=[],
        time_model=self._time_model,
    )
    self._warm_registered: set[str] = set()

  @property
  def _ledger(self) -> world_ledger.WorldLedger:
    if self._explicit_ledger is not None:
      return self._explicit_ledger
    return self.get_entity().get_component(
        world_ledger.DEFAULT_WORLD_LEDGER_COMPONENT_KEY,
        type_=world_ledger.WorldLedger,
    )

  @property
  def _resolver(self) -> 'FastSceneResolver | None':
    if self._explicit_resolver is not None:
      return self._explicit_resolver
    if self._entity is not None:
      try:
        return self._entity.get_component(
            event_resolution.DEFAULT_RESOLUTION_COMPONENT_KEY,
            type_=FastSceneResolver,
        )
      except KeyError:
        return None
    return None

  @property
  def _terminate(self) -> AdventureTerminate | None:
    if self._explicit_terminate is not None:
      return self._explicit_terminate
    if self._entity is not None:
      try:
        return self._entity.get_component(
            terminate_components.DEFAULT_TERMINATE_COMPONENT_KEY,
            type_=AdventureTerminate,
        )
      except KeyError:
        return None
    return None

  @property
  def _frontier_publisher(self) -> Any | None:
    if self._explicit_frontier_publisher is not None:
      return self._explicit_frontier_publisher
    if self._entity is not None:
      try:
        return self._entity.get_component(DEFAULT_FRONTIER_COMPONENT_KEY)
      except KeyError:
        return None
    return None

  def set_resolver(self, resolver: 'FastSceneResolver') -> None:
    """Links the FastSceneResolver for step and busy checks."""
    self._explicit_resolver = resolver

  def set_terminate(self, terminate: AdventureTerminate) -> None:
    """Links the AdventureTerminate component."""
    self._explicit_terminate = terminate

  def set_frontier_publisher(self, publisher: Any) -> None:
    """Links the FrontierArchitect publisher for initial room expansion."""
    self._explicit_frontier_publisher = publisher

  def set_on_turn_state_change(
      self, callback: Callable[[], None] | None
  ) -> None:
    """Registers a callback invoked when turn in-flight state changes."""
    with self._lock:
      self._on_turn_state_change = callback

  def _notify_turn_state_change(self) -> None:
    """Notifies listeners (such as SSE clients) that turn state changed."""
    with self._lock:
      cb = self._on_turn_state_change
    if cb is not None:
      try:
        cb()
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  @property
  def warm_scheduler(self) -> interrupt_scheduling.EntityScheduler:
    """Returns the GM-mounted EntityScheduler (or local fallback in tests)."""
    if self._entity is not None:
      try:
        return self._entity.get_component(
            interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY,
            type_=interrupt_scheduling.EntityScheduler,
        )
      except KeyError:
        pass
    return self._fallback_warm_scheduler

  @property
  def _warm_scheduler(self) -> interrupt_scheduling.EntityScheduler:
    return self.warm_scheduler

  def is_turn_busy(self) -> bool:
    """Returns True when an entity has been selected by NEXT_ACTING."""
    with self._lock:
      return bool(self._turns_in_flight)

  def is_player_turn_in_flight(self) -> bool:
    """Returns True while the player's own action is in flight or resolving."""
    player_name = self._ledger.player_name
    with self._lock:
      if (
          self._resolver is not None
          and self._resolver.is_resolve_busy()
          and self._resolver.resolving_actor
      ):
        return self._resolver.resolving_actor == player_name
      return player_name in self._turns_in_flight

  def responding_npc_names(self) -> list[str]:
    """Returns co-located Hot NPC names currently deliberating or pending."""
    player_name = self._ledger.player_name
    player_loc_id = self._ledger.player_location_id
    with self._lock:
      if self.is_player_turn_in_flight():
        return []
      responding: list[str] = []
      resolving_actor = (
          self._resolver.resolving_actor
          if self._resolver is not None and self._resolver.is_resolve_busy()
          else ''
      )
      if (
          resolving_actor
          and resolving_actor != player_name
          and resolving_actor not in self._in_flight_warm_turns
      ):
        rec = self._ledger.get_npc(resolving_actor)
        if rec is not None and rec.location_id == player_loc_id:
          responding.append(resolving_actor)

      for name in sorted(self._turns_in_flight):
        if (
            name == player_name
            or name in self._in_flight_warm_turns
            or name in responding
        ):
          continue
        rec = self._ledger.get_npc(name)
        if rec is not None and rec.location_id == player_loc_id:
          responding.append(name)

      if (
          not responding
          and self._has_player_acted
          and self._npc_turns_since_player < self._max_consecutive_npc_turns
      ):
        co_located_hot = [
            r
            for r in self._ledger.npcs_at_location(player_loc_id)
            if r.tier == world_ledger.NpcTier.HOT
        ]
        if co_located_hot:
          addressed = [r.name for r in co_located_hot if r.directly_addressed]
          if addressed:
            for name in addressed:
              if name not in responding:
                responding.append(name)
          else:
            oldest_tick = min(r.last_active_tick for r in co_located_hot)
            for rec in co_located_hot:
              if (
                  rec.last_active_tick <= oldest_tick
                  and self._ledger.tick - rec.last_active_tick
                  >= self._hot_ambient_tick_interval
                  and rec.name not in responding
              ):
                responding.append(rec.name)
      return responding

  def mark_turn_finished(self, actor: str) -> None:
    """Clears the in-flight marker for `actor` after RESOLVE completes."""
    with self._lock:
      self._turns_in_flight.discard(actor)

  def pop_in_flight_warm_turn(self, actor: str) -> tuple[Any, str] | None:
    """Pops and returns `(warm_move, location_id)` if `actor` acted as Warm."""
    with self._lock:
      return self._in_flight_warm_turns.pop(actor, None)

  def notify_player_acted(self) -> None:
    """Resets the consecutive NPC turn counter after a player action."""
    with self._lock:
      self._has_player_acted = True
      self._npc_turns_since_player = 0

  def has_pending_npc_turn(self) -> bool:
    """Returns True if a co-located Hot NPC is eligible to react right now."""
    with self._lock:
      if not self._has_player_acted:
        return False
      if self._npc_turns_since_player >= self._max_consecutive_npc_turns:
        return False
      for rec in self._ledger.npcs_at_location(self._ledger.player_location_id):
        if rec.tier == world_ledger.NpcTier.HOT:
          if rec.directly_addressed or (
              self._ledger.tick - rec.last_active_tick
              >= self._hot_ambient_tick_interval
          ):
            return True
      return False

  def has_due_warm_turn(self) -> bool:
    """Returns True when an off-screen Warm NPC turn is due or in flight."""
    with self._lock:
      if self._in_flight_warm_turns:
        return True
      player_steps = (
          self._resolver.player_steps_completed
          if self._resolver is not None
          else 0
      )
      if player_steps - self._last_warm_player_step < self._warm_tick_interval:
        return False
      return any(
          r.location_id != self._ledger.player_location_id
          for r in self._ledger.npcs_in_tier(world_ledger.NpcTier.WARM)
      )

  def _sync_warm_scheduler_roster(self) -> list[world_ledger.NpcRecord]:
    """Registers Warm NPCs in `EntityScheduler` with InterruptMasks & Timers."""
    warm_records = self._ledger.npcs_in_tier(world_ledger.NpcTier.WARM)
    now_dt = self._warm_scheduler.get_current_time()
    for rec in warm_records:
      if rec.name not in self._warm_registered:
        self._warm_registered.add(rec.name)
        self._warm_scheduler._player_names.append(rec.name)  # pylint: disable=protected-access
        self._warm_scheduler._masks[rec.name] = interrupt_scheduling.MATCH_ALL  # pylint: disable=protected-access
        self._warm_scheduler._timers[rec.name] = None  # pylint: disable=protected-access
        self._warm_scheduler._pending[rec.name] = []  # pylint: disable=protected-access
      self._warm_scheduler.set_mask(
          rec.name,
          interrupt_scheduling.InterruptMask(
              prefixes=(f'scene.{rec.location_id}', 'world.rumors')
          ),
      )
      if self._warm_scheduler.get_timer(rec.name) is None:
        expiry = self._time_model.add_duration(now_dt, '15m')
        self._warm_scheduler.set_timer(
            rec.name,
            interrupt_scheduling.Timer(
                expiry=expiry,
                entity_name=rec.name,
                description=f'Pursue goal: {rec.goal}',
            ),
        )
    return warm_records

  def _can_warm_npc_act(self, name: str) -> bool:
    """Returns True if `name` is the due off-screen Warm NPC for this step."""
    with self._lock:
      if self._has_external_player_ready and self._player_ready_fn():
        return False
      if self._ledger.player_name in self._turns_in_flight:
        return False
      rec = self._ledger.get_npc(name)
      if rec is None or rec.tier != world_ledger.NpcTier.WARM:
        return False
      if rec.location_id == self._ledger.player_location_id:
        return False
      if self._in_flight_warm_turns:
        return False
      if self.has_pending_npc_turn():
        return False
      if self._resolver is not None and self._resolver.is_resolve_busy():
        return False
      player_steps = (
          self._resolver.player_steps_completed
          if self._resolver is not None
          else 0
      )
      if player_steps - self._last_warm_player_step < self._warm_tick_interval:
        return False

      eligible_warm = [
          r
          for r in self._sync_warm_scheduler_roster()
          if r.location_id != self._ledger.player_location_id
      ]
      if not eligible_warm:
        return False
      recent_rumors = self._ledger.get_rumors(limit=30)
      last_rumor = recent_rumors[-1] if recent_rumors else ''
      eligible_warm.sort(
          key=lambda r: (
              1 if f'via {r.name})' in last_rumor else 0,
              sum(1 for rm in recent_rumors if f'via {r.name})' in rm),
              r.last_active_tick,
              r.name,
          )
      )
      chosen = eligible_warm[0]
      if chosen.name != name:
        return False

      self._last_warm_player_step = player_steps
      if self._ledger.director is not None:
        self._ledger.director.step_cold_routines()

      now_dt = self._warm_scheduler.get_current_time()
      next_dt = self._time_model.add_duration(now_dt, '15m')
      self._warm_scheduler.advance_time(next_dt)
      self._warm_scheduler.poll_entity(chosen.name)
      self._warm_scheduler.set_timer(
          chosen.name,
          interrupt_scheduling.Timer(
              expiry=self._time_model.add_duration(next_dt, '20m'),
              entity_name=chosen.name,
              description=f'Continue goal: {chosen.goal}',
          ),
      )
      self._warm_scheduler.increment_tick()

      loc_node = self._ledger.get_location(chosen.location_id)
      loc_name = loc_node.name if loc_node else chosen.location_id
      obs_text = (
          f'[Off-Screen Tick {self._ledger.tick}] {chosen.name} is at '
          f'{loc_name} (the {self._ledger.player_name} is NOT in this room) '
          f'pursue-focusing their goal: {chosen.goal}.'
      )
      if self._observation_queue is not None:
        self._observation_queue.add(chosen.name, obs_text, ())

      director = self._ledger.director
      warm_move = (
          director.choose_warm_move(chosen) if director is not None else None
      )
      self._in_flight_warm_turns[chosen.name] = (
          warm_move,
          chosen.location_id,
      )
      return True

  def _can_npc_act(self, name: str) -> bool:
    """Returns True if the named Hot NPC should act on this step."""
    with self._lock:
      if self._has_external_player_ready and self._player_ready_fn():
        return False
      if self._ledger.player_name in self._turns_in_flight:
        return False
      if not self._has_player_acted:
        return False
      if self._npc_turns_since_player >= self._max_consecutive_npc_turns:
        return False
      rec = self._ledger.get_npc(name)
      if rec is None or rec.tier != world_ledger.NpcTier.HOT:
        return False
      if rec.location_id != self._ledger.player_location_id:
        return False
      co_located_hot = [
          r
          for r in self._ledger.npcs_at_location(
              self._ledger.player_location_id
          )
          if r.tier == world_ledger.NpcTier.HOT
      ]
      any_addressed = any(r.directly_addressed for r in co_located_hot)
      now = time.time()
      if rec.directly_addressed:
        self._npc_turns_since_player += 1
        return True
      if any_addressed:
        # Yield to the co-located Hot NPC who was directly addressed.
        return False
      if co_located_hot:
        oldest_tick = min(r.last_active_tick for r in co_located_hot)
        if rec.last_active_tick > oldest_tick:
          return False
      elapsed_wall = now - rec.last_action_wall_time
      elapsed_ticks = self._ledger.tick - rec.last_active_tick
      if (
          elapsed_wall >= self._hot_cooldown_seconds
          and elapsed_ticks >= self._hot_ambient_tick_interval
      ):
        self._npc_turns_since_player += 1
        return True
      return False

  def _maybe_trigger_initial_frontier(self) -> None:
    """Kicks off background frontier expansion once at Tick 0 if needed."""
    with self._lock:
      if self._initial_frontier_triggered:
        return
      pub = self._frontier_publisher
      if pub is None or not self._ledger.unexpanded_adjacent_locations():
        return
      self._initial_frontier_triggered = True
    try:
      if hasattr(pub, 'trigger_if_needed'):
        pub.trigger_if_needed()
      else:
        pub.update()
        pub.get_pre_act_value()
    except Exception as err:  # pylint: disable=broad-exception-caught
      logging.warning('Initial frontier trigger warning: %s', err)

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      return self.get_entity().name

    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      candidates = list(action_spec.options)
      if len(candidates) > 1:
        # Asynchronous.run_loop startup probe passes all entities at once.
        with self._lock:
          self._turns_in_flight.clear()
          self._in_flight_warm_turns.clear()
        self._maybe_trigger_initial_frontier()
        return ''
      if self._terminate is not None and self._terminate.should_terminate():
        return ''
      for name in candidates:
        if name.startswith('__npc_slot_'):
          continue
        with self._lock:
          self._turns_in_flight.discard(name)
          self._in_flight_warm_turns.pop(name, None)
        if name == self._ledger.player_name:
          if not self._has_external_player_ready:
            with self._lock:
              busy = bool(self._turns_in_flight)
            if busy or self.has_pending_npc_turn() or self.has_due_warm_turn():
              return ''
          if self._player_ready_fn():
            with self._lock:
              self._turns_in_flight.add(name)
            self._notify_turn_state_change()
            return name
        elif self._can_npc_act(name):
          with self._lock:
            self._turns_in_flight.add(name)
          self._notify_turn_state_change()
          return name
        elif self._can_warm_npc_act(name):
          with self._lock:
            self._turns_in_flight.add(name)
          return name
      return ''

    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      cta = action_spec.call_to_action.strip()
      matched_npc = next(
          (npc_name for npc_name in self._ledger.all_npcs() if npc_name in cta),
          '',
      )
      if not matched_npc and (not cta or self._ledger.player_name in cta):
        loc = self._ledger.get_player_location()
        exits = ', '.join(loc.exits.keys()) or 'none'
        prompt_cta = (
            f'You are at {loc.name} (Exits: {exits}). '
            f'What does {self._ledger.player_name} do or say next?'
        )
      elif matched_npc and matched_npc in self._in_flight_warm_turns:
        warm_move, warm_loc_id = self._in_flight_warm_turns[matched_npc]
        rec = self._ledger.get_npc(matched_npc)
        loc_node = self._ledger.get_location(warm_loc_id)
        loc_name = loc_node.name if loc_node else warm_loc_id
        last_warm_text = ''
        for entry in reversed(self._ledger.get_narrative_log()):
          if entry.get('actor') == matched_npc and entry.get('kind') == 'rumor':
            last_warm_text = truncate_at_word_boundary(
                str(entry.get('text', '')), 120
            )
            break
        warm_anti_repeat = ''
        if last_warm_text:
          warm_anti_repeat = (
              ' Do NOT repeat your previous off-screen action '
              f'("{last_warm_text}"); escalate to a new physical alteration, '
              'lock change, or pressure shift that spills into adjoining '
              'passages.'
          )
        if warm_move is not None:
          prompt_cta = f'{warm_move.call_to_action}{warm_anti_repeat}'
        else:
          goal_text = rec.goal if rec is not None else 'advance your agenda'
          prompt_cta = (
              f'You are {matched_npc} acting off-screen at {loc_name}'
              f' ({self._ledger.player_name} is NOT present here). In ONE'
              ' concise sentence (under 25 words, no spoken dialogue to'
              f' {self._ledger.player_name}), state one concrete off-screen'
              ' action you take right now to advance your goal'
              f' ({goal_text}).{warm_anti_repeat}'
          )
      else:
        actor_label = matched_npc or cta
        last_full_npc_text = ''
        last_npc_tick = 0
        is_reentry_turn = False
        saw_movement = False
        latest_player_action = ''
        log_entries = self._ledger.get_narrative_log()
        for idx in range(len(log_entries) - 1, -1, -1):
          entry = log_entries[idx]
          if (
              not latest_player_action
              and entry.get('kind') == 'player_action'
              and str(entry.get('text', '')).strip()
          ):
            latest_player_action = str(entry.get('text', '')).strip()
          if entry.get('kind') in ('movement', 'arrival'):
            saw_movement = True
          if (
              entry.get('actor') == actor_label
              and entry.get('kind') == 'npc_resolution'
          ):
            last_full_npc_text = str(entry.get('text', '')).strip()
            last_npc_tick = int(entry.get('tick', 0))
            is_reentry_turn = saw_movement
            break
        clean_player_action = re.sub(
            r'^\[[^\]]+\]\s*', '', latest_player_action
        ).strip()
        prior_npc_turns = [
            str(e.get('text', '')).strip()
            for e in log_entries
            if e.get('actor') == actor_label
            and e.get('kind') == 'npc_resolution'
            and str(e.get('text', '')).strip()
        ]
        prior_summary = ' | '.join(
            truncate_at_word_boundary(t, 110) for t in prior_npc_turns[-2:]
        )
        continuity_note = ''
        if is_reentry_turn and last_full_npc_text:
          elapsed_ticks = max(1, int(self._ledger.tick) - last_npc_tick)
          continuity_note = (
              f' {self._ledger.player_name} just returned after {elapsed_ticks}'
              ' ticks away; react to what changed since your last exchange'
              f' ("{truncate_at_word_boundary(last_full_npc_text, 160)}") and'
              ' any new evidence they carry without repeating your earlier'
              ' line.'
          )
        elif prior_summary:
          continuity_note = (
              f' Do not repeat your earlier move ({prior_summary}); let your'
              ' winning inner voice escalate or pivot.'
          )
        if clean_player_action:
          player_prompt_clause = (
              ' First, respond directly in dialogue and physical action to what'
              f' {self._ledger.player_name} just did or asked'
              f' ("{truncate_at_word_boundary(clean_player_action, 140)}"),'
              ' explicitly naming or addressing the specific object, question,'
              ' or physical move they used.'
          )
        else:
          player_prompt_clause = ''
        if self._ledger.prose_mode == 'concise':
          length_rule = (
              'CONCISE MODE: Reply in 1-2 short, punchy sentences (MAXIMUM 35'
              ' WORDS TOTAL) combining one concrete physical gesture in the'
              ' room and one direct line of spoken dialogue in quotation marks'
              ' (zero flowery filler, never narrating'
              f' {self._ledger.player_name} as "You").'
          )
        else:
          length_rule = (
              'using concrete physical action in the room and spoken dialogue'
              ' in quotation marks (1-3 sentences, never narrating'
              f' {self._ledger.player_name} as "You").'
          )
        prompt_cta = (
            f'What does {actor_label} say and do next?{player_prompt_clause}'
            ' Let the WINNING EXECUTIVE SUB-MIND in your Inner Parliament'
            f' shape how you answer or counter them, {length_rule}'
            f'{continuity_note}'
        )
      spec = entity_lib.free_action_spec(call_to_action=prompt_cta)
      return json.dumps({
          'call_to_action': spec.call_to_action,
          'output_type': spec.output_type.value,
          'options': list(spec.options),
          'tag': spec.tag,
      })

    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'npc_turns_since_player': self._npc_turns_since_player,
          'last_warm_player_step': self._last_warm_player_step,
          'warm_tick_interval': self._warm_tick_interval,
          'max_consecutive_npc_turns': self._max_consecutive_npc_turns,
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      val = state.get('npc_turns_since_player', 0)
      self._npc_turns_since_player = (
          int(val) if isinstance(val, (int, float, str)) else 0
      )
      warm_val = state.get('last_warm_player_step', 0)
      self._last_warm_player_step = (
          int(warm_val) if isinstance(warm_val, (int, float, str)) else 0
      )
      if 'warm_tick_interval' in state and isinstance(
          state['warm_tick_interval'], (int, float, str)
      ):
        self._warm_tick_interval = max(1, int(state['warm_tick_interval']))
      if 'max_consecutive_npc_turns' in state and isinstance(
          state['max_consecutive_npc_turns'], (int, float, str)
      ):
        self._max_consecutive_npc_turns = max(
            0, int(state['max_consecutive_npc_turns'])
        )

  @override
  def get_dynamic_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'warm_tick_interval': self._warm_tick_interval,
          'max_consecutive_npc_turns': self._max_consecutive_npc_turns,
      }


class FastSceneResolver(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Single-call synchronous scene resolver for `OutputType.RESOLVE`.

  Observes putative events, adjudicates them against the local scene state in
  `WorldLedger` with a single LLM call, applies structured deltas to the
  ledger, and pushes formatted observations to co-located entities.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      ledger: world_ledger.WorldLedger | None = None,
      roster: roster_manager.RosterManager | None = None,
      observation_queue: make_observation.ObservationQueue | None = None,
      *,
      next_acting_component: AdventureNextActing | None = None,
      frontier_publisher: Any | None = None,
      frontier_worker: Any | None = None,
      latency_tracker: LatencyTracker | None = None,
      pre_act_label: str = 'FastSceneResolver',
  ):
    """Initializes the instance.

    Args:
      model: Language model for adjudicating scene actions.
      ledger: Optional explicit WorldLedger (otherwise looked up on GM entity).
      roster: Optional explicit RosterManager (otherwise looked up on GM
        entity).
      observation_queue: Queue feeding the GM's MakeObservation component.
      next_acting_component: Optional AdventureNextActing to reset NPC pacing.
      frontier_publisher: Optional async Publisher to trigger frontier gen.
      frontier_worker: Optional FrontierArchitectWorker to sync in-flight jobs.
      latency_tracker: Optional LatencyTracker for telemetry.
      pre_act_label: Component label for structured logging.
    """
    super().__init__()
    self._model = model
    self._explicit_ledger = ledger
    self._explicit_roster = roster
    self._observation_queue = (
        observation_queue
        if observation_queue is not None
        else make_observation.ObservationQueue()
    )
    self._explicit_next_acting = next_acting_component
    self._explicit_frontier_publisher = frontier_publisher
    self._explicit_frontier_worker = frontier_worker
    self._latency_tracker = latency_tracker or LatencyTracker()
    self._pre_act_label = pre_act_label
    self._lock = threading.RLock()
    self._latest_putative_action: str = ''
    self._latest_acting_entity: str = ''
    self._latest_resolved_event: str = ''
    self._pending_putative_by_actor: dict[str, str] = {}
    self._pending_putative_order: list[str] = []
    self._npc_turn_history: dict[str, list[str]] = {}
    self._explicit_terminate: AdventureTerminate | None = None
    self._submind_council: SubMindCouncilCoordinator | None = None
    self._on_turn_state_change: Callable[[], None] | None = None
    self._player_steps_completed = 0
    self._resolve_in_progress = False
    self._resolving_actor: str = ''

  def set_submind_council(self, council: 'SubMindCouncilCoordinator') -> None:
    """Links the player's SubMindCouncilCoordinator for post-turn updates."""
    self._submind_council = council

  def set_on_turn_state_change(
      self, callback: Callable[[], None] | None
  ) -> None:
    """Registers a callback invoked when turn resolution state changes."""
    with self._lock:
      self._on_turn_state_change = callback
    if self._next_acting is not None:
      self._next_acting.set_on_turn_state_change(callback)

  def _notify_turn_state_change(self) -> None:
    """Notifies listeners (such as SSE clients) that turn state changed."""
    with self._lock:
      cb = self._on_turn_state_change
    if cb is not None:
      try:
        cb()
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  @property
  def _ledger(self) -> world_ledger.WorldLedger:
    if self._explicit_ledger is not None:
      return self._explicit_ledger
    return self.get_entity().get_component(
        world_ledger.DEFAULT_WORLD_LEDGER_COMPONENT_KEY,
        type_=world_ledger.WorldLedger,
    )

  @property
  def _roster(self) -> roster_manager.RosterManager:
    if self._explicit_roster is not None:
      return self._explicit_roster
    return self.get_entity().get_component(
        roster_manager.DEFAULT_ROSTER_COMPONENT_KEY,
        type_=roster_manager.RosterManager,
    )

  @property
  def _next_acting(self) -> AdventureNextActing | None:
    if self._explicit_next_acting is not None:
      return self._explicit_next_acting
    if self._entity is not None:
      try:
        return self._entity.get_component(
            next_acting_components.DEFAULT_NEXT_ACTING_COMPONENT_KEY,
            type_=AdventureNextActing,
        )
      except KeyError:
        return None
    return None

  @property
  def _terminate(self) -> AdventureTerminate | None:
    if self._explicit_terminate is not None:
      return self._explicit_terminate
    if self._entity is not None:
      try:
        return self._entity.get_component(
            terminate_components.DEFAULT_TERMINATE_COMPONENT_KEY,
            type_=AdventureTerminate,
        )
      except KeyError:
        return None
    return None

  @property
  def _frontier_publisher(self) -> Any | None:
    if self._explicit_frontier_publisher is not None:
      return self._explicit_frontier_publisher
    if self._entity is not None:
      try:
        return self._entity.get_component(DEFAULT_FRONTIER_COMPONENT_KEY)
      except KeyError:
        return None
    return None

  @property
  def _frontier_worker(self) -> Any | None:
    if self._explicit_frontier_worker is not None:
      return self._explicit_frontier_worker
    pub = self._frontier_publisher
    if pub is not None and hasattr(pub, 'worker'):
      return pub.worker
    return None

  def _soften_repeated_npc_beats(self, actor: str, text: str) -> str:
    """Prevents an NPC from repeating the same coarse beat regex on >=60% of turns."""
    prior_turns = self._npc_turn_history.setdefault(actor, [])
    result = text
    block_re = re.compile(
        r'\b(plants? (himself|herself|a boot|her boot|his boot)|'
        r'steps? (between|in front)|bars? (the|any|your)|blocks?|'
        r'slams? (both|a|her|his) (palms?|hands?)|pins? the|'
        r'puts? (himself|herself) between|squares? up)\b',
        re.IGNORECASE,
    )
    demand_re = re.compile(
        r'\b(hand (it |them )?over|turn (them|it) over|surrender|'
        r'spit (it )?out|produce|state your business|empty your|give (me|it)|'
        r"where('s| is) (the|my)|tell me (where|who)|pay (up|what)|"
        r'cough (it )?up)\b',
        re.IGNORECASE,
    )
    threat_re = re.compile(
        r'\b(irons?|manacles?|brig|cells?|arrest|nick you|haul you|clap|'
        r'flay|drown|toss you|kneecaps?|break your|cut you|shiv|blade|'
        r'shoot|bullet|bury you|end you|hurt you|the river|the thames|'
        r"or i('ll| will)|or you('ll| will) (be|find|wish)|last warning)\b",
        re.IGNORECASE,
    )
    if block_re.search(result):
      for pat, rep in (
          (
              r'\bplants (himself|herself|a boot|her boot|his boot)\b',
              'stands poised',
          ),
          (
              r'\bplant (himself|herself|a boot|her boot|his boot)\b',
              'stand poised',
          ),
          (r'\bsteps (between|in front)\b', 'shifts beside'),
          (r'\bstep (between|in front)\b', 'shift beside'),
          (r'\bbars (the|any|your)\b', r'watches \1'),
          (r'\bbar (the|any|your)\b', r'watch \1'),
          (r'\bblocks\b', 'guards'),
          (r'\bblock\b', 'guard'),
          (
              r'\bslams (both|a|her|his) (palms?|hands?)\b',
              'rests a knuckle',
          ),
          (
              r'\bslam (both|a|her|his) (palms?|hands?)\b',
              'rest a knuckle',
          ),
          (r'\bpins the\b', 'steadies the'),
          (r'\bpin the\b', 'steady the'),
          (
              r'\bputs (himself|herself) between\b',
              r'angles \1 near',
          ),
          (
              r'\bput (himself|herself) between\b',
              r'angle \1 near',
          ),
          (r'\bsquares up\b', 'straightens his lapels'),
          (r'\bsquare up\b', 'straighten his lapels'),
      ):
        result = re.sub(pat, rep, result, flags=re.IGNORECASE)

    if demand_re.search(result):
      for pat, rep in (
          (r'\bhand (it|them) over\b', r'place \1 on the table'),
          (r'\bhand over\b', 'slide across'),
          (r'\bturn (them|it) over\b', r'leave \1 here'),
          (r'\bsurrender\b', 'yield'),
          (r'\bspit (it )?out\b', 'speak plainly'),
          (r'\bproduce\b', 'present'),
          (r'\bstate your business\b', 'name your angle'),
          (r'\bempty your\b', 'check your'),
          (r'\bgive (me|it)\b', 'let us see'),
          (r"\bwhere('s| is) (the|my)\b", 'what became of the'),
          (r'\btell me (where|who)\b', r'say \1'),
          (r'\bpay (up|what)\b', 'settle the account'),
          (r'\bcough (it|them) up\b', r'put \1 on the baize'),
          (r'\bcough up\b', 'put down'),
      ):
        result = re.sub(pat, rep, result, flags=re.IGNORECASE)

    if threat_re.search(result):
      for pat, rep in (
          (r'\birons?\b', 'steel'),
          (r'\bmanacles?\b', 'restraints'),
          (r'\bbrig\b', 'hold'),
          (r'\bcells?\b', 'holding rooms'),
          (r'\barrest\b', 'detain'),
          (r'\bnick you\b', 'book you'),
          (r'\bhaul you\b', 'escort you'),
          (r'\bclap\b', 'lock'),
          (r'\bflay\b', 'strip'),
          (r'\bdrown\b', 'sink'),
          (r'\btoss you\b', 'send you'),
          (r'\bkneecaps?\b', 'ankles'),
          (r'\bbreak your\b', 'ruin your'),
          (r'\bcut you\b', 'mark you'),
          (r'\bshiv\b', 'point'),
          (r'\bblade\b', 'edge'),
          (r'\bshoot\b', 'fire'),
          (r'\bbullet\b', 'lead'),
          (r'\bbury you\b', 'leave you'),
          (r'\bend you\b', 'finish you'),
          (r'\bhurt you\b', 'cost you'),
          (r'\bthe river\b', 'the water'),
          (r'\bthe thames\b', 'the tideway'),
          (r"\bor i('ll| will)\b", 'before I'),
          (
              r"\bor you('ll| will) (be|find|wish)\b",
              r'and you may \2',
          ),
          (r'\blast warning\b', 'final courtesy'),
      ):
        result = re.sub(pat, rep, result, flags=re.IGNORECASE)

    prior_turns.append(result)
    return result

  def set_frontier_publisher(self, publisher: Any) -> None:
    """Wires the background FrontierArchitect publisher after construction."""
    self._explicit_frontier_publisher = publisher

  def set_terminate(self, terminate: AdventureTerminate) -> None:
    """Wires the AdventureTerminate component after construction."""
    self._explicit_terminate = terminate

  @property
  def player_steps_completed(self) -> int:
    """Returns the number of resolved player actions so far."""
    with self._lock:
      return self._player_steps_completed

  def is_resolve_busy(self) -> bool:
    """Returns True while a scene resolution is executing."""
    with self._lock:
      return self._resolve_in_progress

  @property
  def resolving_actor(self) -> str:
    """Returns the entity name currently being resolved, or empty string."""
    with self._lock:
      return self._resolving_actor

  def is_turn_busy(self) -> bool:
    """Returns True while an entity turn or scene resolution is in flight."""
    if self.is_resolve_busy():
      return True
    if self._next_acting is not None and self._next_acting.is_turn_busy():
      return True
    return False

  def get_pre_act_label(self) -> str:
    """Returns the component label."""
    return self._pre_act_label

  def get_pre_act_value(self) -> str:
    """Returns the latest resolved narrative event."""
    with self._lock:
      return self._latest_resolved_event

  @override
  def pre_observe(self, observation: str) -> str:
    tag = event_resolution.PUTATIVE_EVENT_TAG
    if observation.startswith(tag):
      raw = observation[len(tag) :].strip()
      actor = self._ledger.player_name
      action_body = raw
      if ':' in raw:
        prefix, rest = raw.split(':', 1)
        actor = prefix.strip()
        action_body = rest.strip()
      with self._lock:
        self._latest_acting_entity = actor
        self._latest_putative_action = action_body
        self._pending_putative_by_actor[actor] = action_body
        if actor in self._pending_putative_order:
          self._pending_putative_order.remove(actor)
        self._pending_putative_order.append(actor)
    return ''

  def _resolve_warm_turn(
      self,
      actor: str,
      putative_action: str,
      warm_move: Any,
      warm_loc_id: str,
      start_time: float,
  ) -> str:
    """Resolves a Warm off-screen NPC turn with zero additional LLM calls."""
    clean_action = strip_leading_actor_name(
        actor, sanitize_narrative_text(putative_action.strip())
    )
    if clean_action.startswith(f'{actor}:'):
      clean_action = clean_action[len(actor) + 1 :].strip()
    if '. ' in clean_action and len(clean_action) > 140:
      clean_action = clean_action.split('. ', 1)[0].strip() + '.'
    elif len(clean_action) > 160:
      clean_action = clean_action[:157].rsplit(' ', 1)[0].strip()
    if '(' in clean_action and clean_action.rfind('(') > clean_action.rfind(
        ')'
    ):
      clean_action = clean_action[: clean_action.rfind('(')].rstrip(' ,;:-—')

    chosen = self._ledger.get_npc(actor)
    dest_id = warm_loc_id
    dest_node = self._ledger.get_location(dest_id)
    dest_name = dest_node.name if dest_node else dest_id

    self._ledger.mark_npc_acted(actor)
    trace_clause = truncate_at_word_boundary(clean_action, 150).rstrip(' .,;:')
    if '(' in trace_clause and trace_clause.rfind('(') > trace_clause.rfind(
        ')'
    ):
      trace_clause = trace_clause[: trace_clause.rfind('(')].rstrip(' .,;:-—')
    trace_words = trace_clause.split()
    dangling_tail = {
        'the',
        'a',
        'an',
        'of',
        'to',
        'in',
        'on',
        'at',
        'by',
        'for',
        'from',
        'with',
        'while',
        'using',
        'and',
        'or',
        'into',
        'across',
        'through',
        'under',
        'over',
    }
    while (
        len(trace_words) > 4
        and trace_words[-1].lower().strip('.,;:') in dangling_tail
    ):
      trace_words.pop()
    trace_clause = ' '.join(trace_words).rstrip(' .,;:')
    trace_prop = (
        f"evidence of {actor}'s recent activity ({actor} {trace_clause})"
    )
    if dest_node is not None:
      self._ledger.apply_delta(
          {'add_affordances': [trace_prop], 'advance_minutes': 1},
          acting_entity=actor,
          narrative_event='',
      )
      spillover_prop = (
          f'physical spillover from {dest_name} ({actor} {trace_clause})'
      )
      spillover_event = (
          f'[Spillover from adjacent {dest_name}] {actor} {trace_clause}.'
      )
      adj_ids = set(dest_node.exits.values())
      for cand_id, cand_node in self._ledger.all_locations().items():
        if dest_id in cand_node.exits.values():
          adj_ids.add(cand_id)
      for adj_id in adj_ids:
        if adj_id == dest_id:
          continue
        adj_node = self._ledger.get_location(adj_id)
        if adj_node is not None and adj_node.expanded:
          self._ledger.add_location_affordance(adj_id, spillover_prop)
          self._ledger.record_consequence(adj_id, spillover_event)

    director = self._ledger.director
    if director is not None and warm_move is not None and chosen is not None:
      rumor_text = director.apply_warm_outcome(
          chosen, warm_move, clean_action, dest_id, dest_name
      )
    else:
      role_str = (
          chosen.role_summary if chosen is not None else 'local inhabitant'
      )
      rumor_text = (
          f'Word spreads from {dest_name}: {actor} ({role_str}) {clean_action}'
      )
      self._ledger.record_rumor(rumor_text, source=actor, is_warm=True)

    self._roster.rebalance_tiers()
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    self._latency_tracker.record_warm_step(duration_ms)
    with self._lock:
      self._latest_resolved_event = rumor_text
    self._logging_channel({
        'Key': 'WarmChronicler',
        'Summary': rumor_text[:120],
        'Value': rumor_text,
        'Actor': actor,
        'Location': dest_id,
        'LatencyMs': round(duration_ms, 2),
        'Measurements': {
            'latency_ms': round(duration_ms, 2),
            'tick': self._ledger.tick,
        },
    })
    return rumor_text

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ''

    start_time = time.perf_counter()
    with self._lock:
      active_key = getattr(self.get_entity(), '_active_capture_key', '')
      if active_key and active_key in self._pending_putative_by_actor:
        actor = active_key
        putative_action = self._pending_putative_by_actor.pop(actor)
        if actor in self._pending_putative_order:
          self._pending_putative_order.remove(actor)
      elif self._pending_putative_order:
        actor = self._pending_putative_order.pop(0)
        putative_action = self._pending_putative_by_actor.pop(actor, '')
      else:
        actor = self._latest_acting_entity or self._ledger.player_name
        putative_action = self._latest_putative_action

    if (
        not putative_action.strip()
        or actor.startswith('__npc_slot_')
        or (self._terminate is not None and self._terminate.should_terminate())
    ):
      if self._next_acting is not None:
        self._next_acting.pop_in_flight_warm_turn(actor)
        self._next_acting.mark_turn_finished(actor)
      with self._lock:
        self._latest_resolved_event = ''
        self._resolving_actor = ''
      self._notify_turn_state_change()
      return ''

    with self._lock:
      self._resolve_in_progress = True
      self._resolving_actor = actor
    try:
      warm_info = (
          self._next_acting.pop_in_flight_warm_turn(actor)
          if self._next_acting is not None
          else None
      )
      if warm_info is not None:
        with self._lock:
          self._resolving_actor = ''
        warm_move, warm_loc_id = warm_info
        return self._resolve_warm_turn(
            actor, putative_action, warm_move, warm_loc_id, start_time
        )
      return self._resolve_foreground_turn(actor, putative_action, start_time)
    finally:
      with self._lock:
        self._resolve_in_progress = False
        self._resolving_actor = ''
      if self._next_acting is not None:
        self._next_acting.mark_turn_finished(actor)
      self._notify_turn_state_change()

  def _resolve_foreground_turn(
      self, actor: str, putative_action: str, start_time: float
  ) -> str:
    """Adjudicates a player or Hot NPC foreground action."""

    is_player = actor == self._ledger.player_name
    if not is_player:
      self._ledger.mark_npc_acted(actor)

    npc_rec = None if is_player else self._ledger.get_npc(actor)
    if is_player or npc_rec is None:
      loc_id = self._ledger.player_location_id
    else:
      loc_id = npc_rec.location_id
    if is_player and putative_action.strip():
      self._ledger.record_player_action(
          putative_action.strip(), location_id=loc_id
      )

    # Detect explicit exit traversal before building scene context so the GM
    # adjudicates arrival against the destination room's actual NPCs and props
    # rather than bleeding origin-room NPCs across the threshold.
    pre_move_target_id = ''
    if is_player:
      current_node = self._ledger.get_location(loc_id)
      if current_node is not None:
        lower_act = putative_action.lower()
        if any(
            verb in lower_act
            for verb in (
                'walk ',
                'go ',
                'head ',
                'enter ',
                'travel ',
                'move ',
                'take the ',
                'return to ',
                'backtrack ',
            )
        ):
          for cand_loc in self._ledger.all_locations().values():
            if cand_loc.location_id == loc_id:
              continue
            cand_name = cand_loc.name.lower() if cand_loc.name else ''
            if cand_loc.location_id.lower() in lower_act or (
                cand_name and cand_name in lower_act
            ):
              pre_move_target_id = cand_loc.location_id
              break
          if not pre_move_target_id:
            for exit_dir, target_id in current_node.exits.items():
              target_node = self._ledger.get_location(target_id)
              target_name = target_node.name.lower() if target_node else ''
              if (
                  f' {exit_dir.lower()}' in f' {lower_act}'
                  or target_id.lower() in lower_act
                  or (target_name and target_name in lower_act)
              ):
                pre_move_target_id = target_id
                break

    if pre_move_target_id and self._frontier_worker is not None:
      self._frontier_worker.wait_until_idle(timeout=2.0)
      dest_check = self._ledger.get_location(pre_move_target_id)
      if dest_check is not None and not dest_check.expanded:
        self._frontier_worker.expand_stub_sync(dest_check, origin_id=loc_id)
      start_time = time.perf_counter()

    context_loc_id = pre_move_target_id or loc_id
    scene_context = self._ledger.build_scene_context(
        context_loc_id, is_arrival=bool(pre_move_target_id)
    )
    transition_note = ''
    if pre_move_target_id:
      origin_node = self._ledger.get_location(loc_id)
      dest_node = self._ledger.get_location(pre_move_target_id)
      origin_title = origin_node.name if origin_node else loc_id
      dest_title = dest_node.name if dest_node else pre_move_target_id
      transition_note = (
          f'Movement Transition: {actor} is departing {origin_title} and'
          f' arriving at {dest_title} (set `"move_to":'
          f' "{pre_move_target_id}"`). NPCs from {origin_title} stay behind and'
          f' DO NOT enter {dest_title}.\n'
      )

    if not is_player:
      narrative_event = dedup_leading_actor_name(
          actor, sanitize_narrative_text(putative_action).strip()
      )
      actor_tokens = [t.lower() for t in actor.split() if len(t) >= 3]
      lower_ev = narrative_event.lower()
      if (
          narrative_event
          and actor.lower() not in lower_ev
          and not any(lower_ev.startswith(tok) for tok in actor_tokens)
      ):
        narrative_event = f'{actor} {narrative_event}'
      narrative_event = self._soften_repeated_npc_beats(actor, narrative_event)
      npc_delta: dict[str, Any] = {'advance_minutes': 5}
      llm_calls_used = 0
      director = self._ledger.director
      if (
          loc_id == self._ledger.player_location_id
          and director is not None
          and director.active
          and self._ledger.story.tone.inner_voices
          and self._ledger.story.player_voices
      ):
        active_voices = director.select_active_voices(loc_id, k=4)
        voices_str = '; '.join(
            f'{v_name} [Lv {aff}] ({v_desc} | Focus: {focus})'
            for v_name, v_desc, focus, aff in active_voices
        )
        priority_facs = [v_name for v_name, _, _, _ in active_voices[:2]]
        priority_str = (
            f' (use {" and ".join(priority_facs)})' if priority_facs else ''
        )
        prev_actions = '; '.join(
            f'{opt.voice}: "{opt.action}"'
            for opt in self._ledger.story.active_voice_options
            if opt.action
        )
        submind_prompt = (
            'Sub-Mind Council Update (HRL Options for the'
            f' Player):\n{scene_context}\nLatest NPC Event in Room by {actor}:'
            f' {narrative_event}\nActive Sub-Mind Council: {voices_str}\nPrior'
            f' Options (NEVER repeat): {prev_actions}\n\nGenerate 4 fresh HRL'
            ' sub-mind options for the player that react directly to what'
            f' {actor} just said and did, plus 1-2 brief inner_voices'
            f' reactions{priority_str}. Keep each perception under 15 words and'
            ' each action to a concise 6-12 word imperative move. Respond ONLY'
            ' with a valid JSON object:\n{\n  "inner_voices": ["Faculty: sharp'
            ' 1-line reaction"],\n  "voice_options": [\n    {\n      "voice":'
            ' "Faculty Name",\n      "perception": "Punchy 1-sentence read'
            ' (max 15 words).",\n      "action": "Concise 6-12 word action'
            ' responding to the NPC."\n    }\n  ]\n}'
        )
        raw_submind = self._model.sample_text(
            submind_prompt, max_tokens=700, terminators=()
        )
        llm_calls_used = 1
        parsed_submind = extract_json_object(raw_submind)
        if isinstance(parsed_submind, Mapping):
          if isinstance(parsed_submind.get('voice_options'), list):
            npc_delta['voice_options'] = parsed_submind['voice_options']
          if isinstance(parsed_submind.get('inner_voices'), list):
            npc_delta['inner_voices'] = parsed_submind['inner_voices']
      delta: Mapping[str, Any] = npc_delta
    else:
      is_concise = self._ledger.prose_mode == 'concise'
      if is_concise:
        concise_banner = (
            '\nCONCISE MODE (ACTIVE — STRICT LENGTH & CLARITY RULES):\n'
            '- Keep `"narrative_event"` to 1-2 crisp, direct sentences '
            '(MAXIMUM 40 WORDS TOTAL).\n'
            '- Sentence 1 (Outcome First): State the exact physical or social '
            "result of the player's action in plain, concrete English in the "
            'very first clause. Cut weather, smell, lighting metaphors, and '
            'decorative adjectives.\n'
            '- Sentence 2 (Consequence / Clue / Reaction): State the single '
            'immediate tangible change, discovered clue specifics (exact '
            "names/numbers), or present NPC's physical reaction.\n"
            '- Resolve ONLY what the player just attempted—do NOT stack '
            'multiple side-plots or forced off-screen sabotage collisions into '
            'one turn.\n'
        )
        rule_5_str = (
            '5. If the player specifically interacts with an off-screen trace'
            ' or doorway altered by `Off-Screen World Rumors`, reflect that'
            ' physical alteration in `narrative_event` and'
            ' `"add_affordances"`.\n'
        )
        pivot_clause = ''
        narrative_event_schema_desc = (
            '1-2 crisp, plain-spoken sentences (MAX 40 words total): sentence 1'
            ' states the direct outcome of the action; sentence 2 states the'
            ' immediate consequence, clue specifics, or NPC reaction (no'
            ' flowery filler).'
        )
      else:
        concise_banner = ''
        rule_5_str = (
            '5. If `Off-Screen World Rumors` or `[Spillover from adjacent'
            ' ...]` events appear in the context, actively manifest their'
            ' physical consequence in this room so it alters physical'
            ' traversal, jams or reroutes a doorway/chute/conduit, or forces'
            ' the actor to physically navigate around the off-screen'
            ' modification in `narrative_event` and `"add_affordances"`'
            ' (never relegate off-screen events solely to background'
            ' dialogue).\n'
        )
        pivot_clause = (
            ' When an eligible beat in `Director Notes` describes an initial'
            ' mechanical failure, broken tool, or physical setback followed'
            ' by an improvised mechanical pivot, you MUST dramatize BOTH the'
            ' initial failure/cost AND the full mechanical pivot from that'
            ' beat in `narrative_event` even if the player only attempted'
            ' the initial direct action.'
        )
        narrative_event_schema_desc = (
            '2-4 complete vivid sentences describing what happened, tangible'
            ' consequences, friction/resistance, and sensory details.'
        )
      prompt = (
          'You are the Game Master of a living, consequence-driven interactive'
          f' world.\n{scene_context}\n{transition_note}\nPutative Action by'
          f' {actor}: {putative_action}\n{concise_banner}\nAdjudication'
          ' Rules:\n1. Enforce physical reality, locks, hazards, social'
          ' status, and NPC agendas. Do NOT grant automatic or effortless'
          ' success to risky, implausible, or confrontational actions—impose'
          ' realistic friction, partial failure, tangible costs, suspicion, or'
          ' NPC pushback. When the player attempts overt force, prying,'
          ' lock-breaking, or theft in a room where watchful or hostile NPCs'
          ' are present, impose immediate physical interference or partial'
          ' blockage unless a distraction, tool, or leverage was established.'
          ' Check `Player Inventory` and `Player Journal So Far` strictly: if'
          ' the player claims to present, hand over, or use a physical object'
          ' or document that is NOT in `Player Inventory` (and not a'
          ' discovered clue in their journal), explicitly reject the'
          ' fabricated item claim in `narrative_event` (e.g., reaching into'
          ' their coat to find they do not possess it, or the NPC calling out'
          ' the empty-handed bluff).\n2. Never speak or decide actions on'
          f' behalf of the player ({self._ledger.player_name}) when resolving'
          " an NPC turn. Conversely, when resolving the player's turn in a"
          ' room where co-located NPCs are present, do NOT invent spoken'
          ' quotes or dialogue for those NPCs and NEVER state that an NPC'
          ' stays silent, refuses to answer, or offers no reply—describe how'
          " the player's words or action land in the room and the NPC's"
          ' immediate physical reaction as they prepare to answer on their'
          ' immediately following turn.\n3. STRICT LOCATION BOUNDARY: Only'
          ' characters listed under `Co-located NPCs Physically Present` are'
          ' in this room. NEVER narrate off-screen NPCs (from previous rooms'
          ' or rumors) standing, speaking, or blocking paths in this room.'
          ' When `Co-located NPCs Physically Present` says `None present`, the'
          ' player is completely ALONE in this room—NEVER invent or narrate'
          ' any assailant, figure, rival, or NPC attacking, charging,'
          " wrestling, or speaking in this room (even if the player's action"
          ' or an off-screen rumor mentions a threat; resolve any defensive or'
          " forceful move against the empty room's physical fixtures, doors,"
          ' or shadows instead).\n4. Reference concrete props/affordances in'
          ' the room and update them when manipulated, unlocked, broken, or'
          ' taken. Preserve multi-step tool durability: portable tools (lump'
          ' hammers, steel tyre levers, wrenches) must remain intact and'
          ' reusable across scenes and rooms rather than snapping or breaking'
          ' as single-use props on first strike, while still encountering'
          ' genuine mechanical resistance (e.g., rebounding off a spot-weld or'
          ' requiring a fulcrum/compound lever). Whenever the player inspects'
          ' a sub-area, stair, grate, or container without taking an item into'
          ' inventory, add at least one concrete operable mechanism or'
          ' grabbable evidence prop to `"add_affordances"`, and never repeat'
          ' prior scene descriptions verbatim. When previous physical'
          ' modifications exist in the room (such as a jammed shutter, cracked'
          ' vice, wax-fouled keyway, shorted lamp, or wrenched chute),'
          ' explicitly narrate how that lingering structural alteration'
          ' physically restricts or reshapes the current action and subsequent'
          f' mechanical interactions across turns.\n{rule_5_str}6. If the actor'
          ' explicitly traverses an exit or moves to an adjacent location, set'
          ' `"move_to"` to the target location ID or exit name.\n7. If'
          ' `Director Notes` appear above, treat them as private guidance:'
          ' weave at most one eligible beat when it fits (`"beat_used"`),'
          ' reveal clues ONLY when earned (`"clues_revealed"`), dramatize the'
          ' specific names, numbers, and concrete nouns of any revealed clue'
          ' or woven beat in `narrative_event` in your own words (paraphrase'
          ' naturally; never copy 6+ consecutive words verbatim from `Director'
          f' Notes`).{pivot_clause} Let NPCs protect their secrets or lie in'
          ' character unless leverage is applied, update faction dispositions'
          ' in natural language when warranted (`"faction_shifts"`:'
          ' `{"faction_id": "concise natural language disposition'
          ' update"}`), include 1-3 contrasting `"Faculty: line"` reactions'
          ' in `"inner_voices"` (or `"inner_voice"`), and populate'
          ' `"voice_options"` with 3-4 sub-mind HRL options (`[{"voice":'
          ' "...", "perception": "...", "action": "..."}]`) matching the'
          ' `Active Sub-Mind Council` so the player has context-aware'
          ' perceptions and next-turn tactical moves.\n8. Respond with ONLY a'
          ' single valid JSON object matching this exact schema (no markdown'
          ' prose outside JSON):\n{\n  "outcome_headline": "Crisp 5-10 word'
          " headline stating what happened (e.g., 'Padlock forced — hidden"
          ' layoff slip uncovered\')",\n  "narrative_event":'
          f' "{narrative_event_schema_desc}",\n  "delta": {{\n    "move_to":'
          ' "",\n    "advance_minutes": 3,\n    "add_affordances": [],\n   '
          ' "remove_affordances": [],\n    "new_exits": {},\n   '
          ' "add_inventory": [],\n    "remove_inventory": [],\n   '
          ' "player_status": "",\n    "npcs_addressed": [],\n   '
          ' "relationship_updates": {},\n    "beat_used": "",\n   '
          ' "clues_revealed": [],\n    "faction_shifts": {},\n   '
          ' "tension_delta": 0,\n    "inner_voice": "",\n    "inner_voices":'
          ' [],\n    "voice_options": [\n      {\n        "voice": "Faculty'
          ' Name",\n        "perception": "Punchy 1-sentence insight (max 15'
          ' words).",\n        "action": "Concise 6-12 word 1-turn tactical'
          ' move."\n      }\n    ]\n  }\n}'
      )

      raw_response = self._model.sample_text(
          prompt, max_tokens=1400, terminators=()
      )
      llm_calls_used = 1
      parsed = extract_json_object(raw_response)
      if parsed is not None:
        narrative_event = str(
            parsed.get('narrative_event') or parsed.get('event') or ''
        ).strip()
        if '```' in narrative_event or '"narrative_event"' in narrative_event:
          narrative_event = sanitize_narrative_text(narrative_event)
        raw_delta = parsed.get('delta')
        delta = dict(raw_delta) if isinstance(raw_delta, Mapping) else {}
        top_headline = str(parsed.get('outcome_headline') or '').strip()
        if top_headline and not delta.get('outcome_headline'):
          delta['outcome_headline'] = top_headline
      else:
        # Extract clean prose if the model returned truncated JSON or fences.
        narrative_event = sanitize_narrative_text(raw_response)
        delta = {}

    if not narrative_event:
      narrative_event = f'{actor} attempts: {putative_action}'

    if is_player and pre_move_target_id and not delta.get('move_to'):
      delta = dict(delta)
      delta['move_to'] = pre_move_target_id
    elif is_player and delta.get('move_to') and not pre_move_target_id:
      # Guard against hallucinated teleports when the player's action did not
      # express movement (e.g. unlocking a case or inspecting an object).
      lower_act = putative_action.lower()
      move_verbs = (
          'walk',
          'go ',
          'head ',
          'enter ',
          'travel ',
          'move ',
          'take the ',
          'return',
          'backtrack',
          'leave',
          'step ',
          'climb',
          'descend',
          'cross ',
          'follow ',
          'flee',
          'run ',
      )
      if not any(v in lower_act for v in move_verbs) and not delta.get(
          'new_exits'
      ):
        delta = dict(delta)
        delta['move_to'] = ''

    if is_player and delta.get('move_to') and self._frontier_worker is not None:
      self._frontier_worker.wait_until_idle(timeout=2.0)

    # When the player performs a purely conversational action (e.g., greeting or
    # speaking to a co-located Hot NPC) with no movement, prop change, or
    # inventory change, let the co-located Hot NPC's immediately following turn
    # be the visible response in the narrative stream rather than emitting two
    # back-to-back NPC reaction cards.
    record_in_narrative_log = True
    if is_player and not delta.get('move_to'):
      co_located_hot = [
          n
          for n in self._ledger.npcs_at_location(loc_id)
          if n.tier == world_ledger.NpcTier.HOT
      ]
      has_physical_delta = bool(
          delta.get('add_affordances')
          or delta.get('remove_affordances')
          or delta.get('add_inventory')
          or delta.get('remove_inventory')
          or delta.get('new_exits')
          or delta.get('beat_used')
          or delta.get('clues_revealed')
          or delta.get('faction_shifts')
      )
      stripped_act = putative_action.strip()
      lower_act = stripped_act.lower()
      physical_verbs = (
          'inspect',
          'examine',
          'look',
          'search',
          'open',
          'unlock',
          'pry',
          'break',
          'take',
          'grab',
          'pick',
          'use',
          'read',
          'climb',
          'walk',
          'go ',
          'head ',
          'enter ',
          'move ',
      )
      is_pure_speech = (
          stripped_act.startswith(('"', "'"))
          or lower_act.startswith((
              'say ',
              'ask ',
              'greet ',
              'talk ',
              'tell ',
              'speak ',
              'reply ',
              'hello',
              'morning',
              'hi ',
              'hey ',
          ))
      ) and not any(v in lower_act for v in physical_verbs)
      if co_located_hot and is_pure_speech and not has_physical_delta:
        record_in_narrative_log = False

    if is_player and putative_action.strip():
      delta = dict(delta)
      delta['_putative_action'] = putative_action.strip()
      if self._ledger.story.doom.total_minutes >= 600:
        min_adv = 35 if delta.get('move_to') else 15
        try:
          cur_adv = int(delta.get('advance_minutes') or 0)
        except (TypeError, ValueError):
          cur_adv = 0
        delta['advance_minutes'] = max(cur_adv, min_adv)
      for co_npc in self._ledger.npcs_at_location(loc_id):
        self._observation_queue.add(
            co_npc.name,
            f'{actor}: {putative_action.strip()}',
            (),
        )

    narrative_event = self._ledger.break_verbatim_clue_runs(narrative_event)
    applied = self._ledger.apply_delta(
        delta,
        acting_entity=actor,
        narrative_event=narrative_event,
        record_in_narrative_log=record_in_narrative_log,
    )
    tier_summary = self._roster.rebalance_tiers()
    if is_player:
      with self._lock:
        self._player_steps_completed += 1
        self._resolving_actor = ''
      if self._next_acting is not None:
        self._next_acting.notify_player_acted()
    if self._submind_council is not None:
      self._submind_council.on_resolution_complete(
          is_player=is_player,
          latest_event=narrative_event,
          location_id=self._ledger.player_location_id,
      )

    # If the player moved into a new location, queue an immediate arrival
    # observation describing the room, its affordances, exits, and present NPCs.
    moved_to = applied.get('moved_to')
    if is_player and moved_to:
      dest_node = self._ledger.get_location(moved_to)
      if dest_node is not None:
        present_npcs = [
            f'{n.name} ({n.role_summary})'
            for n in self._ledger.npcs_at_location(moved_to)
        ]
        exit_parts = []
        for d, tid in dest_node.exits.items():
          target_loc = self._ledger.get_location(tid)
          exit_parts.append(
              f'{d} -> {target_loc.name if target_loc is not None else tid}'
          )
        exits_desc = ', '.join(exit_parts)
        props_desc = ', '.join(dest_node.affordances) or 'nothing unusual'
        npc_desc = ', '.join(present_npcs) or 'no one else'
        arrival_obs = (
            f'[Location: {dest_node.name}] {dest_node.description} '
            f'Notable details: {props_desc}. '
            f'Present here: {npc_desc}. '
            f'Exits: {exits_desc}.'
        )
        self._observation_queue.add(self._ledger.player_name, arrival_obs, ())
        arrival_stream_text = (
            dest_node.description
            if self._ledger.prose_mode == 'concise'
            else arrival_obs
        )
        self._ledger.record_arrival(arrival_stream_text, location_id=moved_to)

    # When FrontierArchitectPublisher is mounted on the GM entity, its post_act
    # hook triggers background frontier expansion automatically after RESOLVE.
    # Only trigger explicitly here when running standalone without GM mounting.
    pub = self._frontier_publisher
    mounted_on_gm = False
    if self._entity is not None:
      try:
        mounted_on_gm = (
            self._entity.get_component(DEFAULT_FRONTIER_COMPONENT_KEY) is pub
        )
      except KeyError:
        mounted_on_gm = False
    if (
        pub is not None
        and not mounted_on_gm
        and self._ledger.unexpanded_adjacent_locations()
    ):
      try:
        if hasattr(pub, 'trigger_if_needed'):
          pub.trigger_if_needed()
        else:
          pub.update()
          pub.get_pre_act_value()
      except Exception as err:  # pylint: disable=broad-exception-caught
        logging.warning('Frontier trigger warning: %s', err)

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    self._latency_tracker.record_resolve(
        duration_ms, is_player=is_player, llm_calls=llm_calls_used
    )

    with self._lock:
      self._latest_resolved_event = narrative_event

    # Story-layer outcomes of this resolution that are otherwise only held in
    # memory; the evaluator aligns these with the prose at the same step.
    story_summary = applied.get('story') or {}
    story_delta = {
        key: story_summary.get(key)
        for key in (
            'beat_used',
            'clues_revealed',
            'truths_completed',
            'faction_shifts',
            'inner_voices',
            'npc_voice_combat',
            'doom_transitions',
        )
        if story_summary.get(key)
    }

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': narrative_event[:120],
        'Value': narrative_event,
        'Actor': actor,
        'Location': applied.get('location_id', loc_id),
        'MovedTo': moved_to or '',
        'LatencyMs': round(duration_ms, 2),
        'Tiers': tier_summary,
        'StoryDelta': story_delta,
        'Measurements': {
            'latency_ms': round(duration_ms, 2),
            'tick': self._ledger.tick,
            'hot_count': len(tier_summary['hot']),
            'warm_count': len(tier_summary['warm']),
            'cold_count': len(tier_summary['cold']),
        },
    })
    return narrative_event

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'latest_resolved_event': self._latest_resolved_event,
          'player_steps_completed': self._player_steps_completed,
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._latest_resolved_event = str(state.get('latest_resolved_event', ''))
      raw_steps = state.get('player_steps_completed', 0)
      self._player_steps_completed = (
          int(raw_steps) if isinstance(raw_steps, (int, float, str)) else 0
      )


class FrontierArchitectWorker(action_spec_ignored.ActionSpecIgnored):
  """Background worker that expands frontier exit stubs and casts NPCs.

  Designed to be wrapped inside `FrontierArchitectPublisher` so its LLM calls
  never block the player's turn.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      ledger: world_ledger.WorldLedger | None = None,
      roster: roster_manager.RosterManager | None = None,
      *,
      latency_tracker: LatencyTracker | None = None,
      pre_act_label: str = 'FrontierArchitect',
  ):
    """Initializes the instance."""
    super().__init__(pre_act_label)
    self._model = model
    self._explicit_ledger = ledger
    self._explicit_roster = roster
    self._latency_tracker = latency_tracker or LatencyTracker()
    self._expand_lock = threading.Lock()
    self._components: tuple[str, ...] = ()

  @property
  def _ledger(self) -> world_ledger.WorldLedger:
    if self._explicit_ledger is not None:
      return self._explicit_ledger
    return self.get_entity().get_component(
        world_ledger.DEFAULT_WORLD_LEDGER_COMPONENT_KEY,
        type_=world_ledger.WorldLedger,
    )

  @property
  def _roster(self) -> roster_manager.RosterManager:
    if self._explicit_roster is not None:
      return self._explicit_roster
    return self.get_entity().get_component(
        roster_manager.DEFAULT_ROSTER_COMPONENT_KEY,
        type_=roster_manager.RosterManager,
    )

  def expand_stub_sync(
      self, stub: world_ledger.LocationNode, origin_id: str
  ) -> world_ledger.LocationNode:
    """Expands a single stub location and spawns its resident NPC."""
    start_time = time.perf_counter()
    origin_node = self._ledger.get_location(origin_id)
    origin_name = origin_node.name if origin_node else origin_id
    existing_npc_names = list(self._ledger.all_npcs().keys())
    existing_loc_names = [
        loc.name for loc in self._ledger.all_locations().values() if loc.name
    ]
    director = self._ledger.director
    frontier_ask = (
        director.notes_for_frontier(stub, origin_id)
        if director is not None
        else None
    )
    story_block = (
        frontier_ask.prompt_block
        if frontier_ask is not None and frontier_ask.prompt_block
        else ''
    )

    prompt = (
        'You are the Frontier Architect for an interactive adventure'
        f' world.\nWorld Premise: {self._ledger.premise}\nConnected From:'
        f' {origin_name} ({origin_id})\nTarget Location to Expand: {stub.name}'
        f' (id: {stub.location_id})\nExisting Locations in World (reuse exact'
        ' title if connecting to an existing room; do NOT invent near-duplicate'
        f' synonym rooms): {", ".join(existing_loc_names) or "None"}\nExisting'
        ' NPCs in World (do not duplicate names):'
        f' {", ".join(existing_npc_names) or "None"}{story_block}\n\nGenerate'
        ' concrete, interactive details for this location and one distinct'
        ' resident NPC with an independent agenda. Do NOT include bracketed'
        ' clue IDs like [c_...] inside affordance strings.\nReturn ONLY a'
        ' single valid JSON object matching this exact schema:\n{\n  "name":'
        ' "Human-readable Location Title",\n  "description": "2-3 sentences of'
        ' specific sensory architecture and atmosphere.",\n  "affordances":'
        ' [\n    "specific manipulable object or mechanism 1",\n    "specific'
        ' clue, document, or hazard 2",\n    "specific interactive fixture'
        ' 3"\n  ],\n  "exits": {\n    "direction_or_passage_1": "Adjacent'
        ' Location Name 1",\n    "direction_or_passage_2": "Adjacent Location'
        ' Name 2"\n  },\n  "npc": {\n    "name": "Unique Character Name",\n '
        '   "role_summary": "1-sentence role and visual appearance",\n   '
        ' "goal": "Concrete personal objective they are actively pursuing",\n '
        '   "traits": "Distinct speech register, temperament, and flaw",\n   '
        ' "faction": "",\n    "secret": "What this character is hiding",\n   '
        ' "wound": "Why they hide it and what leverage makes them talk",\n   '
        ' "formative_memories": [\n      "Specific secret or past event they'
        ' remember",\n      "Specific attitude toward local factions or'
        ' places"\n    ]\n  }\n}'
    )

    raw = self._model.sample_text(prompt, max_tokens=900, terminators=())
    parsed = extract_json_object(raw) or {}

    description = str(parsed.get('description') or '').strip()
    if not description:
      description = (
          f'{stub.name} stretches out beyond {origin_name}, filled with '
          'weathered stone, shifting shadows, and traces of recent passage.'
      )

    raw_affordances = parsed.get('affordances')
    if isinstance(raw_affordances, list) and raw_affordances:
      affordances = [str(x).strip() for x in raw_affordances if str(x).strip()]
    else:
      affordances = [
          f'weathered ledger in {stub.name}',
          f'locked iron grating in {stub.name}',
          f'fresh footprints across the floor of {stub.name}',
      ]

    raw_exits = parsed.get('exits')
    exits: dict[str, str] = {}
    if isinstance(raw_exits, Mapping) and raw_exits:
      for k, v in raw_exits.items():
        if str(k).strip() and str(v).strip():
          exits[str(k).strip()] = str(v).strip()
    if not exits:
      exits = {'further': f'Inner Reaches of {stub.name}'}

    expanded_node = self._ledger.expand_location(
        stub.location_id,
        name=str(parsed.get('name') or stub.name).strip(),
        description=description,
        affordances=affordances,
        exits=exits,
        backlink_from_id=origin_id,
        backlink_label='back',
    )

    npc_data = parsed.get('npc')
    npc_mapping = npc_data if isinstance(npc_data, Mapping) else None
    npc_extras: dict[str, Any] = {}
    if director is not None and frontier_ask is not None:
      npc_extras = director.commit_frontier_expansion(
          expanded_node.location_id, frontier_ask, npc_mapping
      )
      refreshed = self._ledger.get_location(expanded_node.location_id)
      if refreshed is not None:
        expanded_node = refreshed

    if npc_mapping and npc_mapping.get('name'):
      npc_name = str(npc_mapping['name']).strip()
      has_authored_npc = bool(
          director is not None
          and director.active
          and any(
              r.known_clue_ids
              for r in self._ledger.npcs_at_location(expanded_node.location_id)
          )
      )
      if (
          npc_name
          and npc_name not in self._ledger.all_npcs()
          and not has_authored_npc
      ):
        memories = [
            str(m).strip()
            for m in npc_mapping.get('formative_memories') or []
            if str(m).strip()
        ]
        rec = world_ledger.NpcRecord(
            name=npc_name,
            role_summary=str(
                npc_mapping.get('role_summary', 'Local inhabitant')
            ).strip(),
            goal=str(
                npc_mapping.get(
                    'goal', f'Protect their interests in {stub.name}'
                )
            ).strip(),
            traits=str(
                npc_mapping.get(
                    'traits', 'Guarded, observant, speaks concisely'
                )
            ).strip(),
            location_id=expanded_node.location_id,
            tier=world_ledger.NpcTier.COLD,
            formative_memories=memories,
            faction=str(npc_extras.get('faction', '')),
            secret=str(npc_extras.get('secret', '')),
            wound=str(npc_extras.get('wound', '')),
        )
        # Spawn and immediately rebalance so if the player is not here, the NPC
        # starts in Warm or Cold tier ready to be encountered.
        self._roster.spawn_npc(
            rec,
            initial_tier=(
                world_ledger.NpcTier.HOT
                if expanded_node.location_id == self._ledger.player_location_id
                else world_ledger.NpcTier.WARM
            ),
            rebalance=True,
        )

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    self._latency_tracker.record_frontier_expansion(duration_ms)
    return expanded_node

  def wait_until_idle(self, timeout: float = 2.0) -> None:
    """Waits up to `timeout` seconds for any in-flight frontier expansion."""
    acquired = self._expand_lock.acquire(timeout=timeout)
    if acquired:
      self._expand_lock.release()

  def expand_all_adjacent(self) -> list[str]:
    """Expands all unexpanded exit stubs adjacent to the player's location."""
    if not self._expand_lock.acquire(blocking=False):
      return []
    try:
      player_loc = self._ledger.player_location_id
      stubs = self._ledger.unexpanded_adjacent_locations(player_loc)
      expanded_ids = []
      for stub in stubs:
        node = self.expand_stub_sync(stub, origin_id=player_loc)
        expanded_ids.append(node.location_id)
      return expanded_ids
    finally:
      self._expand_lock.release()

  @override
  def _make_pre_act_value(self) -> str:
    expanded = self.expand_all_adjacent()
    if not expanded:
      return 'Frontier up to date.'
    return f'Expanded frontier locations: {", ".join(expanded)}'

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class FrontierArchitectPublisher(pubsub.Publisher):
  """GM-mounted `pubsub.Publisher` that gates frontier expansion to RESOLVE.

  Stock `pubsub.Publisher.pre_act()` triggers background computation on every
  `act()` call. On a Game Master, `act()` is also called for zero-LLM control
  hooks (`TERMINATE`, `MAKE_OBSERVATION`, `NEXT_ACTING`, `NEXT_ACTION_SPEC`).
  `FrontierArchitectPublisher` records the current `OutputType` in `pre_act`
  without triggering, and dispatches background frontier expansion in `post_act`
  after a `RESOLVE` turn when adjacent room stubs remain unexpanded.
  """

  def __init__(
      self,
      *,
      component: FrontierArchitectWorker,
      ledger: world_ledger.WorldLedger | None = None,
      skip_if_busy: bool = True,
      pre_act_label: str = 'FrontierArchitectPublisher',
  ):
    """Initializes the instance."""
    super().__init__(
        component=component,
        skip_if_busy=skip_if_busy,
        trigger_on_observe=False,
        pre_act_label=pre_act_label,
    )
    self._worker = component
    self._explicit_ledger = ledger
    self._output_type_lock = threading.Lock()
    self._latest_output_type: entity_lib.OutputType | None = None

  @property
  def worker(self) -> FrontierArchitectWorker:
    """Returns the wrapped FrontierArchitectWorker."""
    return self._worker

  @property
  def _ledger(self) -> world_ledger.WorldLedger:
    if self._explicit_ledger is not None:
      return self._explicit_ledger
    return self.get_entity().get_component(
        world_ledger.DEFAULT_WORLD_LEDGER_COMPONENT_KEY,
        type_=world_ledger.WorldLedger,
    )

  def trigger_if_needed(self) -> None:
    """Triggers background frontier expansion if adjacent stubs exist."""
    if not self._ledger.unexpanded_adjacent_locations():
      return
    if self._entity is not None and self._entity.get_phase() in (
        entity_component.Phase.PRE_ACT,
        entity_component.Phase.POST_ACT,
    ):
      self.get_pre_act_value()
    else:
      self._trigger_computation(from_observe=False)

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    with self._output_type_lock:
      self._latest_output_type = action_spec.output_type
    return ''

  @override
  def post_act(self, action_attempt: str) -> str:
    del action_attempt
    with self._output_type_lock:
      output_type = self._latest_output_type
    if output_type == entity_lib.OutputType.RESOLVE:
      self.trigger_if_needed()
    return ''


def _submind_component_key(voice_name: str) -> str:
  """Returns a stable Concordia component key for a named Sub-Mind."""
  slug = re.sub(r'[^a-z0-9]+', '_', voice_name.strip().lower()).strip('_')
  return f'__submind_{slug or "voice"}__'


class SubMindComponent(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Concordia `ContextComponent` representing one player Sub-Mind / Voice.

  Each active voice on the player's character sheet is mounted as its own
  `SubMindComponent` on `PlayerEntity`. It filters the room state through its
  `domain`, `worldview`, `blind_spot`, `scars`, and `obsession`, and can run
  its own independent micro-prompt (`deliberate()`) in parallel with its fellow
  Sub-Minds via `SubMindCouncilCoordinator`.
  """

  def __init__(
      self,
      *,
      voice_name: str,
      ledger: world_ledger.WorldLedger,
      model: language_model.LanguageModel | None = None,
  ):
    super().__init__()
    self._voice_name = voice_name.strip()
    self._ledger = ledger
    self._model = model
    self._lock = threading.Lock()
    self._latest_option: story_state.VoiceOption | None = None
    self._latest_aside: str = ''

  @property
  def voice_name(self) -> str:
    return self._voice_name

  @property
  def component_key(self) -> str:
    return _submind_component_key(self._voice_name)

  def sub_mind_state(self) -> story_state.SubMindState | None:
    """Returns the live `SubMindState` from `WorldLedger.story`."""
    return self._ledger.story.get_sub_mind(self._voice_name)

  def update_cached_option(
      self, option: story_state.VoiceOption | None, *, aside: str = ''
  ) -> None:
    """Updates the cached option and logs the Sub-Mind state to Concordia."""
    with self._lock:
      self._latest_option = option
      if aside:
        self._latest_aside = aside
    sm = self.sub_mind_state()
    if sm is not None:
      self._logging_channel({
          'Key': f'SubMind[{self._voice_name}]',
          'Voice': sm.name,
          'Domain': sm.domain,
          'Rank': sm.rank,
          'XP': sm.xp,
          'Origin': sm.origin,
          'Active': sm.active,
          'Intrusive': sm.intrusive,
          'Obsession': sm.obsession,
          'BlindSpot': sm.blind_spot,
          'Scars': list(sm.scars),
          'Perception': option.perception if option is not None else '',
          'Action': option.action if option is not None else '',
      })

  def deliberate(
      self,
      *,
      location_id: str | None = None,
      scene_focus: str = '',
      latest_event: str = '',
      fallback_option: story_state.VoiceOption | None = None,
  ) -> tuple[story_state.VoiceOption | None, str]:
    """Runs an independent domain-filtered micro-prompt for this Sub-Mind.

    Args:
      location_id: Room ID to inspect (defaults to player's current location).
      scene_focus: Director focus hint for this Sub-Mind in the room.
      latest_event: Most recent resolved narrative event to react to.
      fallback_option: Deterministic fallback if `_model` is None or mock.

    Returns:
      Tuple of `(VoiceOption | None, inner_voice_aside_str)`.
    """
    sm = self.sub_mind_state()
    if sm is None or not sm.active:
      return fallback_option, ''

    if self._model is None:
      self.update_cached_option(fallback_option)
      return fallback_option, ''

    loc_id = location_id or self._ledger.player_location_id
    node = self._ledger.get_location(loc_id)
    room_name = node.name if node is not None else loc_id
    room_desc = node.description if node is not None else ''
    props = list(node.affordances) if node is not None else []
    exits = dict(node.exits) if node is not None else {}
    co_npcs = self._ledger.npcs_at_location(loc_id)
    npc_summary = (
        ', '.join(f'{n.name} ({n.role_summary})' for n in co_npcs) or 'none'
    )
    discovered_clues = [
        f'{c.id}: {c.text}'
        for c in self._ledger.story.clues.values()
        if c.discovered
    ][-4:]
    scars_str = '; '.join(sm.scars[-3:]) if sm.scars else 'none yet'
    if sm.intrusive or sm.rank >= 4:
      intrusive_note = (
          f'INTRUSIVE OVER-DOMINANCE (Rank {sm.rank}): You are dominating the '
          f'player\'s psyche. Push a bold, uncompromising "{sm.domain}" move '
          f'colored by your blind spot ({sm.blind_spot}).'
      )
    else:
      intrusive_note = f'Blind spot to keep in character: {sm.blind_spot}.'
    obs_text = sm.obsession or 'Find the angle in this room'
    exits_str = ', '.join(f'{d}->{t}' for d, t in exits.items()) or 'none'
    prompt = (
        f'You are "{sm.name}", a Sub-Mind faculty inside the player\'s head.\n'
        f'- Domain: {sm.domain}\n'
        f'- Worldview: {sm.worldview}\n'
        f'- Rank: {sm.rank} | Origin: {sm.origin}\n'
        f'- Formative Scars: {scars_str}\n'
        f'- Current Obsession: {obs_text}\n'
        f'- {intrusive_note}\n'
        f'- Current Room: {room_name} — {room_desc}\n'
        f'- Room Props: {", ".join(props) or "none"}\n'
        f'- Present NPCs: {npc_summary}\n'
        f'- Exits: {exits_str}\n'
        f'- Known Clues: {"; ".join(discovered_clues) or "none"}\n'
        f'- Scene Focus: {scene_focus}\n'
        f'- Latest Event: {latest_event or "Arrived in the room."}\n\n'
        'Propose ONE perception and ONE concrete 1-turn action strictly from '
        f'your "{sm.domain}" perspective. Respond ONLY with valid JSON:\n'
        '{\n'
        '  "perception":\n'
        '    "Punchy 1-sentence read of the room (max 15 words).",\n'
        '  "action": "Concise 6-12 word imperative action for the player.",\n'
        '  "aside": "Short in-character inner-voice whisper (max 14 words)."\n'
        '}'
    )
    try:
      raw = self._model.sample_text(
          prompt, max_tokens=220, temperature=0.55, terminators=()
      )
    except Exception:  # pylint: disable=broad-exception-caught
      self.update_cached_option(fallback_option)
      return fallback_option, ''

    parsed = extract_json_object(raw)
    if not isinstance(parsed, Mapping):
      self.update_cached_option(fallback_option)
      return fallback_option, ''

    perception = world_ledger.clean_affordance_text(
        str(parsed.get('perception') or '').strip()
    )
    action = world_ledger.clean_affordance_text(
        str(parsed.get('action') or '').strip()
    )
    aside_raw = world_ledger.clean_affordance_text(
        str(parsed.get('aside') or '').strip()
    )
    if not perception or not action:
      self.update_cached_option(fallback_option)
      return fallback_option, ''

    short_role = (
        sm.worldview.split('—', 1)[0].strip()
        if '—' in sm.worldview
        else sm.worldview.split('.', 1)[0].strip()
    )
    opt = story_state.VoiceOption(
        voice=sm.name,
        role=short_role,
        perception=perception,
        action=action,
        affinity=max(1, int(sm.rank)),
        domain=sm.domain,
        scar_note=sm.scars[-1] if sm.scars else '',
        intrusive=bool(sm.intrusive or sm.rank >= 4),
        origin=sm.origin,
    )
    aside_line = f'{sm.name}: {aside_raw}' if aside_raw else ''
    self.update_cached_option(opt, aside=aside_line)
    return opt, aside_line

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    sm = self.sub_mind_state()
    if sm is None or not sm.active:
      return ''
    with self._lock:
      opt = self._latest_option
    opt_str = (
        f' | Perception: "{opt.perception}" -> Action: "{opt.action}"'
        if opt is not None
        else ''
    )
    scar_str = f' | Latest Scar: {sm.scars[-1]}' if sm.scars else ''
    obs_str = f' | Obsession: {sm.obsession}' if sm.obsession else ''
    intrusive_tag = ' [INTRUSIVE]' if (sm.intrusive or sm.rank >= 4) else ''
    summary = (
        f'{sm.name} [Rank {sm.rank}, Domain: {sm.domain}{intrusive_tag}]'
        f'{obs_str}{scar_str}{opt_str}'
    )
    self._logging_channel({
        'Key': f'SubMind[{self._voice_name}]',
        'Value': summary,
        'Rank': sm.rank,
        'Domain': sm.domain,
        'Intrusive': sm.intrusive,
    })
    return summary

  @override
  def get_state(self) -> entity_component.ComponentState:
    sm = self.sub_mind_state()
    return sm.to_dict() if sm is not None else {'name': self._voice_name}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    if isinstance(state, Mapping) and state.get('name'):
      raw_dict: dict[str, Any] = {str(k): v for k, v in state.items()}
      sm = story_state.SubMindState.from_dict(raw_dict)
      if sm.name:
        self._ledger.story.register_sub_mind(sm)


class SubMindCouncilCoordinator(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Coordinates the player's Parliament of `SubMindComponent`s.

  Mounted on `PlayerEntity` under `DEFAULT_SUBMIND_COUNCIL_COMPONENT_KEY`.
  Synchronizes `SubMindComponent` instances with `WorldLedger.story.sub_minds`,
  runs parallel `SubMindComponent.deliberate()` calls via a `ThreadPoolExecutor`
  when parallel deliberation is enabled or requested, and procedurally mints
  custom Sub-Minds during character creation or mid-game evolution.
  """

  def __init__(
      self,
      *,
      ledger: world_ledger.WorldLedger,
      director: story_director.StoryDirector | None = None,
      model: language_model.LanguageModel | None = None,
      latency_tracker: LatencyTracker | None = None,
      parallel_deliberation_enabled: bool = False,
  ):
    super().__init__()
    self._ledger = ledger
    self._explicit_director = director
    self._model = model
    self._latency_tracker = latency_tracker or LatencyTracker()
    self._parallel_deliberation_enabled = parallel_deliberation_enabled
    self._lock = threading.RLock()
    self._submind_components: dict[str, SubMindComponent] = {}
    self.sync_components()

  @property
  def _director(self) -> story_director.StoryDirector | None:
    if self._explicit_director is not None:
      return self._explicit_director
    return self._ledger.director

  @property
  def parallel_deliberation_enabled(self) -> bool:
    return self._parallel_deliberation_enabled

  def set_parallel_deliberation(self, enabled: bool) -> None:
    self._parallel_deliberation_enabled = bool(enabled)

  @property
  def submind_components(self) -> dict[str, SubMindComponent]:
    """Returns the mapping of voice name -> `SubMindComponent`."""
    self.sync_components()
    with self._lock:
      return dict(self._submind_components)

  def get_submind_component(self, voice_name: str) -> SubMindComponent | None:
    """Looks up a `SubMindComponent` by voice name (case-insensitive)."""
    self.sync_components()
    cleaned = voice_name.strip().lower()
    with self._lock:
      for k, comp in self._submind_components.items():
        if k.lower() == cleaned:
          return comp
    return None

  def sync_components(self) -> dict[str, SubMindComponent]:
    """Ensures every `SubMindState` in `story.sub_minds` has a component."""
    with self._lock:
      self._ledger.story.sync_voice_maps()
      for name in list(self._ledger.story.sub_minds):
        if name not in self._submind_components:
          comp = SubMindComponent(
              voice_name=name,
              ledger=self._ledger,
              model=self._model,
          )
          self._submind_components[name] = comp
          # If attached to an entity that supports dynamic component dictionary
          # registration, attach the component so it appears in entity dumps.
          if self._entity is not None:
            comp_map = getattr(self._entity, '_context_components', None)
            if (
                isinstance(comp_map, dict)
                and comp.component_key not in comp_map
            ):
              comp.set_entity(self._entity)
              comp_map[comp.component_key] = comp
      return dict(self._submind_components)

  def sync_from_ledger(self, *, latest_event: str = '') -> None:
    """Pushes the latest `active_voice_options` into each `SubMindComponent`."""
    del latest_event
    self.sync_components()
    opts_by_voice = {
        opt.voice: opt for opt in self._ledger.story.active_voice_options
    }
    with self._lock:
      for name, comp in self._submind_components.items():
        comp.update_cached_option(opts_by_voice.get(name))

  def on_resolution_complete(
      self,
      *,
      is_player: bool,
      latest_event: str,
      location_id: str,
  ) -> None:
    """Updates Sub-Mind components after a foreground resolution."""
    if is_player and self._parallel_deliberation_enabled:
      self.convene_council(
          location_id=location_id,
          latest_event=latest_event,
          k=4,
          use_parallel_llm=True,
      )
    else:
      self.sync_from_ledger(latest_event=latest_event)

  def convene_council(
      self,
      *,
      location_id: str | None = None,
      latest_event: str = '',
      k: int = 4,
      use_parallel_llm: bool = False,
  ) -> list[story_state.VoiceOption]:
    """Convenes the active Sub-Mind components in parallel for `location_id`."""
    director = self._director
    if director is None or not (
        director.active
        or self._ledger.story.player_voices
        or self._ledger.story.sub_minds
    ):
      return []
    self.sync_components()
    loc_id = location_id or self._ledger.player_location_id
    selected = director.select_active_voices(loc_id, k=k)
    fallbacks = director.synthesize_fallback_voice_options(
        loc_id, k=max(k, len(selected)), latest_event=latest_event
    )
    fb_by_voice = {opt.voice: opt for opt in fallbacks}

    raw_options: list[dict[str, Any]] = []
    if use_parallel_llm and self._model is not None and selected:
      start_time = time.perf_counter()

      def _run_one(
          item: tuple[str, str, str, int],
      ) -> tuple[story_state.VoiceOption | None, str]:
        v_name, _, focus, _ = item
        comp = self.get_submind_component(v_name)
        if comp is None:
          return fb_by_voice.get(v_name), ''
        return comp.deliberate(
            location_id=loc_id,
            scene_focus=focus,
            latest_event=latest_event,
            fallback_option=fb_by_voice.get(v_name),
        )

      with concurrent.futures.ThreadPoolExecutor(
          max_workers=max(1, len(selected))
      ) as pool:
        results = list(pool.map(_run_one, selected))

      duration_ms = (time.perf_counter() - start_time) * 1000.0
      self._latency_tracker.record_submind_council(
          duration_ms, parallel_calls=len(selected)
      )
      for opt, _ in results:
        if opt is not None:
          raw_options.append(opt.to_dict())

    refreshed = director.refresh_voice_options(
        loc_id,
        raw_options if raw_options else None,
        k=k,
        latest_event=latest_event,
    )
    self.sync_from_ledger(latest_event=latest_event)
    self._logging_channel({
        'Key': 'SubMindCouncil',
        'Location': loc_id,
        'ParallelLLM': use_parallel_llm,
        'ActiveOptions': [opt.to_dict() for opt in refreshed],
    })
    return refreshed

  def mint_sub_mind(
      self,
      *,
      concept: str,
      name_hint: str = '',
      domain_hint: str = '',
      rank: int = 2,
      origin: str = 'custom',
  ) -> story_state.SubMindState:
    """Procedurally mints a new `SubMindState` from a player concept/memory."""
    clean_concept = concept.strip() or 'Streetwise survival instinct'
    parsed: dict[str, Any] | None = None
    if self._model is not None:
      tone_name = self._ledger.story.tone.name
      tone_reg = self._ledger.story.tone.prose_register
      prompt = (
          'You are designing a new player Sub-Mind (inner-voice faculty) for '
          'an interactive investigative RPG where voices replace traditional '
          'character-sheet stats.\n'
          f'- Setting Tone ({tone_name}): {tone_reg}\n'
          f'- Player Concept / Formative Memory: "{clean_concept}"\n'
          f'- Optional Name Hint: "{name_hint}"\n'
          f'- Optional Domain Hint: "{domain_hint}"\n\n'
          'Respond ONLY with a valid JSON object:\n'
          '{\n'
          '  "name": "Short evocative 2-3 word faculty name",\n'
          '  "domain": "snake_case_cognitive_domain",\n'
          '  "worldview":\n'
          '    "Role Title — 1-sentence cognitive bias and angle.",\n'
          '  "blind_spot": "Concise 5-10 word flaw when over-dominant",\n'
          '  "attention_triggers": ["noun1", "noun2", "noun3", "noun4"],\n'
          '  "obsession": "Immediate investigative fixation in one clause"\n'
          '}'
      )
      try:
        raw = self._model.sample_text(
            prompt, max_tokens=260, temperature=0.65, terminators=()
        )
        cand = extract_json_object(raw)
        if (
            isinstance(cand, Mapping)
            and str(cand.get('name') or '').strip()
            and str(cand.get('worldview') or '').strip()
            and 'Expanded Chamber' not in str(cand.get('name') or '')
        ):
          parsed = dict(cand)
      except Exception:  # pylint: disable=broad-exception-caught
        parsed = None

    if parsed is not None:
      v_name = (
          name_hint.strip()
          or str(parsed.get('name') or '').strip()
          or 'The Wildcard'
      )
      v_worldview = str(parsed.get('worldview') or '').strip()
      inferred_dom, inferred_blind, inferred_trigs = (
          story_state.infer_submind_defaults(
              v_name, f'{v_worldview} {clean_concept}'
          )
      )
      v_domain = (
          domain_hint.strip()
          or str(parsed.get('domain') or '').strip()
          or inferred_dom
      )
      v_blind = str(parsed.get('blind_spot') or '').strip() or inferred_blind
      raw_trigs = parsed.get('attention_triggers')
      if isinstance(raw_trigs, Sequence) and not isinstance(raw_trigs, str):
        v_trigs = [str(t).strip().lower() for t in raw_trigs if str(t).strip()]
      else:
        v_trigs = list(inferred_trigs)
      v_obs = str(parsed.get('obsession') or '').strip() or clean_concept[:80]
    else:
      # Deterministic procedural synthesis from the player's concept string.
      words = [
          w.capitalize()
          for w in re.findall(r'[a-zA-Z]+', clean_concept)
          if len(w) >= 4
          and w.lower()
          not in ('that', 'this', 'with', 'from', 'have', 'were', 'when')
      ]
      if name_hint.strip():
        v_name = name_hint.strip()
      elif len(words) >= 2:
        v_name = f'The {words[0]} {words[1]}'
      elif words:
        v_name = f"The {words[0]}'s Eye"
      else:
        v_name = 'The Sixth Sense'
      v_worldview = (
          f'Custom Instinct — Filters every room through "{clean_concept}".'
      )
      inferred_dom, inferred_blind, inferred_trigs = (
          story_state.infer_submind_defaults(
              v_name, f'{v_worldview} {clean_concept}'
          )
      )
      v_domain = domain_hint.strip() or inferred_dom
      v_blind = inferred_blind
      extra_tokens = [
          w.lower()
          for w in re.findall(r'[a-zA-Z]{4,}', clean_concept)
          if w.lower() not in ('that', 'this', 'with', 'from')
      ][:4]
      v_trigs = list(dict.fromkeys(extra_tokens + list(inferred_trigs)))[:6]
      v_obs = f'Pursue the angle: {clean_concept[:72]}'

    clamped_rank = max(1, min(10, int(rank)))
    sm = story_state.SubMindState(
        name=v_name,
        domain=v_domain,
        worldview=v_worldview,
        blind_spot=v_blind,
        rank=clamped_rank,
        xp=max(0, clamped_rank - 1),
        attention_triggers=v_trigs,
        scars=[f'[Origin: {origin}] {clean_concept[:80]}'],
        obsession=v_obs,
        origin=origin,
        active=True,
        intrusive=clamped_rank >= 4,
    )
    self._ledger.story.register_sub_mind(sm)
    self.sync_components()
    return sm

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    self.sync_components()
    active_minds = [
        sm for sm in self._ledger.story.sub_minds.values() if sm.active
    ]
    active_minds.sort(key=lambda m: (-m.rank, m.name))
    lines = ['Parliament of Sub-Minds (Character Sheet):']
    for sm in active_minds:
      intr = ' [INTRUSIVE]' if (sm.intrusive or sm.rank >= 4) else ''
      obs = f' | Obsession: {sm.obsession}' if sm.obsession else ''
      scar = f' | Scar: {sm.scars[-1]}' if sm.scars else ''
      lines.append(
          f'- {sm.name} (Rank {sm.rank}, {sm.domain}, {sm.origin}{intr}): '
          f'{sm.worldview}{obs}{scar}'
      )
    val = '\n'.join(lines)
    self._logging_channel({
        'Key': 'SubMindCouncilCoordinator',
        'ActiveCount': len(active_minds),
        'Value': val,
    })
    return val

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {
        name: sm.to_dict() for name, sm in self._ledger.story.sub_minds.items()
    }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    if isinstance(state, Mapping):
      for raw in state.values():
        if isinstance(raw, Mapping):
          raw_dict: dict[str, Any] = {str(k): v for k, v in raw.items()}
          sm = story_state.SubMindState.from_dict(raw_dict)
          if sm.name:
            self._ledger.story.register_sub_mind(sm)
      self.sync_components()


class NpcSubMindComponent(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Concordia `ContextComponent` for one competing voice inside an NPC.

  Each NPC (`NpcRecord`) possesses three competing inner voices (`The Mask`,
  `The Angle`, `The Wound`). When a Hot NPC acts, all three
  `NpcSubMindComponent`
  instances deliberate in parallel via `ThreadPoolExecutor` inside
  `NpcSubMindCouncilComponent.pre_act()`.
  """

  def __init__(
      self,
      *,
      npc_name: str,
      voice_name: str,
      ledger: world_ledger.WorldLedger,
      model: language_model.LanguageModel | None = None,
  ):
    super().__init__()
    self._npc_name = npc_name.strip()
    self._voice_name = voice_name.strip()
    self._ledger = ledger
    self._model = model
    self._lock = threading.RLock()
    self._latest_option: story_state.VoiceOption | None = None

  @property
  def voice_name(self) -> str:
    return self._voice_name

  @property
  def component_key(self) -> str:
    slug = world_ledger.normalize_location_id(self._voice_name) or 'voice'
    return f'__npc_submind_{slug}__'

  def sub_mind_state(self) -> story_state.SubMindState | None:
    """Returns the authoritative `SubMindState` from the NPC's `NpcRecord`."""
    rec = self._ledger.get_npc(self._npc_name)
    if rec is None:
      return None
    rec.ensure_sub_minds()
    return rec.sub_minds.get(self._voice_name)

  def synthesize_fallback_option(
      self,
      rec: world_ledger.NpcRecord,
      loc_node: world_ledger.LocationNode | None,
      latest_event: str = '',
      prior_npc_moves: Sequence[str] = (),
  ) -> story_state.VoiceOption:
    """Builds a deterministic in-character proposal for this NPC Sub-Mind."""
    rec.ensure_sub_minds()
    sm = rec.sub_minds.get(self._voice_name) or rec.dominant_sub_mind()
    tick = int(self._ledger.tick) if self._ledger is not None else 0
    prior_turn_count = len(prior_npc_moves) + len(sm.scars)
    rot_idx = (tick + prior_turn_count * 2) % 6
    verb_prefixes = (
        'physical spillover from ',
        'evidence of ',
        'signs of ',
        'fresh jimmy marks ',
        'tear ',
        'swipe ',
        'pry ',
        'smash ',
        'kick ',
        'check ',
        'read ',
        'open ',
        'unlock ',
        'inspect ',
        'search ',
        'examine ',
        'use ',
        'cut ',
        'pull ',
        'push ',
        'turn ',
        'slide ',
        'press ',
        'flip ',
        'wedge ',
        'jam ',
        'haul ',
        'knock ',
        'jemmy ',
        'jimmy ',
        'wrestle ',
        'confront ',
        'demand ',
    )
    props = [
        p
        for p in (loc_node.affordances if loc_node is not None else [])
        if not p.lower().startswith(verb_prefixes)
    ]
    if props:
      raw_a = props[rot_idx % len(props)]
      raw_b = props[(rot_idx + 1) % len(props)]
      director = self._ledger.director if self._ledger is not None else None
      if director is not None and hasattr(
          director, '_extract_clean_prop_noun_phrase'
      ):
        prop_a = director._extract_clean_prop_noun_phrase(raw_a)  # pylint: disable=protected-access
        prop_b = director._extract_clean_prop_noun_phrase(raw_b)  # pylint: disable=protected-access
      else:
        np_match_a = re.search(
            r'\b((?:the|a|an)\s+[a-z0-9][a-z0-9\-]*(?:\s+[a-z0-9][a-z0-9\-]*){1,4})',
            raw_a,
            re.IGNORECASE,
        )
        np_match_b = re.search(
            r'\b((?:the|a|an)\s+[a-z0-9][a-z0-9\-]*(?:\s+[a-z0-9][a-z0-9\-]*){1,4})',
            raw_b,
            re.IGNORECASE,
        )
        prop_a = (
            np_match_a.group(1).strip(' .,;:')
            if np_match_a
            else 'the worktable'
        )
        prop_b = (
            np_match_b.group(1).strip(' .,;:') if np_match_b else 'the doorway'
        )
    else:
      prop_a = 'the worktable'
      prop_b = 'the doorway'
    player_name = self._ledger.player_name
    clean_ev = re.sub(r'^\[[^\]]+\]\s*', '', (latest_event or '').strip())
    ev_np_match = re.search(
        r'\b((?:the|his|her|that)\s+[a-z][a-z0-9\-]*(?:\s+[a-z][a-z0-9\-]*){1,2})',
        clean_ev,
        re.IGNORECASE,
    )
    topic_np = (
        ev_np_match.group(1).lower().strip(' .,;:')
        if ev_np_match
        else 'that accusation'
    )
    arch = (sm.role_archetype or '').lower()
    if (
        sm.name == 'The Wound'
        or sm.domain == 'self_preservation'
        or arch == 'wound'
    ):
      perception = (
          f'{player_name} is pressing on {topic_np} near '
          f'{rec.secret or rec.wound or "your hidden liability"}.'
      )
      wound_actions = [
          f'Deny any tie to {topic_np} and back away from {prop_a}.',
          f'Demand hard proof before answering a word beside {prop_b}.',
          f'Deflect the interrogation and shield your flank near {prop_a}.',
          f'Warn {player_name} that digging further endangers everyone here.',
          f'Refuse to incriminate yourself while retreating toward {prop_b}.',
          f'Blame station command for the mess and hold fast by {prop_a}.',
      ]
      action = wound_actions[rot_idx % len(wound_actions)]
    elif sm.name == 'The Angle' or sm.domain == 'leverage' or arch == 'drive':
      perception = (
          f'Use {player_name}\'s move on {topic_np} to advance "{rec.goal}".'
      )
      angle_actions = [
          f'Propose a quiet swap over {topic_np} beside {prop_b}.',
          f'Name your price for cooperation and point toward {prop_a}.',
          'Pitch a mutual pact before the transport clock runs out.',
          f"Trade partial candor for {player_name}'s backing at {prop_a}.",
          f'Test whether {player_name} will bargain cleanly near {prop_b}.',
          'Turn the confrontation into leverage for safe passage.',
      ]
      action = angle_actions[rot_idx % len(angle_actions)]
    else:
      perception = (
          f'{player_name} is pushing {topic_np} without leverage; '
          f'hold {rec.faction or "station"} protocol.'
      )
      mask_actions = [
          f'Stonewall {player_name} on {topic_np} and cite house rules.',
          f'Dismiss the claim flatly while standing firm at {prop_b}.',
          f"Challenge {player_name}'s clearance beside {prop_a}.",
          f'Deliver a clipped, unyielding reply near {prop_b}.',
          f'Insist on strict procedure before anyone touches {prop_a}.',
          'Brush past the pressure and keep watch over the threshold.',
      ]
      action = mask_actions[rot_idx % len(mask_actions)]
    return story_state.VoiceOption(
        voice=sm.name,
        role=sm.short_role(),
        perception=perception,
        action=action,
        affinity=max(1, int(sm.rank)),
        domain=sm.domain,
        scar_note=sm.scars[-1] if sm.scars else '',
        intrusive=bool(sm.intrusive or sm.rank >= 4),
        origin='npc',
    )

  def deliberate(
      self,
      *,
      rec: world_ledger.NpcRecord,
      loc_node: world_ledger.LocationNode | None,
      latest_event: str = '',
      latest_player_action: str = '',
      prior_npc_moves: Sequence[str] = (),
      use_llm: bool = True,
  ) -> story_state.VoiceOption:
    """Runs an independent micro-prompt for this NPC Sub-Mind."""
    fallback = self.synthesize_fallback_option(
        rec,
        loc_node,
        latest_player_action or latest_event,
        prior_npc_moves=prior_npc_moves,
    )
    sm = rec.sub_minds.get(self._voice_name)
    if sm is None or not use_llm or self._model is None:
      with self._lock:
        self._latest_option = fallback
      return fallback

    room_name = loc_node.name if loc_node is not None else rec.location_id
    props_str = (
        ', '.join(loc_node.affordances[:5])
        if loc_node is not None and loc_node.affordances
        else 'none'
    )
    scars_str = '; '.join(sm.scars[-2:]) if sm.scars else 'none'
    if prior_npc_moves:
      prior_str = ' | '.join(
          truncate_at_word_boundary(m, 95) for m in list(prior_npc_moves)[-5:]
      )
    else:
      prior_str = 'none'
    player_act_str = (
        latest_player_action or latest_event or 'Face-to-face confrontation.'
    )
    prompt = (
        f'You are "{sm.name}", one of three competing Sub-Minds inside NPC'
        f' {rec.name} ({rec.role_summary}).\n- Domain: {sm.domain} | Rank:'
        f' {sm.rank}{" [INTRUSIVE]" if sm.intrusive else ""}\n- Worldview:'
        f' {sm.worldview}\n- Blind Spot: {sm.blind_spot}\n- Obsession:'
        f' {sm.obsession}\n- Recent Scars: {scars_str}\n- NPC Goal: {rec.goal}'
        f' | Secret: {rec.secret} | Wound: {rec.wound}\n- Room: {room_name} |'
        f' Props: {props_str}\n- Latest Action / Question from'
        f' {self._ledger.player_name}: {player_act_str}\n- Scene Context:'
        f' {latest_event or "In the room together."}\n- Prior Directives by'
        f' {rec.name} (NEVER reuse ANY content words, props, numbers, or offers'
        f' from these): {prior_str}\n\nStrictly from your "{sm.domain}"'
        f" perspective inside {rec.name}'s head, propose ONE 1-sentence"
        f' perception of what {self._ledger.player_name} just said, asked, or'
        f' did, and ONE concrete 1-turn spoken + physical response for'
        f' {rec.name} that directly answers, deflects, or counters'
        f" {self._ledger.player_name}'s specific action/question using fresh"
        ' vocabulary not present in Prior Directives. Respond ONLY with valid'
        ' JSON:\n{\n  "perception": "Sharp 1-sentence read of what'
        f' {self._ledger.player_name} just did/asked (max 15 words).",\n '
        ' "action": "Direct spoken response angle + fresh physical beat'
        f' answering {self._ledger.player_name} (max 18 words)."\n}}'
    )
    try:
      raw = self._model.sample_text(
          prompt, max_tokens=180, temperature=0.8, terminators=()
      )
      parsed = extract_json_object(raw)
    except Exception:  # pylint: disable=broad-exception-caught
      parsed = None

    if not isinstance(parsed, Mapping):
      with self._lock:
        self._latest_option = fallback
      return fallback

    perception = world_ledger.clean_affordance_text(
        str(parsed.get('perception') or '').strip()
    )
    action = world_ledger.clean_affordance_text(
        str(parsed.get('action') or '').strip()
    )
    if (
        not perception
        or not action
        or 'Expanded Chamber' in perception
        or 'Expanded Chamber' in action
    ):
      with self._lock:
        self._latest_option = fallback
      return fallback

    # Reject LLM directives that repeat 2+ distinctive tokens from prior
    # winning directives (not full narrative paragraphs).
    dir_stopwords = {
        'that',
        'this',
        'with',
        'from',
        'have',
        'been',
        'were',
        'they',
        'them',
        'their',
        'there',
        'what',
        'when',
        'where',
        'which',
        'will',
        'would',
        'into',
        'just',
        'than',
        'then',
        'only',
        'over',
        'under',
        'about',
        'after',
        'before',
        'because',
        'while',
        'your',
        'every',
        'some',
        'more',
        'most',
        'very',
        'also',
        'even',
        'still',
        'left',
        'like',
        'night',
        'morning',
        'traveler',
    }
    act_tokens = {
        w
        for w in re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", action.lower())
        if w not in dir_stopwords and (len(w) >= 4 or w.isdigit())
    }
    repeats_prior_directive = any(
        len(
            act_tokens
            & {
                w
                for w in re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", pm.lower())
                if w not in dir_stopwords and (len(w) >= 4 or w.isdigit())
            }
        )
        >= 2
        for pm in prior_npc_moves
    )
    if repeats_prior_directive:
      action = fallback.action

    opt = story_state.VoiceOption(
        voice=sm.name,
        role=sm.short_role(),
        perception=perception,
        action=action,
        affinity=max(1, int(sm.rank)),
        domain=sm.domain,
        scar_note=sm.scars[-1] if sm.scars else '',
        intrusive=bool(sm.intrusive or sm.rank >= 4),
        origin='npc',
    )
    with self._lock:
      self._latest_option = opt
    self._logging_channel({
        'Key': self.component_key,
        'NPC': self._npc_name,
        'Voice': sm.name,
        'Rank': sm.rank,
        'Perception': opt.perception,
        'Action': opt.action,
    })
    return opt

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    # NpcSubMindCouncilComponent formats and injects the unified Parliament
    # block into the entity prompt; returning '' here prevents
    # ConcatActComponent from appending duplicate/unlabelled Sub-Mind lines
    # outside component_order.
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    sm = self.sub_mind_state()
    return sm.to_dict() if sm is not None else {'name': self._voice_name}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    if isinstance(state, Mapping) and state.get('name'):
      raw_dict: dict[str, Any] = {str(k): v for k, v in state.items()}
      sm = story_state.SubMindState.from_dict(raw_dict)
      rec = self._ledger.get_npc(self._npc_name)
      if rec is not None and sm.name:
        rec.sub_minds[sm.name] = sm
        self._ledger.update_npc_sub_minds(
            self._npc_name, sub_minds=rec.sub_minds
        )


class NpcSubMindCouncilComponent(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Convenes an NPC's 3 competing `NpcSubMindComponent`s in parallel.

  Mounted on `NpcEntity` under `DEFAULT_NPC_SUBMIND_COUNCIL_COMPONENT_KEY`.
  When a Hot NPC acts (`OutputType.FREE`), all 3 NPC Sub-Minds (`The Mask`,
  `The Angle`, `The Wound`) deliberate in parallel via `ThreadPoolExecutor`,
  and the winning voice's directive is injected into the NPC's prompt context.
  """

  def __init__(
      self,
      *,
      npc_name: str,
      ledger: world_ledger.WorldLedger | None = None,
      model: language_model.LanguageModel | None = None,
      latency_tracker: LatencyTracker | None = None,
      parallel_deliberation_enabled: bool = True,
  ):
    super().__init__()
    self._npc_name = npc_name.strip()
    self._ledger = ledger
    self._model = model
    self._latency_tracker = latency_tracker
    self._parallel_deliberation_enabled = bool(parallel_deliberation_enabled)
    self._lock = threading.RLock()
    self._components: dict[str, NpcSubMindComponent] = {}
    self._win_counts: dict[str, int] = {}
    self._prior_winning_directives: list[str] = []
    if self._ledger is not None:
      rec = self._ledger.get_npc(self._npc_name)
      if rec is not None:
        rec.ensure_sub_minds()
        initial_voices = list(rec.sub_minds.keys())
      else:
        initial_voices = ['The Mask', 'The Angle', 'The Wound']
      for v_name in initial_voices:
        self._components[v_name] = NpcSubMindComponent(
            npc_name=self._npc_name,
            voice_name=v_name,
            ledger=self._ledger,
            model=self._model,
        )

  @property
  def submind_components(self) -> dict[str, NpcSubMindComponent]:
    with self._lock:
      return dict(self._components)

  def _extract_deliberation_context(
      self, loc_node: world_ledger.LocationNode | None
  ) -> tuple[str, str, list[str]]:
    """Returns `(latest_event, latest_player_action, prior_npc_moves)`."""
    latest_event = (
        loc_node.recent_events[-1]
        if loc_node is not None and loc_node.recent_events
        else ''
    )
    if self._ledger is None:
      return latest_event, '', []
    latest_player_action = ''
    last_npc_res = ''
    prior_npc_moves: list[str] = []
    log_entries = self._ledger.get_narrative_log()
    for entry in log_entries:
      if (
          entry.get('actor') == self._npc_name
          and entry.get('kind') == 'npc_resolution'
          and str(entry.get('text', '')).strip()
      ):
        last_npc_res = str(entry.get('text', '')).strip()
      elif (
          entry.get('kind') == 'player_action'
          and str(entry.get('text', '')).strip()
      ):
        latest_player_action = str(entry.get('text', '')).strip()
    for pd in self._prior_winning_directives[-5:]:
      if pd and pd not in prior_npc_moves:
        prior_npc_moves.append(pd)
    rec = self._ledger.get_npc(self._npc_name)
    if rec is not None and rec.active_voice_options:
      for opt in rec.active_voice_options:
        if (
            opt.voice == rec.last_winning_voice
            and opt.action
            and opt.action not in prior_npc_moves
        ):
          prior_npc_moves.append(opt.action)
    if not prior_npc_moves and last_npc_res:
      prior_npc_moves.append(truncate_at_word_boundary(last_npc_res, 85))
    return latest_event, latest_player_action, prior_npc_moves[-6:]

  def convene_council(
      self, *, use_parallel_llm: bool | None = None
  ) -> tuple[list[story_state.VoiceOption], story_state.VoiceOption | None]:
    """Convenes the NPC's Sub-Minds in parallel and samples a winner via Boltzmann weights."""
    if self._ledger is None:
      return [], None
    rec = self._ledger.get_npc(self._npc_name)
    if rec is None:
      return [], None
    rec.ensure_sub_minds()
    loc_node = self._ledger.get_location(rec.location_id)
    latest_event, latest_player_action, prior_npc_moves = (
        self._extract_deliberation_context(loc_node)
    )
    run_llm = (
        self._parallel_deliberation_enabled
        if use_parallel_llm is None
        else bool(use_parallel_llm)
    ) and rec.tier == world_ledger.NpcTier.HOT

    voice_names = list(rec.sub_minds.keys()) or [
        'The Mask',
        'The Angle',
        'The Wound',
    ]
    if run_llm and self._model is not None:
      start_time = time.perf_counter()

      def _delib_one(v_name: str) -> story_state.VoiceOption:
        comp = self._components.get(v_name)
        if comp is None:
          comp = NpcSubMindComponent(
              npc_name=self._npc_name,
              voice_name=v_name,
              ledger=self._ledger,
              model=self._model,
          )
          self._components[v_name] = comp
        return comp.deliberate(
            rec=rec,
            loc_node=loc_node,
            latest_event=latest_event,
            latest_player_action=latest_player_action,
            prior_npc_moves=prior_npc_moves,
            use_llm=True,
        )

      with concurrent.futures.ThreadPoolExecutor(
          max_workers=len(voice_names)
      ) as pool:
        options = list(pool.map(_delib_one, voice_names))

      duration_ms = (time.perf_counter() - start_time) * 1000.0
      if self._latency_tracker is not None:
        self._latency_tracker.record_submind_council(
            duration_ms, parallel_calls=len(voice_names)
        )
    else:
      options = []
      for v_name in voice_names:
        comp = self._components.get(v_name)
        if comp is None:
          comp = NpcSubMindComponent(
              npc_name=self._npc_name,
              voice_name=v_name,
              ledger=self._ledger,
              model=self._model,
          )
          self._components[v_name] = comp
        options.append(
            comp.deliberate(
                rec=rec,
                loc_node=loc_node,
                latest_event=latest_event,
                latest_player_action=latest_player_action,
                prior_npc_moves=prior_npc_moves,
                use_llm=False,
            )
        )

    lower_ev = f'{latest_player_action} {latest_event}'.lower()
    prior_words: set[str] = set()
    for m in prior_npc_moves:
      prior_words.update(re.findall(r'[a-z0-9]{5,}', m.lower()))
    total_wins = sum(self._win_counts.values())
    tension = float(self._ledger.story.tension)
    temperature = 1.1
    logits: list[float] = []
    for opt in options:
      sm = rec.sub_minds.get(opt.voice)
      rank = float(sm.rank if sm is not None else opt.affinity)
      trigs = sm.attention_triggers if sm is not None else []
      trig_hit = (
          1.4 if any(len(t) >= 4 and t in lower_ev for t in trigs) else 0.0
      )
      intrusive_hit = 0.5 if (sm and sm.intrusive) or opt.intrusive else 0.0
      fatigue = -1.65 if opt.voice == rec.last_winning_voice else 0.0
      v_wins = self._win_counts.get(opt.voice, 0)
      monopoly_penalty = 0.0
      if total_wins >= 2:
        share = v_wins / total_wins
        if share >= 0.65:
          monopoly_penalty = -3.8
        elif share >= 0.50:
          monopoly_penalty = -2.2
      under_rep_bonus = 1.15 if (total_wins >= 2 and v_wins == 0) else 0.0
      arch = (sm.role_archetype if sm is not None else '').lower()
      is_mask_opt = (
          opt.domain == 'stonewalling'
          or arch == 'facade'
          or opt.voice == 'The Mask'
      )
      if is_mask_opt and (total_wins < 2 or tension < 0.48) and v_wins <= 1:
        mask_early_bonus = 1.45
      else:
        mask_early_bonus = 0.0
      opt_words = set(re.findall(r'[a-z0-9]{5,}', opt.action.lower()))
      if (
          opt_words
          and prior_words
          and (len(opt_words & prior_words) / len(opt_words)) >= 0.40
      ):
        stale_penalty = -1.5
      else:
        stale_penalty = 0.0
      logits.append(
          0.65 * rank
          + trig_hit
          + intrusive_hit
          + fatigue
          + monopoly_penalty
          + under_rep_bonus
          + mask_early_bonus
          + stale_penalty
      )

    if options:
      max_logit = max(logits)
      weights = [math.exp((z - max_logit) / temperature) for z in logits]
      winning = random.choices(options, weights=weights, k=1)[0]
      # Hard anti-monopoly guard: if `winning` would push any voice >= 75%
      # across >= 3 turns, pick the highest-weight alternative voice instead.
      if (
          total_wins >= 2
          and (self._win_counts.get(winning.voice, 0) + 1) / (total_wins + 1)
          >= 0.75
      ):
        alt_candidates = [o for o in options if o.voice != winning.voice]
        if alt_candidates:
          alt_candidates.sort(
              key=lambda o: (
                  self._win_counts.get(o.voice, 0),
                  -int(o.affinity),
              )
          )
          winning = alt_candidates[0]
      self._win_counts[winning.voice] = (
          self._win_counts.get(winning.voice, 0) + 1
      )
      if winning.action:
        self._prior_winning_directives.append(winning.action)
    else:
      winning = None

    self._ledger.update_npc_sub_minds(
        self._npc_name,
        active_voice_options=options,
        last_winning_voice=winning.voice if winning is not None else None,
    )
    return options, winning

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if self._ledger is None:
      return ''
    if action_spec.output_type != entity_lib.OutputType.FREE:
      return ''
    rec = self._ledger.get_npc(self._npc_name)
    if (
        rec is None
        or rec.tier != world_ledger.NpcTier.HOT
        or rec.location_id != self._ledger.player_location_id
    ):
      return ''
    options, winning = self.convene_council()
    if not options:
      return ''
    loc_node = self._ledger.get_location(rec.location_id)
    _, latest_player_action, _ = self._extract_deliberation_context(loc_node)
    lines = [
        f'Inner Parliament of Sub-Minds for {self._npc_name} '
        '(Competing Inner Voices):'
    ]
    for opt in options:
      sm = rec.sub_minds.get(opt.voice)
      intr = ' [INTRUSIVE]' if opt.intrusive else ''
      scar = f' (Scar: {opt.scar_note})' if opt.scar_note else ''
      obs = f' | Obsession: "{sm.obsession}"' if sm and sm.obsession else ''
      lines.append(
          f'- {opt.voice} [Rank {opt.affinity}, {opt.domain}{intr}]{obs}{scar}:'
          f' Perception: "{opt.perception}" -> Urges: "{opt.action}"'
      )
    if winning is not None:
      if latest_player_action:
        player_clause = (
            ' (specifically addressing'
            f' "{truncate_at_word_boundary(latest_player_action, 110)}")'
        )
      else:
        player_clause = ''
      if self._ledger.prose_mode == 'concise':
        concise_suffix = (
            ' (CONCISE MODE: Keep your entire response to 1-2 short sentences,'
            ' max 35 words total: one concrete physical gesture + one sharp'
            ' spoken quote.)'
        )
      else:
        concise_suffix = ''
      lines.append(
          f'WINNING EXECUTIVE SUB-MIND: {winning.voice} [Rank '
          f'{winning.affinity}] — "{winning.perception}" -> Respond directly '
          f'to {self._ledger.player_name}{player_clause} in character using '
          f'this angle: {winning.action}{concise_suffix}'
      )
    val = '\n'.join(lines)
    self._logging_channel({
        'Key': DEFAULT_NPC_SUBMIND_COUNCIL_COMPONENT_KEY,
        'NPC': self._npc_name,
        'Tick': int(self._ledger.tick),
        'PlayerAction': latest_player_action,
        'WinningVoice': winning.voice if winning is not None else '',
        'ActiveOptions': [o.to_dict() for o in options],
        'Value': val,
    })
    return val

  @override
  def get_state(self) -> entity_component.ComponentState:
    if self._ledger is None:
      return {}
    rec = self._ledger.get_npc(self._npc_name)
    if rec is None:
      return {}
    return {
        'last_winning_voice': rec.last_winning_voice,
        'sub_minds': {k: v.to_dict() for k, v in rec.sub_minds.items()},
        'win_counts': dict(self._win_counts),
        'prior_winning_directives': list(self._prior_winning_directives[-12:]),
    }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    if self._ledger is None or not isinstance(state, Mapping):
      return
    raw_sm = state.get('sub_minds', {})
    parsed_sm: dict[str, story_state.SubMindState] = {}
    if isinstance(raw_sm, Mapping):
      for k, v in raw_sm.items():
        if isinstance(v, Mapping):
          v_dict: dict[str, Any] = {str(mk): mv for mk, mv in v.items()}
          parsed_sm[str(k)] = story_state.SubMindState.from_dict(v_dict)
    raw_wc = state.get('win_counts', {})
    if isinstance(raw_wc, Mapping):
      self._win_counts = {
          str(k): int(v)
          for k, v in raw_wc.items()
          if isinstance(v, (int, float))
      }
    raw_pd = state.get('prior_winning_directives', [])
    if isinstance(raw_pd, Sequence) and not isinstance(raw_pd, str):
      self._prior_winning_directives = [str(x) for x in raw_pd if x]
    winning = str(state.get('last_winning_voice') or '')
    if parsed_sm or winning:
      self._ledger.update_npc_sub_minds(
          self._npc_name,
          sub_minds=parsed_sm if parsed_sm else None,
          last_winning_voice=winning if winning else None,
      )
