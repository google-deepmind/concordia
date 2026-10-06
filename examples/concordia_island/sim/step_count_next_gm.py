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

"""Component helping a game master pick which game master to use next based on step count."""

from collections.abc import Iterable, Sequence
import logging
import math
import random
import threading
from typing import Any

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

from examples.concordia_island.sim import component_state
from examples.concordia_island.sim import internet_forum
from examples.concordia_island.sim import matching
from examples.concordia_island.sim import social_scheduler as social_scheduler_lib

# Good date venues on the island
DATE_VENUES = [
    'cafe',
    'restaurant',
    'beach',
    'park',
    'swell_bar',
    'rooftop_lounge',
]


class StepCountNextGM(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Switches back to Island GM after fixed steps and schedules dates.

  Used by the Instagram GM to switch back to the Island GM after the night
  session completes. It also extracts mutual matches from the forum and
  schedules them as dates in the shared SocialScheduler.
  """

  def __init__(
      self,
      player_names: Sequence[str],
      island_gm_name: str = 'island rules',
      instagram_gm_name: str = 'instagram_rules',
      max_steps: int = 2,
      clock_key: str = 'clock',
      forum_component_key: str = internet_forum.DEFAULT_FORUM_COMPONENT_KEY,
      social_scheduler_key: str = (
          social_scheduler_lib.DEFAULT_SOCIAL_SCHEDULER_KEY
      ),
      date_hour: int = 19,
      date_turns: int = 80,
      seed: int = 42,
      round_completion_quorum: float = 0.9,
      turn_budget_factor: float = 1.5,
      pre_act_label: str = '\nNext Game Master',
      mark_night_complete_on_clock: bool = True,
  ):
    """Initializes the component.

    Args:
      player_names: Names of all player entities.
      island_gm_name: Name of the target GM to switch to when finished (e.g.
        'marketplace_rules' or 'island rules').
      instagram_gm_name: Name of the current nighttime GM (e.g. 'x_rules').
      max_steps: Number of rounds each entity plays before it is considered
        finished for the night.
      clock_key: Key of the clock component (shared).
      forum_component_key: Key of the forum state component.
      social_scheduler_key: Key of the social scheduler component (shared).
      date_hour: Hour (0-23) to schedule dates.
      date_turns: Number of conversation turns for scheduled dates.
      seed: Random seed for venue selection.
      round_completion_quorum: Fraction of players that must each have played
        `max_steps` rounds before the night ends. Using a quorum rather than
        requiring every player makes the night robust to entity loops that die
        or never route to this game master; requiring all of them made the exit
        condition unreachable and the night ran forever.
      turn_budget_factor: Absolute safety valve. The night is force-ended once
        the total number of counted queries exceeds
        `max_steps * len(player_names) * turn_budget_factor`, which bounds how
        far the fastest entities can overshoot while waiting for the quorum.
        Set to 0 to disable.
      pre_act_label: Label for logging.
      mark_night_complete_on_clock: Whether to call
        clock.mark_nighttime_completed() when this GM's night session completes.
        Set to False on intermediate nighttime GMs (e.g. X when followed by
        Marketplace) so the clock only marks nighttime completed after the final
        nighttime GM finishes.
    """
    super().__init__()
    self._player_names = list(player_names)
    self._island_gm_name = island_gm_name
    self._instagram_gm_name = instagram_gm_name
    self._max_steps = max_steps
    self._clock_key = clock_key
    self._forum_component_key = forum_component_key
    self._social_scheduler_key = social_scheduler_key
    self._date_hour = date_hour
    self._date_turns = date_turns
    self._rng = random.Random(seed)
    self._round_completion_quorum = round_completion_quorum
    self._turn_budget_factor = turn_budget_factor
    self._pre_act_label = pre_act_label
    self._mark_night_complete_on_clock = mark_night_complete_on_clock
    self._lock = threading.Lock()
    # Rounds completed by the engine-level (simultaneous / sequential) path,
    # where a single query represents one round for every player at once.
    self._steps_taken = 0
    # Rounds played so far tonight, per entity. This is the primary counter
    # when every entity queries independently (asynchronous engine).
    self._entity_steps: dict[str, int] = {}
    # Entities that have already played `max_steps` rounds tonight.
    self._finished_entities: set[str] = set()
    # Total counted queries tonight, used by the turn-budget safety valve.
    self._total_queries: int = 0
    self._last_seen_day: int = -1
    self._night_completed: bool = False
    # Simulated day on which the current night was completed. A new night can
    # only start once the clock has moved past it.
    self._night_completed_day: int = -1
    self._all_matches = []
    # Retained for backwards compatibility of get_state/set_state payloads.
    self._round_queries = set()

  def _schedule_dates(self) -> None:
    """Extract matches from forum and add to SocialScheduler."""
    # `get_component` is declared to return `BaseComponent`, so every
    # subclass-specific call below (get_selections, add_event, _current_dt) is
    # invisible to the type checker. The lookup is dynamic by Concordia's
    # design, so say so once here rather than suppressing at each use site.
    try:
      gm = self.get_entity()
      forum: Any = gm.get_component(self._forum_component_key)
      scheduler: Any = gm.get_component(self._social_scheduler_key)
      clock: Any = gm.get_component(self._clock_key)
    except (AttributeError, KeyError) as e:
      logging.warning(
          'StepCountNextGM: Failed to access required components: %s', e
      )
      return

    selections = forum.get_selections()
    matches = matching.compute_mutual_matches(selections)
    current_day = clock._current_dt.day  # pylint: disable=protected-access

    logging.info(
        'StepCountNextGM: Night ended. Found %d matches on day %d',
        len(matches),
        current_day,
    )

    # Determine date type based on match history
    past_pairs = set()
    for night_matches in self._all_matches:
      for pair in night_matches:
        past_pairs.add(tuple(sorted(pair)))

    for p1, p2 in matches:
      pair_key = tuple(sorted([p1, p2]))
      is_rematch = pair_key in past_pairs
      prompt_type = 'second_date' if is_rematch else 'first_date'
      theme = 'second_date' if is_rematch else 'first_date'

      venue = self._rng.choice(DATE_VENUES)

      event = social_scheduler_lib.SocialEvent(
          participants=(p1, p2),
          venue=venue,
          theme=theme,
          tick_hour=self._date_hour,
          day=current_day,
          prompt_type=prompt_type,
          max_turns=self._date_turns,
          scheduled_by='x',
      )
      scheduler.add_event(event)
      logging.info(
          'StepCountNextGM: Scheduled %s for %s & %s at %s',
          prompt_type,
          p1,
          p2,
          venue,
      )

    # Schedule rumination for unmatched singles
    unmatched = matching.get_unmatched_singles(self._player_names, matches)
    for name in unmatched:
      event = social_scheduler_lib.SocialEvent(
          participants=(name,),
          venue='home',
          theme='single_rumination',
          tick_hour=self._date_hour,
          day=current_day,
          prompt_type='single_rumination',
          max_turns=20,
          scheduled_by='x',
      )
      scheduler.add_event(event)

    self._all_matches.append(matches)

  def _reset_night_state(self) -> None:
    """Clears all per-night bookkeeping so a fresh night can start.

    Caller must hold self._lock.
    """
    self._entity_steps.clear()
    self._finished_entities.clear()
    self._round_queries.clear()
    self._steps_taken = 0
    self._total_queries = 0
    self._night_completed = False
    self._night_completed_day = -1

  def _turn_budget(self) -> int:
    """Absolute cap on counted queries per night (0 means no cap)."""
    if self._turn_budget_factor <= 0:
      return 0
    return int(
        self._max_steps * max(1, len(self._player_names))
        * self._turn_budget_factor
    )

  def _night_should_end(self) -> bool:
    """Whether the nighttime session is finished. Caller must hold the lock."""
    # Engine-level path (simultaneous / sequential engine): one query is one
    # round for every player at once.
    if self._steps_taken >= self._max_steps:
      return True

    num_players = len(self._player_names)
    if num_players == 0:
      return True

    # Per-entity path (asynchronous engine): end once a quorum of players have
    # each played max_steps rounds. A quorum rather than all players is required
    # because entity loops can die or never route here, which would otherwise
    # make the exit condition unreachable.
    quorum = max(
        1, min(num_players, math.ceil(self._round_completion_quorum
                                      * num_players))
    )
    if len(self._finished_entities) >= quorum:
      return True

    # Safety valve: never let a night run unboundedly, whatever happens to the
    # individual entity loops.
    budget = self._turn_budget()
    if budget and self._total_queries >= budget:
      logging.warning(
          'StepCountNextGM: force-ending night after %d queries (budget %d); '
          'only %d/%d players reached %d rounds.',
          self._total_queries,
          budget,
          len(self._finished_entities),
          num_players,
          self._max_steps,
      )
      return True

    return False

  def _complete_night(self, current_day: int) -> None:
    """Marks the night finished and runs end-of-night side effects.

    Caller must hold self._lock.

    Args:
      current_day: Day reported by the clock, or -1 if unavailable.
    """
    self._night_completed = True
    self._night_completed_day = current_day
    logging.info(
        'StepCountNextGM: night complete on day %d after %d queries '
        '(%d/%d players finished %d rounds).',
        current_day,
        self._total_queries,
        len(self._finished_entities),
        len(self._player_names),
        self._max_steps,
    )
    try:
      self._schedule_dates()
    except Exception as e:  # pylint: disable=broad-except
      # Scheduling dates is a side effect; it must never prevent the night
      # from ending.
      logging.warning('StepCountNextGM: _schedule_dates failed: %s', e)
    if self._mark_night_complete_on_clock:
      try:
        clock = self.get_entity().get_component(self._clock_key)
        if hasattr(clock, 'mark_nighttime_completed'):
          clock.mark_nighttime_completed(current_day)
          if current_day > 1:
            clock.mark_nighttime_completed(current_day - 1)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning('StepCountNextGM: Could not mark night complete: %s', e)

    if current_day == -1:
      # No usable clock, so we cannot detect when the next night starts. Reset
      # immediately so the component remains reusable.
      self._reset_night_state()

  def _read_clock_day(self) -> int:
    """Returns the current simulated day, or -1 if the clock is unavailable."""
    try:
      clock = self.get_entity().get_component(self._clock_key)
      if hasattr(clock, '_current_dt'):
        return clock._current_dt.day  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
      pass
    return -1

  def _identify_requesting_entity(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Returns the player this NEXT_GAME_MASTER query is about.

    Args:
      action_spec: The action spec of the incoming query.

    Returns:
      The player name, or '' if the query cannot be attributed to a player this
      game master owns.
    """
    # Primary: parse from the action_spec tag, when the engine embeds the
    # entity id as 'next_game_master:<entity_id>' in the tag field.
    tag = getattr(action_spec, 'tag', '') or ''
    if isinstance(tag, str) and ':' in tag:
      candidate = tag.split(':', 1)[1]
      if candidate in self._player_names:
        return candidate

    # Fallback: read per-thread capture key or _active_capture_key.
    try:
      gm = self.get_entity()
      thread_id = threading.current_thread().ident
      if hasattr(gm, 'get_capture_key_for_thread'):
        key = gm.get_capture_key_for_thread(thread_id)
        if isinstance(key, str) and key in self._player_names:
          return key
      if hasattr(gm, '_capture_key_by_thread'):
        key = gm._capture_key_by_thread.get(thread_id)  # pylint: disable=protected-access
        if isinstance(key, str) and key in self._player_names:
          return key
      if hasattr(gm, '_active_capture_key'):
        key = gm._active_capture_key  # pylint: disable=protected-access
        if isinstance(key, str) and key in self._player_names:
          return key
    except Exception:  # pylint: disable=broad-except
      pass
    return ''

  def _budget_exhausted(self) -> bool:
    """Whether the absolute per-night query budget has been spent.

    Caller must hold self._lock.

    Returns:
      True if a budget is configured and has been reached.
    """
    budget = self._turn_budget()
    return bool(budget) and self._total_queries >= budget

  def _rounds_owed(self, entity_name: str) -> bool:
    """Whether `entity_name` still has rounds left in tonight's budget.

    Caller must hold self._lock.

    Args:
      entity_name: Player name, or '' for an unattributed query.

    Returns:
      True if the named player has played fewer than `max_steps` rounds.
    """
    if not entity_name:
      return False
    return self._entity_steps.get(entity_name, 0) < self._max_steps

  def _serve_after_quorum(self, requesting_entity: str) -> str:
    """Routes a query that arrives after the night was declared complete.

    The quorum exists so that a dead or never-routed entity loop cannot hang
    the night, not to cut short entities that are alive and merely slow. Any
    straggler still short of `max_steps` rounds is therefore served the rest of
    its night, and `_turn_budget` remains the backstop that guarantees the
    night terminates.

    Caller must hold self._lock.

    Args:
      requesting_entity: Player this query is about, or '' if unattributed.

    Returns:
      Name of the game master to run next for this query.
    """
    if not self._rounds_owed(requesting_entity):
      return self._island_gm_name

    if self._budget_exhausted():
      logging.warning(
          'StepCountNextGM: turn budget (%d) exhausted; closing out %s on '
          '%d/%d rounds.',
          self._turn_budget(),
          requesting_entity,
          self._entity_steps.get(requesting_entity, 0),
          self._max_steps,
      )
      return self._island_gm_name

    rounds_played = self._entity_steps.get(requesting_entity, 0) + 1
    self._entity_steps[requesting_entity] = rounds_played
    self._total_queries += 1
    if rounds_played >= self._max_steps:
      self._finished_entities.add(requesting_entity)
    logging.info(
        'StepCountNextGM: serving straggler %s round %d/%d after quorum.',
        requesting_entity,
        rounds_played,
        self._max_steps,
    )
    return self._instagram_gm_name

  def _serve_round(
      self,
      requesting_entity: str,
      tag: str,
      current_day: int,
  ) -> str:
    """Counts one query against tonight's budget and routes it.

    Caller must hold self._lock.

    Args:
      requesting_entity: Player this query is about, or '' if unattributed.
      tag: Raw action spec tag, used to recognise engine-level global queries.
      current_day: Day reported by the clock, or -1 if unavailable.

    Returns:
      Name of the game master to run next for this query.
    """
    counted = False
    # `rounds_played` / `_steps_taken` count the round this query is *about to*
    # grant, not one already played. The caller is therefore still owed the
    # marketplace whenever the round it just claimed is within budget.
    owed_round = False
    if requesting_entity:
      rounds_played = self._entity_steps.get(requesting_entity, 0) + 1
      self._entity_steps[requesting_entity] = rounds_played
      if rounds_played >= self._max_steps:
        self._finished_entities.add(requesting_entity)
      owed_round = rounds_played <= self._max_steps
      # Legacy bookkeeping only: `_round_queries` no longer gates the
      # night-exit decision (that is what made the barrier unreachable), but it
      # is still serialized so old checkpoints round-trip.
      self._round_queries.add(requesting_entity)
      counted = True
    elif not tag or tag == 'next_game_master':
      # Engine-level global query per step (simultaneous / sequential engine):
      # one query represents a full round for every player.
      self._steps_taken += 1
      owed_round = self._steps_taken <= self._max_steps
      counted = True
    # Any other tag names an entity we do not own; ignore it entirely.

    if counted:
      self._total_queries += 1

    logging.info(
        'StepCountNextGM.pre_act: tag=%r, requesting_entity=%r, '
        'finished=%d/%d, steps_taken=%d/%d, total_queries=%d',
        tag,
        requesting_entity,
        len(self._finished_entities),
        len(self._player_names),
        self._steps_taken,
        self._max_steps,
        self._total_queries,
    )

    if counted and self._night_should_end():
      self._complete_night(current_day)
      # The query that trips the quorum still owes this caller the round it
      # just claimed; returning the island GM here is what made the
      # quorum-tripping entity finish the night one round short. The turn
      # budget is the one condition that overrides this, since it exists
      # precisely to stop a runaway night.
      if owed_round and not self._budget_exhausted():
        return self._instagram_gm_name
      return self._island_gm_name
    if not owed_round:
      return self._island_gm_name
    return self._instagram_gm_name

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      with self._lock:
        current_day = self._read_clock_day()
        if current_day != -1:
          self._last_seen_day = current_day

        # A completed night stays completed until the simulated day advances.
        # Note that per-night state is deliberately NOT reset on a day change
        # while a night is still in progress: the clock keeps ticking during
        # the marketplace, and resetting mid-night wiped the round counters
        # before they could ever reach max_steps, so the night never ended.
        if (
            self._night_completed
            and current_day != -1
            and current_day != self._night_completed_day
        ):
          self._reset_night_state()

        requesting_entity = self._identify_requesting_entity(action_spec)
        tag = getattr(action_spec, 'tag', '') or ''

        if self._night_completed:
          result = self._serve_after_quorum(requesting_entity)
        else:
          result = self._serve_round(requesting_entity, tag, current_day)

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result,
        'Value': result,
        'steps_taken': self._steps_taken,
        'max_steps': self._max_steps,
        'finished_entities': len(self._finished_entities),
        'num_players': len(self._player_names),
        'total_queries': self._total_queries,
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'steps_taken': self._steps_taken,
          'entity_steps': dict(self._entity_steps),
          'finished_entities': list(self._finished_entities),
          'round_queries': list(self._round_queries),
          'total_queries': self._total_queries,
          'last_seen_day': self._last_seen_day,
          'night_completed': self._night_completed,
          'night_completed_day': self._night_completed_day,
          'all_matches': [
              [list(m) for m in night] for night in self._all_matches
          ],
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._steps_taken = component_state.as_int(state, 'steps_taken')
      self._entity_steps = component_state.as_str_int_map(
          state, 'entity_steps'
      )
      self._finished_entities = component_state.as_str_set(
          state, 'finished_entities'
      )
      self._round_queries = component_state.as_str_set(state, 'round_queries')
      self._total_queries = component_state.as_int(state, 'total_queries')
      self._last_seen_day = component_state.as_int(state, 'last_seen_day', -1)
      self._night_completed = component_state.as_bool(state, 'night_completed')
      self._night_completed_day = component_state.as_int(
          state, 'night_completed_day', -1
      )
      raw_matches = state.get('all_matches')
      self._all_matches = []
      if isinstance(raw_matches, Iterable) and not isinstance(
          raw_matches, (str, bytes)
      ):
        for night in raw_matches:
          if isinstance(night, Iterable) and not isinstance(
              night, (str, bytes)
          ):
            night_matches = []
            for pair in night:
              if isinstance(pair, Iterable) and not isinstance(
                  pair, (str, bytes)
              ):
                night_matches.append(tuple(str(x) for x in pair))
            self._all_matches.append(night_matches)
