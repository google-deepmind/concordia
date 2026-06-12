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

"""Reusable helper components for the resource dilemma Game Master prefabs.

This module contains components shared across multiple GM prefabs:

  - ConstantNextGameMaster: Always returns a fixed next GM name.
  - ResourceVoterResolution: Runs concurrent voting and tallies results.
  - ResourcePolicyResolution: Runs concurrent policy generation.
  - ResourceHarvestResolution: Runs concurrent harvesting and updates sim state.
  - ResourceSimStateUpdater: Parses resolved events and updates sim state.
  - TurnLimitedNextGameMaster: Forces GM transition after a turn limit.
  - ResourceTerminate: Terminates on stock depletion or cycle exhaustion.
"""

import collections
from collections.abc import Sequence
import json
import random
import re

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.components.game_master import next_game_master
from examples.resource_dilemma import simulation_state as sim_state_lib
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.utils import concurrency
from typing_extensions import override

# Regex for the structured HARVEST tag that agents are instructed to produce.
# Matches CATCH, DIVERT, GRAZE, USE, or generic HARVEST followed by a number.
_HARVEST_TAG_PATTERN = re.compile(
    r'(?:CATCH|DIVERT|GRAZE|USE|HARVEST)\s+(\d+(?:\.\d+)?)'
    r'(?:\s*(?:tonnes?|tons?|acre[- ]?feet|hectares?|Gbps|units?))?',
    re.IGNORECASE,
)

# Regex for extracting numeric harvest amounts from agent text.
_HARVEST_AMOUNT_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:tonnes?|tons?|acre[- ]?feet|hectares?|Gbps|units?)',
    re.IGNORECASE,
)

# Regex for extracting resource stock levels from GM announcements.
_STOCK_LEVEL_PATTERN = re.compile(
    r'(?:(?:fish|water|pasture|bandwidth|resource)\s+)?'
    r'stock\s*(?:is|at|:|=)?\s*(\d+(?:\.\d+)?)',
    re.IGNORECASE,
)


def extract_harvest_amount(text: str) -> float | None:
  """Parse a numeric harvest amount from agent response text.

  Parsing priority:
    1. Structured ``CATCH/DIVERT/GRAZE/USE X`` tag (highest confidence).
    2. Explicit ``N tonnes/acre-feet/hectares/Gbps`` pattern.
    3. Single bare number when exactly one appears.

  Args:
    text: The agent response text to parse.

  Returns:
    The parsed harvest amount, or None if no valid amount is found.
  """
  # Priority 1: structured HARVEST tag.
  tag_match = _HARVEST_TAG_PATTERN.search(text)
  if tag_match:
    try:
      return float(tag_match.group(1))
    except ValueError:
      pass

  # Priority 2: explicit "N <unit>" pattern.
  match = _HARVEST_AMOUNT_PATTERN.search(text)
  if match:
    try:
      return float(match.group(1))
    except ValueError:
      return None
  # Fallback: single bare number in a reasonable range.
  numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
  if len(numbers) == 1:
    try:
      val = float(numbers[0])
      return val
    except ValueError:
      pass
  return None


def _extract_explicit_amounts(text: str) -> list[float]:
  """Parse all explicit 'N <unit>' patterns from text."""
  amounts = []
  for match in _HARVEST_AMOUNT_PATTERN.finditer(text):
    try:
      amounts.append(float(match.group(1)))
    except ValueError:
      pass
  return amounts


class ConstantNextGameMaster(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that always returns a fixed next game master name."""

  def __init__(self, next_gm_name: str):
    super().__init__()
    self._next_gm_name = next_gm_name

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if not action_spec:
      return ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      self._logging_channel({'next_gm': self._next_gm_name})
      return self._next_gm_name
    return ''

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass


class ResourceVoterResolution(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Runs a concurrent voting round among all agents during the RESOLVE phase.

  When the engine enters the RESOLVE phase for the voting GM, this component:
    1. Asks each entity to vote (YES or NO on each proposal).
    2. Tallies votes.
    3. Announces results to all entities.
    4. Records the result in shared memory.
  """

  def __init__(
      self,
      player_agents: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
      model: language_model.LanguageModel,
      memory_bank: associative_memory.AssociativeMemoryBank,
      gm_name: str,
      candidates: Sequence[str],
      sim_state: sim_state_lib.ResourceSimulationState | None = None,
  ):
    super().__init__()
    self._player_agents = player_agents
    self._model = model
    self._memory_bank = memory_bank
    self._gm_name = gm_name
    self._candidates = candidates
    self._sim_state = sim_state

  def _get_player_vote(
      self,
      voter: entity_agent_with_logging.EntityAgentWithLogging,
  ) -> str:
    """Asks a single player to vote for a leader."""
    call_to_action = (
        f'{voter.name}, please vote for a leader from the candidates. '
        f'Candidates are: {", ".join(self._candidates)}'
    )
    action_spec = entity_lib.ActionSpec(
        call_to_action=call_to_action,
        output_type=entity_lib.OutputType.CHOICE,
        options=tuple(self._candidates),
        tag='vote',
    )
    voter.observe(f'The community is voting for a leader. {call_to_action}')
    vote_decision = voter.act(action_spec=action_spec)
    return vote_decision

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if (
        not action_spec
        or action_spec.output_type != entity_lib.OutputType.RESOLVE
    ):
      return ''

    # Conduct the vote concurrently across all players
    votes = collections.defaultdict(int)
    print(f'\n--- [{self._gm_name}] STARTING VOTE ---')
    vote_results = concurrency.map_parallel(
        self._get_player_vote, self._player_agents
    )
    voter_logs = {}
    for player, vote in zip(self._player_agents, vote_results):
      voter_logs[player.name] = vote
      if vote in self._candidates:
        votes[vote] += 1
      else:
        votes['INVALID/ABSTAIN'] += 1
      print(f'{player.name} votes: {vote}')
    print(f'--- [{self._gm_name}] VOTING COMPLETE ---')

    tally = dict(votes)
    candidates_tally = {c: tally.get(c, 0) for c in self._candidates}

    if candidates_tally:
      max_votes = max(candidates_tally.values())
      winners = [c for c, v in candidates_tally.items() if v == max_votes]
      winner = random.choice(winners)
      winner_votes = candidates_tally[winner]
      result_summary = (
          f'ELECTION RESULT: Winner is {winner} with {winner_votes} votes. '
          f'Full tally: {candidates_tally}'
      )
    else:
      winner = None
      result_summary = 'ELECTION RESULT: No candidates, no winner.'

    # Record in shared memory and notify all players
    self._memory_bank.add(f'[{self._gm_name}] {result_summary}')
    for player in self._player_agents:
      player.observe(result_summary)

    print(f'[{self._gm_name}] {result_summary}')

    # Write election result to sim_state so the harvest phase can read the
    # winning policy directly without scanning the memory bank.
    if self._sim_state is not None and winner is not None:
      self._sim_state.election_winner = winner
      # Find the winning leader's policy from memory.
      policy_memories = self._memory_bank.scan(
          lambda x: winner in x and 'proposed policy:' in x
      )
      if policy_memories:
        winner_policy_text = (
            policy_memories[-1].split('proposed policy:')[-1].strip()
        )
        self._sim_state.active_policy = winner_policy_text
        print(
            f'[{self._gm_name}] Active policy set from {winner}:'
            f' {winner_policy_text}'
        )
      else:
        self._sim_state.active_policy = ''
    elif self._sim_state is not None:
      # No winner (no candidates) — clear active policy.
      self._sim_state.election_winner = ''
      self._sim_state.active_policy = ''

    self._logging_channel({
        'winner': winner,
        'outcome': 'Vote conducted',
        'summary': result_summary,
        'tally': tally,
        'voter_logs': voter_logs,
    })
    return json.dumps({
        'summary': result_summary,
        'individual_actions': voter_logs,
    })

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass


class ResourcePolicyResolution(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Runs a concurrent policy generation round among leaders during the RESOLVE phase."""

  def __init__(
      self,
      leaders: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
      model: language_model.LanguageModel,
      memory_bank: associative_memory.AssociativeMemoryBank,
      gm_name: str,
      sim_state: sim_state_lib.ResourceSimulationState | None = None,
  ):
    super().__init__()
    self._leaders = leaders
    self._model = model
    self._memory_bank = memory_bank
    self._gm_name = gm_name
    self._sim_state = sim_state

  def _get_leader_policy(
      self,
      leader: entity_agent_with_logging.EntityAgentWithLogging,
  ) -> str:
    """Asks a single leader to propose a policy."""
    call_to_action = (
        f'{leader.name}, please propose your policy agenda for governing the'
        ' shared resource. Your goal is to come up with rules for the'
        ' community to sustain the resource and prevent depletion. Include a'
        ' recommended usage limit and a penalty schedule. Review the events'
        ' of the past cycle (if any), including who followed or broke the'
        ' rules and whether overuse occurred, and use these lessons to craft'
        ' your policy. Recall and use your memories of past cycles and'
        ' interactions to inform your policy. CRITICAL: if the resource is'
        ' completely depleted, it collapses permanently and everyone —'
        ' including you — loses EVERYTHING. Your policy must prevent total'
        ' collapse. Consider what happened in past cycles and how others'
        ' are likely to react to your policies.'
    )
    action_spec = entity_lib.ActionSpec(
        call_to_action=call_to_action,
        output_type=entity_lib.OutputType.FREE,
        tag='policy_generation',
    )
    leader.observe(f'It is time to propose a policy. {call_to_action}')
    policy = leader.act(action_spec=action_spec)
    return policy

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if (
        not action_spec
        or action_spec.output_type != entity_lib.OutputType.RESOLVE
    ):
      return ''

    # Reset per-cycle policy fields so that if no election follows,
    # the harvest phase starts with an empty active policy.
    if self._sim_state is not None:
      self._sim_state.election_winner = ''
      self._sim_state.active_policy = ''

    print(f'\n--- [{self._gm_name}] STARTING POLICY GENERATION ---')
    policies = concurrency.map_parallel(self._get_leader_policy, self._leaders)
    print(f'--- [{self._gm_name}] POLICY GENERATION COMPLETE ---')

    individual_actions = {}
    summary_lines = []
    for leader, policy in zip(self._leaders, policies):
      individual_actions[leader.name] = policy
      summary_lines.append(f'{leader.name} proposed: {policy}')
      self._memory_bank.add(
          f'[{self._gm_name}] {leader.name} proposed policy: {policy}'
      )

    result_summary = '\n'.join(summary_lines)

    return json.dumps({
        'summary': result_summary,
        'individual_actions': individual_actions,
    })

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass


class ResourceHarvestResolution(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Runs a concurrent harvesting round among all agents during the RESOLVE phase.

  After collecting each agent's harvest decision, this component parses the
  numeric harvest amounts and updates the shared ``ResourceSimulationState``:

    - ``cycle_harvest_total`` is accumulated from individual harvests.
    - ``resource_level`` is reduced by the total harvest (clamped to 0).
  """

  def __init__(
      self,
      players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
      model: language_model.LanguageModel,
      memory_bank: associative_memory.AssociativeMemoryBank,
      gm_name: str,
      sim_state: sim_state_lib.ResourceSimulationState | None = None,
      call_to_action: str = '',
  ):
    super().__init__()
    self._players = players
    self._model = model
    self._memory_bank = memory_bank
    self._gm_name = gm_name
    self._current_policy = ''
    self._sim_state = sim_state
    self._did_harvest = False
    self._call_to_action = call_to_action

  def _get_player_harvest(
      self,
      player: entity_agent_with_logging.EntityAgentWithLogging,
  ) -> str:
    """Asks a single player to harvest."""
    call_to_action = self._call_to_action or (
        f'{player.name}, remember that many users share this resource and'
        ' the total usage from everyone combined determines whether the'
        ' resource survives. If the resource is completely depleted, it'
        ' collapses permanently and everyone loses — including you.'
        ' How much do you decide to use this cycle (0-20)?'
        ' You MUST end your response with your final decision on a new'
        ' line in exactly this format: HARVEST X'
        ' (where X is a single number).'
    )
    if self._current_policy:
      call_to_action = (
          f'The active policy is: {self._current_policy}. ' + call_to_action
      )

    action_spec = entity_lib.ActionSpec(
        call_to_action=call_to_action,
        output_type=entity_lib.OutputType.FREE,
        tag='harvesting',
    )
    player.observe(f'It is time to harvest. {call_to_action}')
    harvest = player.act(action_spec=action_spec)
    return harvest

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if (
        not action_spec
        or action_spec.output_type != entity_lib.OutputType.RESOLVE
    ):
      return ''

    # Advance cycle from the PREVIOUS harvest so that the logger's
    # post_act (which runs in the same step as this pre_act) still
    # sees the old cycle number.  Skipped on the very first harvest.
    if self._did_harvest and self._sim_state is not None:
      self._sim_state.current_cycle += 1
      self._sim_state.cycle_harvest_total = 0.0
    self._did_harvest = False

    # NOTE: discussion_completed is reset by ResourceTerminate during the
    # TERMINATE action spec, not here.  Resetting during RESOLVE would
    # clear the flag before the next iteration's terminate check can
    # observe it, because the Simultaneous engine loop checks terminate()
    # AFTER resolve() has already run on the same GM.

    # Retrieve the winning leader's policy from sim_state (set by the election
    # phase).  Falls back to an empty string if no election was held this cycle.
    if self._sim_state is not None:
      self._current_policy = self._sim_state.active_policy
      if self._current_policy:
        print(
            f'[{self._gm_name}] Active policy from'
            f' {self._sim_state.election_winner}: {self._current_policy}'
        )
      else:
        print(f'[{self._gm_name}] No active policy this cycle.')
    else:
      self._current_policy = ''

    print(f'\n--- [{self._gm_name}] STARTING HARVEST ---')
    harvests = concurrency.map_parallel(self._get_player_harvest, self._players)
    print(f'--- [{self._gm_name}] HARVEST COMPLETE ---')

    individual_actions = {}
    summary_lines = []
    total_harvest = 0.0
    for player, harvest in zip(self._players, harvests):
      individual_actions[player.name] = harvest
      summary_lines.append(f'{player.name} decided to use: {harvest}')
      self._memory_bank.add(
          f'[{self._gm_name}] {player.name} decided to use: {harvest}'
      )
      # Parse numeric harvest for sim state update
      parsed = extract_harvest_amount(harvest)
      if parsed is not None:
        total_harvest += parsed

    # Update simulation state with harvest results
    if self._sim_state is not None:
      self._sim_state.cycle_harvest_total += total_harvest
      self._sim_state.resource_level = max(
          0.0, self._sim_state.resource_level - total_harvest
      )

    result_summary = '\n'.join(summary_lines)

    self._did_harvest = True
    return json.dumps({
        'summary': result_summary,
        'individual_actions': individual_actions,
    })

  def post_act(self, action_attempt: str) -> str:
    # Cycle increment is deferred to the start of the next pre_act(RESOLVE)
    # so that the logger can read the current cycle in its own post_act.
    return ''

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass


class TurnLimitedNextGameMaster(next_game_master.NextGameMaster):
  """NextGameMaster that forces transition after a turn limit."""

  def __init__(
      self,
      limit: int,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._limit = int(limit)
    self._turns = 0

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      self._turns += 1
      if self._turns >= self._limit:
        self._turns = 0  # Reset for the next cycle's discussion
        # Force transition to the first choice (usually the next GM)
        self._currently_active_game_master = self._game_master_names[0]
        return self._currently_active_game_master
    return super().pre_act(action_spec)


class ResourceSimStateUpdater(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Parses resolved event text and updates ``ResourceSimulationState``.

  This component is used on GMs that use LLM-based event resolution
  where harvest amounts and stock levels appear in free-form natural
  language text rather than structured harvest resolution.

  In ``post_act`` it parses:
    - Individual harvest amounts → accumulates ``cycle_harvest_total`` and
      deducts from ``resource_level``.
    - Stock level announcements → sets ``resource_level`` directly.
  """

  def __init__(
      self,
      sim_state: sim_state_lib.ResourceSimulationState,
  ):
    super().__init__()
    self._sim_state = sim_state
    self._did_resolve = False

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec and action_spec.output_type == entity_lib.OutputType.RESOLVE:
      # Advance cycle from the PREVIOUS resolve so that the logger's
      # post_act sees the old cycle number.
      if self._did_resolve:
        self._sim_state.current_cycle += 1
      self._did_resolve = True
    return ''

  def post_act(self, action_attempt: str) -> str:
    """Parse the resolved event text and update sim state."""
    # Reset the cycle harvest total at beginning of post_act for new cycle.
    self._sim_state.cycle_harvest_total = 0.0

    # Try JSON format (individual_actions dict)
    harvest_occurred = False
    try:
      data = json.loads(action_attempt)
      individual_actions = data.get('individual_actions', {})
      total_harvest = 0.0
      for _, action in individual_actions.items():
        parsed = extract_harvest_amount(str(action))
        if parsed is not None:
          total_harvest += parsed
      if total_harvest > 0.0:
        self._sim_state.cycle_harvest_total += total_harvest
        self._sim_state.resource_level = max(
            0.0, self._sim_state.resource_level - total_harvest
        )
        harvest_occurred = True
    except (json.JSONDecodeError, AttributeError):
      pass

    # Fallback: parse free-form text harvest amounts if no harvest in JSON.
    if not harvest_occurred:
      amounts = _extract_explicit_amounts(action_attempt)
      if amounts:
        total_harvest = sum(amounts)
        self._sim_state.cycle_harvest_total += total_harvest
        self._sim_state.resource_level = max(
            0.0, self._sim_state.resource_level - total_harvest
        )

    # Also check for stock level announcements from the GM
    stock_match = _STOCK_LEVEL_PATTERN.search(action_attempt)
    if stock_match:
      try:
        self._sim_state.resource_level = float(stock_match.group(1))
      except ValueError:
        pass

    # Cycle increment is now handled at the start of the next
    # pre_act(RESOLVE) so the logger sees the correct cycle number.
    self._did_resolve = False

    return ''

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass


class ResourceTerminate(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Terminate when resource stock drops below threshold or cycles are exhausted.

  This component replaces ``NeverTerminate`` on resource GMs.  It checks
  the shared ``ResourceSimulationState`` for two conditions:

    1. Resource level has dropped below ``stock_threshold`` (default 5.0)
       — the resource has effectively collapsed.
    2. The current cycle exceeds the configured total number of cycles.

  Termination is deferred until the discussion phase has completed at
  least once in the current cycle, ensuring agents always have an
  opportunity to deliberate before the simulation stops.

  Once termination is triggered, the ``terminated`` flag on the shared
  state is set so that all subsequent GMs also honour the decision.
  """

  def __init__(
      self,
      sim_state: sim_state_lib.ResourceSimulationState,
      stock_threshold: float = 5.0,
      phase: str = '',
  ):
    super().__init__()
    self._sim_state = sim_state
    self._threshold = stock_threshold
    self._phase = phase

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      s = self._sim_state

      # Never terminate during discussion — let the full turn limit run.
      # Mark discussion as complete so the *next* phase can terminate.
      if self._phase == 'discussion':
        s.discussion_completed = True
        return entity_lib.BINARY_OPTIONS['negative']

      # If already terminated, propagate immediately.
      if s.terminated:
        return entity_lib.BINARY_OPTIONS['affirmative']

      # Check termination conditions.
      should_terminate = (
          s.resource_level < self._threshold
          or s.current_cycle >= s.total_cycles
      )

      if should_terminate:
        # Defer until discussion has happened at least once this cycle.
        if not s.discussion_completed:
          return entity_lib.BINARY_OPTIONS['negative']
        s.terminated = True
        return entity_lib.BINARY_OPTIONS['affirmative']

      # Not terminating — apply end-of-cycle regeneration (stock
      # doubling) if a full cycle just completed (discussion finished).
      # Regeneration doubles the remaining stock, capped at carrying
      # capacity.  This only fires when discussion_completed is True,
      # so it runs exactly once per cycle.
      if s.discussion_completed:
        before = s.resource_level
        s.resource_level = min(s.resource_level * 2.0, s.carrying_capacity)
        print(
            f'[ResourceTerminate] Regeneration: {before:.1f}'
            f' -> {s.resource_level:.1f}'
            f' (cap {s.carrying_capacity:.0f})'
        )

      # Reset the discussion flag so that the NEXT cycle's terminate
      # check will wait for discussion to complete again.  Doing this
      # here (during TERMINATE) instead of in ResourceHarvestResolution
      # (during RESOLVE) avoids a race: the engine's run_loop calls
      # terminate() at the top of the loop, AFTER the previous
      # iteration's resolve() already ran.
      s.discussion_completed = False
      return entity_lib.BINARY_OPTIONS['negative']
    return ''

  def get_state(self) -> entity_component.ComponentState:
    return {
        'current_cycle': self._sim_state.current_cycle,
        'total_cycles': self._sim_state.total_cycles,
        'resource_level': self._sim_state.resource_level,
        'cycle_harvest_total': self._sim_state.cycle_harvest_total,
        'terminated': self._sim_state.terminated,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'current_cycle' in state:
      self._sim_state.current_cycle = state['current_cycle']
    if 'total_cycles' in state:
      self._sim_state.total_cycles = state['total_cycles']
    if 'resource_level' in state:
      self._sim_state.resource_level = state['resource_level']
    if 'cycle_harvest_total' in state:
      self._sim_state.cycle_harvest_total = state['cycle_harvest_total']
    if 'terminated' in state:
      self._sim_state.terminated = state['terminated']
