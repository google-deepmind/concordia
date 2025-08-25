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

"""Temporal dynamics module for negotiation game master."""

import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

from concordia.prefabs.game_master.negotiation.components import negotiation_modules


@dataclasses.dataclass
class RelationshipState:
  """Tracks relationship dynamics over time."""
  trust_level: float  # 0-1
  commitment_level: float  # 0-1
  history_length: int  # Number of interactions
  recent_trajectory: str  # 'improving', 'stable', 'declining'
  long_term_value: float  # Projected long-term relationship value


@dataclasses.dataclass
class TemporalCommitment:
  """Represents a time-bound commitment."""
  committer: str
  commitment_type: str
  terms: Dict[str, Any]
  time_horizon: str  # 'immediate', 'short', 'medium', 'long'
  created_round: int
  deadline_round: Optional[int]
  fulfilled: bool = False


class TemporalDynamicsGM(negotiation_modules.NegotiationGMModule):
  """GM module for managing temporal aspects of negotiations."""

  def __init__(
      self,
      name: str = 'temporal_dynamics',
      priority: int = 80,
      config: Optional[Dict[str, Any]] = None,
  ):
    """Initialize temporal dynamics module."""
    super().__init__(name, priority, config)

    # Relationship tracking
    self._relationships: Dict[Tuple[str, str], RelationshipState] = {}

    # Commitment tracking
    self._commitments: List[TemporalCommitment] = []

    # Phase timing tracking
    self._phase_durations: Dict[str, int] = {
        'opening': 0,
        'bargaining': 0,
        'closing': 0,
    }
    self._phase_transitions: List[Tuple[str, int]] = []

    # Reputation scores
    self._reputation_scores: Dict[str, float] = {}

    # Configuration
    self._track_relationships = self._config.get('track_relationships', True)
    self._enforce_commitments = self._config.get('enforce_commitments', True)
    self._reputation_weight = self._config.get('reputation_weight', 0.3)
    self._discount_factor = self._config.get('discount_factor', 0.9)

  def get_supported_agent_modules(self) -> Set[str]:
    """Return agent modules this supports."""
    return {'temporal_strategy'}

  def get_relationship(self, party1: str, party2: str) -> RelationshipState:
    """Get relationship state between two parties."""
    key = tuple(sorted([party1, party2]))
    if key not in self._relationships:
      self._relationships[key] = RelationshipState(
          trust_level=0.5,
          commitment_level=0.0,
          history_length=0,
          recent_trajectory='stable',
          long_term_value=0.5,
      )
    return self._relationships[key]

  def update_relationship(
      self,
      party1: str,
      party2: str,
      trust_delta: float = 0.0,
      commitment_delta: float = 0.0,
  ) -> None:
    """Update relationship metrics between parties."""
    relationship = self.get_relationship(party1, party2)

    # Update trust (bounded 0-1)
    old_trust = relationship.trust_level
    relationship.trust_level = max(0, min(1, relationship.trust_level + trust_delta))

    # Update commitment (bounded 0-1)
    relationship.commitment_level = max(0, min(1, relationship.commitment_level + commitment_delta))

    # Update history
    relationship.history_length += 1

    # Determine trajectory
    if relationship.trust_level > old_trust + 0.05:
      relationship.recent_trajectory = 'improving'
    elif relationship.trust_level < old_trust - 0.05:
      relationship.recent_trajectory = 'declining'
    else:
      relationship.recent_trajectory = 'stable'

    # Calculate long-term value
    relationship.long_term_value = (
        relationship.trust_level * 0.4 +
        relationship.commitment_level * 0.3 +
        (1.0 if relationship.recent_trajectory == 'improving' else 0.5) * 0.3
    )

  def record_commitment(
      self,
      committer: str,
      commitment_type: str,
      terms: Dict[str, Any],
      time_horizon: str,
      current_round: int,
      deadline_round: Optional[int] = None,
  ) -> None:
    """Record a temporal commitment."""
    commitment = TemporalCommitment(
        committer=committer,
        commitment_type=commitment_type,
        terms=terms,
        time_horizon=time_horizon,
        created_round=current_round,
        deadline_round=deadline_round,
    )
    self._commitments.append(commitment)

  def check_commitment_violations(
      self,
      current_round: int,
  ) -> List[TemporalCommitment]:
    """Check for commitment deadline violations."""
    violations = []
    for commitment in self._commitments:
      if (not commitment.fulfilled and
          commitment.deadline_round and
          current_round > commitment.deadline_round):
        violations.append(commitment)
    return violations

  def validate_action(
      self,
      actor: str,
      action: str,
      context: negotiation_modules.ModuleContext,
  ) -> Tuple[bool, Optional[str]]:
    """Validate action against temporal commitments."""
    if not self._enforce_commitments:
      return True, None

    # Check if action violates any active commitments
    for commitment in self._commitments:
      if commitment.committer == actor and not commitment.fulfilled:
        # Simple check - in practice would parse action more carefully
        if 'withdraw' in action.lower() or 'cancel' in action.lower():
          if commitment.time_horizon in ['medium', 'long']:
            return False, f"Cannot withdraw - violates {commitment.time_horizon}-term commitment"

    return True, None

  def update_state(
      self,
      event: str,
      actor: str,
      context: negotiation_modules.ModuleContext,
  ) -> None:
    """Update temporal state based on events."""
    # Track phase durations
    current_phase = context.current_phase
    if current_phase in self._phase_durations:
      self._phase_durations[current_phase] += 1

    # Track phase transitions
    if (self._phase_transitions and
        self._phase_transitions[-1][0] != current_phase):
      self._phase_transitions.append((current_phase, context.current_round))
    elif not self._phase_transitions:
      self._phase_transitions.append((current_phase, context.current_round))

    # Update relationships based on actions
    action_lower = event.lower()
    for participant in context.participants:
      if participant != actor:
        # Positive relationship updates
        if any(word in action_lower for word in ['agree', 'accept', 'compromise']):
          self.update_relationship(actor, participant, trust_delta=0.1, commitment_delta=0.05)
        # Negative relationship updates
        elif any(word in action_lower for word in ['reject', 'refuse', 'withdraw']):
          self.update_relationship(actor, participant, trust_delta=-0.05)
        # Commitment actions
        elif any(word in action_lower for word in ['commit', 'promise', 'guarantee']):
          self.update_relationship(actor, participant, commitment_delta=0.1)
          # Record commitment
          if 'long' in action_lower:
            time_horizon = 'long'
          elif 'short' in action_lower:
            time_horizon = 'short'
          else:
            time_horizon = 'medium'
          self.record_commitment(
              actor, 'promise', {'action': event},
              time_horizon, context.current_round
          )

    # Update reputation based on commitment fulfillment
    violations = self.check_commitment_violations(context.current_round)
    for violation in violations:
      if violation.committer not in self._reputation_scores:
        self._reputation_scores[violation.committer] = 1.0
      self._reputation_scores[violation.committer] *= 0.9  # Reputation penalty

  def get_observation_context(
      self,
      observer: str,
      context: negotiation_modules.ModuleContext,
  ) -> str:
    """Get temporal context for observations."""
    observation = "\nTEMPORAL DYNAMICS:\n"

    # Phase timing
    total_rounds = sum(self._phase_durations.values())
    if total_rounds > 0:
      observation += f"Negotiation duration: {total_rounds} rounds\n"
      if context.current_phase == 'opening' and total_rounds > 3:
        observation += "- Extended opening phase may indicate relationship building\n"
      elif context.current_phase == 'bargaining' and self._phase_durations['bargaining'] > 10:
        observation += "- Prolonged bargaining suggests complex issues\n"

    # Relationship status with other parties
    for participant in context.participants:
      if participant != observer:
        relationship = self.get_relationship(observer, participant)
        observation += f"\nRelationship with {participant}:\n"
        observation += f"- Trust: {relationship.trust_level:.1%} ({relationship.recent_trajectory})\n"
        observation += f"- Commitment: {relationship.commitment_level:.1%}\n"
        observation += f"- Long-term value: {relationship.long_term_value:.1%}\n"

    # Active commitments
    observer_commitments = [c for c in self._commitments
                           if c.committer == observer and not c.fulfilled]
    if observer_commitments:
      observation += f"\nYour active commitments: {len(observer_commitments)}\n"
      for commitment in observer_commitments[-2:]:  # Show last 2
        observation += f"- {commitment.commitment_type} ({commitment.time_horizon}-term)\n"

    # Reputation status
    if observer in self._reputation_scores:
      observation += f"\nYour reputation score: {self._reputation_scores[observer]:.1%}\n"

    return observation

  def get_module_report(self) -> str:
    """Get temporal dynamics report."""
    report = "TEMPORAL DYNAMICS REPORT:\n\n"

    # Phase analysis
    total_rounds = sum(self._phase_durations.values())
    if total_rounds > 0:
      report += "Phase Distribution:\n"
      for phase, duration in self._phase_durations.items():
        if duration > 0:
          report += f"- {phase}: {duration} rounds ({duration/total_rounds:.0%})\n"

    # Relationship summary
    if self._relationships:
      report += "\nRelationship Matrix:\n"
      for (p1, p2), rel in self._relationships.items():
        report += f"- {p1} <-> {p2}: "
        report += f"Trust {rel.trust_level:.0%}, "
        report += f"Commitment {rel.commitment_level:.0%}, "
        report += f"{rel.recent_trajectory}\n"

    # Commitment summary
    active_commitments = [c for c in self._commitments if not c.fulfilled]
    if active_commitments:
      report += f"\nActive Commitments: {len(active_commitments)}\n"
      horizon_counts = {}
      for c in active_commitments:
        horizon_counts[c.time_horizon] = horizon_counts.get(c.time_horizon, 0) + 1
      for horizon, count in horizon_counts.items():
        report += f"- {horizon}-term: {count}\n"

    # Reputation standings
    if self._reputation_scores:
      report += "\nReputation Scores:\n"
      for party, score in sorted(self._reputation_scores.items(),
                                key=lambda x: x[1], reverse=True):
        report += f"- {party}: {score:.0%}\n"

    # Temporal insights
    if total_rounds > 10:
      avg_trust = sum(r.trust_level for r in self._relationships.values()) / len(self._relationships) if self._relationships else 0.5
      if avg_trust > 0.7:
        report += "\nInsight: High trust levels suggest successful relationship building\n"
      elif avg_trust < 0.3:
        report += "\nInsight: Low trust levels may hinder agreement\n"

    return report

  def get_state(self) -> str:
    """Get the component state for saving/restoring."""
    state_dict = {
        'relationships': len(self._relationships),
        'commitments': len(self._commitments),
        'reputation': len(self._reputation_scores),
        'milestones': len(self._milestones),
    }
    return str(state_dict)

  def set_state(self, state: str) -> None:
    """Set the component state from a saved string."""
    # Since this tracks dynamic data, we only restore basic structure
    pass


# Register the module
negotiation_modules.NegotiationGMModuleRegistry.register(
    'temporal_dynamics',
    TemporalDynamicsGM
)
