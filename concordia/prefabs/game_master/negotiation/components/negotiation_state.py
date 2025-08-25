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

"""Negotiation state tracking component for game master."""

import dataclasses
import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from concordia.typing import entity_component


@dataclasses.dataclass
class NegotiationOffer:
  """Represents an offer in a negotiation."""
  offerer: str
  recipient: str
  offer_type: str  # 'initial', 'counter', 'final'
  terms: Dict[str, Any]
  timestamp: datetime.datetime
  round_number: int
  is_accepted: bool = False
  is_rejected: bool = False
  rejection_reason: Optional[str] = None


@dataclasses.dataclass
class NegotiationState:
  """Current state of a negotiation."""
  negotiation_id: str
  participants: List[str]
  phase: str  # 'opening', 'bargaining', 'closing', 'completed', 'failed'
  current_round: int
  offers_history: List[NegotiationOffer]
  agreements: Dict[str, Any]
  active_offer: Optional[NegotiationOffer]
  deadline: Optional[datetime.datetime]
  termination_reason: Optional[str]

  def get_latest_offer(self) -> Optional[NegotiationOffer]:
    """Get the most recent offer."""
    return self.offers_history[-1] if self.offers_history else None

  def get_participant_offers(self, participant: str) -> List[NegotiationOffer]:
    """Get all offers made by a specific participant."""
    return [o for o in self.offers_history if o.offerer == participant]


class NegotiationStateTracker(entity_component.ContextComponent):
  """Tracks negotiation state and history for the game master.

  This component:
  - Maintains current negotiation state
  - Tracks offer/counteroffer history
  - Monitors negotiation phases
  - Records agreements and outcomes
  - Handles multi-party negotiations
  """

  def __init__(
      self,
      initial_phase: str = 'opening',
      max_rounds: int = 20,
      enable_deadlines: bool = True,
      track_relationships: bool = True,
  ):
    """Initialize negotiation state tracker.

    Args:
      initial_phase: Starting phase of negotiations
      max_rounds: Maximum rounds before forced termination
      enable_deadlines: Whether to enforce time limits
      track_relationships: Whether to track relationship changes
    """
    self._initial_phase = initial_phase
    self._max_rounds = max_rounds
    self._enable_deadlines = enable_deadlines
    self._track_relationships = track_relationships

    # Active negotiations
    self._negotiations: Dict[str, NegotiationState] = {}

    # Relationship tracking
    self._relationships: Dict[Tuple[str, str], float] = {}

    # Statistics
    self._completed_negotiations = 0
    self._failed_negotiations = 0

  def start_negotiation(
      self,
      negotiation_id: str,
      participants: List[str],
      deadline: Optional[datetime.datetime] = None,
  ) -> NegotiationState:
    """Start a new negotiation."""
    state = NegotiationState(
        negotiation_id=negotiation_id,
        participants=participants,
        phase=self._initial_phase,
        current_round=0,
        offers_history=[],
        agreements={},
        active_offer=None,
        deadline=deadline if self._enable_deadlines else None,
        termination_reason=None,
    )

    self._negotiations[negotiation_id] = state

    # Initialize relationships if tracking
    if self._track_relationships:
      for i, p1 in enumerate(participants):
        for p2 in participants[i+1:]:
          key = tuple(sorted([p1, p2]))
          if key not in self._relationships:
            self._relationships[key] = 0.5  # Neutral starting relationship

    return state

  def record_offer(
      self,
      negotiation_id: str,
      offerer: str,
      recipient: str,
      offer_type: str,
      terms: Dict[str, Any],
  ) -> NegotiationOffer:
    """Record a new offer in the negotiation."""
    if negotiation_id not in self._negotiations:
      raise ValueError(f"Unknown negotiation: {negotiation_id}")

    state = self._negotiations[negotiation_id]

    # Create offer
    offer = NegotiationOffer(
        offerer=offerer,
        recipient=recipient,
        offer_type=offer_type,
        terms=terms,
        timestamp=datetime.datetime.now(),
        round_number=state.current_round,
    )

    # Update state
    state.offers_history.append(offer)
    state.active_offer = offer

    # Update phase if needed
    if state.phase == 'opening' and len(state.offers_history) > 2:
      state.phase = 'bargaining'

    return offer

  def accept_offer(
      self,
      negotiation_id: str,
      offer: NegotiationOffer,
      acceptor: str,
  ) -> None:
    """Accept an offer."""
    if negotiation_id not in self._negotiations:
      raise ValueError(f"Unknown negotiation: {negotiation_id}")

    state = self._negotiations[negotiation_id]
    offer.is_accepted = True

    # Record agreement
    state.agreements[f"round_{offer.round_number}"] = {
        'terms': offer.terms,
        'offerer': offer.offerer,
        'acceptor': acceptor,
        'timestamp': datetime.datetime.now(),
    }

    # Update phase
    state.phase = 'closing'

    # Update relationships positively
    if self._track_relationships:
      key = tuple(sorted([offer.offerer, acceptor]))
      if key in self._relationships:
        self._relationships[key] = min(1.0, self._relationships[key] + 0.1)

  def reject_offer(
      self,
      negotiation_id: str,
      offer: NegotiationOffer,
      rejector: str,
      reason: Optional[str] = None,
  ) -> None:
    """Reject an offer."""
    if negotiation_id not in self._negotiations:
      raise ValueError(f"Unknown negotiation: {negotiation_id}")

    offer.is_rejected = True
    offer.rejection_reason = reason

    # Clear active offer
    self._negotiations[negotiation_id].active_offer = None

    # Slight negative relationship impact
    if self._track_relationships:
      key = tuple(sorted([offer.offerer, rejector]))
      if key in self._relationships:
        self._relationships[key] = max(0.0, self._relationships[key] - 0.05)

  def advance_round(self, negotiation_id: str) -> None:
    """Move to the next negotiation round."""
    if negotiation_id not in self._negotiations:
      raise ValueError(f"Unknown negotiation: {negotiation_id}")

    state = self._negotiations[negotiation_id]
    state.current_round += 1

    # Check max rounds
    if state.current_round >= self._max_rounds:
      self.terminate_negotiation(
          negotiation_id,
          reason="Maximum rounds reached",
          success=False,
      )

  def terminate_negotiation(
      self,
      negotiation_id: str,
      reason: str,
      success: bool,
  ) -> None:
    """Terminate a negotiation."""
    if negotiation_id not in self._negotiations:
      raise ValueError(f"Unknown negotiation: {negotiation_id}")

    state = self._negotiations[negotiation_id]
    state.phase = 'completed' if success else 'failed'
    state.termination_reason = reason

    # Update statistics
    if success:
      self._completed_negotiations += 1
    else:
      self._failed_negotiations += 1

  def get_negotiation_summary(self, negotiation_id: str) -> str:
    """Get a summary of the negotiation state."""
    if negotiation_id not in self._negotiations:
      return f"No negotiation found with ID: {negotiation_id}"

    state = self._negotiations[negotiation_id]

    summary = f"NEGOTIATION STATUS ({negotiation_id}):\n"
    summary += f"Participants: {', '.join(state.participants)}\n"
    summary += f"Phase: {state.phase}\n"
    summary += f"Round: {state.current_round}\n"
    summary += f"Total offers: {len(state.offers_history)}\n"

    if state.active_offer:
      summary += f"\nActive offer from {state.active_offer.offerer}:\n"
      for key, value in state.active_offer.terms.items():
        summary += f"  - {key}: {value}\n"

    if state.agreements:
      summary += f"\nAgreements reached: {len(state.agreements)}\n"

    if state.termination_reason:
      summary += f"\nTermination reason: {state.termination_reason}\n"

    return summary

  def get_relationship_status(self, participant1: str, participant2: str) -> float:
    """Get relationship score between two participants."""
    key = tuple(sorted([participant1, participant2]))
    return self._relationships.get(key, 0.5)

  def pre_act(self, action_spec) -> str:
    """Provide negotiation state context."""
    # Find active negotiations
    active_negotiations = [
        neg_id for neg_id, state in self._negotiations.items()
        if state.phase not in ['completed', 'failed']
    ]

    if not active_negotiations:
      return "No active negotiations."

    context = "ACTIVE NEGOTIATIONS:\n\n"
    for neg_id in active_negotiations:
      context += self.get_negotiation_summary(neg_id) + "\n"

    return context

  def post_act(self, action_attempt: str) -> None:
    """Process actions that might affect negotiation state."""
    # This would parse action_attempt to update state
    # In practice, the game master would call our methods directly
    pass

  def pre_observe(self, observation: str) -> None:
    """Process observations."""
    pass

  def post_observe(self) -> None:
    """Post-observation processing."""
    pass

  def update(self) -> None:
    """Update internal state."""
    # Check deadlines if enabled
    if self._enable_deadlines:
      current_time = datetime.datetime.now()
      for neg_id, state in self._negotiations.items():
        if state.deadline and current_time > state.deadline:
          if state.phase not in ['completed', 'failed']:
            self.terminate_negotiation(
                neg_id,
                reason="Deadline exceeded",
                success=False,
            )

  @property
  def name(self) -> str:
    """Component name."""
    return 'NegotiationStateTracker'

  def get_state(self) -> str:
    """Get component state for saving."""
    return f"{self._completed_negotiations}|{self._failed_negotiations}"

  def set_state(self, state: str) -> None:
    """Restore component state."""
    if '|' in state:
      completed, failed = state.split('|', 1)
      self._completed_negotiations = int(completed)
      self._failed_negotiations = int(failed)
