"""Specialized memory component for negotiation contexts."""

import dataclasses
import datetime
from typing import Dict, List, Optional, Any

from concordia.associative_memory import basic_associative_memory
from concordia.typing import entity_component


@dataclasses.dataclass
class Offer:
    """Represents a negotiation offer."""
    offerer: str
    recipient: str
    content: str
    value: Optional[float] = None
    timestamp: Optional[datetime.datetime] = None
    round_number: int = 0
    offer_type: str = 'standard'  # standard, counter, final


@dataclasses.dataclass
class NegotiationOutcome:
    """Represents the outcome of a negotiation."""
    agreement_reached: bool
    final_value: Optional[float]
    rounds_taken: int
    participants: List[str]
    summary: str


class NegotiationMemory(entity_component.ContextComponent):
    """Memory component specialized for tracking negotiation history.

    This component extends standard memory with:
    - Structured storage of offers and counteroffers
    - Pattern recognition from past negotiations
    - Outcome tracking and learning
    - Strategic memory retrieval
    """

    def __init__(
        self,
        agent_name: str,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
        verbose: bool = False,
    ):
        """Initialize negotiation memory.

        Args:
            agent_name: Name of the agent
            memory_bank: Associative memory bank for storage
            verbose: Whether to print debug information
        """
        self._agent_name = agent_name
        self._memory = memory_bank
        self._verbose = verbose

        # Negotiation-specific tracking
        self._offer_history: List[Offer] = []
        self._negotiation_outcomes: List[NegotiationOutcome] = []
        self._current_negotiation_partners: List[str] = []
        self._patterns_learned: Dict[str, Any] = {}

        # Current negotiation state
        self._current_best_offer: Optional[Offer] = None
        self._concessions_made: List[float] = []
        self._value_discovered: float = 0.0

    def remember_offer(self, offer: Offer) -> None:
        """Store an offer in structured memory."""
        self._offer_history.append(offer)

        # Also store in associative memory for retrieval
        memory_text = (
            f"Negotiation offer from {offer.offerer} to {offer.recipient}: "
            f"{offer.content}"
        )
        if offer.value is not None:
            memory_text += f" (value: {offer.value})"

        # Basic associative memory doesn't support metadata
        self._memory.add(text=memory_text)

        # Update current best if applicable
        if offer.recipient == self._agent_name and offer.value:
            if (self._current_best_offer is None or
                offer.value > self._current_best_offer.value):
                self._current_best_offer = offer

    def remember_outcome(self, outcome: NegotiationOutcome) -> None:
        """Store negotiation outcome for learning."""
        self._negotiation_outcomes.append(outcome)

        # Store in associative memory
        outcome_text = (
            f"Negotiation outcome: {'Agreement' if outcome.agreement_reached else 'No agreement'}. "
            f"Participants: {', '.join(outcome.participants)}. "
            f"Rounds: {outcome.rounds_taken}. "
            f"{outcome.summary}"
        )

        # Basic associative memory doesn't support metadata
        self._memory.add(text=outcome_text)

        # Learn patterns from outcome
        self._extract_patterns(outcome)

    def _extract_patterns(self, outcome: NegotiationOutcome) -> None:
        """Extract learnable patterns from negotiation outcome."""
        # Track success rates by strategy
        if outcome.agreement_reached:
            strategy_key = f"rounds_to_agreement"
            if strategy_key not in self._patterns_learned:
                self._patterns_learned[strategy_key] = []
            self._patterns_learned[strategy_key].append(outcome.rounds_taken)

    def get_similar_negotiations(self, context: str, limit: int = 3) -> List[str]:
        """Retrieve similar past negotiations."""
        # Search for similar negotiation contexts
        # Note: Basic associative memory doesn't support metadata filtering
        # We'll retrieve more results and filter manually
        results = self._memory.retrieve_associative(
            query=context,
            k=limit * 3  # Get more results to filter
        )

        # Results are already strings
        return list(results)[:limit]

    def get_successful_patterns(self) -> str:
        """Retrieve patterns from successful negotiations."""
        successful_outcomes = [
            o for o in self._negotiation_outcomes
            if o.agreement_reached
        ]

        if not successful_outcomes:
            return "No successful negotiations yet to learn from."

        # Analyze patterns
        avg_rounds = sum(o.rounds_taken for o in successful_outcomes) / len(successful_outcomes)

        patterns = f"Successful negotiation patterns:\n"
        patterns += f"- Average rounds to agreement: {avg_rounds:.1f}\n"
        patterns += f"- Success rate: {len(successful_outcomes)}/{len(self._negotiation_outcomes)}\n"

        return patterns

    def get_current_negotiation_summary(self) -> str:
        """Summarize current negotiation state."""
        if not self._offer_history:
            return "No offers exchanged yet."

        summary = f"Negotiation summary:\n"
        summary += f"- Offers exchanged: {len(self._offer_history)}\n"

        if self._current_best_offer:
            summary += f"- Best offer received: {self._current_best_offer.value}\n"

        if self._concessions_made:
            total_concession = sum(self._concessions_made)
            summary += f"- Total concessions made: {total_concession}\n"

        # Recent offer trend
        if len(self._offer_history) >= 2:
            last_two = self._offer_history[-2:]
            if all(o.value is not None for o in last_two):
                trend = last_two[1].value - last_two[0].value
                summary += f"- Recent trend: {'improving' if trend > 0 else 'declining'}\n"

        return summary

    def pre_act(self, action_spec) -> str:
        """Provide negotiation memory context before action."""
        context = "NEGOTIATION MEMORY:\n"

        # Current negotiation state
        context += self.get_current_negotiation_summary() + "\n"

        # Relevant past patterns
        context += self.get_successful_patterns() + "\n"

        # Similar past negotiations
        similar = self.get_similar_negotiations("current negotiation", limit=2)
        if similar:
            context += "Similar past negotiations:\n"
            for s in similar:
                context += f"- {s}\n"

        return context

    def post_act(self, action_attempt: str) -> str:
        """Update memory after action."""
        # Parse if action was an offer
        if 'offer' in action_attempt.lower():
            # Simple parsing - in real implementation would be more sophisticated
            offer = Offer(
                offerer=self._agent_name,
                recipient='other_party',  # Would parse from context
                content=action_attempt,
                round_number=len(self._offer_history) + 1,
            )
            self.remember_offer(offer)
        return ""

    def pre_observe(self, observation: str) -> str:
        """Process negotiation observations."""
        # Parse if observation contains an offer
        if 'offer' in observation.lower():
            # Simple parsing for now
            offer = Offer(
                offerer='other_party',  # Would parse from observation
                recipient=self._agent_name,
                content=observation,
                round_number=len(self._offer_history) + 1,
            )
            self.remember_offer(offer)

        # Detect negotiation end
        if any(phrase in observation.lower() for phrase in ['deal', 'agree', 'accept']):
            outcome = NegotiationOutcome(
                agreement_reached=True,
                final_value=self._current_best_offer.value if self._current_best_offer else None,
                rounds_taken=len(self._offer_history),
                participants=list(set([o.offerer for o in self._offer_history] +
                                     [o.recipient for o in self._offer_history])),
                summary=observation,
            )
            self.remember_outcome(outcome)
        return ""

    def post_observe(self) -> str:
        """Post-observation processing."""
        return ""

    def update(self) -> None:
        """Update internal state."""
        # Could implement memory consolidation or pattern learning here
        pass

    @property
    def name(self) -> str:
        """Component name."""
        return 'NegotiationMemory'

    def get_state(self) -> str:
        """Get the component state for saving/restoring."""
        # For now, return empty state. Could serialize offer history if needed.
        return ''

    def set_state(self, state: str) -> None:
        """Set the component state from a saved string."""
        # For now, do nothing. Could deserialize offer history if needed.
        pass
