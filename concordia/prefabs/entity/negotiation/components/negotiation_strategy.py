"""Basic negotiation strategy component."""

import abc
import dataclasses
from typing import Dict, List, Optional, Tuple

from concordia.typing import entity_component

from concordia.prefabs.entity.negotiation.config import StrategyConfig


@dataclasses.dataclass
class StrategyState:
    """Current state of negotiation from strategic perspective."""
    current_position: float = 0.0
    opponent_position: float = 0.0
    rounds_elapsed: int = 0
    concessions_made: List[float] = dataclasses.field(default_factory=list)
    zone_of_agreement: Optional[Tuple[float, float]] = None
    negotiation_temperature: float = 1.0  # High = early, Low = late stage


class NegotiationStrategy(abc.ABC):
    """Abstract base class for negotiation strategies."""

    @abc.abstractmethod
    def get_opening_position(self, reservation_value: float, target_value: float) -> float:
        """Determine opening offer position."""
        pass

    @abc.abstractmethod
    def calculate_concession(self, state: StrategyState) -> float:
        """Calculate next concession amount."""
        pass

    @abc.abstractmethod
    def should_accept_offer(self, offer: float, state: StrategyState) -> bool:
        """Decide whether to accept current offer."""
        pass

    @abc.abstractmethod
    def get_tactical_guidance(self, state: StrategyState) -> str:
        """Provide tactical guidance for current situation."""
        pass


class CooperativeStrategy(NegotiationStrategy):
    """Cooperative negotiation strategy focused on mutual gains."""

    def get_opening_position(self, reservation_value: float, target_value: float) -> float:
        """Start with reasonable position showing good faith."""
        # Open at 70% of the distance between reservation and target
        return reservation_value + 0.7 * (target_value - reservation_value)

    def calculate_concession(self, state: StrategyState) -> float:
        """Make steady, predictable concessions to build trust."""
        if state.rounds_elapsed < 3:
            # Larger early concessions to signal cooperation
            return 0.15 * (state.current_position - state.opponent_position)
        else:
            # Smaller but consistent concessions
            return 0.10 * (state.current_position - state.opponent_position)

    def should_accept_offer(self, offer: float, state: StrategyState) -> bool:
        """Accept if offer is reasonable and shows good faith."""
        # Accept if within threshold of our current position
        return offer >= StrategyConfig.COOPERATIVE_ACCEPTANCE_THRESHOLD * state.current_position

    def get_tactical_guidance(self, state: StrategyState) -> str:
        """Provide cooperative tactical guidance."""
        guidance = "COOPERATIVE TACTICS:\n"

        if state.rounds_elapsed < 3:
            guidance += "- Make meaningful concessions to build trust\n"
            guidance += "- Share information about your priorities\n"
            guidance += "- Acknowledge the other party's interests\n"
        elif state.rounds_elapsed < 8:
            guidance += "- Continue steady concessions\n"
            guidance += "- Explore creative solutions\n"
            guidance += "- Suggest package deals\n"
        else:
            guidance += "- Work toward final agreement\n"
            guidance += "- Emphasize mutual benefits\n"
            guidance += "- Ensure both parties feel satisfied\n"

        return guidance


class CompetitiveStrategy(NegotiationStrategy):
    """Competitive negotiation strategy focused on value claiming."""

    def get_opening_position(self, reservation_value: float, target_value: float) -> float:
        """Start with ambitious position to anchor high."""
        # Open at 120% of target value
        return target_value * 1.2

    def calculate_concession(self, state: StrategyState) -> float:
        """Make minimal concessions, decreasing over time."""
        base_concession = StrategyConfig.BASE_CONCESSION_RATE * (state.current_position - state.opponent_position)

        # Reduce concessions as negotiation progresses
        time_factor = max(0.3, 1.0 - (state.rounds_elapsed / 20))

        return base_concession * time_factor

    def should_accept_offer(self, offer: float, state: StrategyState) -> bool:
        """Only accept if offer meets high standards."""
        # Accept only if within competitive threshold of current position
        return offer >= StrategyConfig.COMPETITIVE_ACCEPTANCE_THRESHOLD * state.current_position

    def get_tactical_guidance(self, state: StrategyState) -> str:
        """Provide competitive tactical guidance."""
        guidance = "COMPETITIVE TACTICS:\n"

        if state.rounds_elapsed < 3:
            guidance += "- Anchor high with ambitious opening\n"
            guidance += "- Emphasize your alternatives (BATNA)\n"
            guidance += "- Make them make the first concession\n"
        elif state.rounds_elapsed < 8:
            guidance += "- Concede slowly and reluctantly\n"
            guidance += "- Extract value for each concession\n"
            guidance += "- Use time pressure to your advantage\n"
        else:
            guidance += "- Hold firm near your target\n"
            guidance += "- Be willing to walk away\n"
            guidance += "- Make 'final' offers credible\n"

        return guidance


class IntegrativeStrategy(NegotiationStrategy):
    """Integrative negotiation strategy focused on expanding value."""

    def get_opening_position(self, reservation_value: float, target_value: float) -> float:
        """Start with exploratory position to enable value creation."""
        # Open at 85% of distance to target
        return reservation_value + 0.85 * (target_value - reservation_value)

    def calculate_concession(self, state: StrategyState) -> float:
        """Make strategic concessions tied to value creation."""
        if state.zone_of_agreement:
            # If we've identified ZOPA, move toward middle
            zopa_middle = sum(state.zone_of_agreement) / 2
            return 0.2 * (state.current_position - zopa_middle)
        else:
            # Exploratory concessions to find ZOPA
            return 0.1 * (state.current_position - state.opponent_position)

    def should_accept_offer(self, offer: float, state: StrategyState) -> bool:
        """Accept if offer represents good integrated value."""
        # Consider total value created, not just distribution
        if state.zone_of_agreement:
            zopa_middle = sum(state.zone_of_agreement) / 2
            # Accept if reasonably close to middle of ZOPA
            return abs(offer - zopa_middle) < 0.2 * (state.zone_of_agreement[1] - state.zone_of_agreement[0])
        else:
            # Standard acceptance criteria
            return offer >= StrategyConfig.INTEGRATIVE_ACCEPTANCE_THRESHOLD * state.current_position

    def get_tactical_guidance(self, state: StrategyState) -> str:
        """Provide integrative tactical guidance."""
        guidance = "INTEGRATIVE TACTICS:\n"

        if state.rounds_elapsed < 3:
            guidance += "- Ask questions to understand their interests\n"
            guidance += "- Identify all negotiable issues\n"
            guidance += "- Look for different valuations to trade on\n"
        elif state.rounds_elapsed < 8:
            guidance += "- Propose creative package deals\n"
            guidance += "- Suggest conditional agreements\n"
            guidance += "- Find ways to expand the pie\n"
        else:
            guidance += "- Optimize the total value created\n"
            guidance += "- Ensure fair distribution\n"
            guidance += "- Document all aspects of complex deal\n"

        return guidance


class BasicNegotiationStrategy(entity_component.ContextComponent):
    """Component that provides basic negotiation strategies.

    This component:
    - Tracks negotiation state
    - Provides strategic guidance based on chosen style
    - Calculates appropriate concessions
    - Advises on offer acceptance
    """

    def __init__(
        self,
        agent_name: str,
        negotiation_style: str = 'integrative',
        reservation_value: float = 0.0,
        target_value: float = 100.0,
        verbose: bool = False,
    ):
        """Initialize negotiation strategy component.

        Args:
            agent_name: Name of the agent
            negotiation_style: One of 'cooperative', 'competitive', 'integrative'
            reservation_value: Minimum acceptable value (BATNA)
            target_value: Ideal outcome value
            verbose: Whether to print debug information
        """
        self._agent_name = agent_name
        self._style = negotiation_style
        self._reservation_value = reservation_value
        self._target_value = target_value
        self._verbose = verbose

        # Select strategy implementation
        if negotiation_style == 'cooperative':
            self._strategy = CooperativeStrategy()
        elif negotiation_style == 'competitive':
            self._strategy = CompetitiveStrategy()
        else:
            self._strategy = IntegrativeStrategy()

        # Initialize state
        self._state = StrategyState()
        self._state.current_position = self._strategy.get_opening_position(
            reservation_value, target_value
        )

        # Track opponent offers and initial targets
        self._last_opponent_offer: Optional[float] = None
        self._initial_target: float = target_value

    def update_state(self, opponent_offer: Optional[float] = None) -> None:
        """Update strategic state based on negotiation progress.

        Note: rounds_elapsed is incremented in post_act(), not here.
        """
        if opponent_offer is not None:
            self._state.opponent_position = opponent_offer

            # Try to infer ZOPA
            if self._state.opponent_position < self._state.current_position:
                # They're below us, so ZOPA might be between their position and our reservation
                self._state.zone_of_agreement = (
                    max(self._state.opponent_position, self._reservation_value),
                    self._state.current_position
                )

    def get_strategic_context(self) -> str:
        """Get current strategic context and guidance."""
        context = f"NEGOTIATION STRATEGY ({self._style.upper()}):\n\n"

        # Current positions
        context += f"Your current position: {self._state.current_position:.2f}\n"
        context += f"Your reservation value: {self._reservation_value:.2f}\n"
        context += f"Your target value: {self._target_value:.2f}\n"

        if self._state.opponent_position > 0:
            context += f"Opponent's last position: {self._state.opponent_position:.2f}\n"

        if self._state.zone_of_agreement:
            context += f"Estimated ZOPA: {self._state.zone_of_agreement[0]:.2f} - {self._state.zone_of_agreement[1]:.2f}\n"

        context += f"\nRounds elapsed: {self._state.rounds_elapsed}\n"
        context += f"Negotiation temperature: {self._state.negotiation_temperature:.2f}\n"

        # Strategic guidance
        context += "\n" + self._strategy.get_tactical_guidance(self._state)

        # Concession guidance
        if self._state.opponent_position > 0:
            suggested_concession = self._strategy.calculate_concession(self._state)
            new_position = self._state.current_position - suggested_concession

            if new_position >= self._reservation_value:
                context += f"\nSuggested next offer: {new_position:.2f} (concession of {suggested_concession:.2f})\n"
            else:
                context += f"\nWARNING: Further concessions would go below reservation value!\n"

        return context

    def pre_act(self, action_spec) -> str:
        """Provide strategic context before action."""
        return self.get_strategic_context()

    def post_act(self, action_attempt: str) -> str:
        """Update after action."""
        # Always increment rounds on each action
        self._state.rounds_elapsed += 1
        self._state.negotiation_temperature = max(0.1, 1.0 - (self._state.rounds_elapsed / 20))
        return ""

    def pre_observe(self, observation: str) -> str:
        """Process strategic observations."""
        # Enhanced parsing to detect opponent offers and extract values
        if 'offer' in observation.lower():
            parsed_value = self._parse_offer_value(observation)
            if parsed_value is not None:
                self._last_opponent_offer = parsed_value
                # Adjust our strategy based on the offer
                self._adjust_strategy_for_offer(parsed_value)
            self.update_state()
        return ""
    
    def _parse_offer_value(self, text: str) -> Optional[float]:
        """Parse monetary values from text."""
        import re
        
        # Look for currency amounts like $150, 150.00, USD 150, etc.
        patterns = [
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $150, $1,200.50
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|USD|\$)',  # 150 dollars, 150 USD
            r'(?:USD|dollars?)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # USD 150
            r'\b(\d+(?:,\d{3})*(?:\.\d{2})?)\b',  # Plain numbers as fallback
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the first match and clean it
                value_str = matches[0].replace(',', '')
                try:
                    return float(value_str)
                except ValueError:
                    continue
        
        return None
    
    def _adjust_strategy_for_offer(self, offer_value: float) -> None:
        """Adjust strategy based on opponent's offer."""
        # If offer is better than our reservation, become more cooperative
        if ((self._style == 'competitive' and offer_value > self._reservation_value) or
            (self._style == 'cooperative' and offer_value >= self._reservation_value * 0.9)):
            # Adjust target to be more reasonable
            gap = abs(self._target_value - offer_value)
            self._target_value = offer_value + (gap * 0.3)
            
        # If offer is much worse than expected, become more firm
        elif offer_value < self._reservation_value * 0.8:
            # Don't adjust target downward too much
            self._target_value = max(self._target_value, self._reservation_value * 1.1)

    def post_observe(self) -> str:
        """Post-observation processing."""
        return ""

    def update(self) -> None:
        """Update internal state."""
        pass

    @property
    def name(self) -> str:
        """Component name."""
        return 'BasicNegotiationStrategy'

    def get_state(self) -> str:
        """Get the component state for saving/restoring."""
        return f'{self._state.current_position}|{self._state.rounds_elapsed}'

    def set_state(self, state: str) -> None:
        """Set the component state from a saved string."""
        if '|' in state:
            position, rounds = state.split('|', 1)
            self._state.current_position = float(position)
            self._state.rounds_elapsed = int(rounds)
