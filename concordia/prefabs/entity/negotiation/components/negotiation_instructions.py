
"""Negotiation-specific instructions component."""

from typing import Optional

from concordia.typing import entity_component


class NegotiationInstructions(entity_component.ContextComponent):
    """Instructions component specialized for negotiation contexts.

    This component provides dynamic negotiation guidance based on:
    - Current negotiation phase (opening, middle, closing)
    - Active negotiation style
    - Ethical constraints
    - Situational factors
    """

    def __init__(
        self,
        agent_name: str,
        goal: str,
        negotiation_style: str = 'integrative',
        reservation_value: float = 0.0,
        ethical_constraints: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize negotiation instructions.

        Args:
            agent_name: Name of the agent
            goal: Primary negotiation objective
            negotiation_style: One of 'cooperative', 'competitive', 'integrative'
            reservation_value: Minimum acceptable value (BATNA)
            ethical_constraints: Optional ethical guidelines
            verbose: Whether to print debug information
        """
        self._agent_name = agent_name
        self._goal = goal
        self._style = negotiation_style
        self._reservation_value = reservation_value
        self._ethics = ethical_constraints or 'Be honest and fair. Do not deceive.'
        self._verbose = verbose

        # Track negotiation state
        self._negotiation_phase = 'opening'
        self._rounds_completed = 0
        self._last_offer_made = None
        self._last_offer_received = None

        # Base instructions
        self._base_instructions = self._generate_base_instructions()

    def _generate_base_instructions(self) -> str:
        """Generate base negotiation instructions."""
        instructions = (
            f'You are {self._agent_name}, a skilled negotiator.\n\n'
            f'PRIMARY GOAL: {self._goal}\n'
            f'NEGOTIATION STYLE: {self._style}\n'
            f'RESERVATION VALUE: {self._reservation_value} (your minimum acceptable outcome)\n'
            f'ETHICS: {self._ethics}\n\n'
            'CORE NEGOTIATION PRINCIPLES:\n'
            '1. Know your BATNA and reservation value at all times\n'
            '2. Understand interests behind positions\n'
            '3. Create value before claiming it\n'
            '4. Maintain professionalism and build rapport\n'
            '5. Document offers clearly: "I offer X for Y"\n\n'
        )

        # Add style-specific guidance
        if self._style == 'cooperative':
            instructions += (
                'COOPERATIVE STYLE GUIDANCE:\n'
                '- Build trust through transparency\n'
                '- Share information to find mutual gains\n'
                '- Make fair and reasonable offers\n'
                '- Focus on long-term relationships\n'
                '- Acknowledge the other party\'s needs\n\n'
            )
        elif self._style == 'competitive':
            instructions += (
                'COMPETITIVE STYLE GUIDANCE:\n'
                '- Maximize your own value capture\n'
                '- Control information strategically\n'
                '- Start with ambitious positions\n'
                '- Use leverage when available\n'
                '- Be willing to walk away\n\n'
            )
        else:  # integrative
            instructions += (
                'INTEGRATIVE STYLE GUIDANCE:\n'
                '- Expand the pie before dividing it\n'
                '- Explore all issues and priorities\n'
                '- Find creative win-win solutions\n'
                '- Trade on different valuations\n'
                '- Ask "What if..." questions\n\n'
            )

        return instructions

    def get_phase_specific_guidance(self) -> str:
        """Get guidance specific to current negotiation phase."""
        if self._negotiation_phase == 'opening':
            return (
                'OPENING PHASE:\n'
                '- Build rapport and establish communication norms\n'
                '- Explore interests and priorities\n'
                '- Set collaborative or competitive tone\n'
                '- Make or solicit initial offers carefully\n'
            )
        elif self._negotiation_phase == 'middle':
            return (
                'MIDDLE PHASE:\n'
                '- Exchange offers and counteroffers\n'
                '- Look for trade-offs and package deals\n'
                '- Test different options and scenarios\n'
                '- Build on areas of agreement\n'
            )
        else:  # closing
            return (
                'CLOSING PHASE:\n'
                '- Finalize remaining issues\n'
                '- Ensure mutual understanding\n'
                '- Document the agreement clearly\n'
                '- Preserve relationship for future\n'
            )

    def pre_act(self, action_spec) -> str:
        """Provide negotiation instructions before action."""
        # Update phase based on rounds
        if self._rounds_completed < 3:
            self._negotiation_phase = 'opening'
        elif self._rounds_completed < 10:
            self._negotiation_phase = 'middle'
        else:
            self._negotiation_phase = 'closing'

        # Build contextual instructions
        instructions = self._base_instructions
        instructions += self.get_phase_specific_guidance()

        # Add situational guidance
        if self._last_offer_received:
            instructions += f'\nLAST OFFER RECEIVED: {self._last_offer_received}\n'
            instructions += 'Consider: Is this above your reservation value? '
            instructions += 'Can you find creative ways to improve the deal?\n'

        if self._last_offer_made:
            instructions += f'\nYOUR LAST OFFER: {self._last_offer_made}\n'

        # Add tactical reminders
        instructions += '\nREMEMBER:\n'
        if self._style == 'cooperative':
            instructions += '- Share information to build trust\n'
        elif self._style == 'competitive':
            instructions += '- Every concession should get something in return\n'
        else:
            instructions += '- Look for ways to expand value for both parties\n'

        return instructions

    def post_act(self, action_attempt: str) -> str:
        """Update state after action."""
        # Always increment round counter on each action
        self._rounds_completed += 1

        # Track if we made an offer
        if 'offer' in action_attempt.lower():
            self._last_offer_made = action_attempt

        if self._verbose:
            print(f'[{self._agent_name}] Negotiation round {self._rounds_completed}')
        return ""

    def pre_observe(self, observation: str) -> str:
        """Process incoming observations."""
        # Track if we received an offer
        if 'offer' in observation.lower():
            self._last_offer_received = observation
        return ""

    def post_observe(self) -> str:
        """Post-observation processing."""
        return ""

    def update(self) -> None:
        """Update internal state."""
        pass

    @property
    def name(self) -> str:
        """Component name."""
        return 'NegotiationInstructions'

    def get_state(self) -> str:
        """Get the component state for saving/restoring."""
        return f'{self._negotiation_phase}|{self._rounds_completed}'

    def set_state(self, state: str) -> None:
        """Set the component state from a saved string."""
        if '|' in state:
            phase, rounds = state.split('|', 1)
            self._negotiation_phase = phase
            self._rounds_completed = int(rounds)
