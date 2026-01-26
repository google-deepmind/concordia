"""Cultural adaptation component for cross-cultural negotiations."""

import dataclasses
import enum
from typing import Dict, List, Optional, Tuple

from concordia.language_model import language_model
from concordia.typing import entity_component


class CulturalDimension(enum.Enum):
    """Key cultural dimensions that affect negotiation."""
    INDIVIDUALISM_COLLECTIVISM = "individualism_vs_collectivism"
    CONTEXT_LEVEL = "high_vs_low_context"
    POWER_DISTANCE = "power_distance"
    TIME_ORIENTATION = "time_orientation"
    RELATIONSHIP_FOCUS = "relationship_vs_task"


@dataclasses.dataclass
class CulturalProfile:
    """Defines a cultural negotiation profile."""
    name: str
    individualism_score: float  # 0 = collectivist, 1 = individualist
    context_level: float  # 0 = low context, 1 = high context
    power_distance: float  # 0 = low, 1 = high
    time_flexibility: float  # 0 = monochronic, 1 = polychronic
    relationship_importance: float  # 0 = task-focused, 1 = relationship-focused

    # Communication patterns
    directness: float  # 0 = indirect, 1 = direct
    formality: float  # 0 = informal, 1 = formal
    emotional_expression: float  # 0 = neutral, 1 = expressive

    # Negotiation characteristics
    opening_relationship_time: str  # Time to invest in relationship building
    decision_making_style: str  # individual, consultative, consensus
    trust_building_approach: str  # competence-based, relationship-based
    conflict_handling: str  # direct confrontation, indirect/face-saving

    def get_distance_from(self, other: 'CulturalProfile') -> float:
        """Calculate cultural distance from another profile."""
        dimensions = [
            abs(self.individualism_score - other.individualism_score),
            abs(self.context_level - other.context_level),
            abs(self.power_distance - other.power_distance),
            abs(self.time_flexibility - other.time_flexibility),
            abs(self.relationship_importance - other.relationship_importance),
        ]
        return sum(dimensions) / len(dimensions)


# Pre-defined cultural profiles
CULTURAL_PROFILES = {
    'western_business': CulturalProfile(
        name='Western Business (USA/UK)',
        individualism_score=0.9,
        context_level=0.2,
        power_distance=0.3,
        time_flexibility=0.2,
        relationship_importance=0.3,
        directness=0.9,
        formality=0.4,
        emotional_expression=0.5,
        opening_relationship_time='minimal',
        decision_making_style='individual',
        trust_building_approach='competence-based',
        conflict_handling='direct confrontation',
    ),

    'east_asian': CulturalProfile(
        name='East Asian (Japan/China)',
        individualism_score=0.2,
        context_level=0.9,
        power_distance=0.8,
        time_flexibility=0.6,
        relationship_importance=0.9,
        directness=0.1,
        formality=0.9,
        emotional_expression=0.2,
        opening_relationship_time='extensive',
        decision_making_style='consensus',
        trust_building_approach='relationship-based',
        conflict_handling='indirect/face-saving',
    ),

    'middle_eastern': CulturalProfile(
        name='Middle Eastern',
        individualism_score=0.3,
        context_level=0.8,
        power_distance=0.9,
        time_flexibility=0.8,
        relationship_importance=0.9,
        directness=0.4,
        formality=0.8,
        emotional_expression=0.7,
        opening_relationship_time='substantial',
        decision_making_style='hierarchical',
        trust_building_approach='relationship-based',
        conflict_handling='indirect/honor-preserving',
    ),

    'latin_american': CulturalProfile(
        name='Latin American',
        individualism_score=0.3,
        context_level=0.7,
        power_distance=0.7,
        time_flexibility=0.9,
        relationship_importance=0.8,
        directness=0.4,
        formality=0.6,
        emotional_expression=0.9,
        opening_relationship_time='moderate',
        decision_making_style='consultative',
        trust_building_approach='personal-based',
        conflict_handling='indirect/personal',
    ),

    'northern_european': CulturalProfile(
        name='Northern European',
        individualism_score=0.7,
        context_level=0.1,
        power_distance=0.2,
        time_flexibility=0.1,
        relationship_importance=0.2,
        directness=1.0,
        formality=0.3,
        emotional_expression=0.2,
        opening_relationship_time='minimal',
        decision_making_style='consensus',
        trust_building_approach='competence-based',
        conflict_handling='direct/factual',
    ),
}


class CulturalAdaptation(entity_component.ContextComponent):
    """Component that adapts negotiation style to cultural contexts.

    This component:
    - Detects cultural cues from counterpart's communication
    - Adjusts communication style to match or bridge cultural gaps
    - Provides culturally-appropriate negotiation guidance
    - Tracks cultural dynamics throughout negotiation
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        own_culture: str = 'western_business',
        adaptation_level: float = 0.7,
        detect_culture: bool = True,
    ):
        """Initialize cultural adaptation component.

        Args:
            model: Language model for analysis
            own_culture: Agent's base cultural profile
            adaptation_level: How much to adapt (0 = none, 1 = full)
            detect_culture: Whether to auto-detect counterpart's culture
        """
        self._model = model
        self._own_profile = CULTURAL_PROFILES[own_culture]
        self._adaptation_level = adaptation_level
        self._detect_culture = detect_culture

        # State tracking
        self._detected_culture: Optional[str] = None
        self._counterpart_profile: Optional[CulturalProfile] = None
        self._cultural_bridges: List[str] = []
        self._adaptation_history: List[Tuple[str, str]] = []

    def detect_cultural_style(self, communication: str) -> Optional[str]:
        """Detect cultural style from communication patterns."""
        if not self._detect_culture:
            return None

        # Analyze communication for cultural markers
        prompt = f'''Analyze this communication for cultural negotiation style:

Communication: {communication}

Consider these cultural indicators:
- Directness vs indirectness
- Focus on task vs relationship
- Formal vs informal language
- Individual vs group orientation
- Time sensitivity vs flexibility

Which cultural profile best matches:
1. western_business - Direct, task-focused, individualistic
2. east_asian - Indirect, relationship-focused, group-oriented
3. middle_eastern - Formal, hierarchical, honor-focused
4. latin_american - Personal, warm, flexible
5. northern_european - Direct, egalitarian, fact-based

Return only the profile name (e.g., western_business).'''

        response = self._model.sample_text(prompt, max_tokens=20)

        # Validate response
        profile_name = response.strip().lower()
        if profile_name in CULTURAL_PROFILES:
            self._detected_culture = profile_name
            self._counterpart_profile = CULTURAL_PROFILES[profile_name]
            return profile_name

        return None

    def get_adapted_communication_style(self) -> str:
        """Get communication style adapted to counterpart's culture."""
        if not self._counterpart_profile:
            return self._get_own_style()

        # Calculate adaptation based on cultural distance and adaptation level
        distance = self._own_profile.get_distance_from(self._counterpart_profile)
        actual_adaptation = min(distance, self._adaptation_level)

        # Interpolate between own and counterpart styles
        adapted_directness = self._interpolate(
            self._own_profile.directness,
            self._counterpart_profile.directness,
            actual_adaptation
        )

        adapted_formality = self._interpolate(
            self._own_profile.formality,
            self._counterpart_profile.formality,
            actual_adaptation
        )

        # Generate style guidance
        style = f"ADAPTED COMMUNICATION STYLE:\n"

        # Directness
        if adapted_directness < 0.3:
            style += "- Use indirect communication, implications, and suggestions\n"
        elif adapted_directness < 0.7:
            style += "- Balance directness with diplomatic language\n"
        else:
            style += "- Communicate directly and explicitly\n"

        # Formality
        if adapted_formality < 0.3:
            style += "- Keep communication informal and friendly\n"
        elif adapted_formality < 0.7:
            style += "- Maintain professional but approachable tone\n"
        else:
            style += "- Use formal language and proper titles\n"

        # Context level
        if self._counterpart_profile.context_level > 0.6:
            style += "- Pay attention to non-verbal cues and implied meanings\n"
            style += "- Allow for pauses and silence in conversation\n"
        else:
            style += "- Be explicit and detailed in communication\n"
            style += "- Document agreements clearly\n"

        # Relationship focus
        if self._counterpart_profile.relationship_importance > 0.6:
            style += "- Invest time in relationship building\n"
            style += "- Show personal interest and warmth\n"
        else:
            style += "- Focus on business objectives\n"
            style += "- Demonstrate competence and efficiency\n"

        return style

    def get_cultural_bridge_strategies(self) -> List[str]:
        """Get strategies for bridging cultural gaps."""
        if not self._counterpart_profile:
            return []

        strategies = []

        # Time orientation bridge
        if abs(self._own_profile.time_flexibility -
               self._counterpart_profile.time_flexibility) > 0.5:
            if self._counterpart_profile.time_flexibility > 0.5:
                strategies.append(
                    "Allow extra time for relationship building and process"
                )
            else:
                strategies.append(
                    "Respect time constraints while building rapport efficiently"
                )

        # Decision-making bridge
        if (self._own_profile.individualism_score > 0.7 and
            self._counterpart_profile.individualism_score < 0.3):
            strategies.append(
                "Allow time for group consultation and consensus building"
            )
        elif (self._own_profile.individualism_score < 0.3 and
              self._counterpart_profile.individualism_score > 0.7):
            strategies.append(
                "Be prepared for quick individual decisions"
            )

        # Power distance bridge
        if abs(self._own_profile.power_distance -
               self._counterpart_profile.power_distance) > 0.5:
            if self._counterpart_profile.power_distance > 0.7:
                strategies.append(
                    "Show respect for hierarchy and formal protocols"
                )
            else:
                strategies.append(
                    "Engage directly and minimize formality"
                )

        return strategies

    def _interpolate(self, own_value: float, other_value: float,
                     level: float) -> float:
        """Interpolate between own and other cultural values."""
        return own_value + (other_value - own_value) * level

    def _get_own_style(self) -> str:
        """Get communication style for own culture."""
        return f"COMMUNICATION STYLE ({self._own_profile.name}):\n" + \
               f"- Directness level: {self._own_profile.directness:.1f}\n" + \
               f"- Formality level: {self._own_profile.formality:.1f}\n" + \
               f"- Focus: {'relationship' if self._own_profile.relationship_importance > 0.5 else 'task'}\n"

    def pre_act(self, action_spec) -> str:
        """Provide cultural adaptation context before action."""
        context = "CULTURAL ADAPTATION:\n\n"

        # Current cultural context
        if self._detected_culture:
            context += f"Detected culture: {self._counterpart_profile.name}\n"
            context += f"Cultural distance: {self._own_profile.get_distance_from(self._counterpart_profile):.2f}\n\n"
        else:
            context += "No cultural profile detected yet.\n\n"

        # Communication style
        context += self.get_adapted_communication_style() + "\n"

        # Bridge strategies
        bridges = self.get_cultural_bridge_strategies()
        if bridges:
            context += "CULTURAL BRIDGE STRATEGIES:\n"
            for strategy in bridges:
                context += f"- {strategy}\n"
            context += "\n"

        # Specific guidance
        if self._counterpart_profile:
            context += "CULTURAL CONSIDERATIONS:\n"
            context += f"- Opening approach: {self._counterpart_profile.opening_relationship_time} relationship building\n"
            context += f"- Decision style: {self._counterpart_profile.decision_making_style}\n"
            context += f"- Trust building: {self._counterpart_profile.trust_building_approach}\n"
            context += f"- Conflict approach: {self._counterpart_profile.conflict_handling}\n"

        return context

    def post_act(self, action_attempt: str) -> str:
        """Track cultural adaptation in action."""
        self._adaptation_history.append((
            self._detected_culture or 'unknown',
            action_attempt
        ))
        return ""

    def pre_observe(self, observation: str) -> str:
        """Detect cultural cues from observations."""
        # Try to detect culture from substantial communications
        if len(observation) > 100 and 'said:' in observation:
            self.detect_cultural_style(observation)
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
        return 'CulturalAdaptation'

    def get_state(self) -> str:
        """Get component state."""
        return f'{self._detected_culture}|{len(self._adaptation_history)}'

    def set_state(self, state: str) -> None:
        """Set component state."""
        if '|' in state:
            culture, history_len = state.split('|', 1)
            if culture != 'None' and culture in CULTURAL_PROFILES:
                self._detected_culture = culture
                self._counterpart_profile = CULTURAL_PROFILES[culture]
