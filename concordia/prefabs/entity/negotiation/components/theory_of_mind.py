"""Theory of mind component for emotional intelligence and recursive reasoning."""

import dataclasses
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import deque

from concordia.typing import entity_component
from concordia.typing import entity as entity_lib
from concordia.prefabs.entity.negotiation.config import (
    DeceptionDetectionConfig,
    TheoryOfMindConfig,
)


@dataclasses.dataclass
class EmotionalState:
    """Represents the emotional state of a negotiation participant."""
    emotions: Dict[str, float]  # emotion_name -> intensity (0-1)
    valence: float  # Overall positive/negative affect (-1 to 1)
    arousal: float  # Activation level (0-1)
    confidence: float  # How certain we are about this assessment (0-1)
    timestamp: str
    triggers: List[str] = dataclasses.field(default_factory=list)

    def dominant_emotion(self) -> Tuple[str, float]:
        """Get the strongest emotion and its intensity."""
        if not self.emotions:
            return "neutral", 0.0
        return max(self.emotions.items(), key=lambda x: x[1])

    def emotional_intensity(self) -> float:
        """Calculate overall emotional intensity."""
        return max(self.emotions.values()) if self.emotions else 0.0


@dataclasses.dataclass
class MentalModel:
    """Mental model of a counterpart's psychological state."""
    counterpart_id: str
    goals: Dict[str, float]  # goal -> probability
    personality_traits: Dict[str, float]  # trait -> score (0-1)
    emotional_state: EmotionalState
    strategies: Dict[str, float]  # strategy -> likelihood
    constraints: List[str]
    deception_indicators: Dict[str, float]
    last_updated: str


@dataclasses.dataclass
class RecursiveBelief:
    """Represents beliefs at different levels of recursion."""
    level: int  # 0=direct, 1=first-order, 2=second-order, etc.
    believer: str  # Who holds this belief
    content: Dict[str, Any]  # What is believed
    confidence: float  # Certainty level (0-1)
    evidence: List[str]  # Supporting observations


class TheoryOfMind(entity_component.ContextComponent):
    """Component for theory of mind and emotional intelligence in negotiations."""

    def __init__(
        self,
        model: Any,
        max_recursion_depth: int = 3,
        emotion_sensitivity: float = 0.7,
        empathy_level: float = 0.8,
    ):
        """Initialize theory of mind component.

        Args:
            model: Language model for analysis
            max_recursion_depth: Maximum levels of recursive reasoning
            emotion_sensitivity: Sensitivity to emotional cues (0-1)
            empathy_level: Level of empathic responding (0-1)
        """
        self._model = model
        self._max_recursion_depth = max_recursion_depth
        self._emotion_sensitivity = emotion_sensitivity
        self._empathy_level = empathy_level

        # Mental models of counterparts
        self._mental_models: Dict[str, MentalModel] = {}

        # Recursive belief hierarchy
        self._belief_hierarchy: Dict[int, List[RecursiveBelief]] = {}

        # Emotional intelligence state
        self._emotion_history: deque = deque(maxlen=TheoryOfMindConfig.EMOTION_HISTORY_SIZE)
        self._empathy_strategies: Dict[str, str] = {
            'frustration': 'validation_and_problem_solving',
            'anxiety': 'reassurance_and_clarity',
            'anger': 'de_escalation_and_perspective_taking',
            'excitement': 'enthusiasm_matching',
            'disappointment': 'understanding_and_alternative_solutions',
            'confusion': 'clarification_and_simplification'
        }

        # Deception detection
        self._baseline_patterns: Dict[str, float] = {}
        self._deception_indicators: List[str] = []

    def _detect_emotions(self, communication: str) -> EmotionalState:
        """Detect emotional state from communication."""
        prompt = f"""Analyze the emotional content of this negotiation communication:

Communication: {communication}

Assess the following emotions on a scale of 0.0 to 1.0:
- Anger/Frustration
- Fear/Anxiety
- Joy/Satisfaction
- Sadness/Disappointment
- Surprise
- Trust/Confidence
- Anticipation/Excitement

Also assess:
- Overall valence (positive/negative): -1.0 to 1.0
- Arousal level (calm/excited): 0.0 to 1.0
- Confidence in assessment: 0.0 to 1.0

Format: anger:X.X fear:X.X joy:X.X sadness:X.X surprise:X.X trust:X.X anticipation:X.X valence:X.X arousal:X.X confidence:X.X"""

        response = self._model.sample_text(prompt)

        # Parse emotional assessment
        emotions = {
            'anger': 0.0, 'fear': 0.0, 'joy': 0.0, 'sadness': 0.0,
            'surprise': 0.0, 'trust': 0.0, 'anticipation': 0.0
        }
        valence = 0.0
        arousal = 0.0
        confidence = 0.5

        # Extract values from response
        for line in response.split('\n'):
            line = line.lower().strip()
            for emotion in emotions.keys():
                if f"{emotion}:" in line:
                    try:
                        value = float(line.split(':')[1].strip())
                        emotions[emotion] = max(0.0, min(1.0, value))
                    except (ValueError, IndexError):
                        pass

            if "valence:" in line:
                try:
                    valence = float(line.split(':')[1].strip())
                    valence = max(-1.0, min(1.0, valence))
                except (ValueError, IndexError):
                    pass

            if "arousal:" in line:
                try:
                    arousal = float(line.split(':')[1].strip())
                    arousal = max(0.0, min(1.0, arousal))
                except (ValueError, IndexError):
                    pass

            if "confidence:" in line:
                try:
                    confidence = float(line.split(':')[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, IndexError):
                    pass

        # Identify emotional triggers
        triggers = self._identify_emotional_triggers(communication, emotions)

        return EmotionalState(
            emotions=emotions,
            valence=valence,
            arousal=arousal,
            confidence=confidence,
            timestamp="current",
            triggers=triggers
        )

    def _identify_emotional_triggers(self, communication: str, emotions: Dict[str, float]) -> List[str]:
        """Identify what triggered the detected emotions."""
        triggers = []

        # Common emotional triggers in negotiations
        trigger_patterns = {
            'deadline_pressure': ['deadline', 'time', 'urgent', 'quickly'],
            'price_concern': ['expensive', 'cost', 'budget', 'afford'],
            'trust_issue': ['promise', 'guarantee', 'reliable', 'honest'],
            'complexity': ['complicated', 'complex', 'difficult', 'confusing'],
            'progress': ['progress', 'moving forward', 'agreement', 'solution']
        }

        communication_lower = communication.lower()
        for trigger, keywords in trigger_patterns.items():
            if any(keyword in communication_lower for keyword in keywords):
                triggers.append(trigger)

        return triggers

    def _infer_goals(self, observations: List[str], behaviors: List[str]) -> Dict[str, float]:
        """Infer counterpart's goals from observations and behaviors."""
        prompt = f"""Analyze these negotiation observations to infer the counterpart's likely goals:

Observations: {observations}
Behaviors: {behaviors}

Rate the likelihood (0.0 to 1.0) that the counterpart has these goals:
- Maximize financial value
- Minimize costs/risks
- Build long-term relationship
- Establish favorable precedent
- Gain market information
- Save time/expedite process
- Maintain reputation
- Avoid conflict/confrontation
- Learn about our capabilities
- Test our flexibility

Format: financial:X.X costs:X.X relationship:X.X precedent:X.X information:X.X time:X.X reputation:X.X conflict:X.X capabilities:X.X flexibility:X.X"""

        response = self._model.sample_text(prompt)

        goals = {
            'financial': 0.5, 'costs': 0.5, 'relationship': 0.5, 'precedent': 0.5,
            'information': 0.5, 'time': 0.5, 'reputation': 0.5, 'conflict': 0.5,
            'capabilities': 0.5, 'flexibility': 0.5
        }

        # Parse goal likelihoods
        for line in response.split('\n'):
            line = line.lower().strip()
            for goal in goals.keys():
                if f"{goal}:" in line:
                    try:
                        value = float(line.split(':')[1].strip())
                        goals[goal] = max(0.0, min(1.0, value))
                    except (ValueError, IndexError):
                        pass

        return goals

    def _assess_personality(self, communication_patterns: List[str]) -> Dict[str, float]:
        """Assess personality traits from communication patterns."""
        prompt = f"""Analyze these communication patterns to assess personality traits:

Communication patterns: {communication_patterns}

Rate each Big Five personality trait (0.0 to 1.0):
- Openness: creativity, openness to new experiences
- Conscientiousness: organization, attention to detail
- Extraversion: social energy, assertiveness
- Agreeableness: cooperation, empathy
- Neuroticism: emotional instability, anxiety

Also assess negotiation-specific traits:
- Risk tolerance: willingness to take risks
- Patience: tolerance for lengthy processes
- Assertiveness: directness in communication
- Analytical thinking: focus on data and logic

Format: openness:X.X conscientiousness:X.X extraversion:X.X agreeableness:X.X neuroticism:X.X risk_tolerance:X.X patience:X.X assertiveness:X.X analytical:X.X"""

        response = self._model.sample_text(prompt)

        traits = {
            'openness': 0.5, 'conscientiousness': 0.5, 'extraversion': 0.5,
            'agreeableness': 0.5, 'neuroticism': 0.5, 'risk_tolerance': 0.5,
            'patience': 0.5, 'assertiveness': 0.5, 'analytical': 0.5
        }

        # Parse personality scores
        for line in response.split('\n'):
            line = line.lower().strip()
            for trait in traits.keys():
                if f"{trait}:" in line:
                    try:
                        value = float(line.split(':')[1].strip())
                        traits[trait] = max(0.0, min(1.0, value))
                    except (ValueError, IndexError):
                        pass

        return traits

    def _detect_deception(self, statement: str, baseline_patterns: Dict[str, float]) -> Dict[str, float]:
        """Detect potential deception indicators in statements.

        Uses configurable multipliers from DeceptionDetectionConfig.
        Based on research:
          - DePaulo et al. (2003): avg effect size d=0.25 for significant cues
          - Hauch et al. (2015): liars NOT more uncertain, but more negative emotion
          - Newman et al. (2003): fewer self-references, more negative words
        """
        deception_indicators = {
            'linguistic_complexity': 0.0,
            'evasiveness': 0.0,
            'inconsistency': 0.0,
            'over_certainty': 0.0,
            'defensive_language': 0.0,
            'negative_emotion': 0.0,
        }

        statement_lower = statement.lower()
        words = statement_lower.split()
        total_words = len(words)

        # Linguistic complexity (overly complex explanations)
        if total_words > 0:
            complex_words = len([
                word for word in words
                if len(word) > DeceptionDetectionConfig.COMPLEX_WORD_MIN_LENGTH
            ])
            complexity_ratio = complex_words / total_words
            deception_indicators['linguistic_complexity'] = min(
                1.0,
                complexity_ratio * DeceptionDetectionConfig.LINGUISTIC_COMPLEXITY_MULTIPLIER
            )

        # Evasiveness patterns (NOTE: Hauch 2015 shows this is WEAK cue)
        evasiveness_count = sum(
            1 for phrase in DeceptionDetectionConfig.EVASIVE_PHRASES
            if phrase in statement_lower
        )
        deception_indicators['evasiveness'] = min(
            1.0,
            evasiveness_count * DeceptionDetectionConfig.EVASIVENESS_MULTIPLIER
        )

        # Over-certainty / truth emphasis ("protest too much" - reliable cue)
        certainty_count = sum(
            1 for phrase in DeceptionDetectionConfig.CERTAINTY_WORDS
            if phrase in statement_lower
        )
        deception_indicators['over_certainty'] = min(
            1.0,
            certainty_count * DeceptionDetectionConfig.OVER_CERTAINTY_MULTIPLIER
        )

        # Defensive language (reliable cue)
        defensive_count = sum(
            1 for phrase in DeceptionDetectionConfig.DEFENSIVE_PHRASES
            if phrase in statement_lower
        )
        deception_indicators['defensive_language'] = min(
            1.0,
            defensive_count * DeceptionDetectionConfig.DEFENSIVE_LANGUAGE_MULTIPLIER
        )

        # Negative emotion words (Hauch 2015: liars express MORE negative emotion)
        negative_count = sum(
            1 for word in DeceptionDetectionConfig.NEGATIVE_EMOTION_WORDS
            if word in words  # Match whole words only
        )
        deception_indicators['negative_emotion'] = min(
            1.0,
            negative_count * DeceptionDetectionConfig.NEGATIVE_EMOTION_MULTIPLIER
        )

        return deception_indicators

    def _build_recursive_beliefs(self, context: str, depth: int = 2) -> Dict[int, List[RecursiveBelief]]:
        """Build recursive belief hierarchy."""
        belief_hierarchy = {}

        for level in range(depth + 1):
            if level == 0:
                # Level 0: Direct beliefs about the situation
                beliefs = [RecursiveBelief(
                    level=0,
                    believer="self",
                    content={"situation_assessment": "analyzing_negotiation_context"},
                    confidence=0.8,
                    evidence=[context]
                )]
            elif level == 1:
                # Level 1: What I think they believe
                beliefs = [RecursiveBelief(
                    level=1,
                    believer="self",
                    content={"counterpart_believes": "we_are_motivated_to_close"},
                    confidence=TheoryOfMindConfig.BASE_BELIEF_CONFIDENCE - TheoryOfMindConfig.BELIEF_CONFIDENCE_DECAY,
                    evidence=["behavioral_observations"]
                )]
            else:
                # Higher levels: What I think they think I believe, etc.
                # Confidence decays with each level but has a floor
                decayed_confidence = TheoryOfMindConfig.BASE_BELIEF_CONFIDENCE - (level * TheoryOfMindConfig.BELIEF_CONFIDENCE_DECAY)
                beliefs = [RecursiveBelief(
                    level=level,
                    believer="self",
                    content={f"level_{level}_belief": "recursive_reasoning"},
                    confidence=max(TheoryOfMindConfig.MIN_BELIEF_CONFIDENCE, decayed_confidence),
                    evidence=["meta_reasoning"]
                )]

            belief_hierarchy[level] = beliefs

        return belief_hierarchy

    def _generate_empathic_response(self, emotional_state: EmotionalState) -> str:
        """Generate empathic response based on detected emotions."""
        dominant_emotion, intensity = emotional_state.dominant_emotion()

        if intensity < TheoryOfMindConfig.LOW_EMOTION_THRESHOLD:
            return "I appreciate your perspective on this."

        empathy_templates = {
            'anger': [
                "I can see this situation is frustrating for you.",
                "I understand your concerns about this approach.",
                "Let's see if we can address what's bothering you about this."
            ],
            'fear': [
                "I sense you have some concerns about moving forward.",
                "Your caution here is completely understandable.",
                "Let's make sure we address any worries you might have."
            ],
            'sadness': [
                "I can see this isn't meeting your expectations.",
                "I understand your disappointment with how this is developing.",
                "Let's see how we can better align with what you're hoping for."
            ],
            'joy': [
                "I'm glad this resonates with you.",
                "I share your enthusiasm about these possibilities.",
                "It's great to see you're excited about this direction."
            ],
            'surprise': [
                "I can see this is different from what you were expecting.",
                "Let me clarify how we arrived at this approach.",
                "I understand this might be unexpected."
            ]
        }

        templates = empathy_templates.get(dominant_emotion, ["I appreciate your perspective."])

        # Select response based on intensity (using configurable thresholds)
        if intensity > TheoryOfMindConfig.HIGH_EMOTION_THRESHOLD:
            return templates[0]  # Strong acknowledgment
        elif intensity > TheoryOfMindConfig.MODERATE_EMOTION_THRESHOLD:
            return templates[1] if len(templates) > 1 else templates[0]
        else:
            return templates[-1]  # Gentler acknowledgment

    def _update_mental_model(self, counterpart_id: str, observations: List[str]) -> None:
        """Update mental model of counterpart."""
        # Detect emotions
        latest_communication = observations[-1] if observations else ""
        emotional_state = self._detect_emotions(latest_communication)

        # Infer goals and assess personality
        goals = self._infer_goals(observations, [])
        personality_traits = self._assess_personality(observations)

        # Detect deception indicators
        deception_indicators = self._detect_deception(latest_communication, self._baseline_patterns)

        # Create or update mental model
        self._mental_models[counterpart_id] = MentalModel(
            counterpart_id=counterpart_id,
            goals=goals,
            personality_traits=personality_traits,
            emotional_state=emotional_state,
            strategies={},  # To be filled based on behavior patterns
            constraints=[],  # To be inferred from statements
            deception_indicators=deception_indicators,
            last_updated="current"
        )

    def _generate_theory_of_mind_guidance(self, context: str) -> str:
        """Generate guidance based on theory of mind analysis."""
        # Analyze context for social and emotional cues
        emotional_state = self._detect_emotions(context)

        # Build recursive beliefs
        self._belief_hierarchy = self._build_recursive_beliefs(context, self._max_recursion_depth)

        # Generate empathic response
        empathic_response = self._generate_empathic_response(emotional_state)

        guidance = f"""🧠 Theory of Mind Analysis

**Emotional Intelligence Assessment:**
• Dominant emotion: {emotional_state.dominant_emotion()[0]} (intensity: {emotional_state.dominant_emotion()[1]:.2f})
• Emotional valence: {emotional_state.valence:.2f} ({'positive' if emotional_state.valence > 0 else 'negative' if emotional_state.valence < 0 else 'neutral'})
• Arousal level: {emotional_state.arousal:.2f} ({'high' if emotional_state.arousal > 0.6 else 'moderate' if emotional_state.arousal > 0.3 else 'low'})
• Assessment confidence: {emotional_state.confidence:.2f}

**Empathic Response Strategy:**
• Recommended approach: {empathic_response}
• Emotional triggers detected: {', '.join(emotional_state.triggers) if emotional_state.triggers else 'None identified'}

**Recursive Reasoning (Theory of Mind):**"""

        for level in self._belief_hierarchy:
            if level == 0:
                guidance += f"\n• Level {level} (Direct): I believe the situation requires careful attention to emotional dynamics"
            elif level == 1:
                guidance += f"\n• Level {level} (First-order): They likely believe I am analyzing their emotional state"
            elif level == 2:
                guidance += f"\n• Level {level} (Second-order): They may think I think they are trying to influence me emotionally"
            else:
                guidance += f"\n• Level {level} (Higher-order): Complex recursive reasoning about mutual mental modeling"

        guidance += f"""

**Social Intelligence Recommendations:**
• Emotional regulation: {'Help them manage emotions' if emotional_state.emotional_intensity() > 0.6 else 'Maintain current emotional tone'}
• Perspective taking: Acknowledge their viewpoint and concerns
• Trust building: {'Address potential trust issues' if any(self._mental_models.values()) and max([sum(m.deception_indicators.values()) for m in self._mental_models.values()], default=0) > 0.5 else 'Continue building rapport'}
• Strategic empathy: Use emotional understanding to find mutual solutions

**Communication Adaptation:**
• Mirror emotional tone while maintaining professionalism
• Validate emotions before addressing content
• Use perspective-taking language ("I can see...", "From your viewpoint...")
• Adjust complexity based on emotional state"""

        return guidance

    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        """Provide theory of mind guidance before action."""
        context = action_spec.call_to_action

        # Generate theory of mind analysis and guidance
        guidance = self._generate_theory_of_mind_guidance(context)

        return f"\n{guidance}"

    def post_act(self, action_attempt: str) -> str:
        """Update theory of mind state after action."""
        # Analyze our own action for emotional impact
        our_emotional_tone = self._detect_emotions(action_attempt)
        self._emotion_history.append(our_emotional_tone)

        return ""

    def pre_observe(self, observation: str) -> str:
        """Called before observation is processed. Store observation for post_observe."""
        self._last_observation = observation
        return ""

    def post_observe(self) -> str:
        """Called after observation is processed. Uses stored observation."""
        observation = getattr(self, '_last_observation', '')
        if not observation:
            return ""

        # Update mental model based on observation
        self._update_mental_model("counterpart", [observation])

        # Detect and store emotional state
        emotional_state = self._detect_emotions(observation)
        self._emotion_history.append(emotional_state)

        # Update deception detection baseline
        if len(self._emotion_history) > TheoryOfMindConfig.MIN_OBSERVATIONS_FOR_BASELINE:
            # Simple baseline: average emotional patterns
            recent_emotions = list(self._emotion_history)[-TheoryOfMindConfig.MIN_OBSERVATIONS_FOR_BASELINE:]
            for emotion in emotional_state.emotions:
                self._baseline_patterns[emotion] = np.mean([es.emotions.get(emotion, 0) for es in recent_emotions])

        return ""

    def observe(self, observation: str) -> None:
        """Process observations for theory of mind insights (legacy method)."""
        self._last_observation = observation
        self.post_observe()

    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        current_mental_models = {}
        for counterpart_id, model in self._mental_models.items():
            current_mental_models[counterpart_id] = {
                'dominant_emotion': model.emotional_state.dominant_emotion()[0],
                'emotion_intensity': model.emotional_state.emotional_intensity(),
                'valence': model.emotional_state.valence,
                'top_goals': sorted(model.goals.items(), key=lambda x: x[1], reverse=True)[:3],
                'personality_summary': {k: v for k, v in model.personality_traits.items() if v > 0.6},
                'deception_risk': sum(model.deception_indicators.values()) / len(model.deception_indicators) if model.deception_indicators else 0.0
            }

        return {
            'mental_models': current_mental_models,
            'recursion_depth': self._max_recursion_depth,
            'emotion_history_length': len(self._emotion_history),
            'empathy_level': self._empathy_level,
            'recent_emotional_trend': self._get_emotional_trend()
        }

    def _get_emotional_trend(self) -> str:
        """Analyze recent emotional trend."""
        if len(self._emotion_history) < TheoryOfMindConfig.MIN_HISTORY_FOR_TREND:
            return "insufficient_data"

        recent_valences = [es.valence for es in list(self._emotion_history)[-TheoryOfMindConfig.MIN_HISTORY_FOR_TREND:]]

        if all(v > TheoryOfMindConfig.POSITIVE_TREND_THRESHOLD for v in recent_valences):
            return "increasingly_positive"
        elif all(v < TheoryOfMindConfig.NEGATIVE_TREND_THRESHOLD for v in recent_valences):
            return "increasingly_negative"
        elif recent_valences[-1] > recent_valences[0]:
            return "improving"
        elif recent_valences[-1] < recent_valences[0]:
            return "declining"
        else:
            return "stable"

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set component state."""
        self._max_recursion_depth = state.get('recursion_depth', 3)
        self._empathy_level = state.get('empathy_level', 0.8)

    def get_action_attempt(
        self,
        context: Any,  # ComponentContextMapping
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        """Generate emotionally intelligent negotiation action based on theory of mind analysis."""
        # Get the context from call_to_action
        situation_context = action_spec.call_to_action
        
        # Analyze emotional state of the situation
        emotional_state = self._detect_emotions(situation_context)
        
        # Generate empathic response if emotions are intense
        empathic_response = ""
        if emotional_state.emotional_intensity() > TheoryOfMindConfig.EMPATHY_TRIGGER_THRESHOLD:
            empathic_response = self._generate_empathic_response(emotional_state)
        
        # Build recursive reasoning about the situation
        self._belief_hierarchy = self._build_recursive_beliefs(situation_context, min(2, self._max_recursion_depth))
        
        # Generate action based on emotional intelligence
        prompt = f"""Based on theory of mind and emotional intelligence analysis, generate a negotiation action:

Situation: {situation_context}

Emotional Analysis:
- Dominant emotion: {emotional_state.dominant_emotion()[0]} (intensity: {emotional_state.dominant_emotion()[1]:.2f})
- Emotional valence: {emotional_state.valence:.2f}
- Triggers detected: {', '.join(emotional_state.triggers) if emotional_state.triggers else 'None'}

Empathic Response Strategy: {empathic_response if empathic_response else 'Maintain neutral professional tone'}

Theory of Mind Insights:
- Level 1 (What they likely believe): They may think I am focused primarily on my own interests
- Level 2 (What they think I think): They might believe I'm trying to read their emotional state

Generate a negotiation action that:
1. Acknowledges any strong emotions detected
2. Demonstrates understanding of their perspective
3. Builds trust through emotional validation
4. Moves the negotiation forward constructively
5. Uses emotionally intelligent language

Action:"""

        response = self._model.sample_text(prompt)
        
        # Clean up the response to extract just the action
        action = response.strip()
        if action.lower().startswith('action:'):
            action = action[7:].strip()
        
        # Add empathic framing if strong emotions detected
        if emotional_state.emotional_intensity() > TheoryOfMindConfig.EMPATHIC_FRAMING_THRESHOLD and empathic_response:
            action = f"{empathic_response} {action}"
        
        return action

    def update(self) -> None:
        """Update theory of mind component."""
        # Decay old emotional assessments
        if len(self._emotion_history) > 10:
            # Keep recent emotional history
            self._emotion_history = deque(list(self._emotion_history)[-10:], maxlen=20)

        # Update mental model confidence based on interaction count
        for model in self._mental_models.values():
            # Gradually increase confidence with more observations
            if hasattr(model, 'observation_count'):
                model.observation_count = getattr(model, 'observation_count', 0) + 1
