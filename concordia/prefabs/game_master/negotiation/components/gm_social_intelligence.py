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

"""Social intelligence module for negotiation game master."""

import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

from concordia.prefabs.game_master.negotiation.components import negotiation_modules


@dataclasses.dataclass
class EmotionalReading:
  """Represents an emotional state assessment."""
  participant: str
  primary_emotion: str
  intensity: float  # 0-1
  valence: float  # -1 (negative) to 1 (positive)
  confidence: float  # 0-1
  triggers: List[str]
  round_number: int


@dataclasses.dataclass
class MentalModelSnapshot:
  """Snapshot of one party's mental model of another."""
  modeler: str
  subject: str
  perceived_goals: List[str]
  perceived_constraints: List[str]
  trust_assessment: float  # 0-1
  predicted_strategy: str
  accuracy_estimate: float  # 0-1


@dataclasses.dataclass
class DeceptionIndicator:
  """Tracks potential deception or manipulation."""
  actor: str
  indicator_type: str  # 'inconsistency', 'misdirection', 'withholding'
  description: str
  severity: float  # 0-1
  round_number: int


class SocialIntelligenceGM(negotiation_modules.NegotiationGMModule):
  """GM module for managing emotional and social dynamics."""

  # Emotion categories
  EMOTION_CATEGORIES = {
      'positive': ['happy', 'satisfied', 'hopeful', 'confident', 'excited'],
      'negative': ['frustrated', 'angry', 'disappointed', 'anxious', 'suspicious'],
      'neutral': ['calm', 'patient', 'curious', 'uncertain'],
  }

  # Emotion keywords for detection
  EMOTION_INDICATORS = {
      'frustrated': ['frustrated', 'frustrating', 'difficult', 'stuck', 'impossible', 'unfair'],
      'angry': ['angry', 'mad', 'furious', 'unacceptable', 'outrageous', 'insulting', 'ridiculous'],
      'hopeful': ['hopeful', 'hope', 'optimistic', 'possible', 'opportunity', 'potential'],
      'confident': ['confident', 'certain', 'definitely', 'assured', 'guaranteed'],
      'anxious': ['anxious', 'worried', 'concerned', 'unsure', 'risky'],
      'suspicious': ['suspicious', 'doubt', 'question', 'unclear', 'hidden'],
      'excited': ['excited', 'thrilled', 'enthusiastic'],
      'satisfied': ['satisfied', 'pleased', 'content'],
      'disappointed': ['disappointed', 'let down', 'discouraged'],
  }

  def __init__(
      self,
      name: str = 'social_intelligence',
      priority: int = 90,
      config: Optional[Dict[str, Any]] = None,
  ):
    """Initialize social intelligence module."""
    super().__init__(name, priority, config)

    # Emotional tracking
    self._emotional_history: List[EmotionalReading] = []
    self._current_emotions: Dict[str, EmotionalReading] = {}

    # Mental model tracking
    self._mental_models: Dict[Tuple[str, str], MentalModelSnapshot] = {}

    # Deception tracking
    self._deception_indicators: List[DeceptionIndicator] = []
    self._consistency_tracking: Dict[str, List[Dict[str, Any]]] = {}

    # Empathy and rapport tracking
    self._empathy_scores: Dict[Tuple[str, str], float] = {}
    self._rapport_levels: Dict[Tuple[str, str], float] = {}

    # Configuration
    self._track_emotions = self._config.get('track_emotions', True)
    self._detect_deception = self._config.get('detect_deception', True)
    self._emotion_sensitivity = self._config.get('emotion_sensitivity', 0.7)
    self._empathy_threshold = self._config.get('empathy_threshold', 0.6)

  def get_supported_agent_modules(self) -> Set[str]:
    """Return agent modules this supports."""
    return {'theory_of_mind'}

  def detect_emotion(
      self,
      text: str,
      actor: str,
      round_number: int,
  ) -> Optional[EmotionalReading]:
    """Detect emotional state from text."""
    if not self._track_emotions:
      return None

    # Simple keyword-based emotion detection
    detected_emotions = []
    for emotion, keywords in self.EMOTION_INDICATORS.items():
      if any(keyword in text.lower() for keyword in keywords):
        detected_emotions.append(emotion)

    if not detected_emotions:
      # Default to neutral if no clear emotion
      primary_emotion = 'calm'
      intensity = 0.3
    else:
      # Take the first detected emotion
      primary_emotion = detected_emotions[0]
      intensity = min(0.8, len(detected_emotions) * 0.3)

    # Determine valence
    if primary_emotion in ['happy', 'satisfied', 'hopeful', 'confident']:
      valence = 0.7
    elif primary_emotion in ['frustrated', 'angry', 'disappointed', 'anxious']:
      valence = -0.7
    else:
      valence = 0.0

    reading = EmotionalReading(
        participant=actor,
        primary_emotion=primary_emotion,
        intensity=intensity,
        valence=valence,
        confidence=0.6,  # Simple detection has moderate confidence
        triggers=detected_emotions,
        round_number=round_number,
    )

    return reading

  def check_consistency(
      self,
      actor: str,
      statement: str,
      round_number: int,
  ) -> Optional[DeceptionIndicator]:
    """Check for inconsistencies that might indicate deception."""
    if not self._detect_deception:
      return None

    # Track statements for consistency checking
    if actor not in self._consistency_tracking:
      self._consistency_tracking[actor] = []

    # Extract key claims from statement
    current_claims = []
    if ('price' in statement.lower() or 'pay' in statement.lower() or
        '$' in statement or any(word in statement.lower() for word in ['dollar', 'cost', 'amount'])):
      # Extract price mentions
      current_claims.append(('price', statement))
    if 'willing' in statement.lower() or 'accept' in statement.lower():
      current_claims.append(('willingness', statement))
    if 'cannot' in statement.lower() or 'impossible' in statement.lower():
      current_claims.append(('constraint', statement))

    # Check against history
    for claim_type, claim_text in current_claims:
      for past_statement in self._consistency_tracking[actor]:
        if past_statement['type'] == claim_type:
          # Simple contradiction check
          if ('cannot' in claim_text.lower() and
              'can' in past_statement['text'].lower() and
              'cannot' not in past_statement['text'].lower()):
            return DeceptionIndicator(
                actor=actor,
                indicator_type='inconsistency',
                description=f"Contradicts earlier statement about {claim_type}",
                severity=0.7,
                round_number=round_number,
            )
          # Check for price amount contradictions (simplified)
          import re
          current_amounts = re.findall(r'\$?(\d+)', claim_text)
          past_amounts = re.findall(r'\$?(\d+)', past_statement['text'])
          if current_amounts and past_amounts:
            current_val = int(current_amounts[0])
            past_val = int(past_amounts[0])
            # Check if amounts are contradictory in context
            if ('cannot' in claim_text.lower() and 'can' in past_statement['text'].lower() and 
                current_val < past_val):
              return DeceptionIndicator(
                  actor=actor,
                  indicator_type='inconsistency',
                  description=f"Contradicts earlier price statement: {past_val} vs {current_val}",
                  severity=0.8,
                  round_number=round_number,
              )

    # Store current claims
    for claim_type, claim_text in current_claims:
      self._consistency_tracking[actor].append({
          'type': claim_type,
          'text': claim_text,
          'round': round_number,
      })

    return None

  def update_mental_model(
      self,
      modeler: str,
      subject: str,
      observation: str,
  ) -> None:
    """Update one party's mental model of another."""
    key = (modeler, subject)

    # Extract perceived information from observation
    perceived_goals = []
    perceived_constraints = []

    if 'wants' in observation.lower() or 'seeks' in observation.lower():
      perceived_goals.append(observation)
    if 'cannot' in observation.lower() or 'limited' in observation.lower():
      perceived_constraints.append(observation)

    # Simple strategy inference
    if 'aggressive' in observation.lower() or 'demand' in observation.lower():
      predicted_strategy = 'competitive'
    elif 'cooperate' in observation.lower() or 'together' in observation.lower():
      predicted_strategy = 'collaborative'
    else:
      predicted_strategy = 'unknown'

    # Update or create mental model
    if key in self._mental_models:
      model = self._mental_models[key]
      model.perceived_goals.extend(perceived_goals)
      model.perceived_constraints.extend(perceived_constraints)
      if predicted_strategy != 'unknown':
        model.predicted_strategy = predicted_strategy
    else:
      self._mental_models[key] = MentalModelSnapshot(
          modeler=modeler,
          subject=subject,
          perceived_goals=perceived_goals,
          perceived_constraints=perceived_constraints,
          trust_assessment=0.5,
          predicted_strategy=predicted_strategy,
          accuracy_estimate=0.5,
      )

  def validate_action(
      self,
      actor: str,
      action: str,
      context: negotiation_modules.ModuleContext,
  ) -> Tuple[bool, Optional[str]]:
    """Validate action for emotional appropriateness."""
    # Check if action might damage rapport
    current_emotion = self._current_emotions.get(actor)

    if current_emotion and current_emotion.valence < -0.5:
      # Actor is in negative emotional state
      if any(word in action.lower() for word in ['demand', 'ultimatum', 'final']):
        return False, "Escalating when already emotional may damage negotiation"

    # Check for manipulation attempts
    deception = self.check_consistency(actor, action, context.current_round)
    if deception and deception.severity > 0.8:
      return False, f"Potential deception detected: {deception.description}"

    return True, None

  def update_state(
      self,
      event: str,
      actor: str,
      context: negotiation_modules.ModuleContext,
  ) -> None:
    """Update social/emotional state based on events."""
    # Detect emotions
    emotion = self.detect_emotion(event, actor, context.current_round)
    if emotion:
      self._emotional_history.append(emotion)
      self._current_emotions[actor] = emotion

    # Check for deception
    deception = self.check_consistency(actor, event, context.current_round)
    if deception:
      self._deception_indicators.append(deception)

    # Update empathy and rapport
    for participant in context.participants:
      if participant != actor:
        key = tuple(sorted([actor, participant]))

        # Update based on emotional mirroring
        if (actor in self._current_emotions and
            participant in self._current_emotions):
          actor_emotion = self._current_emotions[actor]
          participant_emotion = self._current_emotions[participant]

          # Similar emotions increase rapport
          if actor_emotion.primary_emotion == participant_emotion.primary_emotion:
            self._rapport_levels[key] = self._rapport_levels.get(key, 0.5) + 0.05

          # Responding to negative emotions with empathy
          if (participant_emotion.valence < 0 and
              any(word in event.lower() for word in ['understand', 'appreciate', 'acknowledge'])):
            self._empathy_scores[key] = self._empathy_scores.get(key, 0.5) + 0.1

        # Update mental models
        self.update_mental_model(actor, participant, event)

  def get_observation_context(
      self,
      observer: str,
      context: negotiation_modules.ModuleContext,
  ) -> str:
    """Get social/emotional context for observations."""
    observation = "\nSOCIAL DYNAMICS:\n"

    # Current emotional climate
    if self._current_emotions:
      observation += "\nEmotional Climate:\n"
      for participant, emotion in self._current_emotions.items():
        if participant != observer:
          observation += f"- {participant}: {emotion.primary_emotion} "
          observation += f"(intensity: {emotion.intensity:.0%})\n"

    # Mental models
    observer_models = [(k, v) for k, v in self._mental_models.items() if k[0] == observer]
    if observer_models:
      observation += "\nYour assessment of others:\n"
      for (_, subject), model in observer_models:
        observation += f"- {subject}: "
        observation += f"Strategy likely {model.predicted_strategy}, "
        observation += f"Trust level {model.trust_assessment:.0%}\n"

    # Rapport levels
    rapport_with_observer = {k: v for k, v in self._rapport_levels.items() if observer in k}
    if rapport_with_observer:
      observation += "\nRapport levels:\n"
      for parties, level in rapport_with_observer.items():
        other = parties[0] if parties[1] == observer else parties[1]
        observation += f"- With {other}: {level:.0%}\n"

    # Recent deception indicators
    if self._deception_indicators and self._detect_deception:
      recent_deceptions = self._deception_indicators[-2:]
      if recent_deceptions:
        observation += "\nTrust concerns:\n"
        for deception in recent_deceptions:
          observation += f"- {deception.actor}: {deception.description}\n"

    return observation

  def get_module_report(self) -> str:
    """Get social intelligence report."""
    report = "SOCIAL INTELLIGENCE REPORT:\n\n"

    # Emotional summary
    if self._emotional_history:
      report += "Emotional Journey:\n"
      emotion_counts = {}
      for reading in self._emotional_history:
        emotion_counts[reading.primary_emotion] = emotion_counts.get(reading.primary_emotion, 0) + 1

      for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        report += f"- {emotion}: {count} occurrences\n"

      # Average valence
      avg_valence = sum(r.valence for r in self._emotional_history) / len(self._emotional_history)
      if avg_valence > 0.3:
        report += f"\nOverall tone: Positive ({avg_valence:.1f})\n"
      elif avg_valence < -0.3:
        report += f"\nOverall tone: Negative ({avg_valence:.1f})\n"
      else:
        report += f"\nOverall tone: Neutral ({avg_valence:.1f})\n"

    # Mental models summary
    if self._mental_models:
      report += "\nMental Model Insights:\n"
      strategy_counts = {}
      for model in self._mental_models.values():
        strategy_counts[model.predicted_strategy] = strategy_counts.get(model.predicted_strategy, 0) + 1

      for strategy, count in strategy_counts.items():
        report += f"- {strategy} strategy: {count} assessments\n"

    # Trust and deception
    if self._deception_indicators:
      report += f"\nDeception Indicators: {len(self._deception_indicators)}\n"
      by_type = {}
      for indicator in self._deception_indicators:
        by_type[indicator.indicator_type] = by_type.get(indicator.indicator_type, 0) + 1

      for itype, count in by_type.items():
        report += f"- {itype}: {count}\n"

    # Rapport summary
    if self._rapport_levels:
      avg_rapport = sum(self._rapport_levels.values()) / len(self._rapport_levels)
      report += f"\nAverage Rapport: {avg_rapport:.0%}\n"

      if avg_rapport > 0.7:
        report += "- High rapport suggests collaborative potential\n"
      elif avg_rapport < 0.3:
        report += "- Low rapport may hinder agreement\n"

    return report

  def get_state(self) -> str:
    """Get the component state for saving/restoring."""
    state_dict = {
        'emotions': len(self._emotional_history),
        'models': len(self._mental_models),
        'deception': len(self._deception_indicators),
        'empathy': len(self._empathy_scores),
        'rapport': len(self._rapport_levels),
    }
    return str(state_dict)

  def set_state(self, state: str) -> None:
    """Set the component state from a saved string."""
    # Since this tracks dynamic data, we only restore basic structure
    # Full restoration would require serializing all tracking data
    pass


# Register the module
negotiation_modules.NegotiationGMModuleRegistry.register(
    'social_intelligence',
    SocialIntelligenceGM
)
