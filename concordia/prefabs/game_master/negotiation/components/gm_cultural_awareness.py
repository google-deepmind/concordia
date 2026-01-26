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

"""Cultural awareness module for negotiation game master."""

import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

from concordia.prefabs.game_master.negotiation.components import negotiation_modules


@dataclasses.dataclass
class CulturalProfile:
  """Represents cultural characteristics relevant to negotiation."""
  culture_name: str
  communication_style: str  # 'direct', 'indirect'
  context_level: str  # 'high_context', 'low_context'
  time_orientation: str  # 'monochronic', 'polychronic'
  relationship_importance: float  # 0-1
  hierarchy_sensitivity: float  # 0-1
  face_saving_importance: float  # 0-1
  negotiation_pace: str  # 'fast', 'moderate', 'slow'


class CulturalAwarenessGM(negotiation_modules.NegotiationGMModule):
  """GM module for managing cultural dynamics in negotiations."""

  # Predefined cultural profiles
  CULTURAL_PROFILES = {
      'western_business': CulturalProfile(
          culture_name='Western Business',
          communication_style='direct',
          context_level='low_context',
          time_orientation='monochronic',
          relationship_importance=0.3,
          hierarchy_sensitivity=0.4,
          face_saving_importance=0.3,
          negotiation_pace='fast',
      ),
      'japanese_business': CulturalProfile(
          culture_name='Japanese Business',
          communication_style='indirect',
          context_level='high_context',
          time_orientation='polychronic',
          relationship_importance=0.8,
          hierarchy_sensitivity=0.9,
          face_saving_importance=0.9,
          negotiation_pace='slow',
      ),
      'middle_eastern': CulturalProfile(
          culture_name='Middle Eastern',
          communication_style='indirect',
          context_level='high_context',
          time_orientation='polychronic',
          relationship_importance=0.9,
          hierarchy_sensitivity=0.7,
          face_saving_importance=0.8,
          negotiation_pace='moderate',
      ),
      'latin_american': CulturalProfile(
          culture_name='Latin American',
          communication_style='direct',
          context_level='high_context',
          time_orientation='polychronic',
          relationship_importance=0.8,
          hierarchy_sensitivity=0.6,
          face_saving_importance=0.7,
          negotiation_pace='moderate',
      ),
      'east_asian': CulturalProfile(
          culture_name='East Asian',
          communication_style='indirect',
          context_level='high_context',
          time_orientation='polychronic',
          relationship_importance=0.8,
          hierarchy_sensitivity=0.8,
          face_saving_importance=0.9,
          negotiation_pace='slow',
      ),
  }

  def __init__(
      self,
      name: str = 'cultural_awareness',
      priority: int = 85,
      config: Optional[Dict[str, Any]] = None,
  ):
    """Initialize cultural awareness module."""
    super().__init__(name, priority, config)

    # Track cultural profiles of participants
    self._participant_cultures: Dict[str, str] = {}

    # Track cultural incidents
    self._cultural_incidents: List[Dict[str, Any]] = []

    # Track cultural adaptation efforts
    self._adaptation_tracking: Dict[str, List[str]] = {}

    # Configuration
    self._enforce_protocols = self._config.get('enforce_protocols', True)
    self._track_adaptation = self._config.get('track_adaptation', True)
    self._sensitivity_level = self._config.get('sensitivity', 0.8)

  def get_supported_agent_modules(self) -> Set[str]:
    """Return agent modules this supports."""
    return {'cultural_adaptation'}

  def set_participant_culture(self, participant: str, culture: str) -> None:
    """Set the cultural profile for a participant."""
    if culture in self.CULTURAL_PROFILES:
      self._participant_cultures[participant] = culture
      self.set_module_state(f'culture_{participant}', culture)

  def detect_cultural_violation(
      self,
      actor: str,
      action: str,
      recipient: str,
  ) -> Optional[str]:
    """Detect potential cultural protocol violations."""
    actor_culture = self._participant_cultures.get(actor)
    recipient_culture = self._participant_cultures.get(recipient)

    # Need at least recipient culture to detect violations
    if not recipient_culture:
      return None

    recipient_profile = self.CULTURAL_PROFILES[recipient_culture]

    # Check for face-saving violations (always check regardless of actor culture)
    if recipient_profile.face_saving_importance > 0.7:
      if any(word in action.lower() for word in ['wrong', 'mistake', 'fault', 'blame']):
        return f"Direct criticism threatens face in {recipient_culture} culture"

    # Check for hierarchy violations
    if recipient_profile.hierarchy_sensitivity > 0.7:
      if any(word in action.lower() for word in ['demand', 'insist', 'must']):
        return f"Overly assertive language may violate {recipient_culture} hierarchy norms"

    # Check for communication style mismatch (only if both cultures known)
    if actor_culture:
      actor_profile = self.CULTURAL_PROFILES[actor_culture]
      if (actor_profile.communication_style == 'direct' and
          recipient_profile.communication_style == 'indirect'):
        if any(word in action.lower() for word in ['no', 'reject', 'refuse', 'impossible']):
          return f"Too direct rejection may offend {recipient_culture} sensibilities"

    return None

  def validate_action(
      self,
      actor: str,
      action: str,
      context: negotiation_modules.ModuleContext,
  ) -> Tuple[bool, Optional[str]]:
    """Validate action for cultural appropriateness."""
    if not self._enforce_protocols:
      return True, None

    # Check each potential recipient
    for participant in context.participants:
      if participant != actor:
        violation = self.detect_cultural_violation(actor, action, participant)
        if violation and self._sensitivity_level > 0.5:
          return False, f"Cultural sensitivity warning: {violation}"

    return True, None

  def update_state(
      self,
      event: str,
      actor: str,
      context: negotiation_modules.ModuleContext,
  ) -> None:
    """Update state based on negotiation events."""
    # Track cultural adaptation
    if self._track_adaptation and 'adapt' in event.lower():
      if actor not in self._adaptation_tracking:
        self._adaptation_tracking[actor] = []
      self._adaptation_tracking[actor].append(event)

    # Record cultural incidents
    for participant in context.participants:
      if participant != actor:
        violation = self.detect_cultural_violation(actor, event, participant)
        if violation:
          self._cultural_incidents.append({
              'round': context.current_round,
              'actor': actor,
              'recipient': participant,
              'issue': violation,
              'event': event,
          })

  def get_observation_context(
      self,
      observer: str,
      context: negotiation_modules.ModuleContext,
  ) -> str:
    """Get cultural context for observations."""
    observer_culture = self._participant_cultures.get(observer)
    if not observer_culture:
      return ""

    profile = self.CULTURAL_PROFILES[observer_culture]

    observation = f"\nCULTURAL CONTEXT ({profile.culture_name}):\n"

    # Add relevant cultural guidance
    if context.current_phase == 'opening':
      if profile.relationship_importance > 0.7:
        observation += "- Building rapport is essential before discussing business\n"
      if profile.negotiation_pace == 'slow':
        observation += "- Patience is valued; avoid rushing to business matters\n"

    elif context.current_phase == 'bargaining':
      if profile.communication_style == 'indirect':
        observation += "- Pay attention to subtle cues and implied meanings\n"
      if profile.face_saving_importance > 0.7:
        observation += "- Maintain respect and avoid direct confrontation\n"

    # Note cultural differences with other participants
    for participant, culture in self._participant_cultures.items():
      if participant != observer and culture != observer_culture:
        other_profile = self.CULTURAL_PROFILES[culture]
        if abs(profile.relationship_importance - other_profile.relationship_importance) > 0.4:
          observation += f"- Note: {participant} has different relationship expectations\n"
        if profile.negotiation_pace != other_profile.negotiation_pace:
          observation += f"- Note: {participant} prefers {other_profile.negotiation_pace} pace\n"

    return observation

  def get_module_report(self) -> str:
    """Get cultural dynamics report."""
    report = "CULTURAL DYNAMICS REPORT:\n\n"

    # Participant cultures
    if self._participant_cultures:
      report += "Participant Cultures:\n"
      for participant, culture in self._participant_cultures.items():
        profile = self.CULTURAL_PROFILES[culture]
        report += f"- {participant}: {culture} "
        report += f"({profile.communication_style}, {profile.negotiation_pace} pace)\n"

    # Cultural incidents
    if self._cultural_incidents:
      report += f"\nCultural Incidents: {len(self._cultural_incidents)}\n"
      for incident in self._cultural_incidents[-3:]:  # Last 3 incidents
        report += f"- Round {incident['round']}: {incident['issue']}\n"

    # Adaptation tracking
    if self._adaptation_tracking:
      report += "\nCultural Adaptation Efforts:\n"
      for participant, efforts in self._adaptation_tracking.items():
        report += f"- {participant}: {len(efforts)} adaptations\n"

    # Cross-cultural dynamics
    cultures = set(self._participant_cultures.values())
    if len(cultures) > 1:
      report += f"\nCross-cultural negotiation: {len(cultures)} cultures involved\n"

      # Identify potential friction points
      high_context = sum(1 for c in cultures
                        if self.CULTURAL_PROFILES.get(c, None) and
                        self.CULTURAL_PROFILES[c].context_level == 'high_context')
      low_context = len(cultures) - high_context

      if high_context > 0 and low_context > 0:
        report += "- Warning: Mix of high/low context cultures\n"

    return report

  def get_state(self) -> str:
    """Get the component state for saving/restoring."""
    state_dict = {
        'participants': len(self._participant_cultures),
        'violations': len(self._violation_history),
        'adaptations': sum(len(efforts) for efforts in self._adaptation_tracking.values()),
    }
    return str(state_dict)

  def set_state(self, state: str) -> None:
    """Set the component state from a saved string."""
    # Since this tracks dynamic data, we only restore basic structure
    pass


# Register the module
negotiation_modules.NegotiationGMModuleRegistry.register(
    'cultural_awareness',
    CulturalAwarenessGM
)
