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

"""Uncertainty management module for negotiation game master."""

import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

from concordia.prefabs.game_master.negotiation.components import negotiation_modules


@dataclasses.dataclass
class InformationAsymmetry:
  """Tracks information imbalance between parties."""
  actor: str
  information_type: str  # 'preference', 'constraint', 'reservation_value', 'external'
  knowledge_level: float  # 0-1
  visibility: Set[str]  # Who can see this information
  strategic_value: float  # 0-1
  revelation_cost: float  # 0-1


@dataclasses.dataclass
class UncertaintyMetrics:
  """Metrics for tracking uncertainty in negotiations."""
  participant: str
  preference_uncertainty: float  # 0-1
  constraint_uncertainty: float  # 0-1
  outcome_uncertainty: float  # 0-1
  opponent_model_confidence: float  # 0-1
  information_seeking_behavior: float  # 0-1


@dataclasses.dataclass
class InformationRequest:
  """Represents a request for information."""
  requester: str
  target: str
  information_type: str
  specificity: str  # 'general', 'specific', 'precise'
  urgency: float  # 0-1
  round_requested: int
  granted: bool = False


class UncertaintyManagementGM(negotiation_modules.NegotiationGMModule):
  """GM module for managing information asymmetry and uncertainty."""

  def __init__(
      self,
      name: str = 'uncertainty_management',
      priority: int = 75,
      config: Optional[Dict[str, Any]] = None,
  ):
    """Initialize uncertainty management module."""
    super().__init__(name, priority, config)

    # Information tracking
    self._information_asymmetries: List[InformationAsymmetry] = []
    self._uncertainty_metrics: Dict[str, UncertaintyMetrics] = {}
    self._information_requests: List[InformationRequest] = []

    # Strategic information management
    self._information_revelation_history: List[Dict[str, Any]] = []
    self._uncertainty_exploitation: Dict[str, List[str]] = {}

    # Market for information
    self._information_values: Dict[str, float] = {}
    self._information_trading: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # Configuration
    self._track_asymmetries = self._config.get('track_asymmetries', True)
    self._allow_information_trading = self._config.get('allow_information_trading', True)
    self._uncertainty_threshold = self._config.get('uncertainty_threshold', 0.7)
    self._information_decay_rate = self._config.get('information_decay_rate', 0.95)

  def get_supported_agent_modules(self) -> Set[str]:
    """Return agent modules this supports."""
    return {'uncertainty_aware'}

  def track_information_asymmetry(
      self,
      actor: str,
      information_type: str,
      knowledge_level: float,
      visible_to: Set[str],
      strategic_value: float = 0.5,
  ) -> None:
    """Track information asymmetry between parties."""
    if not self._track_asymmetries:
      return

    asymmetry = InformationAsymmetry(
        actor=actor,
        information_type=information_type,
        knowledge_level=knowledge_level,
        visibility=visible_to,
        strategic_value=strategic_value,
        revelation_cost=1 - strategic_value,  # Inverse relationship
    )
    self._information_asymmetries.append(asymmetry)

  def calculate_uncertainty_metrics(
      self,
      participant: str,
      context: negotiation_modules.ModuleContext,
  ) -> UncertaintyMetrics:
    """Calculate uncertainty metrics for a participant."""
    if participant not in self._uncertainty_metrics:
      self._uncertainty_metrics[participant] = UncertaintyMetrics(
          participant=participant,
          preference_uncertainty=0.8,  # Start with high uncertainty
          constraint_uncertainty=0.7,
          outcome_uncertainty=0.9,
          opponent_model_confidence=0.3,
          information_seeking_behavior=0.5,
      )

    metrics = self._uncertainty_metrics[participant]

    # Update based on negotiation progress
    total_rounds = context.current_round
    if total_rounds > 5:
      # Uncertainty typically decreases over time
      decay_factor = self._information_decay_rate ** (total_rounds - 5)
      metrics.preference_uncertainty *= decay_factor
      metrics.constraint_uncertainty *= decay_factor
      metrics.outcome_uncertainty *= decay_factor

    # Increase confidence if information was revealed
    recent_revelations = [r for r in self._information_revelation_history
                         if r.get('recipient') == participant and
                         r.get('round', 0) > total_rounds - 3]
    
    confidence_boost = min(0.3, len(recent_revelations) * 0.1)
    metrics.opponent_model_confidence = min(1.0, metrics.opponent_model_confidence + confidence_boost)

    return metrics

  def assess_information_value(
      self,
      information_type: str,
      requester: str,
      provider: str,
      context: negotiation_modules.ModuleContext,
  ) -> float:
    """Assess the value of information for strategic purposes."""
    base_value = 0.5

    # Value increases with uncertainty
    requester_metrics = self.calculate_uncertainty_metrics(requester, context)
    if information_type == 'preference':
      uncertainty_factor = requester_metrics.preference_uncertainty
    elif information_type == 'constraint':
      uncertainty_factor = requester_metrics.constraint_uncertainty
    else:
      uncertainty_factor = requester_metrics.outcome_uncertainty

    # Value increases with negotiation criticality
    phase_multiplier = {
        'opening': 0.7,
        'bargaining': 1.0,
        'closing': 1.3,
    }.get(context.current_phase, 1.0)

    # Value affected by information asymmetry
    asymmetry_bonus = 0.0
    for asymmetry in self._information_asymmetries:
      if (asymmetry.actor == provider and
          asymmetry.information_type == information_type and
          requester not in asymmetry.visibility):
        asymmetry_bonus = asymmetry.strategic_value * 0.3
        break

    total_value = min(1.0, base_value + uncertainty_factor * 0.3 + asymmetry_bonus) * phase_multiplier
    return total_value

  def process_information_request(
      self,
      requester: str,
      target: str,
      request_text: str,
      context: negotiation_modules.ModuleContext,
  ) -> Tuple[bool, Optional[str]]:
    """Process a request for information."""
    # Parse request type
    information_type = 'general'
    if any(word in request_text.lower() for word in ['preference', 'want', 'value']):
      information_type = 'preference'
    elif any(word in request_text.lower() for word in ['constraint', 'limit', 'cannot']):
      information_type = 'constraint'
    elif any(word in request_text.lower() for word in ['reservation', 'minimum', 'bottom line']):
      information_type = 'reservation_value'

    # Determine specificity
    specificity = 'general'
    if any(word in request_text.lower() for word in ['exactly', 'precisely', 'specific']):
      specificity = 'precise'
    elif any(word in request_text.lower() for word in ['about', 'roughly', 'approximately']):
      specificity = 'specific'

    # Calculate urgency
    urgency = 0.5
    if any(word in request_text.lower() for word in ['urgent', 'need', 'must know']):
      urgency = 0.9
    elif any(word in request_text.lower() for word in ['curious', 'wondering', 'interested']):
      urgency = 0.3

    # Create request
    request = InformationRequest(
        requester=requester,
        target=target,
        information_type=information_type,
        specificity=specificity,
        urgency=urgency,
        round_requested=context.current_round,
    )
    self._information_requests.append(request)

    # Assess if request should be granted
    information_value = self.assess_information_value(information_type, requester, target, context)
    
    # Simple decision model - in practice would be more sophisticated
    grant_probability = 0.7 if urgency > 0.6 else 0.4
    if information_value > 0.8:
      grant_probability *= 0.6  # Valuable information less likely to be shared
    
    granted = grant_probability > 0.5  # Simplified decision
    request.granted = granted

    if granted:
      return True, f"Information request granted: {information_type}"
    else:
      return False, f"Information request declined: strategic value too high"

  def validate_action(
      self,
      actor: str,
      action: str,
      context: negotiation_modules.ModuleContext,
  ) -> Tuple[bool, Optional[str]]:
    """Validate action for uncertainty management compliance."""
    # Check for information revelation
    if any(word in action.lower() for word in ['reveal', 'tell', 'share', 'disclose']):
      # This could affect information asymmetries
      self._information_revelation_history.append({
          'revealer': actor,
          'round': context.current_round,
          'action': action,
      })

    # Check if action exploits uncertainty inappropriately
    if any(word in action.lower() for word in ['exploit', 'take advantage', 'mislead']):
      return False, "Exploitative behavior detected in uncertain environment"

    return True, None

  def update_state(
      self,
      event: str,
      actor: str,
      context: negotiation_modules.ModuleContext,
  ) -> None:
    """Update uncertainty state based on events."""
    # Track information-seeking behavior
    if any(word in event.lower() for word in ['ask', 'inquire', 'question', 'clarify']):
      if actor not in self._uncertainty_exploitation:
        self._uncertainty_exploitation[actor] = []
      self._uncertainty_exploitation[actor].append('information_seeking')

      # Update metrics
      metrics = self.calculate_uncertainty_metrics(actor, context)
      metrics.information_seeking_behavior = min(1.0, metrics.information_seeking_behavior + 0.1)

    # Track information revelation
    if any(word in event.lower() for word in ['reveal', 'share', 'disclose', 'tell']):
      self._information_revelation_history.append({
          'revealer': actor,
          'recipient': 'all',  # Simplified - could parse specific recipients
          'round': context.current_round,
          'event': event,
      })

      # Reduce uncertainty for all participants
      for participant in context.participants:
        if participant != actor:
          metrics = self.calculate_uncertainty_metrics(participant, context)
          metrics.opponent_model_confidence = min(1.0, metrics.opponent_model_confidence + 0.05)

    # Update information values based on revealed information
    if context.current_round % 3 == 0:  # Update every 3 rounds
      for info_type in ['preference', 'constraint', 'reservation_value']:
        current_value = self._information_values.get(info_type, 0.5)
        # Information becomes less valuable as more is revealed
        revelation_factor = len([r for r in self._information_revelation_history
                               if info_type in r.get('event', '').lower()]) * 0.05
        self._information_values[info_type] = max(0.1, current_value - revelation_factor)

  def get_observation_context(
      self,
      observer: str,
      context: negotiation_modules.ModuleContext,
  ) -> str:
    """Get uncertainty context for observations."""
    observation = "\nUNCERTAINTY ANALYSIS:\n"

    # Personal uncertainty metrics
    metrics = self.calculate_uncertainty_metrics(observer, context)
    observation += f"\nYour uncertainty levels:\n"
    observation += f"- Preference uncertainty: {metrics.preference_uncertainty:.0%}\n"
    observation += f"- Constraint uncertainty: {metrics.constraint_uncertainty:.0%}\n"
    observation += f"- Outcome uncertainty: {metrics.outcome_uncertainty:.0%}\n"
    observation += f"- Opponent model confidence: {metrics.opponent_model_confidence:.0%}\n"

    # Information landscape
    observer_asymmetries = [a for a in self._information_asymmetries
                           if observer in a.visibility]
    if observer_asymmetries:
      observation += f"\nAvailable information insights: {len(observer_asymmetries)}\n"
      for asymmetry in observer_asymmetries[-3:]:  # Show last 3
        observation += f"- {asymmetry.information_type} from {asymmetry.actor} "
        observation += f"(strategic value: {asymmetry.strategic_value:.0%})\n"

    # Information requests
    observer_requests = [r for r in self._information_requests
                        if r.requester == observer]
    if observer_requests:
      recent_requests = [r for r in observer_requests if r.round_requested > context.current_round - 5]
      observation += f"\nRecent information requests: {len(recent_requests)}\n"
      granted_requests = sum(1 for r in recent_requests if r.granted)
      observation += f"- Granted: {granted_requests}/{len(recent_requests)}\n"

    # Strategic guidance
    if metrics.preference_uncertainty > self._uncertainty_threshold:
      observation += "\n⚠️ High preference uncertainty - consider information gathering\n"
    
    if metrics.opponent_model_confidence < 0.3:
      observation += "\n⚠️ Low opponent model confidence - more observation needed\n"

    # Information market insights
    if self._allow_information_trading:
      high_value_info = [info for info, value in self._information_values.items() if value > 0.7]
      if high_value_info:
        observation += f"\nHigh-value information types: {', '.join(high_value_info)}\n"

    return observation

  def get_module_report(self) -> str:
    """Get uncertainty management report."""
    report = "UNCERTAINTY MANAGEMENT REPORT:\n\n"

    # Information asymmetry summary
    if self._information_asymmetries:
      report += f"Information Asymmetries: {len(self._information_asymmetries)}\n"
      
      by_type = {}
      for asymmetry in self._information_asymmetries:
        by_type[asymmetry.information_type] = by_type.get(asymmetry.information_type, 0) + 1
      
      for info_type, count in by_type.items():
        report += f"- {info_type}: {count}\n"

      # Average strategic value
      avg_strategic_value = sum(a.strategic_value for a in self._information_asymmetries) / len(self._information_asymmetries)
      report += f"\nAverage strategic value: {avg_strategic_value:.0%}\n"

    # Uncertainty metrics summary
    if self._uncertainty_metrics:
      report += f"\nParticipant Uncertainty Analysis:\n"
      for participant, metrics in self._uncertainty_metrics.items():
        report += f"- {participant}:\n"
        report += f"  Preference: {metrics.preference_uncertainty:.0%}, "
        report += f"  Constraint: {metrics.constraint_uncertainty:.0%}, "
        report += f"  Outcome: {metrics.outcome_uncertainty:.0%}\n"
        report += f"  Opponent confidence: {metrics.opponent_model_confidence:.0%}\n"

    # Information requests
    if self._information_requests:
      report += f"\nInformation Requests: {len(self._information_requests)}\n"
      granted = sum(1 for r in self._information_requests if r.granted)
      report += f"- Granted: {granted}/{len(self._information_requests)} ({granted/len(self._information_requests):.0%})\n"

    # Information market
    if self._information_values:
      report += "\nInformation Market Values:\n"
      for info_type, value in sorted(self._information_values.items(), key=lambda x: x[1], reverse=True):
        report += f"- {info_type}: {value:.0%}\n"

    # Strategic insights
    total_revelations = len(self._information_revelation_history)
    if total_revelations > 0:
      recent_revelations = len([r for r in self._information_revelation_history
                               if r.get('round', 0) > max(1, max(r.get('round', 0) for r in self._information_revelation_history) - 5)])
      report += f"\nInformation Revelation Trend:\n"
      report += f"- Total revelations: {total_revelations}\n"
      report += f"- Recent activity: {recent_revelations} in last 5 rounds\n"

      if recent_revelations > total_revelations * 0.5:
        report += "- Pattern: High information sharing phase\n"
      elif recent_revelations < total_revelations * 0.2:
        report += "- Pattern: Information hoarding phase\n"

    return report

  def get_state(self) -> str:
    """Get the component state for saving/restoring."""
    state_dict = {
        'asymmetries': len(self._information_asymmetries),
        'requests': len(self._information_requests),
        'metrics': len(self._uncertainty_metrics),
    }
    return str(state_dict)

  def set_state(self, state: str) -> None:
    """Set the component state from a saved string."""
    # Since this tracks dynamic data, we only restore basic structure
    pass


# Register the module
negotiation_modules.NegotiationGMModuleRegistry.register(
    'uncertainty_management',
    UncertaintyManagementGM
)