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

"""Strategy evolution module for negotiation game master."""

import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

from concordia.prefabs.game_master.negotiation.components import negotiation_modules


@dataclasses.dataclass
class StrategySnapshot:
  """Snapshot of a participant's strategy at a point in time."""
  participant: str
  round_number: int
  strategy_type: str  # 'cooperative', 'competitive', 'integrative', 'mixed'
  aggressiveness: float  # 0-1
  concession_rate: float  # 0-1
  information_sharing: float  # 0-1
  coalition_tendency: float  # 0-1
  risk_tolerance: float  # 0-1
  tactics_used: List[str]


@dataclasses.dataclass
class StrategyTransition:
  """Records a strategy change by a participant."""
  participant: str
  from_strategy: str
  to_strategy: str
  trigger_event: str
  transition_round: int
  effectiveness_before: float  # 0-1
  effectiveness_after: float  # 0-1
  adaptation_type: str  # 'reactive', 'proactive', 'imitative'


@dataclasses.dataclass
class PerformanceMetrics:
  """Performance metrics for strategy evaluation."""
  participant: str
  strategy_type: str
  rounds_active: int
  success_rate: float  # 0-1
  efficiency: float  # 0-1
  adaptability_score: float  # 0-1
  counterpart_responses: Dict[str, float]  # How others respond


class StrategyEvolutionGM(negotiation_modules.NegotiationGMModule):
  """GM module for tracking and facilitating strategy evolution."""

  # Strategy archetypes
  STRATEGY_INDICATORS = {
      'cooperative': ['agree', 'collaborate', 'mutual', 'together', 'help'],
      'competitive': ['better', 'win', 'superior', 'demand', 'insist'],
      'integrative': ['creative', 'alternative', 'expand', 'value', 'both'],
      'defensive': ['protect', 'maintain', 'preserve', 'careful', 'cautious'],
      'adaptive': ['adjust', 'flexible', 'consider', 'depending', 'adapt'],
  }

  def __init__(
      self,
      name: str = 'strategy_evolution',
      priority: int = 65,
      config: Optional[Dict[str, Any]] = None,
  ):
    """Initialize strategy evolution module."""
    super().__init__(name, priority, config)

    # Strategy tracking
    self._strategy_snapshots: List[StrategySnapshot] = []
    self._strategy_transitions: List[StrategyTransition] = []
    self._current_strategies: Dict[str, str] = {}

    # Performance tracking
    self._performance_metrics: Dict[Tuple[str, str], PerformanceMetrics] = {}
    self._outcome_history: List[Dict[str, Any]] = []

    # Learning patterns
    self._learning_indicators: Dict[str, List[str]] = {}
    self._adaptation_triggers: Dict[str, List[str]] = {}

    # Evolution dynamics
    self._strategy_diffusion: Dict[str, Dict[str, int]] = {}  # Strategy spread tracking
    self._innovation_events: List[Dict[str, Any]] = []

    # Configuration
    self._track_evolution = self._config.get('track_evolution', True)
    self._adaptation_sensitivity = self._config.get('adaptation_sensitivity', 0.6)
    self._performance_window = self._config.get('performance_window', 5)
    self._innovation_threshold = self._config.get('innovation_threshold', 0.8)

  def get_supported_agent_modules(self) -> Set[str]:
    """Return agent modules this supports."""
    return {'strategy_evolution'}

  def analyze_strategy_from_action(
      self,
      participant: str,
      action: str,
      context: negotiation_modules.ModuleContext,
  ) -> StrategySnapshot:
    """Analyze current strategy from participant's action."""
    # Detect strategy type based on keywords
    strategy_scores = {}
    for strategy, keywords in self.STRATEGY_INDICATORS.items():
      score = sum(1 for keyword in keywords if keyword in action.lower())
      strategy_scores[strategy] = score

    # Primary strategy is highest scoring, or 'mixed' if tie/unclear
    if strategy_scores:
      primary_strategy = max(strategy_scores, key=strategy_scores.get)
      if strategy_scores[primary_strategy] == 0:
        primary_strategy = 'mixed'
    else:
      primary_strategy = 'mixed'

    # Analyze strategy dimensions
    aggressiveness = 0.5
    if any(word in action.lower() for word in ['demand', 'must', 'insist', 'require']):
      aggressiveness += 0.3
    elif any(word in action.lower() for word in ['suggest', 'perhaps', 'might']):
      aggressiveness -= 0.2

    concession_rate = 0.5
    if any(word in action.lower() for word in ['reduce', 'lower', 'accept less']):
      concession_rate += 0.3
    elif any(word in action.lower() for word in ['maintain', 'firm', 'non-negotiable']):
      concession_rate -= 0.2

    information_sharing = 0.5
    if any(word in action.lower() for word in ['share', 'tell', 'reveal', 'explain']):
      information_sharing += 0.3
    elif any(word in action.lower() for word in ['private', 'confidential', 'cannot say']):
      information_sharing -= 0.3

    coalition_tendency = 0.5
    if any(word in action.lower() for word in ['together', 'alliance', 'coordinate']):
      coalition_tendency += 0.4
    elif any(word in action.lower() for word in ['alone', 'individual', 'separate']):
      coalition_tendency -= 0.3

    # Extract tactics used
    tactics_used = []
    tactic_keywords = {
        'anchoring': ['first offer', 'starting point', 'initial'],
        'reciprocity': ['return favor', 'mutual', 'exchange'],
        'deadline_pressure': ['time', 'deadline', 'urgent'],
        'value_creation': ['expand', 'both benefit', 'win-win'],
        'information_seeking': ['question', 'clarify', 'understand'],
    }

    for tactic, keywords in tactic_keywords.items():
      if any(keyword in action.lower() for keyword in keywords):
        tactics_used.append(tactic)

    # Bound values to 0-1 range
    aggressiveness = max(0, min(1, aggressiveness))
    concession_rate = max(0, min(1, concession_rate))
    information_sharing = max(0, min(1, information_sharing))
    coalition_tendency = max(0, min(1, coalition_tendency))

    snapshot = StrategySnapshot(
        participant=participant,
        round_number=context.current_round,
        strategy_type=primary_strategy,
        aggressiveness=aggressiveness,
        concession_rate=concession_rate,
        information_sharing=information_sharing,
        coalition_tendency=coalition_tendency,
        risk_tolerance=0.5,  # Would need more sophisticated analysis
        tactics_used=tactics_used,
    )

    return snapshot

  def detect_strategy_change(
      self,
      participant: str,
      new_snapshot: StrategySnapshot,
  ) -> Optional[StrategyTransition]:
    """Detect if participant changed strategies."""
    # Find previous strategy
    previous_snapshots = [s for s in self._strategy_snapshots 
                         if s.participant == participant]
    if not previous_snapshots:
      return None

    latest_previous = previous_snapshots[-1]
    
    # Check for strategy type change
    if latest_previous.strategy_type != new_snapshot.strategy_type:
      # Determine adaptation type
      adaptation_type = 'reactive'  # Default
      
      # Look for triggers in recent events
      trigger_event = f"Strategy shift from {latest_previous.strategy_type} to {new_snapshot.strategy_type}"
      
      # Calculate effectiveness before/after (simplified)
      effectiveness_before = self._calculate_strategy_effectiveness(
          participant, latest_previous.strategy_type, latest_previous.round_number
      )
      effectiveness_after = 0.5  # Unknown until observed

      transition = StrategyTransition(
          participant=participant,
          from_strategy=latest_previous.strategy_type,
          to_strategy=new_snapshot.strategy_type,
          trigger_event=trigger_event,
          transition_round=new_snapshot.round_number,
          effectiveness_before=effectiveness_before,
          effectiveness_after=effectiveness_after,
          adaptation_type=adaptation_type,
      )

      return transition

    return None

  def _calculate_strategy_effectiveness(
      self,
      participant: str,
      strategy_type: str,
      round_number: int,
  ) -> float:
    """Calculate effectiveness of a strategy (simplified)."""
    # In practice, would analyze outcomes, agreement rates, satisfaction, etc.
    # For now, return a simplified calculation based on strategy characteristics
    
    effectiveness_map = {
        'cooperative': 0.7,
        'competitive': 0.6,
        'integrative': 0.8,
        'defensive': 0.5,
        'adaptive': 0.75,
        'mixed': 0.6,
    }
    
    base_effectiveness = effectiveness_map.get(strategy_type, 0.5)
    
    # Add some variation based on context (simplified)
    context_modifier = (round_number % 10) * 0.05  # Simple round-based variation
    
    return max(0, min(1, base_effectiveness + context_modifier - 0.25))

  def track_strategy_diffusion(
      self,
      context: negotiation_modules.ModuleContext,
  ) -> None:
    """Track how strategies spread between participants."""
    # Analyze recent strategy transitions for diffusion patterns
    recent_transitions = [t for t in self._strategy_transitions 
                         if t.transition_round > context.current_round - 5]

    for transition in recent_transitions:
      to_strategy = transition.to_strategy
      participant = transition.participant

      # Initialize diffusion tracking
      if to_strategy not in self._strategy_diffusion:
        self._strategy_diffusion[to_strategy] = {}

      # Check if other participants recently used this strategy
      other_participants = [p for p in context.participants if p != participant]
      for other in other_participants:
        other_snapshots = [s for s in self._strategy_snapshots[-10:]  # Recent snapshots
                          if s.participant == other and s.strategy_type == to_strategy]
        if other_snapshots:
          # Record potential diffusion
          diffusion_key = f"{other}->{participant}"
          if diffusion_key not in self._strategy_diffusion[to_strategy]:
            self._strategy_diffusion[to_strategy][diffusion_key] = 0
          self._strategy_diffusion[to_strategy][diffusion_key] += 1

  def detect_innovation(
      self,
      snapshot: StrategySnapshot,
      context: negotiation_modules.ModuleContext,
  ) -> Optional[Dict[str, Any]]:
    """Detect innovative strategy behaviors."""
    # Look for novel tactic combinations
    unique_tactics = set(snapshot.tactics_used)
    
    # Compare with historical tactic usage
    historical_combinations = set()
    for past_snapshot in self._strategy_snapshots:
      if len(past_snapshot.tactics_used) >= 2:
        historical_combinations.add(tuple(sorted(past_snapshot.tactics_used)))

    current_combination = tuple(sorted(snapshot.tactics_used))
    
    # Innovation if: new combination + high performance potential
    if (len(unique_tactics) >= 3 and 
        current_combination not in historical_combinations and
        len(current_combination) >= 2):
      
      innovation = {
          'participant': snapshot.participant,
          'innovation_type': 'tactic_combination',
          'tactics': list(unique_tactics),
          'round': snapshot.round_number,
          'novelty_score': len(unique_tactics) / 5.0,  # Normalize by max expected
      }
      
      self._innovation_events.append(innovation)
      return innovation

    return None

  def validate_action(
      self,
      actor: str,
      action: str,
      context: negotiation_modules.ModuleContext,
  ) -> Tuple[bool, Optional[str]]:
    """Validate action for strategy evolution compliance."""
    # Generally allow all actions - evolution module is observational
    # Could add constraints on rapid strategy changes if desired
    
    # Check for potentially harmful strategy oscillation
    actor_transitions = [t for t in self._strategy_transitions 
                        if t.participant == actor and 
                        t.transition_round > context.current_round - 3]
    
    if len(actor_transitions) > 2:
      return False, "Excessive strategy changes may confuse negotiation"

    return True, None

  def update_state(
      self,
      event: str,
      actor: str,
      context: negotiation_modules.ModuleContext,
  ) -> None:
    """Update strategy evolution state based on events."""
    if not self._track_evolution:
      return

    # Analyze current strategy
    snapshot = self.analyze_strategy_from_action(actor, event, context)
    self._strategy_snapshots.append(snapshot)

    # Update current strategy tracking
    self._current_strategies[actor] = snapshot.strategy_type

    # Detect strategy changes
    transition = self.detect_strategy_change(actor, snapshot)
    if transition:
      self._strategy_transitions.append(transition)

    # Track learning indicators
    if any(word in event.lower() for word in ['learn', 'realize', 'understand', 'see now']):
      if actor not in self._learning_indicators:
        self._learning_indicators[actor] = []
      self._learning_indicators[actor].append(event)

    # Track adaptation triggers
    if any(word in event.lower() for word in ['because', 'since', 'due to', 'respond to']):
      if actor not in self._adaptation_triggers:
        self._adaptation_triggers[actor] = []
      self._adaptation_triggers[actor].append(event)

    # Detect innovation
    innovation = self.detect_innovation(snapshot, context)

    # Update strategy diffusion tracking
    if context.current_round % 3 == 0:  # Every 3 rounds
      self.track_strategy_diffusion(context)

    # Update performance metrics
    self._update_performance_metrics(actor, snapshot, context)

  def _update_performance_metrics(
      self,
      participant: str,
      snapshot: StrategySnapshot,
      context: negotiation_modules.ModuleContext,
  ) -> None:
    """Update performance metrics for strategies."""
    key = (participant, snapshot.strategy_type)
    
    if key not in self._performance_metrics:
      self._performance_metrics[key] = PerformanceMetrics(
          participant=participant,
          strategy_type=snapshot.strategy_type,
          rounds_active=0,
          success_rate=0.5,
          efficiency=0.5,
          adaptability_score=0.5,
          counterpart_responses={},
      )

    metrics = self._performance_metrics[key]
    metrics.rounds_active += 1

    # Update adaptability score based on strategy flexibility
    recent_snapshots = [s for s in self._strategy_snapshots[-5:] 
                       if s.participant == participant]
    if len(recent_snapshots) > 1:
      tactic_variety = len(set(tactic for s in recent_snapshots for tactic in s.tactics_used))
      metrics.adaptability_score = min(1.0, tactic_variety / 10.0)

  def get_observation_context(
      self,
      observer: str,
      context: negotiation_modules.ModuleContext,
  ) -> str:
    """Get strategy evolution context for observations."""
    observation = "\nSTRATEGY EVOLUTION:\n"

    # Current strategy
    current_strategy = self._current_strategies.get(observer, 'unknown')
    observation += f"\nYour current strategy: {current_strategy}\n"

    # Recent strategy transitions
    observer_transitions = [t for t in self._strategy_transitions 
                           if t.participant == observer]
    if observer_transitions:
      recent_transition = observer_transitions[-1]
      observation += f"Last strategy change: {recent_transition.from_strategy} → {recent_transition.to_strategy}\n"
      observation += f"- Effectiveness before: {recent_transition.effectiveness_before:.0%}\n"

    # Performance metrics
    performance_data = [(k, v) for k, v in self._performance_metrics.items() 
                       if k[0] == observer]
    if performance_data:
      observation += "\nStrategy Performance:\n"
      for (participant, strategy), metrics in performance_data:
        observation += f"- {strategy}: {metrics.success_rate:.0%} success, "
        observation += f"{metrics.adaptability_score:.0%} adaptability\n"

    # Learning insights
    if observer in self._learning_indicators:
      recent_learning = self._learning_indicators[observer][-3:]  # Last 3
      if recent_learning:
        observation += f"\nRecent learning events: {len(recent_learning)}\n"

    # Strategy landscape
    other_strategies = {p: s for p, s in self._current_strategies.items() 
                       if p != observer}
    if other_strategies:
      strategy_counts = {}
      for strategy in other_strategies.values():
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
      
      observation += "\nStrategy landscape:\n"
      for strategy, count in strategy_counts.items():
        observation += f"- {strategy}: {count} participants\n"

    # Innovation opportunities
    if self._innovation_events:
      recent_innovations = [i for i in self._innovation_events 
                           if i['round'] > context.current_round - 5]
      if recent_innovations:
        observation += f"\nRecent innovations: {len(recent_innovations)}\n"

    # Strategic recommendations
    observer_snapshots = [s for s in self._strategy_snapshots 
                         if s.participant == observer]
    if len(observer_snapshots) >= 3:
      recent_effectiveness = self._calculate_strategy_effectiveness(
          observer, current_strategy, context.current_round
      )
      
      if recent_effectiveness < 0.4:
        observation += "\n💡 Consider strategy adaptation - current approach shows limited effectiveness\n"
      elif recent_effectiveness > 0.8:
        observation += "\n✅ Current strategy performing well - maintain approach\n"

    return observation

  def get_module_report(self) -> str:
    """Get strategy evolution report."""
    report = "STRATEGY EVOLUTION REPORT:\n\n"

    # Strategy distribution
    if self._current_strategies:
      strategy_counts = {}
      for strategy in self._current_strategies.values():
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
      
      report += "Current Strategy Distribution:\n"
      for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
        report += f"- {strategy}: {count} participants\n"

    # Evolution dynamics
    if self._strategy_transitions:
      report += f"\nStrategy Transitions: {len(self._strategy_transitions)}\n"
      
      transition_patterns = {}
      for transition in self._strategy_transitions:
        pattern = f"{transition.from_strategy} → {transition.to_strategy}"
        transition_patterns[pattern] = transition_patterns.get(pattern, 0) + 1
      
      report += "Most common transitions:\n"
      for pattern, count in sorted(transition_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        report += f"- {pattern}: {count}\n"

    # Performance analysis
    if self._performance_metrics:
      report += "\nStrategy Performance:\n"
      for (participant, strategy), metrics in self._performance_metrics.items():
        if metrics.rounds_active >= 3:  # Only report strategies with sufficient data
          report += f"- {participant} ({strategy}): "
          report += f"{metrics.success_rate:.0%} success, "
          report += f"{metrics.adaptability_score:.0%} adaptability\n"

    # Learning indicators
    total_learning_events = sum(len(events) for events in self._learning_indicators.values())
    if total_learning_events > 0:
      report += f"\nLearning Activity: {total_learning_events} events\n"
      
      most_learning = max(self._learning_indicators.items(), key=lambda x: len(x[1]))
      report += f"Most adaptive: {most_learning[0]} ({len(most_learning[1])} learning events)\n"

    # Innovation summary
    if self._innovation_events:
      report += f"\nInnovation Events: {len(self._innovation_events)}\n"
      
      innovation_types = {}
      for innovation in self._innovation_events:
        itype = innovation['innovation_type']
        innovation_types[itype] = innovation_types.get(itype, 0) + 1
      
      for itype, count in innovation_types.items():
        report += f"- {itype}: {count}\n"

      # Top innovators
      innovators = {}
      for innovation in self._innovation_events:
        participant = innovation['participant']
        innovators[participant] = innovators.get(participant, 0) + 1
      
      if innovators:
        top_innovator = max(innovators.items(), key=lambda x: x[1])
        report += f"Top innovator: {top_innovator[0]} ({top_innovator[1]} innovations)\n"

    # Strategy diffusion analysis
    if self._strategy_diffusion:
      report += "\nStrategy Diffusion:\n"
      for strategy, diffusion_paths in self._strategy_diffusion.items():
        if diffusion_paths:
          total_diffusion = sum(diffusion_paths.values())
          report += f"- {strategy}: {total_diffusion} diffusion events\n"

    # Evolution insights
    if len(self._strategy_transitions) > 5:
      # Calculate adaptation rate
      recent_transitions = len([t for t in self._strategy_transitions 
                               if t.transition_round > max(t.transition_round for t in self._strategy_transitions) - 10])
      adaptation_rate = recent_transitions / 10  # Per round
      
      if adaptation_rate > 0.5:
        report += "\n🚀 High adaptation rate - dynamic strategy evolution\n"
      elif adaptation_rate < 0.1:
        report += "\n📊 Stable strategy phase - low evolution rate\n"

    return report

  def get_state(self) -> str:
    """Get the component state for saving/restoring."""
    state_dict = {
        'snapshots': len(self._strategy_snapshots),
        'transitions': len(self._strategy_transitions),
        'current_strategies': len(self._current_strategies),
        'innovations': len(self._innovation_events),
    }
    return str(state_dict)

  def set_state(self, state: str) -> None:
    """Set the component state from a saved string."""
    # Since this tracks dynamic data, we only restore basic structure
    pass


# Register the module
negotiation_modules.NegotiationGMModuleRegistry.register(
    'strategy_evolution',
    StrategyEvolutionGM
)