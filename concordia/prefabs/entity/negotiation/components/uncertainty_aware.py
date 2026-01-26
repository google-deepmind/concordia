"""Uncertainty-aware component for probabilistic reasoning in negotiations."""

import dataclasses
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from concordia.typing import entity_component
from concordia.typing import entity as entity_lib


@dataclasses.dataclass
class BeliefDistribution:
    """Represents a belief about a parameter with uncertainty."""
    name: str
    mean: float
    std: float
    confidence: float  # 0-1, how confident we are in this estimate
    evidence_count: int = 0  # Number of observations supporting this belief
    last_updated: Optional[str] = None

    def sample(self, n: int = 1) -> Union[float, List[float]]:
        """Sample from the belief distribution."""
        samples = np.random.normal(self.mean, self.std, n)
        return samples[0] if n == 1 else samples.tolist()

    def update_with_evidence(self, observation: float, reliability: float = 1.0):
        """Bayesian update with new evidence."""
        # Simple Bayesian updating assuming normal distributions
        prior_precision = 1 / (self.std ** 2)
        evidence_precision = reliability / (self.std ** 2)  # Reliability affects precision

        # Update mean (weighted average)
        total_precision = prior_precision + evidence_precision
        new_mean = (prior_precision * self.mean + evidence_precision * observation) / total_precision

        # Update standard deviation (precision increases)
        new_std = 1 / math.sqrt(total_precision)

        # Update confidence based on evidence accumulation
        self.evidence_count += 1
        self.confidence = min(0.95, self.confidence + 0.05 * reliability)

        self.mean = new_mean
        self.std = max(0.01, new_std)  # Prevent std from becoming too small

    def get_confidence_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for the belief."""
        z_score = 1.96 if level == 0.95 else 2.58  # 95% or 99%
        margin = z_score * self.std
        return (self.mean - margin, self.mean + margin)


@dataclasses.dataclass
class ScenarioAnalysis:
    """Analysis of different negotiation scenarios under uncertainty."""
    scenario_name: str
    probability: float
    expected_value: float
    worst_case: float
    best_case: float
    key_assumptions: List[str]
    risk_factors: List[str]


@dataclasses.dataclass
class InformationValue:
    """Value of gathering specific information."""
    question: str
    expected_value_gain: float
    cost_estimate: float
    net_value: float
    confidence_improvement: float
    uncertainty_reduction: float


class UncertaintyAware(entity_component.ContextComponent):
    """Component for probabilistic reasoning and uncertainty management in negotiations."""

    def __init__(
        self,
        model: Any,
        confidence_threshold: float = 0.7,
        risk_tolerance: float = 0.3,
        information_gathering_budget: float = 0.1,
    ):
        """Initialize uncertainty-aware component.

        Args:
            model: Language model for analysis
            confidence_threshold: Minimum confidence for decisions
            risk_tolerance: Tolerance for uncertainty (0-1)
            information_gathering_budget: Fraction of value to spend on info gathering
        """
        self._model = model
        self._confidence_threshold = confidence_threshold
        self._risk_tolerance = risk_tolerance
        self._info_budget = information_gathering_budget

        # Belief state tracking
        self._beliefs: Dict[str, BeliefDistribution] = {}
        self._scenario_probabilities: Dict[str, float] = {}
        self._uncertainty_sources: List[str] = []
        self._information_gaps: List[str] = []

        # Initialize common negotiation beliefs
        self._initialize_default_beliefs()

    def _initialize_default_beliefs(self):
        """Initialize default beliefs about negotiation parameters."""
        # Counterpart's reservation value (start with high uncertainty)
        self._beliefs['counterpart_reservation'] = BeliefDistribution(
            name='Counterpart Reservation Value',
            mean=50000,  # Will be adjusted based on context
            std=25000,   # High initial uncertainty
            confidence=0.3
        )

        # Counterpart's flexibility
        self._beliefs['counterpart_flexibility'] = BeliefDistribution(
            name='Counterpart Flexibility',
            mean=0.5,   # Neutral assumption
            std=0.3,    # Moderate uncertainty
            confidence=0.4
        )

        # Deal success probability
        self._beliefs['deal_probability'] = BeliefDistribution(
            name='Deal Success Probability',
            mean=0.6,   # Moderately optimistic
            std=0.2,    # Moderate uncertainty
            confidence=0.5
        )

        # Market conditions impact
        self._beliefs['market_conditions'] = BeliefDistribution(
            name='Market Conditions Impact',
            mean=0.0,   # Neutral market
            std=0.4,    # High uncertainty about external factors
            confidence=0.3
        )

    def _analyze_uncertainty_context(self, context: str) -> Dict[str, Any]:
        """Analyze context for uncertainty indicators and information gaps."""
        prompt = f"""Analyze this negotiation context for uncertainty and missing information:

Context: {context}

Identify and assess:
1. What key information is missing or uncertain?
2. What are the main sources of uncertainty?
3. How confident can we be about different aspects?
4. What information would be most valuable to gather?
5. What are the potential scenarios we should consider?

Format your response as:
MISSING_INFO: [List key missing information]
UNCERTAINTY_SOURCES: [List main uncertainty sources]
CONFIDENCE_LEVELS: [High/Medium/Low for different aspects]
VALUABLE_INFO: [Most valuable information to gather]
SCENARIOS: [Key scenarios to consider]"""

        response = self._model.sample_text(prompt)

        # Parse response
        analysis = {
            'missing_info': [],
            'uncertainty_sources': [],
            'confidence_levels': {},
            'valuable_info': [],
            'scenarios': []
        }

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('MISSING_INFO:'):
                current_section = 'missing_info'
                analysis['missing_info'].append(line[13:].strip())
            elif line.startswith('UNCERTAINTY_SOURCES:'):
                current_section = 'uncertainty_sources'
                analysis['uncertainty_sources'].append(line[20:].strip())
            elif line.startswith('CONFIDENCE_LEVELS:'):
                current_section = 'confidence_levels'
                # Parse confidence levels
            elif line.startswith('VALUABLE_INFO:'):
                current_section = 'valuable_info'
                analysis['valuable_info'].append(line[14:].strip())
            elif line.startswith('SCENARIOS:'):
                current_section = 'scenarios'
                analysis['scenarios'].append(line[10:].strip())
            elif current_section and line and not any(line.startswith(prefix) for prefix in ['MISSING_INFO:', 'UNCERTAINTY_SOURCES:', 'CONFIDENCE_LEVELS:', 'VALUABLE_INFO:', 'SCENARIOS:']):
                if current_section in ['missing_info', 'uncertainty_sources', 'valuable_info', 'scenarios']:
                    analysis[current_section].append(line)

        return analysis

    def _update_beliefs_from_context(self, context: str):
        """Update beliefs based on new context information."""
        # Extract numerical hints from context
        prompt = f"""Extract any numerical or quantitative information from this context that could update our beliefs:

Context: {context}

Look for information about:
1. Budget ranges or values mentioned
2. Timeline flexibility indicators
3. Priority or importance signals
4. Market condition indicators
5. Relationship quality signals

Provide estimates with confidence levels:
BUDGET_INFO: [value estimate] [confidence 0-1]
TIMELINE_INFO: [flexibility estimate 0-1] [confidence 0-1]
PRIORITY_INFO: [priority level 0-1] [confidence 0-1]
MARKET_INFO: [condition estimate -1 to 1] [confidence 0-1]
RELATIONSHIP_INFO: [quality estimate 0-1] [confidence 0-1]"""

        response = self._model.sample_text(prompt)

        # Parse and update beliefs
        lines = response.split('\n')
        for line in lines:
            if line.startswith('BUDGET_INFO:'):
                parts = line[12:].strip().split()
                if len(parts) >= 2:
                    try:
                        value = float(parts[0])
                        confidence = float(parts[1])
                        self._beliefs['counterpart_reservation'].update_with_evidence(value, confidence)
                    except ValueError:
                        pass
            elif line.startswith('TIMELINE_INFO:'):
                parts = line[13:].strip().split()
                if len(parts) >= 2:
                    try:
                        flexibility = float(parts[0])
                        confidence = float(parts[1])
                        self._beliefs['counterpart_flexibility'].update_with_evidence(flexibility, confidence)
                    except ValueError:
                        pass
            # Similar parsing for other info types

    def _generate_scenarios(self, n_scenarios: int = 5) -> List[ScenarioAnalysis]:
        """Generate scenarios based on current beliefs."""
        scenarios = []

        # Define scenario types with different assumptions
        scenario_templates = [
            {
                'name': 'Optimistic',
                'probability': 0.2,
                'assumptions': ['Counterpart is flexible', 'Market conditions favor deal', 'High trust level'],
                'multipliers': {'reservation': 1.2, 'flexibility': 1.3, 'deal_prob': 1.4}
            },
            {
                'name': 'Realistic',
                'probability': 0.4,
                'assumptions': ['Average market conditions', 'Moderate counterpart flexibility', 'Standard process'],
                'multipliers': {'reservation': 1.0, 'flexibility': 1.0, 'deal_prob': 1.0}
            },
            {
                'name': 'Pessimistic',
                'probability': 0.2,
                'assumptions': ['Counterpart has strong alternatives', 'Market pressure', 'Time constraints'],
                'multipliers': {'reservation': 0.8, 'flexibility': 0.7, 'deal_prob': 0.6}
            },
            {
                'name': 'High Uncertainty',
                'probability': 0.1,
                'assumptions': ['Major unknown factors', 'Volatile environment', 'Information asymmetry'],
                'multipliers': {'reservation': 0.9, 'flexibility': 0.8, 'deal_prob': 0.5}
            },
            {
                'name': 'Breakthrough',
                'probability': 0.1,
                'assumptions': ['Unexpected positive developments', 'Strategic alignment', 'Win-win discovery'],
                'multipliers': {'reservation': 1.3, 'flexibility': 1.4, 'deal_prob': 1.6}
            }
        ]

        for template in scenario_templates:
            # Calculate scenario values based on current beliefs and multipliers
            reservation_belief = self._beliefs['counterpart_reservation']
            flexibility_belief = self._beliefs['counterpart_flexibility']
            deal_prob_belief = self._beliefs['deal_probability']

            # Sample values for this scenario
            reservation_sample = reservation_belief.sample() * template['multipliers']['reservation']
            flexibility_sample = flexibility_belief.sample() * template['multipliers']['flexibility']
            deal_prob_sample = deal_prob_belief.sample() * template['multipliers']['deal_prob']

            # Estimate expected value for scenario
            expected_value = reservation_sample * deal_prob_sample * (1 + flexibility_sample)

            scenario = ScenarioAnalysis(
                scenario_name=template['name'],
                probability=template['probability'],
                expected_value=expected_value,
                worst_case=expected_value * 0.7,  # Assume 30% downside
                best_case=expected_value * 1.3,   # Assume 30% upside
                key_assumptions=template['assumptions'],
                risk_factors=[f"Low confidence in {belief.name}" for belief in self._beliefs.values() if belief.confidence < 0.5]
            )
            scenarios.append(scenario)

        return scenarios

    def _calculate_information_values(self, context: str) -> List[InformationValue]:
        """Calculate value of gathering different types of information."""
        # Current uncertainty level (higher uncertainty = more value from information)
        avg_confidence = np.mean([belief.confidence for belief in self._beliefs.values()])
        uncertainty_level = 1 - avg_confidence

        # Define potential information gathering opportunities
        info_opportunities = [
            {
                'question': 'What is your budget range for this project?',
                'target_belief': 'counterpart_reservation',
                'confidence_gain': 0.3,
                'cost_factor': 0.1
            },
            {
                'question': 'How flexible are you on timeline and terms?',
                'target_belief': 'counterpart_flexibility',
                'confidence_gain': 0.25,
                'cost_factor': 0.05
            },
            {
                'question': 'What are your main priorities and constraints?',
                'target_belief': 'deal_probability',
                'confidence_gain': 0.2,
                'cost_factor': 0.08
            },
            {
                'question': 'How do current market conditions affect your decision?',
                'target_belief': 'market_conditions',
                'confidence_gain': 0.15,
                'cost_factor': 0.03
            }
        ]

        information_values = []

        for opportunity in info_opportunities:
            target_belief = self._beliefs.get(opportunity['target_belief'])
            if target_belief:
                # Calculate expected value gain
                current_uncertainty = 1 - target_belief.confidence
                potential_uncertainty_reduction = opportunity['confidence_gain'] * current_uncertainty

                # Estimate value based on how much this belief affects decisions
                base_value = 10000  # Baseline value for information
                value_gain = base_value * potential_uncertainty_reduction * uncertainty_level

                # Estimate cost
                cost = base_value * opportunity['cost_factor']

                info_value = InformationValue(
                    question=opportunity['question'],
                    expected_value_gain=value_gain,
                    cost_estimate=cost,
                    net_value=value_gain - cost,
                    confidence_improvement=opportunity['confidence_gain'],
                    uncertainty_reduction=potential_uncertainty_reduction
                )
                information_values.append(info_value)

        # Sort by net value
        information_values.sort(key=lambda x: x.net_value, reverse=True)
        return information_values

    def _generate_uncertainty_guidance(self, context: str) -> str:
        """Generate comprehensive uncertainty-aware guidance."""
        # Analyze uncertainty in context
        uncertainty_analysis = self._analyze_uncertainty_context(context)

        # Generate scenarios
        scenarios = self._generate_scenarios()

        # Calculate information values
        info_values = self._calculate_information_values(context)

        # Generate guidance
        guidance = f"""🎲 Uncertainty-Aware Analysis

**Current Belief State:**
"""

        for name, belief in self._beliefs.items():
            ci_low, ci_high = belief.get_confidence_interval()
            guidance += f"• {belief.name}: {belief.mean:.1f} ± {belief.std:.1f} (confidence: {belief.confidence:.2f})\n"
            guidance += f"  95% CI: [{ci_low:.1f}, {ci_high:.1f}]\n"

        guidance += f"\n**Scenario Analysis:**\n"
        for scenario in scenarios:
            guidance += f"• {scenario.scenario_name} ({scenario.probability:.1%}): "
            guidance += f"Expected value {scenario.expected_value:.0f}, "
            guidance += f"Range [{scenario.worst_case:.0f}, {scenario.best_case:.0f}]\n"

        guidance += f"\n**Information Gathering Opportunities:**\n"
        for info in info_values[:3]:  # Top 3 opportunities
            if info.net_value > 0:
                guidance += f"• \"{info.question[:50]}...\"\n"
                guidance += f"  Value: ${info.net_value:.0f}, Confidence gain: +{info.confidence_improvement:.1%}\n"

        guidance += f"\n**Uncertainty Management Strategy:**\n"

        # Calculate overall uncertainty level
        avg_confidence = np.mean([belief.confidence for belief in self._beliefs.values()])

        if avg_confidence < self._confidence_threshold:
            guidance += f"• HIGH UNCERTAINTY DETECTED (confidence: {avg_confidence:.1%})\n"
            guidance += f"• Recommend information gathering before major commitments\n"
            guidance += f"• Consider contingent offers and flexible terms\n"
            guidance += f"• Focus on robust strategies that work across scenarios\n"
        else:
            guidance += f"• Moderate uncertainty (confidence: {avg_confidence:.1%})\n"
            guidance += f"• Proceed with standard negotiation approach\n"
            guidance += f"• Monitor for new information that could change beliefs\n"

        guidance += f"\n**Robust Decision Principles:**\n"
        guidance += f"• Base decisions on {avg_confidence:.1%} confidence in current beliefs\n"
        guidance += f"• Prepare contingencies for {scenarios[2].scenario_name.lower()} scenario\n"
        guidance += f"• Maintain flexibility for belief updates\n"
        guidance += f"• Balance information gathering with negotiation progress\n"

        return guidance

    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        """Provide uncertainty-aware guidance before action."""
        context = action_spec.call_to_action

        # Update beliefs based on new context
        self._update_beliefs_from_context(context)

        # Generate uncertainty-aware guidance
        guidance = self._generate_uncertainty_guidance(context)

        return f"\n{guidance}"

    def post_act(self, action_attempt: str) -> str:
        """Update uncertainty state based on action taken."""
        # Analyze action for information gathering or commitment
        if '?' in action_attempt or 'ask' in action_attempt.lower() or 'question' in action_attempt.lower():
            # Information gathering action
            for belief in self._beliefs.values():
                belief.confidence = min(0.95, belief.confidence + 0.02)  # Small confidence boost

        return ""

    def observe(self, observation: str) -> None:
        """Process observations to update beliefs."""
        # Update beliefs based on new observations
        self._update_beliefs_from_context(observation)

        # Extract any explicit information from counterpart
        if 'budget' in observation.lower() or '$' in observation:
            # Try to extract budget information
            import re
            numbers = re.findall(r'\$?(\d+(?:,\d+)*(?:\.\d+)?)', observation)
            if numbers:
                try:
                    value = float(numbers[0].replace(',', ''))
                    self._beliefs['counterpart_reservation'].update_with_evidence(value, 0.8)
                except ValueError:
                    pass

    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        return {
            'beliefs': {
                name: {
                    'mean': belief.mean,
                    'std': belief.std,
                    'confidence': belief.confidence,
                    'evidence_count': belief.evidence_count,
                }
                for name, belief in self._beliefs.items()
            },
            'avg_confidence': np.mean([belief.confidence for belief in self._beliefs.values()]),
            'uncertainty_level': 1 - np.mean([belief.confidence for belief in self._beliefs.values()]),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set component state."""
        for name, belief_data in state.get('beliefs', {}).items():
            if name in self._beliefs:
                self._beliefs[name].mean = belief_data.get('mean', self._beliefs[name].mean)
                self._beliefs[name].std = belief_data.get('std', self._beliefs[name].std)
                self._beliefs[name].confidence = belief_data.get('confidence', self._beliefs[name].confidence)
                self._beliefs[name].evidence_count = belief_data.get('evidence_count', self._beliefs[name].evidence_count)

    def get_action_attempt(
        self,
        context: Any,  # ComponentContextMapping
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        """Generate uncertainty-aware negotiation action with robust decision making."""
        situation_context = action_spec.call_to_action
        
        # Update beliefs from current context
        self._update_beliefs_from_context(situation_context)
        
        # Analyze uncertainty in the situation
        uncertainty_analysis = self._analyze_uncertainty_context(situation_context)
        
        # Generate scenarios for decision making
        scenarios = self._generate_scenarios()
        
        # Calculate information values
        info_values = self._calculate_information_values(situation_context)
        
        # Calculate overall confidence level
        avg_confidence = np.mean([belief.confidence for belief in self._beliefs.values()])
        uncertainty_level = 1 - avg_confidence
        
        # Decide whether to gather information or make a proposal
        should_gather_info = (
            avg_confidence < self._confidence_threshold and 
            info_values and 
            info_values[0].net_value > 0
        )
        
        if should_gather_info:
            # Generate information-gathering action
            top_info_question = info_values[0]
            prompt = f"""Based on uncertainty analysis, generate an information-gathering negotiation action:

Situation: {situation_context}

Uncertainty Analysis:
- Average confidence: {avg_confidence:.1%}
- Uncertainty level: {uncertainty_level:.1%}
- Missing information: {', '.join(uncertainty_analysis.get('missing_info', []))}

Most Valuable Information to Gather:
Question: {top_info_question.question}
Expected Value: ${top_info_question.expected_value_gain:.0f}
Confidence Improvement: +{top_info_question.confidence_improvement:.1%}

Generate a negotiation action that:
1. Asks the most valuable information-gathering question
2. Explains why this information would help both parties
3. Demonstrates thoughtful preparation and analysis
4. Maintains negotiation momentum while reducing uncertainty
5. Shows professional competence despite information gaps

Action:"""
        
        else:
            # Generate proposal/response action with uncertainty management
            best_scenario = max(scenarios, key=lambda s: s.expected_value * s.probability)
            worst_scenario = min(scenarios, key=lambda s: s.expected_value * s.probability)
            
            # Get confidence intervals for key beliefs
            reservation_ci = self._beliefs['counterpart_reservation'].get_confidence_interval()
            flexibility_ci = self._beliefs['counterpart_flexibility'].get_confidence_interval()
            
            prompt = f"""Based on uncertainty analysis and scenario planning, generate a robust negotiation action:

Situation: {situation_context}

Current Beliefs (with confidence intervals):
- Counterpart reservation: ${self._beliefs['counterpart_reservation'].mean:.0f} (95% CI: ${reservation_ci[0]:.0f} - ${reservation_ci[1]:.0f})
- Counterpart flexibility: {self._beliefs['counterpart_flexibility'].mean:.1f} (95% CI: {flexibility_ci[0]:.1f} - {flexibility_ci[1]:.1f})
- Deal probability: {self._beliefs['deal_probability'].mean:.1%}

Scenario Analysis:
- Best case: {best_scenario.scenario_name} - Expected value: ${best_scenario.expected_value:.0f}
- Worst case: {worst_scenario.scenario_name} - Expected value: ${worst_scenario.expected_value:.0f}
- Average confidence: {avg_confidence:.1%}

Risk Management:
- Risk tolerance: {self._risk_tolerance:.1%}
- Key uncertainties: {', '.join(uncertainty_analysis.get('uncertainty_sources', []))}

Generate a negotiation action that:
1. Makes a robust proposal that works across scenarios
2. Acknowledges and manages key uncertainties
3. Includes contingencies for different outcomes  
4. Demonstrates analytical sophistication
5. Balances confidence with appropriate caution given uncertainty level

Action:"""

        response = self._model.sample_text(prompt)
        
        # Clean up response
        action = response.strip()
        if action.lower().startswith('action:'):
            action = action[7:].strip()
        
        # Add uncertainty framing based on confidence level
        if should_gather_info:
            action = f"To make the best decision for both of us, {action.lower()}"
        elif avg_confidence < 0.5:
            action = f"While there are several factors to consider, {action.lower()}"
        elif avg_confidence > 0.8:
            action = f"Based on our analysis, {action.lower()}"
        
        return action

    def update(self) -> None:
        """Update uncertainty-aware component state."""
        # Gradually decay confidence over time if no new evidence
        for belief in self._beliefs.values():
            if belief.evidence_count == 0:
                belief.confidence *= 0.99  # Slow decay
                belief.std = min(belief.std * 1.01, belief.std * 2)  # Increase uncertainty
