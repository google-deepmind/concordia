"""Uncertainty-aware component for probabilistic reasoning in negotiations."""

import dataclasses
import math
import random
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from concordia.typing import entity_component
from concordia.typing import entity as entity_lib
from pydantic import BaseModel, ValidationError, Field

# To model pricing distribution beliefs
@dataclasses.dataclass
class NormalInverseGamma:
    '''Represents a Normal-Inverse-Gamma conjugate prior for Bayesian updating.'''
    name: str 
    mu: float  
    lambda_: float
    a: float
    b: float
    confidence: float = 0.5  # Initial confidence level (0-1)
    evidence_count: int = 0  # Number of observations supporting this belief
    last_updated: Optional[str] = None

    # Helpers
    def _get_t_critical(self, confidence: float, df: float) -> float:
        """
        Approximates the t-critical value without scipy.
        Uses a lookup table for small df and Z-score for large df.
        Confidence should only be 0.90, 0.95, or 0.99.
        """
        t_table_90 = {
            1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
            6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
            12: 1.782, 14: 1.761, 16: 1.746, 18: 1.734, 20: 1.725,
            25: 1.708, 30: 1.697
        }
        t_table_95 = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            12: 2.179, 14: 2.145, 16: 2.120, 18: 2.101, 20: 2.086,
            25: 2.060, 30: 2.042
        }
        t_table_99 = {
            1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032,
            6: 3.707, 7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169,
            12: 3.055, 14: 2.977, 16: 2.921, 18: 2.878, 20: 2.845,
            25: 2.787, 30: 2.750
        }

        if confidence == 0.90:
            t_table = t_table_90
            if df > 30:
                return 1.645  # use normal approximation
        elif confidence == 0.95:
            t_table = t_table_95
            if df > 30:
                return 1.96  # use normal approximation
        elif confidence == 0.99:
            t_table = t_table_99
            if df > 30:
                return 2.576  # use normal approximation
        
        # Find the closest key in the table for small df
        closest_df = max(k for k in t_table if k <= df)
        return t_table[closest_df]

    @property
    def get_expected_mean(self) -> float: 
        '''Calculate the expected mean of the distribution.'''
        return self.mu 

    @property
    def get_expected_variance(self) -> float:
        '''Calculate the expected variance of the distribution.'''
        if self.a > 1:
            return self.b / (self.a - 1)
        else:
            return self.b 
        
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        '''Calculate the predictive mean range for a given confidence level.'''
        df = 2 * self.a
        scale = np.sqrt(self.b * (self.lambda_ + 1) / (self.lambda_ * self.a))

        t_value = self._get_t_critical(confidence, df)
        margin = t_value * scale

        return (self.mu - margin, self.mu + margin)
    
    def sample(self, n: int = 1) -> Union[float, List[float]]:
        '''Sample from the Normal-Inverse-Gamma distribution.'''
        rng=np.random.default_rng()
        tau_sq_samples = 1 / rng.gamma(self.a, 1 / self.b, n)# first sample from inverse gamma distribution
        mu_samples=rng.normal(self.mu, np.sqrt(1 / (self.lambda_ * tau_sq_samples)), n) # then sample from normal distribution
        return mu_samples[0] if n == 1 else mu_samples.tolist()
    
    def update_with_evidence(self, observation: float, reliability: float = 1.0):
        '''Update the distribution with new evidence using standard Bayesian update.'''
        # Update mean and lambda
        new_lambda = self.lambda_ + reliability
        new_mu = (self.lambda_ * self.mu + reliability * observation) / new_lambda

        # Update variance parameters(a and b)
        new_a = self.a + 0.5 * reliability # 1 observation
        diff = observation - self.mu
        new_b = self.b + (self.lambda_ * reliability * (diff **2)) / (2 * new_lambda)

        # update confidence
        self.confidence= min(0.95, self.confidence + 0.05 * reliability)
        self.evidence_count += 1

        # Assign updated parameters
        self.mu = new_mu
        self.lambda_ = new_lambda
        self.a = new_a
        self.b = new_b


    # TODO: revise this method to be more principled
    def update_flexibility_multiplicative(self, flexibility_score: float, reliability: float = 1.0):
            """
            Updates variance by scaling the current belief.
            
            Args:
                flexibility_score: 
                    0.0 (Rigid)    -> Shrink variance (Multiplier < 1.0)
                    0.5 (Neutral)  -> No change (Multiplier = 1.0)
                    1.0 (Flexible) -> Grow variance (Multiplier > 1.0)
                reliability: 
                    Multiplier dampening factor (0.0 to 1.0)
            """
            # Define max parameters to avoid extreme scaling
            max_shrink = 0.5
            max_grow = 2.0
            
            # 2. Map flexibility score to some form of multiplier
            if flexibility_score < 0.5:
                # Range [0.0, 0.5] maps to [max_shrink, 1.0]
                raw_multiplier = max_shrink + (flexibility_score * 2 * (1.0 - max_shrink))
            else:
                # Range [0.5, 1.0] maps to [1.0, max_grow]
                raw_multiplier = 1.0 + ((flexibility_score - 0.5) * 2 * (max_grow - 1.0))

            # If reliability is low, pull the multiplier back towards 1.0 (no change)
            # Formula: Final = 1.0 + (Target - 1.0) * Reliability
            final_multiplier = 1.0 + (raw_multiplier - 1.0) * reliability
            
            self.b *= final_multiplier
            
            # to ensure b stays within reasonable bounds (TODO: edit if needed)
            self.b = max(1e4, min(self.b, 1e11))
    
@dataclasses.dataclass
class BeliefDistribution:
    """Represents a belief about a parameter with uncertainty."""
    name: str
    mean: float
    std: float
    confidence: float  # 0-1, how confident we are in this estimate
    evidence_count: int = 0  # Number of observations supporting this belief
    last_updated: Optional[str] = None

    @property
    def get_expected_mean(self) -> float:
        """Get the expected mean of the belief."""
        return self.mean
    
    @property
    def get_expected_variance(self) -> float:
        """Get the expected variance of the belief."""
        return self.std ** 2
    
    def sample(self, n: int = 1) -> Union[float, List[float]]:
        """Sample from the belief distribution."""
        samples = np.random.normal(self.mean, self.std, n)
        return samples[0] if n == 1 else samples.tolist()

    def update_with_evidence(self, observation: float, reliability: float = 1.0):
        """Bayesian update with new evidence."""
        # Simple Bayesian updating assuming normal distributions (both prior and likelihood are normal)
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


# Dataclasses for structured analysis outputs
class UncertainInfo(BaseModel):
    '''Schema for uncertain information.'''
    claim: str
    confidence_level: float # 0-1

class UncertaintyContext(BaseModel):
    '''Schema for uncertainty contexts.'''
    missing_info: List[str]
    uncertainty_sources: List[UncertainInfo]
    valuable_info: List[str]
    scenarios: List[str]

@dataclasses.dataclass
class ScenarioAnalysis:
    """Analysis of different negotiation scenarios under uncertainty."""
    scenario_name: str
    likelihood: float
    value: float
    key_assumptions: List[str]
    risk_factors: List[str]


class InformationValue(BaseModel):
    """Value of gathering specific information."""
    question: str = Field(description="The specific question to ask to gather information")
    priority_score: float = Field(ge=0, le=1, description="The degree of reduction in uncertainty (0-1); higher means more valuable")
    cost_factor: float = Field(ge=0, le=1, description="Relative cost of obtaining this information (0-1); higher means more costly")

class InformationValueResponse(BaseModel):
    information_values: Optional[List[InformationValue]] = None

class UpdateOwnBeliefInfoMetadata(BaseModel):
    '''Metadata for belief info updates during negotiations.'''
    estimate: float
    confidence: float

class UpdateOwnBeliefInfo(BaseModel):
    '''Information to update belief during negotiations.'''
    reservation_info: Optional[UpdateOwnBeliefInfoMetadata] = Field(None, description="Information about own reservation value")

class UpdateOpposingBeliefInfoMetadata(BaseModel):
    '''Metadata for belief info updates during negotiations.'''
    estimate: float
    confidence: float

class UpdateOpposingBeliefInfo(BaseModel):
    '''Information to update belief during negotiations.'''
    budget_info: Optional[UpdateOpposingBeliefInfoMetadata] = None
    flexibility_info: Optional[UpdateOpposingBeliefInfoMetadata] = None

class UncertainSeller(entity_component.ContextComponent):
    """Component for probabilistic reasoning and uncertainty management in negotiations. (seller's side)"""

    def __init__(
        self,
        model: Any,
        confidence_threshold: float = 0.7,
        risk_tolerance: float = 0.3,
        information_gathering_budget: float = 0.1,
        own_reservation_: float = 0.0, # note that this is the minimum reservation price for the seller
        mu: float = 0.0,
        lambda_: float = 1.0,
        a: float = 1.0,
        b: float = 1.0,
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
        # TODO: think whether we need a separate belief for own reservation price (should not be to simulate information asymmetry of the product)
        self._own_reservation = own_reservation_ # for seller, we first assume that they are sure of their reservation price
        self._info_budget = information_gathering_budget

        # Belief state tracking
        self._beliefs: Dict[str, BeliefDistribution | NormalInverseGamma] = {}
        self._uncertainty_sources: List[str] = []
        self._information_gaps: List[str] = []

        # Initialize common negotiation beliefs
        self._initialize_default_beliefs(mu, lambda_, a, b, own_reservation_)


    def _initialize_default_beliefs(self, mu: float = 0.0, lambda_: float = 1.0, a: float = 1.0, b: float = 1.0, own_reservation_: float = 0.0):
        """Initialize default beliefs about negotiation parameters."""
        # Counterpart's reservation value (start with high uncertainty)
        self._beliefs['counterpart_reservation'] = NormalInverseGamma(
            name="Counterpart's Reservation Value",
            mu=mu, #TODO: to determine based on initial logs before initialisation of negotiation
            lambda_=lambda_,
            a=a,
            b=b
        )

    def _analyze_uncertainty_context(self, context: str) -> UncertaintyContext:
        """Analyze context for uncertainty indicators and information gaps."""
        formatted_beliefs="\n".join(
            [f"{name}: mean={belief.get_expected_mean:.2f}, std={belief.get_expected_mean:.2f}" for name, belief in self._beliefs.items()] # TODO: might need to add confidence for NIG, but decide later on
        )
        prompt = f"""
        You are looking from the perspective of a seller in a negotiation with perfect information of your house that you plan to sell. The counterpart has imperfect information about your own reservation price and flexibility.
        However, there is uncertainty about the buyer's reservation value and flexibility.
        Based off your current beliefs given below, analyze this negotiation context for uncertainty and missing information:
        Beliefs: {formatted_beliefs}
        Context: {context}

        Identify and assess:
        1. What key information is missing?
        2. What are the main sources of uncertainty and how uncertain are they? Determine the degree of uncertainty based on your current beliefs (0 - 1 scale).
        3. What information would be most valuable to gather given our preferences?
        4. What are the potential scenarios we should consider?

        Format your response using the JSON schema provided. It should be as follows:
        missing_info: [List key missing information]
        uncertainty_sources: List of {{"claim": [uncertain claim], "confidence_level": [0-1]}}
        valuable_info: [List of most valuable information to gather given preferences]
        scenarios: [Key scenarios to consider]"""

        response = self._model.sample_text(prompt, json_schema=UncertaintyContext.model_json_schema())

        # Load JSON response and validate schema
        try:
            analysis = UncertaintyContext.model_validate_json(response)
        except ValidationError as e:
            # current fallback TODO: improve error handling
            analysis = UncertaintyContext(
                missing_info=[],
                uncertainty_sources=[],
                valuable_info=[],
                scenarios=[]
            )

        return analysis

    def _update_counterpart_reservation_from_context(self, context: str):
        """Update beliefs based on new context information."""
        prompt = f"""
        You are looking from the perspective of a seller in a negotiation with perfect information. The counterpart has imperfect information about your own budget and flexibility regarding the house. However, there is uncertainty about the buyer's budget and flexibility.
        Given a context, your task is to extract information regarding the counterpart's budget and flexibility in the negotiation, if there is any:
        
        Context: {context}

        First, focus on extracting BUDGET_INFO, the counterpart's budget or reservation value (in dollars). Determine the confidence level (0-1) through the amount of trust you have in this information.

        After that, extract other relevant information on:
        1. Sense of urgency or timeline flexibility of the sale
        2. Current relationship strength with the counterpart
        Consider these other relevant information to determine FLEXIBILITY_INFO (the flexibility of the sale). Determine the confidence level (0-1) through the amount of trust you have in this information.

        Return a response using the JSON schema provided. Provide estimates with confidence levels:
        budget_info: [value estimate] [confidence 0-1]
        flexibility_info: [flexibility estimate 0-1] [confidence 0-1]
        """

        response = self._model.sample_text(prompt, json_schema=UpdateOpposingBeliefInfo.model_json_schema())

        # Load JSON response
        info_update = UpdateOpposingBeliefInfo.model_validate_json(response)
        if info_update.budget_info:
            self._beliefs['counterpart_reservation'].update_with_evidence(
                info_update.budget_info.estimate,
                info_update.budget_info.confidence
            )
        if info_update.flexibility_info:
            self._beliefs['counterpart_flexibility'].update_flexibility_multiplicative(
                info_update.flexibility_info.estimate,
                info_update.flexibility_info.confidence
            )

    def _generate_scenarios(self) -> List[ScenarioAnalysis]:
        """
        Generate scenarios based on current beliefs.
        We will generate 3 types of scenarios: Optimistic, Pessimistic and Realistic. 
        The expected value for each scenario will be calculated based on sampled beliefs and risk tolerance parameter.
        """
        def _get_risk_adjusted_weights(risk_tolerance: float) -> Dict[str, float]:
            """
            Calculates scenario weights based on risk tolerance.
            
            Logic:
            - We start with a neutral stance: [25% Pessimistic, 50% Realistic, 25% Optimistic]
            - Risk Tolerance acts as a slider moving mass from Pessimistic to Optimistic.
            """
            # TODO: Refine this logic as needed
            # Define neutral likelihood for now
            base_pessimistic = 0.25
            base_optimistic = 0.25
            
            # Scale based on risk tolerance (max movement of 25% from each tail)
            shift = (risk_tolerance - 0.5) * 0.5  
            
            # Calculate likelihoods
            l_pess = base_pessimistic - shift
            l_opt  = base_optimistic + shift
            l_real = 1.0 - l_pess - l_opt # to ensure sum to 1.0
            
            return {
                'Pessimistic': max(0.0, min(1.0, l_pess)),
                'Realistic': max(0.0, min(1.0, l_real)),
                'Optimistic': max(0.0, min(1.0, l_opt))
            }
        likelihoods = _get_risk_adjusted_weights(self._risk_tolerance)
        scenarios = []
        cp_belief=self._beliefs['counterpart_reservation']
        # calculate student t parameters for predictive distribution
        df=2*cp_belief.a
        if df <= 2:
            var_cp = 1e9 # cap for now
        else:
            scale=cp_belief.b*(cp_belief.lambda_+1)/(cp_belief.lambda_*cp_belief.a)
            tail_inflation = df / (df - 2)
            var_cp = scale * tail_inflation
        # counterpart's distribution summary statistics
        mu_cp = cp_belief.mu

        # own distribution's summary statistics
        own_reservation=self._own_reservation # for the seller, we assume they know their own reservation price

        # find surplus; for the seller, its 
        mu_diff = mu_cp - own_reservation

        # TODO: we assume independence for now (i.e. covariance = 0) but assumption is weak since we are talking about the same product. 
        # However, it is fine for now, since we assume maximum variance between the differences => more conservative estimates for ZOPA. 
        zopa_dist = NormalDist(mu_diff, math.sqrt(var_cp + 0)) # variance of own reservation is 0 since we assume full knowledge
        p_upper = 0.5 + (self._confidence_threshold / 2.0)
        
        z_width = NormalDist(mu=0, sigma=1).inv_cdf(p_upper)
        
        val_optimistic = zopa_dist.mean + (z_width * zopa_dist.stdev)
        val_realistic  = zopa_dist.mean 
        val_pessimistic = zopa_dist.mean - (z_width * zopa_dist.stdev)
        # generate scenarios based on mean and std deviations
        for name, likelihood in likelihoods.items():
            if name == 'Pessimistic': 
                scenarios.append(
                    ScenarioAnalysis(
                        scenario_name=f"{name}: Deal Possible" if val_pessimistic > 0 else f"{name}: No Deal",
                        likelihood=likelihood,
                        value=max(0.0, val_pessimistic),
                        key_assumptions=[f"Counterpart reservation at {cp_belief.get_expected_mean:.2f}", f"Your reservation at {own_reservation:.2f}"],
                        risk_factors=[f"Counterpart reservation uncertainty (std: {math.sqrt(cp_belief.get_expected_variance):.2f})"]
                    )
                )
            elif name == 'Realistic':
                scenarios.append(
                    ScenarioAnalysis(
                        scenario_name=f"{name}: Deal Possible" if val_realistic > 0 else f"{name}: No Deal",
                        likelihood=likelihood,
                        value=max(0.0, val_realistic),
                        key_assumptions=[f"Counterpart reservation at {cp_belief.get_expected_mean:.2f}", f"Your reservation at {own_reservation:.2f}"],
                        risk_factors=[f"Counterpart reservation uncertainty (std: {math.sqrt(cp_belief.get_expected_variance):.2f})"]
                    )
                )
            else:
                scenarios.append(
                    ScenarioAnalysis(
                        scenario_name=f"{name}: Deal Possible" if val_optimistic > 0 else f"{name}: No Deal",
                        likelihood=likelihood,
                        value=max(0.0, val_optimistic),
                        key_assumptions=[f"Counterpart reservation at {cp_belief.get_expected_mean:.2f}", f"Your reservation at {own_reservation:.2f}"],
                        risk_factors=[f"Counterpart reservation uncertainty (std: {math.sqrt(cp_belief.get_expected_variance):.2f})"]
                    )
                )
        return scenarios
    
    def _calculate_information_values(self, context: str, uncertainty_analysis: UncertaintyContext) -> List[InformationValue]:
        """Calculate value of gathering different types of information."""
        # Current uncertainty level (higher uncertainty = more value from information)

        # Determine the info_opportunities using a LLM. 
        prompt = f"""
        You are looking from the perspective of a seller in a negotiation with perfect information of your house that you plan to sell. The counterpart has imperfect information about your own reservation price and flexibility.
        However, there is uncertainty about the counterpart's reservation value and flexibility. Given the negotiation context and uncertainty analysis below, identify up to 10 specific questions that can be asked to gather valuable information that affects the counterpart's reservation value and flexibility.

        Context: {context}
        Uncertainty Analysis: {uncertainty_analysis.model_dump_json()}
        
        For each question, estimate 
        1. priority_score (0-1): how much confidence improvement this information could provide.
        2. cost_factor (0-1): the relative cost of obtaining this information based off how tedious/difficult it is to obtain.

        Return a list of information gathering opportunities in the JSON schema provided.
        """

        response = self._model.sample_text(prompt, json_schema=InformationValueResponse.model_json_schema())
        # Load JSON response and validate schema
        try:
            info_opportunities = [InformationValueResponse.model_validate_json(item) for item in response]
        except ValidationError as e:
            # current fallback TODO: improve error handling
            info_opportunities = []

        # Sort by rank
        info_opportunities.sort(key=lambda x: x.priority_score, reverse=True)
        return info_opportunities

    def _generate_uncertainty_guidance(self, context: str) -> str:
        """Generate comprehensive uncertainty-aware guidance."""
        # Analyze uncertainty in context
        uncertainty_analysis = self._analyze_uncertainty_context(context)

        # Generate scenarios
        scenarios = self._generate_scenarios()

        # Calculate information values
        info_values = self._calculate_information_values(context, uncertainty_analysis)

        # Generate guidance
        guidance = f""" Uncertainty-Aware Analysis
**Current Belief State:**
"""

        for name, belief in self._beliefs.items():
            guidance += f"• {belief.name}: {belief.get_expected_mean:.1f} ± {belief.get_expected_variance:.1f} \n"
            guidance += f"  95% CI: [{belief.get_confidence_interval()[0]:.1f}, {belief.get_confidence_interval()[1]:.1f}]\n"

        guidance += f"\n**Scenario Analysis:**\n"
        for scenario in scenarios:
            guidance += f"• {scenario.scenario_name} ({scenario.likelihood:.1%}): "
            guidance += f"Expected value {scenario.value:.0f}, "

        guidance += f"\n**Information Gathering Opportunities:**\n"
        for info in info_values[:3]:  # Top 3 opportunities
            if info.net_value > 0:
                guidance += f"• \"{info.question[:50]}...\"\n"
                guidance += f"  Priority Score: ${info.priority_score:.0f}, Cost Factor: +{info.cost_factor:.2f}\n"

        guidance += f"\n**Uncertainty Management Strategy:**\n"

        # Calculate overall uncertainty level
        avg_uncertainty = np.mean([1 - confidence_level for info in uncertainty_analysis.uncertainty_sources for confidence_level in [info.confidence_level]])

        if 1-avg_uncertainty < self._confidence_threshold:
            guidance += f"• HIGH UNCERTAINTY DETECTED (uncertainty: {avg_uncertainty:.1%})\n"
            guidance += f"• Recommend information gathering before major commitments\n"
            guidance += f"• Consider contingent offers and flexible terms\n"
            guidance += f"• Focus on robust strategies that work across scenarios\n"
        else:
            guidance += f"• Moderate uncertainty (uncertainty: {1-avg_uncertainty:.1%})\n"
            guidance += f"• Proceed with standard negotiation approach\n"
            guidance += f"• Monitor for new information that could change beliefs\n"

        guidance += f"\n**Robust Decision Principles:**\n"
        guidance += f"• Base decisions on {1-avg_uncertainty:.1%} confidence in current beliefs\n"
        guidance += f"• Maintain flexibility for belief updates\n"
        guidance += f"• Balance information gathering with negotiation progress\n"

        return guidance

    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        """Provide uncertainty-aware guidance before action."""
        context = action_spec.call_to_action

        # Update beliefs based on new context
        self._update_counterpart_reservation_from_context(context)

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
        self._update_counterpart_reservation_from_context(observation)

    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        return {
            'beliefs': {
                name: {
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
                self._beliefs[name].confidence = belief_data.get('confidence', self._beliefs[name].confidence)
                self._beliefs[name].evidence_count = belief_data.get('evidence_count', self._beliefs[name].evidence_count)


    def update(self) -> None:
        """Update uncertainty-aware component state."""
        # Gradually decay confidence over time if no new evidence
        for belief in self._beliefs.values():
            if belief.evidence_count == 0:
                belief.confidence *= 0.99  # Slow decay
                belief.std = min(belief.std * 1.01, belief.std * 2)  # Increase uncertainty
