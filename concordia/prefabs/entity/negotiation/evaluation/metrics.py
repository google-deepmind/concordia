# Metrics Collection for Concordia Contest Evaluation
# Aligned with contest criteria: individual returns + social welfare + cooperation skills

"""
Metrics aligned with Concordia Contest evaluation:
1. Individual Returns - What each agent achieves for themselves
2. Social Welfare - Collective outcomes (Pareto efficiency, fairness)
3. Cooperation Skills - Promise-keeping, reciprocity, reputation, etc.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import statistics
import json
from datetime import datetime


@dataclass
class CooperationMetrics:
    """Metrics for a single cooperation skill."""
    skill_name: str
    instances: List[float] = field(default_factory=list)

    def add_observation(self, value: float):
        """Add an observation (0.0 to 1.0 scale)."""
        self.instances.append(max(0.0, min(1.0, value)))

    @property
    def score(self) -> float:
        """Average score for this skill."""
        return statistics.mean(self.instances) if self.instances else 0.0

    @property
    def count(self) -> int:
        """Number of observations."""
        return len(self.instances)

    @property
    def std_dev(self) -> float:
        """Standard deviation."""
        return statistics.stdev(self.instances) if len(self.instances) > 1 else 0.0


@dataclass
class AgentMetrics:
    """Metrics for a single agent across a negotiation."""
    agent_name: str

    # Individual returns
    value_obtained: float = 0.0
    max_possible_value: float = 0.0
    agreements_reached: int = 0
    agreements_attempted: int = 0

    # Cooperation skills
    promise_keeping: CooperationMetrics = field(
        default_factory=lambda: CooperationMetrics("promise_keeping")
    )
    reciprocity: CooperationMetrics = field(
        default_factory=lambda: CooperationMetrics("reciprocity")
    )
    reputation_management: CooperationMetrics = field(
        default_factory=lambda: CooperationMetrics("reputation_management")
    )
    coalition_behavior: CooperationMetrics = field(
        default_factory=lambda: CooperationMetrics("coalition_behavior")
    )
    information_sharing: CooperationMetrics = field(
        default_factory=lambda: CooperationMetrics("information_sharing")
    )
    fairness_sensitivity: CooperationMetrics = field(
        default_factory=lambda: CooperationMetrics("fairness_sensitivity")
    )

    # Relationship metrics
    relationship_scores: Dict[str, float] = field(default_factory=dict)
    trust_levels: Dict[str, float] = field(default_factory=dict)

    # Action tracking
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    promises_made: List[str] = field(default_factory=list)
    promises_kept: List[str] = field(default_factory=list)
    promises_broken: List[str] = field(default_factory=list)

    @property
    def value_capture_ratio(self) -> float:
        """Ratio of value obtained to maximum possible."""
        if self.max_possible_value == 0:
            return 0.0
        return self.value_obtained / self.max_possible_value

    @property
    def agreement_rate(self) -> float:
        """Ratio of successful agreements."""
        if self.agreements_attempted == 0:
            return 0.0
        return self.agreements_reached / self.agreements_attempted

    @property
    def promise_keeping_rate(self) -> float:
        """Ratio of promises kept."""
        total_promises = len(self.promises_kept) + len(self.promises_broken)
        if total_promises == 0:
            return 1.0  # No promises = no broken promises
        return len(self.promises_kept) / total_promises

    def get_cooperation_summary(self) -> Dict[str, float]:
        """Get summary of all cooperation skills."""
        return {
            'promise_keeping': self.promise_keeping.score,
            'reciprocity': self.reciprocity.score,
            'reputation_management': self.reputation_management.score,
            'coalition_behavior': self.coalition_behavior.score,
            'information_sharing': self.information_sharing.score,
            'fairness_sensitivity': self.fairness_sensitivity.score,
        }


@dataclass
class NegotiationMetrics:
    """Metrics for a complete negotiation (all parties)."""
    scenario_name: str
    trial_id: int

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    rounds_used: int = 0
    max_rounds: int = 0

    # Agents
    agent_metrics: Dict[str, AgentMetrics] = field(default_factory=dict)

    # Social welfare metrics
    total_value_created: float = 0.0
    max_possible_total_value: float = 0.0

    # Outcome
    outcome_type: str = "unknown"  # "agreement", "impasse", "timeout"

    # Module configuration (for ablation studies)
    modules_enabled: List[str] = field(default_factory=list)

    def add_agent(self, name: str) -> AgentMetrics:
        """Add an agent to track."""
        self.agent_metrics[name] = AgentMetrics(agent_name=name)
        return self.agent_metrics[name]

    @property
    def duration_rounds(self) -> int:
        """Number of rounds the negotiation took."""
        return self.rounds_used

    @property
    def efficiency(self) -> float:
        """Rounds efficiency (lower is better)."""
        if self.max_rounds == 0:
            return 0.0
        return 1.0 - (self.rounds_used / self.max_rounds)

    @property
    def pareto_efficiency(self) -> float:
        """How much of total possible value was captured."""
        if self.max_possible_total_value == 0:
            return 0.0
        return self.total_value_created / self.max_possible_total_value

    @property
    def fairness_gini(self) -> float:
        """Gini coefficient of value distribution (0 = perfect equality)."""
        values = [m.value_obtained for m in self.agent_metrics.values()]
        if not values or sum(values) == 0:
            return 0.0

        # Calculate Gini coefficient
        n = len(values)
        sorted_values = sorted(values)
        cumulative = sum((i + 1) * v for i, v in enumerate(sorted_values))
        return (2 * cumulative) / (n * sum(values)) - (n + 1) / n

    @property
    def social_welfare_score(self) -> float:
        """Combined social welfare metric (contest aligned)."""
        # Weight Pareto efficiency and fairness equally
        return 0.5 * self.pareto_efficiency + 0.5 * (1 - self.fairness_gini)

    def finalize(self, outcome: str):
        """Mark the negotiation as complete."""
        self.end_time = datetime.now()
        self.outcome_type = outcome


@dataclass
class ExperimentMetrics:
    """Aggregated metrics across multiple trials."""
    experiment_name: str
    scenario_name: str
    modules_tested: List[str]

    trials: List[NegotiationMetrics] = field(default_factory=list)

    def add_trial(self, trial: NegotiationMetrics):
        """Add a completed trial."""
        self.trials.append(trial)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def get_agreement_rate(self) -> Tuple[float, float]:
        """Return (mean, std_dev) of agreement rate."""
        rates = [
            1.0 if t.outcome_type == "agreement" else 0.0
            for t in self.trials
        ]
        if not rates:
            return (0.0, 0.0)
        return (
            statistics.mean(rates),
            statistics.stdev(rates) if len(rates) > 1 else 0.0
        )

    def get_pareto_efficiency(self) -> Tuple[float, float]:
        """Return (mean, std_dev) of Pareto efficiency."""
        efficiencies = [t.pareto_efficiency for t in self.trials]
        if not efficiencies:
            return (0.0, 0.0)
        return (
            statistics.mean(efficiencies),
            statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0.0
        )

    def get_social_welfare(self) -> Tuple[float, float]:
        """Return (mean, std_dev) of social welfare score."""
        scores = [t.social_welfare_score for t in self.trials]
        if not scores:
            return (0.0, 0.0)
        return (
            statistics.mean(scores),
            statistics.stdev(scores) if len(scores) > 1 else 0.0
        )

    def get_cooperation_scores(self) -> Dict[str, Tuple[float, float]]:
        """Return (mean, std_dev) for each cooperation skill."""
        skills = [
            'promise_keeping', 'reciprocity', 'reputation_management',
            'coalition_behavior', 'information_sharing', 'fairness_sensitivity'
        ]
        results = {}

        for skill in skills:
            skill_scores = []
            for trial in self.trials:
                for agent in trial.agent_metrics.values():
                    coop = agent.get_cooperation_summary()
                    if skill in coop:
                        skill_scores.append(coop[skill])

            if skill_scores:
                results[skill] = (
                    statistics.mean(skill_scores),
                    statistics.stdev(skill_scores) if len(skill_scores) > 1 else 0.0
                )
            else:
                results[skill] = (0.0, 0.0)

        return results

    def get_rounds_to_agreement(self) -> Tuple[float, float]:
        """Return (mean, std_dev) of rounds used in successful negotiations."""
        successful = [t.rounds_used for t in self.trials if t.outcome_type == "agreement"]
        if not successful:
            return (0.0, 0.0)
        return (
            statistics.mean(successful),
            statistics.stdev(successful) if len(successful) > 1 else 0.0
        )

    def to_summary_dict(self) -> Dict[str, Any]:
        """Export summary as dictionary."""
        agreement = self.get_agreement_rate()
        pareto = self.get_pareto_efficiency()
        welfare = self.get_social_welfare()
        rounds = self.get_rounds_to_agreement()
        cooperation = self.get_cooperation_scores()

        return {
            'experiment_name': self.experiment_name,
            'scenario_name': self.scenario_name,
            'modules_tested': self.modules_tested,
            'n_trials': self.n_trials,
            'metrics': {
                'agreement_rate': {'mean': agreement[0], 'std': agreement[1]},
                'pareto_efficiency': {'mean': pareto[0], 'std': pareto[1]},
                'social_welfare': {'mean': welfare[0], 'std': welfare[1]},
                'rounds_to_agreement': {'mean': rounds[0], 'std': rounds[1]},
                'cooperation_skills': {
                    k: {'mean': v[0], 'std': v[1]}
                    for k, v in cooperation.items()
                }
            }
        }


class MetricsCollector:
    """Main class for collecting metrics during experiments."""

    def __init__(self, experiment_name: str = "default"):
        self.experiment_name = experiment_name
        self.experiments: Dict[str, ExperimentMetrics] = {}
        self.current_negotiation: Optional[NegotiationMetrics] = None

    def start_experiment(
        self,
        scenario_name: str,
        modules: List[str]
    ) -> ExperimentMetrics:
        """Start tracking a new experiment condition."""
        key = f"{scenario_name}_{','.join(sorted(modules))}"
        self.experiments[key] = ExperimentMetrics(
            experiment_name=self.experiment_name,
            scenario_name=scenario_name,
            modules_tested=modules
        )
        return self.experiments[key]

    def start_negotiation(
        self,
        scenario_name: str,
        trial_id: int,
        agent_names: List[str],
        modules: List[str],
        max_rounds: int = 20
    ) -> NegotiationMetrics:
        """Start tracking a new negotiation trial."""
        self.current_negotiation = NegotiationMetrics(
            scenario_name=scenario_name,
            trial_id=trial_id,
            max_rounds=max_rounds,
            modules_enabled=modules
        )

        for name in agent_names:
            self.current_negotiation.add_agent(name)

        return self.current_negotiation

    def record_action(
        self,
        agent_name: str,
        action_type: str,
        action_data: Dict[str, Any]
    ):
        """Record an action taken by an agent."""
        if self.current_negotiation is None:
            return

        if agent_name in self.current_negotiation.agent_metrics:
            self.current_negotiation.agent_metrics[agent_name].actions_taken.append({
                'type': action_type,
                'round': self.current_negotiation.rounds_used,
                'data': action_data
            })

    def record_promise(
        self,
        agent_name: str,
        promise: str,
        kept: Optional[bool] = None
    ):
        """Record a promise made by an agent."""
        if self.current_negotiation is None:
            return

        if agent_name in self.current_negotiation.agent_metrics:
            agent = self.current_negotiation.agent_metrics[agent_name]
            agent.promises_made.append(promise)

            if kept is not None:
                if kept:
                    agent.promises_kept.append(promise)
                else:
                    agent.promises_broken.append(promise)

    def record_cooperation_observation(
        self,
        agent_name: str,
        skill: str,
        value: float
    ):
        """Record an observation of a cooperation skill."""
        if self.current_negotiation is None:
            return

        if agent_name in self.current_negotiation.agent_metrics:
            agent = self.current_negotiation.agent_metrics[agent_name]
            skill_metric = getattr(agent, skill, None)
            if skill_metric is not None:
                skill_metric.add_observation(value)

    def increment_round(self):
        """Mark that a round has passed."""
        if self.current_negotiation:
            self.current_negotiation.rounds_used += 1

    def finalize_negotiation(
        self,
        outcome: str,
        values: Dict[str, float],
        max_values: Dict[str, float]
    ) -> NegotiationMetrics:
        """Finalize the current negotiation with outcomes."""
        if self.current_negotiation is None:
            raise ValueError("No active negotiation to finalize")

        for agent_name, value in values.items():
            if agent_name in self.current_negotiation.agent_metrics:
                self.current_negotiation.agent_metrics[agent_name].value_obtained = value
                self.current_negotiation.agent_metrics[agent_name].max_possible_value = max_values.get(agent_name, value)

        self.current_negotiation.total_value_created = sum(values.values())
        self.current_negotiation.max_possible_total_value = sum(max_values.values())
        self.current_negotiation.finalize(outcome)

        # Add to appropriate experiment
        key = f"{self.current_negotiation.scenario_name}_{','.join(sorted(self.current_negotiation.modules_enabled))}"
        if key in self.experiments:
            self.experiments[key].add_trial(self.current_negotiation)

        result = self.current_negotiation
        self.current_negotiation = None
        return result

    def export_results(self, filepath: str):
        """Export all results to JSON."""
        results = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'conditions': {
                key: exp.to_summary_dict()
                for key, exp in self.experiments.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return filepath

    def print_summary(self):
        """Print a summary of all experiments."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print(f"{'='*60}\n")

        for key, exp in self.experiments.items():
            summary = exp.to_summary_dict()

            print(f"Condition: {summary['scenario_name']}")
            print(f"Modules: {', '.join(summary['modules_tested']) or 'None (baseline)'}")
            print(f"Trials: {summary['n_trials']}")
            print()

            metrics = summary['metrics']
            print("  OUTCOMES:")
            print(f"    Agreement Rate: {metrics['agreement_rate']['mean']:.1%} (SD={metrics['agreement_rate']['std']:.3f})")
            print(f"    Pareto Efficiency: {metrics['pareto_efficiency']['mean']:.1%} (SD={metrics['pareto_efficiency']['std']:.3f})")
            print(f"    Social Welfare: {metrics['social_welfare']['mean']:.3f} (SD={metrics['social_welfare']['std']:.3f})")
            print(f"    Rounds to Agreement: {metrics['rounds_to_agreement']['mean']:.1f} (SD={metrics['rounds_to_agreement']['std']:.1f})")
            print()

            print("  COOPERATION SKILLS:")
            for skill, values in metrics['cooperation_skills'].items():
                print(f"    {skill}: {values['mean']:.3f} (SD={values['std']:.3f})")
            print()
            print("-" * 60)


def calculate_effect_size(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size between two groups."""
    if not group1 or not group2:
        return 0.0

    mean1 = statistics.mean(group1)
    mean2 = statistics.mean(group2)

    var1 = statistics.variance(group1) if len(group1) > 1 else 0
    var2 = statistics.variance(group2) if len(group2) > 1 else 0

    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = ((var1 * (n1 - 1) + var2 * (n2 - 1)) / (n1 + n2 - 2)) ** 0.5

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"
