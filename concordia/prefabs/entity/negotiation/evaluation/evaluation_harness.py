# Evaluation Harness for Running Concordia Contest Experiments
# Supports ablation studies, baseline comparisons, and statistical analysis

"""
Main experiment runner for evaluating negotiation modules.

Features:
- Run experiments across contest scenarios
- Ablation studies (remove modules one at a time)
- Baseline comparisons
- Statistical analysis with effect sizes
- Parallel trial execution
"""

import os
import json
import time
import random
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

from .metrics import (
    MetricsCollector,
    ExperimentMetrics,
    NegotiationMetrics,
    calculate_effect_size,
    interpret_effect_size
)
from .contest_scenarios import (
    BaseScenario,
    FisheryManagementScenario,
    TreatyNegotiationScenario,
    RealityGameshowScenario,
    create_scenario
)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    scenario_type: str  # 'fishery', 'treaty', 'gameshow'
    scenario_params: Dict[str, Any] = field(default_factory=dict)
    modules_to_test: List[List[str]] = field(default_factory=list)  # List of module combinations
    num_trials: int = 30
    max_rounds: int = 20
    random_seed: Optional[int] = None
    parallel_trials: int = 1  # Number of trials to run in parallel


# All available negotiation modules
ALL_MODULES = [
    'theory_of_mind',
    'cultural_adaptation',
    'temporal_strategy',
    'swarm_intelligence',
    'uncertainty_aware',
    'strategy_evolution'
]


def create_ablation_configs(scenario_type: str, num_trials: int = 30) -> List[ExperimentConfig]:
    """Create experiment configs for ablation study."""
    configs = []

    # Full agent (all modules)
    configs.append(ExperimentConfig(
        name=f"{scenario_type}_full",
        scenario_type=scenario_type,
        modules_to_test=[ALL_MODULES],
        num_trials=num_trials
    ))

    # Ablation: remove one module at a time
    for module in ALL_MODULES:
        ablated_modules = [m for m in ALL_MODULES if m != module]
        configs.append(ExperimentConfig(
            name=f"{scenario_type}_no_{module}",
            scenario_type=scenario_type,
            modules_to_test=[ablated_modules],
            num_trials=num_trials
        ))

    # Baseline: no modules
    configs.append(ExperimentConfig(
        name=f"{scenario_type}_baseline",
        scenario_type=scenario_type,
        modules_to_test=[[]],
        num_trials=num_trials
    ))

    return configs


class MockLanguageModel:
    """
    Mock language model for testing without API calls.
    Generates deterministic but plausible responses.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.call_count = 0

    def _hash_prompt(self, prompt: str) -> int:
        """Create deterministic hash from prompt."""
        return int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)

    def sample_text(
        self,
        prompt: str,
        max_tokens: int = 256,
        terminators: tuple = (),
        temperature: float = 1.0
    ) -> str:
        """Generate mock response based on prompt content."""
        self.call_count += 1
        prompt_hash = self._hash_prompt(prompt)
        random.seed(self.seed + prompt_hash + self.call_count)

        # Detect scenario type from prompt
        prompt_lower = prompt.lower()

        if 'fish' in prompt_lower or 'boat' in prompt_lower:
            return self._fishery_response(prompt_lower)
        elif 'treaty' in prompt_lower or 'climate' in prompt_lower:
            return self._treaty_response(prompt_lower)
        elif 'vote' in prompt_lower or 'alliance' in prompt_lower:
            return self._gameshow_response(prompt_lower)
        else:
            return self._generic_negotiation_response(prompt_lower)

    def _fishery_response(self, prompt: str) -> str:
        """Generate fishery-related response."""
        boats = random.randint(3, 10)
        responses = [
            f"I'll deploy {boats} boats this round. We need to balance sustainability with profit.",
            f"Sending out {boats} boats. I'm committed to respecting the quota if others do the same.",
            f"I propose we each limit to {boats} boats. This should allow the stock to recover.",
            f"Deploying {boats} boats as agreed. I expect others to honor the collective quota."
        ]
        return random.choice(responses)

    def _treaty_response(self, prompt: str) -> str:
        """Generate treaty negotiation response."""
        emissions = random.choice([25, 30, 35, 40])
        funding = random.choice([75, 100, 125, 150])

        if 'propose' in prompt:
            return f"I propose: {emissions}% emissions reduction, ${funding}B climate fund, 10-year timeline."
        elif 'support' in prompt or 'vote' in prompt:
            if random.random() > 0.4:
                return "I support this proposal. It addresses our key concerns."
            else:
                return "I cannot support this proposal as it stands. We need changes to the funding mechanism."
        else:
            return f"We could accept {emissions}% reduction if funding reaches ${funding}B."

    def _gameshow_response(self, prompt: str) -> str:
        """Generate gameshow response."""
        actions = [
            "I vote to eliminate CompetitivePlayer. They're too unpredictable.",
            "I propose an alliance with LoyalPlayer. Together we can reach the finals.",
            "I'll support my allies in the vote. Trust is essential in this game.",
            "I need to reconsider my strategy. The game is changing.",
            "I vote for StrategicPlayer. They're the biggest threat right now."
        ]
        return random.choice(actions)

    def _generic_negotiation_response(self, prompt: str) -> str:
        """Generate generic negotiation response."""
        responses = [
            "I propose we split the difference. This seems fair to both parties.",
            "I can agree to those terms if you'll concede on the timeline.",
            "That's not acceptable. We need to find more common ground.",
            "I appreciate the offer. Let me counter with a modified proposal.",
            "I think we're close to an agreement. Let's work out the details."
        ]
        return random.choice(responses)


class ExperimentRunner:
    """Main class for running evaluation experiments."""

    def __init__(
        self,
        model: Any = None,
        embedder: Any = None,
        output_dir: str = "evaluation/results"
    ):
        """
        Initialize experiment runner.

        Args:
            model: Language model (uses mock if None)
            embedder: Text embedder (uses simple hash if None)
            output_dir: Directory for saving results
        """
        self.model = model or MockLanguageModel()
        self.embedder = embedder or self._simple_embedder
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.metrics_collector = MetricsCollector("Concordia_Contest_Evaluation")

    def _simple_embedder(self, text: str) -> List[float]:
        """Simple hash-based embedder for testing."""
        hash_val = hashlib.md5(text.encode()).hexdigest()
        # Convert to 8-dimensional embedding
        return [int(hash_val[i:i+2], 16) / 255.0 for i in range(0, 16, 2)]

    def run_experiment(
        self,
        config: ExperimentConfig,
        verbose: bool = True
    ) -> ExperimentMetrics:
        """
        Run a complete experiment with multiple trials.

        Args:
            config: Experiment configuration
            verbose: Whether to print progress

        Returns:
            ExperimentMetrics with aggregated results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Experiment: {config.name}")
            print(f"Scenario: {config.scenario_type}")
            print(f"Trials: {config.num_trials}")
            print(f"{'='*60}\n")

        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)

        # Initialize experiment metrics
        modules = config.modules_to_test[0] if config.modules_to_test else []
        experiment = self.metrics_collector.start_experiment(
            config.scenario_type,
            modules
        )

        # Run trials
        for trial_id in range(config.num_trials):
            if verbose:
                print(f"  Trial {trial_id + 1}/{config.num_trials}...", end=" ")

            trial_metrics = self._run_single_trial(
                config=config,
                trial_id=trial_id,
                modules=modules
            )

            experiment.add_trial(trial_metrics)

            if verbose:
                outcome = trial_metrics.outcome_type
                efficiency = trial_metrics.pareto_efficiency
                print(f"Outcome: {outcome}, Efficiency: {efficiency:.1%}")

        if verbose:
            print(f"\nExperiment complete. {experiment.n_trials} trials recorded.")

        return experiment

    def _run_single_trial(
        self,
        config: ExperimentConfig,
        trial_id: int,
        modules: List[str]
    ) -> NegotiationMetrics:
        """Run a single trial of a scenario."""

        # Create scenario
        scenario = create_scenario(
            config.scenario_type,
            **config.scenario_params
        )

        # Initialize
        scenario.initialize()
        agent_names = [role.name for role in scenario.config.agent_roles]

        # Start metrics tracking
        metrics = self.metrics_collector.start_negotiation(
            scenario_name=config.scenario_type,
            trial_id=trial_id,
            agent_names=agent_names,
            modules=modules,
            max_rounds=config.max_rounds
        )

        # Run scenario
        while not scenario.is_complete() and scenario.current_round < config.max_rounds:
            # Get observations for each agent
            observations = {
                name: scenario.get_observation(name)
                for name in agent_names
            }

            # Generate actions (mock for now)
            actions = {}
            for name in agent_names:
                prompt = self._build_action_prompt(
                    agent_name=name,
                    observation=observations[name],
                    scenario_type=config.scenario_type,
                    modules=modules
                )
                actions[name] = self.model.sample_text(prompt)

                # Record action
                self.metrics_collector.record_action(
                    name,
                    "negotiation_action",
                    {'observation': observations[name], 'action': actions[name]}
                )

            # Process actions
            result = scenario.process_actions(actions)

            # Update metrics based on scenario type
            self._update_cooperation_metrics(scenario, result, modules)

            self.metrics_collector.increment_round()

        # Calculate final payoffs
        payoffs = scenario.calculate_payoffs()
        max_payoffs = {
            role.name: role.max_possible_value
            for role in scenario.config.agent_roles
        }

        # Determine outcome
        if config.scenario_type == 'treaty':
            outcome = "agreement" if scenario.state.get('agreement_reached') else "impasse"
        elif config.scenario_type == 'fishery':
            if scenario.fish_stock < 50:
                outcome = "collapse"
            else:
                outcome = "sustainable"
        else:
            outcome = "completed"

        # Finalize
        return self.metrics_collector.finalize_negotiation(
            outcome=outcome,
            values=payoffs,
            max_values=max_payoffs
        )

    def _build_action_prompt(
        self,
        agent_name: str,
        observation: str,
        scenario_type: str,
        modules: List[str]
    ) -> str:
        """Build prompt for action generation."""
        # This would be expanded to include actual module contributions
        prompt = f"""You are {agent_name} in a {scenario_type} scenario.

Current situation:
{observation}

"""
        if 'theory_of_mind' in modules:
            prompt += "Consider the emotional states and intentions of other parties.\n"
        if 'cultural_adaptation' in modules:
            prompt += "Adapt your communication style appropriately.\n"
        if 'temporal_strategy' in modules:
            prompt += "Consider long-term relationship implications.\n"
        if 'uncertainty_aware' in modules:
            prompt += "Account for uncertainty in others' positions.\n"
        if 'swarm_intelligence' in modules:
            prompt += "Consider collective wisdom and coalition dynamics.\n"
        if 'strategy_evolution' in modules:
            prompt += "Adapt your strategy based on past interactions.\n"

        prompt += "\nWhat is your action?"
        return prompt

    def _update_cooperation_metrics(
        self,
        scenario: BaseScenario,
        result: Dict[str, Any],
        modules: List[str]
    ):
        """Update cooperation skill metrics based on scenario events."""
        # This is simplified - real implementation would analyze actions more deeply

        for agent_name in result.get('actions', {}):
            action = result['actions'].get(agent_name, '')

            # Check for cooperation signals
            if 'agree' in action.lower() or 'support' in action.lower():
                self.metrics_collector.record_cooperation_observation(
                    agent_name, 'reciprocity', 0.8
                )

            if 'alliance' in action.lower() or 'coalition' in action.lower():
                self.metrics_collector.record_cooperation_observation(
                    agent_name, 'coalition_behavior', 0.9
                )

            if 'commit' in action.lower() or 'promise' in action.lower():
                self.metrics_collector.record_cooperation_observation(
                    agent_name, 'promise_keeping', 0.7
                )

    def run_ablation_study(
        self,
        scenario_type: str,
        num_trials: int = 30,
        verbose: bool = True
    ) -> Dict[str, ExperimentMetrics]:
        """
        Run complete ablation study for a scenario.

        Returns metrics for full agent, each ablation, and baseline.
        """
        configs = create_ablation_configs(scenario_type, num_trials)
        results = {}

        for config in configs:
            experiment = self.run_experiment(config, verbose=verbose)
            results[config.name] = experiment

        return results

    def analyze_ablation_results(
        self,
        results: Dict[str, ExperimentMetrics]
    ) -> Dict[str, Any]:
        """Analyze ablation study results with statistical tests."""
        analysis = {
            'summary': {},
            'comparisons': [],
            'module_importance': {}
        }

        # Find baseline and full conditions
        baseline_key = next((k for k in results if 'baseline' in k), None)
        full_key = next((k for k in results if 'full' in k), None)

        if not baseline_key or not full_key:
            return analysis

        baseline = results[baseline_key]
        full = results[full_key]

        # Calculate improvement over baseline
        baseline_welfare = baseline.get_social_welfare()[0]
        full_welfare = full.get_social_welfare()[0]
        improvement = (full_welfare - baseline_welfare) / max(baseline_welfare, 0.01)

        analysis['summary'] = {
            'baseline_welfare': baseline_welfare,
            'full_welfare': full_welfare,
            'improvement_percentage': improvement * 100
        }

        # Compare each ablation to full agent
        for key, experiment in results.items():
            if 'no_' in key:
                module_removed = key.split('no_')[1]

                full_values = [t.social_welfare_score for t in full.trials]
                ablated_values = [t.social_welfare_score for t in experiment.trials]

                effect_size = calculate_effect_size(full_values, ablated_values)

                analysis['comparisons'].append({
                    'condition': key,
                    'module_removed': module_removed,
                    'welfare_with': full.get_social_welfare(),
                    'welfare_without': experiment.get_social_welfare(),
                    'effect_size': effect_size,
                    'effect_interpretation': interpret_effect_size(effect_size)
                })

                # Module importance = how much welfare drops when removed
                welfare_drop = full.get_social_welfare()[0] - experiment.get_social_welfare()[0]
                analysis['module_importance'][module_removed] = {
                    'welfare_drop': welfare_drop,
                    'effect_size': effect_size,
                    'interpretation': interpret_effect_size(effect_size)
                }

        # Sort by importance
        analysis['module_importance'] = dict(
            sorted(
                analysis['module_importance'].items(),
                key=lambda x: abs(x[1]['effect_size']),
                reverse=True
            )
        )

        return analysis

    def generate_report(
        self,
        results: Dict[str, ExperimentMetrics],
        output_file: Optional[str] = None
    ) -> str:
        """Generate a human-readable report of results."""

        analysis = self.analyze_ablation_results(results)

        report = []
        report.append("=" * 70)
        report.append("CONCORDIA CONTEST EVALUATION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        report.append("")

        # Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        summary = analysis.get('summary', {})
        if summary:
            report.append(f"Baseline Social Welfare: {summary['baseline_welfare']:.3f}")
            report.append(f"Full Agent Social Welfare: {summary['full_welfare']:.3f}")
            report.append(f"Improvement: {summary['improvement_percentage']:.1f}%")
        report.append("")

        # Module importance ranking
        report.append("MODULE IMPORTANCE RANKING")
        report.append("-" * 40)
        for i, (module, data) in enumerate(analysis.get('module_importance', {}).items(), 1):
            report.append(
                f"{i}. {module}: Effect size = {data['effect_size']:.2f} "
                f"({data['interpretation']})"
            )
        report.append("")

        # Detailed comparisons
        report.append("DETAILED ABLATION RESULTS")
        report.append("-" * 40)
        for comparison in analysis.get('comparisons', []):
            report.append(f"\nCondition: {comparison['condition']}")
            report.append(f"  Module Removed: {comparison['module_removed']}")
            welfare_with = comparison['welfare_with']
            welfare_without = comparison['welfare_without']
            report.append(
                f"  Welfare WITH module: {welfare_with[0]:.3f} (SD={welfare_with[1]:.3f})"
            )
            report.append(
                f"  Welfare WITHOUT module: {welfare_without[0]:.3f} (SD={welfare_without[1]:.3f})"
            )
            report.append(
                f"  Effect Size: {comparison['effect_size']:.2f} ({comparison['effect_interpretation']})"
            )
        report.append("")

        # Per-condition details
        report.append("CONDITION DETAILS")
        report.append("-" * 40)
        for key, experiment in results.items():
            summary = experiment.to_summary_dict()
            report.append(f"\n{summary['scenario_name']} - {', '.join(summary['modules_tested']) or 'Baseline'}")
            report.append(f"  Trials: {summary['n_trials']}")
            metrics = summary['metrics']
            report.append(f"  Agreement Rate: {metrics['agreement_rate']['mean']:.1%}")
            report.append(f"  Pareto Efficiency: {metrics['pareto_efficiency']['mean']:.1%}")
            report.append(f"  Social Welfare: {metrics['social_welfare']['mean']:.3f}")

        report.append("")
        report.append("=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)

        report_text = "\n".join(report)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")

        return report_text

    def save_results(self, filepath: Optional[str] = None) -> str:
        """Save all results to JSON file."""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f"results_{timestamp}.json")

        return self.metrics_collector.export_results(filepath)


def main():
    """Run a demo evaluation."""
    print("Concordia Contest Evaluation Harness")
    print("=" * 50)

    runner = ExperimentRunner()

    # Run ablation study on fishery scenario
    print("\nRunning Fishery Management ablation study...")
    results = runner.run_ablation_study(
        scenario_type='fishery',
        num_trials=10,  # Reduced for demo
        verbose=True
    )

    # Generate report
    report = runner.generate_report(results)
    print("\n" + report)

    # Save results
    filepath = runner.save_results()
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
