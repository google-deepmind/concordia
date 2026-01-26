#!/usr/bin/env python3
# CLI Tool for Running Concordia Contest Evaluation Experiments

"""
Command-line interface for running evaluation experiments.

Usage:
    python run_experiments.py --scenario fishery --trials 30 --ablation
    python run_experiments.py --scenario all --trials 50 --output results/
    python run_experiments.py --quick  # Quick sanity check with 5 trials
"""

import argparse
import os
import json
from datetime import datetime

from .evaluation_harness import (
    ExperimentRunner,
    ExperimentConfig,
    ALL_MODULES,
    create_ablation_configs
)
from .metrics import MetricsCollector
from .baseline_agents import create_all_baselines


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Concordia Contest evaluation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ablation study on fishery scenario
  python run_experiments.py --scenario fishery --trials 30 --ablation

  # Run all scenarios with 50 trials each
  python run_experiments.py --scenario all --trials 50

  # Quick sanity check (5 trials, all scenarios)
  python run_experiments.py --quick

  # Full paper-worthy evaluation
  python run_experiments.py --full --output results/paper/
        """
    )

    # Scenario selection
    parser.add_argument(
        '--scenario', '-s',
        choices=['fishery', 'treaty', 'gameshow', 'all'],
        default='all',
        help='Which scenario(s) to run (default: all)'
    )

    # Trial configuration
    parser.add_argument(
        '--trials', '-n',
        type=int,
        default=30,
        help='Number of trials per condition (default: 30)'
    )

    # Study type
    parser.add_argument(
        '--ablation', '-a',
        action='store_true',
        help='Run full ablation study (each module removed)'
    )

    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='Only compare full agent to baselines'
    )

    parser.add_argument(
        '--module-isolation',
        action='store_true',
        help='Test each module in isolation'
    )

    # Presets
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick sanity check (5 trials, limited conditions)'
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help='Full paper-worthy evaluation (50 trials, all conditions)'
    )

    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation/results',
        help='Output directory for results (default: evaluation/results)'
    )

    parser.add_argument(
        '--format',
        choices=['json', 'text', 'both'],
        default='both',
        help='Output format (default: both)'
    )

    # Execution
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without executing'
    )

    return parser.parse_args()


def get_scenarios(scenario_arg: str) -> list:
    """Get list of scenarios to run."""
    if scenario_arg == 'all':
        return ['fishery', 'treaty', 'gameshow']
    return [scenario_arg]


def create_experiment_plan(args) -> list:
    """Create list of experiment configurations based on arguments."""
    scenarios = get_scenarios(args.scenario)
    configs = []

    # Handle presets
    if args.quick:
        trials = 5
        scenarios = ['fishery']  # Just one scenario for quick check
    elif args.full:
        trials = 50
    else:
        trials = args.trials

    for scenario in scenarios:
        if args.baseline_only:
            # Just full agent vs baseline
            configs.append(ExperimentConfig(
                name=f"{scenario}_full",
                scenario_type=scenario,
                modules_to_test=[ALL_MODULES],
                num_trials=trials,
                random_seed=args.seed
            ))
            configs.append(ExperimentConfig(
                name=f"{scenario}_baseline",
                scenario_type=scenario,
                modules_to_test=[[]],
                num_trials=trials,
                random_seed=args.seed
            ))

        elif args.module_isolation:
            # Each module in isolation
            for module in ALL_MODULES:
                configs.append(ExperimentConfig(
                    name=f"{scenario}_only_{module}",
                    scenario_type=scenario,
                    modules_to_test=[[module]],
                    num_trials=trials,
                    random_seed=args.seed
                ))
            # Plus baseline
            configs.append(ExperimentConfig(
                name=f"{scenario}_baseline",
                scenario_type=scenario,
                modules_to_test=[[]],
                num_trials=trials,
                random_seed=args.seed
            ))

        elif args.ablation or args.full:
            # Full ablation study
            configs.extend(create_ablation_configs(scenario, trials))
            # Update seeds
            for config in configs:
                config.random_seed = args.seed

        else:
            # Default: just full agent vs baseline
            configs.append(ExperimentConfig(
                name=f"{scenario}_full",
                scenario_type=scenario,
                modules_to_test=[ALL_MODULES],
                num_trials=trials,
                random_seed=args.seed
            ))
            configs.append(ExperimentConfig(
                name=f"{scenario}_baseline",
                scenario_type=scenario,
                modules_to_test=[[]],
                num_trials=trials,
                random_seed=args.seed
            ))

    return configs


def print_experiment_plan(configs: list):
    """Print what experiments will be run."""
    print("\nExperiment Plan:")
    print("=" * 60)

    scenarios = set(c.scenario_type for c in configs)
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Total conditions: {len(configs)}")
    print(f"Trials per condition: {configs[0].num_trials if configs else 0}")
    print(f"Total trials: {len(configs) * (configs[0].num_trials if configs else 0)}")

    print("\nConditions:")
    for config in configs:
        modules = config.modules_to_test[0] if config.modules_to_test else []
        module_str = ', '.join(modules) if modules else 'None (baseline)'
        print(f"  - {config.name}: {module_str}")

    print("=" * 60)


def run_experiments(configs: list, args) -> dict:
    """Run all experiments and return results."""
    runner = ExperimentRunner(output_dir=args.output)
    results = {}

    total = len(configs)
    for i, config in enumerate(configs, 1):
        if args.verbose:
            print(f"\n[{i}/{total}] Running: {config.name}")

        experiment = runner.run_experiment(config, verbose=args.verbose)
        results[config.name] = experiment

        if args.verbose:
            summary = experiment.to_summary_dict()
            print(f"  Agreement Rate: {summary['metrics']['agreement_rate']['mean']:.1%}")
            print(f"  Social Welfare: {summary['metrics']['social_welfare']['mean']:.3f}")

    return results, runner


def save_results(results: dict, runner: ExperimentRunner, args):
    """Save results in requested format(s)."""
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    saved_files = []

    if args.format in ['json', 'both']:
        json_path = os.path.join(args.output, f"results_{timestamp}.json")
        runner.save_results(json_path)
        saved_files.append(json_path)

    if args.format in ['text', 'both']:
        report_path = os.path.join(args.output, f"report_{timestamp}.txt")
        runner.generate_report(results, report_path)
        saved_files.append(report_path)

    return saved_files


def print_summary(results: dict, runner: ExperimentRunner):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Group by scenario
    scenarios = {}
    for name, exp in results.items():
        scenario = exp.scenario_name
        if scenario not in scenarios:
            scenarios[scenario] = {}
        scenarios[scenario][name] = exp

    for scenario, exps in scenarios.items():
        print(f"\n{scenario.upper()}")
        print("-" * 40)

        # Find baseline and full
        baseline = next((e for n, e in exps.items() if 'baseline' in n), None)
        full = next((e for n, e in exps.items() if 'full' in n), None)

        if baseline and full:
            baseline_welfare = baseline.get_social_welfare()[0]
            full_welfare = full.get_social_welfare()[0]
            improvement = (full_welfare - baseline_welfare) / max(baseline_welfare, 0.001) * 100

            print(f"  Baseline Social Welfare: {baseline_welfare:.3f}")
            print(f"  Full Agent Social Welfare: {full_welfare:.3f}")
            print(f"  Improvement: {improvement:+.1f}%")

        # Print each condition
        print("\n  All Conditions:")
        for name, exp in exps.items():
            summary = exp.to_summary_dict()
            modules = summary['modules_tested']
            module_str = ', '.join(modules) if modules else 'baseline'
            welfare = summary['metrics']['social_welfare']['mean']
            agreement = summary['metrics']['agreement_rate']['mean']
            print(f"    {module_str:40} Welfare: {welfare:.3f}  Agreement: {agreement:.0%}")


def main():
    """Main entry point."""
    args = parse_args()

    print("Concordia Contest Evaluation Framework")
    print("=" * 50)

    # Create experiment plan
    configs = create_experiment_plan(args)

    if not configs:
        print("No experiments to run!")
        return 1

    # Print plan
    print_experiment_plan(configs)

    if args.dry_run:
        print("\n[Dry run - not executing experiments]")
        return 0

    # Confirm before running full evaluation
    if args.full and not args.verbose:
        total_trials = len(configs) * 50
        print(f"\nThis will run {total_trials} trials. Continue? (y/n) ", end="")
        if input().lower() != 'y':
            print("Cancelled.")
            return 0

    # Run experiments
    print("\nRunning experiments...")
    results, runner = run_experiments(configs, args)

    # Print summary
    print_summary(results, runner)

    # Save results
    saved_files = save_results(results, runner, args)
    print(f"\nResults saved to: {', '.join(saved_files)}")

    # Generate report
    if args.format in ['text', 'both']:
        report = runner.generate_report(results)
        if args.verbose:
            print("\n" + report)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
