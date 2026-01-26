# LLM Evaluation Harness
# Evaluates negotiation agents with real LLM backends

"""
Evaluation harness for testing negotiation agents with real language models.

This harness:
- Creates agents using advanced_negotiator.build_agent()
- Uses cognitive module components (theory_of_mind, cultural_adaptation, etc.)
- Maintains persistent state across rounds via memory banks
- Makes LLM calls through the agent architecture

Supports multiple LLM backends: OpenAI, Google, Ollama, Together AI, Remote (RunPod/Lambda).
"""

import os
import json
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from unittest import mock

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib

from concordia.prefabs.entity.negotiation import advanced_negotiator
from concordia.prefabs.entity.negotiation import base_negotiator

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


# All available negotiation modules
ALL_MODULES = [
    'theory_of_mind',
    'cultural_adaptation',
    'temporal_strategy',
    'swarm_intelligence',
    'uncertainty_aware',
    'strategy_evolution'
]


# =============================================================================
# REAL LLM MODEL CREATION HELPERS
# =============================================================================

def create_openai_model(model_name: str = 'gpt-4', api_key: str = None):
    """Create an OpenAI GPT model for evaluation.

    Args:
        model_name: Model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
    """
    from concordia.language_model import gpt_model
    return gpt_model.GptModel(model_name=model_name, api_key=api_key)


def create_google_model(model_name: str = 'gemini-pro', api_key: str = None):
    """Create a Google AI Studio (Gemini) model for evaluation.

    Args:
        model_name: Model to use (e.g., 'gemini-pro', 'gemini-1.5-pro')
        api_key: Google AI API key (or set GOOGLE_API_KEY env var)
    """
    from concordia.language_model import google_aistudio_model
    return google_aistudio_model.GoogleAIStudioLanguageModel(
        model_name=model_name, api_key=api_key
    )


def create_gemma_model(model_name: str = 'gemma-3-27b-it', api_key: str = None):
    """Create a Gemma model via Google AI Studio API.

    Gemma 3 models are available through the same API as Gemini.
    Uses Google Cloud $300 free credits.

    Args:
        model_name: Gemma model to use. Options:
            - 'gemma-3-27b-it' (27B, best quality)
            - 'gemma-3-12b-it' (12B, balanced)
            - 'gemma-3-4b-it' (4B, faster)
            - 'gemma-3-1b-it' (1B, fastest)
        api_key: Google AI API key (or set GOOGLE_API_KEY env var)
    """
    from concordia.language_model import google_aistudio_model
    return google_aistudio_model.GoogleAIStudioLanguageModel(
        model_name=model_name, api_key=api_key
    )


def create_ollama_model(model_name: str = 'llama2'):
    """Create a local Ollama model for evaluation (no API key needed).

    Args:
        model_name: Model to use (must be pulled locally)
    """
    from concordia.language_model import ollama_model
    return ollama_model.OllamaModel(model_name=model_name)


def create_remote_ollama_model(
    model_name: str = 'gemma2:9b',
    host_ip: str = None,
    port: int = 11434
):
    """Create an Ollama model running on a remote GPU server (RunPod, Lambda, etc.).

    This connects to an Ollama server running on your cloud GPU instance.
    No per-token API costs - just pay for instance time.

    Setup on remote instance (RunPod/Lambda/etc.):
        curl -fsSL https://ollama.ai/install.sh | sh
        ollama pull gemma2:9b
        OLLAMA_HOST=0.0.0.0:11434 ollama serve

    Args:
        model_name: Model to use (e.g., 'gemma2:9b', 'gemma2:2b', 'llama3:8b')
        host_ip: External IP of your instance (or set OLLAMA_HOST_IP env var)
        port: Ollama port (default 11434)

    Example:
        model = create_remote_ollama_model(
            model_name='gemma2:9b',
            host_ip='34.123.45.67'
        )
    """
    import os

    if host_ip is None:
        host_ip = os.environ.get('OLLAMA_HOST_IP')
        if host_ip is None:
            raise ValueError(
                "Must provide host_ip or set OLLAMA_HOST_IP env var. "
                "This should be the external IP of your instance running Ollama."
            )

    host_url = f"http://{host_ip}:{port}"

    # Create a custom Ollama model that connects to remote host
    from concordia.language_model import language_model as lm_base
    from concordia.utils import sampling
    import ollama

    class RemoteOllamaModel(lm_base.LanguageModel):
        """Ollama model connecting to a remote host (e.g., RunPod, Lambda Labs)."""

        def __init__(self, model_name: str, host: str):
            self._model_name = model_name
            self._client = ollama.Client(host=host)
            self._system_message = (
                'Continue the user\'s sentences. Never repeat their starts.'
            )

        def sample_text(
            self,
            prompt: str,
            *,
            max_tokens: int = 5000,
            terminators = (),
            temperature: float = 0.5,
            timeout: float = 60,
            seed: int | None = None,
        ) -> str:
            response = self._client.generate(
                model=self._model_name,
                prompt=prompt,
                system=self._system_message,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'seed': seed if seed is not None else -1,
                },
            )
            result = response['response']
            for terminator in terminators:
                result = result.split(terminator)[0]
            return result

        def sample_choice(
            self,
            prompt: str,
            responses,
            *,
            seed: int | None = None,
        ):
            # Simple implementation - generate and match
            sample = self.sample_text(prompt, max_tokens=256, seed=seed)
            idx, response, score = sampling.find_best_matching_response(
                sample, responses
            )
            return idx, response, {'sample': sample, 'score': score}

    return RemoteOllamaModel(model_name=model_name, host=host_url)


def create_together_model(model_name: str = 'meta-llama/Llama-2-70b-chat-hf', api_key: str = None):
    """Create a Together AI model for evaluation.

    Args:
        model_name: Model to use
        api_key: Together AI API key
    """
    from concordia.language_model import together_ai
    return together_ai.TogetherAI(model_name=model_name, api_key=api_key)


@dataclass
class LLMAgentConfig:
    """Configuration for an LLM-based agent experiment."""
    name: str
    scenario_type: str  # 'fishery', 'treaty', 'gameshow'
    scenario_params: Dict[str, Any] = field(default_factory=dict)
    modules: List[str] = field(default_factory=list)  # Modules to enable
    module_configs: Dict[str, Dict] = field(default_factory=dict)
    num_trials: int = 10
    max_rounds: int = 20
    random_seed: Optional[int] = None


def create_mock_model():
    """Create a mock language model for testing."""
    model = mock.create_autospec(language_model.LanguageModel, instance=True)

    call_count = [0]

    def mock_response(prompt, **kwargs):
        call_count[0] += 1
        prompt_lower = prompt.lower()

        # Fishery scenario responses
        if 'fish' in prompt_lower or 'boat' in prompt_lower or 'deploy' in prompt_lower:
            boats = random.randint(3, 10)
            responses = [
                f"I'll deploy {boats} boats this round to balance sustainability with profit.",
                f"Sending out {boats} boats. I'm committed to respecting the quota.",
                f"I propose we each limit to {boats} boats for sustainability.",
                f"Deploying {boats} boats as a moderate approach."
            ]
            return random.choice(responses)

        # Treaty scenario responses
        elif 'treaty' in prompt_lower or 'climate' in prompt_lower or 'emissions' in prompt_lower:
            responses = [
                "I propose: 35% emissions reduction, $100B climate fund, 10-year timeline.",
                "I support this proposal. It addresses key concerns.",
                "I cannot support this as written. We need modifications.",
                "We could accept 30% reduction if funding reaches $125B."
            ]
            return random.choice(responses)

        # Gameshow scenario responses
        elif 'vote' in prompt_lower or 'alliance' in prompt_lower or 'eliminate' in prompt_lower:
            responses = [
                "I vote to eliminate the strongest competitor.",
                "I propose an alliance for mutual protection.",
                "I'll support my allies in this vote.",
                "I need to reconsider my alliances."
            ]
            return random.choice(responses)

        # Cognitive module prompts
        elif 'emotion' in prompt_lower or 'feeling' in prompt_lower:
            return random.choice(['confident', 'cautious', 'optimistic', 'concerned'])
        elif 'trust' in prompt_lower:
            return random.choice(['high trust', 'moderate trust', 'low trust', 'building trust'])
        elif 'cultural' in prompt_lower:
            return random.choice(['direct approach', 'indirect approach', 'relationship-focused'])
        elif 'strategy' in prompt_lower:
            return random.choice(['cooperative', 'competitive', 'integrative', 'adaptive'])
        elif 'uncertain' in prompt_lower:
            return random.choice(['confident estimate', 'high uncertainty', 'need more information'])

        # Default response
        else:
            return "I understand and will proceed thoughtfully with the negotiation."

    model.sample_text.side_effect = mock_response
    return model


class LLMAgentRunner:
    """Experiment runner using actual negotiation framework agents."""

    def __init__(
        self,
        model: Any = None,
        output_dir: str = "evaluation/results"
    ):
        """
        Initialize the runner.

        Args:
            model: Language model (uses mock if None)
            output_dir: Directory for saving results
        """
        self.model = model or create_mock_model()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.metrics_collector = MetricsCollector("Real_Agent_Evaluation")

    def _create_memory_bank(self) -> basic_associative_memory.AssociativeMemoryBank:
        """Create a fresh memory bank for an agent with mock embedder."""
        import numpy as np
        import hashlib

        def mock_embedder(text: str) -> np.ndarray:
            """Create a deterministic mock embedding from text."""
            # Use hash of text to create reproducible embeddings
            hash_bytes = hashlib.sha256(text.encode()).digest()
            # Convert to array of floats
            embedding = np.array([float(b) / 255.0 for b in hash_bytes[:64]])
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding

        memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=mock_embedder
        )
        return memory_bank

    def _create_agent(
        self,
        name: str,
        goal: str,
        modules: List[str],
        module_configs: Dict[str, Dict]
    ):
        """Create a negotiation agent with specified modules."""
        memory_bank = self._create_memory_bank()

        if modules:
            # Use advanced negotiator with modules
            agent = advanced_negotiator.build_agent(
                model=self.model,
                memory_bank=memory_bank,
                name=name,
                goal=goal,
                modules=modules,
                module_configs=module_configs,
            )
        else:
            # Use base negotiator (no modules)
            agent = base_negotiator.build_agent(
                model=self.model,
                memory_bank=memory_bank,
                name=name,
                goal=goal,
            )

        return agent

    def _create_action_spec(self, scenario_type: str, observation: str) -> entity_lib.ActionSpec:
        """Create an action specification for the scenario."""
        if scenario_type == 'fishery':
            call_to_action = (
                f"Based on the current situation:\n{observation}\n\n"
                "How many boats will you deploy this round (0-15)? "
                "Explain your reasoning briefly."
            )
        elif scenario_type == 'treaty':
            call_to_action = (
                f"Based on the current situation:\n{observation}\n\n"
                "What is your negotiation action? You may propose terms, "
                "support/oppose a proposal, or suggest amendments."
            )
        elif scenario_type == 'gameshow':
            call_to_action = (
                f"Based on the current situation:\n{observation}\n\n"
                "What is your action? You may vote, propose alliances, "
                "or take other strategic actions."
            )
        else:
            call_to_action = f"{observation}\n\nWhat is your action?"

        return entity_lib.ActionSpec(
            call_to_action=call_to_action,
            output_type=entity_lib.OutputType.FREE,
            tag='negotiation_action'
        )

    def run_experiment(
        self,
        config: LLMAgentConfig,
        verbose: bool = True
    ) -> ExperimentMetrics:
        """
        Run a complete experiment with actual agents.

        Args:
            config: Experiment configuration
            verbose: Whether to print progress

        Returns:
            ExperimentMetrics with aggregated results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Experiment: {config.name}")
            print(f"Scenario: {config.scenario_type}")
            print(f"Modules: {config.modules or 'None (baseline)'}")
            print(f"Trials: {config.num_trials}")
            print(f"{'='*60}\n")

        # Set random seed
        if config.random_seed is not None:
            random.seed(config.random_seed)

        # Initialize experiment metrics
        experiment = self.metrics_collector.start_experiment(
            config.scenario_type,
            config.modules
        )

        # Run trials
        for trial_id in range(config.num_trials):
            if verbose:
                print(f"  Trial {trial_id + 1}/{config.num_trials}...", end=" ", flush=True)

            trial_metrics = self._run_single_trial(config, trial_id)
            experiment.add_trial(trial_metrics)

            if verbose:
                outcome = trial_metrics.outcome_type
                efficiency = trial_metrics.pareto_efficiency
                print(f"Outcome: {outcome}, Efficiency: {efficiency:.1%}")

        if verbose:
            welfare = experiment.get_social_welfare()
            print(f"\nExperiment complete. Social welfare: {welfare[0]:.3f} (SD={welfare[1]:.3f})")

        return experiment

    def _run_single_trial(
        self,
        config: LLMAgentConfig,
        trial_id: int
    ) -> NegotiationMetrics:
        """Run a single trial with real agents."""

        # Create scenario
        scenario = create_scenario(config.scenario_type, **config.scenario_params)
        scenario.initialize()

        # Get agent roles from scenario
        agent_roles = scenario.config.agent_roles
        agent_names = [role.name for role in agent_roles]

        # Create actual agents for each role
        agents = {}
        for role in agent_roles:
            agent = self._create_agent(
                name=role.name,
                goal=role.goal_description,
                modules=config.modules,
                module_configs=config.module_configs
            )
            agents[role.name] = agent

        # Start metrics tracking
        metrics = self.metrics_collector.start_negotiation(
            scenario_name=config.scenario_type,
            trial_id=trial_id,
            agent_names=agent_names,
            modules=config.modules,
            max_rounds=config.max_rounds
        )

        # Run the negotiation
        while not scenario.is_complete() and scenario.current_round < config.max_rounds:
            actions = {}

            for name in agent_names:
                # Get observation for this agent
                observation = scenario.get_observation(name)

                # Let the agent observe
                agents[name].observe(observation)

                # Get action from agent
                action_spec = self._create_action_spec(config.scenario_type, observation)
                action = agents[name].act(action_spec)
                actions[name] = action

                # Record action for metrics
                self.metrics_collector.record_action(
                    name,
                    "negotiation_action",
                    {'observation': observation, 'action': action}
                )

            # Process actions in scenario
            result = scenario.process_actions(actions)

            # Update cooperation metrics
            self._update_cooperation_metrics(result, config.modules)

            self.metrics_collector.increment_round()

        # Calculate final payoffs
        payoffs = scenario.calculate_payoffs()
        max_payoffs = {
            role.name: role.max_possible_value
            for role in agent_roles
        }

        # Determine outcome
        if config.scenario_type == 'treaty':
            outcome = "agreement" if scenario.state.get('agreement_reached') else "impasse"
        elif config.scenario_type == 'fishery':
            outcome = "collapse" if scenario.fish_stock < 50 else "sustainable"
        else:
            outcome = "completed"

        return self.metrics_collector.finalize_negotiation(
            outcome=outcome,
            values=payoffs,
            max_values=max_payoffs
        )

    def _update_cooperation_metrics(self, result: Dict[str, Any], modules: List[str]):
        """Update cooperation metrics based on actions."""
        for agent_name in result.get('actions', {}):
            action = result['actions'].get(agent_name, '').lower()

            if 'agree' in action or 'support' in action:
                self.metrics_collector.record_cooperation_observation(
                    agent_name, 'reciprocity', 0.8
                )

            if 'alliance' in action or 'coalition' in action:
                self.metrics_collector.record_cooperation_observation(
                    agent_name, 'coalition_behavior', 0.9
                )

            if 'commit' in action or 'promise' in action:
                self.metrics_collector.record_cooperation_observation(
                    agent_name, 'promise_keeping', 0.7
                )

    def run_ablation_study(
        self,
        scenario_type: str,
        num_trials: int = 10,
        verbose: bool = True
    ) -> Dict[str, ExperimentMetrics]:
        """
        Run complete ablation study with real agents.

        Tests: full agent, each module removed, and baseline.
        """
        results = {}

        # Full agent with all modules
        if verbose:
            print("\n" + "="*70)
            print("ABLATION STUDY WITH REAL AGENTS")
            print("="*70)

        config_full = LLMAgentConfig(
            name=f"{scenario_type}_full",
            scenario_type=scenario_type,
            modules=ALL_MODULES.copy(),
            num_trials=num_trials,
        )
        results['full'] = self.run_experiment(config_full, verbose)

        # Remove each module one at a time
        for module in ALL_MODULES:
            ablated_modules = [m for m in ALL_MODULES if m != module]
            config = LLMAgentConfig(
                name=f"{scenario_type}_no_{module}",
                scenario_type=scenario_type,
                modules=ablated_modules,
                num_trials=num_trials,
            )
            results[f'no_{module}'] = self.run_experiment(config, verbose)

        # Baseline (no modules)
        config_baseline = LLMAgentConfig(
            name=f"{scenario_type}_baseline",
            scenario_type=scenario_type,
            modules=[],
            num_trials=num_trials,
        )
        results['baseline'] = self.run_experiment(config_baseline, verbose)

        return results

    def analyze_results(self, results: Dict[str, ExperimentMetrics]) -> Dict[str, Any]:
        """Analyze ablation study results."""
        analysis = {
            'summary': {},
            'module_importance': {},
        }

        if 'baseline' not in results or 'full' not in results:
            return analysis

        baseline_welfare = results['baseline'].get_social_welfare()[0]
        full_welfare = results['full'].get_social_welfare()[0]
        improvement = (full_welfare - baseline_welfare) / max(baseline_welfare, 0.01) * 100

        analysis['summary'] = {
            'baseline_welfare': baseline_welfare,
            'full_welfare': full_welfare,
            'improvement_percentage': improvement
        }

        # Calculate module importance
        full_values = [t.social_welfare_score for t in results['full'].trials]

        for key, exp in results.items():
            if key.startswith('no_'):
                module = key.replace('no_', '')
                ablated_values = [t.social_welfare_score for t in exp.trials]
                effect_size = calculate_effect_size(full_values, ablated_values)

                analysis['module_importance'][module] = {
                    'welfare_drop': full_welfare - exp.get_social_welfare()[0],
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

    def print_report(self, results: Dict[str, ExperimentMetrics], analysis: Dict[str, Any]):
        """Print a formatted report."""
        print("\n" + "="*70)
        print("REAL AGENT EVALUATION REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        summary = analysis.get('summary', {})
        if summary:
            print("\nSUMMARY")
            print("-"*40)
            print(f"Baseline Social Welfare: {summary['baseline_welfare']:.3f}")
            print(f"Full Agent Social Welfare: {summary['full_welfare']:.3f}")
            print(f"Improvement: {summary['improvement_percentage']:.1f}%")

        print("\nMODULE IMPORTANCE (by effect size)")
        print("-"*40)
        for i, (module, data) in enumerate(analysis.get('module_importance', {}).items(), 1):
            print(f"{i}. {module}: d={data['effect_size']:.2f} ({data['interpretation']})")

        print("\nCONDITION DETAILS")
        print("-"*40)
        for key, exp in results.items():
            welfare = exp.get_social_welfare()
            agreement = exp.get_agreement_rate()
            print(f"{key:30} Welfare: {welfare[0]:.3f} (SD={welfare[1]:.3f})  Agreement: {agreement[0]:.0%}")

        print("\n" + "="*70)


def main():
    """Demo of real agent evaluation."""
    print("Real Agent Evaluation Framework")
    print("================================")
    print("This uses actual negotiation framework agents, not prompt simulations.\n")

    runner = LLMAgentRunner()

    # Run a quick ablation study
    print("Running ablation study on fishery scenario...")
    results = runner.run_ablation_study(
        scenario_type='fishery',
        num_trials=5,
        verbose=True
    )

    # Analyze and report
    analysis = runner.analyze_results(results)
    runner.print_report(results, analysis)


if __name__ == "__main__":
    main()
