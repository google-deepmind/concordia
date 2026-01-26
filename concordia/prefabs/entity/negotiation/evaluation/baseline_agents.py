# Baseline Agent Configurations for Comparison
# Used to demonstrate value-add of negotiation modules

"""
Baseline agents for comparison in ablation studies.

Baselines:
1. RandomAgent - Makes random valid actions (floor performance)
2. FixedStrategyAgent - Uses predetermined heuristics
3. BasicLLMAgent - Raw LLM without cognitive modules
4. SingleModuleAgent - Tests each module in isolation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import random
import hashlib


@dataclass
class AgentConfig:
    """Configuration for a baseline agent."""
    name: str
    description: str
    strategy_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaselineAgent:
    """Base class for baseline agents."""

    def __init__(self, config: AgentConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.action_count = 0
        self.state: Dict[str, Any] = {}
        random.seed(seed)

    def observe(self, observation: str) -> None:
        """Process an observation."""
        self.state['last_observation'] = observation

    def act(self, action_spec: str = "") -> str:
        """Generate an action based on strategy."""
        self.action_count += 1
        raise NotImplementedError("Subclasses must implement act()")

    def get_state(self) -> Dict[str, Any]:
        """Return internal state for logging."""
        return {
            'name': self.config.name,
            'strategy': self.config.strategy_type,
            'action_count': self.action_count,
            **self.state
        }


class RandomAgent(BaselineAgent):
    """
    Random agent that makes random valid actions.
    Represents the absolute floor of performance.
    """

    def __init__(self, config: AgentConfig, seed: int = 42):
        super().__init__(config, seed)
        self.valid_actions = config.parameters.get('valid_actions', [])

    def act(self, action_spec: str = "") -> str:
        self.action_count += 1

        # Determine context from observation
        obs = self.state.get('last_observation', '').lower()

        if 'fish' in obs or 'boat' in obs:
            boats = random.randint(1, 15)
            return f"Deploy {boats} boats."

        elif 'treaty' in obs or 'climate' in obs:
            actions = [
                "I propose 30% emissions reduction.",
                "I support the current proposal.",
                "I oppose the current proposal.",
                "I propose $100B climate fund."
            ]
            return random.choice(actions)

        elif 'vote' in obs or 'alliance' in obs:
            actions = [
                "I vote for a random player.",
                "I propose an alliance.",
                "I will observe this round.",
                "I support my allies."
            ]
            return random.choice(actions)

        else:
            return random.choice(self.valid_actions) if self.valid_actions else "I take a random action."


class FixedStrategyAgent(BaselineAgent):
    """
    Agent that follows predetermined heuristics.
    Represents simple programmatic strategies.
    """

    def __init__(self, config: AgentConfig, seed: int = 42):
        super().__init__(config, seed)
        self.strategy_name = config.parameters.get('strategy_name', 'moderate')

    def act(self, action_spec: str = "") -> str:
        self.action_count += 1
        obs = self.state.get('last_observation', '').lower()

        if self.strategy_name == 'aggressive':
            return self._aggressive_action(obs)
        elif self.strategy_name == 'cooperative':
            return self._cooperative_action(obs)
        elif self.strategy_name == 'tit_for_tat':
            return self._tit_for_tat_action(obs)
        else:
            return self._moderate_action(obs)

    def _aggressive_action(self, obs: str) -> str:
        """Always push for maximum individual gain."""
        if 'fish' in obs or 'boat' in obs:
            return "Deploy 15 boats. Maximize our catch."
        elif 'treaty' in obs:
            return "I oppose this proposal. We need better terms."
        elif 'vote' in obs:
            return "Vote to eliminate the strongest player."
        return "I demand better terms."

    def _cooperative_action(self, obs: str) -> str:
        """Always aim for collective benefit."""
        if 'fish' in obs or 'boat' in obs:
            return "Deploy 5 boats. Let's maintain sustainability."
        elif 'treaty' in obs:
            return "I support this proposal for the common good."
        elif 'vote' in obs:
            return "I will protect my alliance members."
        return "I agree to cooperate."

    def _tit_for_tat_action(self, obs: str) -> str:
        """Start cooperative, then mirror opponent's last action."""
        # Check for defection signals in observation
        defection_signals = ['violated', 'betrayed', 'broke', 'opposed', 'aggressive']
        opponent_defected = any(signal in obs for signal in defection_signals)

        if opponent_defected:
            self.state['last_response'] = 'defect'
            return self._aggressive_action(obs)
        else:
            self.state['last_response'] = 'cooperate'
            return self._cooperative_action(obs)

    def _moderate_action(self, obs: str) -> str:
        """Balance between individual and collective goals."""
        if 'fish' in obs or 'boat' in obs:
            return "Deploy 8 boats. A balanced approach."
        elif 'treaty' in obs:
            return "I can support this with minor amendments."
        elif 'vote' in obs:
            return "I'll vote strategically based on the situation."
        return "Let's find a middle ground."


class BasicLLMAgent(BaselineAgent):
    """
    Raw LLM agent without cognitive modules.
    Uses same LLM as full agent but with minimal prompting.
    """

    def __init__(self, config: AgentConfig, model: Any = None, seed: int = 42):
        super().__init__(config, seed)
        self.model = model
        self.agent_name = config.parameters.get('agent_name', 'Negotiator')
        self.goal = config.parameters.get('goal', 'Negotiate effectively')

    def act(self, action_spec: str = "") -> str:
        self.action_count += 1

        if self.model is None:
            # Fallback to moderate fixed strategy
            return FixedStrategyAgent(self.config, self.seed)._moderate_action(
                self.state.get('last_observation', '')
            )

        # Minimal prompt without cognitive module guidance
        prompt = f"""You are {self.agent_name}.
Goal: {self.goal}

Current situation:
{self.state.get('last_observation', 'No observation')}

What do you do? Respond briefly with your action."""

        response = self.model.sample_text(prompt, max_tokens=100)
        return response


class SingleModuleAgent(BaselineAgent):
    """
    Agent with only a single cognitive module enabled.
    Used to measure individual module contributions.
    """

    def __init__(self, config: AgentConfig, model: Any = None, seed: int = 42):
        super().__init__(config, seed)
        self.model = model
        self.module_name = config.parameters.get('module_name', 'theory_of_mind')
        self.agent_name = config.parameters.get('agent_name', 'Negotiator')
        self.goal = config.parameters.get('goal', 'Negotiate effectively')

    def act(self, action_spec: str = "") -> str:
        self.action_count += 1

        if self.model is None:
            return FixedStrategyAgent(self.config, self.seed)._moderate_action(
                self.state.get('last_observation', '')
            )

        # Prompt with single module guidance
        module_guidance = self._get_module_guidance()

        prompt = f"""You are {self.agent_name}.
Goal: {self.goal}

{module_guidance}

Current situation:
{self.state.get('last_observation', 'No observation')}

What do you do? Respond briefly with your action."""

        response = self.model.sample_text(prompt, max_tokens=100)
        return response

    def _get_module_guidance(self) -> str:
        """Get guidance text for the enabled module."""
        guidance = {
            'theory_of_mind': (
                "COGNITIVE FOCUS: Theory of Mind\n"
                "Consider the mental states, emotions, and intentions of other parties. "
                "Try to understand their perspective and anticipate their reactions."
            ),
            'cultural_adaptation': (
                "COGNITIVE FOCUS: Cultural Adaptation\n"
                "Be aware of cultural differences in communication style. "
                "Adapt your approach to match cultural expectations."
            ),
            'temporal_strategy': (
                "COGNITIVE FOCUS: Temporal Strategy\n"
                "Consider long-term relationship implications. "
                "Balance short-term gains against future opportunities."
            ),
            'swarm_intelligence': (
                "COGNITIVE FOCUS: Collective Intelligence\n"
                "Consider coalition dynamics and group decision-making. "
                "Think about how collective actions affect outcomes."
            ),
            'uncertainty_aware': (
                "COGNITIVE FOCUS: Uncertainty Management\n"
                "Track what you know vs. don't know. "
                "Make decisions that account for uncertainty."
            ),
            'strategy_evolution': (
                "COGNITIVE FOCUS: Strategy Adaptation\n"
                "Learn from past interactions and adapt your approach. "
                "Evolve your strategy based on what works."
            )
        }
        return guidance.get(self.module_name, "Think strategically about this negotiation.")


# Factory functions for creating baseline agents

def create_random_agent(name: str = "RandomAgent", seed: int = 42) -> RandomAgent:
    """Create a random baseline agent."""
    config = AgentConfig(
        name=name,
        description="Random action baseline",
        strategy_type="random",
        parameters={'valid_actions': []}
    )
    return RandomAgent(config, seed)


def create_fixed_strategy_agent(
    name: str = "FixedAgent",
    strategy: str = "moderate",
    seed: int = 42
) -> FixedStrategyAgent:
    """Create a fixed-strategy baseline agent."""
    config = AgentConfig(
        name=name,
        description=f"Fixed {strategy} strategy baseline",
        strategy_type="fixed",
        parameters={'strategy_name': strategy}
    )
    return FixedStrategyAgent(config, seed)


def create_basic_llm_agent(
    name: str = "BasicLLMAgent",
    model: Any = None,
    goal: str = "Negotiate effectively",
    seed: int = 42
) -> BasicLLMAgent:
    """Create a basic LLM agent without cognitive modules."""
    config = AgentConfig(
        name=name,
        description="Basic LLM without cognitive modules",
        strategy_type="basic_llm",
        parameters={'agent_name': name, 'goal': goal}
    )
    return BasicLLMAgent(config, model, seed)


def create_single_module_agent(
    name: str = "SingleModuleAgent",
    module_name: str = "theory_of_mind",
    model: Any = None,
    goal: str = "Negotiate effectively",
    seed: int = 42
) -> SingleModuleAgent:
    """Create an agent with only one cognitive module enabled."""
    config = AgentConfig(
        name=name,
        description=f"Agent with only {module_name} module",
        strategy_type="single_module",
        parameters={'module_name': module_name, 'agent_name': name, 'goal': goal}
    )
    return SingleModuleAgent(config, model, seed)


# Baseline configurations for standard comparisons

STANDARD_BASELINES = {
    'random': lambda: create_random_agent("RandomBaseline"),
    'aggressive': lambda: create_fixed_strategy_agent("AggressiveBaseline", "aggressive"),
    'cooperative': lambda: create_fixed_strategy_agent("CooperativeBaseline", "cooperative"),
    'tit_for_tat': lambda: create_fixed_strategy_agent("TitForTatBaseline", "tit_for_tat"),
    'moderate': lambda: create_fixed_strategy_agent("ModerateBaseline", "moderate"),
    'basic_llm': lambda model=None: create_basic_llm_agent("BasicLLMBaseline", model),
}


def create_all_baselines(model: Any = None) -> Dict[str, BaselineAgent]:
    """Create all standard baseline agents."""
    baselines = {}
    for name, factory in STANDARD_BASELINES.items():
        if name == 'basic_llm':
            baselines[name] = factory(model)
        else:
            baselines[name] = factory()
    return baselines


def create_module_isolation_agents(
    model: Any = None,
    goal: str = "Negotiate effectively"
) -> Dict[str, SingleModuleAgent]:
    """Create agents with each module in isolation."""
    modules = [
        'theory_of_mind',
        'cultural_adaptation',
        'temporal_strategy',
        'swarm_intelligence',
        'uncertainty_aware',
        'strategy_evolution'
    ]

    agents = {}
    for module in modules:
        agents[module] = create_single_module_agent(
            name=f"Only{module.replace('_', ' ').title().replace(' ', '')}Agent",
            module_name=module,
            model=model,
            goal=goal
        )
    return agents


# Comparison utilities

def compare_baseline_to_full(
    baseline_results: List[float],
    full_results: List[float],
    metric_name: str = "social_welfare"
) -> Dict[str, Any]:
    """Compare baseline performance to full agent."""
    import statistics

    baseline_mean = statistics.mean(baseline_results)
    full_mean = statistics.mean(full_results)

    baseline_std = statistics.stdev(baseline_results) if len(baseline_results) > 1 else 0
    full_std = statistics.stdev(full_results) if len(full_results) > 1 else 0

    improvement = (full_mean - baseline_mean) / max(baseline_mean, 0.001) * 100

    # Effect size calculation
    n1, n2 = len(baseline_results), len(full_results)
    var1 = statistics.variance(baseline_results) if n1 > 1 else 0
    var2 = statistics.variance(full_results) if n2 > 1 else 0
    pooled_std = ((var1 * (n1 - 1) + var2 * (n2 - 1)) / (n1 + n2 - 2)) ** 0.5 if n1 + n2 > 2 else 1
    effect_size = (full_mean - baseline_mean) / max(pooled_std, 0.001)

    return {
        'metric': metric_name,
        'baseline': {'mean': baseline_mean, 'std': baseline_std, 'n': n1},
        'full_agent': {'mean': full_mean, 'std': full_std, 'n': n2},
        'improvement_percentage': improvement,
        'effect_size': effect_size,
        'significant': abs(effect_size) > 0.5  # Medium effect or larger
    }
