"""Strategy evolution component for meta-learning and continual adaptation."""

import dataclasses
import random
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from collections import deque

from concordia.typing import entity_component
from concordia.typing import entity as entity_lib


@dataclasses.dataclass
class NegotiationEpisode:
    """Records a complete negotiation experience for learning."""
    context: str
    strategy_used: str
    actions_taken: List[str]
    counterpart_responses: List[str]
    outcome_value: float
    relationship_impact: float
    duration: int
    success_metrics: Dict[str, float]
    lessons_learned: List[str]


@dataclasses.dataclass
class StrategyGenome:
    """Represents an evolving negotiation strategy."""
    strategy_id: str
    tactics: List[str]
    parameters: Dict[str, float]
    weights: Dict[str, float]
    fitness_history: List[float]
    context_specialization: List[str]
    generation: int = 0

    def mutate(self, mutation_rate: float = 0.1) -> 'StrategyGenome':
        """Create a mutated version of this strategy."""
        new_genome = StrategyGenome(
            strategy_id=f"{self.strategy_id}_mut_{random.randint(1000, 9999)}",
            tactics=self.tactics.copy(),
            parameters=self.parameters.copy(),
            weights=self.weights.copy(),
            fitness_history=[],
            context_specialization=self.context_specialization.copy(),
            generation=self.generation + 1
        )

        # Parameter mutation
        for param in new_genome.parameters:
            if random.random() < mutation_rate:
                noise = random.gauss(0, 0.1)
                new_genome.parameters[param] = max(0.01, min(1.0,
                    new_genome.parameters[param] + noise))

        # Weight mutation
        for weight in new_genome.weights:
            if random.random() < mutation_rate:
                noise = random.gauss(0, 0.05)
                new_genome.weights[weight] = max(0.01, min(1.0,
                    new_genome.weights[weight] + noise))

        # Tactic mutation (add/remove/modify)
        if random.random() < mutation_rate:
            mutation_type = random.choice(['add', 'remove', 'modify'])
            if mutation_type == 'add' and len(new_genome.tactics) < 10:
                new_tactic = random.choice([
                    'collaborative_framing', 'value_stacking', 'deadline_pressure',
                    'anchoring_high', 'anchoring_low', 'information_gathering',
                    'rapport_building', 'alternative_creation', 'package_deals',
                    'gradual_concession'
                ])
                if new_tactic not in new_genome.tactics:
                    new_genome.tactics.append(new_tactic)
            elif mutation_type == 'remove' and len(new_genome.tactics) > 3:
                new_genome.tactics.remove(random.choice(new_genome.tactics))
            elif mutation_type == 'modify':
                # Modify order
                random.shuffle(new_genome.tactics)

        return new_genome

    def crossover(self, other: 'StrategyGenome') -> Tuple['StrategyGenome', 'StrategyGenome']:
        """Create offspring through crossover with another strategy."""
        # Blend parameters
        child1_params = {}
        child2_params = {}
        for param in self.parameters:
            if param in other.parameters:
                alpha = random.random()
                child1_params[param] = alpha * self.parameters[param] + (1-alpha) * other.parameters[param]
                child2_params[param] = (1-alpha) * self.parameters[param] + alpha * other.parameters[param]
            else:
                child1_params[param] = self.parameters[param]
                child2_params[param] = self.parameters[param]

        # Blend weights similarly
        child1_weights = {}
        child2_weights = {}
        for weight in self.weights:
            if weight in other.weights:
                alpha = random.random()
                child1_weights[weight] = alpha * self.weights[weight] + (1-alpha) * other.weights[weight]
                child2_weights[weight] = (1-alpha) * self.weights[weight] + alpha * other.weights[weight]
            else:
                child1_weights[weight] = self.weights[weight]
                child2_weights[weight] = self.weights[weight]

        # Combine tactics
        combined_tactics = list(set(self.tactics + other.tactics))
        random.shuffle(combined_tactics)

        split_point = len(combined_tactics) // 2
        child1_tactics = combined_tactics[:split_point]
        child2_tactics = combined_tactics[split_point:]

        child1 = StrategyGenome(
            strategy_id=f"cross_{self.strategy_id}_{other.strategy_id}_{random.randint(1000,9999)}",
            tactics=child1_tactics,
            parameters=child1_params,
            weights=child1_weights,
            fitness_history=[],
            context_specialization=list(set(self.context_specialization + other.context_specialization)),
            generation=max(self.generation, other.generation) + 1
        )

        child2 = StrategyGenome(
            strategy_id=f"cross_{other.strategy_id}_{self.strategy_id}_{random.randint(1000,9999)}",
            tactics=child2_tactics,
            parameters=child2_params,
            weights=child2_weights,
            fitness_history=[],
            context_specialization=list(set(self.context_specialization + other.context_specialization)),
            generation=max(self.generation, other.generation) + 1
        )

        return child1, child2


@dataclasses.dataclass
class PerformanceMetrics:
    """Multi-objective performance evaluation."""
    deal_value: float
    relationship_quality: float
    negotiation_speed: float
    fairness: float
    creativity: float
    robustness: float

    def weighted_fitness(self, weights: Dict[str, float]) -> float:
        """Calculate weighted fitness score."""
        return (
            weights.get('deal_value', 0.3) * self.deal_value +
            weights.get('relationship_quality', 0.2) * self.relationship_quality +
            weights.get('negotiation_speed', 0.1) * self.negotiation_speed +
            weights.get('fairness', 0.15) * self.fairness +
            weights.get('creativity', 0.1) * self.creativity +
            weights.get('robustness', 0.15) * self.robustness
        )


class ExperienceReplayBuffer:
    """Intelligent experience replay for continual learning."""

    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.importance_scores = deque(maxlen=max_size)

    def add_experience(self, episode: NegotiationEpisode, importance: float = 1.0):
        """Add experience with importance weighting."""
        self.buffer.append(episode)
        self.importance_scores.append(importance)

    def sample_strategic(self, n_samples: int, current_context: str) -> List[NegotiationEpisode]:
        """Sample experiences strategically for learning."""
        if len(self.buffer) == 0:
            return []

        n_samples = min(n_samples, len(self.buffer))

        # Calculate relevance scores based on context similarity
        relevance_scores = []
        for episode in self.buffer:
            relevance = self._calculate_context_similarity(episode.context, current_context)
            relevance_scores.append(relevance)

        # Sample mix of relevant and diverse experiences
        relevant_indices = np.argsort(relevance_scores)[-n_samples//2:]
        diverse_indices = np.random.choice(
            len(self.buffer),
            size=n_samples//2,
            replace=False,
            p=np.array(list(self.importance_scores)) / sum(self.importance_scores)
        )

        selected_indices = list(relevant_indices) + list(diverse_indices)
        return [self.buffer[i] for i in selected_indices[:n_samples]]

    def _calculate_context_similarity(self, context1: str, context2: str) -> float:
        """Simple context similarity based on keyword overlap."""
        words1 = set(context1.lower().split())
        words2 = set(context2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


class StrategyEvolution(entity_component.ContextComponent):
    """Component for strategy evolution and meta-learning in negotiations."""

    def __init__(
        self,
        model: Any,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        learning_rate: float = 0.01,
    ):
        """Initialize strategy evolution component.

        Args:
            model: Language model for analysis
            population_size: Size of strategy population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            learning_rate: Meta-learning rate
        """
        self._model = model
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._learning_rate = learning_rate

        # Strategy population
        self._strategy_population: List[StrategyGenome] = []
        self._current_strategy: Optional[StrategyGenome] = None

        # Experience management
        self._experience_buffer = ExperienceReplayBuffer(max_size=1000)
        self._current_episode: Optional[NegotiationEpisode] = None

        # Learning state
        self._generation_count = 0
        self._total_negotiations = 0
        self._performance_history: List[PerformanceMetrics] = []

        # Meta-learning parameters
        self._context_adaptation_weights = {
            'high_stakes': 1.2,
            'relationship_focused': 1.1,
            'time_pressure': 0.9,
            'competitive': 1.0,
            'collaborative': 1.1
        }

        self._initialize_population()

    def _initialize_population(self):
        """Initialize diverse strategy population."""
        base_tactics = [
            'rapport_building', 'value_creation', 'information_gathering',
            'anchoring', 'concession_management', 'deadline_management'
        ]

        for i in range(self._population_size):
            # Create diverse initial strategies
            tactics = random.sample(base_tactics, random.randint(3, 6))

            # Random parameters
            parameters = {
                'aggressiveness': random.uniform(0.2, 0.8),
                'flexibility': random.uniform(0.3, 0.9),
                'patience': random.uniform(0.2, 0.8),
                'risk_tolerance': random.uniform(0.1, 0.7),
                'creativity': random.uniform(0.3, 0.9)
            }

            # Random weights
            weights = {
                'deal_value': random.uniform(0.2, 0.5),
                'relationship_quality': random.uniform(0.1, 0.4),
                'negotiation_speed': random.uniform(0.05, 0.2),
                'fairness': random.uniform(0.1, 0.3),
                'creativity': random.uniform(0.05, 0.2),
                'robustness': random.uniform(0.1, 0.3)
            }

            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}

            strategy = StrategyGenome(
                strategy_id=f"init_strategy_{i}",
                tactics=tactics,
                parameters=parameters,
                weights=weights,
                fitness_history=[],
                context_specialization=[],
                generation=0
            )

            self._strategy_population.append(strategy)

        # Select initial strategy
        self._current_strategy = random.choice(self._strategy_population)

    def _analyze_negotiation_context(self, context: str) -> Dict[str, float]:
        """Analyze context for meta-learning features."""
        prompt = f"""Analyze this negotiation context for strategic characteristics:

Context: {context}

Rate each characteristic from 0.0 to 1.0:
1. Stakes level (0.0 = low stakes, 1.0 = high stakes)
2. Relationship importance (0.0 = transactional, 1.0 = relationship critical)
3. Time pressure (0.0 = no urgency, 1.0 = extreme urgency)
4. Competitive intensity (0.0 = collaborative, 1.0 = highly competitive)
5. Complexity level (0.0 = simple, 1.0 = very complex)
6. Uncertainty level (0.0 = clear information, 1.0 = high uncertainty)

Format: stakes:X.X relationship:X.X time_pressure:X.X competitive:X.X complexity:X.X uncertainty:X.X"""

        response = self._model.sample_text(prompt)

        # Parse response
        context_features = {
            'stakes': 0.5,
            'relationship': 0.5,
            'time_pressure': 0.5,
            'competitive': 0.5,
            'complexity': 0.5,
            'uncertainty': 0.5
        }

        for line in response.split('\n'):
            for feature in context_features.keys():
                if f"{feature}:" in line.lower():
                    try:
                        value = float(line.split(':')[1].strip())
                        context_features[feature] = max(0.0, min(1.0, value))
                    except (ValueError, IndexError):
                        pass

        return context_features

    def _select_strategy_for_context(self, context_features: Dict[str, float]) -> StrategyGenome:
        """Select best strategy for current context using meta-learning."""
        if not self._strategy_population:
            return self._current_strategy

        strategy_scores = []  # Use list of tuples instead of dict

        for strategy in self._strategy_population:
            # Calculate fitness based on context match and historical performance
            context_match = self._calculate_context_match(strategy, context_features)
            historical_fitness = np.mean(strategy.fitness_history) if strategy.fitness_history else 0.5

            # Combine scores with meta-learning weights
            combined_score = (
                0.6 * historical_fitness +
                0.4 * context_match
            )

            strategy_scores.append((strategy, combined_score))

        # Select best strategy with some exploration
        if random.random() < 0.1:  # 10% exploration
            return random.choice(self._strategy_population)
        else:  # 90% exploitation
            return max(strategy_scores, key=lambda x: x[1])[0]

    def _calculate_context_match(self, strategy: StrategyGenome, context_features: Dict[str, float]) -> float:
        """Calculate how well a strategy matches the current context."""
        match_score = 0.0

        # Match based on strategy parameters and context features
        if context_features['stakes'] > 0.7:  # High stakes
            match_score += strategy.parameters.get('risk_tolerance', 0.5) * 0.8

        if context_features['relationship'] > 0.6:  # Relationship important
            match_score += strategy.parameters.get('flexibility', 0.5) * 0.7
            match_score += strategy.weights.get('relationship_quality', 0.2) * 2.0

        if context_features['time_pressure'] > 0.6:  # Time pressure
            match_score += (1.0 - strategy.parameters.get('patience', 0.5)) * 0.6
            match_score += strategy.weights.get('negotiation_speed', 0.1) * 3.0

        if context_features['competitive'] > 0.6:  # Competitive
            match_score += strategy.parameters.get('aggressiveness', 0.5) * 0.8

        if context_features['uncertainty'] > 0.6:  # High uncertainty
            match_score += strategy.parameters.get('flexibility', 0.5) * 0.7

        # Tactic matching
        tactic_match = 0.0
        if 'rapport_building' in strategy.tactics and context_features['relationship'] > 0.5:
            tactic_match += 0.2
        if 'deadline_management' in strategy.tactics and context_features['time_pressure'] > 0.5:
            tactic_match += 0.2
        if 'anchoring' in strategy.tactics and context_features['competitive'] > 0.5:
            tactic_match += 0.2

        return (match_score + tactic_match) / 2.0

    def _generate_evolution_guidance(self, context: str, strategy: StrategyGenome) -> str:
        """Generate strategy guidance based on evolution and context."""
        context_features = self._analyze_negotiation_context(context)

        guidance = f"""🧬 Strategy Evolution Guidance

**Current Strategy**: {strategy.strategy_id} (Generation {strategy.generation})

**Strategy Profile**:
• Tactics: {', '.join(strategy.tactics[:3])}{'...' if len(strategy.tactics) > 3 else ''}
• Aggressiveness: {strategy.parameters.get('aggressiveness', 0.5):.1f}
• Flexibility: {strategy.parameters.get('flexibility', 0.5):.1f}
• Risk Tolerance: {strategy.parameters.get('risk_tolerance', 0.5):.1f}
• Historical Performance: {np.mean(strategy.fitness_history) if strategy.fitness_history else 'New strategy':.2f}

**Context Analysis**:
• Stakes Level: {context_features['stakes']:.1f} ({'High' if context_features['stakes'] > 0.6 else 'Medium' if context_features['stakes'] > 0.3 else 'Low'})
• Relationship Importance: {context_features['relationship']:.1f}
• Time Pressure: {context_features['time_pressure']:.1f}
• Competitive Intensity: {context_features['competitive']:.1f}
• Uncertainty Level: {context_features['uncertainty']:.1f}

**Strategic Adaptations**:"""

        # Generate context-specific adaptations
        if context_features['stakes'] > 0.7:
            guidance += "\n• HIGH STAKES: Increase preparation, reduce risk-taking, focus on robust outcomes"

        if context_features['relationship'] > 0.6:
            guidance += "\n• RELATIONSHIP FOCUS: Emphasize trust-building, long-term value, collaborative tactics"

        if context_features['time_pressure'] > 0.6:
            guidance += "\n• TIME PRESSURE: Streamline process, focus on key issues, avoid perfectionism"

        if context_features['competitive'] > 0.6:
            guidance += "\n• COMPETITIVE ENVIRONMENT: Strengthen positions, prepare alternatives, strategic anchoring"

        if context_features['uncertainty'] > 0.6:
            guidance += "\n• HIGH UNCERTAINTY: Increase information gathering, maintain flexibility, scenario planning"

        guidance += f"\n\n**Evolution Insights**:"
        guidance += f"\n• Strategy Population: {len(self._strategy_population)} variants"
        guidance += f"\n• Generation: {self._generation_count}"
        guidance += f"\n• Total Experience: {self._total_negotiations} negotiations"
        guidance += f"\n• Learning Progress: {'Exploration' if self._generation_count < 5 else 'Optimization'} phase"

        return guidance

    def _evolve_population(self):
        """Evolve strategy population using genetic algorithm."""
        if len(self._strategy_population) < 4:
            return

        # Calculate fitness for all strategies
        strategy_fitness = {}
        for strategy in self._strategy_population:
            if strategy.fitness_history:
                # Recent performance weighted more heavily
                weights = np.exp(np.arange(len(strategy.fitness_history)) * 0.1)
                weights = weights / weights.sum()
                fitness = np.average(strategy.fitness_history, weights=weights)
            else:
                fitness = 0.1  # Low fitness for untested strategies
            strategy_fitness[strategy] = fitness

        # Selection: tournament selection
        new_population = []
        elite_count = max(2, self._population_size // 10)  # Keep top performers

        # Elitism: keep best strategies
        sorted_strategies = sorted(strategy_fitness.items(), key=lambda x: x[1], reverse=True)
        for strategy, _ in sorted_strategies[:elite_count]:
            new_population.append(strategy)

        # Generate offspring
        while len(new_population) < self._population_size:
            # Tournament selection
            parent1 = self._tournament_selection(strategy_fitness, tournament_size=3)
            parent2 = self._tournament_selection(strategy_fitness, tournament_size=3)

            if random.random() < self._crossover_rate:
                child1, child2 = parent1.crossover(parent2)
                new_population.extend([child1, child2])
            else:
                # Mutation only
                new_population.append(parent1.mutate(self._mutation_rate))

        # Trim to population size
        self._strategy_population = new_population[:self._population_size]
        self._generation_count += 1

    def _tournament_selection(self, fitness_dict: Dict[StrategyGenome, float], tournament_size: int = 3) -> StrategyGenome:
        """Select strategy using tournament selection."""
        tournament = random.sample(list(fitness_dict.keys()), min(tournament_size, len(fitness_dict)))
        return max(tournament, key=lambda s: fitness_dict[s])

    def get_action_attempt(
        self,
        context: Any,  # ComponentContextMapping
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        """Generate evolved negotiation action based on adaptive strategies."""
        situation_context = action_spec.call_to_action
        
        # Analyze context for strategy selection
        context_features = self._analyze_negotiation_context(situation_context)
        
        # Select best strategy for current context
        selected_strategy = self._select_strategy_for_context(context_features)
        
        # Update current strategy
        if selected_strategy != self._current_strategy:
            self._current_strategy = selected_strategy
        
        # Start episode tracking
        self._current_episode = NegotiationEpisode(
            context=situation_context,
            strategy_used=self._current_strategy.strategy_id,
            actions_taken=[],
            counterpart_responses=[],
            outcome_value=0.0,
            relationship_impact=0.0,
            duration=0,
            success_metrics={},
            lessons_learned=[]
        )
        
        # Generate action based on evolved strategy
        strategy = self._current_strategy
        prompt = f"""Based on evolved strategy and meta-learning, generate a negotiation action:

Situation: {situation_context}

Current Strategy: {strategy.strategy_id} (Generation {strategy.generation})

Strategy Profile:
- Core Tactics: {', '.join(strategy.tactics)}
- Aggressiveness: {strategy.parameters.get('aggressiveness', 0.5):.2f}
- Flexibility: {strategy.parameters.get('flexibility', 0.5):.2f}  
- Risk Tolerance: {strategy.parameters.get('risk_tolerance', 0.5):.2f}
- Patience Level: {strategy.parameters.get('patience', 0.5):.2f}
- Creativity: {strategy.parameters.get('creativity', 0.5):.2f}

Context Analysis:
- Stakes Level: {context_features['stakes']:.2f}
- Relationship Importance: {context_features['relationship']:.2f}
- Time Pressure: {context_features['time_pressure']:.2f}
- Competitive Intensity: {context_features['competitive']:.2f}
- Uncertainty Level: {context_features['uncertainty']:.2f}

Historical Performance: {np.mean(strategy.fitness_history) if strategy.fitness_history else 'New strategy'}

Meta-Learning Insights:
- Total Negotiations: {self._total_negotiations}
- Generation: {self._generation_count}
- Learning Rate: {self._learning_rate:.4f}

Generate a negotiation action that:
1. Applies the primary tactic: {strategy.tactics[0] if strategy.tactics else 'balanced_approach'}
2. Matches aggressiveness level of {strategy.parameters.get('aggressiveness', 0.5):.2f}
3. Shows flexibility level of {strategy.parameters.get('flexibility', 0.5):.2f}
4. Adapts to context features (stakes: {context_features['stakes']:.1f}, competitive: {context_features['competitive']:.1f})
5. Demonstrates learning from {len(strategy.fitness_history)} previous experiences

Action:"""

        response = self._model.sample_text(prompt)
        
        # Clean up response
        action = response.strip()
        if action.lower().startswith('action:'):
            action = action[7:].strip()
        
        # Apply strategy-specific modifications
        aggressiveness = strategy.parameters.get('aggressiveness', 0.5)
        flexibility = strategy.parameters.get('flexibility', 0.5)
        
        # Modify action based on strategy parameters
        if aggressiveness > 0.7:
            action = f"Let me be direct: {action.lower()}"
        elif aggressiveness < 0.3 and flexibility > 0.6:
            action = f"I'm open to exploring options here - {action.lower()}"
        
        # Add strategy evolution context if this is an evolved strategy
        if strategy.generation > 2:
            action = f"Based on our refined approach, {action.lower()}"
        
        # Track action for learning
        if self._current_episode:
            self._current_episode.actions_taken.append(action)
            self._current_episode.duration += 1
        
        return action

    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        """Provide evolution-based guidance before action."""
        context = action_spec.call_to_action

        # Analyze context and select appropriate strategy
        context_features = self._analyze_negotiation_context(context)
        selected_strategy = self._select_strategy_for_context(context_features)

        # Update current strategy if different
        if selected_strategy != self._current_strategy:
            self._current_strategy = selected_strategy

        # Start new episode tracking
        self._current_episode = NegotiationEpisode(
            context=context,
            strategy_used=self._current_strategy.strategy_id,
            actions_taken=[],
            counterpart_responses=[],
            outcome_value=0.0,
            relationship_impact=0.0,
            duration=0,
            success_metrics={},
            lessons_learned=[]
        )

        # Generate evolution-based guidance
        guidance = self._generate_evolution_guidance(context, self._current_strategy)

        return f"\n{guidance}"

    def post_act(self, action_attempt: str) -> str:
        """Update evolution state after action."""
        if self._current_episode:
            self._current_episode.actions_taken.append(action_attempt)
            self._current_episode.duration += 1

        return ""

    def observe(self, observation: str) -> None:
        """Process observations for strategy learning."""
        if self._current_episode:
            self._current_episode.counterpart_responses.append(observation)

            # Simple outcome assessment from observation
            if any(word in observation.lower() for word in ['agree', 'accept', 'deal', 'success']):
                self._current_episode.outcome_value = 0.8
                self._current_episode.relationship_impact = 0.7
            elif any(word in observation.lower() for word in ['reject', 'no deal', 'failed']):
                self._current_episode.outcome_value = 0.2
                self._current_episode.relationship_impact = 0.3

    def _finalize_episode(self, final_outcome: Dict[str, float]):
        """Finalize current episode and update learning."""
        if not self._current_episode or not self._current_strategy:
            return

        # Update episode with final outcome
        self._current_episode.outcome_value = final_outcome.get('value', 0.5)
        self._current_episode.relationship_impact = final_outcome.get('relationship', 0.5)

        # Calculate performance metrics
        performance = PerformanceMetrics(
            deal_value=final_outcome.get('value', 0.5),
            relationship_quality=final_outcome.get('relationship', 0.5),
            negotiation_speed=max(0.1, 1.0 - (self._current_episode.duration / 20.0)),
            fairness=final_outcome.get('fairness', 0.5),
            creativity=final_outcome.get('creativity', 0.5),
            robustness=final_outcome.get('robustness', 0.5)
        )

        # Update strategy fitness
        fitness = performance.weighted_fitness(self._current_strategy.weights)
        self._current_strategy.fitness_history.append(fitness)

        # Add to experience buffer
        importance = max(0.1, abs(fitness - 0.5) * 2)  # Extreme outcomes are more important
        self._experience_buffer.add_experience(self._current_episode, importance)

        # Update counters
        self._total_negotiations += 1
        self._performance_history.append(performance)

        # Evolve population periodically
        if self._total_negotiations % 10 == 0:
            self._evolve_population()

            # Meta-learning: adjust learning rate based on progress
            if len(self._performance_history) >= 20:
                recent_performance = [p.weighted_fitness({'deal_value': 0.4, 'relationship_quality': 0.3, 'fairness': 0.3})
                                    for p in self._performance_history[-10:]]
                older_performance = [p.weighted_fitness({'deal_value': 0.4, 'relationship_quality': 0.3, 'fairness': 0.3})
                                   for p in self._performance_history[-20:-10]]

                if np.mean(recent_performance) > np.mean(older_performance):
                    self._learning_rate *= 1.05  # Increase learning rate if improving
                else:
                    self._learning_rate *= 0.95  # Decrease if not improving

                self._learning_rate = max(0.001, min(0.1, self._learning_rate))

        self._current_episode = None

    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        return {
            'generation_count': self._generation_count,
            'total_negotiations': self._total_negotiations,
            'population_size': len(self._strategy_population),
            'current_strategy': {
                'id': self._current_strategy.strategy_id if self._current_strategy else None,
                'generation': self._current_strategy.generation if self._current_strategy else 0,
                'fitness_history': self._current_strategy.fitness_history if self._current_strategy else [],
                'tactics': self._current_strategy.tactics if self._current_strategy else []
            },
            'learning_rate': self._learning_rate,
            'experience_buffer_size': len(self._experience_buffer.buffer),
            'avg_recent_performance': np.mean([p.weighted_fitness({'deal_value': 0.4, 'relationship_quality': 0.3, 'fairness': 0.3})
                                             for p in self._performance_history[-5:]]) if len(self._performance_history) >= 5 else 0.0
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set component state."""
        self._generation_count = state.get('generation_count', 0)
        self._total_negotiations = state.get('total_negotiations', 0)
        self._learning_rate = state.get('learning_rate', 0.01)

    def update(self) -> None:
        """Update strategy evolution component."""
        # Decay old experiences
        if len(self._experience_buffer.buffer) > 0:
            # Reduce importance of old experiences
            for i in range(len(self._experience_buffer.importance_scores)):
                self._experience_buffer.importance_scores[i] *= 0.99
